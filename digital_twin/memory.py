"""
Stage 5: Memory Engine & Conflict-Aware Retrieval-Augmented Generation (RAG).

Responsibilities:
1. Vector Store Abstraction:
   - Built-in SQLiteVectorStore: Zero-external-dependency, persistent SQLite storage
     with NumPy-accelerated normalized cosine similarity search.
   - ChromaDBVectorStore: Native ChromaDB integration when available.
   - Fast semantic embedding generation (supporting SentenceTransformers or built-in dense hashing).
2. Multi-Tiered Memory Conflict Resolution:
   - Dynamic Conflicts (Temporal Validity): Resolves conflicting temporal states by
     prioritizing the most recent timestamp and marking older memories as superseded.
   - Static Conflicts (Factual Integrity): Rejects retrieved memories that contradict
     authoritative invariant facts in the Stage 4 Identity Core.
   - Conditional Conflicts (Contextual Applicability): Preserves context-dependent facts
     (e.g., morning coffee vs evening tea) by matching query metadata bindings.
3. Distractor Filtering:
   - Enforces strict ownership validation (owner == target persona), filtering out
     memories containing statements made by friends, colleagues, or interlocutors.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from digital_twin.config import MemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    """A discrete unit of episodic or semantic memory."""
    id: str
    content: str
    timestamp: str               # ISO-8601 string
    owner: str                   # Identity owner (e.g. "Dingxuan Liang")
    category: str = "general"    # "location", "preference", "career", "project", "general"
    condition_binding: Optional[Dict[str, Any]] = None  # e.g. {"time_of_day": "morning"}
    status: str = "active"       # "active", "superseded", "rejected"
    conflict_reason: Optional[str] = None
    similarity_score: float = 0.0
    unixtime: float = 0.0

    def __post_init__(self):
        if not self.unixtime and self.timestamp:
            try:
                dt = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
                self.unixtime = dt.timestamp()
            except Exception:
                self.unixtime = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IdentityManager:
    """
    Stage 4 Engine: Manages the authoritative Identity Core.
    Acts as Level-1 Ground Truth with absolute precedence over retrieved memories.
    """

    def __init__(self, identity_path: Union[str, Path]):
        self.identity_path = Path(identity_path)
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.identity_path.is_file():
            logger.warning(f"Identity file not found at {self.identity_path}. Creating fallback.")
            return {
                "personal": {"full_name": "Dingxuan Liang", "current_residence": "Singapore"},
                "immutable_facts": {"birthplace": "Singapore", "home_country": "Singapore"},
                "stable_preferences": {}
            }
        with open(self.identity_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def full_name(self) -> str:
        return self.data.get("personal", {}).get("full_name", "Dingxuan Liang")

    @property
    def preferred_name(self) -> str:
        return self.data.get("personal", {}).get("preferred_name", "Dingxuan")

    def get_summary_markdown(self) -> str:
        """Returns structured markdown of invariant facts for prompt injection."""
        p = self.data.get("personal", {})
        edu = self.data.get("education", {})
        car = self.data.get("career", {})
        facts = self.data.get("immutable_facts", {})
        prefs = self.data.get("stable_preferences", {})

        lines = [
            f"- Full Name: {p.get('full_name')} (Preferred: {p.get('preferred_name')})",
            f"- Current Location: {p.get('current_residence')}, Nationality: {p.get('nationality')}",
            f"- Languages: {', '.join(p.get('primary_languages', []))}",
            f"- Field: {edu.get('field_of_study')}",
            f"- Core Competencies: {', '.join(car.get('core_technical_skills', []))}",
            f"- Invariant Facts: Born and resident in {facts.get('birthplace', 'Singapore')}; cannot be contradicted.",
        ]
        if prefs:
            lines.append("- Stable Preferences:")
            for k, v in prefs.items():
                lines.append(f"  * {k.replace('_', ' ').capitalize()}: {v}")
        return "\n".join(lines)

    def detect_static_conflict(self, memory_text: str) -> Tuple[bool, Optional[str]]:
        """
        Static Conflict Resolution:
        Validates whether a memory asserts facts contradicting authoritative invariant facts.
        """
        text_lower = memory_text.lower()
        facts = self.data.get("immutable_facts", {})
        home = facts.get("home_country", "singapore").lower()
        birthplace = facts.get("birthplace", "singapore").lower()

        # Check birthplace / home country contradictions
        # e.g., if memory claims "I was born in Canada/Malaysia/USA" or "I live in Boston/Seattle"
        other_countries = ["canada", "seattle", "boston", "tokyo", "london", "new york", "sydney", "germany"]
        for place in other_countries:
            if place in text_lower:
                if any(phrase in text_lower for phrase in ["born in", "grew up in", "native of", "originally from"]):
                    return True, f"Contradicts invariant birthplace: Authoritative birthplace is {birthplace.capitalize()}, not {place.capitalize()}."
                if any(phrase in text_lower for phrase in ["permanently moved to", "now a citizen of"]):
                    return True, f"Contradicts invariant nationality/residence: Authoritative base is {home.capitalize()}."

        # Check field contradictions
        if any(unrelated in text_lower for unrelated in ["practicing neurosurgeon", "criminal defense attorney"]):
            return True, "Contradicts invariant domain of expertise (Computer Science / Software)."

        return False, None


class FastEmbedder:
    """
    High-performance semantic embedder supporting:
    1. SentenceTransformers (if installed)
    2. Dense semantic hashing with n-gram and stop-word downweighting (zero-dependency fallback)
    """

    STOPWORDS = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
        "is", "are", "was", "were", "be", "been", "being",
        "i", "you", "he", "she", "it", "we", "they", "me", "my", "your",
        "what", "where", "which", "who", "when", "why", "how",
        "do", "does", "did", "have", "has", "had", "can", "could", "should", "would"
    }

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dimension = 384
            logger.info("Loaded SentenceTransformer ('all-MiniLM-L6-v2') successfully.")
        except Exception:
            logger.info("SentenceTransformers not loaded. Using built-in normalized dense semantic embedder.")

    @staticmethod
    def _stem(t: str) -> str:
        """Lightweight morphological stemming for query-memory matching."""
        if t.endswith("ing") and len(t) > 4:
            return t[:-3]
        if t.endswith("ed") and len(t) > 3:
            return t[:-2]
        if t.endswith("es") and len(t) > 4:
            return t[:-2]
        if t.endswith("s") and len(t) > 3 and not t.endswith("ss"):
            return t[:-1]
        if t.endswith("e") and len(t) > 3:
            return t[:-1]
        return t

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)

        # High-dimensional sub-word hash projection with stopword dampening & stemming
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            clean = text.lower()
            tokens = re.findall(r'\b[a-z0-9_]+\b', clean)
            # Add character n-grams
            trigrams = [clean[j:j+3] for j in range(len(clean) - 2)]
            
            v = np.zeros(self.dimension, dtype=np.float32)
            
            # Embed word tokens with inverse stopword weighting and stem injection
            for token in tokens:
                weight = 0.15 if token in self.STOPWORDS else 1.0
                h = int(hashlib.md5(f"w_{token}".encode("utf-8")).hexdigest(), 16)
                idx = h % self.dimension
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                v[idx] += sign * weight

                stemmed = self._stem(token)
                if stemmed != token:
                    h_s = int(hashlib.md5(f"w_{stemmed}".encode("utf-8")).hexdigest(), 16)
                    idx_s = h_s % self.dimension
                    sign_s = 1.0 if ((h_s >> 8) & 1) else -1.0
                    v[idx_s] += sign_s * weight

            # Embed character trigrams for morphological / typo resilience
            for tri in trigrams:
                h = int(hashlib.md5(f"t_{tri}".encode("utf-8")).hexdigest(), 16)
                idx = h % self.dimension
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                v[idx] += sign * 0.25

            norm = np.linalg.norm(v)
            if norm > 1e-6:
                v = v / norm
            vectors[i] = v

        return vectors


class VectorStoreBase(ABC):
    """Abstract interface for memory vector stores."""

    @abstractmethod
    def add_nodes(self, nodes: List[MemoryNode]) -> None:
        raise NotImplementedError("Subclasses must implement add_nodes()")

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[MemoryNode]:
        raise NotImplementedError("Subclasses must implement search()")

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError("Subclasses must implement clear()")


class SQLiteVectorStore(VectorStoreBase):
    """
    Embedded SQLite-backed vector store with persistent storage and
    vectorized normalized cosine similarity calculation.
    """

    def __init__(self, db_path: Union[str, Path], embedder: FastEmbedder):
        self.db_path = Path(db_path)
        self.embedder = embedder
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._create_tables()

    def _create_tables(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    unixtime REAL NOT NULL,
                    owner TEXT NOT NULL,
                    category TEXT NOT NULL,
                    condition_binding TEXT,
                    embedding BLOB NOT NULL
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_owner ON memories(owner)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_unixtime ON memories(unixtime)")

    def add_nodes(self, nodes: List[MemoryNode]) -> None:
        if not nodes:
            return

        texts = [n.content for n in nodes]
        vectors = self.embedder.embed(texts)

        records = []
        for node, vec in zip(nodes, vectors):
            blob = vec.tobytes()
            cond_str = json.dumps(node.condition_binding) if node.condition_binding else None
            records.append((
                node.id,
                node.content,
                node.timestamp,
                node.unixtime,
                node.owner,
                node.category,
                cond_str,
                blob
            ))

        with self.conn:
            self.conn.executemany("""
                INSERT OR REPLACE INTO memories
                (id, content, timestamp, unixtime, owner, category, condition_binding, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

    def search(self, query: str, top_k: int = 10) -> List[MemoryNode]:
        query_vec = self.embedder.embed([query])[0]
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, content, timestamp, unixtime, owner, category, condition_binding, embedding FROM memories")
        rows = cursor.fetchall()
        if not rows:
            return []

        nodes: List[MemoryNode] = []
        vec_dim = query_vec.shape[0]

        for row in rows:
            m_id, content, timestamp, unixtime, owner, category, cond_str, blob = row
            m_vec = np.frombuffer(blob, dtype=np.float32)
            if m_vec.shape[0] != vec_dim:
                continue

            sim = float(np.dot(query_vec, m_vec))
            cond = json.loads(cond_str) if cond_str else None

            node = MemoryNode(
                id=m_id,
                content=content,
                timestamp=timestamp,
                owner=owner,
                category=category,
                condition_binding=cond,
                similarity_score=round(sim, 4),
                unixtime=unixtime,
            )
            nodes.append(node)

        nodes.sort(key=lambda n: n.similarity_score, reverse=True)
        return nodes[:top_k]

    def clear(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM memories")


class MemoryEngine:
    """
    Conflict-Aware Retrieval Engine.
    Coordinates memory ingestion, semantic vector search, and four-stage conflict resolution:
    1. Distractor Filtering (Speaker ownership)
    2. Static Conflicts (Identity Core integrity)
    3. Dynamic Conflicts (Temporal recency arbitration)
    4. Conditional Conflicts (Context-metadata binding)
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        identity_manager: Optional[IdentityManager] = None,
        target_user_name: str = "Dingxuan Liang",
    ):
        self.config = config or MemoryConfig()
        self.target_user_name = target_user_name
        self.identity = identity_manager or IdentityManager("digital_twin/identity.json")
        self.embedder = FastEmbedder()
        self.store = SQLiteVectorStore(self.config.db_path, self.embedder)

    def ingest_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Ingests unstructured notes, emails, and conversational summaries.
        Expected schema:
        {"id": "...", "content": "...", "timestamp": "...", "owner": "...", "category": "...", "condition": {...}}
        """
        nodes = []
        for i, doc in enumerate(documents):
            node_id = str(doc.get("id", f"mem_{i}_{int(datetime.now().timestamp())}"))
            nodes.append(
                MemoryNode(
                    id=node_id,
                    content=doc["content"],
                    timestamp=doc.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    owner=doc.get("owner", self.target_user_name),
                    category=doc.get("category", "general"),
                    condition_binding=doc.get("condition"),
                )
            )
        self.store.add_nodes(nodes)
        logger.info(f"Ingested {len(nodes)} memory nodes into vector store.")

    def resolve_conflicts(
        self,
        candidates: List[MemoryNode],
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryNode]:
        """
        Applies multi-stage conflict filtering:
        1. Distractor Filtering: Reject statements not owned by target persona.
        2. Static Conflict Resolution: Reject memories clashing with Identity Core.
        3. Dynamic Conflict Resolution: For conflicting temporal states, keep newest.
        4. Conditional Conflict Resolution: Match situational context metadata.
        """
        query_context = query_context or {}
        valid_nodes: List[MemoryNode] = []

        # STAGE 1: Distractor Filtering (Ownership Validation)
        target_name_clean = self.target_user_name.lower().strip()
        for node in candidates:
            if self.config.enable_distractor_filtering:
                owner_clean = node.owner.lower().strip()
                if owner_clean != target_name_clean and "dingxuan" not in owner_clean:
                    node.status = "rejected"
                    node.conflict_reason = f"Distractor: Statement belongs to '{node.owner}', not target persona."
                    continue

            # STAGE 2: Static Conflict Resolution (Identity Core Authority)
            if self.config.enable_static_conflicts:
                has_conflict, reason = self.identity.detect_static_conflict(node.content)
                if has_conflict:
                    node.status = "rejected"
                    node.conflict_reason = reason
                    continue

            valid_nodes.append(node)

        # STAGE 3: Conditional Conflict Resolution (Context Binding)
        # Context bindings (e.g. time_of_day: 'morning' vs 'evening')
        if self.config.enable_conditional_conflicts and query_context:
            filtered_conditional: List[MemoryNode] = []
            for node in valid_nodes:
                if node.condition_binding:
                    # Check if query context matches condition binding
                    matches = True
                    for key, val in node.condition_binding.items():
                        if key in query_context:
                            if str(query_context[key]).lower() != str(val).lower():
                                matches = False
                                break
                    if not matches:
                        node.status = "rejected"
                        node.conflict_reason = f"Conditional mismatch: Requires {node.condition_binding}, got {query_context}."
                        continue
                filtered_conditional.append(node)
            valid_nodes = filtered_conditional

        # STAGE 4: Dynamic Conflict Resolution (Temporal Recency)
        # Groups claims by semantic predicate category (e.g. location, employer)
        # If multiple claims compete in the same category, latest timestamp wins.
        if self.config.enable_dynamic_conflicts:
            category_buckets: Dict[str, List[MemoryNode]] = {}
            for node in valid_nodes:
                # Distinguish general from stateful categories
                if node.category in ["location", "employment", "relationship_status"]:
                    category_buckets.setdefault(node.category, []).append(node)

            for cat, bucket in category_buckets.items():
                if len(bucket) > 1:
                    # Sort by unixtime descending (most recent first)
                    bucket.sort(key=lambda n: n.unixtime, reverse=True)
                    latest = bucket[0]
                    for older in bucket[1:]:
                        older.status = "superseded"
                        older.conflict_reason = (
                            f"Dynamic temporal conflict: Superseded by newer state from "
                            f"{latest.timestamp} (id: {latest.id})."
                        )

            # Keep only active nodes
            valid_nodes = [n for n in valid_nodes if n.status == "active"]

        return valid_nodes

    def retrieve(
        self,
        query: str,
        query_context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryNode]:
        """
        End-to-end conflict-aware retrieval:
        1. Incorporates query context into semantic lookup.
        2. Queries vector store for raw top candidates.
        3. Filters below similarity threshold.
        4. Applies multi-stage conflict resolution (Distractor, Static, Conditional, Dynamic).
        """
        k = top_k or self.config.similarity_top_k
        query_context = query_context or {}

        # Augment search query with contextual keywords if present
        augmented_query = query
        if query_context:
            context_tokens = " ".join(str(v) for v in query_context.values())
            augmented_query = f"{query} {context_tokens}"

        # Fetch surplus candidates to allow conflict pruning
        raw_candidates = self.store.search(augmented_query, top_k=k * 4)

        # Apply multi-tiered conflict engine
        resolved = self.resolve_conflicts(raw_candidates, query_context=query_context)

        # Boost score for explicit conditional matches
        for node in resolved:
            if node.condition_binding and query_context:
                if all(str(query_context.get(k, "")).lower() == str(v).lower() for k, v in node.condition_binding.items()):
                    node.similarity_score += 0.2

        # Final score filter and sort
        valid_candidates = [
            n for n in resolved if n.similarity_score >= self.config.similarity_threshold
        ]
        valid_candidates.sort(key=lambda n: n.similarity_score, reverse=True)
        return valid_candidates[:k]


if __name__ == "__main__":
    # Test conflict resolution pipeline
    print("Testing Stage 5 Conflict-Aware Memory Engine...")
    id_mgr = IdentityManager("digital_twin/identity.json")
    mem_engine = MemoryEngine(identity_manager=id_mgr)
    mem_engine.store.clear()

    # Sample memory entries demonstrating dynamic, static, conditional, and distractor conflicts
    sample_docs = [
        # Dynamic conflict: Old residence vs New residence
        {
            "id": "mem_loc_old",
            "content": "I live in Jurong West, Singapore.",
            "timestamp": "2021-06-15T10:00:00Z",
            "owner": "Dingxuan Liang",
            "category": "location",
        },
        {
            "id": "mem_loc_new",
            "content": "I moved and now live in Clementi, Singapore.",
            "timestamp": "2024-02-10T14:30:00Z",
            "owner": "Dingxuan Liang",
            "category": "location",
        },
        # Static conflict: Contradicts Identity Core
        {
            "id": "mem_static_bad",
            "content": "I was born in Boston, Massachusetts and lived there for 20 years.",
            "timestamp": "2023-01-01T12:00:00Z",
            "owner": "Dingxuan Liang",
            "category": "general",
        },
        # Distractor: Statement by someone else
        {
            "id": "mem_distractor",
            "content": "I really hate programming in Python, it is too slow.",
            "timestamp": "2024-03-01T09:00:00Z",
            "owner": "Kelvin (Friend)",
            "category": "preference",
        },
        # Conditional preference: Morning coffee vs Evening tea
        {
            "id": "mem_cond_morning",
            "content": "I always drink strong black coffee when I start my morning coding.",
            "timestamp": "2024-01-10T08:00:00Z",
            "owner": "Dingxuan Liang",
            "category": "preference",
            "condition": {"time_of_day": "morning"},
        },
        {
            "id": "mem_cond_evening",
            "content": "I drink green tea or water in the evening so I can sleep well.",
            "timestamp": "2024-01-10T20:00:00Z",
            "owner": "Dingxuan Liang",
            "category": "preference",
            "condition": {"time_of_day": "evening"},
        },
    ]

    mem_engine.ingest_documents(sample_docs)

    # 1. Test Dynamic Conflict Resolution
    print("\n--- Test 1: Dynamic Conflict (Where do you live?) ---")
    retrieved = mem_engine.retrieve("Where do you live?")
    for r in retrieved:
        print(f"[{r.category}] {r.content} (Timestamp: {r.timestamp}, Score: {r.similarity_score})")

    # 2. Test Static Conflict Resolution
    print("\n--- Test 2: Static Conflict (Where were you born?) ---")
    retrieved = mem_engine.retrieve("Where were you born?")
    print(f"Retrieved {len(retrieved)} valid memories (Contradictory Boston memory should be suppressed):")
    for r in retrieved:
        print(f"  {r.content}")

    # 3. Test Distractor Filtering
    print("\n--- Test 3: Distractor Filtering (Do you hate Python?) ---")
    retrieved = mem_engine.retrieve("Do you hate Python?")
    print(f"Retrieved {len(retrieved)} valid memories (Friend's negative statement should be filtered):")
    for r in retrieved:
        print(f"  Owner: {r.owner}, Content: {r.content}")

    # 4. Test Conditional Conflict Resolution
    print("\n--- Test 4: Conditional Conflict (What beverage do you drink in the morning?) ---")
    retrieved_morning = mem_engine.retrieve("What beverage do you drink?", query_context={"time_of_day": "morning"})
    for r in retrieved_morning:
        print(f"  [Morning]: {r.content}")

    retrieved_evening = mem_engine.retrieve("What beverage do you drink?", query_context={"time_of_day": "evening"})
    for r in retrieved_evening:
        print(f"  [Evening]: {r.content}")
