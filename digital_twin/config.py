"""
Configuration and hyperparameters for the Digital Twin architecture.

Defines strongly-typed dataclasses for:
1. Chat preprocessing (burst window, session timeout, PII masking)
2. Personality extraction (metrics, psychometric assessment, exemplar filters)
3. Memory and vector RAG (conflict resolution thresholds, vector store type)
4. LLM orchestration and inference backends (Transformers, Ollama, API, Mock)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PreprocessorConfig:
    """Hyperparameters for Stage 1: Chat log cleaning and sessionization."""
    # Burst concatenation: maximum seconds between messages from same sender to merge
    burst_window_seconds: int = 180
    # Session segmentation: idle gap threshold in hours to split conversational threads
    session_idle_gap_hours: float = 18.0
    # Data sanitization
    anonymize_pii: bool = True
    mask_phone_numbers: bool = True
    mask_emails: bool = True
    mask_ip_addresses: bool = True
    filter_service_messages: bool = True
    # Target identity filters (defaulted to Dingxuan Liang from dataset)
    target_user_id: str = "5711494385"
    target_user_name: str = "Dingxuan Liang"


@dataclass
class PersonalityConfig:
    """Hyperparameters for Stage 2 & 3: Extraction and Few-shot mining."""
    # Minimum words for an exemplar response to be considered representative
    min_exemplar_words: int = 2
    max_exemplar_words: int = 120
    # Number of few-shot pairs to inject into system prompt
    few_shot_count: int = 4
    # Minimum stylistic score for exemplar curation
    min_style_score: float = 0.15
    # Big Five and MBTI extraction
    enable_psychometrics: bool = True


@dataclass
class MemoryConfig:
    """Hyperparameters for Stage 5: Memory engine & conflict-aware RAG."""
    # Vector store backend: 'sqlite' (zero-dependency, cosine similarity) or 'chromadb'
    vector_store_type: str = "sqlite"
    db_path: str = "digital_twin_memory.db"
    similarity_top_k: int = 5
    similarity_threshold: float = 0.15
    
    # Conflict resolution flags
    enable_dynamic_conflicts: bool = True       # Prioritize newer timestamps
    enable_static_conflicts: bool = True        # Identity Core absolute authority
    enable_conditional_conflicts: bool = True   # Time-of-day / context binding
    enable_distractor_filtering: bool = True    # Enforce owner == target_user


@dataclass
class LLMConfig:
    """Hyperparameters for Stage 6: Disciplined LLM Orchestration."""
    # Backend options: 'transformers', 'ollama', 'api', 'mock'
    backend: str = "mock"
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    ollama_endpoint: str = "http://localhost:11434"
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    
    # Refusal & uncertainty guardrail
    target_persona_name: str = "Dingxuan"
    refusal_phrase: str = "I don't have enough information to know how Dingxuan would answer that."


@dataclass
class TwinConfig:
    """Master configuration container for the entire Digital Twin pipeline."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    raw_chat_path: str = "telegramChatHistory.json"
    identity_file: str = "identity.json"
    
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def get_identity_path(self) -> Path:
        """Resolve path to identity.json."""
        candidate = Path(self.identity_file)
        if candidate.is_file():
            return candidate
        return self.base_dir / self.identity_file
