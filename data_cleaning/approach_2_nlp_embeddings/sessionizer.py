"""
Approach 2: NLP Embeddings Semantic Sessionizer
Optimized for private_supergroup chats (formal/long-form discussions).
Uses sentence-transformers to detect topic shifts via cosine similarity.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

APPROACH_DIR = Path(__file__).resolve().parent
CLEANING_DIR = APPROACH_DIR.parent
PROJECT_ROOT = CLEANING_DIR.parent
for p in [str(PROJECT_ROOT), str(CLEANING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.models import Turn, Conversation, merge_into_turns
from shared.data_loader import prepare_messages


def sessionize_with_embeddings(
    chats: List[Dict[str, Any]],
    idle_gap_seconds: int = 14400,
    burst_window_seconds: int = 120,
    similarity_threshold: float = 0.3,
    min_turns: int = 2,
    require_target_user: bool = True,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Conversation]:
    """
    Sessionizes supergroup chats using sentence embeddings and cosine similarity.

    Pipeline:
    1. Pre-split by idle gap (default 4 hours)
    2. Burst-merge messages into turns (default 120s window)
    3. Embed each turn's text with sentence-transformers
    4. Split when cosine similarity between consecutive turns drops below threshold
    5. Filter for training quality (min turns, target user participation)
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("[Error] Required packages not installed.")
        print("Please run: pip install -r approach_2_nlp_embeddings/requirements.txt")
        sys.exit(1)

    print(f"Loading sentence-transformers model: {model_name}...")
    model = SentenceTransformer(model_name)
    embed_dim = model.get_sentence_embedding_dimension()

    all_conversations: List[Conversation] = []
    conv_counter = 0

    for chat_idx, chat in enumerate(chats):
        chat_id = chat.get("id", 0)
        chat_name = chat.get("name", f"Chat {chat_id}")
        chat_type = chat.get("type", "private_supergroup")

        messages = prepare_messages(chat)
        if not messages:
            continue

        print(f"  [{chat_idx + 1}/{len(chats)}] Processing '{chat_name}' ({len(messages)} messages)...")

        # Step 1: Pre-split by idle gap
        coarse_sessions: List[List[Dict]] = []
        current_session_msgs = [messages[0]]

        for i in range(1, len(messages)):
            time_diff = messages[i]["date_unixtime"] - messages[i - 1]["date_unixtime"]
            if time_diff > idle_gap_seconds:
                coarse_sessions.append(current_session_msgs)
                current_session_msgs = [messages[i]]
            else:
                current_session_msgs.append(messages[i])
        if current_session_msgs:
            coarse_sessions.append(current_session_msgs)

        print(f"    Pre-split into {len(coarse_sessions)} coarse sessions by {idle_gap_seconds // 3600}h idle gap")

        # Step 2-4: For each coarse session, merge → embed → split by similarity
        for session_msgs in coarse_sessions:
            turns = merge_into_turns(session_msgs, burst_window_seconds)
            if len(turns) < min_turns:
                continue

            # Step 3: Embed turn texts
            turn_texts = [t.text.strip() if t.text.strip() else "[empty]" for t in turns]
            embeddings = model.encode(turn_texts, show_progress_bar=False)

            # Step 4: Cosine similarity topic detection
            current_topic_turns: List[Turn] = [turns[0]]

            for i in range(1, len(turns)):
                prev_emb = embeddings[i - 1].reshape(1, -1)
                curr_emb = embeddings[i].reshape(1, -1)

                # Handle zero vectors
                if np.linalg.norm(prev_emb) < 1e-8 or np.linalg.norm(curr_emb) < 1e-8:
                    sim = 0.0
                else:
                    sim = float(cosine_similarity(prev_emb, curr_emb)[0][0])

                if sim < similarity_threshold:
                    # Topic shifted — finalize current conversation
                    conv_counter += 1
                    conv = Conversation(
                        conversation_id=f"{chat_id}_emb_{conv_counter:05d}",
                        chat_id=chat_id,
                        chat_name=chat_name,
                        chat_type=chat_type,
                        approach="nlp_embeddings",
                        turns=current_topic_turns,
                    )
                    conv.finalize()
                    all_conversations.append(conv)
                    current_topic_turns = [turns[i]]
                else:
                    current_topic_turns.append(turns[i])

            # Flush remaining turns
            if current_topic_turns:
                conv_counter += 1
                conv = Conversation(
                    conversation_id=f"{chat_id}_emb_{conv_counter:05d}",
                    chat_id=chat_id,
                    chat_name=chat_name,
                    chat_type=chat_type,
                    approach="nlp_embeddings",
                    turns=current_topic_turns,
                )
                conv.finalize()
                all_conversations.append(conv)

    # Step 5: Training quality filter
    before_count = len(all_conversations)
    filtered: List[Conversation] = []

    for conv in all_conversations:
        if conv.turn_count < min_turns:
            continue
        if require_target_user and not conv.has_target_user_participation:
            continue
        filtered.append(conv)

    print(f"  Quality filter: {before_count} → {len(filtered)} conversations "
          f"(removed {before_count - len(filtered)} without target user or < {min_turns} turns)")

    return filtered
