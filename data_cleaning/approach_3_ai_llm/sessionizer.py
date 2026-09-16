"""
Approach 3: AI/LLM Contextual Disentanglement Sessionizer
Optimized for private_group chats (messy multi-user conversations).
Uses Gemini API for thread disentanglement, with rule-based fallback.
"""

import json
import os
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

APPROACH_DIR = Path(__file__).resolve().parent
CLEANING_DIR = APPROACH_DIR.parent
PROJECT_ROOT = CLEANING_DIR.parent
for p in [str(PROJECT_ROOT), str(CLEANING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.models import Turn, Conversation, merge_into_turns
from shared.data_loader import prepare_messages


def sessionize_with_llm(
    chats: List[Dict[str, Any]],
    idle_gap_seconds: int = 7200,
    burst_window_seconds: int = 60,
    chunk_size: int = 50,
    min_turns: int = 2,
    require_target_user: bool = True,
    dry_run: bool = False,
) -> List[Conversation]:
    """
    Sessionizes group chats using LLM-based thread disentanglement.

    Pipeline:
    1. Pre-split by idle gap (default 2 hours)
    2. Burst-merge messages into turns (default 60s window)
    3. Batch turns into LLM-sized chunks (default 50 turns)
    4. Send each chunk to Gemini for thread assignment
    5. Fall back to rule-based if API unavailable
    6. Filter for training quality
    """
    # Determine whether to use the API
    use_api = not dry_run
    client = None

    if use_api:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[Info] GEMINI_API_KEY not set. Using rule-based fallback mode.")
            use_api = False

    if use_api:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            print("[Info] Gemini API client initialized successfully.")
        except ImportError:
            print("[Info] google-genai package not installed. Using rule-based fallback mode.")
            print("       Install with: pip install -r approach_3_ai_llm/requirements.txt")
            use_api = False
        except Exception as e:
            print(f"[Info] Failed to initialize Gemini client: {e}. Using fallback.")
            use_api = False

    if dry_run:
        print("[Info] Dry-run mode: using rule-based fallback (no API calls).")

    all_conversations: List[Conversation] = []
    conv_counter = 0

    for chat_idx, chat in enumerate(chats):
        chat_id = chat.get("id", 0)
        chat_name = chat.get("name", f"Chat {chat_id}")
        chat_type = chat.get("type", "private_group")

        messages = prepare_messages(chat)
        if not messages:
            continue

        print(f"  [{chat_idx + 1}/{len(chats)}] Processing '{chat_name}' ({len(messages)} messages)...")

        # Step 1: Pre-split by idle gap
        temporal_sessions: List[List[Dict]] = []
        current_session_msgs = [messages[0]]

        for i in range(1, len(messages)):
            time_diff = messages[i]["date_unixtime"] - messages[i - 1]["date_unixtime"]
            if time_diff > idle_gap_seconds:
                temporal_sessions.append(current_session_msgs)
                current_session_msgs = [messages[i]]
            else:
                current_session_msgs.append(messages[i])
        if current_session_msgs:
            temporal_sessions.append(current_session_msgs)

        print(f"    Pre-split into {len(temporal_sessions)} temporal sessions")

        for session_msgs in temporal_sessions:
            # Step 2: Burst merge
            turns = merge_into_turns(session_msgs, burst_window_seconds)
            if not turns:
                continue

            # Step 3: Batch into chunks
            chunks = [turns[i:i + chunk_size] for i in range(0, len(turns), chunk_size)]

            for chunk in chunks:
                if use_api:
                    # Step 4: LLM thread disentanglement
                    thread_groups = _llm_disentangle(client, chunk)
                else:
                    # Fallback: treat entire chunk as one thread
                    thread_groups = {"thread_1": chunk}

                # Create conversations from thread groups
                for thread_id, thread_turns in thread_groups.items():
                    if len(thread_turns) < min_turns:
                        continue
                    if require_target_user and not any(t.is_target_user for t in thread_turns):
                        continue

                    conv_counter += 1
                    conv = Conversation(
                        conversation_id=f"{chat_id}_llm_{conv_counter:05d}",
                        chat_id=chat_id,
                        chat_name=chat_name,
                        chat_type=chat_type,
                        approach="ai_llm",
                        turns=thread_turns,
                    )
                    conv.finalize()
                    all_conversations.append(conv)

    print(f"  Total conversations extracted: {len(all_conversations)}")
    return all_conversations


def _llm_disentangle(client: Any, chunk: List[Turn]) -> Dict[str, List[Turn]]:
    """
    Sends a chunk of turns to Gemini for thread assignment.
    Returns a dict mapping thread_id -> list of turns.
    Falls back to single-thread if API call fails.
    """
    prompt = _build_prompt(chunk)

    try:
        from google.genai import types

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        assignments = json.loads(response.text)
        thread_groups: Dict[str, List[Turn]] = {}

        for item in assignments:
            t_idx = item.get("turn_index")
            th_id = item.get("thread_id", "thread_default")
            if t_idx is not None and 0 <= t_idx < len(chunk):
                thread_groups.setdefault(th_id, []).append(chunk[t_idx])

        # Rate limiting
        time.sleep(1)

        if thread_groups:
            return thread_groups

    except json.JSONDecodeError:
        print("    [Fallback] Failed to parse LLM JSON response.")
    except Exception as e:
        print(f"    [Fallback] API call failed: {e}")

    # Fallback: single thread
    return {"thread_fallback": chunk}


def _build_prompt(turns: List[Turn]) -> str:
    """Builds the LLM prompt for thread disentanglement."""
    lines = [
        "You are analyzing a multi-user group chat. Your task is to identify distinct conversation threads.",
        "",
        'Each turn below has an index. Assign each turn to a thread_id (e.g., "thread_1", "thread_2", etc.).',
        "Messages about the same topic, or that are responses to the same discussion, should share a thread_id.",
        "",
        "Turns:",
    ]

    for i, turn in enumerate(turns):
        sender = turn.sender_name or turn.sender_id
        text_preview = turn.text[:200] if turn.text else "[empty]"
        lines.append(f"Index {i}: [{sender}] {text_preview}")

    lines.append("")
    lines.append('Respond with ONLY a JSON array: [{"turn_index": 0, "thread_id": "thread_1"}, ...]')
    return "\n".join(lines)
