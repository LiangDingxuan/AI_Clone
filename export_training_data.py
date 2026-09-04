"""
Training Data Exporter and Distractor Filter (export_training_data.py)
Author: AI Software Engineer
Project: Context-Adaptive Personality Chatbot ("Dingxuan")

Transforms clean sessionized conversations from Approach 1, 2, and 3 into
standardized conversational training datasets (ChatML/OpenAI JSONL format)
for Supervised Fine-Tuning (SFT / QLoRA).

Features:
1. Strips third-party distractor artifacts (bot commands, system events, media tags).
2. Formats multi-party group chat context using bracketed sender tags [Name]:.
3. Enforces strict conversational turn alternation: system -> user -> assistant.
4. Injects context-adaptive system prompts synthesized by SystemPromptSynthesizer.
5. Performs stratified 85/15 train/val split across personal, group, and supergroup contexts.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from profiler import MultiContextProfiler, is_dingxuan_sender, load_context_sessions, normalize_session_turns
from synthesizer import SystemPromptSynthesizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_training_data")

# Regex for detecting non-conversational noise
SERVICE_EVENT_PATTERNS = [
    re.compile(r"^(pinned\s+a\s+message|joined\s+the\s+group|left\s+the\s+group|changed\s+the\s+group)", re.IGNORECASE),
    re.compile(r"^<media\s+omitted>$", re.IGNORECASE),
    re.compile(r"^http[s]?://\S+$", re.IGNORECASE),  # Standalone URLs with no commentary
    re.compile(r"^/[a-zA-Z0-9_]+(\s|$)", re.IGNORECASE),  # Bot commands like /start, /help
]


def is_distractor_message(text: str) -> bool:
    """Returns True if the message is a non-conversational artifact or service notification."""
    clean = text.strip()
    if not clean:
        return True
    for pat in SERVICE_EVENT_PATTERNS:
        if pat.search(clean):
            return True
    return False


def format_conversation_for_sft(
    turns: List[Dict[str, Any]],
    system_prompt: str,
    context_type: str,
) -> Optional[Dict[str, List[Dict[str, str]]]]:
    """
    Converts a list of turns into a strictly alternating ChatML conversation:
    [
      {"role": "system", "content": ...},
      {"role": "user", "content": ...},
      {"role": "assistant", "content": ...}
    ]

    Rules:
    1. Skip leading assistant messages (so conversation opens with user inquiry/statement).
    2. Merge consecutive other-user messages into a single user turn:
       - In group/supergroup: prefix with sender name '[Sender]: text' to avoid third-party blur.
       - In personal_chat: simple newline concatenation.
    3. Merge consecutive Dingxuan messages into a single assistant turn.
    4. Ensure the episode ends with an assistant turn (omitting dangling user queries).
    5. Discard episodes with fewer than 1 complete (user, assistant) exchange.
    """
    # 1. Clean turns
    cleaned_turns: List[Dict[str, Any]] = []
    for t in turns:
        txt = t["text"].strip()
        if not txt or is_distractor_message(txt):
            continue
        cleaned_turns.append({
            "sender_name": t.get("sender_name") or "User",
            "text": txt,
            "is_target_user": bool(t.get("is_target_user")),
        })

    if not cleaned_turns:
        return None

    # 2. Skip leading assistant turns
    start_idx = 0
    while start_idx < len(cleaned_turns) and cleaned_turns[start_idx]["is_target_user"]:
        start_idx += 1

    if start_idx >= len(cleaned_turns):
        return None

    cleaned_turns = cleaned_turns[start_idx:]

    # 3. Build alternating blocks
    is_multi_party = context_type in ["group", "supergroup"]
    chat_messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    current_role: Optional[str] = None
    current_buffer: List[str] = []

    for t in cleaned_turns:
        role = "assistant" if t["is_target_user"] else "user"
        txt = t["text"]
        sender = t["sender_name"].strip()

        # Format turn text
        if role == "user" and is_multi_party:
            formatted_turn = f"[{sender}]: {txt}"
        else:
            formatted_turn = txt

        if role == current_role:
            current_buffer.append(formatted_turn)
        else:
            if current_role is not None and current_buffer:
                chat_messages.append({
                    "role": current_role,
                    "content": "\n".join(current_buffer),
                })
            current_role = role
            current_buffer = [formatted_turn]

    # Flush remaining buffer
    if current_role is not None and current_buffer:
        chat_messages.append({
            "role": current_role,
            "content": "\n".join(current_buffer),
        })

    # 4. Enforce ending on assistant turn
    while len(chat_messages) > 1 and chat_messages[-1]["role"] != "assistant":
        chat_messages.pop()

    # 5. Validation: Need at least 1 user turn and 1 assistant turn
    user_turns = sum(1 for m in chat_messages if m["role"] == "user")
    assistant_turns = sum(1 for m in chat_messages if m["role"] == "assistant")

    if user_turns < 1 or assistant_turns < 1:
        return None

    return {"messages": chat_messages}


class DatasetExporter:
    """Orchestrates multi-context loading, formatting, splitting, and JSONL export."""

    DEFAULT_SOURCES = {
        "personal_chat": Path("approach_1_rule_based/output/sessions_personal_chat.json"),
        "supergroup": Path("approach_2_nlp_embeddings/output/sessions_supergroup.json"),
        "group": Path("approach_3_ai_llm/output/sessions_group.json"),
    }

    def __init__(
        self,
        profiler: Optional[MultiContextProfiler] = None,
        synthesizer: Optional[SystemPromptSynthesizer] = None,
    ):
        self.profiler = profiler or MultiContextProfiler()
        self.synthesizer = synthesizer or SystemPromptSynthesizer(self.profiler)

    def export(
        self,
        output_dir: Path | str = "data",
        val_ratio: float = 0.15,
        seed: int = 42,
        custom_paths: Optional[Dict[str, Path | str]] = None,
    ) -> Dict[str, Any]:
        random.seed(seed)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        sources = custom_paths or self.DEFAULT_SOURCES

        episodes_by_context: Dict[str, List[Dict[str, Any]]] = {
            "personal_chat": [],
            "supergroup": [],
            "group": [],
        }

        total_sessions_inspected = 0
        total_dingxuan_turns_modeled = 0

        for ctx_key, file_path in sources.items():
            path = Path(file_path)
            if not path.is_file():
                logger.warning("Source file not found for context '%s': %s", ctx_key, path)
                continue

            sessions = load_context_sessions(path)
            total_sessions_inspected += len(sessions)

            # Build context system prompt
            system_prompt = self.synthesizer.build_system_prompt(context_type=ctx_key)

            for sess in sessions:
                raw_turns = normalize_session_turns(sess)
                sample = format_conversation_for_sft(
                    turns=raw_turns,
                    system_prompt=system_prompt,
                    context_type=ctx_key,
                )
                if sample:
                    episodes_by_context[ctx_key].append(sample)
                    # Count assistant turns
                    asst_count = sum(1 for m in sample["messages"] if m["role"] == "assistant")
                    total_dingxuan_turns_modeled += asst_count

        train_episodes: List[Dict[str, Any]] = []
        val_episodes: List[Dict[str, Any]] = []
        context_stats: Dict[str, Dict[str, int]] = {}

        # Stratified split per context
        for ctx_key, episodes in episodes_by_context.items():
            random.shuffle(episodes)
            n_total = len(episodes)
            n_val = max(1, int(n_total * val_ratio)) if n_total > 0 else 0
            n_train = n_total - n_val

            val_split = episodes[:n_val]
            train_split = episodes[n_val:]

            train_episodes.extend(train_split)
            val_episodes.extend(val_split)

            context_stats[ctx_key] = {
                "total_valid_episodes": n_total,
                "train_episodes": n_train,
                "val_episodes": n_val,
            }

        # Shuffle final combined pools
        random.shuffle(train_episodes)
        random.shuffle(val_episodes)

        train_file = out_path / "train.jsonl"
        val_file = out_path / "val.jsonl"
        summary_file = out_path / "training_data_summary.json"

        # Write train.jsonl
        with open(train_file, "w", encoding="utf-8") as f:
            for item in train_episodes:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Write val.jsonl
        with open(val_file, "w", encoding="utf-8") as f:
            for item in val_episodes:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        summary = {
            "total_raw_sessions_inspected": total_sessions_inspected,
            "total_formatted_episodes": len(train_episodes) + len(val_episodes),
            "train_episodes": len(train_episodes),
            "val_episodes": len(val_episodes),
            "val_ratio": val_ratio,
            "total_assistant_turns_modeled": total_dingxuan_turns_modeled,
            "context_breakdown": context_stats,
            "train_file": str(train_file),
            "val_file": str(val_file),
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(
            "Export complete: %d train episodes, %d val episodes across %d modeled assistant turns.",
            len(train_episodes),
            len(val_episodes),
            total_dingxuan_turns_modeled,
        )
        return summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Export cleaned sessions into QLoRA SFT training datasets.")
    parser.add_argument("--output-dir", "-o", default="data", help="Output directory for JSONL files")
    parser.add_argument("--val-ratio", "-v", type=float, default=0.15, help="Validation set ratio (default: 0.15)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    exporter = DatasetExporter()
    summary = exporter.export(output_dir=args.output_dir, val_ratio=args.val_ratio, seed=args.seed)

    print("=" * 70)
    print(" TRAINING DATASET EXPORT SUMMARY")
    print("=" * 70)
    print(f" Total Sessions Inspected : {summary['total_raw_sessions_inspected']}")
    print(f" Formatted SFT Episodes   : {summary['total_formatted_episodes']}")
    print(f" Train Split              : {summary['train_episodes']} episodes -> {summary['train_file']}")
    print(f" Validation Split         : {summary['val_episodes']} episodes -> {summary['val_file']}")
    print(f" Dingxuan Turns Modeled   : {summary['total_assistant_turns_modeled']}")
    print(" Breakdown by Context:")
    for ctx, stats in summary["context_breakdown"].items():
        print(f"   - {ctx:15s}: {stats['total_valid_episodes']} episodes (Train: {stats['train_episodes']}, Val: {stats['val_episodes']})")
    print("=" * 70)


if __name__ == "__main__":
    main()
