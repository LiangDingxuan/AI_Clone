"""
Data Cleaning & Sessionization Pipeline Runner.

Executes all Telegram chat history cleaning, sorting, and sessionization steps:
  Step 1: Sort raw Telegram export by chat type (chat_sorter.py)
  Step 2: Approach 1 - Rule-based sessionizer for personal chats (DMs)
  Step 3: Approach 2 - NLP embedding semantic sessionizer for supergroups
  Step 4: Approach 3 - AI/LLM contextual disentanglement for private groups
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root and data_cleaning to sys.path
CLEANING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLEANING_DIR.parent
for p in [str(PROJECT_ROOT), str(CLEANING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_cleaning.chat_sorter import (
    export_sorted_chats,
    load_telegram_data,
    print_summary_table,
    sort_chats_by_type,
)
from data_cleaning.shared.data_loader import get_chats_by_types, load_sorted_data
from data_cleaning.shared.models import export_conversations
from data_cleaning.approach_1_rule_based.sessionizer import sessionize_personal_chats
from data_cleaning.approach_2_nlp_embeddings.sessionizer import sessionize_with_embeddings
from data_cleaning.approach_3_ai_llm.sessionizer import sessionize_with_llm


def resolve_file(path_str: str, default_dir: Path) -> Path:
    """Finds a file in path_str directly, under default_dir, or under PROJECT_ROOT."""
    p = Path(path_str)
    if p.is_file():
        return p
    if (default_dir / path_str).is_file():
        return default_dir / path_str
    if (PROJECT_ROOT / path_str).is_file():
        return PROJECT_ROOT / path_str
    return default_dir / path_str


def run_cleaning_pipeline(
    raw_input: str = "telegramChatHistory.json",
    sorted_output: str = "sorted_chats_by_type.json",
    summary_output: str = "chats_summary.json",
    skip_step1: bool = False,
    skip_approach1: bool = False,
    skip_approach2: bool = False,
    skip_approach3: bool = False,
    idle_gap_dm: float = 3.0,
    burst_window_dm: int = 90,
    idle_gap_supergroup: float = 4.0,
    burst_window_supergroup: int = 120,
    similarity_threshold: float = 0.3,
    supergroup_model: str = "all-MiniLM-L6-v2",
    idle_gap_group: float = 2.0,
    burst_window_group: int = 60,
    chunk_size_group: int = 50,
    llm_dry_run: bool = False,
):
    start_total_time = time.time()

    print("=" * 80)
    print(" DATA CLEANING & SESSIONIZATION PIPELINE - LIANG DINGXUAN")
    print(" Categorization & Multi-Approach Conversational Sessionization")
    print("=" * 80)

    # STEP 1: Sort raw export
    sorted_path = resolve_file(sorted_output, CLEANING_DIR)
    if skip_step1 and sorted_path.exists():
        print(f"\n[Step 1/4] Skipping raw chat sorting (using existing {sorted_path}).")
    else:
        print("\n" + "-" * 80)
        print("[Step 1/4] Sorting Raw Telegram Export by Chat Type")
        print("-" * 80)
        step1_start = time.time()

        raw_path = resolve_file(raw_input, PROJECT_ROOT)
        print(f"Loading raw export: {raw_path}...")
        raw_data = load_telegram_data(str(raw_path))

        print("Categorizing and filtering out empty chats (total_messages <= 0)...")
        metadata, sorted_chats = sort_chats_by_type(
            data=raw_data,
            include_left_chats=True,
            min_messages=1,
            clean_text=True,
            remove_empty_service=True,
        )

        print_summary_table(metadata)

        summary_path = CLEANING_DIR / summary_output
        export_sorted_chats(
            metadata=metadata,
            sorted_chats=sorted_chats,
            output_path=str(sorted_path),
            summary_path=str(summary_path),
        )
        print(f"[Step 1 Complete] Took {time.time() - step1_start:.2f}s")

    sorted_data = load_sorted_data(sorted_path)
    results_summary = {}

    # STEP 2: Approach 1 - Personal Chats
    if skip_approach1:
        print("\n[Step 2/4] Skipping Approach 1 (Personal Chats).")
    else:
        print("\n" + "-" * 80)
        print("[Step 2/4] Approach 1: Rule-Based Sessionizer (1-on-1 Personal Chats)")
        print(f"  Method: Idle Gap ({idle_gap_dm}h) + Burst Merge ({burst_window_dm}s) + Reply-Chain Union-Find")
        print("-" * 80)
        step2_start = time.time()

        dm_chats = get_chats_by_types(sorted_data, ["personal_chat"])
        dm_output = CLEANING_DIR / "approach_1_rule_based" / "output" / "sessions_personal_chat.json"

        dm_convs = sessionize_personal_chats(
            chats=dm_chats,
            idle_gap_seconds=int(idle_gap_dm * 3600),
            burst_window_seconds=burst_window_dm,
            min_turns=2,
            require_target_user=True,
        )

        export_conversations(dm_convs, str(dm_output), approach_name="rule_based")
        results_summary["Approach 1 (Personal Chats)"] = {
            "conversations": len(dm_convs),
            "target_turns": sum(c.target_user_turn_count for c in dm_convs),
            "total_turns": sum(c.turn_count for c in dm_convs),
            "output_file": dm_output,
        }
        print(f"[Step 2 Complete] Took {time.time() - step2_start:.2f}s")

    # STEP 3: Approach 2 - Supergroups
    if skip_approach2:
        print("\n[Step 3/4] Skipping Approach 2 (Supergroups).")
    else:
        print("\n" + "-" * 80)
        print("[Step 3/4] Approach 2: NLP Embeddings Semantic Sessionizer (Supergroups)")
        print(f"  Method: SentenceTransformer ('{supergroup_model}') + Cosine Similarity (<{similarity_threshold})")
        print("-" * 80)
        step3_start = time.time()

        supergroup_chats = get_chats_by_types(sorted_data, ["private_supergroup"])
        sg_output = CLEANING_DIR / "approach_2_nlp_embeddings" / "output" / "sessions_supergroup.json"

        sg_convs = sessionize_with_embeddings(
            chats=supergroup_chats,
            idle_gap_seconds=int(idle_gap_supergroup * 3600),
            burst_window_seconds=burst_window_supergroup,
            similarity_threshold=similarity_threshold,
            min_turns=2,
            require_target_user=True,
            model_name=supergroup_model,
        )

        export_conversations(sg_convs, str(sg_output), approach_name="nlp_embeddings")
        results_summary["Approach 2 (Supergroups)"] = {
            "conversations": len(sg_convs),
            "target_turns": sum(c.target_user_turn_count for c in sg_convs),
            "total_turns": sum(c.turn_count for c in sg_convs),
            "output_file": sg_output,
        }
        print(f"[Step 3 Complete] Took {time.time() - step3_start:.2f}s")

    # STEP 4: Approach 3 - Private Groups
    if skip_approach3:
        print("\n[Step 4/4] Skipping Approach 3 (Private Groups).")
    else:
        print("\n" + "-" * 80)
        print("[Step 4/4] Approach 3: AI/LLM Contextual Disentanglement (Private Groups)")
        has_key = bool(os.environ.get("GEMINI_API_KEY"))
        mode_str = "Gemini 2.5 Flash API" if (has_key and not llm_dry_run) else "Deterministic Fallback (Dry-Run)"
        print(f"  Mode: {mode_str} | Idle Gap: {idle_gap_group}h | Chunk Size: {chunk_size_group} turns")
        print("-" * 80)
        step4_start = time.time()

        group_chats = get_chats_by_types(sorted_data, ["private_group"])
        grp_output = CLEANING_DIR / "approach_3_ai_llm" / "output" / "sessions_group.json"

        grp_convs = sessionize_with_llm(
            chats=group_chats,
            idle_gap_seconds=int(idle_gap_group * 3600),
            burst_window_seconds=burst_window_group,
            chunk_size=chunk_size_group,
            min_turns=2,
            require_target_user=True,
            dry_run=llm_dry_run or (not has_key),
        )

        export_conversations(grp_convs, str(grp_output), approach_name="ai_llm")
        results_summary["Approach 3 (Private Groups)"] = {
            "conversations": len(grp_convs),
            "target_turns": sum(c.target_user_turn_count for c in grp_convs),
            "total_turns": sum(c.turn_count for c in grp_convs),
            "output_file": grp_output,
        }
        print(f"[Step 4 Complete] Took {time.time() - step4_start:.2f}s")

    total_pipeline_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(" DATA CLEANING PIPELINE COMPLETE")
    print(f" Total Elapsed Time: {total_pipeline_time:.2f}s")
    print("=" * 80)
    for name, stats in results_summary.items():
        print(f"  - {name}: {stats['conversations']} sessions -> {stats['output_file']}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Run data cleaning and sessionization pipeline for AI Clone."
    )
    parser.add_argument("--input", "-i", default="telegramChatHistory.json", help="Path to raw Telegram JSON export")
    parser.add_argument("--sorted-output", default="sorted_chats_by_type.json", help="Path to sorted chats JSON")
    parser.add_argument("--summary-output", default="chats_summary.json", help="Path to summary index JSON")
    parser.add_argument("--skip-step1", action="store_true", help="Skip Step 1 (raw chat sorting) if sorted file exists")
    parser.add_argument("--skip-approach1", action="store_true", help="Skip Approach 1 (Personal Chats)")
    parser.add_argument("--skip-approach2", action="store_true", help="Skip Approach 2 (Supergroups)")
    parser.add_argument("--skip-approach3", action="store_true", help="Skip Approach 3 (Private Groups)")
    parser.add_argument("--idle-gap-dm", type=float, default=3.0, help="Idle gap in hours for personal chats (default: 3.0)")
    parser.add_argument("--burst-window-dm", type=int, default=90, help="Burst window in seconds for personal chats (default: 90)")
    parser.add_argument("--idle-gap-supergroup", type=float, default=4.0, help="Idle gap in hours for supergroups (default: 4.0)")
    parser.add_argument("--burst-window-supergroup", type=int, default=120, help="Burst window in seconds for supergroups (default: 120)")
    parser.add_argument("--similarity-threshold", type=float, default=0.3, help="Cosine similarity threshold for topic shift (default: 0.3)")
    parser.add_argument("--supergroup-model", default="all-MiniLM-L6-v2", help="Embedding model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--idle-gap-group", type=float, default=2.0, help="Idle gap in hours for group chats (default: 2.0)")
    parser.add_argument("--burst-window-group", type=int, default=60, help="Burst window in seconds for group chats (default: 60)")
    parser.add_argument("--chunk-size-group", type=int, default=50, help="Turn chunk size for group chats (default: 50)")
    parser.add_argument("--dry-run-llm", action="store_true", help="Force dry-run fallback mode for Approach 3")

    args = parser.parse_args()

    run_cleaning_pipeline(
        raw_input=args.input,
        sorted_output=args.sorted_output,
        summary_output=args.summary_output,
        skip_step1=args.skip_step1,
        skip_approach1=args.skip_approach1,
        skip_approach2=args.skip_approach2,
        skip_approach3=args.skip_approach3,
        idle_gap_dm=args.idle_gap_dm,
        burst_window_dm=args.burst_window_dm,
        idle_gap_supergroup=args.idle_gap_supergroup,
        burst_window_supergroup=args.burst_window_supergroup,
        similarity_threshold=args.similarity_threshold,
        supergroup_model=args.supergroup_model,
        idle_gap_group=args.idle_gap_group,
        burst_window_group=args.burst_window_group,
        chunk_size_group=args.chunk_size_group,
        llm_dry_run=args.dry_run_llm,
    )


if __name__ == "__main__":
    main()
