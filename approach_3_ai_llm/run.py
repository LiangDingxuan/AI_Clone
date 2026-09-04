"""
Approach 3 CLI: AI/LLM Contextual Disentanglement for Group Chats.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from shared.data_loader import load_sorted_data, get_chats_by_types
from shared.models import export_conversations


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Approach 3: AI/LLM Contextual Disentanglement for group chats."
    )
    parser.add_argument("--input", "-i", default="sorted_chats_by_type.json",
                        help="Path to sorted_chats_by_type.json (default: sorted_chats_by_type.json)")
    parser.add_argument("--output", "-o", default="approach_3_ai_llm/output/sessions_group.json",
                        help="Output path (default: approach_3_ai_llm/output/sessions_group.json)")
    parser.add_argument("--idle-gap", type=float, default=2.0,
                        help="Idle gap in hours (default: 2.0)")
    parser.add_argument("--burst-window", type=int, default=60,
                        help="Burst merge window in seconds (default: 60)")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Turns per LLM chunk (default: 50)")
    parser.add_argument("--min-turns", type=int, default=2,
                        help="Minimum turns per conversation (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use rule-based fallback without making API calls")

    args = parser.parse_args()

    input_path = project_root / args.input
    output_path = project_root / args.output

    data = load_sorted_data(str(input_path))
    chats = get_chats_by_types(data, ["private_group"])

    if not chats:
        print("No private_group chats found. Nothing to process.")
        return

    from sessionizer import sessionize_with_llm

    idle_gap_seconds = int(args.idle_gap * 3600)

    print(f"\nSessionizing {len(chats)} group chats...")
    print(f"  Idle gap: {args.idle_gap}h ({idle_gap_seconds}s)")
    print(f"  Burst window: {args.burst_window}s")
    print(f"  Chunk size: {args.chunk_size} turns")
    print(f"  Dry run: {args.dry_run}")
    print()

    conversations = sessionize_with_llm(
        chats=chats,
        idle_gap_seconds=idle_gap_seconds,
        burst_window_seconds=args.burst_window,
        chunk_size=args.chunk_size,
        min_turns=args.min_turns,
        require_target_user=True,
        dry_run=args.dry_run,
    )

    export_conversations(conversations, str(output_path), approach_name="ai_llm")

    print(f"\nApproach 3 complete! {len(conversations)} conversations extracted.")


if __name__ == "__main__":
    main()
