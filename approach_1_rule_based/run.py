"""
Approach 1 CLI: Rule-Based Sessionizer for Personal Chats (DMs).
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.data_loader import load_sorted_data, get_chats_by_types
from shared.models import export_conversations


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Approach 1: Rule-Based Sessionizer for personal chats (DMs)."
    )
    parser.add_argument("--input", "-i", default="sorted_chats_by_type.json",
                        help="Path to sorted_chats_by_type.json (default: sorted_chats_by_type.json)")
    parser.add_argument("--output", "-o", default="approach_1_rule_based/output/sessions_personal_chat.json",
                        help="Output path (default: approach_1_rule_based/output/sessions_personal_chat.json)")
    parser.add_argument("--idle-gap", type=float, default=3.0,
                        help="Idle gap in hours (default: 3.0)")
    parser.add_argument("--burst-window", type=int, default=90,
                        help="Burst merge window in seconds (default: 90)")
    parser.add_argument("--min-turns", type=int, default=2,
                        help="Minimum turns per conversation (default: 2)")

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    input_path = root_dir / args.input
    output_path = root_dir / args.output

    data = load_sorted_data(str(input_path))
    chats = get_chats_by_types(data, ["personal_chat"])

    if not chats:
        print("No personal_chat chats found. Nothing to process.")
        return

    from sessionizer import sessionize_personal_chats

    idle_gap_seconds = int(args.idle_gap * 3600)

    print(f"\nSessionizing {len(chats)} personal chats...")
    print(f"  Idle gap: {args.idle_gap}h ({idle_gap_seconds}s)")
    print(f"  Burst window: {args.burst_window}s")
    print()

    conversations = sessionize_personal_chats(
        chats=chats,
        idle_gap_seconds=idle_gap_seconds,
        burst_window_seconds=args.burst_window,
        min_turns=args.min_turns,
        require_target_user=True,
    )

    export_conversations(conversations, str(output_path), approach_name="rule_based")

    print(f"\nApproach 1 complete! {len(conversations)} conversations extracted.")


if __name__ == "__main__":
    main()
