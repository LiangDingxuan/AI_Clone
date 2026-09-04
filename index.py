"""
Main entry point for AI Clone dataset preprocessing.
Sorts and cleans telegramChatHistory.json by chat type.
"""

import argparse
import sys
from pathlib import Path

from chat_sorter import (
    export_sorted_chats,
    load_telegram_data,
    print_summary_table,
    sort_chats_by_type,
)


def main():
    # Ensure UTF-8 output encoding in Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Sort and clean Telegram chat history by chat type for AI personality cloning."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="telegramChatHistory.json",
        help="Path to source Telegram JSON export (default: telegramChatHistory.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="sorted_chats_by_type.json",
        help="Path to output sorted JSON file (default: sorted_chats_by_type.json)",
    )
    parser.add_argument(
        "--summary-output",
        "-s",
        default="chats_summary.json",
        help="Path to lightweight summary JSON file (default: chats_summary.json)",
    )
    parser.add_argument(
        "--min-messages",
        "-m",
        type=int,
        default=1,
        help="Minimum total messages required to include a chat (default: 1, excludes empty chats)",
    )
    parser.add_argument(
        "--exclude-left",
        action="store_true",
        help="Exclude left/archived chats from export (default: False, left chats are included)",
    )
    parser.add_argument(
        "--keep-empty-service",
        action="store_true",
        help="Keep empty service messages like join/leave events (default: False, removes them)",
    )

    args = parser.parse_args()

    print(f"Loading Telegram data from: {args.input}")
    data = load_telegram_data(args.input)

    print("Sorting and cleaning chats by type...")
    metadata, sorted_chats = sort_chats_by_type(
        data=data,
        include_left_chats=not args.exclude_left,
        min_messages=args.min_messages,
        clean_text=True,
        remove_empty_service=not args.keep_empty_service,
    )

    print_summary_table(metadata)

    export_sorted_chats(
        metadata=metadata,
        sorted_chats=sorted_chats,
        output_path=args.output,
        summary_path=args.summary_output,
    )

    print("Processing complete!")


if __name__ == "__main__":
    main()