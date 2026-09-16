"""
Main entry point for AI Clone dataset preprocessing.
Sorts and cleans telegramChatHistory.json by chat type.
"""

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from data_cleaning.chat_sorter import (
        export_sorted_chats,
        load_telegram_data,
        print_summary_table,
        sort_chats_by_type,
    )
except ImportError:
    from chat_sorter import (
        export_sorted_chats,
        load_telegram_data,
        print_summary_table,
        sort_chats_by_type,
    )


def resolve_input_path(raw_path: str) -> str:
    """Finds raw export in current dir, project root, or as given."""
    p = Path(raw_path)
    if p.is_file():
        return str(p)
    if (PROJECT_ROOT / raw_path).is_file():
        return str(PROJECT_ROOT / raw_path)
    if (CURRENT_DIR / raw_path).is_file():
        return str(CURRENT_DIR / raw_path)
    return raw_path


def main():
    # Ensure UTF-8 output encoding in Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    default_input = "telegramChatHistory.json"
    if not Path(default_input).is_file() and (PROJECT_ROOT / default_input).is_file():
        default_input = str(PROJECT_ROOT / default_input)

    default_output = str(CURRENT_DIR / "sorted_chats_by_type.json")
    default_summary = str(CURRENT_DIR / "chats_summary.json")

    parser = argparse.ArgumentParser(
        description="Sort and clean Telegram chat history by chat type for AI personality cloning."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=default_input,
        help="Path to source Telegram JSON export (default: telegramChatHistory.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=default_output,
        help=f"Path to output sorted JSON file (default: {default_output})",
    )
    parser.add_argument(
        "--summary-output",
        "-s",
        default=default_summary,
        help=f"Path to lightweight summary JSON file (default: {default_summary})",
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

    input_file = resolve_input_path(args.input)
    print(f"Loading Telegram data from: {input_file}")
    data = load_telegram_data(input_file)

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