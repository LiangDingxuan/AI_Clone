"""
Approach 2 CLI: NLP Embeddings Semantic Sessionizer for Supergroups.
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
        description="Approach 2: NLP Embeddings Semantic Sessionizer for supergroup chats."
    )
    parser.add_argument("--input", "-i", default="sorted_chats_by_type.json",
                        help="Path to sorted_chats_by_type.json (default: sorted_chats_by_type.json)")
    parser.add_argument("--output", "-o", default="approach_2_nlp_embeddings/output/sessions_supergroup.json",
                        help="Output path (default: approach_2_nlp_embeddings/output/sessions_supergroup.json)")
    parser.add_argument("--idle-gap", type=float, default=4.0,
                        help="Idle gap in hours for coarse temporal split (default: 4.0)")
    parser.add_argument("--burst-window", type=int, default=120,
                        help="Burst merge window in seconds (default: 120)")
    parser.add_argument("--similarity-threshold", type=float, default=0.3,
                        help="Cosine similarity threshold for topic shift (default: 0.3)")
    parser.add_argument("--min-turns", type=int, default=2,
                        help="Minimum turns per conversation (default: 2)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)")

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    input_path = root_dir / args.input
    output_path = root_dir / args.output

    data = load_sorted_data(str(input_path))
    chats = get_chats_by_types(data, ["private_supergroup"])

    if not chats:
        print("No private_supergroup chats found. Nothing to process.")
        return

    # Lazy import to give a clear error if deps missing
    from sessionizer import sessionize_with_embeddings

    idle_gap_seconds = int(args.idle_gap * 3600)

    print(f"\nSessionizing {len(chats)} supergroup chats...")
    print(f"  Idle gap: {args.idle_gap}h ({idle_gap_seconds}s)")
    print(f"  Burst window: {args.burst_window}s")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print(f"  Model: {args.model}")
    print()

    conversations = sessionize_with_embeddings(
        chats=chats,
        idle_gap_seconds=idle_gap_seconds,
        burst_window_seconds=args.burst_window,
        similarity_threshold=args.similarity_threshold,
        min_turns=args.min_turns,
        require_target_user=True,
        model_name=args.model,
    )

    export_conversations(conversations, str(output_path), approach_name="nlp_embeddings")

    print(f"\nApproach 2 complete! {len(conversations)} conversations extracted.")


if __name__ == "__main__":
    main()
