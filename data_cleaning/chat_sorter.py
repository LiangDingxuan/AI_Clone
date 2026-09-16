"""
Chat Sorter & Cleaner for Telegram Export Data
Specifically designed for training an AI clone of Liang Dingxuan (@LiangDingxuan).
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def normalize_text(text_val: Any) -> str:
    """
    Normalizes Telegram message text.
    In Telegram exports, text can be:
      - A plain string: "Hello"
      - A list of mixed strings and formatting dicts:
        ["Hello ", {"type": "bold", "text": "world"}]
    This function converts all representations into a single clean string.
    """
    if isinstance(text_val, str):
        return text_val
    if isinstance(text_val, list):
        parts = []
        for item in text_val:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    return ""


def load_telegram_data(file_path: str | Path) -> Dict[str, Any]:
    """
    Loads Telegram export JSON data safely.
    Includes auto-repair logic if the JSON contains syntax anomalies
    (such as orphaned braces or trailing commas from manual edits).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_content = f.read()

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as err:
        print(f"[Warning] Direct JSON load failed ({err}). Attempting auto-repair...")
        # Common repair: orphaned braces or missing commas around stickers / drafts / chats
        repaired = re.sub(r'\]\s*\}\s*,\s*"chats"\s*:', r'],\n "chats":', raw_content)
        repaired = re.sub(r'\}\s*\}\s*,\s*"chats"\s*:', r'},\n "chats":', repaired)
        repaired = re.sub(r'\]\s*\}\s*,\s*"left_chats"\s*:', r'],\n "left_chats":', repaired)
        try:
            data = json.loads(repaired)
            print("[Info] Successfully loaded JSON after auto-repair!")
            return data
        except json.JSONDecodeError as second_err:
            raise ValueError(
                f"Failed to parse JSON file {file_path} even after repair: {second_err}"
            ) from second_err


def identify_target_user(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identifies the primary user profile from personal_information or default fallback.
    """
    personal_info = data.get("personal_information", {})
    user_id = personal_info.get("user_id", 5711494385)
    first_name = personal_info.get("first_name", "Dingxuan")
    last_name = personal_info.get("last_name", "Liang")
    username = personal_info.get("username", "@LiangDingxuan")

    full_name = f"{first_name} {last_name}".strip()
    return {
        "user_id": user_id,
        "from_id_str": f"user{user_id}",
        "full_name": full_name,
        "username": username,
    }


def is_target_user_message(msg: Dict[str, Any], target_user: Dict[str, Any]) -> bool:
    """
    Checks if a message was sent by Liang Dingxuan.
    Checks both from_id ('user5711494385') and author name.
    """
    from_id = str(msg.get("from_id", ""))
    target_id = target_user["from_id_str"]
    if from_id == target_id:
        return True

    from_name = str(msg.get("from", "")).strip().lower()
    full_name = target_user["full_name"].lower()
    if from_name and (from_name == full_name or ("dingxuan" in from_name and "liang" in from_name)):
        return True

    return False


def clean_message(
    msg: Dict[str, Any],
    target_user: Dict[str, Any],
    clean_text: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Cleans a single message, keeping essential fields for AI training.
    """
    msg_type = msg.get("type", "message")
    raw_text = msg.get("text", "")
    text = normalize_text(raw_text) if clean_text else raw_text

    is_sender = is_target_user_message(msg, target_user)

    cleaned = {
        "id": msg.get("id"),
        "type": msg_type,
        "date": msg.get("date"),
        "date_unixtime": msg.get("date_unixtime"),
        "from": msg.get("from"),
        "from_id": msg.get("from_id"),
        "is_target_user": is_sender,
        "text": text,
    }

    # Retain reply context if present
    if "reply_to_message_id" in msg:
        cleaned["reply_to_message_id"] = msg["reply_to_message_id"]

    # Retain forwarded context if present
    if "forwarded_from" in msg:
        cleaned["forwarded_from"] = msg["forwarded_from"]

    # Retain media information if present
    if "media_type" in msg:
        cleaned["media_type"] = msg["media_type"]
    elif "photo" in msg:
        cleaned["media_type"] = "photo"
    elif "file" in msg:
        cleaned["media_type"] = msg.get("mime_type", "file")

    return cleaned


def process_chat(
    chat: Dict[str, Any],
    target_user: Dict[str, Any],
    is_left: bool = False,
    clean_text: bool = True,
    remove_empty_service: bool = True,
) -> Dict[str, Any]:
    """
    Processes and cleans a single chat dictionary.
    Computes statistics and cleans message records.
    """
    chat_type = chat.get("type", "unknown")
    chat_name = chat.get("name")
    if not chat_name:
        chat_name = "Saved Messages" if chat_type == "saved_messages" else f"Chat {chat.get('id', 'Unknown')}"

    raw_messages = chat.get("messages", [])
    cleaned_messages = []
    target_user_msg_count = 0
    service_msg_count = 0

    for m in raw_messages:
        m_type = m.get("type", "message")
        if m_type == "service":
            service_msg_count += 1
            if remove_empty_service and not m.get("text"):
                continue  # Skip empty service noise

        cleaned_m = clean_message(m, target_user, clean_text=clean_text)
        if cleaned_m["is_target_user"]:
            target_user_msg_count += 1
        cleaned_messages.append(cleaned_m)

    first_date = cleaned_messages[0]["date"] if cleaned_messages else None
    last_date = cleaned_messages[-1]["date"] if cleaned_messages else None

    return {
        "id": chat.get("id"),
        "name": chat_name,
        "type": chat_type,
        "is_left": is_left,
        "total_messages": len(cleaned_messages),
        "raw_message_count": len(raw_messages),
        "target_user_messages": target_user_msg_count,
        "other_messages": len(cleaned_messages) - target_user_msg_count,
        "first_message_date": first_date,
        "last_message_date": last_date,
        "messages": cleaned_messages,
    }


def sort_chats_by_type(
    data: Dict[str, Any],
    include_left_chats: bool = True,
    min_messages: int = 1,
    clean_text: bool = True,
    remove_empty_service: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    Main sorting engine.
    1. Collects all chats from 'chats.list' and optionally 'left_chats.list'.
    2. Groups chats by 'type', excluding any chats where total_messages <= 0.
    3. Sorts chats within each type (by target user messages desc, then total messages desc).
    4. Orders the types themselves logically for persona training.
    """
    target_user = identify_target_user(data)
    active_chats = data.get("chats", {}).get("list", [])
    left_chats = data.get("left_chats", {}).get("list", []) if include_left_chats else []

    # Desired ordering of chat types (most conversational 1-on-1 first)
    type_priority = [
        "personal_chat",
        "private_group",
        "private_supergroup",
        "public_supergroup",
        "saved_messages",
        "private_channel",
        "public_channel",
    ]

    all_chats_processed: List[Dict[str, Any]] = []

    for c in active_chats:
        p = process_chat(
            c,
            target_user,
            is_left=False,
            clean_text=clean_text,
            remove_empty_service=remove_empty_service,
        )
        if p["total_messages"] > 0 and p["total_messages"] >= min_messages:
            all_chats_processed.append(p)

    for c in left_chats:
        p = process_chat(
            c,
            target_user,
            is_left=True,
            clean_text=clean_text,
            remove_empty_service=remove_empty_service,
        )
        if p["total_messages"] > 0 and p["total_messages"] >= min_messages:
            all_chats_processed.append(p)

    # Group by type
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_chats_processed:
        t = c["type"]
        grouped.setdefault(t, []).append(c)

    # Sort within each type: chats with most target user messages first, then total messages
    for t in grouped:
        grouped[t].sort(
            key=lambda item: (item["target_user_messages"], item["total_messages"]),
            reverse=True,
        )

    # Order keys according to type_priority, followed by any remaining types (skip empty categories)
    sorted_grouped: Dict[str, List[Dict[str, Any]]] = {}
    for t in type_priority:
        if t in grouped and len(grouped[t]) > 0:
            sorted_grouped[t] = grouped[t]

    for t in sorted(grouped.keys()):
        if t not in sorted_grouped and len(grouped[t]) > 0:
            sorted_grouped[t] = grouped[t]

    # Generate summary stats
    summary_by_type = {}
    total_chats = 0
    total_messages = 0
    total_target_msgs = 0

    for t, chat_list in sorted_grouped.items():
        type_total_msgs = sum(c["total_messages"] for c in chat_list)
        type_target_msgs = sum(c["target_user_messages"] for c in chat_list)
        total_chats += len(chat_list)
        total_messages += type_total_msgs
        total_target_msgs += type_target_msgs

        summary_by_type[t] = {
            "chat_count": len(chat_list),
            "total_messages": type_total_msgs,
            "target_user_messages": type_target_msgs,
            "other_messages": type_total_msgs - type_target_msgs,
        }

    metadata = {
        "target_user": target_user["full_name"],
        "user_id": target_user["user_id"],
        "username": target_user["username"],
        "total_chats": total_chats,
        "total_messages": total_messages,
        "total_target_user_messages": total_target_msgs,
        "chats_by_type_summary": summary_by_type,
    }

    return metadata, sorted_grouped


def export_sorted_chats(
    metadata: Dict[str, Any],
    sorted_chats: Dict[str, List[Dict[str, Any]]],
    output_path: str | Path,
    summary_path: Optional[str | Path] = None,
) -> None:
    """
    Saves sorted chats to output JSON file.
    Optionally saves a separate lightweight summary file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    output_payload = {
        "export_info": metadata,
        "chats_by_type": sorted_chats,
    }

    print(f"Writing sorted chats to {out_file.resolve()}...")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(f"[Done] Wrote {out_file.name} ({out_file.stat().st_size / (1024 * 1024):.2f} MB)")

    if summary_path:
        sum_file = Path(summary_path)
        sum_file.parent.mkdir(parents=True, exist_ok=True)
        # Create a detailed inspection list without heavy messages array
        summary_payload = {
            "export_info": metadata,
            "chats_by_type_index": {
                t: [
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "is_left": c["is_left"],
                        "total_messages": c["total_messages"],
                        "target_user_messages": c["target_user_messages"],
                        "first_date": c["first_message_date"],
                        "last_date": c["last_message_date"],
                    }
                    for c in chat_list
                ]
                for t, chat_list in sorted_chats.items()
            },
        }
        with open(sum_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)
        print(f"[Done] Wrote summary to {sum_file.name} ({sum_file.stat().st_size / 1024:.2f} KB)")


def print_summary_table(metadata: Dict[str, Any]) -> None:
    """
    Prints a formatted summary table to stdout.
    """
    print("\n" + "=" * 80)
    print(" TELEGRAM CHAT HISTORY - SORTED BY TYPE FOR AI CLONE TRAINING")
    print(f" Target User: {metadata['target_user']} ({metadata['username']} / ID: {metadata['user_id']})")
    print("=" * 80)
    print(f"{'Chat Type':<25} | {'Chats':<8} | {'Total Msgs':<12} | {'Liang Dingxuan':<16} | {'Others':<10}")
    print("-" * 80)

    summary = metadata["chats_by_type_summary"]
    for chat_type, stats in summary.items():
        print(
            f"{chat_type:<25} | "
            f"{stats['chat_count']:<8} | "
            f"{stats['total_messages']:<12} | "
            f"{stats['target_user_messages']:<16} | "
            f"{stats['other_messages']:<10}"
        )

    print("-" * 80)
    print(
        f"{'TOTAL':<25} | "
        f"{metadata['total_chats']:<8} | "
        f"{metadata['total_messages']:<12} | "
        f"{metadata['total_target_user_messages']:<16} | "
        f"{metadata['total_messages'] - metadata['total_target_user_messages']:<10}"
    )
    print("=" * 80 + "\n")