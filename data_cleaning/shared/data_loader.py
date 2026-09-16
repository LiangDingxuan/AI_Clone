"""
Shared data loader for all three sessionization approaches.
Loads sorted_chats_by_type.json and provides filtered access by chat type.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_sorted_data(path: str | Path) -> Dict[str, Any]:
    """
    Loads the sorted_chats_by_type.json file.
    Returns the full parsed JSON dict with keys: export_info, chats_by_type.
    """
    p = Path(path)
    if not p.exists():
        shared_dir = Path(__file__).resolve().parent
        data_cleaning_dir = shared_dir.parent
        project_root = data_cleaning_dir.parent
        if (data_cleaning_dir / path).exists():
            p = data_cleaning_dir / path
        elif (project_root / path).exists():
            p = project_root / path
        elif (data_cleaning_dir / p.name).exists():
            p = data_cleaning_dir / p.name
        elif (project_root / p.name).exists():
            p = project_root / p.name
        else:
            raise FileNotFoundError(f"Sorted data file not found: {Path(path).resolve()}")

    print(f"Loading sorted data from {p.name}...")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_chats = data.get("export_info", {}).get("total_chats", 0)
    total_msgs = data.get("export_info", {}).get("total_messages", 0)
    print(f"Loaded {total_chats} chats with {total_msgs} total messages.")
    return data


def get_chats_by_types(
    data: Dict[str, Any],
    chat_types: List[str],
    min_target_user_messages: int = 0,
) -> List[Dict[str, Any]]:
    """
    Filters and returns chats matching the specified type(s).

    Args:
        data: Full sorted_chats_by_type.json parsed dict.
        chat_types: List of types to include (e.g. ["personal_chat"]).
        min_target_user_messages: Only include chats where the target user
            has at least this many messages (default 0 = include all).

    Returns:
        List of chat dicts, each with keys: id, name, type, messages, etc.
    """
    chats_by_type = data.get("chats_by_type", {})
    result = []

    for ct in chat_types:
        chat_list = chats_by_type.get(ct, [])
        for chat in chat_list:
            if chat.get("target_user_messages", 0) >= min_target_user_messages:
                result.append(chat)

    print(f"Selected {len(result)} chats of type(s) {chat_types}")
    return result


def prepare_messages(chat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns the messages list from a chat dict with date_unixtime cast to int.
    Filters out service messages and messages with empty text (unless they have media).
    Messages are sorted by timestamp.
    """
    messages = chat.get("messages", [])
    prepared = []

    for msg in messages:
        # Cast date_unixtime from string to int
        try:
            msg["date_unixtime"] = int(msg["date_unixtime"])
        except (ValueError, TypeError, KeyError):
            continue  # Skip messages without valid timestamps

        # Skip service messages (join/leave/pin events)
        if msg.get("type") == "service":
            continue

        prepared.append(msg)

    # Sort by timestamp (handles both positive and negative message IDs)
    prepared.sort(key=lambda m: m["date_unixtime"])
    return prepared


def get_target_user_id(data: Dict[str, Any]) -> str:
    """
    Returns the target user's from_id string (e.g. 'user5711494385').
    """
    export_info = data.get("export_info", {})
    user_id = export_info.get("user_id", 5711494385)
    return f"user{user_id}"


def get_target_user_name(data: Dict[str, Any]) -> str:
    """
    Returns the target user's display name (e.g. 'Dingxuan Liang').
    """
    return data.get("export_info", {}).get("target_user", "Dingxuan Liang")

