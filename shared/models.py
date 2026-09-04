"""
Shared data models for conversation sessionization.
Provides consistent output format across all three approaches.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    """
    A single conversational turn — one or more burst-merged messages
    from the same sender in rapid succession.
    """
    sender_name: str
    sender_id: str
    is_target_user: bool
    text: str
    start_time: str          # ISO 8601 datetime string
    end_time: str             # ISO 8601 datetime string
    start_unixtime: int
    end_unixtime: int
    message_ids: List[int] = field(default_factory=list)
    message_count: int = 1
    has_media: bool = False
    has_reply: bool = False
    reply_to_message_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Turn":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Conversation:
    """
    A complete conversation session — a coherent sequence of turns
    that belong to the same topical exchange.
    """
    conversation_id: str
    chat_id: int
    chat_name: str
    chat_type: str
    approach: str             # "rule_based", "nlp_embeddings", or "ai_llm"
    turns: List[Turn] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    start_unixtime: int = 0
    end_unixtime: int = 0
    turn_count: int = 0
    target_user_turn_count: int = 0
    has_target_user_participation: bool = False

    def finalize(self):
        """Compute derived fields from turns."""
        if self.turns:
            self.start_time = self.turns[0].start_time
            self.end_time = self.turns[-1].end_time
            self.start_unixtime = self.turns[0].start_unixtime
            self.end_unixtime = self.turns[-1].end_unixtime
            self.participants = sorted(set(
                t.sender_name for t in self.turns if t.sender_name
            ))
            self.turn_count = len(self.turns)
            self.target_user_turn_count = sum(
                1 for t in self.turns if t.is_target_user
            )
            self.has_target_user_participation = self.target_user_turn_count > 0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "conversation_id": self.conversation_id,
            "chat_id": self.chat_id,
            "chat_name": self.chat_name,
            "chat_type": self.chat_type,
            "approach": self.approach,
            "participants": self.participants,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_unixtime": self.start_unixtime,
            "end_unixtime": self.end_unixtime,
            "turn_count": self.turn_count,
            "target_user_turn_count": self.target_user_turn_count,
            "has_target_user_participation": self.has_target_user_participation,
            "turns": [t.to_dict() for t in self.turns],
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Conversation":
        turns = [Turn.from_dict(t) for t in d.pop("turns", [])]
        conv = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        conv.turns = turns
        return conv


def merge_into_turns(
    messages: List[Dict[str, Any]],
    burst_window_seconds: int = 90,
) -> List[Turn]:
    """
    Burst-merges consecutive messages from the same sender within
    burst_window_seconds into single Turn objects.

    This is shared logic used by all three approaches.
    """
    if not messages:
        return []

    turns: List[Turn] = []
    current_burst: List[Dict[str, Any]] = [messages[0]]

    for msg in messages[1:]:
        prev = current_burst[-1]
        time_delta = msg["date_unixtime"] - prev["date_unixtime"]
        same_sender = msg.get("from_id") == prev.get("from_id")

        if same_sender and time_delta <= burst_window_seconds:
            current_burst.append(msg)
        else:
            turns.append(_flush_burst(current_burst))
            current_burst = [msg]

    if current_burst:
        turns.append(_flush_burst(current_burst))

    return turns


def _flush_burst(burst: List[Dict[str, Any]]) -> Turn:
    """Convert a burst of messages into a single Turn."""
    first = burst[0]
    last = burst[-1]

    combined_text = "\n".join(
        m.get("text", "") for m in burst if m.get("text", "").strip()
    )

    has_media = any("media_type" in m for m in burst)
    has_reply = any("reply_to_message_id" in m for m in burst)
    reply_id = None
    for m in burst:
        if "reply_to_message_id" in m:
            reply_id = m["reply_to_message_id"]
            break

    return Turn(
        sender_name=first.get("from", "Unknown"),
        sender_id=first.get("from_id", ""),
        is_target_user=first.get("is_target_user", False),
        text=combined_text,
        start_time=first.get("date", ""),
        end_time=last.get("date", ""),
        start_unixtime=first["date_unixtime"],
        end_unixtime=last["date_unixtime"],
        message_ids=[m.get("id", 0) for m in burst],
        message_count=len(burst),
        has_media=has_media,
        has_reply=has_reply,
        reply_to_message_id=reply_id,
    )


def export_conversations(
    conversations: List[Conversation],
    output_path: str | Path,
    approach_name: str = "",
) -> None:
    """
    Exports conversations to a JSON file with summary metadata.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Compute summary stats
    total_turns = sum(c.turn_count for c in conversations)
    total_target_turns = sum(c.target_user_turn_count for c in conversations)
    with_target = sum(1 for c in conversations if c.has_target_user_participation)
    unique_chats = len(set(c.chat_id for c in conversations))

    payload = {
        "summary": {
            "approach": approach_name,
            "total_conversations": len(conversations),
            "conversations_with_target_user": with_target,
            "total_turns": total_turns,
            "target_user_turns": total_target_turns,
            "unique_chats": unique_chats,
        },
        "conversations": [c.to_dict() for c in conversations],
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"[Done] Exported {len(conversations)} conversations to {out.name} ({size_mb:.2f} MB)")
    print(f"       {with_target} conversations have Liang Dingxuan participation")
    print(f"       {total_turns} total turns, {total_target_turns} from target user")

