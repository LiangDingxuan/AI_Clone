"""
Stage 1: Chat History Preprocessing Engine.

Responsibilities:
1. Multi-format Ingestion: Telegram JSON export, Generic JSON, CSV, and plain text chat logs.
2. Anonymization & Noise Cleaning: Strips PII (phone numbers, emails, IP addresses),
   system logs, embedded timestamps, and filters out noise (service events, empty media).
3. Consecutive Message Concatenation: Groups rapid bursts from the same sender within
   a configured time window (e.g. 180s) to reflect human pacing.
4. Session Segmentation: Partitions ongoing conversations into discrete sessions
   separated by temporal idle gaps (e.g. >= 18 hours).
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Generator, List, Optional, Union

import pandas as pd

from digital_twin.config import PreprocessorConfig

logger = logging.getLogger(__name__)


@dataclass
class RawMessage:
    """Internal representation of a single extracted message prior to grouping."""
    sender_name: str
    sender_id: str
    timestamp: datetime
    unixtime: float
    text: str
    chat_id: str = "default"
    chat_name: str = "General"
    is_service: bool = False
    reply_to_id: Optional[int] = None


@dataclass
class ChatTurn:
    """A single conversational turn, potentially combining consecutive bursts from one sender."""
    sender_name: str
    sender_id: str
    start_time: str
    end_time: str
    timestamp_unixtime: float
    text: str
    message_count: int
    chat_id: str
    chat_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatSession:
    """A segmented conversational session separated by temporal idle boundaries."""
    session_id: str
    chat_id: str
    chat_name: str
    start_time: str
    end_time: str
    participants: List[str]
    turns: List[ChatTurn] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "chat_id": self.chat_id,
            "chat_name": self.chat_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "participants": self.participants,
            "turn_count": len(self.turns),
            "turns": [t.to_dict() for t in self.turns],
        }


class ChatPreprocessor:
    """
    Robust preprocessing pipeline implementing:
    - PII scrubbing & noise suppression
    - Burst concatenation
    - Temporal session segmentation
    """

    # Comprehensive regular expressions for PII and noise
    EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
    PHONE_REGEX = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b')
    IP_REGEX = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
    EMBEDDED_DATETIME_REGEX = re.compile(
        r'\b\d{4}[-/]\d{2}[-/]\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\b'
    )

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()

    def clean_and_anonymize(self, text: str) -> str:
        """
        Cleans text body by stripping PII and noise according to configuration.
        Preserves linguistic markers (slang, casing, contractions, emojis).
        """
        if not text:
            return ""

        cleaned = text

        if self.config.anonymize_pii:
            if self.config.mask_emails:
                cleaned = self.EMAIL_REGEX.sub("[EMAIL]", cleaned)
            if self.config.mask_phone_numbers:
                cleaned = self.PHONE_REGEX.sub("[PHONE]", cleaned)
            if self.config.mask_ip_addresses:
                cleaned = self.IP_REGEX.sub("[IP_ADDRESS]", cleaned)

        # Remove explicit inline timestamps accidentally embedded into copy-pasted text
        cleaned = self.EMBEDDED_DATETIME_REGEX.sub("", cleaned)

        # Normalize noisy extra spaces while preserving intentional newlines
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in cleaned.splitlines()]
        cleaned = "\n".join([line for line in lines if line])

        return cleaned.strip()

    def extract_telegram_text(self, text_field: Union[str, List[Any]]) -> str:
        """
        Handles Telegram export peculiarities where text can be:
        1. Plain string
        2. List of strings and text entity dicts (e.g. {'type': 'link', 'text': 'http...'})
        """
        if isinstance(text_field, str):
            return text_field
        elif isinstance(text_field, list):
            chunks = []
            for item in text_field:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict) and "text" in item:
                    # Ignore pure bot command strings if desired, else retain text
                    chunks.append(str(item["text"]))
            return "".join(chunks)
        return ""

    def parse_telegram_json(
        self, file_path: Union[str, Path], max_messages: Optional[int] = None
    ) -> List[RawMessage]:
        """
        Parses a Telegram Desktop JSON export (`result.json` or `telegramChatHistory.json`).
        Extracts both 1-on-1 personal chats and group chats while discarding non-text noise.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Telegram export file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_messages: List[RawMessage] = []
        chats = data.get("chats", {}).get("list", [])
        if not chats and "messages" in data:
            # Single chat export format
            chats = [data]

        for chat in chats:
            chat_id = str(chat.get("id", "chat_unknown"))
            chat_name = chat.get("name") or chat.get("type", "Unnamed_Chat")
            messages = chat.get("messages", [])

            for msg in messages:
                # Discard service notifications unless explicitly allowed
                msg_type = msg.get("type", "")
                if msg_type != "message":
                    if self.config.filter_service_messages:
                        continue

                sender_name = msg.get("from") or "Unknown"
                sender_id = str(msg.get("from_id", sender_name))
                
                # Extract and clean text
                raw_text = self.extract_telegram_text(msg.get("text", ""))
                cleaned_text = self.clean_and_anonymize(raw_text)

                # Skip completely blank messages (e.g. stickers or photos without captions)
                if not cleaned_text:
                    continue

                # Parse timestamp
                unixtime = float(msg.get("date_unixtime", 0))
                if unixtime > 0:
                    dt = datetime.fromtimestamp(unixtime, tz=timezone.utc)
                else:
                    date_str = msg.get("date", "")
                    try:
                        dt = datetime.fromisoformat(date_str)
                        unixtime = dt.timestamp()
                    except Exception:
                        dt = datetime.now(timezone.utc)
                        unixtime = dt.timestamp()

                raw_messages.append(
                    RawMessage(
                        sender_name=sender_name,
                        sender_id=sender_id,
                        timestamp=dt,
                        unixtime=unixtime,
                        text=cleaned_text,
                        chat_id=chat_id,
                        chat_name=chat_name,
                        is_service=msg_type != "message",
                        reply_to_id=msg.get("reply_to_message_id"),
                    )
                )

                if max_messages and len(raw_messages) >= max_messages:
                    break

            if max_messages and len(raw_messages) >= max_messages:
                break

        # Chronological sort across messages
        raw_messages.sort(key=lambda m: m.unixtime)
        return raw_messages

    def parse_csv(self, file_path: Union[str, Path]) -> List[RawMessage]:
        """
        Parses a generic CSV chat export.
        Attempts to detect columns: [sender, timestamp/date, text/message, chat_id].
        """
        df = pd.read_csv(file_path)
        col_map = {c.lower(): c for c in df.columns}

        sender_col = col_map.get("sender") or col_map.get("from") or col_map.get("author") or "sender"
        text_col = col_map.get("text") or col_map.get("message") or col_map.get("content") or "text"
        time_col = col_map.get("timestamp") or col_map.get("date") or col_map.get("time") or "timestamp"
        chat_col = col_map.get("chat_id") or col_map.get("chat") or None

        raw_messages: List[RawMessage] = []
        for _, row in df.iterrows():
            text_val = str(row.get(text_col, ""))
            cleaned = self.clean_and_anonymize(text_val)
            if not cleaned:
                continue

            sender_name = str(row.get(sender_col, "Unknown"))
            t_val = row.get(time_col)
            try:
                dt = pd.to_datetime(t_val).to_pydatetime()
                unixtime = dt.timestamp()
            except Exception:
                dt = datetime.now(timezone.utc)
                unixtime = dt.timestamp()

            chat_id = str(row.get(chat_col, "default_chat")) if chat_col else "default_chat"

            raw_messages.append(
                RawMessage(
                    sender_name=sender_name,
                    sender_id=sender_name,
                    timestamp=dt,
                    unixtime=unixtime,
                    text=cleaned,
                    chat_id=chat_id,
                )
            )

        raw_messages.sort(key=lambda m: m.unixtime)
        return raw_messages

    def concatenate_bursts(self, messages: List[RawMessage]) -> List[ChatTurn]:
        """
        Requirement 1: Consecutive Message Concatenation.
        Combines multiple short, consecutive messages from the same sender
        occurring within `burst_window_seconds` into a single coherent conversational block.
        """
        if not messages:
            return []

        turns: List[ChatTurn] = []
        current_burst: List[RawMessage] = [messages[0]]

        for msg in messages[1:]:
            prev = current_burst[-1]
            time_delta = msg.unixtime - prev.unixtime
            same_sender = (msg.sender_id == prev.sender_id)
            same_chat = (msg.chat_id == prev.chat_id)

            if same_sender and same_chat and (time_delta <= self.config.burst_window_seconds):
                current_burst.append(msg)
            else:
                # Flush the current burst into a single ChatTurn
                combined_text = "\n".join(m.text for m in current_burst)
                first_msg = current_burst[0]
                last_msg = current_burst[-1]

                turns.append(
                    ChatTurn(
                        sender_name=first_msg.sender_name,
                        sender_id=first_msg.sender_id,
                        start_time=first_msg.timestamp.isoformat(),
                        end_time=last_msg.timestamp.isoformat(),
                        timestamp_unixtime=first_msg.unixtime,
                        text=combined_text,
                        message_count=len(current_burst),
                        chat_id=first_msg.chat_id,
                        chat_name=first_msg.chat_name,
                    )
                )
                current_burst = [msg]

        # Flush final burst
        if current_burst:
            combined_text = "\n".join(m.text for m in current_burst)
            first_msg = current_burst[0]
            last_msg = current_burst[-1]
            turns.append(
                ChatTurn(
                    sender_name=first_msg.sender_name,
                    sender_id=first_msg.sender_id,
                    start_time=first_msg.timestamp.isoformat(),
                    end_time=last_msg.timestamp.isoformat(),
                    timestamp_unixtime=first_msg.unixtime,
                    text=combined_text,
                    message_count=len(current_burst),
                    chat_id=first_msg.chat_id,
                    chat_name=first_msg.chat_name,
                )
            )

        return turns

    def segment_sessions(self, turns: List[ChatTurn]) -> List[ChatSession]:
        """
        Requirement 2: Session Segmentation.
        Splits conversation streams into distinct logical sessions when idle gaps
        exceed `session_idle_gap_hours` (e.g. 18 hours), preventing topic conflation.
        """
        if not turns:
            return []

        # Group turns by chat_id first so sessions belong to individual threads
        chats_map: Dict[str, List[ChatTurn]] = {}
        for turn in turns:
            chats_map.setdefault(turn.chat_id, []).append(turn)

        sessions: List[ChatSession] = []
        gap_threshold_seconds = self.config.session_idle_gap_hours * 3600.0

        for chat_id, chat_turns in chats_map.items():
            chat_turns.sort(key=lambda t: t.timestamp_unixtime)
            current_session_turns: List[ChatTurn] = [chat_turns[0]]
            session_counter = 1

            for turn in chat_turns[1:]:
                prev_turn = current_session_turns[-1]
                gap = turn.timestamp_unixtime - prev_turn.timestamp_unixtime

                if gap > gap_threshold_seconds:
                    # Finalize current session
                    participants = sorted(list({t.sender_name for t in current_session_turns}))
                    sessions.append(
                        ChatSession(
                            session_id=f"{chat_id}_sess_{session_counter:04d}",
                            chat_id=chat_id,
                            chat_name=current_session_turns[0].chat_name,
                            start_time=current_session_turns[0].start_time,
                            end_time=current_session_turns[-1].end_time,
                            participants=participants,
                            turns=current_session_turns,
                        )
                    )
                    session_counter += 1
                    current_session_turns = [turn]
                else:
                    current_session_turns.append(turn)

            if current_session_turns:
                participants = sorted(list({t.sender_name for t in current_session_turns}))
                sessions.append(
                    ChatSession(
                        session_id=f"{chat_id}_sess_{session_counter:04d}",
                        chat_id=chat_id,
                        chat_name=current_session_turns[0].chat_name,
                        start_time=current_session_turns[0].start_time,
                        end_time=current_session_turns[-1].end_time,
                        participants=participants,
                        turns=current_session_turns,
                    )
                )

        # Sort all sessions chronologically by start_time
        sessions.sort(key=lambda s: s.turns[0].timestamp_unixtime if s.turns else 0)
        return sessions

    def process(
        self, file_path: Union[str, Path], max_messages: Optional[int] = None
    ) -> List[ChatSession]:
        """
        Executes the complete Stage 1 preprocessing pipeline on an input file.
        Detects file extension and routes appropriately.
        """
        p = Path(file_path)
        if p.suffix.lower() == ".json":
            raw = self.parse_telegram_json(p, max_messages=max_messages)
        elif p.suffix.lower() in [".csv", ".tsv"]:
            raw = self.parse_csv(p)
        else:
            raise ValueError(f"Unsupported file format: {p.suffix}. Expected .json or .csv")

        logger.info(f"Loaded {len(raw)} raw messages from {p.name}")
        turns = self.concatenate_bursts(raw)
        logger.info(f"Concatenated into {len(turns)} turns via {self.config.burst_window_seconds}s burst window")
        sessions = self.segment_sessions(turns)
        logger.info(f"Segmented into {len(sessions)} sessions (idle gap > {self.config.session_idle_gap_hours}h)")
        return sessions

    def to_dataframe(self, sessions: List[ChatSession]) -> pd.DataFrame:
        """Converts sessions into a flattened pandas DataFrame for analysis."""
        rows = []
        for sess in sessions:
            for i, turn in enumerate(sess.turns):
                rows.append({
                    "session_id": sess.session_id,
                    "chat_id": sess.chat_id,
                    "chat_name": sess.chat_name,
                    "turn_index": i,
                    "sender_name": turn.sender_name,
                    "sender_id": turn.sender_id,
                    "start_time": turn.start_time,
                    "end_time": turn.end_time,
                    "timestamp_unixtime": turn.timestamp_unixtime,
                    "text": turn.text,
                    "message_count": turn.message_count,
                })
        return pd.DataFrame(rows)

    def save_sessions_to_json(self, sessions: List[ChatSession], output_path: Union[str, Path]) -> None:
        """Serializes processed sessions to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in sessions], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    input_file = sys.argv[1] if len(sys.argv) > 1 else "telegramChatHistory.json"
    preprocessor = ChatPreprocessor()
    print(f"Running preprocessing on {input_file}...")
    sess = preprocessor.process(input_file, max_messages=5000)
    print(f"Extracted {len(sess)} sessions.")
    if sess:
        print("Sample session:", sess[0].session_id, "with", len(sess[0].turns), "turns.")
        print("First turn:", sess[0].turns[0])
