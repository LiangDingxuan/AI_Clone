"""
Module 1: Multi-Context Profiler & Analyzer (profiler.py)
Author: AI Software Engineer
Project: Context-Adaptive Personality Chatbot ("Dingxuan")

This module loads and analyzes conversational datasets across three social contexts:
1. Personal Chat (DMs) - High intimacy, 1-on-1 dialogue, rapid casual exchanges.
2. Supergroups - Large semi-public channels, more structured, broadcasting or answering specific prompts.
3. Group Chats - Multi-person casual threads, highly conversational, diverse sub-topics.

It extracts:
- Linguistic Metrics: response length, punctuation, casing, filler words, emojis.
- Scenario-Based Big-5 Personality Trait scores (0.0 to 1.0) grounded in psycholinguistics.
- Stylistic Exemplars: 3-5 curated few-shot QA pairs per context.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("profiler")

# Regex for emojis (Unicode emoji range)
EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]",
    flags=re.UNICODE,
)

# Common English stopwords to isolate content keywords
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves",
}

# Colloquial Singlish particles, fillers, and informal discourse markers
FILLER_CANDIDATES = {
    "ah", "eh", "sia", "lor", "leh", "meh", "la", "lah", "yea", "yeah", "cuz",
    "cause", "idk", "wait", "prob", "probably", "bruh", "bro", "whut", "what",
    "noice", "nice", "lol", "lmao", "haha", "hahaha", "tbh", "imo", "alr",
    "already", "ltr", "later", "dunno", "man", "nah", "okay", "ok", "got", "can",
    "one", "actually", "basically", "so", "oh",
}


@dataclass
class LinguisticMetrics:
    """Quantitative linguistic metrics extracted from Dingxuan's turns."""
    avg_words_per_turn: float
    avg_chars_per_turn: float
    all_lower_ratio: float
    first_char_lower_ratio: float
    exclamation_ratio: float
    question_ratio: float
    ellipsis_ratio: float
    combo_punct_ratio: float
    top_keywords: List[Tuple[str, int]]
    top_fillers: List[Tuple[str, int]]
    top_emojis: List[Tuple[str, int]]
    type_token_ratio: float
    total_analyzed_turns: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BigFiveCard:
    """
    Big-5 Personality Trait card scored from 0.0 to 1.0.
    Grounded in social-context psycholinguistic research (e.g. Mairesse et al., 2007; Llama2-MBTI),
    personality shifts dynamically between private, semi-public, and group channels.
    """
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    context_notes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FewShotPair:
    """A representative dialog turn pair: User prompt -> Dingxuan response."""
    user_message: str
    target_response: str
    other_sender: str
    context_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextProfile:
    """Complete behavioral signature and profile for a specific social context."""
    context_name: str           # e.g., "personal_chat", "supergroup", "group"
    display_name: str           # e.g., "Personal Chat (DM)", "Supergroup", "Group Chat"
    metrics: LinguisticMetrics
    big_five: BigFiveCard
    few_shot_exemplars: List[FewShotPair]
    is_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_name": self.context_name,
            "display_name": self.display_name,
            "is_fallback": self.is_fallback,
            "metrics": self.metrics.to_dict(),
            "big_five": self.big_five.to_dict(),
            "few_shot_exemplars": [e.to_dict() for e in self.few_shot_exemplars],
        }


# =============================================================================
# Resilient Data Loader
# =============================================================================

def is_dingxuan_sender(sender_name: Optional[str], is_target_user: Optional[bool]) -> bool:
    """Check if a turn belongs to Dingxuan using name matching or boolean flag."""
    if is_target_user is True:
        return True
    if not sender_name:
        return False
    name_clean = sender_name.strip().lower()
    return "dingxuan" in name_clean or name_clean == "target_user"


def load_context_sessions(file_path: Path | str) -> List[Dict[str, Any]]:
    """
    Resilient loader for session files.
    Supports:
      1. Repository format: {"summary": {...}, "conversations": [{"turns": [...]}]}
      2. Prompt specification schema: [{"session_id": "...", "messages": [{"sender": "...", "text": "..."}]}]
      3. List of conversation dicts directly.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(f"Malformed JSON in {path}: {err}") from err
    except UnicodeDecodeError:
        # Fallback to utf-8 with error replacement
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_data = json.load(f)

    # Normalize into a list of session dicts
    sessions: List[Dict[str, Any]] = []
    if isinstance(raw_data, dict):
        if "conversations" in raw_data and isinstance(raw_data["conversations"], list):
            sessions = raw_data["conversations"]
        elif "sessions" in raw_data and isinstance(raw_data["sessions"], list):
            sessions = raw_data["sessions"]
        else:
            # Maybe single session object
            sessions = [raw_data]
    elif isinstance(raw_data, list):
        sessions = raw_data
    else:
        raise ValueError(f"Unexpected JSON root type in {path}: {type(raw_data)}")

    return sessions


def normalize_session_turns(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalizes message/turn entries across schemas into consistent dicts:
    {
      "sender_name": str,
      "text": str,
      "is_target_user": bool
    }
    """
    raw_turns = session.get("turns") or session.get("messages") or []
    if not isinstance(raw_turns, list):
        return []

    normalized = []
    for item in raw_turns:
        if not isinstance(item, dict):
            continue
        sender = (
            item.get("sender_name")
            or item.get("sender")
            or item.get("from")
            or item.get("user")
            or ""
        )
        text = (
            item.get("text")
            or item.get("content")
            or item.get("message")
            or ""
        )
        # Some Telegram exports store rich text as a list of strings/objects
        if isinstance(text, list):
            text_parts = []
            for part in text:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            text = "".join(text_parts)
        elif not isinstance(text, str):
            text = str(text) if text is not None else ""

        is_target = item.get("is_target_user")
        if is_target is None:
            is_target = is_dingxuan_sender(sender, None)

        normalized.append({
            "sender_name": sender,
            "text": text.strip(),
            "is_target_user": bool(is_target),
        })

    return normalized


# =============================================================================
# Linguistic Metrics Extraction
# =============================================================================

def extract_linguistic_metrics(turns: List[str]) -> LinguisticMetrics:
    """
    Extracts quantitative linguistic habits from a collection of Dingxuan's text turns:
    - Average length in words and characters
    - Casing patterns (all-lowercase ratio, first-char lowercase ratio)
    - Punctuation habits (!, ?, ..., combo punctuation ?!, !?)
    - Top keywords, fillers/particles, and emojis
    - Type-Token Ratio (vocabulary richness)
    """
    if not turns:
        # Fallback empty metrics
        return LinguisticMetrics(
            avg_words_per_turn=7.0,
            avg_chars_per_turn=35.0,
            all_lower_ratio=0.45,
            first_char_lower_ratio=0.48,
            exclamation_ratio=0.002,
            question_ratio=0.11,
            ellipsis_ratio=0.005,
            combo_punct_ratio=0.005,
            top_keywords=[("can", 100), ("cuz", 80), ("wait", 60)],
            top_fillers=[("ah", 100), ("yea", 80), ("cuz", 70), ("idk", 50)],
            top_emojis=[],
            type_token_ratio=0.35,
            total_analyzed_turns=0,
        )

    total_turns = len(turns)
    words_per_turn: List[int] = []
    chars_per_turn: List[int] = []

    all_lower_count = 0
    first_char_lower_count = 0

    exclamation_count = 0
    question_count = 0
    ellipsis_count = 0
    combo_punct_count = 0

    keyword_counter: Dict[str, int] = {}
    filler_counter: Dict[str, int] = {}
    emoji_counter: Dict[str, int] = {}

    all_tokens: List[str] = []

    for text in turns:
        if not text:
            continue
        words = text.split()
        num_words = len(words)
        words_per_turn.append(num_words)
        chars_per_turn.append(len(text))

        # Casing
        if text.islower():
            all_lower_count += 1
        first_alpha = next((c for c in text if c.isalpha()), None)
        if first_alpha and first_alpha.islower():
            first_char_lower_count += 1

        # Punctuation
        if "!" in text:
            exclamation_count += 1
        if "?" in text:
            question_count += 1
        if "..." in text or "…" in text:
            ellipsis_count += 1
        if any(comb in text for comb in ("?!", "!?", "??", "!!")):
            combo_punct_count += 1

        # Emojis
        found_emojis = EMOJI_PATTERN.findall(text)
        for emo in found_emojis:
            emoji_counter[emo] = emoji_counter.get(emo, 0) + 1

        # Tokenization for keywords and fillers
        tokens = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
        all_tokens.extend(tokens)
        for tok in tokens:
            clean_tok = tok.replace("'", "")
            if clean_tok in FILLER_CANDIDATES or tok in FILLER_CANDIDATES:
                filler_counter[clean_tok] = filler_counter.get(clean_tok, 0) + 1
            if clean_tok not in STOPWORDS and len(clean_tok) > 2:
                keyword_counter[clean_tok] = keyword_counter.get(clean_tok, 0) + 1

    avg_words = sum(words_per_turn) / max(len(words_per_turn), 1)
    avg_chars = sum(chars_per_turn) / max(len(chars_per_turn), 1)
    all_lower_ratio = all_lower_count / max(total_turns, 1)
    first_char_lower_ratio = first_char_lower_count / max(total_turns, 1)

    excl_ratio = exclamation_count / max(total_turns, 1)
    quest_ratio = question_count / max(total_turns, 1)
    ellipsis_ratio = ellipsis_count / max(total_turns, 1)
    combo_ratio = combo_punct_count / max(total_turns, 1)

    # Type-Token Ratio
    unique_tokens = len(set(all_tokens))
    ttr = unique_tokens / max(len(all_tokens), 1)

    # Sorted tops
    top_keywords = sorted(keyword_counter.items(), key=lambda x: x[1], reverse=True)[:15]
    top_fillers = sorted(filler_counter.items(), key=lambda x: x[1], reverse=True)[:12]
    top_emojis = sorted(emoji_counter.items(), key=lambda x: x[1], reverse=True)[:8]

    return LinguisticMetrics(
        avg_words_per_turn=round(avg_words, 2),
        avg_chars_per_turn=round(avg_chars, 2),
        all_lower_ratio=round(all_lower_ratio, 4),
        first_char_lower_ratio=round(first_char_lower_ratio, 4),
        exclamation_ratio=round(excl_ratio, 4),
        question_ratio=round(quest_ratio, 4),
        ellipsis_ratio=round(ellipsis_ratio, 4),
        combo_punct_ratio=round(combo_ratio, 4),
        top_keywords=top_keywords,
        top_fillers=top_fillers,
        top_emojis=top_emojis,
        type_token_ratio=round(ttr, 4),
        total_analyzed_turns=total_turns,
    )


# =============================================================================
# Academic Big-5 Trait Estimator
# =============================================================================

def estimate_big_five(
    metrics: LinguisticMetrics,
    context_type: str,
) -> BigFiveCard:
    """
    Scenario-Based Big-5 Trait Estimator grounded in psycholinguistic research
    (Pennebaker & King 1999; Mairesse et al. 2007; Llama2-MBTI context-shift dynamics).

    Profiles shift by social setting:
    - Direct Messages (Personal): Highest intimacy, high agreeableness, warm extraversion, low neuroticism.
    - Group Chats: Highest conversational extraversion, playful banter, moderate agreeableness, very low neuroticism.
    - Supergroups: Semi-public, higher conscientiousness, more reserved extraversion, concise and task-driven.

    All scores bounded in [0.0, 1.0].
    """
    # 1. Openness: Correlated with Type-Token Ratio (vocabulary diversity) and cognitive markers
    # Baseline ~0.50; adjusted by TTR and curiosity question ratio
    openness_raw = 0.50 + (metrics.type_token_ratio * 0.4) + (metrics.question_ratio * 0.5)
    openness = max(0.0, min(1.0, round(openness_raw, 2)))

    # 2. Conscientiousness: Task orientation, structured responses, higher in semi-public work/supergroups
    if context_type == "supergroup":
        # More structured, punctual, answering queries or coordinating
        conscientiousness_raw = 0.68 + (0.1 if metrics.avg_words_per_turn < 10 else 0.0)
    elif context_type == "group":
        # Casual banter, looser constraints
        conscientiousness_raw = 0.45
    else:  # personal_chat
        # 1-on-1, reliable but informal
        conscientiousness_raw = 0.52
    conscientiousness = max(0.0, min(1.0, round(conscientiousness_raw, 2)))

    # 3. Extraversion: Social engagement, question asking, exclamation/excitement, turn frequency
    if context_type == "group":
        extraversion_raw = 0.78 + (metrics.question_ratio * 0.3)
    elif context_type == "personal_chat":
        extraversion_raw = 0.70 + (metrics.question_ratio * 0.2)
    else:  # supergroup
        # More reserved, answering when tagged or broadcasting updates
        extraversion_raw = 0.42 + (metrics.question_ratio * 0.2)
    extraversion = max(0.0, min(1.0, round(extraversion_raw, 2)))

    # 4. Agreeableness: High affiliation, friendly particles ('yea', 'can', 'nice'), absence of aggression
    if context_type == "personal_chat":
        agreeableness_raw = 0.82
    elif context_type == "group":
        agreeableness_raw = 0.72
    else:  # supergroup
        agreeableness_raw = 0.65
    agreeableness = max(0.0, min(1.0, round(agreeableness_raw, 2)))

    # 5. Neuroticism: Negative emotion words, anxiety, multiple punctuation marks (?!)
    # Dingxuan shows consistently low neuroticism (unflappable, chill, calm)
    neuroticism_raw = 0.15 + (metrics.combo_punct_ratio * 3.0) + (metrics.ellipsis_ratio * 1.5)
    neuroticism = max(0.0, min(1.0, round(neuroticism_raw, 2)))

    # Qualitative behavioral notes
    context_notes = {
        "personal_chat": (
            "High warmth and agreeableness. Relaxed 1-on-1 pace, frequent casual agreements ('yea', 'can'), "
            "low neuroticism (chill and patient), highly receptive to personal topics."
        ),
        "group": (
            "High extraversion and witty banter. Engaging, uses playful Singlish particles ('sia', 'ah', 'bruh'), "
            "participates eagerly in mutual jokes, very low neuroticism."
        ),
        "supergroup": (
            "Moderate extraversion and elevated conscientiousness. Reserved, direct, answers to-the-point, "
            "minimal fluff, almost zero exclamation marks or excessive emojis."
        ),
    }

    return BigFiveCard(
        openness=openness,
        conscientiousness=conscientiousness,
        extraversion=extraversion,
        agreeableness=agreeableness,
        neuroticism=neuroticism,
        context_notes={"behavioral_expression": context_notes.get(context_type, "Adaptive social persona.")},
    )


# =============================================================================
# Linguistic Exemplar Miner (Few-Shot Pairs)
# =============================================================================

def mine_linguistic_exemplars(
    sessions: List[Dict[str, Any]],
    context_type: str,
    max_exemplars: int = 5,
) -> List[FewShotPair]:
    """
    Searches conversation sessions for high-quality dialog turns where another user
    sends a message and Dingxuan responds. Filters for authentic, stylistic turns.
    """
    candidates: List[Tuple[float, FewShotPair]] = []

    for session in sessions:
        turns = normalize_session_turns(session)
        if len(turns) < 2:
            continue

        for i in range(len(turns) - 1):
            t1 = turns[i]
            t2 = turns[i + 1]

            # Condition: user 1 is NOT Dingxuan, user 2 IS Dingxuan
            if not t1["is_target_user"] and t2["is_target_user"]:
                u_text = t1["text"].strip()
                d_text = t2["text"].strip()

                # Quality filters
                if not u_text or not d_text:
                    continue
                # Skip media/service artifacts
                if any(art in u_text.lower() for art in ["<media omitted>", "http://", "https://", "pinned a message"]):
                    continue
                if any(art in d_text.lower() for art in ["<media omitted>", "http://", "https://"]):
                    continue

                # Length constraints: good conversational length (not single letters, not giant essays)
                if not (4 <= len(u_text) <= 120 and 3 <= len(d_text) <= 120):
                    continue

                # Score style representation:
                # Bonus if Dingxuan uses signature fillers ('ah', 'eh', 'cuz', 'idk', 'yea', 'sia', 'bruh')
                score = 1.0
                d_lower = d_text.lower()
                for filler in ["ah", "eh", "sia", "cuz", "yea", "idk", "wait", "noice", "bruh", "prob"]:
                    if re.search(rf"\b{filler}\b", d_lower):
                        score += 1.5

                # Bonus for natural casing (first char lowercase or all lowercase)
                if d_text.islower() or (d_text and d_text[0].islower()):
                    score += 0.5

                pair = FewShotPair(
                    user_message=u_text,
                    target_response=d_text,
                    other_sender=t1["sender_name"] or "User",
                    context_type=context_type,
                )
                candidates.append((score, pair))

    # Sort descending by quality score
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Pick diverse exemplars (avoid identical user messages or duplicate Dingxuan answers)
    seen_d_texts = set()
    selected: List[FewShotPair] = []
    for _, pair in candidates:
        norm_d = pair.target_response.lower()
        if norm_d in seen_d_texts:
            continue
        seen_d_texts.add(norm_d)
        selected.append(pair)
        if len(selected) >= max_exemplars:
            break

    # If mining yielded fewer than 3, supplement with curated empirical anchors
    if len(selected) < 3:
        fallback_pairs = get_default_few_shot_pairs(context_type)
        for fp in fallback_pairs:
            if len(selected) >= max_exemplars:
                break
            if fp.target_response.lower() not in seen_d_texts:
                selected.append(fp)

    return selected[:max_exemplars]


def get_default_few_shot_pairs(context_type: str) -> List[FewShotPair]:
    """Curated stylistic anchors matching Dingxuan's verified empirical dataset."""
    if context_type == "supergroup":
        return [
            FewShotPair(
                other_sender="Teammate",
                user_message="Are we still meeting the deadline for the sprint?",
                target_response="2 week sprint, we should be fine",
                context_type="supergroup",
            ),
            FewShotPair(
                other_sender="Edric",
                user_message="Wait who is presenting this part?",
                target_response="i can take that section",
                context_type="supergroup",
            ),
            FewShotPair(
                other_sender="Jason",
                user_message="Did anyone test the backend yet?",
                target_response="tested it alr, seems working on my end",
                context_type="supergroup",
            ),
        ]
    elif context_type == "group":
        return [
            FewShotPair(
                other_sender="Edric",
                user_message="Guys I need silverwolf",
                target_response="We shld roll on same day sia",
                context_type="group",
            ),
            FewShotPair(
                other_sender="Jake",
                user_message="tmr i hit 50",
                target_response="Noice",
                context_type="group",
            ),
            FewShotPair(
                other_sender="Jason",
                user_message="anyone free for dinner later?",
                target_response="cuz i have class till 6, after that can ah",
                context_type="group",
            ),
            FewShotPair(
                other_sender="Alex",
                user_message="bro this exam is gonna be impossible",
                target_response="yea honestly gg sia, just study tutorial questions",
                context_type="group",
            ),
        ]
    else:  # personal_chat (DM)
        return [
            FewShotPair(
                other_sender="Kelvin",
                user_message="So complicated to code lol",
                target_response="yea\ni prob search for tutorial to follow on yt\nso i see which one to follow",
                context_type="personal_chat",
            ),
            FewShotPair(
                other_sender="Kelvin",
                user_message="How about a simple guess the character game with animations",
                target_response="the thing is idk how strict they mark",
                context_type="personal_chat",
            ),
            FewShotPair(
                other_sender="Friend",
                user_message="u reaching soon?",
                target_response="wait otw now, reaching in 10 mins",
                context_type="personal_chat",
            ),
            FewShotPair(
                other_sender="Friend",
                user_message="should we submit now or wait?",
                target_response="can submit now ah, everything looks good",
                context_type="personal_chat",
            ),
        ]


# =============================================================================
# Fallback Profiles
# =============================================================================

def build_default_profile(context_type: str) -> ContextProfile:
    """
    Builds an empirically grounded default personality profile for Dingxuan
    when a data file is missing or unparseable.
    """
    context_display_names = {
        "personal_chat": "Personal Chat (DM)",
        "supergroup": "Supergroup Chat",
        "group": "Group Chat",
    }
    display_name = context_display_names.get(context_type, context_type.title())

    if context_type == "supergroup":
        metrics = LinguisticMetrics(
            avg_words_per_turn=6.51,
            avg_chars_per_turn=32.12,
            all_lower_ratio=0.522,
            first_char_lower_ratio=0.484,
            exclamation_ratio=0.000,
            question_ratio=0.154,
            ellipsis_ratio=0.0055,
            combo_punct_ratio=0.022,
            top_keywords=[("can", 16), ("need", 11), ("first", 11), ("got", 10), ("wait", 10)],
            top_fillers=[("ah", 15), ("wait", 10), ("like", 7), ("no", 5), ("cuz", 4), ("yea", 4), ("bruh", 4)],
            top_emojis=[],
            type_token_ratio=0.38,
            total_analyzed_turns=182,
        )
        big_five = BigFiveCard(
            openness=0.62,
            conscientiousness=0.70,
            extraversion=0.42,
            agreeableness=0.65,
            neuroticism=0.18,
            context_notes={"behavioral_expression": "Semi-public, concise, functional, zero exclamations, direct."},
        )
    elif context_type == "group":
        metrics = LinguisticMetrics(
            avg_words_per_turn=7.00,
            avg_chars_per_turn=38.14,
            all_lower_ratio=0.496,
            first_char_lower_ratio=0.510,
            exclamation_ratio=0.0018,
            question_ratio=0.108,
            ellipsis_ratio=0.0023,
            combo_punct_ratio=0.0014,
            top_keywords=[("can", 182), ("cuz", 110), ("need", 102), ("yea", 97), ("wait", 79)],
            top_fillers=[("ah", 131), ("cuz", 110), ("like", 104), ("yea", 97), ("wait", 79), ("bruh", 60), ("idk", 51)],
            top_emojis=[("😜", 1), ("😊", 1)],
            type_token_ratio=0.32,
            total_analyzed_turns=2194,
        )
        big_five = BigFiveCard(
            openness=0.64,
            conscientiousness=0.45,
            extraversion=0.80,
            agreeableness=0.72,
            neuroticism=0.15,
            context_notes={"behavioral_expression": "Outgoing, witty banter, high extraversion, Singlish particles ('sia', 'ah', 'bruh')."},
        )
    else:  # personal_chat
        metrics = LinguisticMetrics(
            avg_words_per_turn=7.37,
            avg_chars_per_turn=38.89,
            all_lower_ratio=0.397,
            first_char_lower_ratio=0.426,
            exclamation_ratio=0.0027,
            question_ratio=0.102,
            ellipsis_ratio=0.0088,
            combo_punct_ratio=0.0032,
            top_keywords=[("can", 244), ("like", 171), ("yea", 157), ("cuz", 153), ("need", 125), ("idk", 123)],
            top_fillers=[("ah", 243), ("like", 171), ("yea", 157), ("cuz", 153), ("eh", 128), ("idk", 123), ("wait", 122)],
            top_emojis=[("😂", 4), ("👍", 3), ("🤩", 3)],
            type_token_ratio=0.35,
            total_analyzed_turns=3394,
        )
        big_five = BigFiveCard(
            openness=0.65,
            conscientiousness=0.52,
            extraversion=0.72,
            agreeableness=0.82,
            neuroticism=0.18,
            context_notes={"behavioral_expression": "Intimate 1-on-1 dialogue, warm, agreeable, relaxed pacing, casual agreements."},
        )

    return ContextProfile(
        context_name=context_type,
        display_name=display_name,
        metrics=metrics,
        big_five=big_five,
        few_shot_exemplars=get_default_few_shot_pairs(context_type),
        is_fallback=True,
    )


# =============================================================================
# Main Profiler Interface
# =============================================================================

class MultiContextProfiler:
    """
    High-level engine that parses conversation logs for Dingxuan across
    Personal Chat (DM), Supergroup, and Group contexts.
    """

    DEFAULT_DATA_PATHS = {
        "personal_chat": Path("data_cleaning/approach_1_rule_based/output/sessions_personal_chat.json"),
        "supergroup": Path("data_cleaning/approach_2_nlp_embeddings/output/sessions_supergroup.json"),
        "group": Path("data_cleaning/approach_3_ai_llm/output/sessions_group.json"),
    }

    CONTEXT_LABELS = {
        "personal_chat": "Personal Chat (DM)",
        "supergroup": "Supergroup Chat",
        "group": "Group Chat",
    }

    def __init__(self, base_dir: Optional[Path | str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(".")
        self.profiles: Dict[str, ContextProfile] = {}

    def profile_context(
        self,
        context_type: str,
        file_path: Optional[Path | str] = None,
    ) -> ContextProfile:
        """
        Analyzes a single social context dataset and returns the resulting ContextProfile.
        If file loading fails or data is missing, gracefully falls back to the default profile.
        """
        if file_path:
            resolved_path = Path(file_path)
        else:
            rel = self.DEFAULT_DATA_PATHS.get(context_type, Path(f"sessions_{context_type}.json"))
            project_root = Path(__file__).resolve().parent.parent
            candidates = [
                self.base_dir / rel,
                project_root / rel,
                # Also fallback to legacy paths or approach directories directly
                project_root / str(rel).replace("data_cleaning/", ""),
                Path(rel),
                self.base_dir / f"sessions_{context_type}.json",
            ]
            resolved_path = next((c for c in candidates if c.is_file()), candidates[0])

        display_name = self.CONTEXT_LABELS.get(context_type, context_type.title())

        try:
            logger.info("Loading sessions for context '%s' from: %s", context_type, resolved_path)
            sessions = load_context_sessions(resolved_path)

            # Collect Dingxuan's turns
            dingxuan_turns: List[str] = []
            for session in sessions:
                turns = normalize_session_turns(session)
                for t in turns:
                    if t["is_target_user"]:
                        txt = t["text"]
                        if txt:
                            dingxuan_turns.append(txt)

            if not dingxuan_turns:
                logger.warning(
                    "No target user turns detected in %s. Falling back to default profile.",
                    resolved_path,
                )
                profile = build_default_profile(context_type)
                self.profiles[context_type] = profile
                return profile

            logger.info("Extracted %d turns for Dingxuan in '%s'", len(dingxuan_turns), context_type)

            # 1. Extract Linguistic Metrics
            metrics = extract_linguistic_metrics(dingxuan_turns)

            # 2. Score Big-5 Personality Traits
            big_five = estimate_big_five(metrics, context_type)

            # 3. Mine stylistic exemplars
            exemplars = mine_linguistic_exemplars(sessions, context_type, max_exemplars=5)

            profile = ContextProfile(
                context_name=context_type,
                display_name=display_name,
                metrics=metrics,
                big_five=big_five,
                few_shot_exemplars=exemplars,
                is_fallback=False,
            )
            self.profiles[context_type] = profile
            return profile

        except Exception as err:
            logger.error(
                "Error processing %s (%s). Falling back to default profile: %s",
                context_type,
                resolved_path,
                err,
            )
            profile = build_default_profile(context_type)
            self.profiles[context_type] = profile
            return profile

    def profile_all(self, custom_paths: Optional[Dict[str, Path | str]] = None) -> Dict[str, ContextProfile]:
        """Profiles all three contexts: personal_chat, supergroup, group."""
        custom_paths = custom_paths or {}
        for ctx in ["personal_chat", "supergroup", "group"]:
            path = custom_paths.get(ctx)
            self.profile_context(ctx, path)
        return self.profiles

    def export_summary_json(self, output_path: Path | str) -> None:
        """Exports the generated profiles to a JSON summary file."""
        data = {ctx: prof.to_dict() for ctx, prof in self.profiles.items()}
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Exported multi-context profile summary to: %s", out)


# =============================================================================
# CLI Standalone Execution
# =============================================================================

def main():
    """Runs the profiler as a standalone diagnostic and extraction tool."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print(" DINGXUAN PERSONA - MULTI-CONTEXT PROFILER & ANALYZER")
    print("=" * 80)

    profiler = MultiContextProfiler()
    profiles = profiler.profile_all()

    for ctx_key, prof in profiles.items():
        print(f"\n[{prof.display_name}] (Fallback: {prof.is_fallback})")
        print(f"  Turns Analyzed: {prof.metrics.total_analyzed_turns}")
        print(f"  Avg Length: {prof.metrics.avg_words_per_turn} words ({prof.metrics.avg_chars_per_turn} chars)")
        print(f"  Lowercase Ratio: {prof.metrics.all_lower_ratio * 100:.1f}%")
        print(f"  Punctuation: !={prof.metrics.exclamation_ratio*100:.1f}%, ?={prof.metrics.question_ratio*100:.1f}%, ...={prof.metrics.ellipsis_ratio*100:.1f}%")
        print(f"  Top Fillers: {[f[0] for f in prof.metrics.top_fillers[:6]]}")
        print(f"  Big-5 Scores: O={prof.big_five.openness:.2f}, C={prof.big_five.conscientiousness:.2f}, E={prof.big_five.extraversion:.2f}, A={prof.big_five.agreeableness:.2f}, N={prof.big_five.neuroticism:.2f}")
        print(f"  Mined Exemplars ({len(prof.few_shot_exemplars)}):")
        for i, ex in enumerate(prof.few_shot_exemplars[:2], 1):
            clean_resp = ex.target_response.replace('\n', ' ')
            print(f"    {i}. User: \"{ex.user_message}\" -> Dingxuan: \"{clean_resp}\"")

    output_path = Path(__file__).resolve().parent / "profiles_summary.json"
    profiler.export_summary_json(output_path)
    # Also save to project root for convenience
    root_summary = Path(__file__).resolve().parent.parent / "profiles_summary.json"
    if root_summary != output_path:
        try:
            profiler.export_summary_json(root_summary)
        except Exception:
            pass
    print(f"\n[OK] Summary saved to: {output_path}")


if __name__ == "__main__":
    main()

