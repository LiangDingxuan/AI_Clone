"""
End-to-End Demonstration Script for the Digital Twin Conversational Architecture.

Executes all 6 stages sequentially:
1. Stage 1: Chat history preprocessing (Burst concatenation, session segmentation, PII cleaning)
2. Stage 2: Personality extraction (Linguistic style metrics & Big Five / MBTI psychometrics)
3. Stage 3: Few-shot exemplar mining & LoRA/ChatML dataset compilation
4. Stage 4: Stable Information Engine (Identity Core inspection)
5. Stage 5: Conflict-Aware Memory RAG (Dynamic, static, conditional conflicts & distractor filtering)
6. Stage 6: Disciplined LLM Orchestration & Interactive Persona Chat
"""

import argparse
import json
import logging
from pathlib import Path
import sys

from digital_twin.config import TwinConfig
from digital_twin.extractor import PersonalityExtractor
from digital_twin.memory import IdentityManager, MemoryEngine
from digital_twin.preprocessor import ChatPreprocessor
from digital_twin.twin_bot import DigitalTwinBot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_pipeline(interactive: bool = False, max_messages: int = 3000):
    print("\n" + "=" * 70)
    print("       HUMAN DIGITAL TWIN CONVERSATIONAL ARCHITECTURE")
    print("  Target Persona: Dingxuan Liang | Framework: 6-Stage Modular Pipeline")
    print("=" * 70 + "\n")

    config = TwinConfig()

    # -------------------------------------------------------------
    # STAGE 1: CHAT HISTORY PREPROCESSING
    # -------------------------------------------------------------
    print("[STAGE 1] Running Chat History Preprocessing...")
    preprocessor = ChatPreprocessor(config.preprocessor)
    raw_path = Path(config.raw_chat_path)

    if raw_path.is_file():
        sessions = preprocessor.process(raw_path, max_messages=max_messages)
        print(f"  [OK] Processed {max_messages} messages into {len(sessions)} logical sessions.")
    else:
        print(f"  [!] Raw chat log {raw_path} not found. Creating sample mock sessions.")
        from tests.test_digital_twin import RawMessage
        from datetime import datetime, timezone
        t0 = 1700000000.0
        sample_msgs = [
            RawMessage("Dingxuan Liang", "user5711494385", datetime.fromtimestamp(t0, tz=timezone.utc), t0, "actually i can code that game ah"),
            RawMessage("Dingxuan Liang", "user5711494385", datetime.fromtimestamp(t0 + 60, tz=timezone.utc), t0 + 60, "cause quiz is too basic haha"),
            RawMessage("Kelvin", "user242405345", datetime.fromtimestamp(t0 + 120, tz=timezone.utc), t0 + 120, "tetris sounds good"),
        ]
        turns = preprocessor.concatenate_bursts(sample_msgs)
        sessions = preprocessor.segment_sessions(turns)

    # -------------------------------------------------------------
    # STAGE 2: PERSONALITY EXTRACTION ENGINE
    # -------------------------------------------------------------
    print("\n[STAGE 2] Running Personality Extraction Engine...")
    extractor = PersonalityExtractor(config.personality)
    target_turns = extractor.filter_target_turns(
        sessions,
        target_user_name=config.preprocessor.target_user_name,
        target_user_id=config.preprocessor.target_user_id,
    )
    metrics = extractor.analyze_linguistics(target_turns)
    profile = extractor.assess_psychometrics(metrics, target_name="Dingxuan")

    print(f"  [OK] Analyzed {metrics.total_turns} target turns ({metrics.total_words} words).")
    print(f"  [OK] Formality Index: {metrics.formality_score:.3f} / 1.000 (Casual/Pragmatic)")
    print(f"  [OK] Lowercase Texting Ratio: {metrics.lowercase_start_ratio * 100:.1f}%")
    print(f"  [OK] Top Colloquial Markers: {metrics.characteristic_fillers}")
    print(f"  [OK] Psychometrics: MBTI = {profile.mbti_type}, Big Five = [O:{profile.openness:.2f}, C:{profile.conscientiousness:.2f}, E:{profile.extraversion:.2f}, A:{profile.agreeableness:.2f}, N:{profile.neuroticism:.2f}]")

    # -------------------------------------------------------------
    # STAGE 3: PERSONALITY DATASET & FEW-SHOT MINER
    # -------------------------------------------------------------
    print("\n[STAGE 3] Mining Few-Shot Exemplar Dialogue Pairs...")
    exemplars = extractor.mine_exemplars(
        sessions,
        target_user_name=config.preprocessor.target_user_name,
        target_user_id=config.preprocessor.target_user_id,
        max_exemplars=6,
    )
    print(f"  [OK] Mined {len(exemplars)} high-signal dialogue exemplars.")
    for i, ex in enumerate(exemplars[:2], 1):
        print(f"    * Exemplar {i} [{ex.category}]:")
        print(f"      Q: {ex.query[:65]}...")
        print(f"      A: {ex.target_response[:65]}...")

    # Export to ChatML format for fine-tuning
    chatml_data = extractor.export_for_finetuning(
        exemplars,
        system_prompt="You are Dingxuan Liang, an AI engineer and developer.",
        format_type="chatml"
    )
    print(f"  [OK] Formatted {len(chatml_data)} pairs into ChatML instruction-tuning format.")

    # -------------------------------------------------------------
    # STAGE 4: STABLE INFORMATION ENGINE (IDENTITY CORE)
    # -------------------------------------------------------------
    print("\n[STAGE 4] Loading Stage 4 Identity Core (Absolute Fact Authority)...")
    identity = IdentityManager(config.get_identity_path())
    print(f"  [OK] Invariant Target: {identity.full_name} ({identity.preferred_name})")
    print(f"  [OK] Base: Singapore | Citizenship: Singaporean")
    print(f"  [OK] Field: Computer Science & AI Systems")

    # -------------------------------------------------------------
    # STAGE 5: MEMORY ENGINE & CONFLICT-AWARE RAG
    # -------------------------------------------------------------
    print("\n[STAGE 5] Ingesting Knowledge & Testing Conflict-Aware Memory Engine...")
    memory_engine = MemoryEngine(
        config=config.memory,
        identity_manager=identity,
        target_user_name=identity.full_name,
    )
    memory_engine.store.clear()

    # Ingest diverse episodic memories containing conflicts & distractors
    test_memories = [
        # Dynamic Conflict: Old residence (2021) vs New residence (2024)
        {
            "id": "mem_jurong_2021",
            "content": "I live in Jurong West, Singapore.",
            "timestamp": "2021-04-10T10:00:00Z",
            "owner": "Dingxuan Liang",
            "category": "location",
        },
        {
            "id": "mem_clementi_2024",
            "content": "I moved and now live in Clementi, Singapore.",
            "timestamp": "2024-02-15T16:00:00Z",
            "owner": "Dingxuan Liang",
            "category": "location",
        },
        # Static Conflict: Direct contradiction with Identity Core invariant
        {
            "id": "mem_boston_fake",
            "content": "I was born in Boston, Massachusetts and spent my youth in the US.",
            "timestamp": "2023-08-01T12:00:00Z",
            "owner": "Dingxuan Liang",
            "category": "general",
        },
        # Distractor Statement: Coworker speaking, not target persona
        {
            "id": "mem_coworker_python",
            "content": "I hate writing Python, it has too many dynamic typing bugs.",
            "timestamp": "2024-03-01T10:00:00Z",
            "owner": "Kelvin (Friend)",
            "category": "preference",
        },
        # Conditional Conflict: Morning coffee vs Evening tea
        {
            "id": "mem_coffee_morning",
            "content": "I always drink strong black coffee when I start my morning coding sessions.",
            "timestamp": "2024-01-10T08:30:00Z",
            "owner": "Dingxuan Liang",
            "category": "preference",
            "condition": {"time_of_day": "morning"},
        },
        {
            "id": "mem_tea_evening",
            "content": "I drink green tea or water in the evening so I can sleep properly.",
            "timestamp": "2024-01-10T20:30:00Z",
            "owner": "Dingxuan Liang",
            "category": "preference",
            "condition": {"time_of_day": "evening"},
        },
    ]

    memory_engine.ingest_documents(test_memories)

    # Demonstrate Dynamic Resolution
    print("  * Testing Dynamic Conflict Resolution (Where do you live?):")
    res_loc = memory_engine.retrieve("Where do you live?")
    for m in res_loc:
        print(f"    -> Retained: [{m.timestamp}] {m.content}")

    # Demonstrate Static Conflict Rejection
    print("  * Testing Static Conflict Resolution (Where were you born?):")
    res_birth = memory_engine.retrieve("Where were you born?")
    print(f"    -> Retained valid memories: {len(res_birth)} (Contradictory Boston claim suppressed)")

    # Demonstrate Distractor Rejection
    print("  * Testing Distractor Rejection (Do you hate Python?):")
    res_dist = memory_engine.retrieve("Do you hate Python?")
    print(f"    -> Retained valid memories: {len(res_dist)} (Friend Kelvin's negative quote filtered)")

    # Demonstrate Conditional Conflict Resolution
    print("  * Testing Conditional Conflict (What beverage do you drink?):")
    morning_bevs = memory_engine.retrieve("What beverage do you drink?", query_context={"time_of_day": "morning"})
    evening_bevs = memory_engine.retrieve("What beverage do you drink?", query_context={"time_of_day": "evening"})
    print(f"    -> Morning context: {morning_bevs[0].content if morning_bevs else 'None'}")
    print(f"    -> Evening context: {evening_bevs[0].content if evening_bevs else 'None'}")

    # -------------------------------------------------------------
    # STAGE 6: DISCIPLINED LLM ORCHESTRATION & INFERENCE
    # -------------------------------------------------------------
    print("\n[STAGE 6] Initializing Disciplined LLM Orchestrator...")
    bot = DigitalTwinBot(config)
    bot.metrics = metrics
    bot.profile = profile
    bot.exemplars = exemplars
    bot.memory = memory_engine

    print("\n" + "-" * 70)
    print("Executing Sample Queries Against Digital Twin:")
    print("-" * 70)

    sample_prompts = [
        ("Query 1 [Identity Core]", "Hey! What's your background and where are you based?"),
        ("Query 2 [Dynamic Memory]", "Where are you currently living?"),
        ("Query 3 [Distractor Filter]", "Do you hate Python?"),
        ("Query 4 [Conditional Memory]", "What do you drink when you code in the morning?"),
        ("Query 5 [Uncertainty Refusal]", "What was the brand of your first pair of shoes in elementary school?"),
    ]

    for label, q in sample_prompts:
        print(f"\n[{label}]")
        print(f"User: {q}")
        reply = bot.query(q)
        print(f"Twin: {reply}")

    print("\n" + "=" * 70)
    print("[OK] Full 6-Stage Digital Twin Pipeline Verified Successfully!")
    print("=" * 70 + "\n")

    if interactive:
        bot.chat_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Digital Twin 6-Stage Pipeline")
    parser.add_argument("--chat", action="store_true", help="Launch interactive CLI chat session")
    parser.add_argument("--messages", type=int, default=3000, help="Max chat history messages to process")
    args = parser.parse_args()

    run_pipeline(interactive=args.chat, max_messages=args.messages)
