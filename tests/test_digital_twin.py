"""
Unit and Integration Test Suite for the 6-Stage Digital Twin Architecture.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from digital_twin.config import (
    MemoryConfig,
    PersonalityConfig,
    PreprocessorConfig,
    TwinConfig,
)
from digital_twin.preprocessor import (
    ChatPreprocessor,
    ChatTurn,
    ChatSession,
    RawMessage,
)
from digital_twin.extractor import (
    DialogueExemplar,
    LinguisticMetrics,
    PersonalityExtractor,
)
from digital_twin.memory import (
    FastEmbedder,
    IdentityManager,
    MemoryEngine,
    MemoryNode,
    SQLiteVectorStore,
)
from digital_twin.twin_bot import (
    DigitalTwinBot,
    MockTwinBackend,
    SystemPromptBuilder,
)


class TestStage1Preprocessing(unittest.TestCase):
    """Verifies Stage 1: Burst concatenation, sessionization, and PII masking."""

    def setUp(self):
        self.config = PreprocessorConfig(
            burst_window_seconds=180,
            session_idle_gap_hours=18.0,
            anonymize_pii=True,
        )
        self.preprocessor = ChatPreprocessor(self.config)

    def test_pii_cleaning(self):
        dirty = (
            "Contact me at test.user@gmail.com or call +65 9123 4567! "
            "Server IP is 192.168.1.50 at 2024-01-01 12:00:00."
        )
        cleaned = self.preprocessor.clean_and_anonymize(dirty)
        self.assertNotIn("test.user@gmail.com", cleaned)
        self.assertIn("[EMAIL]", cleaned)
        self.assertNotIn("9123 4567", cleaned)
        self.assertIn("[PHONE]", cleaned)
        self.assertNotIn("192.168.1.50", cleaned)
        self.assertIn("[IP_ADDRESS]", cleaned)

    def test_burst_concatenation(self):
        t0 = 1700000000.0
        msgs = [
            RawMessage(
                sender_name="Dingxuan",
                sender_id="u1",
                timestamp=datetime.fromtimestamp(t0, tz=timezone.utc),
                unixtime=t0,
                text="hey bro",
            ),
            RawMessage(
                sender_name="Dingxuan",
                sender_id="u1",
                timestamp=datetime.fromtimestamp(t0 + 60, tz=timezone.utc),
                unixtime=t0 + 60,
                text="did you see the repo?",
            ),
            RawMessage(
                sender_name="Kelvin",
                sender_id="u2",
                timestamp=datetime.fromtimestamp(t0 + 90, tz=timezone.utc),
                unixtime=t0 + 90,
                text="yeah looking at it now",
            ),
        ]
        turns = self.preprocessor.concatenate_bursts(msgs)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].sender_name, "Dingxuan")
        self.assertEqual(turns[0].message_count, 2)
        self.assertIn("hey bro\ndid you see the repo?", turns[0].text)
        self.assertEqual(turns[1].sender_name, "Kelvin")
        self.assertEqual(turns[1].message_count, 1)

    def test_session_segmentation(self):
        t0 = 1700000000.0
        gap_20_hours = 20.0 * 3600.0
        turns = [
            ChatTurn(
                sender_name="Dingxuan",
                sender_id="u1",
                start_time="2023-11-14T10:00:00Z",
                end_time="2023-11-14T10:00:00Z",
                timestamp_unixtime=t0,
                text="Morning sync",
                message_count=1,
                chat_id="chat_1",
                chat_name="Project Chat",
            ),
            ChatTurn(
                sender_name="Kelvin",
                sender_id="u2",
                start_time="2023-11-15T07:00:00Z",
                end_time="2023-11-15T07:00:00Z",
                timestamp_unixtime=t0 + gap_20_hours,
                text="Next day update",
                message_count=1,
                chat_id="chat_1",
                chat_name="Project Chat",
            ),
        ]
        sessions = self.preprocessor.segment_sessions(turns)
        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0].session_id, "chat_1_sess_0001")
        self.assertEqual(sessions[1].session_id, "chat_1_sess_0002")


class TestStage2And3PersonalityAndDataset(unittest.TestCase):
    """Verifies Stage 2 & 3: Stylistic analysis, psychometrics, exemplar mining, and SFT format."""

    def setUp(self):
        self.extractor = PersonalityExtractor(PersonalityConfig())

    def test_linguistic_metrics(self):
        turns = [
            ChatTurn("Dingxuan", "u1", "", "", 0.0, "actually i can code that game ah", 1, "c1", "chat"),
            ChatTurn("Dingxuan", "u1", "", "", 10.0, "cause quiz is too basic haha", 1, "c1", "chat"),
            ChatTurn("Dingxuan", "u1", "", "", 20.0, "lemme check github rn", 1, "c1", "chat"),
        ]
        metrics = self.extractor.analyze_linguistics(turns)
        self.assertEqual(metrics.total_turns, 3)
        self.assertGreater(metrics.humor_frequency, 0.0)
        self.assertIn("ah", metrics.characteristic_fillers)
        self.assertIn("cause", metrics.characteristic_fillers)

    def test_exemplar_mining_and_finetune_export(self):
        turns = [
            ChatTurn("Kelvin", "u2", "2024-01-09T18:50:00Z", "2024-01-09T18:50:00Z", 100.0, "Do you prefer gamer crazy project?", 1, "c1", "chat"),
            ChatTurn("Dingxuan", "u1", "2024-01-09T18:52:00Z", "2024-01-09T18:52:00Z", 220.0, "Actually I don't think we should do quiz cause too many people gonna do quiz ah", 1, "c1", "chat"),
            ChatTurn("Kelvin", "u2", "2024-01-09T19:00:00Z", "2024-01-09T19:00:00Z", 700.0, "What about tetris?", 1, "c1", "chat"),
            ChatTurn("Dingxuan", "u1", "2024-01-09T19:02:00Z", "2024-01-09T19:02:00Z", 820.0, "Tetris can work if we add powerups haha", 1, "c1", "chat"),
        ]
        session = ChatSession("sess_1", "c1", "Dev Chat", "", "", ["Dingxuan", "Kelvin"], turns)
        exemplars = self.extractor.mine_exemplars([session], "Dingxuan", "u1")
        self.assertGreaterEqual(len(exemplars), 1)

        # Verify SFT Export
        chatml = self.extractor.export_for_finetuning(exemplars, "You are Dingxuan.", format_type="chatml")
        self.assertEqual(len(chatml), len(exemplars))
        self.assertEqual(chatml[0]["messages"][0]["role"], "system")
        self.assertEqual(chatml[0]["messages"][1]["role"], "user")
        self.assertEqual(chatml[0]["messages"][2]["role"], "assistant")


class TestStage4IdentityCore(unittest.TestCase):
    """Verifies Stage 4: Authoritative factual JSON profile."""

    def test_identity_file_validity(self):
        id_mgr = IdentityManager("digital_twin/identity.json")
        self.assertEqual(id_mgr.full_name, "Dingxuan Liang")
        self.assertEqual(id_mgr.preferred_name, "Dingxuan")
        summary = id_mgr.get_summary_markdown()
        self.assertIn("Singapore", summary)
        self.assertIn("Computer Science", summary)


class TestStage5ConflictAwareRAG(unittest.TestCase):
    """Verifies Stage 5: Dynamic, static, conditional conflict handling, and distractor rejection."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.config = MemoryConfig(db_path=self.temp_db.name, similarity_threshold=0.10)
        self.id_mgr = IdentityManager("digital_twin/identity.json")
        self.engine = MemoryEngine(
            config=self.config,
            identity_manager=self.id_mgr,
            target_user_name="Dingxuan Liang"
        )
        self.engine.store.clear()

    def tearDown(self):
        try:
            Path(self.temp_db.name).unlink(missing_ok=True)
        except Exception:
            pass

    def test_dynamic_temporal_conflict_resolution(self):
        docs = [
            {
                "id": "loc_old",
                "content": "I live in Jurong West, Singapore.",
                "timestamp": "2021-05-01T10:00:00Z",
                "owner": "Dingxuan Liang",
                "category": "location",
            },
            {
                "id": "loc_new",
                "content": "I moved and now live in Clementi, Singapore.",
                "timestamp": "2024-03-01T10:00:00Z",
                "owner": "Dingxuan Liang",
                "category": "location",
            }
        ]
        self.engine.ingest_documents(docs)
        retrieved = self.engine.retrieve("Where do you live?")
        # Only the latest location should be active
        active_ids = [m.id for m in retrieved]
        self.assertIn("loc_new", active_ids)
        self.assertNotIn("loc_old", active_ids)

    def test_static_conflict_resolution(self):
        docs = [
            {
                "id": "bad_birthplace",
                "content": "I was born in Boston and grew up there.",
                "timestamp": "2024-01-01T10:00:00Z",
                "owner": "Dingxuan Liang",
                "category": "general",
            }
        ]
        self.engine.ingest_documents(docs)
        retrieved = self.engine.retrieve("Where were you born?")
        active_ids = [m.id for m in retrieved]
        self.assertNotIn("bad_birthplace", active_ids)

    def test_distractor_filtering(self):
        docs = [
            {
                "id": "friend_quote",
                "content": "I completely despise writing Python code.",
                "timestamp": "2024-02-01T10:00:00Z",
                "owner": "Alice (Coworker)",
                "category": "preference",
            }
        ]
        self.engine.ingest_documents(docs)
        retrieved = self.engine.retrieve("Do you despise Python?")
        active_ids = [m.id for m in retrieved]
        self.assertNotIn("friend_quote", active_ids)

    def test_conditional_conflict_resolution(self):
        docs = [
            {
                "id": "drink_morning",
                "content": "I drink black coffee in the morning for coding.",
                "timestamp": "2024-01-01T08:00:00Z",
                "owner": "Dingxuan Liang",
                "category": "preference",
                "condition": {"time_of_day": "morning"},
            },
            {
                "id": "drink_evening",
                "content": "I drink green tea in the evening to relax.",
                "timestamp": "2024-01-01T20:00:00Z",
                "owner": "Dingxuan Liang",
                "category": "preference",
                "condition": {"time_of_day": "evening"},
            }
        ]
        self.engine.ingest_documents(docs)
        morning_res = self.engine.retrieve("What do you drink?", query_context={"time_of_day": "morning"})
        morning_ids = [m.id for m in morning_res]
        self.assertIn("drink_morning", morning_ids)
        self.assertNotIn("drink_evening", morning_ids)


class TestStage6PromptAndTwinOrchestration(unittest.TestCase):
    """Verifies Stage 6: Priority prompt hierarchy and uncertainty refusal."""

    def setUp(self):
        self.id_mgr = IdentityManager("digital_twin/identity.json")
        self.metrics = LinguisticMetrics(formality_score=0.35, lowercase_start_ratio=0.7)
        self.profile = self.extractor = PersonalityExtractor().assess_psychometrics(self.metrics)
        self.exemplar = DialogueExemplar("ex1", "chat", "Kelvin", "wanna game?", "can, give me 10 mins", 0.8)
        self.mem = MemoryNode("m1", "I worked on a game project in Clementi", "2024-01-01T00:00:00Z", "Dingxuan Liang", "project")

    def test_prompt_hierarchy_order(self):
        prompt = SystemPromptBuilder.build(
            identity=self.id_mgr,
            metrics=self.metrics,
            profile=self.profile,
            exemplars=[self.exemplar],
            retrieved_memories=[self.mem],
            refusal_phrase="I don't have enough information to know how Dingxuan would answer that."
        )

        # Verify strict priority ordering in the prompt string
        pos_p1 = prompt.find("PRIORITY 1: PERSONALITY RULES")
        pos_few_shot = prompt.find("FEW-SHOT PERSONALITY EXEMPLARS")
        pos_p2 = prompt.find("PRIORITY 2: STABLE PROFILE")
        pos_p3 = prompt.find("PRIORITY 3: RELEVANT CONTEXTUAL MEMORIES")
        pos_refusal = prompt.find("UNCERTAINTY DISCIPLINE & REFUSAL DIRECTIVE")

        self.assertNotEqual(pos_p1, -1)
        self.assertNotEqual(pos_few_shot, -1)
        self.assertNotEqual(pos_p2, -1)
        self.assertNotEqual(pos_p3, -1)
        self.assertNotEqual(pos_refusal, -1)

        self.assertLess(pos_p1, pos_few_shot)
        self.assertLess(pos_few_shot, pos_p2)
        self.assertLess(pos_p2, pos_p3)
        self.assertLess(pos_p3, pos_refusal)

    def test_uncertainty_refusal(self):
        bot = DigitalTwinBot()
        # Query about an obscure topic not in Identity Core or RAG
        reply = bot.query("What was the serial number on your first bicycle?")
        self.assertEqual(reply, bot.config.llm.refusal_phrase)


if __name__ == "__main__":
    unittest.main()
