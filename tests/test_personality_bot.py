"""
Unit and Integration Tests for Context-Adaptive Personality Chatbot ("Dingxuan").
Tests profiler.py, synthesizer.py, and chat_app.py.
"""

import json
import tempfile
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = PROJECT_ROOT / "llm"
CLEANING_DIR = PROJECT_ROOT / "data_cleaning"
for p in [str(PROJECT_ROOT), str(LLM_DIR), str(CLEANING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from llm.chat_app import (
        PersonalityChatApp,
        RefusalDeflectionLayer,
        SimulatedClient,
    )
    from llm.profiler import (
        BigFiveCard,
        ContextProfile,
        LinguisticMetrics,
        MultiContextProfiler,
        build_default_profile,
        estimate_big_five,
        extract_linguistic_metrics,
        is_dingxuan_sender,
        load_context_sessions,
        mine_linguistic_exemplars,
        normalize_session_turns,
    )
    from llm.synthesizer import SystemPromptSynthesizer
except ImportError:
    from chat_app import (
        PersonalityChatApp,
        RefusalDeflectionLayer,
        SimulatedClient,
    )
    from profiler import (
        BigFiveCard,
        ContextProfile,
        LinguisticMetrics,
        MultiContextProfiler,
        build_default_profile,
        estimate_big_five,
        extract_linguistic_metrics,
        is_dingxuan_sender,
        load_context_sessions,
        mine_linguistic_exemplars,
        normalize_session_turns,
    )
    from synthesizer import SystemPromptSynthesizer


class TestProfilerModule(unittest.TestCase):
    """Tests for Module 1: profiler.py."""

    def test_is_dingxuan_sender(self):
        self.assertTrue(is_dingxuan_sender("Dingxuan Liang", False))
        self.assertTrue(is_dingxuan_sender("Dingxuan", False))
        self.assertTrue(is_dingxuan_sender("Random User", True))
        self.assertFalse(is_dingxuan_sender("Edric Yeo", False))
        self.assertFalse(is_dingxuan_sender(None, False))

    def test_load_context_sessions_dual_schema(self):
        """Test loading both repository schema and simplified prompt schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Schema 1: Prompt specification schema
            prompt_schema_path = Path(tmpdir) / "prompt_schema.json"
            prompt_data = [
                {
                    "session_id": "sess_001",
                    "messages": [
                        {"sender": "Kelvin", "text": "Are you reaching soon?"},
                        {"sender": "Dingxuan Liang", "text": "wait otw now, reaching in 5 mins"},
                    ],
                }
            ]
            with open(prompt_schema_path, "w", encoding="utf-8") as f:
                json.dump(prompt_data, f)

            sessions1 = load_context_sessions(prompt_schema_path)
            self.assertEqual(len(sessions1), 1)
            turns1 = normalize_session_turns(sessions1[0])
            self.assertEqual(len(turns1), 2)
            self.assertFalse(turns1[0]["is_target_user"])
            self.assertTrue(turns1[1]["is_target_user"])
            self.assertIn("otw", turns1[1]["text"])

            # Schema 2: Repository schema {"conversations": [...]}
            repo_schema_path = Path(tmpdir) / "repo_schema.json"
            repo_data = {
                "summary": {"total_conversations": 1},
                "conversations": [
                    {
                        "conversation_id": "conv_001",
                        "turns": [
                            {"sender_name": "Jason", "is_target_user": False, "text": "eat what?"},
                            {"sender_name": "Dingxuan Liang", "is_target_user": True, "text": "munch can ah"},
                        ],
                    }
                ],
            }
            with open(repo_schema_path, "w", encoding="utf-8") as f:
                json.dump(repo_data, f)

            sessions2 = load_context_sessions(repo_schema_path)
            self.assertEqual(len(sessions2), 1)
            turns2 = normalize_session_turns(sessions2[0])
            self.assertEqual(len(turns2), 2)
            self.assertFalse(turns2[0]["is_target_user"])
            self.assertTrue(turns2[1]["is_target_user"])

    def test_resilience_missing_and_corrupt_files(self):
        """Test fallback profile generation when file is missing or corrupted."""
        profiler = MultiContextProfiler()

        # Missing file
        missing_prof = profiler.profile_context("personal_chat", file_path="non_existent_file.json")
        self.assertTrue(missing_prof.is_fallback)
        self.assertGreater(missing_prof.metrics.avg_words_per_turn, 0)
        self.assertGreaterEqual(missing_prof.big_five.extraversion, 0.0)

        # Corrupt file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            f.write("{corrupt json...")
            corrupt_path = f.name

        try:
            corrupt_prof = profiler.profile_context("supergroup", file_path=corrupt_path)
            self.assertTrue(corrupt_prof.is_fallback)
            self.assertEqual(corrupt_prof.context_name, "supergroup")
        finally:
            Path(corrupt_path).unlink(missing_ok=True)

    def test_extract_linguistic_metrics(self):
        sample_turns = [
            "yea can but idk how to do it ah",
            "wait i check first",
            "whut time?",
            "noice",
            "bruh idk why it failed?!",
        ]
        metrics = extract_linguistic_metrics(sample_turns)
        self.assertEqual(metrics.total_analyzed_turns, 5)
        self.assertGreater(metrics.avg_words_per_turn, 0)
        self.assertGreater(metrics.question_ratio, 0)
        self.assertGreater(metrics.combo_punct_ratio, 0)
        top_filler_keys = [f[0] for f in metrics.top_fillers]
        self.assertTrue(any(fil in top_filler_keys for fil in ["idk", "wait", "ah", "bruh"]))

    def test_estimate_big_five(self):
        """Test that Big-5 estimation stays in [0.0, 1.0] and shifts with context."""
        sample_turns = ["yea can", "noice", "whut time ah?", "wait for me"]
        metrics = extract_linguistic_metrics(sample_turns)

        b5_dm = estimate_big_five(metrics, "personal_chat")
        b5_group = estimate_big_five(metrics, "group")
        b5_super = estimate_big_five(metrics, "supergroup")

        for card in [b5_dm, b5_group, b5_super]:
            for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
                score = getattr(card, trait)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

        # Supergroup has higher conscientiousness than casual group
        self.assertGreater(b5_super.conscientiousness, b5_group.conscientiousness)
        # Group chat has higher extraversion than supergroup
        self.assertGreater(b5_group.extraversion, b5_super.extraversion)
        # DM has highest agreeableness
        self.assertGreaterEqual(b5_dm.agreeableness, b5_super.agreeableness)

    def test_mine_linguistic_exemplars(self):
        sessions = [
            {
                "turns": [
                    {"sender_name": "Friend", "is_target_user": False, "text": "Are you going?"},
                    {"sender_name": "Dingxuan Liang", "is_target_user": True, "text": "yea otw now ah"},
                ]
            }
        ]
        exemplars = mine_linguistic_exemplars(sessions, "personal_chat", max_exemplars=3)
        self.assertGreaterEqual(len(exemplars), 1)
        self.assertIn("otw", exemplars[0].target_response)


class TestSynthesizerModule(unittest.TestCase):
    """Tests for Module 2: synthesizer.py."""

    def setUp(self):
        self.profiler = MultiContextProfiler()
        self.synthesizer = SystemPromptSynthesizer(self.profiler)

    def test_system_prompt_structure(self):
        prompt = self.synthesizer.build_system_prompt(context_type="personal_chat")

        self.assertIn("# Target Persona: Dingxuan", prompt)
        self.assertIn("## Absolute Linguistic Constraints", prompt)
        self.assertIn("## Scenario Big-5 Personality Directives", prompt)
        self.assertIn("## Linguistic Profile & Discourse Markers", prompt)
        self.assertIn("## In-Character Refusal & Deflection Policy", prompt)
        self.assertIn("## Context-Specific Stylistic Exemplars", prompt)

    def test_mode_specific_constraints(self):
        # Supergroup prompt should prohibit/limit emojis and emphasize concise task focus
        sg_prompt = self.synthesizer.build_system_prompt(context_type="supergroup")
        self.assertIn("Emoji Prohibition", sg_prompt)
        self.assertIn("Supergroup", sg_prompt)

        # Group prompt should emphasize extraversion and witty banter
        grp_prompt = self.synthesizer.build_system_prompt(context_type="group")
        self.assertIn("banter", grp_prompt.lower())


class TestChatAppModule(unittest.TestCase):
    """Tests for Module 3: chat_app.py."""

    def test_refusal_deflection_triggers(self):
        """Test that biographical and memory probing triggers deflection."""
        deflection_queries = [
            "Where were you born and what is your birthday?",
            "What is your home address and IC number?",
            "Do you remember what we talked about last year?",
            "What did we discuss back then in 2022?",
            "What is your life story from childhood?",
            "Tell me your background and past history",
            "Where do you live right now?",
            "What primary school did you attend?",
            "Who are your parents?",
            "Are you an AI chatbot?",
        ]
        for query in deflection_queries:
            self.assertTrue(
                RefusalDeflectionLayer.check_query(query),
                f"Expected '{query}' to trigger deflection.",
            )

        benign_queries = [
            "yo are you free for lunch later?",
            "can we meet up at 5pm at Munch?",
            "whut time does the tutorial start?",
            "did you submit the code assignment?",
            "how do we fix this git merge conflict?",
        ]
        for query in benign_queries:
            self.assertFalse(
                RefusalDeflectionLayer.check_query(query),
                f"Expected '{query}' NOT to trigger deflection.",
            )

    def test_refusal_deflection_responses(self):
        """Ensure deflections match Dingxuan's voice and do not leak AI tropes."""
        for ctx in ["personal_chat", "group", "supergroup"]:
            resp = RefusalDeflectionLayer.get_deflection(ctx)
            self.assertIsInstance(resp, str)
            self.assertNotIn("As an AI", resp)
            self.assertNotIn("I cannot fulfill", resp)
            self.assertNotIn("OpenAI", resp)

    def test_sliding_window_buffer(self):
        """Test that history maintains exactly 5 turns (10 messages max)."""
        app = PersonalityChatApp(context_type="personal_chat", window_size=5)

        for i in range(8):
            app.respond(f"test message {i}")

        # Window size is 5 turns => max 10 messages (5 user + 5 assistant)
        self.assertEqual(len(app.history), 10)
        # Most recent turn should be message 7
        self.assertIn("test message 7", app.history[-2]["content"])

    def test_simulated_client_responses(self):
        client = SimulatedClient(context_type="personal_chat")
        prompt = "System instructions..."

        # Greeting test
        r_greet = client.generate_response(prompt, [{"role": "user", "content": "yo"}])
        self.assertIn("yo", r_greet.lower())

        # Question test
        r_time = client.generate_response(prompt, [{"role": "user", "content": "whut time?"}])
        self.assertTrue(len(r_time) > 0)


if __name__ == "__main__":
    unittest.main()

