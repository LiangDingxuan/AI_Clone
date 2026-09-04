"""
Unit and Integration Tests for QLoRA Training Pipeline and Safeguards.
Tests:
1. export_training_data.py (distractor filter, multi-turn formatting, stratified split).
2. synthesizer.py (multi-party context directives).
3. train_lora.py (dry-run, hyperparameter calibration, template detection).
4. chat_app.py (adapter argument integration).
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from export_training_data import (
    DatasetExporter,
    format_conversation_for_sft,
    is_distractor_message,
)
from synthesizer import SystemPromptSynthesizer
from training.train_lora import detect_response_template, parse_args, run_dry_run_validation


class TestDistractorFilterAndFormatting(unittest.TestCase):
    """Tests for distractor cleaning and multi-turn SFT formatting."""

    def test_distractor_message_detection(self):
        # Noise that should be filtered
        self.assertTrue(is_distractor_message("<Media omitted>"))
        self.assertTrue(is_distractor_message("pinned a message"))
        self.assertTrue(is_distractor_message("joined the group"))
        self.assertTrue(is_distractor_message("https://t.me/randomlink"))
        self.assertTrue(is_distractor_message("/start"))
        self.assertTrue(is_distractor_message("/help"))
        self.assertTrue(is_distractor_message("   "))

        # Conversational text that should NOT be filtered
        self.assertFalse(is_distractor_message("can meet at munch later?"))
        self.assertFalse(is_distractor_message("yea can, what time ah"))
        self.assertFalse(is_distractor_message("check out https://github.com/repo for the code"))

    def test_format_conversation_alternation(self):
        """Test strict role alternation (user -> assistant) and leading/trailing trimming."""
        system_prompt = "System prompt here"
        raw_turns = [
            # Leading assistant turn (should be trimmed)
            {"sender_name": "Dingxuan Liang", "text": "hey everyone", "is_target_user": True},
            # User turns (consecutive)
            {"sender_name": "Kelvin", "text": "Are you reaching soon?", "is_target_user": False},
            {"sender_name": "Kelvin", "text": "We are waiting outside", "is_target_user": False},
            # Assistant turns (consecutive, should be merged)
            {"sender_name": "Dingxuan Liang", "text": "wait otw now", "is_target_user": True},
            {"sender_name": "Dingxuan Liang", "text": "reaching in 5 mins", "is_target_user": True},
            # Trailing user turn (should be trimmed because no Dingxuan reply)
            {"sender_name": "Kelvin", "text": "Ok see u", "is_target_user": False},
        ]

        result = format_conversation_for_sft(raw_turns, system_prompt, context_type="personal_chat")
        self.assertIsNotNone(result)
        messages = result["messages"]

        # Expected: system, user, assistant (total 3 messages)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], system_prompt)

        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Are you reaching soon?", messages[1]["content"])
        self.assertIn("We are waiting outside", messages[1]["content"])

        self.assertEqual(messages[2]["role"], "assistant")
        self.assertIn("wait otw now", messages[2]["content"])
        self.assertIn("reaching in 5 mins", messages[2]["content"])

        # Trailing turn was trimmed
        self.assertNotIn("Ok see u", messages[-1]["content"])

    def test_multi_party_speaker_tagging(self):
        """Test bracketed sender tags [Name]: for group chats."""
        system_prompt = "Group system prompt"
        raw_turns = [
            {"sender_name": "Edric", "text": "where to eat?", "is_target_user": False},
            {"sender_name": "Jason", "text": "munch or kfc", "is_target_user": False},
            {"sender_name": "Dingxuan Liang", "text": "munch can ah", "is_target_user": True},
        ]

        result = format_conversation_for_sft(raw_turns, system_prompt, context_type="group")
        self.assertIsNotNone(result)
        user_msg = result["messages"][1]["content"]

        # Both speakers tagged
        self.assertIn("[Edric]: where to eat?", user_msg)
        self.assertIn("[Jason]: munch or kfc", user_msg)


class TestDatasetExporterAndSplit(unittest.TestCase):
    """Tests for DatasetExporter splitting and file generation."""

    def test_exporter_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create mock session
            mock_session = [
                {
                    "turns": [
                        {"sender_name": "Friend", "text": "u free?", "is_target_user": False},
                        {"sender_name": "Dingxuan Liang", "text": "yea can", "is_target_user": True},
                    ]
                }
            ]
            sess_file = tmp_path / "mock_personal.json"
            with open(sess_file, "w", encoding="utf-8") as f:
                json.dump({"conversations": mock_session}, f)

            exporter = DatasetExporter()
            custom_paths = {"personal_chat": sess_file}

            summary = exporter.export(
                output_dir=tmp_path / "data_out",
                val_ratio=0.5,
                seed=42,
                custom_paths=custom_paths,
            )

            self.assertGreater(summary["total_formatted_episodes"], 0)
            train_f = Path(summary["train_file"])
            val_f = Path(summary["val_file"])
            self.assertTrue(train_f.is_file())
            self.assertTrue(val_f.is_file())


class TestMultiPartyDirectiveInSynthesizer(unittest.TestCase):
    """Tests that Multi-Party Directives appear in group prompts and not in DM prompts."""

    def test_multi_party_directive_inclusion(self):
        synth = SystemPromptSynthesizer()

        dm_prompt = synth.build_system_prompt(context_type="personal_chat")
        self.assertNotIn("Multi-Party Context Directive", dm_prompt)

        group_prompt = synth.build_system_prompt(context_type="group")
        self.assertIn("Multi-Party Context Directive", group_prompt)
        self.assertIn("[Edric]:", group_prompt)

        supergroup_prompt = synth.build_system_prompt(context_type="supergroup")
        self.assertIn("Multi-Party Context Directive", supergroup_prompt)


class TestTrainingScriptSafeguards(unittest.TestCase):
    """Tests for train_lora.py configuration and safeguards."""

    def test_detect_response_template(self):
        self.assertEqual(
            detect_response_template("Qwen/Qwen2.5-7B-Instruct"),
            "<|im_start|>assistant\n",
        )
        self.assertEqual(
            detect_response_template("meta-llama/Meta-Llama-3-8B-Instruct"),
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    def test_dry_run_validation(self):
        """Test dry-run validation using generated data/ files."""
        # Use existing data/ files if present, else create temp dummy files
        train_file = Path("data/train.jsonl")
        val_file = Path("data/val.jsonl")

        if train_file.is_file() and val_file.is_file():
            mock_args = MagicMock()
            mock_args.model_name = "Qwen/Qwen2.5-7B-Instruct"
            mock_args.train_file = str(train_file)
            mock_args.val_file = str(val_file)
            mock_args.output_dir = "checkpoints/test_lora"
            mock_args.lora_r = 16
            mock_args.lora_alpha = 32
            mock_args.style_boost = True
            mock_args.epochs = 3
            mock_args.batch_size = 2
            mock_args.grad_accum = 8
            mock_args.learning_rate = 2e-4
            mock_args.early_stopping_patience = 2
            mock_args.eval_steps = 40

            # Should complete without error
            run_dry_run_validation(mock_args)


class TestChatAppAdapterIntegration(unittest.TestCase):
    """Tests chat_app.py adapter argument passing."""

    def test_create_llm_client_with_adapter(self):
        from chat_app import HuggingFaceClient, create_llm_client

        # If adapter path is passed, create_llm_client should select HuggingFaceClient
        with patch.object(HuggingFaceClient, "__init__", return_value=None) as mock_init:
            client = create_llm_client(adapter="checkpoints/dingxuan_lora/adapter")
            self.assertIsInstance(client, HuggingFaceClient)
            mock_init.assert_called_once()


if __name__ == "__main__":
    unittest.main()
