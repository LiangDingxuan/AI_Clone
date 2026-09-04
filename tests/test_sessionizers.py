"""
Unit tests for all three sessionizer approaches and shared utilities.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.models import Turn, Conversation, merge_into_turns
from shared.data_loader import prepare_messages
from approach_1_rule_based.sessionizer import sessionize_personal_chats
from approach_3_ai_llm.sessionizer import sessionize_with_llm


class TestSharedModels(unittest.TestCase):
    def test_merge_into_turns(self):
        # 3 messages: first 2 from same user within burst window
        messages = [
            {"id": 1, "from": "Dingxuan Liang", "from_id": "user5711494385", "is_target_user": True, "text": "hey", "date_unixtime": 1000, "date": "2024-01-01T10:00:00"},
            {"id": 2, "from": "Dingxuan Liang", "from_id": "user5711494385", "is_target_user": True, "text": "you free?", "date_unixtime": 1020, "date": "2024-01-01T10:00:20"},
            {"id": 3, "from": "Kelvin", "from_id": "user242405345", "is_target_user": False, "text": "yeah", "date_unixtime": 1050, "date": "2024-01-01T10:00:50"},
        ]
        turns = merge_into_turns(messages, burst_window_seconds=90)
        self.assertEqual(len(turns), 2)
        # First turn combines msg 1 & 2
        self.assertEqual(turns[0].sender_id, "user5711494385")
        self.assertEqual(turns[0].text, "hey\nyou free?")
        self.assertEqual(turns[0].message_count, 2)
        self.assertTrue(turns[0].is_target_user)
        # Second turn is Kelvin
        self.assertEqual(turns[1].sender_id, "user242405345")
        self.assertEqual(turns[1].text, "yeah")
        self.assertFalse(turns[1].is_target_user)

    def test_conversation_finalize(self):
        turns = [
            Turn(sender_name="Alice", sender_id="user1", is_target_user=False, text="Hi", start_time="2024-01-01T10:00:00", end_time="2024-01-01T10:00:00", start_unixtime=1000, end_unixtime=1000),
            Turn(sender_name="Dingxuan Liang", sender_id="user5711494385", is_target_user=True, text="Hello", start_time="2024-01-01T10:01:00", end_time="2024-01-01T10:01:00", start_unixtime=1060, end_unixtime=1060),
        ]
        conv = Conversation(
            conversation_id="test_001",
            chat_id=123,
            chat_name="Test Chat",
            chat_type="personal_chat",
            approach="rule_based",
            turns=turns,
        )
        conv.finalize()
        self.assertEqual(conv.turn_count, 2)
        self.assertEqual(conv.target_user_turn_count, 1)
        self.assertTrue(conv.has_target_user_participation)
        self.assertIn("Alice", conv.participants)
        self.assertIn("Dingxuan Liang", conv.participants)


class TestApproach1RuleBased(unittest.TestCase):
    def test_sessionize_personal_chats_idle_gap(self):
        # 2 conversations separated by 5 hours (> 3h idle gap)
        chat = {
            "id": 100,
            "name": "Friend A",
            "type": "personal_chat",
            "messages": [
                # Conv 1: at t = 10000
                {"id": 1, "from": "Friend A", "from_id": "user111", "is_target_user": False, "text": "hello", "date_unixtime": "10000", "date": "2024-01-01T10:00:00"},
                {"id": 2, "from": "Dingxuan Liang", "from_id": "user5711494385", "is_target_user": True, "text": "hey there", "date_unixtime": "10060", "date": "2024-01-01T10:01:00"},
                # Conv 2: at t = 50000 (40000s later > 10800s idle gap)
                {"id": 3, "from": "Friend A", "from_id": "user111", "is_target_user": False, "text": "dinner?", "date_unixtime": "50000", "date": "2024-01-01T21:00:00"},
                {"id": 4, "from": "Dingxuan Liang", "from_id": "user5711494385", "is_target_user": True, "text": "sure let's go", "date_unixtime": "50120", "date": "2024-01-01T21:02:00"},
            ]
        }
        convs = sessionize_personal_chats([chat], idle_gap_seconds=10800, burst_window_seconds=90)
        self.assertEqual(len(convs), 2)
        self.assertEqual(convs[0].turn_count, 2)
        self.assertEqual(convs[1].turn_count, 2)

    def test_filter_no_target_user(self):
        # Conversation where target user never participates should be filtered out
        chat = {
            "id": 200,
            "name": "Other",
            "type": "personal_chat",
            "messages": [
                {"id": 1, "from": "Alice", "from_id": "user1", "is_target_user": False, "text": "msg1", "date_unixtime": "1000", "date": "2024-01-01T10:00:00"},
                {"id": 2, "from": "Bob", "from_id": "user2", "is_target_user": False, "text": "msg2", "date_unixtime": "1060", "date": "2024-01-01T10:01:00"},
            ]
        }
        convs = sessionize_personal_chats([chat], require_target_user=True)
        self.assertEqual(len(convs), 0)


class TestApproach3AI_LLM(unittest.TestCase):
    def test_dry_run_fallback(self):
        # Tests that dry-run mode runs cleanly without API key
        chat = {
            "id": 300,
            "name": "Project Group",
            "type": "private_group",
            "messages": [
                {"id": 1, "from": "Colleague", "from_id": "user99", "is_target_user": False, "text": "Can we sync?", "date_unixtime": "1000", "date": "2024-01-01T10:00:00"},
                {"id": 2, "from": "Dingxuan Liang", "from_id": "user5711494385", "is_target_user": True, "text": "Yeah 2pm works", "date_unixtime": "1050", "date": "2024-01-01T10:00:50"},
            ]
        }
        convs = sessionize_with_llm([chat], dry_run=True, require_target_user=True)
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0].chat_id, 300)
        self.assertEqual(convs[0].approach, "ai_llm")
        self.assertTrue(convs[0].has_target_user_participation)


if __name__ == "__main__":
    unittest.main()

