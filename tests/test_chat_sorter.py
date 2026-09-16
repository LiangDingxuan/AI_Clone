"""
Unit tests for chat_sorter module using standard unittest.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANING_DIR = PROJECT_ROOT / "data_cleaning"
for p in [str(PROJECT_ROOT), str(CLEANING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from data_cleaning.chat_sorter import (
        clean_message,
        identify_target_user,
        is_target_user_message,
        load_telegram_data,
        normalize_text,
        process_chat,
        sort_chats_by_type,
    )
except ImportError:
    from chat_sorter import (
        clean_message,
        identify_target_user,
        is_target_user_message,
        load_telegram_data,
        normalize_text,
        process_chat,
        sort_chats_by_type,
    )


class TestChatSorter(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("Hello world"), "Hello world")
        self.assertEqual(
            normalize_text(["Part 1", " ", {"type": "bold", "text": "Part 2"}]),
            "Part 1 Part 2",
        )
        self.assertEqual(normalize_text([]), "")
        self.assertEqual(normalize_text(None), "")

    def test_identify_target_user(self):
        sample_data = {
            "personal_information": {
                "user_id": 5711494385,
                "first_name": "Dingxuan",
                "last_name": "Liang",
                "username": "@LiangDingxuan",
            }
        }
        target = identify_target_user(sample_data)
        self.assertEqual(target["user_id"], 5711494385)
        self.assertEqual(target["from_id_str"], "user5711494385")
        self.assertEqual(target["full_name"], "Dingxuan Liang")
        self.assertEqual(target["username"], "@LiangDingxuan")

    def test_is_target_user_message(self):
        target = {
            "user_id": 5711494385,
            "from_id_str": "user5711494385",
            "full_name": "Dingxuan Liang",
            "username": "@LiangDingxuan",
        }
        # Match by from_id
        self.assertTrue(is_target_user_message({"from_id": "user5711494385", "from": "Someone"}, target))
        # Match by full name
        self.assertTrue(is_target_user_message({"from_id": "user9999", "from": "Dingxuan Liang"}, target))
        # Non-match
        self.assertFalse(is_target_user_message({"from_id": "user1234", "from": "Alice"}, target))

    def test_sort_chats_by_type_logic(self):
        mock_data = {
            "personal_information": {
                "user_id": 5711494385,
                "first_name": "Dingxuan",
                "last_name": "Liang",
            },
            "chats": {
                "list": [
                    {
                        "id": 1,
                        "name": "Friend A",
                        "type": "personal_chat",
                        "messages": [
                            {"id": 101, "from_id": "user5711494385", "from": "Dingxuan Liang", "text": "Hey"},
                            {"id": 102, "from_id": "user999", "from": "Friend A", "text": "Hello"},
                        ],
                    },
                    {
                        "id": 2,
                        "name": "Friend B",
                        "type": "personal_chat",
                        "messages": [
                            {"id": 201, "from_id": "user5711494385", "from": "Dingxuan Liang", "text": "Yo"},
                            {"id": 202, "from_id": "user5711494385", "from": "Dingxuan Liang", "text": "What's up"},
                            {"id": 203, "from_id": "user888", "from": "Friend B", "text": "Chilling"},
                        ],
                    },
                    {
                        "id": 3,
                        "name": "Gaming Group",
                        "type": "private_group",
                        "messages": [
                            {"id": 301, "from_id": "user111", "from": "Gamer 1", "text": "Game night?"},
                        ],
                    },
                    {
                        "id": 4,
                        "name": "Empty Chat",
                        "type": "public_channel",
                        "messages": [],
                    },
                ]
            },
        }

        metadata, sorted_chats = sort_chats_by_type(mock_data)

        # Total chats should be 3 (Empty Chat should be excluded!)
        self.assertEqual(metadata["total_chats"], 3)
        self.assertEqual(metadata["total_messages"], 6)
        self.assertEqual(metadata["total_target_user_messages"], 3)

        self.assertIn("personal_chat", sorted_chats)
        self.assertIn("private_group", sorted_chats)
        # Empty category public_channel should not even be in sorted_chats
        self.assertNotIn("public_channel", sorted_chats)

        # Friend B has 2 Dingxuan messages, Friend A has 1.
        personal_chats = sorted_chats["personal_chat"]
        self.assertEqual(len(personal_chats), 2)
        self.assertEqual(personal_chats[0]["name"], "Friend B")
        self.assertEqual(personal_chats[0]["target_user_messages"], 2)
        self.assertEqual(personal_chats[1]["name"], "Friend A")
        self.assertEqual(personal_chats[1]["target_user_messages"], 1)

    def test_load_telegram_data_auto_repair(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            broken_json = """{
  "personal_information": {"user_id": 5711494385, "first_name": "Dingxuan", "last_name": "Liang"},
  "drafts": []
  },
 "chats": {
  "list": []
 }
}"""
            file_p = Path(tmp_dir) / "broken.json"
            file_p.write_text(broken_json, encoding="utf-8")

            data = load_telegram_data(file_p)
            self.assertIn("chats", data)
            self.assertEqual(data["personal_information"]["first_name"], "Dingxuan")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

