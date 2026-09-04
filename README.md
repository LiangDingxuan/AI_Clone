# AI Clone: Telegram Chat History Preprocessing & Sessionization Pipeline

Data cleaning, sorting, and conversation sessionization pipeline designed to process Telegram chat history exports to train an AI chatbot clone mimicking the personality and conversation style of **Liang Dingxuan** (`@LiangDingxuan` / ID: `5711494385`).

---

## Table of Contents
1. [Overview & Results](#overview--results)
2. [Quick Start (All-in-One Runner)](#quick-start-all-in-one-runner)
3. [End-to-End Sorting & Sessionization Steps](#end-to-end-sorting--sessionization-steps)
   - [Step 1: Raw Chat Sorting & Filtering](#step-1-raw-chat-sorting--filtering)
   - [Step 2: Approach 1 — Rule-Based (1-on-1 Personal Chats)](#step-2-approach-1--rule-based-1-on-1-personal-chats)
   - [Step 3: Approach 2 — NLP Embeddings (Supergroups / Long-Form)](#step-3-approach-2--nlp-embeddings-supergroups--long-form)
   - [Step 4: Approach 4 — AI/LLM Contextual Disentanglement (Group Chats)](#step-4-approach-3--aillm-contextual-disentanglement-group-chats)
4. [Project Structure](#project-structure)
5. [Data Models & Output Format](#data-models--output-format)
6. [Running Individual Approaches](#running-individual-approaches)
7. [Running Tests](#running-tests)

---

## Overview & Results

Raw Telegram export files (`telegramChatHistory.json`) contain continuous multi-year message streams where messages from different people and topics interleave. This pipeline transforms raw data into discrete, topic-coherent conversational episodes ready for AI fine-tuning or few-shot training.

### Pipeline Dataset Summary

| Stage / Approach | Target Data Type | Method | Conversations | Dingxuan Turns | Total Turns | Output File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 1: Chat Sorter** | All Chats | Sort by Type & Prune 0-message chats | 71 active chats | 9,716 msgs | 146,591 msgs | `sorted_chats_by_type.json` (52 MB)<br>`chats_summary.json` (19 KB) |
| **Step 2: Approach 1** | `personal_chat` (DMs) | Idle Gap (3h) + Burst Merge (90s) + Reply Graph | **909** | **3,478** | 7,621 | `approach_1_rule_based/output/sessions_personal_chat.json` (4.79 MB) |
| **Step 3: Approach 2** | `private_supergroup` | SentenceTransformer + Cosine Similarity (<0.3) | **190** | **201** | 499 | `approach_2_nlp_embeddings/output/sessions_supergroup.json` (0.39 MB) |
| **Step 4: Approach 3** | `private_group` | Gemini 2.5 Flash LLM / Fallback Chunking | **641** | **2,277** | 18,272 | `approach_3_ai_llm/output/sessions_group.json` (10.64 MB) |
| **TOTAL** | | | **1,740** | **5,956** | **26,392** | **15.82 MB Clean Training Data** |

---

## Quick Start (All-in-One Runner)

Execute the full end-to-end pipeline from `telegramChatHistory.json` through all three sessionization approaches with a single command:

```bash
# Run full pipeline (offline / dry-run mode for Approach 3):
python run_all.py --dry-run-llm

# Run full pipeline with live Gemini API for Approach 3:
set GEMINI_API_KEY=your_api_key_here
python run_all.py
```

### Useful CLI Flags for `run_all.py`:
- `--skip-step1`: Skip re-sorting raw chats if `sorted_chats_by_type.json` already exists.
- `--skip-approach1`: Skip personal chat sessionizer.
- `--skip-approach2`: Skip supergroup embedding sessionizer.
- `--skip-approach3`: Skip group chat LLM sessionizer.
- `--idle-gap-dm <hours>`: Customize DM idle gap (default: `3.0`).
- `--similarity-threshold <float>`: Customize embedding topic shift threshold (default: `0.3`).

---

## End-to-End Sorting & Sessionization Steps

### Step 1: Raw Chat Sorting & Filtering

**Script**: `chat_sorter.py` / `index.py`  
**Input**: `telegramChatHistory.json` (~76 MB raw Telegram desktop export)  
**Outputs**: `sorted_chats_by_type.json` (52 MB), `chats_summary.json` (19 KB)

#### What this step does:
1. **Auto-Repair JSON**: Automatically detects and fixes syntax anomalies (e.g., dangling closing braces ` },` from manual edits).
2. **Target Persona Detection**: Automatically locates Liang Dingxuan (`user5711494385` / `@LiangDingxuan`) and flags all messages authored by him (`is_target_user: true`).
3. **Text Normalization**: Flattens nested Telegram formatted entity lists (e.g. `['Hello ', {'type': 'bold', 'text': 'world'}]`) into clean plain text strings.
4. **Pruning Empty Chats**: Automatically excludes any chat where `total_messages <= 0`. Pruned 141 empty/inactive chats, leaving 71 active chats.
5. **Categorization by Type**: Groups chats into conversational hierarchy:
   - `personal_chat` (43 active chats — direct 1-on-1 messages)
   - `private_group` (16 active chats — small friend/project groups)
   - `private_supergroup` (9 active chats — large community supergroups)
   - `saved_messages` (1 chat — personal notes & links)
   - `private_channel` (2 active channels — announcements)
6. **Activity Sorting**: Within each category, chats are sorted in descending order of Liang Dingxuan's message volume, placing the richest conversations at the top.

```bash
python index.py
```

---

### Step 2: Approach 1 — Rule-Based (1-on-1 Personal Chats)

**Folder**: `approach_1_rule_based/`  
**Target**: `personal_chat` (43 chats, 13,620 messages)  
**Output**: `approach_1_rule_based/output/sessions_personal_chat.json` (909 conversations)

#### Method:
- **Burst Merge (`BURST_WINDOW = 90s`)**: Multiple consecutive short messages sent by the same user within 90 seconds are concatenated into a single coherent `Turn`.
- **Idle-Gap Splitting (`IDLE_GAP = 3 hours`)**: Silence exceeding 3 hours indicates conversation conclusion and topic reset.
- **Reply Chain Union-Find**: Traces `reply_to_message_id` references. If a user replies to a message from an earlier session within 12 hours, a Disjoint-Set Union (Union-Find) algorithm links the reply back to the originating discussion.
- **Quality Filter**: Discards monologues (< 2 turns) and sessions without Liang Dingxuan.

```bash
python approach_1_rule_based/run.py --idle-gap 3.0 --burst-window 90
```

---

### Step 3: Approach 2 — NLP Embeddings (Supergroups / Long-Form)

**Folder**: `approach_2_nlp_embeddings/`  
**Target**: `private_supergroup` (9 chats, 48,204 messages)  
**Output**: `approach_2_nlp_embeddings/output/sessions_supergroup.json` (190 conversations)

#### Method:
- **Pre-split by Idle Gap (`IDLE_GAP = 4 hours`)**: Coarse temporal chunking.
- **Burst Merge (`BURST_WINDOW = 120s`)**: Merges consecutive turns.
- **SentenceTransformer Embeddings**: Encodes each turn into a 384-dimensional semantic dense vector using `all-MiniLM-L6-v2`.
- **Cosine Similarity Topic Shift**: Computes pairwise cosine similarity between consecutive turns. A drop below `0.30` triggers a topic split.
- **Quality Filter**: Keeps conversations with $\ge 2$ turns and Liang Dingxuan participation.

```bash
# Install dependencies (first time only):
pip install -r approach_2_nlp_embeddings/requirements.txt

# Run:
python approach_2_nlp_embeddings/run.py --similarity-threshold 0.3
```

---

### Step 4: Approach 3 — AI/LLM Contextual Disentanglement (Group Chats)

**Folder**: `approach_3_ai_llm/`  
**Target**: `private_group` (16 chats, 77,858 messages)  
**Output**: `approach_3_ai_llm/output/sessions_group.json` (641 conversations)

#### Method:
- **Temporal Chunking (`IDLE_GAP = 2 hours`, `CHUNK_SIZE = 50 turns`)**: Pre-splits group chat streams into manageable conversational windows.
- **LLM Thread Disentanglement Prompt**: Sends batches of turns to Gemini (`gemini-2.5-flash`) with structured JSON instructions to cluster interleaved turns into separate `thread_id`s.
- **Offline / Dry-Run Fallback**: If no API key is provided or `--dry-run` is specified, it gracefully uses rule-based temporal chunking with zero API dependencies.
- **Quality Filter**: Requires $\ge 2$ turns and Liang Dingxuan participation.

```bash
# Offline fallback mode:
python approach_3_ai_llm/run.py --dry-run

# Live Gemini LLM mode:
set GEMINI_API_KEY=your_key_here
python approach_3_ai_llm/run.py
```

---

## Project Structure

```
d:\LocalUser\AI_Clone\
├── README.md                                   # Documentation (this file)
├── run_all.py                                  # Master pipeline runner
├── index.py                                    # CLI for chat sorting
├── chat_sorter.py                              # Core chat cleaner and sorter
├── chats_summary.json                          # Lightweight index of sorted chats (19 KB)
├── sorted_chats_by_type.json                   # Full cleaned & sorted Telegram export (52 MB)
│
├── shared/                                     # Shared utilities
│   ├── __init__.py
│   ├── data_loader.py                          # Safe loading, type filtering, timestamp casting
│   └── models.py                               # Turn, Conversation dataclasses & JSON serialization
│
├── approach_1_rule_based/                      # Approach 1: DMs
│   ├── sessionizer.py
│   ├── run.py
│   └── output/
│       └── sessions_personal_chat.json         # 909 conversations (4.79 MB)
│
├── approach_2_nlp_embeddings/                  # Approach 2: Supergroups
│   ├── requirements.txt
│   ├── sessionizer.py
│   ├── run.py
│   └── output/
│       └── sessions_supergroup.json            # 190 conversations (0.39 MB)
│
├── approach_3_ai_llm/                          # Approach 3: Group Chats
│   ├── requirements.txt
│   ├── sessionizer.py
│   ├── run.py
│   └── output/
│       └── sessions_group.json                 # 641 conversations (10.64 MB)
│
└── tests/                                      # Automated unit tests
    ├── test_chat_sorter.py
    └── test_sessionizers.py
```

---

## Data Models & Output Format

All three approaches export conversations adhering to the standardized `Conversation` and `Turn` schemas:

```json
{
  "conversation_id": "242405345_1",
  "chat_id": 242405345,
  "chat_name": "Kelvin",
  "chat_type": "personal_chat",
  "approach": "rule_based",
  "participants": ["Dingxuan Liang", "Kelvin"],
  "start_time": "2024-01-09T18:47:24",
  "end_time": "2024-01-09T19:17:01",
  "turn_count": 9,
  "target_user_turn_count": 4,
  "has_target_user_participation": true,
  "turns": [
    {
      "sender_name": "Dingxuan Liang",
      "sender_id": "user5711494385",
      "is_target_user": true,
      "text": "Which do you prefer that is not student school, COVID 19 and inflation",
      "start_time": "2024-01-09T18:47:24",
      "end_time": "2024-01-09T18:47:24",
      "start_unixtime": 1704797244,
      "end_unixtime": 1704797244,
      "message_ids": [48777],
      "message_count": 1,
      "has_media": true,
      "has_reply": false,
      "reply_to_message_id": null
    },
    {
      "sender_name": "Kelvin",
      "sender_id": "user242405345",
      "is_target_user": false,
      "text": "Do u prefer “Gamer crazy”? Since we can create a honkai star rail quiz website with your knowledge of the game",
      "start_time": "2024-01-09T18:50:46",
      "end_time": "2024-01-09T18:50:46",
      "start_unixtime": 1704797446,
      "end_unixtime": 1704797446,
      "message_ids": [48778],
      "message_count": 1,
      "has_media": false,
      "has_reply": true,
      "reply_to_message_id": 48777
    }
  ]
}
```

---

## Running Tests

Run the test suite using Python's built-in `unittest` runner (zero external dependencies):

```bash
python -m unittest discover tests
```

**Output**:
```
Ran 10 tests in 0.011s
OK
```
Tests cover:
- Plain text extraction and entity normalization
- Target user identification and message attribution
- Zero-message chat pruning and chat type categorization
- Burst merging and conversation finalization
- Approach 1 idle-gap splitting and quality filtering
- Approach 3 dry-run fallback execution
