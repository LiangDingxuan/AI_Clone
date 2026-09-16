# AI Clone: Telegram Chat History Preprocessing & Sessionization Pipeline

Data cleaning, sorting, and conversation sessionization pipeline designed to process Telegram chat history exports to train an AI chatbot clone mimicking the personality and conversation style of **Liang Dingxuan** (`@LiangDingxuan` / ID: `5711494385`).

> [!TIP]
> **Complete Operational Runbook Available**: Check [**`FULL_PIPELINE_GUIDE.md`**](FULL_PIPELINE_GUIDE.md) for the exhaustive guide covering fresh device GPU setup, updating `telegramChatHistory.json`, sorting, running all 3 approaches, dataset export, and fine-tuning.

---

## Table of Contents
0. [**Complete Pipeline Runbook (`FULL_PIPELINE_GUIDE.md`)**](FULL_PIPELINE_GUIDE.md)
1. [Overview & Results](#overview--results)
2. [Quick Start (All-in-One Runner)](#quick-start-all-in-one-runner)
3. [End-to-End Sorting & Sessionization Steps](#end-to-end-sorting--sessionization-steps)
   - [Step 1: Raw Chat Sorting & Filtering](#step-1-raw-chat-sorting--filtering)
   - [Step 2: Approach 1 — Rule-Based (1-on-1 Personal Chats)](#step-2-approach-1--rule-based-1-on-1-personal-chats)
   - [Step 3: Approach 2 — NLP Embeddings (Supergroups / Long-Form)](#step-3-approach-2--nlp-embeddings-supergroups--long-form)
   - [Step 4: Approach 3 — AI/LLM Contextual Disentanglement (Group Chats)](#step-4-approach-3--aillm-contextual-disentanglement-group-chats)
4. [Project Structure](#project-structure)
5. [Data Models & Output Format](#data-models--output-format)
6. [Conversational SFT Dataset Compilation](#conversational-sft-dataset-compilation)
7. [Multi-Context Personality Profiler & Prompt Synthesizer](#multi-context-personality-profiler--prompt-synthesizer)
8. [Interactive Chat Application (`chat_app.py`)](#interactive-chat-application-chat_apppy)
9. [Persona Model Fine-Tuning (QLoRA)](#persona-model-fine-tuning-qlora)
10. [Running Automated Tests](#running-automated-tests)

---

## Overview & Results

Raw Telegram export files (`telegramChatHistory.json`) contain continuous multi-year message streams where messages from different people and topics interleave. This pipeline transforms raw data into discrete, topic-coherent conversational episodes ready for AI fine-tuning or few-shot training.

### Pipeline Dataset Summary

| Stage / Component | Target Data Type | Method | Scope / Size | Dingxuan Turns | Output File / Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 1: Chat Sorter** | All Chats | Sort by Type & Prune 0-message chats | 71 active chats | 9,716 msgs | `data_cleaning/sorted_chats_by_type.json` (52 MB)<br>`data_cleaning/chats_summary.json` (19 KB) |
| **Step 2: Approach 1** | `personal_chat` (DMs) | Idle Gap (3h) + Burst Merge (90s) + Reply Graph | **909** conversations | **3,478** turns | `data_cleaning/approach_1_rule_based/output/sessions_personal_chat.json` (4.79 MB) |
| **Step 3: Approach 2** | `private_supergroup` | SentenceTransformer + Cosine Similarity (<0.3) | **190** conversations | **201** turns | `data_cleaning/approach_2_nlp_embeddings/output/sessions_supergroup.json` (0.39 MB) |
| **Step 4: Approach 3** | `private_group` | Gemini 2.5 Flash LLM / Fallback Chunking | **641** conversations | **2,277** turns | `data_cleaning/approach_3_ai_llm/output/sessions_group.json` (10.64 MB) |
| **Step 5: Profiler** | All Cleaned Sessions | Empirical Lexical Mining & Big-5 Trait Scoring | 1,740 episodes | 5,956 turns | `llm/profiles_summary.json` (12 KB) |
| **Step 6: SFT Exporter** | Training JSONL | Distractor Filter, Multi-Party Tags, Strict Alternation | **1,437** SFT episodes | **4,967** assistant turns | `llm/data/train.jsonl` (7.8 MB)<br>`llm/data/val.jsonl` (1.35 MB) |
| **Step 7: QLoRA Trainer** | Open-Source LLMs | Heretic Abliteration, 4-bit NF4, Style-Boost (α=64), Early Stopping | Qwen-2.5 (Abliterated) / LLaMA-3 | Full SFT Adapter | `checkpoints/dingxuan_lora/adapter` |

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

**Script**: `data_cleaning/chat_sorter.py` / `data_cleaning/index.py` (or root `index.py`)  
**Input**: `telegramChatHistory.json` (~76 MB raw Telegram desktop export)  
**Outputs**: `data_cleaning/sorted_chats_by_type.json` (52 MB), `data_cleaning/chats_summary.json` (19 KB)

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
python data_cleaning/index.py
# (or python index.py)
```

---

### Step 2: Approach 1 — Rule-Based (1-on-1 Personal Chats)

**Folder**: `data_cleaning/approach_1_rule_based/`  
**Target**: `personal_chat` (43 chats, 13,620 messages)  
**Output**: `data_cleaning/approach_1_rule_based/output/sessions_personal_chat.json` (909 conversations)

#### Method:
- **Burst Merge (`BURST_WINDOW = 90s`)**: Multiple consecutive short messages sent by the same user within 90 seconds are concatenated into a single coherent `Turn`.
- **Idle-Gap Splitting (`IDLE_GAP = 3 hours`)**: Silence exceeding 3 hours indicates conversation conclusion and topic reset.
- **Reply Chain Union-Find**: Traces `reply_to_message_id` references. If a user replies to a message from an earlier session within 12 hours, a Disjoint-Set Union (Union-Find) algorithm links the reply back to the originating discussion.
- **Quality Filter**: Discards monologues (< 2 turns) and sessions without Liang Dingxuan.

```bash
python data_cleaning/approach_1_rule_based/run.py --idle-gap 3.0 --burst-window 90
```

---

### Step 3: Approach 2 — NLP Embeddings (Supergroups / Long-Form)

**Folder**: `data_cleaning/approach_2_nlp_embeddings/`  
**Target**: `private_supergroup` (9 chats, 48,204 messages)  
**Output**: `data_cleaning/approach_2_nlp_embeddings/output/sessions_supergroup.json` (190 conversations)

#### Method:
- **Pre-split by Idle Gap (`IDLE_GAP = 4 hours`)**: Coarse temporal chunking.
- **Burst Merge (`BURST_WINDOW = 120s`)**: Merges consecutive turns.
- **SentenceTransformer Embeddings**: Encodes each turn into a 384-dimensional semantic dense vector using `all-MiniLM-L6-v2`.
- **Cosine Similarity Topic Shift**: Computes pairwise cosine similarity between consecutive turns. A drop below `0.30` triggers a topic split.
- **Quality Filter**: Keeps conversations with $\ge 2$ turns and Liang Dingxuan participation.

```bash
# Install dependencies (first time only):
pip install -r data_cleaning/approach_2_nlp_embeddings/requirements.txt

# Run:
python data_cleaning/approach_2_nlp_embeddings/run.py --similarity-threshold 0.3
```

---

### Step 4: Approach 3 — AI/LLM Contextual Disentanglement (Group Chats)

**Folder**: `data_cleaning/approach_3_ai_llm/`  
**Target**: `private_group` (16 chats, 77,858 messages)  
**Output**: `data_cleaning/approach_3_ai_llm/output/sessions_group.json` (641 conversations)

#### Method:
- **Temporal Chunking (`IDLE_GAP = 2 hours`, `CHUNK_SIZE = 50 turns`)**: Pre-splits group chat streams into manageable conversational windows.
- **LLM Thread Disentanglement Prompt**: Sends batches of turns to Gemini (`gemini-2.5-flash`) with structured JSON instructions to cluster interleaved turns into separate `thread_id`s.
- **Offline / Dry-Run Fallback**: If no API key is provided or `--dry-run` is specified, it gracefully uses rule-based temporal chunking with zero API dependencies.
- **Quality Filter**: Requires $\ge 2$ turns and Liang Dingxuan participation.

```bash
# Offline fallback mode:
python data_cleaning/approach_3_ai_llm/run.py --dry-run

# Live Gemini LLM mode:
set GEMINI_API_KEY=your_key_here
python data_cleaning/approach_3_ai_llm/run.py
```

---

## Project Structure

```
d:\LocalUser\AI_Clone\
├── README.md                                   # Documentation (this file)
├── FULL_PIPELINE_GUIDE.md                      # Exhaustive end-to-end operational runbook
├── run_all.py                                  # Master pipeline runner (orchestrates both modules)
├── index.py                                    # Root CLI convenience forwarder -> data_cleaning/index.py
├── chat_app.py                                 # Root CLI convenience forwarder -> llm/chat_app.py
├── telegramChatHistory.json                    # Raw Telegram desktop export (~76 MB)
│
├── data_cleaning/                              # DATA CLEANING & SESSIONIZATION MODULE
│   ├── __init__.py                             # Package exports
│   ├── chat_sorter.py                          # Core chat cleaner and sorter
│   ├── index.py                                # CLI entrypoint for chat sorting
│   ├── run_cleaning.py                         # Pipeline runner for Steps 1-4
│   ├── chats_summary.json                      # Lightweight index of sorted chats (19 KB)
│   ├── sorted_chats_by_type.json               # Full cleaned & sorted Telegram export (52 MB)
│   ├── shared/                                 # Shared data loaders & models
│   │   ├── __init__.py
│   │   ├── data_loader.py                      # Safe loading, type filtering, timestamp casting
│   │   └── models.py                           # Turn, Conversation dataclasses & serialization
│   ├── approach_1_rule_based/                  # Approach 1: DMs (idle gap + burst merge)
│   │   ├── sessionizer.py
│   │   ├── run.py
│   │   └── output/sessions_personal_chat.json  # 909 conversations (4.79 MB)
│   ├── approach_2_nlp_embeddings/              # Approach 2: Supergroups (SentenceTransformers)
│   │   ├── requirements.txt
│   │   ├── sessionizer.py
│   │   ├── run.py
│   │   └── output/sessions_supergroup.json     # 190 conversations (0.39 MB)
│   └── approach_3_ai_llm/                      # Approach 3: Group Chats (Gemini disentanglement)
│       ├── requirements.txt
│       ├── sessionizer.py
│       ├── run.py
│       └── output/sessions_group.json          # 641 conversations (10.64 MB)
│
├── llm/                                        # LLM MODELING, SFT, INFERENCE & FINE-TUNING
│   ├── __init__.py                             # Package exports
│   ├── profiler.py                             # Multi-context linguistic & Big-5 personality profiler
│   ├── profiles_summary.json                   # Extracted personality metrics & few-shot exemplars
│   ├── synthesizer.py                          # Context-adaptive system prompt synthesizer
│   ├── export_training_data.py                 # Compiles session conversations into SFT JSONL format
│   ├── chat_app.py                             # Interactive terminal chat sandbox (Simulated/HF/OpenAI)
│   ├── data/                                   # SFT Conversational Training Datasets
│   │   ├── train.jsonl                         # 1,223 training episodes (7.8 MB)
│   │   ├── val.jsonl                           # 214 validation episodes (1.35 MB)
│   │   └── training_data_summary.json          # Dataset statistics & token breakdown
│   └── training/                               # Standalone GPU QLoRA Fine-Tuning Package
│       ├── README.md                           # GPU / Google Colab / Cloud training instructions
│       ├── requirements.txt                    # PyTorch, PEFT, TRL, BitsAndBytes dependencies
│       ├── train_lora.py                       # 4-bit QLoRA trainer with Doppelganger safeguards
│       ├── run_training.sh                     # Turnkey bash training runner (Linux / Cloud GPU)
│       └── run_training.bat                    # Turnkey batch training runner (Windows GPU)
│
└── tests/                                      # Automated unit & integration tests (31 tests passing)
    ├── test_chat_sorter.py
    ├── test_sessionizers.py
    ├── test_personality_bot.py
    └── test_training_pipeline.py
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

## Conversational SFT Dataset Compilation

## Conversational SFT Dataset Compilation

**Script**: `llm/export_training_data.py`  
**Inputs**: Session files from `data_cleaning/approach_*/`  
**Outputs**: `llm/data/train.jsonl` (7.8 MB), `llm/data/val.jsonl` (1.35 MB), `llm/data/training_data_summary.json`

Transforms the 1,740 sessionized conversations into standardized multi-turn ChatML/messages JSONL datasets for QLoRA fine-tuning.

### What this step does:
1. **Third-Party Distractor Filtering**: Strips bot commands (`/start`, `/help`), system events (join/leave, pinned messages), standalone URLs, and `<media omitted>` tags.
2. **Multi-Party Context Formatting**: For group and supergroup chats, merges multiple other-user turns into a single user turn with bracketed sender tags: `[Edric]: let's meet\n[Jason]: what time?`.
3. **Strict Turn Alternation**: Guarantees a strictly alternating `system` $\rightarrow$ `user` $\rightarrow$ `assistant` sequence. Automatically prunes leading assistant turns and trailing unreplied user turns so every episode concludes on an assistant (Dingxuan) turn.
4. **Context-Adaptive System Prompt Injection**: Automatically prepends the exact synthesized system prompt generated by `synthesizer.py` (including the multi-party context directive).
5. **Stratified 85/15 Split**: Performs an 85% train / 15% validation split stratified across personal, group, and supergroup episodes.

```bash
python llm/export_training_data.py --output-dir llm/data/
```

- **Dataset Breakdown**:
  - `llm/data/train.jsonl`: 1,223 episodes (4,270 Dingxuan assistant turns)
  - `llm/data/val.jsonl`: 214 episodes (697 Dingxuan assistant turns)
  - **Total**: 1,437 formatted episodes, 4,967 assistant turns

---

## Multi-Context Personality Profiler & Prompt Synthesizer

A style-first, 2-layer profiling architecture for simulating Dingxuan across 3 distinct social environments:

```
┌──────────────────────────────────────┐
│ 1. Multi-Context Profiler & Analyzer │  (llm/profiler.py)
│    (Reads Personal, Group, & Super)  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 2. Context-Adaptive Prompt Generator │  (llm/synthesizer.py)
│    (Extracts Big-5 & Lexics per Mode)│
└──────────────────────────────────────┘
```

### 1. Multi-Context Profiler (`llm/profiler.py`)
Analyzes 1,740 conversational episodes and 5,956 of Dingxuan's turns across:
- **Personal DMs** (`data_cleaning/approach_1_rule_based/output/sessions_personal_chat.json`)
- **Supergroups** (`data_cleaning/approach_2_nlp_embeddings/output/sessions_supergroup.json`)
- **Group Chats** (`data_cleaning/approach_3_ai_llm/output/sessions_group.json`)

Extracts quantitative metrics (turn length, lowercase ratio, punctuation frequency, Singlish particles like `ah`, `eh`, `cuz`, `idk`, `sia`), scores academic Big-5 personality traits (0.0 to 1.0), and mines representative few-shot QA pairs.

```bash
# Run standalone profiling and export JSON summary:
python llm/profiler.py
```

### 2. Context-Adaptive Prompt Synthesizer (`llm/synthesizer.py`)
Dynamically constructs system prompts with absolute formatting constraints, Big-5 behavioral directives, lexicon guidelines, **Multi-Party Context Directives**, in-character refusal deflection instructions, and few-shot pairs.

```bash
# Inspect generated prompt for a specific context:
python llm/synthesizer.py --context dm
python llm/synthesizer.py --context group
python llm/synthesizer.py --context supergroup
```

---

## Interactive Chat Application (`llm/chat_app.py` / `chat_app.py`)

Provides an interactive terminal conversation sandbox:
- Prompts for chat context on start: `[1] Personal Chat (DM)`, `[2] Group Chat`, `[3] Supergroup Chat`.
- Maintains a 5-turn sliding window buffer.
- **Refusal Deflection Layer**: Intercepts biographical and historical memory queries with in-character deflections (e.g. *"idk tbh, can't rly remember rn lol"*, *"whut why u asking that lol"*).
- Abstract LLM support: HuggingFace local pipeline / PEFT LoRA adapter, OpenAI-compatible endpoints, Anthropic Claude, or offline **Simulated Provider** (zero API key / GPU needed).

```bash
# Run interactive chat (Simulated Provider):
python llm/chat_app.py
# (or python chat_app.py)

# Run with trained LoRA clone:
python chat_app.py --provider hf --adapter checkpoints/dingxuan_lora/adapter

# Run with OpenAI API:
set OPENAI_API_KEY=your_key
python chat_app.py --provider openai --model gpt-4o-mini
```

**Commands inside chat**:
- `/context [1|2|3]`: Switch chat context on the fly
- `/stats`: View current linguistic & Big-5 personality metrics
- `/prompt`: View the synthesized system prompt
- `/clear`: Reset conversation history
- `/exit`: Exit chat

---

## Route B: Persona Model Fine-Tuning (QLoRA)

A complete, production-grade 4-bit QLoRA fine-tuning pipeline for open-source LLMs (**Qwen-2.5-7B-Instruct** or **Meta-Llama-3-8B-Instruct**) trained directly on Liang Dingxuan's 1,437 formatted conversational episodes (4,967 assistant turns).

### 1. Training Architecture & Workflow

```
┌─────────────────────────────────┐
│ 1. llm/export_training_data.py  │  (Filters noise, formats multi-party tags [Name]:,
│    (Creates llm/data/*.jsonl)   │   injects context system prompts, 85/15 split)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 2. llm/training/train_lora.py   │  (Heretic abliterated base model, 4-bit QLoRA
│    (Runs on GPU / Colab / Cloud)│   with BitsAndBytes, Style-Boost α=64, early stopping)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 3. llm/chat_app.py --adapter ...│  (Interactive inference with trained clone)
└─────────────────────────────────┘
```

### 2. Export Training Dataset (CPU / Local)

Transform the sessionized Telegram history into standardized ChatML/messages JSONL datasets:

```bash
python llm/export_training_data.py --output-dir llm/data/
```

- **Output files**:
  - `llm/data/train.jsonl` (1,223 episodes, 4,270 assistant turns — 7.8 MB)
  - `llm/data/val.jsonl` (214 episodes, 697 assistant turns — 1.35 MB)
  - `llm/data/training_data_summary.json` (metadata & context breakdown)

### 3. Training on Another Device with a GPU

After pushing to GitHub, clone the repository on any device with an NVIDIA GPU ($\ge 8\text{ GB}$ VRAM, or Google Colab T4/A100):

#### Linux / Cloud Workstation (RunPod, Lambda, Vast.ai):
```bash
git clone https://github.com/LiangDingxuan/AI_Clone.git
cd AI_Clone
bash llm/training/run_training.sh
```

#### Windows Workstation with GPU:
```cmd
git clone https://github.com/LiangDingxuan/AI_Clone.git
cd AI_Clone
llm\training\run_training.bat
```

#### Google Colab:
```python
!git clone https://github.com/LiangDingxuan/AI_Clone.git
%cd AI_Clone
!pip install -r llm/training/requirements.txt
!python llm/training/train_lora.py --abliterated --style-boost --epochs 3 --merge-adapter
```

### 4. Advanced Hyperparameters & Safeguards

- **Heretic Model Abliteration (`--abliterated`)**:
  - Automatically targets **`huihui-ai/Qwen2.5-7B-Instruct-abliterated`** (abliterated using **Heretic** via Bayesian directional ablation).
  - Removes corporate refusal directions (*"As an AI language model..."*) caused by corporate RLHF safety alignment, preventing the clone from breaking character during banter or colloquial discussions.
  - Legitimate privacy defenses (IC numbers, home address, passwords) remain strictly protected and are handled in-character by `RefusalDeflectionLayer` in `chat_app.py` (*"whut why u asking that lol"*).
- **LoRA Scaling Factor ($\alpha$) Calibration**:
  - `--style-boost`: Automatically sets $r=16, \alpha=64$ ($\alpha/r = 4.0$) and $lr=1.5\times 10^{-4}$ to amplify Dingxuan's colloquial quirks (`ah`, `sia`, `cuz`, `idk`, `yea`) without sounding robotic.
- **Doppelganger Drift Safeguards**:
  - **Completion-Only Validation Loss**: Loss is computed strictly on assistant turns for both train and validation splits (user questions are never penalized).
  - **Early Stopping**: Halts training if validation loss does not improve for 2 evaluations (`--early-stopping-patience 2`), restoring the best checkpoint to prevent style overfitting.
- **Third-Party Distractor Protection**:
  - In group chats, other users are formatted as `[Sender Name]: text`.
  - The system prompt includes an explicit **Multi-Party Context Directive** instructing the model to treat bracketed names as room background context, not statements made by the direct prompter.

### 5. Chatting with Your Trained Clone

Launch the interactive chat interface with your trained LoRA adapter:

```bash
# Using LoRA adapter checkpoint:
python chat_app.py --provider hf --adapter checkpoints/dingxuan_lora/adapter

# Using merged standalone model:
python chat_app.py --provider hf --model checkpoints/dingxuan_lora/merged_model
```

---

## Running Automated Tests

Run the complete test suite using Python's built-in `unittest` runner (zero external dependencies required):

```bash
python -m unittest discover tests
```

**Output**:
```
Ran 31 tests in 1.40s
OK
```

### Test Coverage (31 Unit & Integration Tests):
- **Chat Sorter & Text Cleaning**: Plain text extraction, entity normalization, target user identification (`user5711494385`), 0-message chat pruning, and chat type categorization.
- **Sessionizers**: Burst merging, conversation finalization, Approach 1 idle-gap splitting, Approach 3 dry-run fallback chunking, and dual-schema JSON loading resilience.
- **Profiler & Linguistics**: Response length, lowercase ratio, punctuation frequencies (!, ?, ..., combo), Singlish particle extraction, and Big-5 academic trait estimation across DM, Group, and Supergroup contexts.
- **System Prompt Synthesizer**: System prompt structure, mode-specific formatting constraints, in-character refusal deflection directives, few-shot QA pairing, and **Multi-Party Context Directives** in group chats.
- **Inference Sandbox (`chat_app.py`)**: 5-turn sliding window buffer, SimulatedClient persona response generation, refusal deflection trigger matching, and LoRA adapter integration.
- **Data Export & Distractor Filtering (`export_training_data.py`)**: Detection of bot commands, media placeholders, system event notices, multi-speaker bracketed tag formatting (`[Name]: text`), strictly alternating turn enforcement, and stratified 85/15 train/val splitting.
- **QLoRA Fine-Tuning Safeguards (`train_lora.py`)**: Dry-run CPU validation, LoRA scaling factor calibration (`--style-boost` ratio 4.0), response template detection for Qwen vs. LLaMA-3, early stopping callback configuration, and Heretic abliterated base model flag resolution (`--abliterated`).



