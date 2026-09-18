# Dingxuan AI Clone — GPU QLoRA Fine-Tuning Guide (Route B)

This directory contains the complete training package for fine-tuning open-source LLMs (**Qwen-2.5-7B-Instruct** or **Meta-Llama-3-8B-Instruct**) on Liang Dingxuan's conversational dataset.

---

## 🚀 Quick Start on Another GPU Device

### Option 1: Linux / Cloud GPU Workstation (RunPod, Lambda, Vast.ai)

1. **Clone the repository on your GPU instance**:
   ```bash
   git clone https://github.com/LiangDingxuan/AI_Clone.git
   cd AI_Clone
   ```

2. **Run training in one command**:
   ```bash
   bash llm/training/run_training.sh
   ```

---

### Option 2: Windows Workstation with NVIDIA GPU

1. **Clone and enter repository**:
   ```cmd
   git clone https://github.com/LiangDingxuan/AI_Clone.git
   cd AI_Clone
   ```

2. **Double-click or run**:
   ```cmd
   llm\training\run_training.bat
   ```

---

### Option 3: Free Google Colab (T4 or A100)

1. Open a new notebook on [Google Colab](https://colab.research.google.com/) and set the runtime to **GPU** (`Runtime` -> `Change runtime type` -> `T4 GPU` or `A100`).
2. Run the following cell:

```python
# 1. Reset directory and clone/pull repository (prevents nested folders)
%cd /content
import os
if not os.path.exists("/content/AI_Clone"):
    !git clone https://github.com/LiangDingxuan/AI_Clone.git
%cd /content/AI_Clone
!git pull origin main

# 2. Install dependencies
!pip install -r llm/training/requirements.txt

# 3. Run QLoRA training on abliterated base
!python llm/training/train_lora.py --abliterated --style-boost --epochs 3 --merge-adapter

# 4. Zip and download the checkpoint
!zip -r dingxuan_adapter.zip checkpoints/dingxuan_lora/adapter
from google.colab import files
files.download("dingxuan_adapter.zip")
```

---

## 🧠 Key Features & Safeguards

### 1. Heretic Model Abliteration (`--abliterated`)
Instruction models (`Qwen-2.5-7B-Instruct`) contain internal "refusal direction" vectors installed via corporate RLHF safety training. When discussing casual banter, gaming slang, or edgy topics with friends, base models can misfire and generate corporate refusal boilerplate (*"As an AI language model..."*), breaking character.
- Passing `--abliterated` automatically uses **`huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2`** (abliterated using **Heretic** via Bayesian directional ablation). Supports private/gated repositories via `--hf-token <token>` or `HF_TOKEN` environment variable.
- This completely removes corporate refusal tendencies from the weights without retraining.
- Any necessary refusals (e.g. asking for personal credentials or historical memory probing) are handled cleanly by our in-character [`RefusalDeflectionLayer`](../chat_app.py) (*"whut why u asking that lol"*).

### 2. Style-Boost Scaling Factor ($\alpha$) Calibration
By default, the training runner enables `--style-boost`, which configures:
- **LoRA Rank**: $r = 16$
- **LoRA Alpha**: $\alpha = 64$ (ratio $\alpha / r = 4.0$)
- **Learning Rate**: $1.5 \times 10^{-4}$

Because persona cloning is style-heavy rather than knowledge-heavy, this ratio ensures Dingxuan's signature Singlish fillers (`ah`, `sia`, `cuz`, `idk`, `yea`, `wait`) and punchy brevity are captured effectively without sounding like a generic corporate assistant.

### 3. "Doppelganger Drift" Safeguards (Style Overfitting Protection)
- **Validation Completion Loss**: The validation set loss is computed **strictly on assistant tokens** using our `CustomDataCollatorForCompletionOnlyLM` (with direct token-id matching and `<|im_end|>` preservation). Loss is never computed on user/system prompts, protecting against Doppelganger Drift without requiring fragile chat template patching.
- **Early Stopping**: The trainer evaluates validation loss every 40 steps. If the validation loss fails to improve for 2 consecutive checks while training loss plummets, training halts automatically and restores the best checkpoint (`load_best_model_at_end=True`).
- **Transformers v4 & v5+ Dynamic Compatibility**: `train_lora.py` automatically detects and adapts configuration parameters (`warmup_steps` float ratio vs `warmup_ratio`, `eval_strategy` vs `evaluation_strategy`), ensuring seamless compatibility across different cloud runtime environments without manual parameter tuning.

### 4. Multi-Party Context Directive
In group and supergroup chats, dialogue from third-party peers is formatted as `[Sender Name]: text`. The training samples incorporate an explicit system directive instructing Qwen to treat bracketed names as room background, preventing the model from confusing its identity with other participants.

---

## 💬 Chatting with Your Trained Clone

Once training finishes, the best adapter checkpoint will be saved in `checkpoints/dingxuan_lora/adapter` (or `checkpoints/dingxuan_lora/merged_model` if `--merge-adapter` was used).

Launch the interactive chat sandbox with your fine-tuned clone:

```bash
# Using the LoRA adapter:
python chat_app.py --provider hf --adapter checkpoints/dingxuan_lora/adapter

# Using the merged standalone model:
python chat_app.py --provider hf --model checkpoints/dingxuan_lora/merged_model
```
