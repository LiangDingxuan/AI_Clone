#!/usr/bin/env bash
# ==============================================================================
# Dingxuan Persona QLoRA Training Runner (Linux / Cloud GPU / Colab)
# ==============================================================================
set -e

echo "=== [1/3] Checking GPU Device ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[Warning] nvidia-smi not found. Ensure an NVIDIA GPU is available."
fi

echo ""
echo "=== [2/3] Installing Dependencies ==="
pip install -r training/requirements.txt

echo ""
echo "=== [3/3] Starting QLoRA Fine-Tuning ==="
# Default: Qwen/Qwen2.5-7B-Instruct with style-boost (alpha=64) and Doppelganger safeguards
python training/train_lora.py \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --train-file "data/train.jsonl" \
    --val-file "data/val.jsonl" \
    --output-dir "checkpoints/dingxuan_lora" \
    --style-boost \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    --eval-steps 40 \
    --early-stopping-patience 2 \
    --merge-adapter

echo ""
echo "======================================================================"
echo "[SUCCESS] Training finished! Best checkpoint is in checkpoints/dingxuan_lora"
echo "You can now chat with your clone using:"
echo "  python chat_app.py --provider hf --adapter checkpoints/dingxuan_lora"
echo "======================================================================"
