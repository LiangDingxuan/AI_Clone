"""
QLoRA Fine-Tuning Script with Doppelganger Drift Safeguards (training/train_lora.py)
Author: AI Software Engineer
Project: Context-Adaptive Personality Chatbot ("Dingxuan")

This script performs 4-bit QLoRA fine-tuning on open-source causal language models
(default: Qwen/Qwen2.5-7B-Instruct or meta-llama/Meta-Llama-3-8B-Instruct)
using conversation episodes exported from Telegram chat sessions.

Key Enhancements & Safeguards:
1. LoRA Scaling Factor Calibration:
   Configurable --lora-r and --lora-alpha.
   --style-boost preset configures r=16, alpha=64 (alpha/r = 4) to amplify persona quirks.
2. Doppelganger Drift Safeguards:
   - Completion-only loss masking: loss is calculated strictly on assistant tokens for both train and val.
   - Early stopping: EarlyStoppingCallback with early_stopping_patience=2 halts training if validation loss
     diverges, automatically restoring the best checkpoint.
3. Multi-party bracketed sender tags [Name]: are explicitly accommodated in chat templates.
4. One-click adapter merging with --merge-adapter.
5. Zero-GPU sanity checking with --dry-run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dingxuan Persona QLoRA Fine-Tuning")

    # Model & Data
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model path or HuggingFace hub identifier (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train.jsonl",
        help="Path to training JSONL dataset (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/val.jsonl",
        help="Path to validation JSONL dataset (default: data/val.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/dingxuan_lora",
        help="Directory to save model checkpoints and adapter (default: checkpoints/dingxuan_lora)",
    )

    # LoRA Calibration & Hyperparameters
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank dimension (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA scaling factor alpha (default: 32)",
    )
    parser.add_argument(
        "--style-boost",
        action="store_true",
        help="Preset for style amplification: sets r=16, alpha=64 (ratio alpha/r = 4) and lr=1.5e-4",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default: 0.05)",
    )

    # Training Dynamics & Early Stopping
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8 -> effective batch size 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Initial learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=40,
        help="Run evaluation every N steps (default: 40)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Doppelganger Drift safeguard: stop if val_loss does not improve for N evaluations (default: 2)",
    )
    parser.add_argument(
        "--abliterated",
        action="store_true",
        help="Use Heretic pre-abliterated model (huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2) to remove corporate refusal alignment",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face authorization token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--merge-adapter",
        action="store_true",
        help="Merge LoRA adapter into base model weights and save full model checkpoint at end",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sanity-check data files, hyperparameter configuration, and chat formatting without GPU",
    )

    args = parser.parse_args()
    if args.abliterated and args.model_name == "Qwen/Qwen2.5-7B-Instruct":
        args.model_name = "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2"
    return args


def detect_response_template(model_name: str) -> str:
    """Detects the completion delimiter template for the specified base model."""
    name_lower = model_name.lower()
    if "llama-3" in name_lower or "llama3" in name_lower or "daredevil" in name_lower:
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # Default to ChatML / Qwen format (including huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2)
    return "<|im_start|>assistant\n"


TRAINING_DIR = Path(__file__).resolve().parent
LLM_DIR = TRAINING_DIR.parent
PROJECT_ROOT = LLM_DIR.parent


def resolve_dataset_path(file_path: str) -> str:
    """Resolves dataset path across cwd, llm/data, and project root."""
    p = Path(file_path)
    if p.is_file():
        return str(p)
    candidates = [
        LLM_DIR / file_path,
        PROJECT_ROOT / file_path,
        LLM_DIR / "data" / p.name,
        PROJECT_ROOT / "data" / p.name,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return file_path


def run_dry_run_validation(args: argparse.Namespace) -> None:
    """Performs CPU-friendly validation of data files, schemas, and configurations."""
    print("=" * 70)
    print(" DINGXUAN PERSONA QLORA - DRY RUN VALIDATION")
    print("=" * 70)
    print(f"Base Model Target       : {args.model_name}")
    is_abliterated = args.abliterated or "abliterated" in args.model_name.lower()
    if is_abliterated:
        print("Abliteration Status     : ENABLED (Corporate refusal alignment removed via Heretic)")
    else:
        print("Abliteration Status     : Standard Alignment (Use --abliterated to remove refusal direction)")
    print(f"Response Template       : {repr(detect_response_template(args.model_name))}")
    print(f"Training Dataset Path   : {args.train_file}")
    print(f"Validation Dataset Path : {args.val_file}")
    print(f"Output Checkpoint Dir   : {args.output_dir}")

    # Check files exist
    train_resolved = resolve_dataset_path(args.train_file)
    val_resolved = resolve_dataset_path(args.val_file)
    train_p = Path(train_resolved)
    val_p = Path(val_resolved)

    if not train_p.is_file():
        raise FileNotFoundError(f"Training dataset not found: {train_p}. Run export_training_data.py first.")
    if not val_p.is_file():
        raise FileNotFoundError(f"Validation dataset not found: {val_p}. Run export_training_data.py first.")

    # Count records and check formatting
    train_count = 0
    train_user_turns = 0
    train_asst_turns = 0
    with open(train_p, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            train_count += 1
            train_user_turns += sum(1 for m in msgs if m["role"] == "user")
            train_asst_turns += sum(1 for m in msgs if m["role"] == "assistant")

    val_count = 0
    val_user_turns = 0
    val_asst_turns = 0
    with open(val_p, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            val_count += 1
            val_user_turns += sum(1 for m in msgs if m["role"] == "user")
            val_asst_turns += sum(1 for m in msgs if m["role"] == "assistant")

    # Hyperparameter calibration display
    alpha = 64 if args.style_boost else args.lora_alpha
    lr = 1.5e-4 if args.style_boost else args.learning_rate
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = train_count // effective_batch
    total_steps = steps_per_epoch * args.epochs

    print("-" * 70)
    print(f"Train Episodes Validated: {train_count} (Assistant turns: {train_asst_turns})")
    print(f"Val Episodes Validated  : {val_count} (Assistant turns: {val_asst_turns})")
    print(f"LoRA Calibration        : rank={args.lora_r}, alpha={alpha} (scaling ratio alpha/r={alpha/args.lora_r:.1f})")
    if args.style_boost:
        print("  -> [Style-Boost Active]: Alpha bumped to 64 to amplify Dingxuan's colloquial quirks")
    print(f"Learning Rate           : {lr:.2e}")
    print(f"Effective Batch Size    : {effective_batch} ({args.batch_size} * {args.grad_accum} grad accum)")
    print(f"Estimated Steps/Epoch   : {steps_per_epoch} steps (Total: ~{total_steps} steps over {args.epochs} epochs)")
    print(f"Doppelganger Safeguards :")
    print(f"  - Completion-only loss on validation set: ENABLED")
    print(f"  - Early stopping patience: {args.early_stopping_patience} evaluations (~every {args.eval_steps} steps)")
    print(f"  - Best model checkpoint restoration: ENABLED")
    print("=" * 70)
    print("[SUCCESS] Dry-run validation passed! Ready for training on GPU.")


try:
    import numpy as np
    import torch
    from transformers import DataCollatorForLanguageModeling

    class CustomDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
        """Completion-only collator masking prompt tokens with -100."""

        def __init__(
            self,
            response_template: str | list[int],
            instruction_template: str | list[int] | None = None,
            *c_args,
            mlm: bool = False,
            ignore_index: int = -100,
            **c_kwargs,
        ):
            super().__init__(*c_args, mlm=mlm, **c_kwargs)
            self.instruction_template = instruction_template
            self.response_template = response_template
            if isinstance(response_template, str):
                self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
            else:
                self.response_token_ids = response_template

            self.ignore_index = ignore_index
            self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if self.eos_token_id is None:
                try:
                    self.eos_token_id = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
                except Exception:
                    self.eos_token_id = None

        def torch_call(self, examples: list[Any]) -> dict[str, Any]:
            # Filter non-tensor / nested list structures (such as 'messages') that cannot be padded as tensors
            clean_examples = [
                {k: v for k, v in ex.items() if k in ("input_ids", "attention_mask", "labels")}
                if isinstance(ex, dict) else ex
                for ex in examples
            ]
            batch = super().torch_call(clean_examples)
            labels = batch["labels"].clone()
            input_ids = batch["input_ids"]

            resp_len = len(self.response_token_ids)
            for i in range(len(examples)):
                input_i = input_ids[i]
                input_arr = input_i.cpu().numpy() if hasattr(input_i, "cpu") else np.array(input_i)
                seq_len = len(input_arr)

                # Initialize all positions as ignored (-100)
                labels[i, :] = self.ignore_index

                # Find all occurrences of the response_template in input_ids
                matched_starts = [
                    idx
                    for idx in range(seq_len - resp_len + 1)
                    if input_arr[idx : idx + resp_len].tolist() == self.response_token_ids
                ]

                if not matched_starts:
                    import warnings
                    warnings.warn(
                        f"Could not find response template `{self.response_template}` in sequence {i}. "
                        f"This instance will be ignored in loss calculation."
                    )
                    continue

                for start_idx in matched_starts:
                    content_start = start_idx + resp_len
                    content_end = seq_len
                    for end_candidate in range(content_start, seq_len):
                        if input_arr[end_candidate] in (self.eos_token_id, 151645):
                            content_end = end_candidate + 1
                            break
                    labels[i, content_start:content_end] = input_ids[i, content_start:content_end]

            batch["labels"] = labels
            return batch
except ImportError:
    CustomDataCollatorForCompletionOnlyLM = None


def train(args: argparse.Namespace) -> None:
    """Executes the full QLoRA training loop on GPU."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import inspect
    import warnings
    import numpy as np
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        TrainingArguments,
    )

    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError:
        DataCollatorForCompletionOnlyLM = None

    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    from trl import SFTTrainer

    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not detected. Training a 7B model requires an NVIDIA GPU with at least 8GB VRAM.\n"
            "If testing script functionality without a GPU, please run with '--dry-run'."
        )
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info("Detected GPU: %s (%.1f GB VRAM)", gpu_name, vram_gb)

    # 1. Hyperparameter adjustments for style boost
    lora_alpha = 64 if args.style_boost else args.lora_alpha
    learning_rate = 1.5e-4 if args.style_boost else args.learning_rate

    logger.info(
        "LoRA Calibration: r=%d, alpha=%d (ratio=%.1f), lr=%.2e",
        args.lora_r,
        lora_alpha,
        lora_alpha / args.lora_r,
        learning_rate,
    )

    # 2. Tokenizer & Chat Formatting
    logger.info("Loading tokenizer for: %s", args.model_name)
    hf_token = getattr(args, "hf_token", None) or os.environ.get("HF_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            token=hf_token,
        )
    except Exception as e:
        if "huihui-ai" in args.model_name or getattr(args, "abliterated", False):
            logger.error(
                "Failed to load tokenizer for '%s': %s\n"
                "Tip: 'huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2' is the active abliterated model.\n"
                "If accessing a private repo or gated model, provide --hf-token or set HF_TOKEN env var.\n"
                "You can also use the official base model: --model-name Qwen/Qwen2.5-7B-Instruct",
                args.model_name,
                e,
            )
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. 4-bit Quantization Config (NF4)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # 4. Load Base Model
    logger.info("Loading base model in 4-bit precision...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            token=hf_token,
        )
    except Exception as e:
        if "huihui-ai" in args.model_name or getattr(args, "abliterated", False):
            logger.error(
                "Failed to load base model for '%s': %s\n"
                "Tip: 'huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2' is the active abliterated model.\n"
                "If accessing a private repo or gated model, provide --hf-token or set HF_TOKEN env var.\n"
                "You can also use the official base model: --model-name Qwen/Qwen2.5-7B-Instruct",
                args.model_name,
                e,
            )
        raise
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA Configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Load Datasets
    logger.info("Loading JSONL datasets...")
    train_file = resolve_dataset_path(args.train_file)
    val_file = resolve_dataset_path(args.val_file)
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": val_file,
        },
    )

    # 7. Completion-Only Data Collator (Safeguard against Doppelganger Drift)
    response_template = detect_response_template(args.model_name)
    logger.info("Using response template for completion loss: %s", repr(response_template))
    collator = CustomDataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # 8. Training Arguments & Safeguards
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import dataclasses

    config_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    valid_fields = set()
    if dataclasses.is_dataclass(config_cls):
        valid_fields = {f.name for f in dataclasses.fields(config_cls)}

    # Warmup parameter: transformers v5+ unifies warmup into warmup_steps (float ratio),
    # while transformers v4 uses warmup_ratio (float) and warmup_steps (int).
    warmup_args = {}
    if "warmup_ratio" in valid_fields:
        warmup_args["warmup_ratio"] = 0.05
    elif "warmup_steps" in valid_fields:
        warmup_args["warmup_steps"] = 0.05

    # Evaluation strategy parameter name: eval_strategy (modern) vs evaluation_strategy (legacy)
    strategy_args = {}
    if "eval_strategy" in valid_fields:
        strategy_args["eval_strategy"] = "steps"
    elif "evaluation_strategy" in valid_fields:
        strategy_args["evaluation_strategy"] = "steps"

    # SFT-specific configuration fields
    sft_extra_args = {}
    if "max_length" in valid_fields:
        sft_extra_args["max_length"] = args.max_seq_length
    elif "max_seq_length" in valid_fields:
        sft_extra_args["max_seq_length"] = args.max_seq_length

    # Note: We deliberately do NOT set assistant_only_loss=True in SFTConfig.
    # In TRL 1.0+, assistant_only_loss=True triggers get_training_chat_template(),
    # which fails with ValueError on Qwen2.5 chat templates (missing {% generation %} markers).
    # Instead, completion-only loss masking is robustly and accurately enforced by
    # CustomDataCollatorForCompletionOnlyLM passed as data_collator to SFTTrainer.

    raw_training_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        max_grad_norm=1.0,
        logging_steps=10,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        **warmup_args,
        **strategy_args,
        **sft_extra_args,
    )

    # Filter arguments to ensure only fields accepted by config_cls are passed
    if valid_fields:
        filtered_kwargs = {k: v for k, v in raw_training_kwargs.items() if k in valid_fields}
    else:
        filtered_kwargs = raw_training_kwargs

    try:
        training_args = config_cls(**filtered_kwargs)
    except Exception as e:
        logger.warning(
            "Could not initialize %s with custom kwargs (%s). Falling back to TrainingArguments.",
            config_cls.__name__,
            e,
        )
        ta_fields = {f.name for f in dataclasses.fields(TrainingArguments)} if dataclasses.is_dataclass(TrainingArguments) else set()
        ta_warmup = {"warmup_ratio": 0.05} if "warmup_ratio" in ta_fields else {"warmup_steps": 0.05}
        ta_strat = {"eval_strategy": "steps"} if "eval_strategy" in ta_fields else {"evaluation_strategy": "steps"}
        fallback_raw = dict(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            fp16=(compute_dtype == torch.float16),
            bf16=(compute_dtype == torch.bfloat16),
            max_grad_norm=1.0,
            logging_steps=10,
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            **ta_warmup,
            **ta_strat,
        )
        filtered_fallback = {k: v for k, v in fallback_raw.items() if not ta_fields or k in ta_fields}
        training_args = TrainingArguments(**filtered_fallback)

    # 9. Initialize SFTTrainer with EarlyStoppingCallback
    sft_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "data_collator": collator,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
            ),
        ],
    }

    # Pass tokenizer using modern (processing_class) or legacy (tokenizer) parameter name
    if "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # max_seq_length is passed to SFTTrainer in older TRL; in newer TRL it is inside SFTConfig
    if "max_seq_length" in sft_params:
        trainer_kwargs["max_seq_length"] = args.max_seq_length

    trainer = SFTTrainer(**trainer_kwargs)

    # 10. Execute Training
    logger.info("Starting QLoRA fine-tuning...")
    trainer.train()

    # 11. Save Best LoRA Adapter
    adapter_path = out_dir / "adapter"
    logger.info("Saving best LoRA adapter checkpoint to: %s", adapter_path)
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    # 12. Optional Adapter Merging
    if args.merge_adapter:
        logger.info("Merging LoRA weights with base model for standalone inference...")
        try:
            from peft import PeftModel
            merged_model = trainer.model.merge_and_unload()
            merged_dir = out_dir / "merged_model"
            merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(merged_dir))
            logger.info("Standalone merged weights saved to: %s", merged_dir)
        except Exception as e:
            logger.warning("Could not merge adapter in-memory (likely due to 4-bit base weights): %s", e)
            logger.info("To merge in 16-bit, load base model in fp16 and call PeftModel.merge_and_unload().")

    logger.info("Training pipeline complete! Output saved to: %s", out_dir)


def main():
    args = parse_args()
    if args.dry_run:
        run_dry_run_validation(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
