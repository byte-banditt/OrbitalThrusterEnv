"""Vanilla TRL+peft+bnb training (no unsloth). Runs on Windows/4060.

Defaults to a small model (Qwen2.5-1.5B-Instruct) to fit 8 GB VRAM comfortably.
Override via ORBITAL_BASE_MODEL env var.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

LOCAL_MODEL = os.environ.get("ORBITAL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
os.environ["ORBITAL_BASE_MODEL"] = LOCAL_MODEL

from common import build_prompt, collect_seed_records  # noqa: E402
from rl_utils import (  # noqa: E402
    GRPO_OUTPUT_DIR,
    REWARD_FUNCS,
    SFT_OUTPUT_DIR,
    SYSTEM_PROMPT,
    TRAIN_LOG_DIR,
    RewardCSVLogger,
    filter_records_by_curriculum,
    plot_training_curves,
)


def load_records() -> list[dict]:
    seed_path = ROOT / "training" / "data" / "seed_trajectories.jsonl"
    if not seed_path.exists():
        collect_seed_records(seed_path)
    return [json.loads(line) for line in seed_path.open("r", encoding="utf-8")]


def build_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map={"": 0},
    )
    return model, tokenizer


def attach_lora(model, r: int = 16, alpha: int = 16):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def run_sft(max_steps: int = 60) -> None:
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print(f"[local_train] SFT model={LOCAL_MODEL}")
    model, tokenizer = build_model_and_tokenizer()
    model = attach_lora(model, r=16, alpha=16)

    records = load_records()[:200]
    texts = []
    for r in records:
        chat = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(r["observation"])},
                {"role": "assistant", "content": json.dumps(r["expert_action"])},
            ],
            tokenize=False,
        )
        texts.append({"text": chat})
    dataset = Dataset.from_list(texts)

    SFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(SFT_OUTPUT_DIR),
            max_steps=max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=5,
            save_steps=max_steps,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            report_to=[],
            max_length=2048,
            dataset_text_field="text",
        ),
    )
    trainer.train()
    trainer.save_model(str(SFT_OUTPUT_DIR))
    print(f"[local_train] SFT adapter saved to {SFT_OUTPUT_DIR}")


def run_grpo(max_steps: int = 120) -> None:
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    print(f"[local_train] GRPO model={LOCAL_MODEL}")
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = build_model_and_tokenizer()
    if SFT_OUTPUT_DIR.exists() and any(SFT_OUTPUT_DIR.iterdir()):
        try:
            model = PeftModel.from_pretrained(model, str(SFT_OUTPUT_DIR), is_trainable=True)
            print(f"[local_train] warm-started from SFT adapter {SFT_OUTPUT_DIR}")
        except Exception as exc:
            print(f"[local_train] SFT load failed ({exc}); attaching fresh LoRA.")
            model = attach_lora(model, r=32, alpha=32)
    else:
        model = attach_lora(model, r=32, alpha=32)

    raw = load_records()
    records = filter_records_by_curriculum(raw, target=128)

    def to_prompt(obs: dict) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": build_prompt(obs)}],
            tokenize=False, add_generation_prompt=True,
        )

    dataset = Dataset.from_list([
        {
            "prompt": to_prompt(r["observation"]),
            "task_id": r["task_id"],
            "history_actions": json.dumps(r["history_actions"]),
        }
        for r in records
    ])

    GRPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_logger = RewardCSVLogger(TRAIN_LOG_DIR / "grpo_metrics.csv")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=str(GRPO_OUTPUT_DIR),
            num_generations=2,
            max_completion_length=64,
            max_prompt_length=1280,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=2,
            max_steps=max_steps,
            save_steps=max(50, max_steps // 2),
            temperature=0.9,
            top_p=0.95,
            report_to=[],
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
        ),
        callbacks=[csv_logger.make_callback()],
    )
    trainer.train()
    trainer.save_model(str(GRPO_OUTPUT_DIR))
    plot_training_curves(TRAIN_LOG_DIR / "grpo_metrics.csv", TRAIN_LOG_DIR / "grpo_metrics.png")
    print(f"[local_train] GRPO adapter saved to {GRPO_OUTPUT_DIR}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["sft", "grpo", "all"], default="all")
    parser.add_argument("--sft-steps", type=int, default=60)
    parser.add_argument("--grpo-steps", type=int, default=120)
    args = parser.parse_args()

    if args.phase in {"sft", "all"}:
        run_sft(args.sft_steps)
    if args.phase in {"grpo", "all"}:
        run_grpo(args.grpo_steps)


if __name__ == "__main__":
    main()
