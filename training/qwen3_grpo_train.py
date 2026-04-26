"""GRPO training for OrbitalThrusterEnv with multi-component verifier rewards."""
from __future__ import annotations

import unsloth  # noqa: F401  must precede transformers
import json
import os
from pathlib import Path

from common import ROOT, TASK_IDS, build_prompt
from rl_utils import (
    DEFAULT_MODEL,
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
        from common import collect_seed_records
        collect_seed_records(seed_path)
    records: list[dict] = []
    with seed_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return [r for r in records if r["task_id"] in TASK_IDS]


def main() -> None:
    try:
        import torch
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit("Install training/requirements.txt before running GRPO.") from exc

    raw_records = load_records()
    records = filter_records_by_curriculum(raw_records, target=192)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=DEFAULT_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model, r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32, lora_dropout=0.0,
    )

    if SFT_OUTPUT_DIR.exists() and any(SFT_OUTPUT_DIR.iterdir()):
        try:
            from safetensors.torch import load_file
            adapter_file = next(SFT_OUTPUT_DIR.glob("**/adapter_model.safetensors"), None)
            if adapter_file is not None:
                state = load_file(str(adapter_file))
                missing, unexpected = model.load_state_dict(state, strict=False)
                print(f"Warm-started from SFT adapter: {SFT_OUTPUT_DIR} (missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                print(f"No adapter_model.safetensors under {SFT_OUTPUT_DIR}; using fresh LoRA.")
        except Exception as exc:
            print(f"Could not load SFT adapter ({exc}); using fresh LoRA.")

    def to_chat_prompt(observation: dict) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(observation)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    dataset = Dataset.from_list(
        [
            {
                "prompt": to_chat_prompt(r["observation"]),
                "task_id": r["task_id"],
                "history_actions": json.dumps(r["history_actions"]),
            }
            for r in records
        ]
    )

    GRPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_logger = RewardCSVLogger(TRAIN_LOG_DIR / "grpo_metrics.csv")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=str(GRPO_OUTPUT_DIR),
            num_generations=int(os.environ.get("ORBITAL_NUM_GEN", "6")),
            max_completion_length=96,
            max_prompt_length=1536,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=2,
            max_steps=int(os.environ.get("ORBITAL_GRPO_STEPS", "300")),
            save_steps=100,
            temperature=0.9,
            top_p=0.95,
            report_to=[],
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        ),
        callbacks=[csv_logger.make_callback()],
    )
    trainer.train()
    trainer.save_model(str(GRPO_OUTPUT_DIR))
    plot_training_curves(TRAIN_LOG_DIR / "grpo_metrics.csv", TRAIN_LOG_DIR / "grpo_metrics.png")
    print(f"GRPO adapter saved to {GRPO_OUTPUT_DIR}")
    print(f"Metrics CSV: {TRAIN_LOG_DIR / 'grpo_metrics.csv'}")


if __name__ == "__main__":
    main()
