"""SFT warm-start for OrbitalThrusterEnv. JSON-format + control-mode discipline."""
from __future__ import annotations

import unsloth  # noqa: F401  must precede transformers
import json
import os
from pathlib import Path

from common import ROOT, build_prompt, collect_seed_records
from rl_utils import DEFAULT_MODEL, SFT_OUTPUT_DIR, SYSTEM_PROMPT


def ensure_records(seed_path: Path) -> list[dict[str, object]]:
    if not seed_path.exists():
        return collect_seed_records(seed_path)
    records: list[dict[str, object]] = []
    with seed_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def main() -> None:
    try:
        from unsloth import FastLanguageModel
        import torch
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install training/requirements.txt before running SFT.") from exc

    seed_path = ROOT / "training" / "data" / "seed_trajectories.jsonl"
    records = ensure_records(seed_path)[:384]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=DEFAULT_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
    )

    texts = []
    for record in records:
        user = build_prompt(record["observation"])
        assistant = json.dumps(record["expert_action"], ensure_ascii=True)
        chat = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            tokenize=False,
        )
        texts.append({"text": chat})
    dataset = Dataset.from_list(texts)

    SFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=str(SFT_OUTPUT_DIR),
            max_steps=int(os.environ.get("ORBITAL_SFT_STEPS", "150")),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=5,
            save_steps=80,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            report_to=[],
        ),
    )
    trainer.train()
    trainer.save_model(str(SFT_OUTPUT_DIR))
    print(f"SFT adapter saved to {SFT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
