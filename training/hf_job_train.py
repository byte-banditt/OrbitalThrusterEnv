#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "huggingface_hub>=0.26.0",
#     "torch>=2.6.0",
#     "torchvision",
#     "unsloth>=2025.10.1",
#     "unsloth_zoo>=2025.10.1",
#     "trl>=0.21.0",
#     "transformers>=4.55.0",
#     "datasets>=3.1.0",
#     "accelerate>=1.7.0",
#     "peft>=0.15.0",
#     "bitsandbytes>=0.45.0",
#     "matplotlib>=3.9.0",
#     "pyyaml",
#     "fastapi",
#     "uvicorn",
#     "pydantic>=2.6.0",
#     "openenv-core[core]>=0.2.2",
#     "openai>=1.20.0",
#     "requests>=2.31.0",
# ]
# ///
"""HF Job entrypoint: download repo from HF Space, run SFT + GRPO, upload artifacts.

Run with:
    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN training/hf_job_train.py

Env vars (all optional):
    SOURCE_REPO       (default: pixxel-phantom/orbital-thruster-env)
    SOURCE_REPO_TYPE  (default: space)
    OUTPUT_REPO       (default: pixxel-phantom/orbital-thruster-grpo)
    ORBITAL_BASE_MODEL (default: Qwen/Qwen3-4B-Instruct-2507)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SOURCE_REPO = os.environ.get("SOURCE_REPO", "pixxel-phantom/orbital-thruster-env")
SOURCE_REPO_TYPE = os.environ.get("SOURCE_REPO_TYPE", "space")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "pixxel-phantom/orbital-thruster-grpo")
BASE_MODEL = os.environ.get("ORBITAL_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

WORK = Path("/tmp/orbital_work")
WORK.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"[hf_job] downloading {SOURCE_REPO} ({SOURCE_REPO_TYPE})")
    repo_dir = snapshot_download(
        repo_id=SOURCE_REPO,
        repo_type=SOURCE_REPO_TYPE,
        local_dir=str(WORK / "repo"),
    )
    repo = Path(repo_dir)
    print(f"[hf_job] repo at {repo}")

    os.environ["ORBITAL_BASE_MODEL"] = BASE_MODEL
    env = {**os.environ, "PYTHONPATH": str(repo)}

    def run(cmd: list[str], cwd: Path) -> None:
        print(f"[hf_job] $ {' '.join(cmd)}  (cwd={cwd})")
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True)

    seed_path = repo / "training" / "data" / "seed_trajectories.jsonl"
    if not seed_path.exists():
        run([sys.executable, "training/generate_seed_trajectories.py"], cwd=repo)

    run([sys.executable, "training/qwen3_smoke_sft.py"], cwd=repo)
    run([sys.executable, "training/qwen3_grpo_train.py"], cwd=repo)
    run([sys.executable, "training/eval_trained_model.py"], cwd=repo)

    api = HfApi()
    api.create_repo(repo_id=OUTPUT_REPO, repo_type="model", exist_ok=True)
    print(f"[hf_job] uploading artifacts to {OUTPUT_REPO}")

    artifact_root = WORK / "upload"
    artifact_root.mkdir(exist_ok=True)
    for sub in ["trainer_output/qwen_grpo", "trainer_output/qwen_sft", "outputs"]:
        src = repo / sub
        if src.exists():
            dst = artifact_root / sub
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    readme = artifact_root / "README.md"
    readme.write_text(
        "# OrbitalThrusterEnv — GRPO LoRA adapter\n\n"
        f"Base model: `{BASE_MODEL}`\n\n"
        f"Source env: https://huggingface.co/spaces/{SOURCE_REPO}\n\n"
        "Trained via TRL `GRPOTrainer` + Unsloth on OpenEnv `OrbitalThrusterEnv` flagship task `mission_ops_long_horizon`.\n"
        "5 independent reward funcs (format, env-step, mode-match, anti-spam, fuel-discipline) for anti-reward-hacking.\n\n"
        "## Artifacts\n"
        "- `trainer_output/qwen_grpo/` — final LoRA adapter\n"
        "- `trainer_output/qwen_sft/` — SFT warm-start adapter\n"
        "- `outputs/training/grpo_metrics.png` — reward + loss curves\n"
        "- `outputs/eval_trained/trained_vs_baseline.png` — trained vs baselines on 4 tasks\n"
    )
    api.upload_folder(folder_path=str(artifact_root), repo_id=OUTPUT_REPO, repo_type="model")
    print(f"[hf_job] done. https://huggingface.co/{OUTPUT_REPO}")


if __name__ == "__main__":
    main()
