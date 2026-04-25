from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOption:
    name: str
    params: str
    vram_4bit_gb: float
    strengths: str
    best_for: str
    score: int


MODELS = [
    ModelOption(
        name="Qwen3.5-7B-Instruct",
        params="7B",
        vram_4bit_gb=5.0,
        strengths="strong structured output, stable instruction following",
        best_for="T4 16GB baseline + fast iteration",
        score=8,
    ),
    ModelOption(
        name="Qwen2.5-14B-Instruct",
        params="14B",
        vram_4bit_gb=9.0,
        strengths="best quality/compute tradeoff for planning + control prompts",
        best_for="A100 40GB main training run",
        score=9,
    ),
    ModelOption(
        name="Llama-3.1-8B-Instruct",
        params="8B",
        vram_4bit_gb=5.5,
        strengths="strong general instruction model, robust open tooling",
        best_for="T4 fallback with good quality",
        score=8,
    ),
    ModelOption(
        name="Mistral-7B-Instruct-v0.3",
        params="7B",
        vram_4bit_gb=5.0,
        strengths="fast rollout throughput",
        best_for="high-iteration ablations",
        score=7,
    ),
    ModelOption(
        name="DeepSeek-R1-Distill-Qwen-7B",
        params="7B",
        vram_4bit_gb=5.0,
        strengths="strong reasoning traces for action justification",
        best_for="reasoning-heavy prompts and analysis",
        score=7,
    ),
    ModelOption(
        name="Llama-3.3-70B-Instruct",
        params="70B",
        vram_4bit_gb=40.0,
        strengths="highest raw quality",
        best_for="final showcase eval only (expensive)",
        score=6,
    ),
]


def pick_model(gpu: str, mode: str) -> ModelOption:
    if gpu == "t4":
        pool = [m for m in MODELS if m.vram_4bit_gb <= 8.0]
    elif gpu == "a100":
        pool = [m for m in MODELS if m.vram_4bit_gb <= 12.0]
    else:
        pool = list(MODELS)

    if mode == "quality":
        return max(pool, key=lambda m: (m.score, m.vram_4bit_gb))
    if mode == "speed":
        return min(pool, key=lambda m: (m.vram_4bit_gb, -m.score))
    if mode == "reasoning":
        for name in ("Qwen2.5-14B-Instruct", "DeepSeek-R1-Distill-Qwen-7B"):
            for model in pool:
                if model.name == name:
                    return model
    return max(pool, key=lambda m: m.score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a hackathon LLM for OrbitalThrusterEnv.")
    parser.add_argument("--gpu", choices=["t4", "a100", "any"], default="t4")
    parser.add_argument("--mode", choices=["balanced", "quality", "speed", "reasoning"], default="balanced")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    chosen = pick_model(args.gpu, args.mode)

    if args.json:
        print(json.dumps(chosen.__dict__, indent=2))
        return

    print(f"GPU: {args.gpu}")
    print(f"Mode: {args.mode}")
    print("Recommended model:")
    print(f"- {chosen.name}")
    print(f"- Params: {chosen.params}")
    print(f"- 4-bit VRAM: ~{chosen.vram_4bit_gb} GB")
    print(f"- Why: {chosen.strengths}")
    print(f"- Best for: {chosen.best_for}")
    print("")
    print("Runner-up options:")
    for model in sorted(MODELS, key=lambda m: m.score, reverse=True)[:3]:
        print(f"- {model.name} (score {model.score}/10, ~{model.vram_4bit_gb} GB)")


if __name__ == "__main__":
    main()
