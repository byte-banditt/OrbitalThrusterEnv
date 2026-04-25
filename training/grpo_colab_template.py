"""
Colab-ready GRPO template for OrbitalThrusterEnv.

Copy this file into a Colab notebook and run section by section.
"""

# =========================
# Section 1: Install
# =========================
# !pip install -q unsloth trl>=0.12.0 openenv-core[core] accelerate bitsandbytes datasets transformers
# !pip install -q git+https://github.com/byte-banditt/openenv-hackathon.git
#
# import unsloth, trl, openenv
# print("unsloth", unsloth.__version__)
# print("trl", trl.__version__)
# print("openenv", openenv.__version__)


# =========================
# Section 2: Load model with Unsloth
# =========================
import os
from dataclasses import dataclass
from typing import Any

import torch
from unsloth import FastLanguageModel


CHOSEN_MODEL = os.getenv("CHOSEN_MODEL", "Qwen/Qwen2.5-14B-Instruct")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CHOSEN_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)


# =========================
# Section 3: Connect to deployed environment
# =========================
import asyncio

from client import OrbitalThrusterEnv
from models import OrbitalThrusterAction


ENV_URL = os.getenv("ENV_URL", "https://<your-space>.hf.space")
DEFAULT_TASK = os.getenv("TASK_ID", "detumble_satellite")


async def smoke_test() -> None:
    async with OrbitalThrusterEnv(base_url=ENV_URL) as env:
        reset_result = await env.reset(task_id=DEFAULT_TASK)
        print("reset task:", reset_result.observation.task_id)
        step_result = await env.step(OrbitalThrusterAction(action_type="idle", reason="smoke test"))
        print("step reward:", step_result.reward)


# asyncio.run(smoke_test())


# =========================
# Section 4: Rollout function
# =========================
VALID_ACTIONS = {
    "fire_pitch_pos_small",
    "fire_pitch_neg_small",
    "fire_roll_pos_small",
    "fire_roll_neg_small",
    "fire_yaw_pos_small",
    "fire_yaw_neg_small",
    "fire_pitch_pos_large",
    "fire_pitch_neg_large",
    "fire_roll_pos_large",
    "fire_roll_neg_large",
    "fire_yaw_pos_large",
    "fire_yaw_neg_large",
    "idle",
}


def format_observation(obs: Any, prompt: str) -> str:
    return (
        "You control a spacecraft attitude system.\n"
        "Return JSON: {\"action_type\": ..., \"reason\": ...}\n\n"
        f"Prompt: {prompt}\n"
        f"Task: {obs.task_id}\n"
        f"Difficulty: {obs.difficulty}\n"
        f"Phase: {obs.mission_phase}\n"
        f"Attitude error: {obs.attitude_error_deg.model_dump()}\n"
        f"Angular rate: {obs.current_angular_velocity_dps.model_dump()}\n"
        f"Fuel remaining: {obs.fuel_remaining}\n"
        f"Steps used / budget: {obs.steps_used}/{obs.step_budget}\n"
        "Choose one next action.\n"
    )


def parse_action(action_text: str) -> OrbitalThrusterAction:
    # Minimal robust parser for RL loop stability.
    action_type = "idle"
    for candidate in VALID_ACTIONS:
        if candidate in action_text:
            action_type = candidate
            break
    return OrbitalThrusterAction(action_type=action_type, reason="grpo rollout")


@dataclass
class EpisodeResult:
    total_reward: float
    primary_reward: float
    process_reward: float
    format_reward: float
    efficiency_reward: float
    done: bool
    steps_used: int


async def run_episode(
    env: OrbitalThrusterEnv,
    prompt: str,
    max_steps: int = 24,
    task_id: str = DEFAULT_TASK,
) -> EpisodeResult:
    reset_result = await env.reset(task_id=task_id)
    obs = reset_result.observation
    total_reward = 0.0
    primary_total = 0.0
    process_total = 0.0
    format_total = 0.0
    efficiency_total = 0.0

    for _ in range(max_steps):
        input_text = format_observation(obs, prompt)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=96,
                temperature=0.8,
                do_sample=True,
            )
        action_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        action = parse_action(action_text)

        result = await env.step(action)
        obs = result.observation
        total_reward += result.reward

        # Heuristic decomposition for logging; replace with true reward bundle if server exposes terms.
        primary = max(0.0, 1.0 - (abs(obs.attitude_error_deg.pitch) + abs(obs.attitude_error_deg.roll) + abs(obs.attitude_error_deg.yaw)) / 30.0)
        process = max(0.0, 1.0 - (abs(obs.current_angular_velocity_dps.pitch) + abs(obs.current_angular_velocity_dps.roll) + abs(obs.current_angular_velocity_dps.yaw)) / 3.0)
        fmt = 1.0 if action.action_type in VALID_ACTIONS and len(action.reason.strip()) > 0 else 0.0
        eff = max(0.0, min(1.0, obs.fuel_remaining / 100.0))
        primary_total += primary
        process_total += process
        format_total += fmt
        efficiency_total += eff

        if result.done:
            break

    steps_used = max(1, obs.steps_used)
    return EpisodeResult(
        total_reward=total_reward,
        primary_reward=primary_total / steps_used,
        process_reward=process_total / steps_used,
        format_reward=format_total / steps_used,
        efficiency_reward=efficiency_total / steps_used,
        done=obs.done,
        steps_used=obs.steps_used,
    )


def rollout_fn(prompts, completions=None, **kwargs):
    del completions, kwargs

    async def batch():
        async with OrbitalThrusterEnv(base_url=ENV_URL) as env:
            tasks = [run_episode(env, prompt=p, task_id=DEFAULT_TASK) for p in prompts]
            return await asyncio.gather(*tasks)

    results = asyncio.run(batch())
    return [r.total_reward for r in results]


# =========================
# Section 5: GRPO Trainer
# =========================
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


def get_prompt_dataset() -> Dataset:
    prompts = [
        {"prompt": "Stabilize quickly while preserving fuel."},
        {"prompt": "Dampen angular rates before final pointing trim."},
        {"prompt": "Avoid overshoot and keep final error minimal."},
    ]
    return Dataset.from_list(prompts)


training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=100,
    report_to="wandb",
    num_generations=4,
    max_new_tokens=256,
    temperature=0.8,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=rollout_fn,
    args=training_args,
    train_dataset=get_prompt_dataset(),
    processing_class=tokenizer,
)

# trainer.train()


# =========================
# Section 6: Logging
# =========================
metrics_to_log = {
    "reward/overall": None,
    "reward/primary": None,
    "reward/process": None,
    "reward/format": None,
    "reward/efficiency": None,
    "episode/timeout_rate": None,
    "episode/success_rate": None,
}


# =========================
# Section 7: Plot and save PNGs
# =========================
import matplotlib.pyplot as plt
import pandas as pd


def save_reward_plots(log_df: pd.DataFrame, baseline_reward: float) -> None:
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(log_df["step"], log_df["reward_overall"], label="Trained agent", linewidth=2)
    ax.plot(log_df["step"], log_df["reward_primary"], label="Primary objective", linewidth=2, linestyle="--")
    ax.axhline(baseline_reward, color="gray", linestyle=":", label="Untrained baseline")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward (0-1)")
    ax.set_title("OrbitalThrusterEnv: GRPO Training Progress")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/reward_curve.png", dpi=150)
    plt.close(fig)


def save_before_after_plot(baseline_mean: float, trained_mean: float) -> None:
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Untrained", "Trained"], [baseline_mean, trained_mean])
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Before vs After GRPO")
    plt.tight_layout()
    plt.savefig("plots/before_after.png", dpi=150)
    plt.close(fig)


# =========================
# Section 8: Save model safely
# =========================
def save_model_outputs() -> None:
    model.save_pretrained_merged(
        "final_model",
        tokenizer,
        save_method="merged_16bit",
    )
    model.save_pretrained("final_adapters")
    tokenizer.save_pretrained("final_adapters")


# =========================
# Section 9: Baseline vs Trained comparison
# =========================
async def evaluate_many(task_id: str, prompts: list[str], episodes: int = 20) -> float:
    scores = []
    async with OrbitalThrusterEnv(base_url=ENV_URL) as env:
        for i in range(episodes):
            prompt = prompts[i % len(prompts)]
            result = await run_episode(env, prompt=prompt, task_id=task_id)
            scores.append(result.total_reward)
    return sum(scores) / max(len(scores), 1)


def print_improvement(baseline_mean: float, trained_mean: float) -> None:
    denom = baseline_mean if abs(baseline_mean) > 1e-9 else 1e-9
    pct = ((trained_mean - baseline_mean) / denom) * 100.0
    print(f"Improvement: {pct:.1f}%")
