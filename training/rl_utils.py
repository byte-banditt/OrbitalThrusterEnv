from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from common import ROOT, TASK_IDS, build_prompt, parse_action_json
from inference import VALID_ACTIONS, VALID_CONTROL_MODES, deterministic_controller, tuned_mission_controller
from models import OrbitalThrusterAction
from server.orbital_thruster_environment import OrbitalThrusterEnvironment
from server.tasks import get_task

DEFAULT_MODEL = os.environ.get("ORBITAL_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
SFT_OUTPUT_DIR = ROOT / "trainer_output" / "qwen_sft"
GRPO_OUTPUT_DIR = ROOT / "trainer_output" / "qwen_grpo"
TRAIN_LOG_DIR = ROOT / "outputs" / "training"
TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
SYSTEM_PROMPT = (
    "You are an OrbitalThrusterEnv mission-ops controller. "
    "Reply with one JSON object only: "
    '{"action_type": "...", "control_mode": "...", "reason": "..."}. '
    "No prose, no code fences."
)


def chat_prompt(tokenizer, observation: dict[str, Any]) -> str:
    user = build_prompt(observation)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _extract_text(completion: Any) -> str:
    if isinstance(completion, list) and completion:
        item = completion[0]
        if isinstance(item, dict):
            return str(item.get("content", ""))
    return str(completion)


def _safe_parse(raw: str) -> dict[str, str] | None:
    parsed = parse_action_json(raw)
    if parsed is not None:
        return parsed
    match = JSON_RE.search(raw or "")
    if match:
        return parse_action_json(match.group(0))
    return None


# ------------------------- Reward functions -------------------------

def reward_format(completions, **_kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        parsed = _safe_parse(text)
        if parsed is None:
            rewards.append(-1.0)
        else:
            score = 0.4
            if parsed["action_type"] in VALID_ACTIONS:
                score += 0.2
            if parsed["control_mode"] in VALID_CONTROL_MODES:
                score += 0.2
            if parsed.get("reason"):
                score += 0.2
            rewards.append(min(1.0, score))
    return rewards


def reward_env_step(completions, task_id, history_actions, **_kwargs) -> list[float]:
    rewards = []
    for completion, item_task, item_history in zip(completions, task_id, history_actions):
        parsed = _safe_parse(_extract_text(completion))
        if parsed is None:
            rewards.append(-0.5)
            continue
        try:
            env = OrbitalThrusterEnvironment()
            env.reset(task_id=item_task)
            history = json.loads(item_history) if isinstance(item_history, str) else item_history
            obs = None
            for action_payload in history:
                obs = env.step(OrbitalThrusterAction(**action_payload))
                if obs.done:
                    break
            if obs is not None and obs.done:
                rewards.append(0.0)
                continue
            next_obs = env.step(OrbitalThrusterAction(**parsed))
            rewards.append(float(next_obs.reward))
        except Exception:
            rewards.append(-0.5)
    return rewards


def reward_mode_match(completions, task_id, history_actions, **_kwargs) -> list[float]:
    rewards = []
    for completion, item_task, item_history in zip(completions, task_id, history_actions):
        parsed = _safe_parse(_extract_text(completion))
        if parsed is None:
            rewards.append(-0.2)
            continue
        try:
            history = json.loads(item_history) if isinstance(item_history, str) else item_history
            step_index = len(history)
            task = get_task(item_task)
            recommended = set(task.recommended_modes_for_step(step_index + 1))
            if not recommended:
                rewards.append(0.0)
            elif parsed["control_mode"] in recommended:
                rewards.append(0.25)
            else:
                rewards.append(-0.15)
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_anti_spam(completions, history_actions, **_kwargs) -> list[float]:
    rewards = []
    for completion, item_history in zip(completions, history_actions):
        parsed = _safe_parse(_extract_text(completion))
        if parsed is None:
            rewards.append(0.0)
            continue
        try:
            history = json.loads(item_history) if isinstance(item_history, str) else item_history
            recent = [a.get("action_type") for a in history[-6:]]
            recent.append(parsed["action_type"])
            top_count = Counter(recent).most_common(1)[0][1]
            if top_count >= 6:
                rewards.append(-0.4)
            elif top_count >= 4:
                rewards.append(-0.15)
            else:
                rewards.append(0.05)
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_fuel_discipline(completions, task_id, history_actions, **_kwargs) -> list[float]:
    rewards = []
    for completion, item_task, item_history in zip(completions, task_id, history_actions):
        parsed = _safe_parse(_extract_text(completion))
        if parsed is None:
            rewards.append(0.0)
            continue
        try:
            env = OrbitalThrusterEnvironment()
            obs = env.reset(task_id=item_task)
            history = json.loads(item_history) if isinstance(item_history, str) else item_history
            for action_payload in history:
                obs = env.step(OrbitalThrusterAction(**action_payload))
                if obs.done:
                    break
            fuel_remaining = float(obs.fuel_remaining)
            reserve = float(obs.fuel_reserve_target or 0.0)
            low_fuel = fuel_remaining <= max(reserve, 8.0)
            is_large = "large" in parsed["action_type"]
            is_idle_or_safe = parsed["action_type"] == "idle" or parsed["control_mode"] == "safe_hold"
            if low_fuel and is_large:
                rewards.append(-0.3)
            elif low_fuel and is_idle_or_safe:
                rewards.append(0.15)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


REWARD_FUNCS: list[Callable] = [
    reward_format,
    reward_env_step,
    reward_mode_match,
    reward_anti_spam,
    reward_fuel_discipline,
]


# ------------------------- Curriculum sampling -------------------------

CURRICULUM_WEIGHTS = {
    "detumble_satellite": 0.50,
    "retarget_180_flip": 0.25,
    "long_horizon_precision_hold": 0.15,
    "mission_ops_long_horizon": 0.10,
}


def filter_records_by_curriculum(records: list[dict[str, Any]], target: int = 256) -> list[dict[str, Any]]:
    import random
    rng = random.Random(0)
    by_task: dict[str, list[dict[str, Any]]] = {tid: [] for tid in TASK_IDS}
    for rec in records:
        tid = rec.get("task_id")
        if tid in by_task:
            by_task[tid].append(rec)
    out: list[dict[str, Any]] = []
    for tid, weight in CURRICULUM_WEIGHTS.items():
        pool = by_task[tid]
        if not pool:
            continue
        n = max(1, int(target * weight))
        rng.shuffle(pool)
        for i in range(n):
            out.append(pool[i % len(pool)])
    rng.shuffle(out)
    return out


# ------------------------- LoRA controller wrapper for eval -------------------------

def make_lora_controller(adapter_dir: str | Path, base_model: str = DEFAULT_MODEL):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_dir = str(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    fallback = deterministic_controller

    @torch.inference_mode()
    def controller(observation: dict[str, Any]) -> dict[str, Any]:
        prompt = chat_prompt(tokenizer, observation)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = _safe_parse(text)
        if parsed is None:
            return fallback(observation)
        return parsed

    return controller


# ------------------------- Training metric logger -------------------------

class RewardCSVLogger:
    """Trainer-callback that appends per-step reward components and loss to a CSV."""

    def __init__(self, csv_path: Path):
        from transformers import TrainerCallback

        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: list[str] = []
        self._init_done = False
        self._cb_cls = TrainerCallback

    def make_callback(self):
        outer = self

        class _Callback(self._cb_cls):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                import csv
                row = {"step": state.global_step}
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        row[key] = value
                if not row:
                    return
                if not outer._init_done:
                    outer.fieldnames = list(row.keys())
                    with outer.csv_path.open("w", newline="", encoding="utf-8") as f:
                        csv.DictWriter(f, fieldnames=outer.fieldnames).writeheader()
                    outer._init_done = True
                for key in row:
                    if key not in outer.fieldnames:
                        outer.fieldnames.append(key)
                with outer.csv_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=outer.fieldnames, extrasaction="ignore")
                    writer.writerow(row)

        return _Callback()


def plot_training_curves(csv_path: Path, png_path: Path) -> None:
    import csv
    import matplotlib.pyplot as plt

    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    steps = [int(r["step"]) for r in rows if r.get("step")]
    keys = [k for k in rows[0].keys() if k != "step"]
    reward_keys = [k for k in keys if "reward" in k.lower() or k in {"loss", "kl"}]
    if not reward_keys:
        reward_keys = keys
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in reward_keys:
        ys = []
        xs = []
        for r in rows:
            v = r.get(key)
            if v in (None, ""):
                continue
            try:
                ys.append(float(v))
                xs.append(int(r["step"]))
            except Exception:
                continue
        if ys:
            ax.plot(xs, ys, label=key, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title("GRPO training metrics")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
