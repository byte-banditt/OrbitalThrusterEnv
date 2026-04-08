import json
import os
import re
import sys
import time
from typing import Any

import requests
from openai import OpenAI


PLACEHOLDER_VALUES = {
    "",
    "default",
    "none",
    "null",
    "<set-api-base-url>",
    "<set-model-name>",
}


def _first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value is not None:
            return value.strip()
    return None


def _is_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in PLACEHOLDER_VALUES


API_BASE_URL = _first_env("API_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE") or "<set-api-base-url>"
MODEL_NAME = _first_env("MODEL_NAME", "MODEL_ID", "OPENAI_MODEL", "MODEL") or "<set-model-name>"
HF_TOKEN = _first_env("HF_TOKEN", "OPENAI_API_KEY", "API_KEY")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

TASKS = [
    "detumble_satellite",
    "retarget_180_flip",
    "long_horizon_precision_hold",
]

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

SYSTEM_PROMPT = """
You are a fuel-aware satellite mission-operations controller inside OrbitalThrusterEnv.
Your task is to choose one discrete thruster pulse or idle at each step to minimize attitude
error, suppress angular velocity, and conserve propellant.

You must respond with JSON only and use this schema:
{
  "action_type": "fire_pitch_pos_small" | "fire_pitch_neg_small" |
                 "fire_roll_pos_small" | "fire_roll_neg_small" |
                 "fire_yaw_pos_small" | "fire_yaw_neg_small" |
                 "fire_pitch_pos_large" | "fire_pitch_neg_large" |
                 "fire_roll_pos_large" | "fire_roll_neg_large" |
                 "fire_yaw_pos_large" | "fire_yaw_neg_large" |
                 "idle",
  "reason": "one concise sentence"
}

Controller rules:
1. Read the full state before acting.
2. Favor small pulses near target and large pulses only when a large slew is required.
3. Brake residual angular velocity before it creates overshoot.
4. Preserve fuel on the hard task.
5. If the current rates and errors are already low, prefer idle over unnecessary firings.
6. Output valid JSON only.
""".strip()


def build_llm_client() -> OpenAI | None:
    missing: list[str] = []
    if _is_placeholder(HF_TOKEN):
        missing.append("HF_TOKEN")
    if _is_placeholder(API_BASE_URL):
        missing.append("API_BASE_URL")
    if _is_placeholder(MODEL_NAME):
        missing.append("MODEL_NAME")

    if missing:
        print(
            f"[inference.py] LLM path disabled; missing or placeholder config: {', '.join(missing)}. "
            "Falling back to deterministic controller.",
            file=sys.stderr,
            flush=True,
        )
        return None

    try:
        return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as exc:
        print(
            f"[inference.py] Failed to initialize OpenAI client ({exc}). Falling back to deterministic controller.",
            file=sys.stderr,
            flush=True,
        )
        return None


def safe_post(url: str, payload: dict[str, Any], retries: int = 3, delay: float = 1.5) -> dict[str, Any] | None:
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(delay)
    return None


def parse_llm_response(raw: str) -> dict[str, Any]:
    fallback = {
        "action_type": "idle",
        "reason": "Fallback to idle after parse failure.",
    }
    if not raw:
        return fallback

    text = raw.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return fallback


def build_user_prompt(obs: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Task ID: {obs.get('task_id', '')}",
            f"Difficulty: {obs.get('difficulty', '')}",
            f"Mission phase: {obs.get('mission_phase', '')}",
            f"Current attitude (deg): {json.dumps(obs.get('current_attitude_deg', {}), sort_keys=True)}",
            f"Current angular velocity (deg/s): {json.dumps(obs.get('current_angular_velocity_dps', {}), sort_keys=True)}",
            f"Target attitude (deg): {json.dumps(obs.get('target_attitude_deg', {}), sort_keys=True)}",
            f"Signed attitude error (deg): {json.dumps(obs.get('attitude_error_deg', {}), sort_keys=True)}",
            f"Fuel remaining: {obs.get('fuel_remaining', 0.0)}",
            f"Fuel used: {obs.get('fuel_used', 0.0)}",
            f"Reward so far: {obs.get('reward_so_far', 0.0)}",
            f"Last action: {obs.get('last_action', 'none')}",
            f"Disturbance level: {obs.get('disturbance_level', 0.0)}",
            f"Step budget: {obs.get('step_budget', 0)}",
            f"Steps used: {obs.get('steps_used', 0)}",
            f"Last feedback: {obs.get('last_feedback', 'None')}",
            "",
            "Choose exactly one next action.",
        ]
    )


def _to_axis_map(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for axis in ("pitch", "roll", "yaw"):
            try:
                result[axis] = float(value.get(axis, 0.0))
            except Exception:
                result[axis] = 0.0
        return result
    return {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}


def deterministic_controller(observation: dict[str, Any]) -> dict[str, Any]:
    errors = _to_axis_map(observation.get("attitude_error_deg", {}))
    rates = _to_axis_map(observation.get("current_angular_velocity_dps", {}))
    try:
        fuel_remaining = float(observation.get("fuel_remaining", 0.0))
    except Exception:
        fuel_remaining = 0.0

    max_abs_error = max(abs(errors[axis]) for axis in ("pitch", "roll", "yaw"))
    max_abs_rate = max(abs(rates[axis]) for axis in ("pitch", "roll", "yaw"))
    if fuel_remaining <= 0.0 or (max_abs_error < 0.25 and max_abs_rate < 0.05):
        return {"action_type": "idle", "reason": "Low-error hold; conserving fuel."}

    commands: dict[str, float] = {}
    scores: dict[str, float] = {}
    for axis in ("pitch", "roll", "yaw"):
        # PD-style command: track target error while damping angular-rate overshoot.
        cmd = (0.65 * errors[axis]) - (1.30 * rates[axis])
        commands[axis] = cmd
        scores[axis] = abs(cmd) + (0.20 * abs(errors[axis])) + (0.10 * abs(rates[axis]))

    best_axis = max(scores, key=scores.get)
    best_error = errors[best_axis]
    best_rate = rates[best_axis]
    best_cmd = commands[best_axis]

    if abs(best_cmd) < 0.08 and abs(best_error) < 0.45 and abs(best_rate) < 0.06:
        return {"action_type": "idle", "reason": "Near target; avoid unnecessary firing."}

    direction = "pos" if best_cmd >= 0 else "neg"
    use_large = abs(best_error) > 9.0 or abs(best_rate) > 0.9 or abs(best_cmd) > 3.0
    size = "large" if use_large else "small"
    return {
        "action_type": f"fire_{best_axis}_{direction}_{size}",
        "reason": "Deterministic PD controller response.",
    }


def choose_action(client: OpenAI | None, observation: dict[str, Any]) -> dict[str, Any]:
    if client is None:
        return deterministic_controller(observation)

    fallback_action = deterministic_controller(observation)
    user_prompt = build_user_prompt(observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=250,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        action = parse_llm_response(raw)
    except Exception:
        action = fallback_action

    if "action_type" not in action:
        action = fallback_action
    if "reason" not in action or not isinstance(action["reason"], str) or not action["reason"].strip():
        action["reason"] = fallback_action["reason"]
    if action["action_type"] not in VALID_ACTIONS:
        action = fallback_action
    return action


def log_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def run_task(task_id: str, client: OpenAI | None) -> dict[str, Any]:
    reset_result = safe_post(f"{ENV_URL}/reset", {"task_id": task_id})
    if not reset_result:
        failure = {
            "task_id": task_id,
            "total_reward": 0.0,
            "steps_used": 0,
            "success": False,
        }
        log_event({"event": "START", "task_id": task_id, "difficulty": "unknown", "step_budget": 0})
        log_event({"event": "END", "task_id": task_id, "total_reward": 0.0, "steps_used": 0, "success": False})
        return failure

    observation = reset_result.get("observation", {})
    difficulty = observation.get("difficulty", "unknown")
    step_budget = int(observation.get("step_budget", 0))
    log_event({"event": "START", "task_id": task_id, "difficulty": difficulty, "step_budget": step_budget})

    total_reward = 0.0
    steps_used = 0
    done = bool(reset_result.get("done", False))

    while not done and steps_used < step_budget:
        steps_used += 1
        action = choose_action(client, observation)
        step_result = safe_post(f"{ENV_URL}/step", {"action": action})
        if not step_result:
            observation = dict(observation)
            observation["last_feedback"] = "Step request failed; terminating episode."
            done = True
            reward = -1.0
        else:
            observation = step_result.get("observation", {})
            reward = float(step_result.get("reward") or 0.0)
            done = bool(step_result.get("done", False))

        total_reward = round(total_reward + reward, 6)
        log_event({
            "event": "STEP",
            "task_id": task_id,
            "step": steps_used,
            "action": action.get("action_type", "idle"),
            "reward": round(reward, 6),
            "total_reward": round(total_reward, 6),
            "done": done,
        })

    success = bool(observation.get("success", False))
    result = {
        "task_id": task_id,
        "total_reward": round(total_reward, 6),
        "steps_used": int(observation.get("steps_used", steps_used)),
        "success": success,
    }
    log_event({
        "event": "END",
        "task_id": task_id,
        "total_reward": result["total_reward"],
        "steps_used": result["steps_used"],
        "success": success,
    })
    return result


if __name__ == "__main__":
    llm_client = build_llm_client()
    for task_name in TASKS:
        run_task(task_name, llm_client)
