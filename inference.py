import json
import os
import re
import time
from typing import Any

import requests
from openai import OpenAI


API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ["HF_TOKEN"]
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


def choose_action(client: OpenAI, observation: dict[str, Any]) -> dict[str, Any]:
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
        action = {"action_type": "idle", "reason": "Fallback to idle after model failure."}

    if "action_type" not in action:
        action["action_type"] = "idle"
    if "reason" not in action:
        action["reason"] = "Fallback to idle after incomplete model output."
    if action["action_type"] not in VALID_ACTIONS:
        action["action_type"] = "idle"
        action["reason"] = "Fallback to idle after invalid action output."
    return action


def log_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def run_task(task_id: str, client: OpenAI) -> dict[str, Any]:
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
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    results: list[dict[str, Any]] = []
    for task_name in TASKS:
        results.append(run_task(task_name, llm_client))

    overall = round(sum(item["total_reward"] for item in results) / max(len(results), 1), 6)
    log_event({"event": "SUMMARY", "results": results, "overall_score": overall})
