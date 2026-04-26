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


API_BASE_URL = _first_env("API_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")
MODEL_NAME = _first_env("MODEL_NAME", "MODEL_ID", "OPENAI_MODEL", "MODEL") or "openai/gpt-4.1-mini"
API_KEY = _first_env("API_KEY", "HF_TOKEN", "OPENAI_API_KEY")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = os.getenv("BENCHMARK_NAME", "orbital-thruster-env")

TASKS = [
    "detumble_satellite",
    "retarget_180_flip",
    "long_horizon_precision_hold",
    "mission_ops_long_horizon",
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

VALID_CONTROL_MODES = {
    "detumble",
    "slew",
    "brake",
    "trim",
    "hold",
    "recover",
    "safe_hold",
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
  "control_mode": "detumble" | "slew" | "brake" | "trim" | "hold" | "recover" | "safe_hold",
  "reason": "one concise sentence"
}

Controller rules:
1. Read the full state before acting.
2. Match `control_mode` to the current directive and anomaly state.
3. Favor small pulses near target and large pulses only when a large slew is required.
4. Brake residual angular velocity before it creates overshoot.
5. Preserve fuel on the long-horizon tasks.
6. If the current rates and errors are already low, prefer idle over unnecessary firings.
7. Output valid JSON only.
""".strip()


def build_llm_client() -> OpenAI | None:
    missing: list[str] = []
    if _is_placeholder(API_KEY):
        missing.append("API_KEY")
    if _is_placeholder(API_BASE_URL):
        missing.append("API_BASE_URL")

    if missing:
        print(
            f"[inference.py] LLM path disabled; missing or placeholder config: {', '.join(missing)}. "
            "Falling back to deterministic controller.",
            file=sys.stderr,
            flush=True,
        )
        return None

    try:
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(
            f"[inference.py] Failed to initialize OpenAI client ({exc}). Falling back to deterministic controller.",
            file=sys.stderr,
            flush=True,
        )
        return None


def warmup_llm(client: OpenAI | None) -> None:
    if client is None:
        return
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return exactly: ok"},
                {"role": "user", "content": "ok"},
            ],
            temperature=0.0,
            max_tokens=4,
            stream=False,
        )
    except Exception as exc:
        print(f"[inference.py] Warmup call failed: {exc}", file=sys.stderr, flush=True)


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
        "control_mode": "safe_hold",
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
            f"Mission brief: {obs.get('mission_brief', '')}",
            f"Active directive: {obs.get('active_directive', '')}",
            f"Pending directives count: {obs.get('pending_directives_count', 0)}",
            f"Completed milestones: {json.dumps(obs.get('milestones_completed', []))}",
            f"Anomaly flags: {json.dumps(obs.get('anomaly_flags', []))}",
            f"Fuel reserve target: {obs.get('fuel_reserve_target', 0.0)}",
            f"Phase deadline step: {obs.get('phase_deadline_step', 0)}",
            f"Current attitude (deg): {json.dumps(obs.get('current_attitude_deg', {}), sort_keys=True)}",
            f"Current angular velocity (deg/s): {json.dumps(obs.get('current_angular_velocity_dps', {}), sort_keys=True)}",
            f"Target attitude (deg): {json.dumps(obs.get('target_attitude_deg', {}), sort_keys=True)}",
            f"Signed attitude error (deg): {json.dumps(obs.get('attitude_error_deg', {}), sort_keys=True)}",
            f"Fuel remaining: {obs.get('fuel_remaining', 0.0)}",
            f"Fuel used: {obs.get('fuel_used', 0.0)}",
            f"Reward so far: {obs.get('reward_so_far', 0.0)}",
            f"Last action: {obs.get('last_action', 'none')}",
            f"Reward breakdown: {json.dumps(obs.get('reward_breakdown', {}), sort_keys=True)}",
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


def _select_control_mode(
    observation: dict[str, Any],
    best_axis: str,
    best_cmd: float,
    best_error: float,
    best_rate: float,
    max_abs_error: float,
    max_abs_rate: float,
) -> str:
    phase = str(observation.get("mission_phase", ""))
    anomalies = {str(flag) for flag in observation.get("anomaly_flags", [])}
    fuel_remaining = float(observation.get("fuel_remaining", 0.0) or 0.0)
    fuel_reserve_target = float(observation.get("fuel_reserve_target", 0.0) or 0.0)
    directive = str(observation.get("active_directive", "")).lower()

    if anomalies:
        if max_abs_rate > 0.18 or "recover" in directive:
            return "recover"
        return "safe_hold"
    if fuel_remaining <= max(fuel_reserve_target, 0.0):
        return "safe_hold"
    if "detumble" in phase:
        return "detumble"
    if "retarget" in phase or "slew" in phase:
        if abs(best_error) > 3.0 and abs(best_rate) < abs(best_cmd):
            return "slew"
        if abs(best_rate) > 0.22 and (best_error > 0) != (best_rate > 0):
            return "brake"
        return "trim"
    if "coast" in phase:
        return "hold" if max_abs_error < 1.5 else "trim"
    if "precision" in phase:
        return "hold" if max_abs_error < 0.8 and max_abs_rate < 0.08 else "trim"
    if max_abs_error < 0.6 and max_abs_rate < 0.08:
        return "hold"
    del best_axis
    return "trim"


def _policy_from_params(observation: dict[str, Any], params: dict[str, float]) -> dict[str, Any]:
    errors = _to_axis_map(observation.get("attitude_error_deg", {}))
    rates = _to_axis_map(observation.get("current_angular_velocity_dps", {}))
    try:
        fuel_remaining = float(observation.get("fuel_remaining", 0.0))
    except Exception:
        fuel_remaining = 0.0
    try:
        fuel_reserve_target = float(observation.get("fuel_reserve_target", 0.0))
    except Exception:
        fuel_reserve_target = 0.0

    max_abs_error = max(abs(errors[axis]) for axis in ("pitch", "roll", "yaw"))
    max_abs_rate = max(abs(rates[axis]) for axis in ("pitch", "roll", "yaw"))
    if fuel_remaining <= params["fuel_idle"] or (max_abs_error < params["idle_e"] and max_abs_rate < params["idle_r"]):
        return {
            "action_type": "idle",
            "control_mode": "safe_hold" if fuel_remaining <= max(fuel_reserve_target, params["fuel_idle"]) else "hold",
            "reason": "Low-error hold; conserving fuel.",
        }

    commands: dict[str, float] = {}
    scores: dict[str, float] = {}
    for axis in ("pitch", "roll", "yaw"):
        cmd = (params["kp"] * errors[axis]) - (params["kd"] * rates[axis])
        if str(observation.get("mission_phase", "")) == "precision_hold":
            cmd *= params["hard_scale"]
        commands[axis] = cmd
        scores[axis] = abs(cmd) + (params["e_bonus"] * abs(errors[axis])) + (params["r_bonus"] * abs(rates[axis]))

    best_axis = max(scores, key=scores.get)
    best_error = errors[best_axis]
    best_rate = rates[best_axis]
    best_cmd = commands[best_axis]

    control_mode = _select_control_mode(
        observation,
        best_axis=best_axis,
        best_cmd=best_cmd,
        best_error=best_error,
        best_rate=best_rate,
        max_abs_error=max_abs_error,
        max_abs_rate=max_abs_rate,
    )

    if abs(best_cmd) < params["cmd_idle"]:
        return {
            "action_type": "idle",
            "control_mode": "hold" if control_mode not in {"recover", "safe_hold"} else control_mode,
            "reason": "Near target; avoid unnecessary firing.",
        }

    direction = "pos" if best_cmd >= 0 else "neg"
    use_large = abs(best_error) > params["large_e"] or abs(best_rate) > params["large_r"] or abs(best_cmd) > params["large_cmd"]
    size = "large" if use_large else "small"
    return {
        "action_type": f"fire_{best_axis}_{direction}_{size}",
        "control_mode": control_mode,
        "reason": f"{control_mode} controller response.",
    }


DETERMINISTIC_PARAMS = {
    "kp": 0.65,
    "kd": 1.30,
    "hard_scale": 1.0,
    "e_bonus": 0.20,
    "r_bonus": 0.10,
    "fuel_idle": 0.0,
    "idle_e": 0.25,
    "idle_r": 0.05,
    "cmd_idle": 0.08,
    "large_e": 9.0,
    "large_r": 0.9,
    "large_cmd": 3.0,
}

TUNED_MISSION_PARAMS = {
    "kp": 0.4658720392155283,
    "kd": 0.5876145072661834,
    "hard_scale": 0.2465717883022279,
    "e_bonus": 0.0015498199755242137,
    "r_bonus": 0.26185103060939924,
    "fuel_idle": 1.6848838684281833,
    "idle_e": 0.8080114110184469,
    "idle_r": 0.0880503995881725,
    "cmd_idle": 0.3888568014747084,
    "large_e": 9.137260765134043,
    "large_r": 1.3978094874863687,
    "large_cmd": 2.9953729477574327,
}


def deterministic_controller(observation: dict[str, Any]) -> dict[str, Any]:
    return _policy_from_params(observation, DETERMINISTIC_PARAMS)


def tuned_mission_controller(observation: dict[str, Any]) -> dict[str, Any]:
    return _policy_from_params(observation, TUNED_MISSION_PARAMS)


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
    if "control_mode" not in action:
        action["control_mode"] = fallback_action["control_mode"]
    if "reason" not in action or not isinstance(action["reason"], str) or not action["reason"].strip():
        action["reason"] = fallback_action["reason"]
    if action["action_type"] not in VALID_ACTIONS:
        action = fallback_action
    if action["control_mode"] not in VALID_CONTROL_MODES:
        action["control_mode"] = fallback_action["control_mode"]
    return action


def _one_line(value: str) -> str:
    return " ".join(str(value).split())


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = "null" if not error else _one_line(error)
    print(
        f"[STEP] step={step} action={_one_line(action)} reward={reward:.2f} "
        f"done={_bool_text(done)} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={_bool_text(success)} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


def run_task(task_id: str, client: OpenAI | None) -> dict[str, Any]:
    rewards: list[float] = []
    steps_used = 0
    success = False
    score = 0.0
    observation: dict[str, Any] = {}
    log_start(task_id)

    try:
        reset_result = safe_post(f"{ENV_URL}/reset", {"task_id": task_id})
        if not reset_result:
            return {"task_id": task_id, "score": 0.0, "success": False, "steps_used": 0}

        observation = reset_result.get("observation", {})
        step_budget = int(observation.get("step_budget", 0))
        done = bool(reset_result.get("done", False))

        while not done and steps_used < step_budget:
            steps_used += 1
            action = choose_action(client, observation)
            step_result = safe_post(f"{ENV_URL}/step", {"action": action})
            error_text: str | None = None
            if not step_result:
                error_text = "step request failed"
                done = True
                reward = -1.0
            else:
                observation = step_result.get("observation", {})
                reward = float(step_result.get("reward") or 0.0)
                done = bool(step_result.get("done", False))
                error_candidate = step_result.get("last_action_error")
                if error_candidate is None and isinstance(observation, dict):
                    error_candidate = observation.get("last_action_error")
                if error_candidate:
                    error_text = str(error_candidate)

            rewards.append(reward)
            log_step(
                step=steps_used,
                action=action.get("action_type", "idle"),
                reward=reward,
                done=done,
                error=error_text,
            )

        success = bool(observation.get("success", False))
        steps_used = int(observation.get("steps_used", steps_used))
        if rewards:
            average_reward = sum(rewards) / len(rewards)
            score = max(0.0, min(1.0, (average_reward + 1.0) / 2.0))
    except Exception as exc:
        print(f"[inference.py] Task {task_id} failed: {exc}", file=sys.stderr, flush=True)
    finally:
        log_end(success=success, steps=steps_used, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps_used": steps_used}


if __name__ == "__main__":
    llm_client = build_llm_client()
    warmup_llm(llm_client)
    for task_name in TASKS:
        run_task(task_name, llm_client)
