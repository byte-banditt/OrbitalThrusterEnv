from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
SITE_PACKAGES = ROOT / "venv310" / "Lib" / "site-packages"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SITE_PACKAGES.exists() and str(SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(SITE_PACKAGES))

from inference import (  # noqa: E402
    VALID_ACTIONS,
    VALID_CONTROL_MODES,
    deterministic_controller,
    tuned_mission_controller,
)
from models import OrbitalThrusterAction  # noqa: E402
from server.orbital_thruster_environment import OrbitalThrusterEnvironment  # noqa: E402


TASK_IDS = [
    "detumble_satellite",
    "retarget_180_flip",
    "long_horizon_precision_hold",
    "mission_ops_long_horizon",
]


def random_controller_factory(seed: int = 0) -> Callable[[dict[str, Any]], dict[str, str]]:
    rng = random.Random(seed)
    action_choices = sorted(VALID_ACTIONS)
    mode_choices = sorted(VALID_CONTROL_MODES)

    def _controller(_observation: dict[str, Any]) -> dict[str, str]:
        return {
            "action_type": rng.choice(action_choices),
            "control_mode": rng.choice(mode_choices),
            "reason": "seeded random baseline",
        }

    return _controller


CONTROLLERS: dict[str, Callable[[dict[str, Any]], dict[str, str]]] = {
    "deterministic": deterministic_controller,
    "tuned_pd": tuned_mission_controller,
}


def rollout_task(
    task_id: str,
    controller: Callable[[dict[str, Any]], dict[str, Any]],
    record_history: bool = False,
) -> dict[str, Any]:
    env = OrbitalThrusterEnvironment()
    observation = env.reset(task_id=task_id)
    history: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    while not observation.done:
        observation_payload = observation.model_dump()
        action_payload = controller(observation_payload)
        action = OrbitalThrusterAction(**action_payload)
        if record_history:
            trace.append(
                {
                    "task_id": task_id,
                    "step_index": observation.steps_used,
                    "history_actions": list(history),
                    "observation": observation_payload,
                    "expert_action": action_payload,
                }
            )
        history.append(action_payload)
        observation = env.step(action)

    return {
        "task_id": task_id,
        "success": observation.success,
        "reward_total": observation.reward_so_far,
        "fuel_used": observation.fuel_used,
        "fuel_remaining": observation.fuel_remaining,
        "steps_used": observation.steps_used,
        "milestones_completed_count": len(observation.milestones_completed),
        "directive_completion_ratio": observation.episode_metrics.get("directive_completion_ratio", 0.0),
        "reward_columns": dict(env.state.reward_columns),
        "last_feedback": observation.last_feedback,
        "trace": trace,
    }


def collect_seed_records(
    output_path: Path,
    episodes_per_task: int = 2,
    max_records_per_task: int = 96,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        per_task_records: list[dict[str, Any]] = []
        for _ in range(episodes_per_task):
            rollout = rollout_task(task_id, tuned_mission_controller, record_history=True)
            per_task_records.extend(rollout["trace"])
        records.extend(per_task_records[:max_records_per_task])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return records


def build_prompt(observation: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are a mission-operations spacecraft controller.",
            f"Task: {observation['task_id']}",
            f"Difficulty: {observation['difficulty']}",
            f"Mission phase: {observation['mission_phase']}",
            f"Mission brief: {observation['mission_brief']}",
            f"Active directive: {observation['active_directive']}",
            f"Pending directives: {observation['pending_directives_count']}",
            f"Anomaly flags: {json.dumps(observation['anomaly_flags'])}",
            f"Fuel reserve target: {observation['fuel_reserve_target']}",
            f"Phase deadline step: {observation['phase_deadline_step']}",
            f"Current attitude: {json.dumps(observation['current_attitude_deg'], sort_keys=True)}",
            f"Current rates: {json.dumps(observation['current_angular_velocity_dps'], sort_keys=True)}",
            f"Target attitude: {json.dumps(observation['target_attitude_deg'], sort_keys=True)}",
            f"Attitude error: {json.dumps(observation['attitude_error_deg'], sort_keys=True)}",
            f"Fuel remaining: {observation['fuel_remaining']}",
            f"Reward breakdown: {json.dumps(observation.get('reward_breakdown', {}), sort_keys=True)}",
            "Return JSON with action_type, control_mode, reason.",
        ]
    )


def parse_action_json(text: str) -> dict[str, str] | None:
    text = text.strip()
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    action_type = str(data.get("action_type", ""))
    control_mode = str(data.get("control_mode", ""))
    reason = str(data.get("reason", "")).strip() or "model output"
    if action_type not in VALID_ACTIONS or control_mode not in VALID_CONTROL_MODES:
        return None
    return {
        "action_type": action_type,
        "control_mode": control_mode,
        "reason": reason[:240],
    }
