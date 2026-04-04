import sys
from pathlib import Path
from typing import Any

import requests
import yaml


ROOT = Path(__file__).resolve().parent


class CheckRunner:
    def __init__(self) -> None:
        self.passed = 0
        self.total = 0

    def record(self, condition: bool, label: str) -> None:
        self.total += 1
        marker = "PASS" if condition else "FAIL"
        print(f"{marker} {label}")
        if condition:
            self.passed += 1


def load_manifest() -> dict[str, Any]:
    manifest_path = ROOT / "openenv.yaml"
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def safe_get(url: str) -> tuple[bool, dict[str, Any] | list[Any] | str]:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        try:
            return True, response.json()
        except Exception:
            return True, response.text
    except Exception as exc:
        return False, str(exc)


def safe_post(url: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any] | str]:
    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    env_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    env_url = env_url.rstrip("/")
    runner = CheckRunner()
    manifest = load_manifest()

    required_files = [
        "openenv.yaml",
        "Dockerfile",
        "server/requirements.txt",
        "inference.py",
        "pyproject.toml",
        "models.py",
        "client.py",
        "server/app.py",
        "server/orbital_thruster_environment.py",
    ]
    for file_name in required_files:
        runner.record((ROOT / file_name).exists(), f"{file_name} exists")

    runner.record(bool(manifest.get("name")), "openenv.yaml has name")
    runner.record(bool(manifest.get("version")), "openenv.yaml has version")
    runner.record(isinstance(manifest.get("tasks"), list) and len(manifest["tasks"]) >= 3, "openenv.yaml has at least 3 tasks")
    runner.record(manifest.get("reward_range") == [-1.0, 1.0], "reward_range is [-1.0, 1.0]")
    runner.record(all(isinstance(task.get("id"), str) and task.get("id") for task in manifest.get("tasks", [])), "all task IDs are non-empty")

    ok, health_body = safe_get(f"{env_url}/health")
    runner.record(ok and isinstance(health_body, dict) and health_body.get("status") == "healthy", "GET /health returns healthy")

    ok, root_body = safe_get(f"{env_url}/")
    runner.record(ok and isinstance(root_body, dict) and root_body.get("name") == "orbital-thruster-env", "GET / returns environment metadata")

    ok, tasks_body = safe_get(f"{env_url}/tasks")
    runner.record(ok and isinstance(tasks_body, list) and len(tasks_body) >= 3, "GET /tasks returns at least 3 tasks")

    task_ids = [task["id"] for task in manifest.get("tasks", []) if isinstance(task, dict) and "id" in task]
    probe_action = {"action_type": "idle", "reason": "validation probe"}

    for task_id in task_ids:
        ok, reset_body = safe_post(f"{env_url}/reset", {"task_id": task_id})
        reset_obs = reset_body.get("observation", {}) if ok and isinstance(reset_body, dict) else {}
        runner.record(ok and isinstance(reset_obs, dict) and reset_obs.get("task_id") == task_id, f"POST /reset works for {task_id}")
        runner.record("difficulty" in reset_obs and "step_budget" in reset_obs and "target_attitude_deg" in reset_obs, f"reset observation shape is valid for {task_id}")

        ok, step_body = safe_post(f"{env_url}/step", {"action": probe_action})
        step_obs = step_body.get("observation", {}) if ok and isinstance(step_body, dict) else {}
        reward = step_body.get("reward") if ok and isinstance(step_body, dict) else None
        done = step_body.get("done") if ok and isinstance(step_body, dict) else None
        runner.record(ok and isinstance(step_obs, dict) and isinstance(done, bool), f"POST /step works for {task_id}")
        runner.record(isinstance(reward, (int, float)) and -1.0 <= float(reward) <= 1.0, f"step reward is bounded for {task_id}")

        ok, state_body = safe_get(f"{env_url}/state")
        runner.record(ok and isinstance(state_body, dict) and bool(state_body.get("episode_id")), f"GET /state returns episode for {task_id}")

    print(f"{runner.passed} / {runner.total} checks passed")
    return 0 if runner.passed == runner.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
