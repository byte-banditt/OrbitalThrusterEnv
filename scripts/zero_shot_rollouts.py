from __future__ import annotations

import argparse
import statistics
from typing import Any

from inference import ENV_URL, TASKS, build_llm_client, choose_action, safe_post


def run_one(task_id: str, max_steps: int) -> float:
    reset_result = safe_post(f"{ENV_URL}/reset", {"task_id": task_id})
    if not reset_result:
        return -1.0

    observation: dict[str, Any] = reset_result.get("observation", {})
    step_budget = int(observation.get("step_budget", max_steps))
    steps = min(max_steps, step_budget)

    client = build_llm_client()
    total = 0.0
    done = bool(reset_result.get("done", False))
    used = 0

    while not done and used < steps:
        used += 1
        action = choose_action(client, observation)
        step_result = safe_post(f"{ENV_URL}/step", {"action": action})
        if not step_result:
            break
        total += float(step_result.get("reward") or 0.0)
        observation = step_result.get("observation", observation)
        done = bool(step_result.get("done", False))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot rollouts before RL training.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--task", type=str, default="detumble_satellite", choices=TASKS)
    parser.add_argument("--max-steps", type=int, default=60)
    args = parser.parse_args()

    scores = [run_one(task_id=args.task, max_steps=args.max_steps) for _ in range(args.episodes)]
    mean_reward = statistics.mean(scores) if scores else 0.0
    non_zero = sum(1 for value in scores if abs(value) > 1e-9)

    print(f"task={args.task}")
    print(f"episodes={args.episodes}")
    print(f"mean_reward={mean_reward:.4f}")
    print(f"non_zero_reward_episodes={non_zero}/{args.episodes}")
    print("scores=", ",".join(f"{value:.3f}" for value in scores))


if __name__ == "__main__":
    main()
