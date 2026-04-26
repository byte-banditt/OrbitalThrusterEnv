"""Evaluate trained LoRA adapter as an OrbitalThrusterEnv controller. Compare vs baselines."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from common import CONTROLLERS, ROOT, TASK_IDS, random_controller_factory, rollout_task
from rl_utils import DEFAULT_MODEL, GRPO_OUTPUT_DIR, make_lora_controller


OUTPUT_DIR = ROOT / "outputs" / "eval_trained"


def evaluate(adapter_dir: Path, base_model: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    controllers = {
        "random": random_controller_factory(seed=7),
        "deterministic": CONTROLLERS["deterministic"],
        "tuned_pd": CONTROLLERS["tuned_pd"],
        "trained": make_lora_controller(adapter_dir, base_model=base_model),
    }
    for policy, controller in controllers.items():
        for task_id in TASK_IDS:
            rollout = rollout_task(task_id, controller)
            rows.append({
                "policy": policy,
                "task_id": task_id,
                "success": int(rollout["success"]),
                "reward_total": round(float(rollout["reward_total"]), 4),
                "fuel_used": round(float(rollout["fuel_used"]), 3),
                "steps_used": rollout["steps_used"],
                "milestones_completed_count": rollout["milestones_completed_count"],
                "directive_completion_ratio": round(float(rollout["directive_completion_ratio"]), 4),
            })
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, object]], path: Path) -> None:
    policies = ["random", "deterministic", "tuned_pd", "trained"]
    metrics = [
        ("reward_total", "Reward total"),
        ("success", "Success"),
        ("directive_completion_ratio", "Milestone ratio"),
        ("fuel_used", "Fuel used"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    width = 0.20
    positions = list(range(len(TASK_IDS)))
    for ax, (key, title) in zip(axes, metrics):
        for i, policy in enumerate(policies):
            ys = [
                float(next(r for r in rows if r["policy"] == policy and r["task_id"] == t)[key])
                for t in TASK_IDS
            ]
            shifted = [p + (i - 1.5) * width for p in positions]
            ax.bar(shifted, ys, width=width, label=policy)
        ax.set_title(title)
        ax.set_xticks(positions)
        ax.set_xticklabels(TASK_IDS, rotation=20, ha="right", fontsize=8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=str(GRPO_OUTPUT_DIR))
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    adapter = Path(args.adapter)
    if not adapter.exists():
        raise SystemExit(f"Adapter dir not found: {adapter}")

    rows = evaluate(adapter, args.base_model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, OUTPUT_DIR / "trained_vs_baseline.csv")
    write_plot(rows, OUTPUT_DIR / "trained_vs_baseline.png")
    print(f"Wrote eval artifacts to {OUTPUT_DIR}")
    for row in rows:
        if row["policy"] == "trained":
            print(row)


if __name__ == "__main__":
    main()
