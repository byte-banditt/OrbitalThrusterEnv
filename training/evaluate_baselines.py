from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from common import ROOT, CONTROLLERS, TASK_IDS, random_controller_factory, rollout_task


def evaluate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    controller_map = {
        "random": random_controller_factory(seed=7),
        "deterministic": CONTROLLERS["deterministic"],
        "tuned_pd": CONTROLLERS["tuned_pd"],
    }

    for policy_name, controller in controller_map.items():
        for task_id in TASK_IDS:
            rollout = rollout_task(task_id, controller)
            rows.append(
                {
                    "policy": policy_name,
                    "task_id": task_id,
                    "success": int(rollout["success"]),
                    "reward_total": rollout["reward_total"],
                    "fuel_used": rollout["fuel_used"],
                    "steps_used": rollout["steps_used"],
                    "milestones_completed_count": rollout["milestones_completed_count"],
                    "directive_completion_ratio": rollout["directive_completion_ratio"],
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_order = TASK_IDS
    policy_order = ["random", "deterministic", "tuned_pd"]
    metrics = [
        ("reward_total", "Reward Total"),
        ("success", "Success"),
        ("fuel_used", "Fuel Used"),
        ("directive_completion_ratio", "Milestone Ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for axis, (metric_key, title) in zip(axes, metrics):
        width = 0.24
        positions = list(range(len(task_order)))
        for index, policy in enumerate(policy_order):
            values = []
            for task_id in task_order:
                row = next(row for row in rows if row["policy"] == policy and row["task_id"] == task_id)
                values.append(float(row[metric_key]))
            shifted = [position + ((index - 1) * width) for position in positions]
            axis.bar(shifted, values, width=width, label=policy)
        axis.set_title(title)
        axis.set_xticks(positions)
        axis.set_xticklabels(task_order, rotation=20, ha="right")

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    rows = evaluate()
    output_dir = ROOT / "outputs" / "baseline_eval"
    write_csv(rows, output_dir / "baseline_summary.csv")
    write_plot(rows, output_dir / "baseline_summary.png")
    print(f"Wrote baseline evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
