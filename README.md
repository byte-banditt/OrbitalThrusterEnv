---
title: Orbital Thruster Environment Server
sdk: docker
pinned: false
app_port: 7860
base_path: /web
colorFrom: blue
colorTo: indigo
tags:
  - openenv
---

# OrbitalThrusterEnv: Train LLMs for Fuel-Aware Spacecraft Control

## The Problem
LLMs can explain control theory, but often fail to execute closed-loop control over long horizons under hard constraints.
OrbitalThrusterEnv trains this gap directly: stabilize and retarget a spacecraft while conserving finite propellant.
If solved well, this maps to real mission-operations workflows where poor control burns fuel, misses pointing windows, and risks mission failure.

## The Environment
Agent observes: `task_id`, `difficulty`, `mission_phase`, current attitude/rates, target attitude, signed error, fuel, disturbance level, step counters, reward, success.
Agent can: choose exactly one discrete action each step from 13 commands (`fire_<axis>_<dir>_<size>` or `idle`).
Episode ends when: success criteria are satisfied (easy/medium early stop) or task step budget is exhausted.
One episode looks like: reset task state with seeded disturbances, take one thruster action per step, and receive dense reward + feedback string.
Easy focuses detumbling, medium enforces a controlled large-angle retarget without overshoot, hard enforces long-horizon precision hold under disturbances.

## Reward Design

| Component | Weight | What it measures | Anti-hack guard |
| --- | --- | --- | --- |
| primary_objective | 0.40 | Error reduction toward target with terminal accuracy | Final max-axis error + hold streak + fuel reserve gates |
| process_quality | 0.25 | Smooth, stable control with bounded angular rates | Stability penalty + overshoot penalty + rate tolerance checks |
| format_compliance | 0.20 | Valid action schema and parse-safe output | Strict enum actions, invalid payload fallback penalties |
| efficiency | 0.15 | Fuel and step economy while still solving task | No efficiency credit unless objective quality crosses threshold |

## Results
![Reward Curve](plots/reward_curve.png)
*Reward over training steps. Dashed = untrained baseline (0.XX). Trained agent reaches 0.XX.*

![Before vs After](plots/before_after.png)
*Left: untrained agent output. Right: trained agent output on same task.*

**Summary:** Training improved mean episode reward from **X.XX -> X.XX** (+XX%).

## Quickstart
```powershell
# 1) Install runtime deps
powershell -ExecutionPolicy Bypass -File scripts/setup_deps.ps1

# 2) Run server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# 3) Validate contract
python validate.py http://127.0.0.1:7860
```

Model selection helper:
```powershell
python scripts/select_model.py --gpu t4 --mode balanced
python scripts/select_model.py --gpu a100 --mode quality
```

Training template:
```powershell
# Use this as the base notebook script for Colab
type training\grpo_colab_template.py
```

## Links
- HF Space (live env): `<add-space-url>`
- Colab training notebook: `<add-colab-url>`
- Mini-blog / writeup: `<add-blog-or-youtube-url>`
- WandB training run: `<add-wandb-url>`
- Demo video (<2 min): `<add-demo-url>`

## Local Validation Status
- `openenv-core` installed: `0.2.3`
- `validate.py` status: `32 / 32 checks passed` on local run against `http://127.0.0.1:7860`

## Project Structure
- `server/`: OpenEnv FastAPI environment implementation
- `models.py`: typed action/observation/state contracts
- `client.py`: environment client wrapper
- `openenv.yaml`: manifest metadata for OpenEnv
- `docs/HACKATHON_PLAN.md`: repo map, theme choice, reward audit, and strategy
- `docs/MODEL_OPTIONS.md`: model comparison and selection guidance
- `training/grpo_colab_template.py`: sectioned GRPO training template
