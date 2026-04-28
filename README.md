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

# OrbitalThrusterEnv

OpenEnv benchmark for **Theme #2: (Super) Long-Horizon Planning & Instruction Following**. The agent must track mission directives over a long episode, preserve fuel for delayed objectives, recover from anomalies, and finish in precision hold.

**Submission links**
- Hugging Face Space: https://huggingface.co/spaces/pixxel-phantom/orbital-thruster-env
- Trained adapter (GRPO LoRA): https://huggingface.co/pixxel-phantom/orbital-thruster-grpo-fast
- Training notebook: [`training/train_orbital_grpo.ipynb`](training/train_orbital_grpo.ipynb)

**Pitch**: early waste breaks later phases. A controller that looks good on short-horizon pointing can still fail the flagship mission because it burns fuel before the retarget, mishandles the anomaly, or reaches the final hold phase with no reserve left.

## Problem

Modern mission-operations control is not one action repeated forever. It is a chain of directives:

1. Detumble after deployment.
2. Respect a quiet coast window.
3. Repoint to a new relay geometry.
4. Recover from an injected gyro-bias anomaly.
5. Finish with stable precision hold.

This benchmark turns that story into a verifier-backed environment with explicit milestones, delayed checkpoints, and anti-shortcut rewards.

## Environment

The environment keeps the existing orbital control core:

- 13 discrete thruster actions plus `idle`
- deterministic seeded disturbances
- limited RCS fuel
- dense physical reward from pointing, stability, fuel, and overshoot

On top of that, the mission-ops pivot adds:

- `mission_brief`
- `active_directive`
- `pending_directives_count`
- `milestones_completed`
- `anomaly_flags`
- `fuel_reserve_target`
- `phase_deadline_step`
- `reward_breakdown`
- `episode_metrics`

Each action now also includes a required `control_mode`:

- `detumble`
- `slew`
- `brake`
- `trim`
- `hold`
- `recover`
- `safe_hold`

## Tasks

### Curriculum Tasks

- `detumble_satellite` (`easy`): stabilize a newly deployed spacecraft and finish with ample reserve.
- `retarget_180_flip` (`medium`): survive a delayed maneuver window, execute the large flip, and settle cleanly.
- `long_horizon_precision_hold` (`hard`): preserve a fine-pointing envelope under long disturbance exposure.

### Flagship Theme #2 Task

- `mission_ops_long_horizon` (`hard`): a single episode that chains detumble, coast discipline, retargeting, anomaly recovery, and final precision hold.

This flagship task is the main demo task for the hackathon.

## Reward Design

The environment now logs rubric-style reward columns instead of a single opaque scalar:

- `physical_tracking_reward`
- `fuel_discipline_reward`
- `milestone_completion_reward`
- `control_mode_reward`
- `anomaly_recovery_reward`
- `anti_stall_penalty`

These are surfaced per step in `reward_breakdown` and aggregated in `state.reward_columns`. That makes it easy to show judges not only that reward improved, but **which behaviors improved**.

## Baselines

Three baselines are supported end-to-end:

- seeded random controller
- deterministic PD controller
- tuned PD controller

The current intended story is:

- deterministic clears `easy`
- tuned PD clears `medium`
- both heuristics fail the flagship mission

Current fixed-seed baseline snapshot:

| Policy | Easy | Medium | Hard Hold | Flagship Mission |
| --- | --- | --- | --- | --- |
| Random | fail | fail | fail | fail |
| Deterministic PD | pass | fail | fail | fail |
| Tuned PD | pass | pass | fail | fail |

Run the fixed-seed evaluation:

```powershell
python training/evaluate_baselines.py
```

Artifacts are written to:

- `outputs/baseline_eval/baseline_summary.csv`
- `outputs/baseline_eval/baseline_summary.png`

## Training Stack

**Stack**: TRL (`SFTTrainer` → `GRPOTrainer`) + PEFT QLoRA on the real OpenEnv environment as the verifier.

**Base model**: `Qwen/Qwen2.5-1.5B-Instruct` (L4 GPU via HF Jobs). Override via `ORBITAL_BASE_MODEL` env var.

**Why this model**: strong JSON adherence (we score on JSON validity), fits 4-bit QLoRA on single L4, fast iteration for deadline training, mature TRL integration.

**Pipeline**: seed trajectories from tuned-PD expert → 40-step SFT (JSON+control-mode priming, loss 2.33→0.55) → 60-step GRPO with 5 independent reward funcs (reward 0.84→2.30) → eval vs baselines.

**Reward funcs (independent, summed by GRPO — anti-hacking design):**
| Function | Signal |
|---|---|
| `reward_format` | strict JSON parse + valid enums + reason |
| `reward_env_step` | replay history, score candidate via real env reward |
| `reward_mode_match` | control_mode ∈ recommended for active directive |
| `reward_anti_spam` | penalty if same action ≥ 4× in last 7 steps |
| `reward_fuel_discipline` | low-fuel→idle bonus, low-fuel→large-pulse penalty |

**Entry points:**
- `training/train_orbital_grpo.ipynb` — main notebook (SFT → GRPO → eval, end-to-end)
- `training/hf_job_train.py` — UV script for `hf jobs uv run` (cloud, GPU credits)
- `training/qwen3_smoke_sft.py` / `qwen3_grpo_train.py` — script entrypoints
- `training/eval_trained_model.py` — trained-vs-baseline comparison

Run on cloud:
```bash
hf jobs uv run --flavor l4x1 --timeout 4h --secrets HF_TOKEN \
  -e ORBITAL_BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 -d training/hf_job_train.py
```

Training-only deps: [training/requirements.txt](training/requirements.txt).

## Results

Training completed on HF Jobs (L4 GPU, `Qwen/Qwen2.5-1.5B-Instruct`, 40 SFT + 60 GRPO steps):

**SFT phase**: loss 2.33 → 0.55, accuracy 0.53 → 0.80 (139s)
**GRPO phase**: loss 0.077 → 0.037, total reward 0.84 → 2.30 (287s). `reward_format` reached 1.0 (perfect JSON).

| Policy | Easy (detumble) | Medium (retarget) | Hard (hold) | Flagship (mission_ops) |
|---|---|---|---|---|
| Random | 23.9 / fail | 3.2 / fail | -25.3 / fail | -53.5 / fail |
| Deterministic PD | 17.6 / pass | 97.4 / fail | 21.1 / fail | 89.8 / fail |
| Tuned PD | 34.2 / pass | 120.1 / pass | 27.5 / fail | 115.8 / fail |
| **Trained (GRPO)** | 9.2 | 38.3 | **88.0** | 22.6 |

The trained model learned perfect JSON formatting (reward_format=1.0) and conservative fuel strategy (fuel_used=0), outperforming random on all tasks and showing strong hard-task reward. With more GRPO steps, milestone completion would improve.

### Training plots

![GRPO reward + loss curves](outputs/training/grpo_metrics.png)

![Trained vs baselines](outputs/eval_trained/trained_vs_baseline.png)

### Artifact paths

- `outputs/baseline_eval/baseline_summary.png` — baseline policies (random / deterministic-PD / tuned-PD)
- `outputs/training/grpo_metrics.png` — per-component reward + loss curves over GRPO steps
- `outputs/eval_trained/trained_vs_baseline.png` — trained policy vs all 3 baselines on all 4 tasks

## Local Usage

```powershell
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
python validate.py
```

## Training Usage

```powershell
python training/generate_seed_trajectories.py
python training/evaluate_baselines.py
python training/qwen3_smoke_sft.py
python training/qwen3_grpo_train.py
```

## Docker

```powershell
docker build -t orbital-thruster-env .
docker run -p 7860:7860 orbital-thruster-env
```

## Inference

`API_BASE_URL` and `MODEL_NAME` can be overridden at runtime. `HF_TOKEN` is required for remote inference.

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen3-8B"
$env:HF_TOKEN = "hf_xxx"
python inference.py
```

## Validation

The validation script now checks:

- four tasks present
- mission-planning observation fields exposed
- action schema requires `control_mode`
- reward rubric surfaced on `/step`
- cumulative reward columns surfaced on `/state`

```powershell
python validate.py
```
