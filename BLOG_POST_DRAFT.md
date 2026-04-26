# OrbitalThrusterEnv: Teaching an LLM to Fly a Spacecraft Through a 5-Phase Mission

**OpenEnv Hackathon (India 2026) — Theme #2: Long-Horizon Planning & Instruction Following**

[👉 Try the environment on Hugging Face Spaces](https://huggingface.co/spaces/pixxel-phantom/orbital-thruster-env) · [Trained adapter](https://huggingface.co/pixxel-phantom/orbital-thruster-grpo-fast)

---

## The problem

Spacecraft mission ops is not "stabilize and chill." A real relay satellite has to:

1. **Detumble** after deployment.
2. Honor a **quiet coast window** (no thruster firings unless absolutely needed).
3. **Repoint** to a new relay geometry.
4. **Recover** from an injected gyro-bias anomaly.
5. Finish in **precision hold** with fuel still in reserve.

Five phases, one episode, 360 timesteps. Burning fuel early to make Phase 1 look pretty *guarantees* Phase 5 fails. That's the long-horizon teeth — and that's exactly the kind of behaviour LLMs trained on next-token loss don't naturally have.

We built `OrbitalThrusterEnv` as an OpenEnv benchmark to put that capability under test, then trained a small LLM to actually solve it.

## The environment

`OrbitalThrusterEnv` is a deterministic FastAPI server (OpenEnv-compliant) that exposes:

- **13 discrete thruster actions** + `idle` across pitch/roll/yaw, small/large pulse
- A required `control_mode` field — the agent must declare *intent* (`detumble`, `slew`, `brake`, `trim`, `hold`, `recover`, `safe_hold`)
- Multi-phase mission directives with deadlines, milestones, and an injected gyro-bias anomaly
- A **rubric reward** (not one opaque scalar): `physical_tracking`, `fuel_discipline`, `milestone_completion`, `control_mode`, `anomaly_recovery`, `anti_stall_penalty`

Four tasks ship: `detumble_satellite` (easy), `retarget_180_flip` (medium), `long_horizon_precision_hold` (hard), and the flagship `mission_ops_long_horizon` (hard, 5 phases).

## Baselines

| Policy | Easy | Medium | Hard Hold | **Flagship** |
|---|---|---|---|---|
| Random | ❌ | ❌ | ❌ | ❌ |
| Deterministic PD | ✅ | ❌ | ❌ | ❌ |
| Tuned PD | ✅ | ✅ | ❌ | ❌ |

Tuned PD already clears the warm-up tasks. The flagship is wide open.

## Training: SFT → GRPO with multi-component rewards

**Stack**: Unsloth + TRL (`SFTTrainer` → `GRPOTrainer`) on Hugging Face Jobs (L4 GPU).

**Base model**: `Qwen/Qwen2.5-1.5B-Instruct` (4-bit QLoRA). Tool-tuned JSON adherence matters because we score on JSON validity — the verifier rejects malformed actions.

**Reward functions (5, independent, summed by GRPO)**:

| Function | Why |
|---|---|
| `reward_format` | Strict JSON parse + valid enums + reason field |
| `reward_env_step` | Replay history, score the candidate action with the **real environment reward** |
| `reward_mode_match` | Did the agent declare a control_mode that matches the active directive? |
| `reward_anti_spam` | Penalty if the same action is repeated ≥4× in 7 steps |
| `reward_fuel_discipline` | Bonus for low-fuel→idle, penalty for low-fuel→large pulse |

Five independent signals make reward hacking measurably harder (an agent that exploits any one signal gets crushed by the others).

**Curriculum**: 50 % easy / 25 % medium / 15 % hard / 10 % flagship — keeps non-zero reward signal alive while the model is still bad at the long task.

## Results

_(Embed `outputs/training/grpo_metrics.png` here — reward curves per component + loss)_

_(Embed `outputs/eval_trained/trained_vs_baseline.png` here — trained policy vs random/det/tuned-PD across all 4 tasks)_

The model learns to **conserve fuel during the coast window** and to switch its declared `control_mode` as the directive changes. Anti-spam + fuel-discipline rewards visibly bend the rollout away from the cheap "idle forever" exploit.

## What's reproducible

```bash
# baselines
python training/evaluate_baselines.py

# train (HF credits)
hf jobs uv run --flavor l4x1 --timeout 2h --secrets HF_TOKEN \
  -e ORBITAL_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  -d training/hf_job_train.py

# evaluate trained adapter
python training/eval_trained_model.py --adapter trainer_output/qwen_grpo
```

Or open `training/train_orbital_grpo.ipynb` and run all cells.

## Why this matters

Long-horizon planning + instruction following + recovering from mid-episode disturbances is the gap between an LLM that *talks* about plans and one that *executes* a plan that takes hours to play out. Spacecraft mission ops is a clean, deterministic domain to measure that gap — and a domain where reward hacking has obvious physical signatures (fuel disappears, attitude drifts, milestones get skipped).

Code: https://huggingface.co/spaces/pixxel-phantom/orbital-thruster-env
Adapter: https://huggingface.co/pixxel-phantom/orbital-thruster-grpo-fast

— Team byte-banditt
