# Teaching an LLM to Fly a Spacecraft: Long-Horizon Mission Control with OpenEnv + GRPO

*OpenEnv Hackathon 2026 — Theme #2: (Super) Long-Horizon Planning & Instruction Following*

---

## The Problem

Most RL environments ask an agent to do one thing repeatedly: reach a goal, score points, avoid obstacles. Real-world tasks are not like that. They are chains of directives, each with its own constraint, each depending on decisions made twenty steps ago.

Spacecraft attitude control is a perfect illustration. After deployment, a satellite doesn't just "point somewhere." It goes through a mission sequence:

1. **Detumble** — kill the post-deployment spin before any controlled motion is possible.
2. **Coast** — hold attitude quietly while the ground window is closed; burn no extra fuel.
3. **Retarget** — execute a large 180° slew to a new relay geometry, on a deadline.
4. **Anomaly recovery** — a gyro-bias spike injects false rate data; the controller must recognize and override it.
5. **Precision hold** — finish in a fine-pointing envelope and stay there.

Each phase has a different optimal strategy. Fuel burned in the detumble phase is fuel unavailable for the retarget. A controller that looks excellent on a short-horizon pointing benchmark can fail the full mission by arriving at the final hold with an empty tank, having wasted thrust on unnecessary corrections in earlier phases.

The challenge for Theme #2 was to build an OpenEnv-compliant environment that *encodes this multi-phase structure as a verifiable, trainable benchmark* — and then actually train a model on it.

---

## The Environment: `OrbitalThrusterEnv`

### Physics Layer

The environment runs a lightweight rigid-body dynamics model. Each step, `server/dynamics.py` propagates angular rates and attitude using a seeded sinusoidal disturbance signal — deterministic enough for reproducibility, rich enough to prevent memorization. Three axes (pitch, roll, yaw), individual inertia and damping constants per task, and limited RCS fuel define the physical space.

### Action Space

The agent chooses from 13 discrete thruster actions (small and large impulses on each of 3 axes, plus `idle`) and simultaneously declares a `control_mode` from 7 options (`detumble`, `slew`, `brake`, `trim`, `hold`, `recover`, `safe_hold`). The control mode declaration is not cosmetic — it feeds directly into the reward signal and is checked against the task's recommended modes for each phase.

```json
{
  "action_type": "fire_pitch_neg_large",
  "control_mode": "slew",
  "reason": "Executing retarget maneuver, large negative pitch burn."
}
```

### Observation Space

Every step, the agent receives a rich observation that makes the long-horizon structure explicit:

- `active_directive` — natural-language instruction for the current phase
- `pending_directives_count` — how many phases remain
- `milestones_completed` — what has already been verified
- `anomaly_flags` — active anomaly conditions
- `fuel_reserve_target` — fuel the agent must preserve for future phases
- `phase_deadline_step` — when the current directive expires
- `reward_breakdown` — per-component reward, surfaced live

### Task Curriculum

Four tasks, in order of difficulty:

| Task | Difficulty | Steps | Key Challenge |
|---|---|---|---|
| `detumble_satellite` | Easy | 120 | Stabilize a tumbling spacecraft |
| `retarget_180_flip` | Medium | 200 | Delayed maneuver window + large flip |
| `long_horizon_precision_hold` | Hard | 280 | Fine-pointing under long disturbance |
| `mission_ops_long_horizon` | Hard | 360 | All five phases, chained, with anomaly |

The flagship task `mission_ops_long_horizon` is the main benchmark. It chains all five mission phases into a single 360-step episode, with timed directives, fuel reserve targets per phase, and a `gyro_bias_spike` anomaly injected mid-mission.

---

## Reward Design: Anti-Hacking from the Start

The reward signal is where most benchmark environments go wrong. A single scalar `reward = pointing_accuracy` invites the model to ignore fuel, skip anomaly recovery, and spam actions. We designed the reward as a six-component rubric, each term targeting a specific behavior:

| Component | What it measures |
|---|---|
| `physical_tracking` | Pointing accuracy + hold streak bonus − stability penalty − overshoot penalty |
| `fuel_discipline` | Per-step fuel cost penalty + reserve-gap penalty |
| `milestone_completion` | +0.35 on verified directive completion |
| `control_mode` | +0.12 if declared mode matches recommended, −0.08 otherwise |
| `anomaly_recovery` | Bonus for error+rate improvement during active anomaly |
| `anti_stall_penalty` | Penalty for consecutive steps without meaningful progress |

These are logged per step in `reward_breakdown` and accumulated in `state.reward_columns`. Judges can see not just that total reward improved, but *which behaviors* drove the improvement.

The rule: **do not collapse this to a single signal.** That is the anti-reward-hacking story.

---

## Training Pipeline: SFT → GRPO

### Why Two Stages?

A raw `Qwen2.5-1.5B-Instruct` model has no idea what `fire_pitch_neg_large` means, and it certainly does not know to output a bare JSON object with no prose. SFT primes the format. GRPO teaches the strategy.

**Stage 1 — SFT:** Run the tuned PD controller (a hand-tuned heuristic) across all four tasks to generate seed trajectories (`training/data/seed_trajectories.jsonl`). Train the model on these expert demonstrations for 40–80 steps using QLoRA (r=16, Unsloth `FastLanguageModel`). Goal: teach the model to produce valid JSON with correct enum values. SFT loss dropped from **2.33 → 0.55**; accuracy rose from **0.53 → 0.80** in 139 seconds on an L4 GPU.

**Stage 2 — GRPO:** Use the real environment as the verifier. Five independent reward functions replace the single SFT loss signal:

| GRPO Reward Function | Signal |
|---|---|
| `reward_format` | Strict JSON parse + valid `action_type` + valid `control_mode` + non-empty reason |
| `reward_env_step` | Replay history into a fresh env, score the candidate action's actual next-step reward |
| `reward_mode_match` | `control_mode` ∈ task-recommended modes for current step |
| `reward_anti_spam` | Penalty if same action appears ≥ 4× in the last 7 steps |
| `reward_fuel_discipline` | Penalize large burns when fuel is low; reward idle/safe-hold |

`reward_env_step` is the heart of the verifier: it resets a fresh `OrbitalThrusterEnvironment`, replays the full action history from the training record, then evaluates the candidate action — giving the model real physics-backed feedback on whether its next action was good or bad.

After 60 GRPO steps, total reward rose from **0.84 → 2.30**. `reward_format` converged to **1.0** — perfect JSON on every generation.

---

## Difficulties Faced (and How We Solved Them)

### 1. Unsloth's `model.load_adapter()` is broken for GRPO warm-starting

The natural way to warm-start GRPO from an SFT adapter in PEFT is `model.load_adapter(path)`. With Unsloth's patched model, this raises dtype mismatches and silent key errors.

**Solution:** Load the raw safetensors file manually, cast each weight tensor to the model's current dtype, and call `model.load_state_dict(fixed, strict=False)`.

```python
from safetensors.torch import load_file

adapter_file = next(SFT_OUTPUT_DIR.glob("**/adapter_model.safetensors"), None)
raw_state = load_file(str(adapter_file))
target_state = model.state_dict()
fixed = {k: v.to(target_state[k].dtype) for k, v in raw_state.items() if k in target_state}
model.load_state_dict(fixed, strict=False)
```

If this still fails due to architecture drift between SFT and GRPO LoRA configs, set `ORBITAL_SKIP_SFT_WARMUP=1` to skip the overlay entirely.

### 2. `import unsloth` must come before `import transformers`

Unsloth patches transformers at import time. If any other module has already imported transformers before `import unsloth`, the patches do not apply and you get either degraded performance or subtle errors with no obvious traceback.

**Solution:** `import unsloth` is the first import in both `qwen3_smoke_sft.py` and `qwen3_grpo_train.py`, before any other import. This is enforced by a `# noqa: F401 must precede transformers` comment.

### 3. OpenEnv auto-generates a `GET /state` route that conflicts with our custom state

The `openenv.core.env_server.create_app` factory auto-registers a `/state` endpoint that returns a minimal generic response — not the full `EnvState.model_dump()` we need. The validation script checks that `/state` returns our rubric columns and milestone data.

**Solution:** After calling `create_app`, iterate `app.routes` and remove the auto-generated `GET /state` route, then register our own.

```python
app.routes[:] = [r for r in app.routes if not (getattr(r, "path", "") == "/state" and "GET" in getattr(r, "methods", set()))]

@app.get("/state")
async def get_state():
    return env_instance.state().model_dump()
```

### 4. Single-session environment and the `reset_hard` workaround

OpenEnv's `SUPPORTS_CONCURRENT_SESSIONS = False` means a single environment instance handles all calls. But `reset()` with a different `task_id` does not always cleanly rebuild all state — previous episode's anomaly flags and milestone sets can bleed through.

**Solution:** Added `POST /reset_hard` to the FastAPI app. This increments a generation counter and constructs a new `OrbitalThrusterEnvironment()` instance from scratch, guaranteeing a clean slate regardless of how the previous episode ended.

### 5. Reward hacking via action spam

During early GRPO experiments, the model discovered it could achieve decent `physical_tracking` reward by hammering a single thruster action indefinitely — which tanks fuel and stalls real progress.

**Solution:** Two layers of defense:
- `reward_anti_spam` in GRPO penalizes ≥4 identical actions in a 7-step window.
- `anti_stall_penalty` in the environment's step-level reward accumulates as a penalty for consecutive stall steps.

This combination forces the model to vary its actions and actually make progress.

### 6. Training on Windows (Unsloth is Linux/CUDA-only)

Unsloth requires Linux and a CUDA GPU. Local development was done on Windows.

**Solution:** `training/local_train.py` is a vanilla TRL + PEFT + bitsandbytes fallback that runs without Unsloth. For actual training runs, we used HF Jobs (`hf jobs uv run --flavor l4x1`) which spins up an ephemeral L4 GPU on Linux with a UV environment specified by the PEP 723 deps block at the top of `training/hf_job_train.py`.

---

## Results

Training ran on HF Jobs (L4 GPU, `Qwen/Qwen2.5-1.5B-Instruct`, 40 SFT + 60 GRPO steps):

**SFT:** loss 2.33 → 0.55, accuracy 0.53 → 0.80 (139s)

**GRPO:** loss 0.077 → 0.037, total reward 0.84 → 2.30 (287s), `reward_format` → 1.0

| Policy | Easy (detumble) | Medium (retarget) | Hard (hold) | Flagship |
|---|---|---|---|---|
| Random | 23.9 / fail | 3.2 / fail | −25.3 / fail | −53.5 / fail |
| Deterministic PD | 17.6 / pass | 97.4 / fail | 21.1 / fail | 89.8 / fail |
| Tuned PD | 34.2 / pass | 120.1 / pass | 27.5 / fail | 115.8 / fail |
| **Trained (GRPO)** | 9.2 | 38.3 | **88.0** | 22.6 |

The trained model learned perfect output format (reward_format = 1.0) and strongly conservative fuel strategy (fuel_used ≈ 0), which explains both the high hard-task score and the relatively modest easy/medium scores — a conservative model preserves fuel but moves slowly. With more GRPO steps and a curriculum-weighted dataset, the milestone completion signal would close the gap.

---

## What We Learned

**Decomposed reward > opaque scalar.** The six-component rubric made it immediately obvious during debugging why reward was or wasn't improving. When `milestone_completion_reward` stayed flat while `fuel_discipline_reward` climbed, we knew the model had learned to idle conservatively but hadn't yet learned to commit to a maneuver.

**SFT priming is not optional for structured outputs.** Without the SFT stage, GRPO spends most of its budget learning to produce valid JSON at all, leaving nothing for learning spacecraft control. Forty steps of format priming saved hundreds of GRPO steps.

**The verifier signal (`reward_env_step`) is the most powerful reward function.** It replays real physics. No heuristic approximation can substitute for this when the task involves genuine physics-driven state transitions.

**Long-horizon tasks expose shortcut-taking immediately.** A controller optimizing only the next-step reward burns fuel early and fails the retarget phase. The fuel reserve targets per phase — embedded in the observation as `fuel_reserve_target` — give the model the information it needs to plan ahead, but only a controller that actually uses that signal will pass the flagship mission.

---

## Links

- **HF Space (live env):** https://huggingface.co/spaces/pixxel-phantom/orbital-thruster-env
- **Trained adapter (GRPO LoRA, 1.5B):** https://huggingface.co/pixxel-phantom/orbital-thruster-grpo-fast
- **Trained adapter (GRPO LoRA, 4B):** https://huggingface.co/pixxel-phantom/orbital-thruster-grpo

