# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`OrbitalThrusterEnv` — an OpenEnv-compliant FastAPI environment for **OpenEnv Hackathon Theme #2 (Long-Horizon Planning & Instruction Following)**, plus a TRL+Unsloth training pipeline. The agent (an LLM) controls a spacecraft through a 5-phase mission (`detumble → coast → retarget → anomaly recovery → precision hold`) using 13 discrete thruster actions and a required `control_mode` declaration. Submission is the HF Space `pixxel-phantom/orbital-thruster-env`.

## Run

```bash
# Local server
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
python validate.py                 # 22+ checks against /health, /reset, /step, /state

# Tests
pytest tests/ -q
pytest tests/test_mission_ops_env.py::test_flagship_task_has_long_horizon_directives_and_anomalies -q

# Baseline rollout (random / deterministic-PD / tuned-PD)
python training/evaluate_baselines.py

# Full training (laptop or cloud)
python training/qwen3_smoke_sft.py      # SFT 80 steps QLoRA (Unsloth)
python training/qwen3_grpo_train.py     # GRPO 300 steps with 5 reward funcs
python training/eval_trained_model.py   # trained vs baselines

# Cloud training (HF Jobs, requires token with jobs.write)
hf jobs uv run --flavor l4x1 --timeout 2h --secrets HF_TOKEN \
  -e ORBITAL_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  -e ORBITAL_SFT_STEPS=40 -e ORBITAL_GRPO_STEPS=80 -e ORBITAL_NUM_GEN=4 \
  -e ORBITAL_SKIP_SFT_WARMUP=1 \
  -e OUTPUT_REPO=pixxel-phantom/orbital-thruster-grpo-fast \
  -d training/hf_job_train.py

# Push to HF Space (uses huggingface_hub, ignores venvs/caches)
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='.', repo_id='pixxel-phantom/orbital-thruster-env', repo_type='space', ignore_patterns=['venv*/**','**/__pycache__/**','.git/**','trainer_output/**','*.pyc'])"
```

## Architecture

### Server (OpenEnv-compliant FastAPI)
- `server/app.py` — wires `OrbitalThrusterEnvironment` into `openenv.core.env_server.create_app`. Strips the auto-generated `GET /state` route and re-adds a custom one returning `EnvState.model_dump()`. Exposes `POST /reset_hard` to bump generation and rebuild the env (workaround for single-session env).
- `server/orbital_thruster_environment.py` — `Environment[Action, Observation, EnvState]` impl. Holds the entire mission state in instance fields (rates, attitude, fuel, milestones, anomaly flags). `step()` flow: propagate dynamics → compute errors → detect overshoot → update on-target streak → check directive milestone → score reward → assemble observation. **`SUPPORTS_CONCURRENT_SESSIONS = False`** — only one episode at a time; the FastAPI app is built with `max_concurrent_envs=1`.
- `server/dynamics.py` — pure-Python rigid-body dynamics with seeded sinusoidal disturbances. `propagate()` is the physics step; `signed_angle_error()` and `vector_magnitude()` are reused everywhere.
- `server/reward.py` — `RewardScorer` returns a single scalar reward **and** a 6-key rubric (`physical_tracking`, `fuel_discipline`, `milestone_completion`, `control_mode`, `anomaly_recovery`, `anti_stall_penalty`). `score_episode()` and `is_success()` are separate, difficulty-weighted formulas. **Do not modify reward logic** unless absolutely necessary — judging weight depends on it being interpretable.
- `server/tasks/` — one `MissionTask` per file (`task_easy`, `task_medium`, `task_hard`, `task_flagship`). `task_flagship.py` is the headline 360-step `mission_ops_long_horizon` task with 5 timed directives, fuel reserve targets per phase, and a `gyro_bias_spike` anomaly. Tasks register in `server/tasks/__init__.py::TASK_REGISTRY`.
- `models.py` — Pydantic schemas for `OrbitalThrusterAction` (action_type ∈ 13 enums, `control_mode` ∈ 7 enums, optional reason) and `OrbitalThrusterObservation`. **Schema is the contract** — judges validate the env via `validate.py`.

### Inference / agent contract
- `inference.py` — single source of truth for valid actions/control modes (`VALID_ACTIONS`, `VALID_CONTROL_MODES`), the system prompt, and two non-LLM controllers (`deterministic_controller`, `tuned_mission_controller` — these are the PD baselines and the LLM fallback). `choose_action()` posts to an OpenAI-compatible LLM (HF Router by default) and falls back to the tuned PD if the LLM call fails or produces invalid JSON.

### Training pipeline
- `training/common.py` — shared utilities. `collect_seed_records()` runs the tuned-PD expert and writes `training/data/seed_trajectories.jsonl`; `build_prompt()` is the prompt format used by both SFT and GRPO; `parse_action_json()` validates LLM JSON output against the action schema. Adds `ROOT` (repo root) to `sys.path` on import.
- `training/rl_utils.py` — **5 independent reward functions** (`reward_format`, `reward_env_step`, `reward_mode_match`, `reward_anti_spam`, `reward_fuel_discipline`) consumed by `GRPOTrainer.reward_funcs=[...]`. `reward_env_step` replays the prompt's `history_actions` into a fresh `OrbitalThrusterEnvironment` and scores the candidate action's next-step reward — this is the verifier signal. Also exports `make_lora_controller()` (PEFT model wrapped as a controller for eval), `RewardCSVLogger` (TrainerCallback that writes `outputs/training/grpo_metrics.csv`), and `plot_training_curves()`.
- `training/qwen3_smoke_sft.py` — Unsloth `FastLanguageModel` + TRL `SFTTrainer`, QLoRA r=16, JSON-format priming on tuned-PD traces. Reads `ORBITAL_SFT_STEPS` env var (default 80). **`import unsloth` MUST precede transformers** or patches don't apply.
- `training/qwen3_grpo_train.py` — Unsloth + TRL `GRPOTrainer`. Loads SFT adapter via `safetensors.load_file` + `model.load_state_dict(strict=False)` with dtype-cast to match the model's half precision (the standard `model.load_adapter()` path is broken with Unsloth). Set `ORBITAL_SKIP_SFT_WARMUP=1` to skip the overlay if it produces dtype/key mismatches. Reads `ORBITAL_GRPO_STEPS` (default 300) and `ORBITAL_NUM_GEN` (default 6).
- `training/hf_job_train.py` — UV-script entrypoint for `hf jobs uv run`. PEP 723 deps block at top (`torch>=2.6.0`, `unsloth>=2025.10.1`, etc.). Downloads the HF Space repo via `snapshot_download`, runs SFT then GRPO then eval, uploads `trainer_output/` + `outputs/` to `OUTPUT_REPO` as a model repo with auto-generated README.
- `training/eval_trained_model.py` — rolls trained adapter as a controller across all 4 tasks, compares vs random/deterministic/tuned-PD, writes `outputs/eval_trained/trained_vs_baseline.{csv,png}`.
- `training/local_train.py` — vanilla TRL+peft+bnb fallback (no Unsloth) for Windows. Defaults to `Qwen/Qwen2.5-1.5B-Instruct`. Use only if Unsloth import fails locally.

### Output conventions
- `trainer_output/qwen_sft/` — SFT LoRA adapter
- `trainer_output/qwen_grpo/` — final GRPO LoRA adapter
- `outputs/baseline_eval/baseline_summary.{csv,png}` — random/det/tuned-PD baselines
- `outputs/training/grpo_metrics.{csv,png}` — per-component reward + loss curves
- `outputs/eval_trained/trained_vs_baseline.{csv,png}` — final comparison

## Hard rules

- **Don't modify the reward function or task definitions** unless fixing a real bug. The rubric design is a judging signal; tampering invalidates the multi-component anti-reward-hacking story.
- **Don't reinvent the env API.** It must remain OpenEnv-compliant (`reset`, `step`, `state`, action/observation Pydantic models). `validate.py` enforces 22+ contract checks — if it fails, the submission is invalid.
- **Save LoRA adapters only**. Never naively merge a 4-bit model to 16-bit and re-save — it damages weights (per OpenEnv guide §16). The training scripts already save adapter-only.
- **Curriculum + 5-reward design is the anti-hacking story.** Don't collapse to a single reward signal — judges score 20% on showing improvement *without* reward hacking.
- **The flagship task is `mission_ops_long_horizon`** — all demo plots and the README pitch are framed around it. Other 3 tasks are curriculum/sanity-check.

## Submission state

- HF Space: `pixxel-phantom/orbital-thruster-env` (live, Docker SDK)
- Trained adapter: `pixxel-phantom/orbital-thruster-grpo-fast` (1.5B, deadline run) and `pixxel-phantom/orbital-thruster-grpo` (4B, post-deadline)
- README has placeholder slots for blog/video URLs at the top — fill before submitting
- Submission requirements (themes.md): OpenEnv ✓, training script ✓, loss+reward plots from real run (auto-generated post-training), <2 min video OR HF blog (manual), Space URL ✓, README that motivates + shows results
