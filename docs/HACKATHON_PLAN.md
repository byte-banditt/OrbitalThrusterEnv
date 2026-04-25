# Orbital Thruster Hackathon Plan

This file follows `final.md` execution order and is grounded in the current repository state.

## Step 0: Repo Map

### `server/orbital_thruster_environment.py`
- `reset(task_id=...)`
  - Initializes task config, seeded disturbance coefficients, state counters, fuel, and reward accumulators.
  - Sets initial attitude/rates from task definition and target from phase schedule.
  - Returns a structured observation with `reward=0.0`.
- `step(action)`
  - Parses and applies one discrete thruster action through deterministic dynamics (`server/dynamics.py`).
  - Updates attitude, rates, fuel, overshoot bookkeeping, hold streak, and cumulative reward.
  - Computes per-step reward through `RewardScorer.compute(...)`.
  - Sets `done` on early success (easy/medium) or step budget exhaustion.
- `state` property
  - Returns `EnvState` with `episode_id`, `step_count`, `steps_used`, fuel stats, cumulative reward, and done flag.
- Terminal conditions
  - Early success when `_success(...)` is true and `early_success_allowed` is true.
  - Otherwise done when `step_count >= step_budget`.

### `server/reward.py`
- Reward logic lives here (`RewardScorer`):
  - Per-step dense reward: `pointing + hold_bonus - fuel_penalty - stability_penalty - overshoot_penalty`, clamped to `[-1, 1]`.
  - Episode score: weighted composition by task difficulty.
  - Success criteria: strict threshold checks on final error/rates/fuel + difficulty-specific constraints.

### `server/app.py`
- OpenEnv FastAPI server via `create_app(...)`.
- Adds `/`, `/tasks`, `/reset_hard`.
- Uses `OrbitalThrusterEnvironment` and typed `models.py`.

### `client.py`
- `OrbitalThrusterEnv` extends `EnvClient`.
- Only speaks HTTP payload/response contract; no server internals imported.

### `openenv.yaml`
- Present and valid.
- Name/version/runtime/action/observation/task metadata are complete.

### Packaging
- `pyproject.toml`, `Dockerfile`, `requirements.txt`, `server/requirements.txt` present and coherent.

### README
- Present, with environment summary, tasks, API, and local run instructions.

## Step 1: Theme Scoring

| Theme | Score (0-10) | Reasoning |
| --- | --- | --- |
| Multi-Agent Interactions | 2 | Environment is single-controller spacecraft attitude task; no direct agent-agent interaction. |
| Long-Horizon Planning & Instruction Following | 9 | Hard task requires hundreds of constrained steps with fuel/rate/error tradeoffs and explicit action schema. |
| World Modeling - Professional Tasks | 10 | Mission-ops spacecraft control is professional, safety-constrained, and directly measurable. |
| World Modeling - Personalized Tasks | 1 | Domain is operational aerospace, not personal preference modeling. |
| Self-Improvement | 4 | Possible via meta-policy adaptation, but environment not designed for self-edit loops. |
| Wild Card | 7 | Spacecraft control is novel and judge-memorable, but best framed under professional world modeling. |

### Problem Statement

- Theme: **World Modeling - Professional Tasks**
- One sentence: Current LLMs struggle to execute fuel-aware closed-loop control under disturbance, despite strong offline reasoning.
- Agent does: Observes structured telemetry, decides one discrete thruster action per step, applies correction while minimizing overshoot/rate/fuel waste.
- Success verified by: Programmatic task-specific thresholds on final axis error, angular rate, fuel reserve/budget, hold streak, and on-target fraction.
- Why judges will remember this: It turns "LLM planning" into auditable mission-ops behavior with measurable physics-grounded wins.

## Step 2: Reward Audit & Upgrade Plan

### 2A: Existing Reward Audit

1. `pointing`
- Measures: current error-norm closeness to target.
- Scale: normalized to `[0, 1]`.
- Composability: yes, main positive term.
- Cheapest exploit: oscillate near threshold with low true stability to farm partial score.

2. `hold_bonus`
- Measures: consecutive on-target streak.
- Scale: `[0, 0.18]`.
- Composability: yes.
- Cheapest exploit: tiny oscillation inside tolerance window without robust damping.

3. `fuel_penalty`
- Measures: immediate fuel consumed.
- Scale: non-negative subtraction.
- Composability: yes.
- Cheapest exploit: over-idle to save fuel even when mission objective is missed.

4. `stability_penalty`
- Measures: angular-rate penalty when near target.
- Scale: non-negative subtraction.
- Composability: yes.
- Cheapest exploit: stay just outside stability trigger region to avoid penalty.

5. `overshoot_penalty` (medium/hard)
- Measures: sign-crossing overshoot while still rotating fast.
- Scale: non-negative subtraction.
- Composability: yes.
- Cheapest exploit: many small crossings below trigger thresholds.

### 2B: Multi-Signal Bundle (minimum 4)

Use this conceptual decomposition (map to current metrics):
- `reward_primary_objective` (weight 0.40): final/task correctness (hard to fake).
- `reward_process_quality` (weight 0.25): bounded rates + controlled slew profile.
- `reward_format_compliance` (weight 0.20): valid action schema + non-empty reason string.
- `reward_efficiency` (weight 0.15): fuel and step economy relative to budget.

Exploit analysis to enforce in implementation:

`reward_primary_objective`
1. Exploit: briefly hit tolerance near episode end only.
   Defense: require hold streak minimum and on-target fraction gate.
2. Exploit: optimize one axis, ignore others.
   Defense: max-axis error hard constraint.
3. Exploit: burn fuel to brute-force final snapshot.
   Defense: fuel reserve/budget constraints.

`reward_process_quality`
1. Exploit: aggressive bang-bang control with incidental final accuracy.
   Defense: overshoot accumulator + angular-rate penalties.
2. Exploit: zig-zag around target line.
   Defense: sign-crossing + streak reset logic.
3. Exploit: late stabilization only.
   Defense: rolling-window smoothness and intermediate phase checks.

`reward_format_compliance`
1. Exploit: invalid action strings parsed as defaults.
   Defense: strict enum validation, invalid action => explicit penalty.
2. Exploit: empty rationale spam.
   Defense: minimum reason length and trimmed whitespace check.
3. Exploit: JSON wrappers around junk.
   Defense: strict schema parse; reject extra/malformed fields for training rewards.

`reward_efficiency`
1. Exploit: idle forever to preserve fuel.
   Defense: efficiency only awarded when objective quality crosses threshold.
2. Exploit: one axis solved with excessive steps.
   Defense: steps-used normalization + mission-phase progress checks.
3. Exploit: fuel dump early then idle.
   Defense: per-step marginal fuel penalties + end-state reserve checks.

### 2C: OpenEnv Rubric Integration

Installed `openenv-core==0.2.3` exposes rubric primitives under:
- `openenv.core.rubrics.Rubric`
- `openenv.core.rubrics.WeightedSum`

Recommended next code move:
- Lift reward components into `Rubric` subclasses.
- Compose with `WeightedSum(weights=[0.40, 0.25, 0.20, 0.15])`.
- Keep current success gating as a hard constraint.

## Step 3: OSS LLM Options

| Model | Params | VRAM (4-bit) | Base capability for this env | Score |
| --- | --- | --- | --- | --- |
| Qwen3.5-7B-Instruct | 7B | ~5GB | Strong structured action JSON + good control prompt following | 8 |
| Qwen2.5-14B-Instruct | 14B | ~9GB | Best balance of planning quality vs cost for long-horizon control | 9 |
| Llama-3.1-8B-Instruct | 8B | ~5.5GB | Reliable fallback, strong baseline behavior | 8 |
| Llama-3.3-70B-Instruct | 70B | ~40GB | Highest raw quality, too expensive for heavy RL iterations | 6 |
| Mistral-7B-Instruct-v0.3 | 7B | ~5GB | Fast rollouts, useful for ablation speed | 7 |
| DeepSeek-R1-Distill-Qwen-7B | 7B | ~5GB | Reasoning-heavy action selection, decent fallback | 7 |

### Recommended Choice

Pick **Qwen2.5-14B-Instruct** for A100 runs, and **Qwen3.5-7B-Instruct** for T4 iteration loops.
This environment rewards long-horizon instruction adherence and clean structured outputs more than pure chat fluency.
Use `scripts/select_model.py` to choose by GPU/mode and lock model before training.

## Step 4+ Immediate Execution Checklist

1. Run `scripts/setup_deps.ps1` (and optionally `-TrainingDeps`).
2. Run `scripts/select_model.py --gpu t4 --mode balanced`.
3. Use `training/grpo_colab_template.py` to build the Colab notebook.
4. Populate `plots/reward_curve.png` and `plots/before_after.png`.
5. Fill README links (HF Space, Colab, WandB, video/blog) before submission.
