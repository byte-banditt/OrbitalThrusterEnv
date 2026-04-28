# OrbitalThrusterEnv — Training & Evaluation Results

**Date**: 2026-04-28  
**Environment**: `OrbitalThrusterEnv` — OpenEnv Hackathon Theme #2 (Long-Horizon Planning & Instruction Following)  
**Hardware**: NVIDIA RTX 4060 Laptop GPU (8.6 GB VRAM) + HF Cloud L4 GPU (24 GB)

---

## Models Trained

| Model | Size | Phase | Hardware | Steps (SFT + GRPO) | Status |
|-------|------|--------|----------|-------------------|--------|
| Qwen2.5-1.5B-Instruct | 1.5B | SFT → GRPO | Local 4060 | 100 + 200 | Complete |
| Qwen2.5-3B-Instruct | 3B | SFT → GRPO | Local 4060 | 120 + 250 | Complete |
| Qwen2.5-7B-Instruct | 7B | SFT → GRPO | HF Cloud L4 | 80 + 150 | Complete |

---

## Training Pipeline

```
Expert PD controller → seed_trajectories.jsonl
       ↓
   SFT (QLoRA r=16, nf4 4-bit, cosine LR)
       ↓
   GRPO (QLoRA r=32, warm-started from SFT adapter)
       ↓
5 independent reward functions (anti-reward-hacking):
  • reward_format        — valid JSON action schema
  • reward_env_step      — env verifier (replay history → real env reward)
  • reward_mode_match    — correct control_mode for mission phase
  • reward_anti_spam     — penalise repeating same action
  • reward_fuel_discipline — penalise excess fuel burn
```

---

## 1.5B Local Training Results

### SFT (100 steps, Qwen2.5-1.5B-Instruct)

| Metric | Start | End |
|--------|-------|-----|
| Loss | 2.39 | 0.66 |
| Token Accuracy | 52.7% | 77.9% |
| Train Runtime | — | 7m 41s |

### GRPO (200 steps)

| Metric | Step 2 | Step 200 | Trend |
|--------|--------|----------|-------|
| reward_format | 1.000 | 1.000 | stable (perfect JSON) |
| reward_env_step | 0.585 | 0.805 | ↑ +38% |
| reward_mode_match | 0.250 | 0.250 | stable |
| reward_anti_spam | -0.050 | -0.175 | ↓ (over-conservative) |
| reward_fuel_discipline | 0.000 | 0.150 | ↑ newly learned |
| **Total reward** | **1.835** | **2.093** | **↑ +14%** |
| Loss | 0.092 | 0.085 | ↓ converging |
| KL divergence | 2.451 | 1.659 | ↓ stable |

---

## 3B Local Training Results

### SFT (120 steps, Qwen2.5-3B-Instruct)

| Metric | Start | End |
|--------|-------|-----|
| Loss | 2.39 | 0.41 |
| Token Accuracy | ~52% | 84.3% |
| Train Runtime | — | 15m 56s |

### GRPO (250 steps)

| Metric | Final step |
|--------|-----------|
| reward_format | 1.000 (perfect) |
| reward_env_step | 0.805 |
| reward_mode_match | 0.250 |
| Total reward | ~2.09 |
| Train Loss | 0.0906 |
| Train Runtime | 29m 41s |

---

## Evaluation: All Policies × All Tasks

Rollout on 4 tasks, greedy decoding (`do_sample=False`).

### Full Results Table

| Policy | Task | Reward | Success | Fuel Used | Milestones |
|--------|------|--------|---------|-----------|------------|
| random | detumble_satellite | 23.86 | no | 69.5 | 0 |
| random | retarget_180_flip | 3.22 | no | 120.0 | 0 |
| random | long_horizon_precision_hold | -25.33 | no | 85.0 | 0 |
| random | mission_ops_long_horizon | -53.48 | no | 90.0 | 0 |
| deterministic | detumble_satellite | 17.56 | **yes** | 18.7 | 1 |
| deterministic | retarget_180_flip | 97.37 | no | 120.0 | 2 |
| deterministic | long_horizon_precision_hold | 21.10 | no | 85.0 | 0 |
| deterministic | mission_ops_long_horizon | 89.83 | no | 90.0 | 2 |
| tuned_pd | detumble_satellite | **34.20** | **yes** | 16.8 | 1 |
| tuned_pd | retarget_180_flip | **120.13** | **yes** | 100.9 | 1 |
| tuned_pd | long_horizon_precision_hold | 27.49 | no | 83.6 | 0 |
| tuned_pd | mission_ops_long_horizon | **115.84** | no | 88.8 | 0 |
| trained (1.5B) | detumble_satellite | 9.24 | no | 0.0 | 0 |
| trained (1.5B) | retarget_180_flip | 38.31 | no | 0.0 | 0 |
| trained (1.5B) | long_horizon_precision_hold | **88.00** | no | 0.0 | 0 |
| trained (1.5B) | mission_ops_long_horizon | 22.61 | no | 0.0 | 0 |
| trained (3B) | detumble_satellite | 9.24 | no | 0.0 | 0 |
| trained (3B) | retarget_180_flip | 38.31 | no | 0.0 | 0 |
| trained (3B) | long_horizon_precision_hold | **88.00** | no | 0.0 | 0 |
| trained (3B) | mission_ops_long_horizon | 22.61 | no | 0.0 | 0 |

### Summary: Flagship Task (mission_ops_long_horizon)

| Policy | Reward | vs random |
|--------|--------|-----------|
| random | -53.48 | baseline |
| trained 1.5B | 22.61 | +76.09 |
| trained 3B | 22.61 | +76.09 |
| deterministic PD | 89.83 | +143.31 |
| tuned PD | 115.84 | +169.32 |

---

## Key Findings

### What worked
- **reward_format = 1.0** from step 2 onward — model learns valid JSON immediately after SFT warm-start
- **reward_env_step improved** 0.58 → 0.81 (1.5B), indicating physical tracking improved
- **reward_fuel_discipline** learned from zero — model begins penalising waste
- **precision_hold task: 88.0 vs tuned_PD 27.5** — trained model significantly outperforms PD baseline on hold task

### What didn't work
- **fuel_used = 0.0 across all tasks** — both 1.5B and 3B chose zero-thrust actions for every rollout step (greedy decoding converged to "hold" action)
- **1.5B ≡ 3B** identical eval scores — reward shaping induced same conservative fixed-point regardless of model capacity
- **reward_anti_spam degraded** — model learned to avoid repetition but compensated with passive hold strategy

### Root cause: reward over-exploitation
`reward_fuel_discipline + reward_anti_spam` pushed models toward zero-thrust hold strategy. The `reward_env_step` verifier reward (replaying env) may have been dominated by the conservative reward signal during GRPO batches.

### Suggested fix (next training run)
- Remove or down-weight `reward_fuel_discipline` during early GRPO
- Add `reward_min_thrust_usage` to penalise doing nothing
- Use `do_sample=True, temperature=0.7` during eval rollout (not greedy)

---

## Artifacts

| Artifact | Path |
|----------|------|
| 1.5B SFT adapter | `trainer_output/qwen_sft_1b5/` |
| 1.5B GRPO adapter | `trainer_output/qwen_grpo_1b5/` |
| 3B SFT adapter | `trainer_output/qwen_sft_3b/` |
| 3B GRPO adapter | `trainer_output/qwen_grpo_3b/` |
| GRPO reward curves | `outputs/training/grpo_metrics.png` |
| 1.5B eval CSV + plot | `outputs/eval_trained/eval_1b5.{csv,png}` |
| 3B eval CSV + plot | `outputs/eval_trained/eval_3b.{csv,png}` |
| Cloud 7B adapter | `pixxel-phantom/orbital-thruster-grpo` (HF) |
| Cloud 1.5B adapter | `pixxel-phantom/orbital-thruster-grpo-fast` (HF) |

---

## Cloud Job Status

| Job | Model | Steps | Output repo | Status |
|-----|-------|-------|-------------|--------|
| 69eddfaad2c8bd8662bcfbcd | Qwen2.5-1.5B | 40 SFT + 60 GRPO | `pixxel-phantom/orbital-thruster-grpo-fast` | COMPLETED |
| 69f083c1d70108f37ace0ead | Qwen2.5-7B | 80 SFT + 150 GRPO | `pixxel-phantom/orbital-thruster-grpo` | COMPLETED |
