# Training Guide

## Files
- `train_orbital_grpo.ipynb` — main notebook. Runs full SFT → GRPO → eval.
- `rl_utils.py` — multi-component reward funcs, curriculum filter, LoRA-controller wrapper, CSV logger, plot helper.
- `qwen3_smoke_sft.py` — QLoRA SFT (80 steps) on tuned-PD expert traces. Locks JSON format + control-mode vocabulary.
- `qwen3_grpo_train.py` — GRPO (300 steps) with 5 independent rewards, warm-started from SFT adapter.
- `eval_trained_model.py` — rolls out trained adapter on all 4 tasks, compares vs random / deterministic / tuned-PD.
- `evaluate_baselines.py` — fixed-seed baseline rollout.
- `generate_seed_trajectories.py` — emits expert traces from tuned PD.
- `requirements.txt` — training-only deps (kept out of Space runtime).

## Model
Default: `Qwen/Qwen2.5-3B-Instruct` — fits 4060 laptop (8 GB VRAM) at 4-bit, mature Unsloth/TRL recipe, strong JSON adherence.

Override on Hugging Face credits:
```bash
export ORBITAL_BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

## Run order
```bash
pip install -r training/requirements.txt
python training/generate_seed_trajectories.py     # one-time
python training/evaluate_baselines.py             # baseline CSV+PNG
python training/qwen3_smoke_sft.py                # SFT warm-start
python training/qwen3_grpo_train.py               # GRPO (logs metrics CSV+PNG)
python training/eval_trained_model.py             # trained vs baseline CSV+PNG
```

Or run `training/train_orbital_grpo.ipynb` end-to-end.

## Outputs
- `trainer_output/qwen_sft/` — SFT LoRA adapter
- `trainer_output/qwen_grpo/` — GRPO LoRA adapter (final)
- `outputs/baseline_eval/baseline_summary.{csv,png}` — baseline policies
- `outputs/training/grpo_metrics.{csv,png}` — reward + loss curves per component
- `outputs/training/sample_rollout_flagship.json` — qualitative trace
- `outputs/eval_trained/trained_vs_baseline.{csv,png}` — final comparison

## Reward components (independent, summed by GRPO)
| Function | Range | Signal |
|---|---|---|
| `reward_format` | -1.0 → 1.0 | strict JSON parse + valid enums + reason field |
| `reward_env_step` | -1.0 → 1.0 | replays history, scores candidate via real env reward |
| `reward_mode_match` | -0.2 → 0.25 | control_mode ∈ recommended for active directive |
| `reward_anti_spam` | -0.4 → 0.05 | penalty if same action ≥ 4× in last 7 steps |
| `reward_fuel_discipline` | -0.3 → 0.15 | low-fuel→idle bonus, low-fuel→large-pulse penalty |

Five independent signals reduce reward-hackability per OpenEnv guide §8/§9.

## Hyperparameters
SFT: r=16 α=16, lr=2e-4, cosine, batch 1×grad_accum 8, 80 steps, max_seq 2048.
GRPO: r=32 α=32, lr=5e-6, cosine, batch 1×grad_accum 8, 300 steps, num_generations=6, max_completion=96, T=0.9, top_p=0.95.

## Anti-reward-hacking checks
Notebook §8 asserts `idle_fraction < 0.85` and `top_action_share < 0.85` on flagship rollout — catches all-idle / single-action exploits before claiming success.

## Save warning
Adapter saved as LoRA only. Do NOT upcast 4-bit base then merge naively (per OpenEnv guide §16) — it damages weights.
