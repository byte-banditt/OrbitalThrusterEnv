# OrbitalThrusterEnv — GRPO LoRA adapter

Base model: `Qwen/Qwen2.5-1.5B-Instruct`

Source env: https://huggingface.co/spaces/pixxel-phantom/orbital-thruster-env

Trained via TRL `GRPOTrainer` + Unsloth on OpenEnv `OrbitalThrusterEnv` flagship task `mission_ops_long_horizon`.
5 independent reward funcs (format, env-step, mode-match, anti-spam, fuel-discipline) for anti-reward-hacking.

## Artifacts
- `trainer_output/qwen_grpo/` — final LoRA adapter
- `trainer_output/qwen_sft/` — SFT warm-start adapter
- `outputs/training/grpo_metrics.png` — reward + loss curves
- `outputs/eval_trained/trained_vs_baseline.png` — trained vs baselines on 4 tasks
