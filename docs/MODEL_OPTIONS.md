# LLM Model Options

Use this file to pick the RL base model quickly.

## Quick Picks

1. **T4 16GB, balanced:** `Qwen3.5-7B-Instruct`
2. **A100 40GB, best quality/cost:** `Qwen2.5-14B-Instruct`
3. **Need fastest iteration:** `Mistral-7B-Instruct-v0.3`
4. **Reasoning-heavy fallback:** `DeepSeek-R1-Distill-Qwen-7B`
5. **Premium final showcase only:** `Llama-3.3-70B-Instruct`

## Score Table

| Model | Params | VRAM 4-bit | Pros | Cons | Fit Score |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-14B-Instruct | 14B | ~9GB | Best long-horizon planning + instruction reliability | Slower than 7B | 9/10 |
| Qwen3.5-7B-Instruct | 7B | ~5GB | Strong structured output at low cost | Lower ceiling than 14B | 8/10 |
| Llama-3.1-8B-Instruct | 8B | ~5.5GB | Stable and well supported | Slightly weaker control precision | 8/10 |
| Mistral-7B-Instruct-v0.3 | 7B | ~5GB | Fast rollouts | Lower reward ceiling | 7/10 |
| DeepSeek-R1-Distill-Qwen-7B | 7B | ~5GB | Strong reasoning traces | May over-verbose actions | 7/10 |
| Llama-3.3-70B-Instruct | 70B | ~40GB | Highest raw quality | Expensive + slow for RL | 6/10 |

## CLI Selector

```powershell
python scripts/select_model.py --gpu t4 --mode balanced
python scripts/select_model.py --gpu a100 --mode quality
python scripts/select_model.py --gpu t4 --mode speed
```

## Decision Rule

Pick one model and freeze it for first full run.
Only switch model if 10 zero-shot rollouts return all-zero reward.
