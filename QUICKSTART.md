# Quick Start Guide

**MLOps Pipeline Debugger** for Meta OpenEnv Hackathon

## 30-Second Setup

```bash
# 1. Install dependencies
pip install fastapi uvicorn pydantic openai httpx python-dotenv

# 2. Set environment variable
export HF_TOKEN=your_huggingface_token_here

# 3. Run the complete pipeline
python inference.py
```

That's it! The script will:
- Start the FastAPI server
- Wait for it to be ready
- Run all 3 tasks with an LLM agent
- Print results

## What You'll See

```
[START] task=schema_drift env=mlops-pipeline-debugger model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=check_feature_stats reward=0.15 done=false error=null
[STEP] step=2 action=fix_schema reward=0.35 done=false error=null
[STEP] step=3 action=run_eval_set reward=0.25 done=true error=null
[END] success=true steps=3 score=0.85 rewards=[0.15,0.35,0.25]

[START] task=concept_drift ...
...
```

## Local Development

### Start Server Only

```bash
cd server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Interactive API Testing

```bash
# In separate terminal:
# 1. View API docs
open http://localhost:8000/docs

# 2. Or make HTTP requests
curl -X POST http://localhost:8000/reset?task_id=schema_drift

# 3. Get state
curl http://localhost:8000/state
```

### Manual Test (Python)

```python
import httpx

# Reset
r = httpx.post("http://localhost:8000/reset", params={"task_id": "schema_drift"})
obs = r.json()
print(f"Initial accuracy: {obs['model_accuracy']}")

# Take a step
action = {"action": "check_feature_stats", "parameters": {}}
r = httpx.post("http://localhost:8000/step", json=action)
result = r.json()
print(f"Reward: {result['reward']}")
```

## Docker

```bash
# Build
docker build -t mlops-debugger .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_xxxx \
  mlops-debugger
```

## Environment Variables

| Variable | Value | Required |
|----------|-------|----------|
| `HF_TOKEN` | Your Hugging Face API token | ✅ Yes |
| `API_BASE_URL` | (default: HF router) | ❌ No |
| `MODEL_NAME` | (default: Qwen2.5-72B) | ❌ No |

## Troubleshooting

### ModuleNotFoundError

```bash
# Make sure you're using the right Python
which python3
which python

# Reinstall dependencies
pip install --upgrade pydantic fastapi uvicorn
```

### Connection Refused

```bash
# Server might not be started yet
# inference.py handles this automatically

# Or start manually:
cd server && uvicorn main:app --port 8000
```

### LLM API Errors

```bash
# Verify token
echo $HF_TOKEN

# Test connectivity
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://router.huggingface.co/v1/models
```

## Project Structure

```
mlops-pipeline-debugger/
├── server/                     # FastAPI server
│   ├── main.py                # (endpoints)
│   ├── env.py                 # (state machine)
│   ├── models.py              # (Pydantic models)
│   ├── tasks.py               # (3 tasks)
│   └── graders.py             # (scoring)
├── inference.py               # ← Hackathon evaluates this!
├── openenv.yaml               # (metadata)
└── README.md                  # (full docs)
```

## Scoring

- **schema_drift** (easy, 8 steps): Identify and fix schema type change
- **concept_drift** (medium, 12 steps): Detect drift and retrain
- **gpu_nan_failure** (hard, 15 steps): Complex GPU recovery

Each task scores 0.0-1.0 based on:
- Root cause identified (0.3)
- Fix applied correctly (0.4)
- Accuracy restored (0.3)

**Total possible: 3.0 (average per task: 1.0)**

## Expected Performance

Using Qwen/Qwen2.5-72B-Instruct:
- schema_drift: ~0.75-0.85
- concept_drift: ~0.65-0.75
- gpu_nan_failure: ~0.55-0.70
- **Average: ~0.65-0.75**

## Files to Review

1. **inference.py** - Main entry point (what gets evaluated)
2. **README.md** - Full documentation
3. **openenv.yaml** - OpenEnv specification
4. **server/env.py** - Environment state machine (core logic)

---

**Ready to submit! 🚀**
