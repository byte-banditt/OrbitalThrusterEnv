"""Task definitions for MLOps Pipeline Debugger."""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class Task:
    """Task definition."""
    id: str
    name: str
    difficulty: str
    description: str
    scenario: str
    max_steps: int
    optimal_actions: List[str]
    initial_state: Dict[str, Any]
    baseline_accuracy: float
    target_accuracy: float


# ============================================================================
# TASK 1: Schema Drift (Easy)
# ============================================================================
SCHEMA_DRIFT = Task(
    id="schema_drift",
    name="Schema Drift Detection & Fix",
    difficulty="easy",
    description="Detect and fix a schema change that broke the ingestion pipeline",
    scenario=(
        "An upstream API changed the 'Age' feature from int to string. "
        "The pipeline crashes on type validation. Initial model accuracy is 0 "
        "due to pipeline failure."
    ),
    max_steps=8,
    optimal_actions=[
        "check_feature_stats",
        "fix_schema",
        "run_eval_set",
    ],
    initial_state={
        "pipeline_status": "failed",
        "model_accuracy": 0.0,
        "baseline_accuracy": 0.75,
        "error_messages": [
            "TypeError: cannot perform reduce with dtype float64 and operand type str",
            "[Age] Expected int64 but got object (string)",
            "Pipeline validation failed at feature ingestion stage"
        ],
        "logs": [
            "[2024-04-08 10:45:23] Starting pipeline...",
            "[2024-04-08 10:45:24] Loaded training data: 50000 samples",
            "[2024-04-08 10:45:25] ERROR: Feature validation failed",
            "[2024-04-08 10:45:25] TypeError on Age column",
            "[2024-04-08 10:45:26] Pipeline halted"
        ],
        "feature_stats": {
            "Age": {
                "type": "string",  # PROBLEM: should be int
                "mean": None,
                "null_pct": 0.01,
                "sample_values": ["25", "34", "null", "42", "29"],
                "note": "Schema changed from int to string in upstream API"
            },
            "Income": {
                "type": "float",
                "mean": 52300.5,
                "null_pct": 0.02,
                "sample_values": [45000.0, 52500.0, 61000.0, 38500.0, 55000.0]
            }
        },
        "root_cause_identified": False,
        "fix_applied": False,
    },
    baseline_accuracy=0.75,
    target_accuracy=0.75,
)


# ============================================================================
# TASK 2: Concept Drift (Medium)
# ============================================================================
CONCEPT_DRIFT = Task(
    id="concept_drift",
    name="Concept Drift Detection & Retraining",
    difficulty="medium",
    description="Identify concept drift and trigger retraining on recent data",
    scenario=(
        "Model accuracy has silently dropped from 0.89 to 0.61 over 2 weeks "
        "due to shifting user behavior patterns. The pipeline appears healthy "
        "but outputs are increasingly inaccurate."
    ),
    max_steps=12,
    optimal_actions=[
        "analyze_drift",
        "trigger_retraining",
        "deploy_model",
    ],
    initial_state={
        "pipeline_status": "degraded",
        "model_accuracy": 0.61,
        "baseline_accuracy": 0.82,
        "error_messages": [
            "WARN: Model accuracy degraded from 0.89 (7 days ago) to 0.73 (3 days ago)",
            "WARN: Feature distribution change detected in Age (KL divergence: 0.42)",
        ],
        "logs": [
            "[2024-04-08 09:00:00] Model inference started",
            "[2024-04-08 09:00:05] Batch 1: Accuracy 0.60",
            "[2024-04-08 09:00:10] Batch 2: Accuracy 0.62",
            "[2024-04-08 09:00:15] WARN: Accuracy below threshold (0.82)",
            "[2024-04-08 09:00:20] Model drift detected (concept shift)"
        ],
        "feature_stats": {
            "Age": {
                "type": "float",
                "mean": 38.5,  # shifted from 32.1 two weeks ago
                "null_pct": 0.01,
                "note": "Mean shifted +6.4, indicating age demographic shift",
                "distribution_change": "high"
            },
            "Income": {
                "type": "float",
                "mean": 51200.0,  # shifted from 54300 two weeks ago
                "null_pct": 0.02,
                "note": "Mean dropped, economic shift detected",
                "distribution_change": "medium"
            },
            "Last_Purchase_Days_Ago": {
                "type": "int",
                "mean": 45,  # shifted from 25 two weeks ago
                "null_pct": 0.03,
                "note": "Users now inactive longer (behavior change)",
                "distribution_change": "high"
            }
        },
        "root_cause_identified": False,
        "fix_applied": False,
    },
    baseline_accuracy=0.82,
    target_accuracy=0.82,
)


# ============================================================================
# TASK 3: Silent GPU NaN Failure (Hard)
# ============================================================================
GPU_NAN_FAILURE = Task(
    id="gpu_nan_failure",
    name="Silent GPU NaN Failure Recovery",
    difficulty="hard",
    description="Debug silent NaN GPU failure, recover from checkpoint, and restore model accuracy",
    scenario=(
        "A faulty GPU driver causes NaN loss spikes during fine-tuning. "
        "Model silently produces garbage predictions. No obvious crash — it "
        "'completes' but outputs NaN weights. This is the hardest debugging scenario."
    ),
    max_steps=15,
    optimal_actions=[
        "run_eval_set",
        "view_training_logs",
        "inspect_model_weights",
        "check_gpu_health",
        "restart_training_node",
        "resume_from_checkpoint",
        "deploy_model",
    ],
    initial_state={
        "pipeline_status": "degraded",
        "model_accuracy": 0.12,  # garbage due to NaN weights
        "baseline_accuracy": 0.85,
        "error_messages": [
            "Model predictions contain NaN values",
            "Model weights contain NaN (detected in inference)",
        ],
        "logs": [
            "[2024-04-08 14:00:00] Starting fine-tuning on GPU 0",
            "[2024-04-08 14:00:15] Step 100: loss = 0.462",
            "[2024-04-08 14:00:30] Step 200: loss = 0.421",
            "[2024-04-08 14:00:45] Step 300: loss = NaN (GPU ANOMALY)",
            "[2024-04-08 14:01:00] Step 400: loss = NaN (PERSISTS)",
            "[2024-04-08 14:05:00] Training completed (silent failure - weights corrupted)",
            "[2024-04-08 14:05:05] Model deployed with NaN weights",
            "[2024-04-08 14:10:00] Inference accuracy: 0.12 (garbage predictions)"
        ],
        "feature_stats": {
            "Age": {"type": "float", "mean": 34.2, "null_pct": 0.01},
            "Income": {"type": "float", "mean": 55000.0, "null_pct": 0.02},
        },
        "gpu_health": {
            "status": "degraded",
            "gpu_0": {
                "memory_used": "8.2GB / 24GB",
                "temperature": 85,  # high
                "driver_version": "535.104.05 (VULNERABLE)",
                "fan_speed": 100,
                "power_draw": 280,
                "driver_status": "FAULTY - Known NaN issue in driver",
                "recommendation": "Update to 550.76+ or use different GPU"
            },
            "error_log": [
                "GPU 0: CUDA compute capability 8.0 detected",
                "Driver warning: CUDA kernel may produce NaN on this hardware",
                "Temperature threshold: 82°C (current: 85°C)"
            ]
        },
        "available_checkpoints": [
            {"id": "checkpoint_epoch_3", "step": 300, "loss": 0.428, "status": "clean"},
            {"id": "checkpoint_epoch_2", "step": 200, "loss": 0.441, "status": "clean"},
            {"id": "checkpoint_epoch_1", "step": 100, "loss": 0.487, "status": "clean"},
        ],
        "root_cause_identified": False,
        "fix_applied": False,
    },
    baseline_accuracy=0.85,
    target_accuracy=0.85,
)


TASKS: Dict[str, Task] = {
    SCHEMA_DRIFT.id: SCHEMA_DRIFT,
    CONCEPT_DRIFT.id: CONCEPT_DRIFT,
    GPU_NAN_FAILURE.id: GPU_NAN_FAILURE,
}


def get_task(task_id: str) -> Task:
    """Get task by ID."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}")
    return TASKS[task_id]


def list_tasks() -> List[Task]:
    """List all available tasks."""
    return [SCHEMA_DRIFT, CONCEPT_DRIFT, GPU_NAN_FAILURE]
