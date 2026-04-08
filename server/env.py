"""Core environment logic for MLOps Pipeline Debugger."""

import copy
import math
from typing import Dict, Any, List, Tuple, Optional
from models import MLOpsObservation, MLOpsAction
from tasks import get_task, Task
from graders import get_grader


class MLOpsEnvironment:
    """MLOps Pipeline Debugger environment."""
    
    def __init__(self, task_id: str):
        """Initialize environment with a task."""
        self.task: Task = get_task(task_id)
        self.grader = get_grader(task_id)
        self.reset()
    
    def reset(self) -> MLOpsObservation:
        """Reset environment and return initial observation."""
        # Deep copy initial state
        self.state = copy.deepcopy(self.task.initial_state)
        self.step_count = 0
        self.action_history: List[Dict[str, Any]] = []
        self.wrong_actions_count = 0
        self.done = False
        self.cumulative_reward = 0.0
        
        return self._get_observation()
    
    def step(self, action: MLOpsAction) -> Tuple[MLOpsObservation, float, bool, Dict[str, Any]]:
        """
        Execute one step.
        
        Returns:
            (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode already done. Call reset() to start a new episode.")
        
        self.step_count += 1
        action_name = action.action
        parameters = action.parameters
        
        # Execute action
        reward, action_success = self._execute_action(action_name, parameters)
        
        # Track action
        self.action_history.append({
            "step": self.step_count,
            "action": action_name,
            "parameters": parameters,
            "reward": reward,
            "success": action_success
        })
        
        # Track wrong actions
        if not action_success:
            self.wrong_actions_count += 1
        else:
            self.wrong_actions_count = 0  # reset on success
        
        # Update cumulative reward
        self.cumulative_reward += reward
        
        # Check done conditions
        if self._check_done():
            self.done = True
        elif self.step_count >= self.task.max_steps:
            self.done = True
        elif self.wrong_actions_count >= 3:
            self.done = True
        
        info = {
            "action_success": action_success,
            "cumulative_reward": self.cumulative_reward,
            "wrong_actions_count": self.wrong_actions_count,
        }
        
        observation = self._get_observation()
        
        return observation, reward, self.done, info
    
    def _execute_action(self, action: str, parameters: Dict[str, Any]) -> Tuple[float, bool]:
        """Execute an action and return (reward, success)."""
        reward = 0.0
        success = False
        
        if self.task.id == "schema_drift":
            reward, success = self._execute_schema_drift_action(action, parameters)
        elif self.task.id == "concept_drift":
            reward, success = self._execute_concept_drift_action(action, parameters)
        elif self.task.id == "gpu_nan_failure":
            reward, success = self._execute_gpu_nan_action(action, parameters)
        
        return reward, success
    
    # ========================================================================
    # SCHEMA DRIFT TASK
    # ========================================================================
    
    def _execute_schema_drift_action(self, action: str, parameters: Dict[str, Any]) -> Tuple[float, bool]:
        """Execute actions for schema drift task."""
        reward = 0.0
        success = False
        
        if action == "check_feature_stats":
            # Diagnostically correct action
            self.state["logs"].append(
                "[2024-04-08 10:46:00] Retrieved feature statistics"
            )
            reward = 0.15
            success = True
            self.state["root_cause_identified"] = True
        
        elif action == "view_training_logs":
            # Diagnostically helpful
            self.state["logs"].append(
                "[2024-04-08 10:46:00] Viewing training logs..."
            )
            reward = 0.10
            success = True
        
        elif action == "fix_schema":
            feature = parameters.get("feature", "")
            cast_type = parameters.get("cast_type", "")
            
            if feature == "Age" and cast_type == "int":
                # Correct fix
                self.state["feature_stats"]["Age"]["type"] = "int"
                self.state["feature_stats"]["Age"]["mean"] = 34.2  # now computable
                self.state["logs"].append(
                    f"[2024-04-08 10:46:30] Applied schema fix: Age cast to int"
                )
                self.state["pipeline_status"] = "healthy"
                self.state["fix_applied"] = True
                reward = 0.35
                success = True
            else:
                # Wrong fix
                self.state["logs"].append(
                    f"[2024-04-08 10:46:30] ERROR: Invalid schema fix attempt"
                )
                reward = -0.10
                success = False
        
        elif action == "rollback_model":
            # Not necessary for this task
            self.state["logs"].append(
                "[2024-04-08 10:46:30] Rolled back model (not needed)"
            )
            reward = -0.10
            success = False
        
        elif action == "run_eval_set":
            # Verification step
            if self.state["fix_applied"]:
                # Fix was already applied, eval should succeed
                self.state["model_accuracy"] = 0.78
                self.state["pipeline_status"] = "fixed"
                self.state["logs"].append(
                    "[2024-04-08 10:46:45] Evaluation complete: accuracy 0.78"
                )
                reward = 0.25
                success = True
            else:
                # Fix not yet applied
                self.state["logs"].append(
                    "[2024-04-08 10:46:45] Evaluation failed: schema broken"
                )
                reward = -0.05
                success = False
        
        else:
            reward = -0.05
            success = False
        
        return reward, success
    
    # ========================================================================
    # CONCEPT DRIFT TASK
    # ========================================================================
    
    def _execute_concept_drift_action(self, action: str, parameters: Dict[str, Any]) -> Tuple[float, bool]:
        """Execute actions for concept drift task."""
        reward = 0.0
        success = False
        
        if action == "check_feature_stats":
            self.state["logs"].append(
                "[2024-04-08 09:01:00] Retrieved feature statistics"
            )
            reward = 0.10
            success = True
        
        elif action == "view_training_logs":
            self.state["logs"].append(
                "[2024-04-08 09:01:00] Training logs: Model degradation observed"
            )
            reward = 0.10
            success = True
        
        elif action == "run_eval_set":
            self.state["logs"].append(
                "[2024-04-08 09:01:00] Current eval accuracy: 0.61"
            )
            reward = 0.05
            success = True
        
        elif action == "analyze_drift":
            window_days = parameters.get("window_days", 14)
            if window_days >= 14:
                # Correct analysis window
                self.state["logs"].append(
                    f"[2024-04-08 09:01:15] Drift analysis over {window_days} days: "
                    "KL divergence = 0.42 (HIGH), distribution shifted significantly"
                )
                self.state["root_cause_identified"] = True
                reward = 0.25
                success = True
            else:
                self.state["logs"].append(
                    f"[2024-04-08 09:01:15] Insufficient window: {window_days} < 14"
                )
                reward = 0.05
                success = False
        
        elif action == "trigger_retraining":
            data_slice = parameters.get("data_slice", "")
            learning_rate = parameters.get("learning_rate", 0.001)
            
            # Check if using recent data
            if "last_" in data_slice.lower() or data_slice == "last_30_days":
                if 0.0001 <= learning_rate <= 0.01:
                    self.state["logs"].append(
                        f"[2024-04-08 09:02:00] Retraining started: "
                        f"data_slice={data_slice}, lr={learning_rate}"
                    )
                    self.state["logs"].append(
                        "[2024-04-08 09:05:30] Retraining completed: new model ready"
                    )
                    self.state["fix_applied"] = True
                    reward = 0.30
                    success = True
                else:
                    self.state["logs"].append(
                        f"[2024-04-08 09:02:00] Invalid learning rate: {learning_rate}"
                    )
                    reward = -0.05
                    success = False
            else:
                self.state["logs"].append(
                    f"[2024-04-08 09:02:00] Old data slice detected: {data_slice}"
                )
                reward = -0.05
                success = False
        
        elif action == "deploy_model":
            version = parameters.get("version", "")
            if self.state["fix_applied"]:
                self.state["logs"].append(
                    f"[2024-04-08 09:06:00] Deployed model version: {version}"
                )
                self.state["model_accuracy"] = 0.83
                self.state["pipeline_status"] = "fixed"
                reward = 0.25
                success = True
            else:
                self.state["logs"].append(
                    f"[2024-04-08 09:06:00] Cannot deploy: no retrained model"
                )
                reward = -0.10
                success = False
        
        elif action == "rollback_model":
            self.state["logs"].append(
                "[2024-04-08 09:01:00] Rolled back to previous model"
            )
            reward = -0.10
            success = False
        
        else:
            reward = -0.05
            success = False
        
        return reward, success
    
    # ========================================================================
    # GPU NaN FAILURE TASK
    # ========================================================================
    
    def _execute_gpu_nan_action(self, action: str, parameters: Dict[str, Any]) -> Tuple[float, bool]:
        """Execute actions for GPU NaN failure task."""
        reward = 0.0
        success = False
        
        if action == "run_eval_set":
            self.state["logs"].append(
                "[2024-04-08 14:11:00] Evaluation running..."
            )
            self.state["logs"].append(
                "[2024-04-08 14:11:30] ERROR: Model predictions contain NaN"
            )
            reward = 0.10
            success = True
        
        elif action == "check_feature_stats":
            self.state["logs"].append(
                "[2024-04-08 14:11:00] Feature statistics retrieved (no NaN in data)"
            )
            reward = 0.05
            success = True
        
        elif action == "view_training_logs":
            self.state["logs"].append(
                "[2024-04-08 14:11:00] Training logs show NaN loss at step 300+"
            )
            reward = 0.15
            success = True
        
        elif action == "inspect_model_weights":
            self.state["logs"].append(
                "[2024-04-08 14:11:30] Weight inspection: Layer 5-12 contain NaN values"
            )
            self.state["root_cause_identified"] = True  # NaN weights confirmed
            reward = 0.20
            success = True
        
        elif action == "check_gpu_health":
            self.state["logs"].append(
                "[2024-04-08 14:11:45] GPU health check: Driver FAULTY (535.104.05)"
            )
            self.state["logs"].append(
                "[2024-04-08 14:11:45] Known NaN bug in driver on this hardware"
            )
            self.state["root_cause_identified"] = True  # Root cause found
            reward = 0.25
            success = True
        
        elif action == "restart_training_node":
            clean_env = parameters.get("clean_env", False)
            if clean_env:
                self.state["logs"].append(
                    "[2024-04-08 14:12:00] Training node restarted (clean environment)"
                )
                self.state["pipeline_status"] = "degraded"  # still recovering
                reward = 0.20
                success = True
            else:
                self.state["logs"].append(
                    "[2024-04-08 14:12:00] Node restart without clean environment"
                )
                reward = -0.05
                success = False
        
        elif action == "resume_from_checkpoint":
            checkpoint_id = parameters.get("checkpoint_id", "")
            if checkpoint_id in ["checkpoint_epoch_1", "checkpoint_epoch_2", "checkpoint_epoch_3"]:
                self.state["logs"].append(
                    f"[2024-04-08 14:12:30] Resumed from {checkpoint_id} (clean weights)"
                )
                self.state["fix_applied"] = True
                reward = 0.25
                success = True
            else:
                self.state["logs"].append(
                    f"[2024-04-08 14:12:30] Invalid checkpoint: {checkpoint_id}"
                )
                reward = -0.10
                success = False
        
        elif action == "trigger_retraining":
            # Not ideal for this scenario
            data_slice = parameters.get("data_slice", "")
            self.state["logs"].append(
                f"[2024-04-08 14:12:00] Retraining with {data_slice} (risky with faulty GPU)"
            )
            reward = -0.15  # negative — will likely fail again on same faulty GPU
            success = False
        
        elif action == "deploy_model":
            version = parameters.get("version", "")
            if self.state["fix_applied"]:
                self.state["logs"].append(
                    f"[2024-04-08 14:13:00] Deployed model {version} (from checkpoint)"
                )
                self.state["model_accuracy"] = 0.87
                self.state["pipeline_status"] = "fixed"
                reward = 0.25
                success = True
            else:
                self.state["logs"].append(
                    f"[2024-04-08 14:13:00] Cannot deploy: weights still corrupted"
                )
                reward = -0.10
                success = False
        
        else:
            reward = -0.05
            success = False
        
        return reward, success
    
    def _check_done(self) -> bool:
        """Check if episode is done (success condition)."""
        # Episode ends successfully when:
        # 1. Root cause identified
        # 2. Fix applied
        # 3. Accuracy above baseline
        return (
            self.state["root_cause_identified"] and
            self.state["fix_applied"] and
            self.state["model_accuracy"] >= self.task.baseline_accuracy
        )
    
    def _get_observation(self) -> MLOpsObservation:
        """Get current observation."""
        return MLOpsObservation(
            task_id=self.task.id,
            step=self.step_count,
            pipeline_status=self.state["pipeline_status"],
            model_accuracy=self.state["model_accuracy"],
            baseline_accuracy=self.task.baseline_accuracy,
            available_actions=self._get_available_actions(),
            logs=self.state.get("logs", [])[-5:],  # Last 5 logs
            feature_stats=self.state.get("feature_stats", {}),
            error_messages=self.state.get("error_messages", []),
            hint=self._get_hint(),
        )
    
    def _get_available_actions(self) -> List[str]:
        """Get list of available actions for current task."""
        actions_map = {
            "schema_drift": [
                "check_feature_stats",
                "view_training_logs",
                "fix_schema",
                "rollback_model",
                "run_eval_set",
            ],
            "concept_drift": [
                "check_feature_stats",
                "view_training_logs",
                "run_eval_set",
                "analyze_drift",
                "trigger_retraining",
                "deploy_model",
                "rollback_model",
            ],
            "gpu_nan_failure": [
                "run_eval_set",
                "check_feature_stats",
                "view_training_logs",
                "inspect_model_weights",
                "check_gpu_health",
                "restart_training_node",
                "resume_from_checkpoint",
                "trigger_retraining",
                "deploy_model",
            ],
        }
        return actions_map.get(self.task.id, [])
    
    def _get_hint(self) -> str:
        """Get hint based on current state."""
        hints_map = {
            "schema_drift": (
                "Hint: Check feature statistics first—one of the feature types has changed. "
                "Use fix_schema(feature, cast_type) to correct the type, then run_eval_set()."
            ),
            "concept_drift": (
                "Hint: Use analyze_drift(window_days) to quantify distribution changes. "
                "Then trigger_retraining() on recent data, and deploy_model()."
            ),
            "gpu_nan_failure": (
                "Hint: This is a hard task. The model has NaN weights. "
                "Check GPU health, restart the training node, resume from a clean checkpoint, "
                "then deploy the recovered model."
            ),
        }
        return hints_map.get(self.task.id, "")
    
    def get_state(self) -> Dict[str, Any]:
        """Get full internal state (for /state endpoint)."""
        from models import MLOpsState
        
        # Compute final score using grader
        score = self.grader.grade(
            action_history=self.action_history,
            final_accuracy=self.state["model_accuracy"],
            root_cause_identified=self.state.get("root_cause_identified", False),
            fix_applied=self.state.get("fix_applied", False),
            target_accuracy=self.task.baseline_accuracy,
        )
        
        return MLOpsState(
            task_id=self.task.id,
            step=self.step_count,
            root_cause_identified=self.state.get("root_cause_identified", False),
            fix_applied=self.state.get("fix_applied", False),
            pipeline_status=self.state["pipeline_status"],
            model_accuracy=self.state["model_accuracy"],
            action_history=self.action_history,
            score=score,
        )
