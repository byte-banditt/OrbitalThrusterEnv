"""Graders for MLOps Pipeline Debugger tasks."""

from typing import Dict, Any, List


class TaskGrader:
    """Base grader for tasks."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
    
    def grade(
        self,
        action_history: List[Dict[str, Any]],
        final_accuracy: float,
        root_cause_identified: bool,
        fix_applied: bool,
        target_accuracy: float,
    ) -> float:
        """
        Grade the task performance.
        
        Returns:
            float: Score in [0.0, 1.0]
        
        Scoring breakdown:
            - Root cause correctly identified: +0.3
            - Correct fix action applied with right parameters: +0.4
            - Model accuracy restored above target: +0.3
        """
        score = 0.0
        
        # 0.3 points: Root cause identified
        if root_cause_identified:
            score += 0.3
        
        # 0.4 points: Fix applied correctly
        if fix_applied:
            score += 0.4
        
        # 0.3 points: Accuracy restored
        if final_accuracy >= target_accuracy:
            score += 0.3
        
        return min(1.0, score)


class SchemaDriftGrader(TaskGrader):
    """Grader for schema drift task."""
    
    def __init__(self):
        super().__init__("schema_drift")
    
    def grade(
        self,
        action_history: List[Dict[str, Any]],
        final_accuracy: float,
        root_cause_identified: bool,
        fix_applied: bool,
        target_accuracy: float,
    ) -> float:
        """
        Grade schema drift task.
        
        Root cause: Age feature is string, should be int.
        Correct fix: check_feature_stats() → fix_schema("Age", "int") → run_eval_set()
        """
        score = 0.0
        
        # Check if root cause was identified
        # Root cause: Age should be cast to int
        checked_stats = any(a["action"] == "check_feature_stats" for a in action_history)
        if checked_stats:
            root_cause_identified = True
        
        # Check if correct fix was applied
        fix_schema_calls = [a for a in action_history if a["action"] == "fix_schema"]
        age_fixed_correctly = any(
            a.get("parameters", {}).get("feature") == "Age" and
            a.get("parameters", {}).get("cast_type") == "int"
            for a in fix_schema_calls
        )
        
        run_eval_called = any(a["action"] == "run_eval_set" for a in action_history)
        
        if checked_stats:
            score += 0.2
        
        if age_fixed_correctly and run_eval_called:
            score += 0.5
            fix_applied = True
        
        if final_accuracy >= target_accuracy:
            score += 0.3
        
        return min(1.0, score)


class ConceptDriftGrader(TaskGrader):
    """Grader for concept drift task."""
    
    def __init__(self):
        super().__init__("concept_drift")
    
    def grade(
        self,
        action_history: List[Dict[str, Any]],
        final_accuracy: float,
        root_cause_identified: bool,
        fix_applied: bool,
        target_accuracy: float,
    ) -> float:
        """
        Grade concept drift task.
        
        Root cause: Concept drift detected (data distribution changed).
        Correct fix: analyze_drift() → trigger_retraining(with recent data) → deploy_model()
        """
        score = 0.0
        
        # Check if drift was analyzed
        drift_analyzed = any(a["action"] == "analyze_drift" for a in action_history)
        if drift_analyzed:
            root_cause_identified = True
            score += 0.2
        
        # Check if retraining was triggered on recent data
        retraining_calls = [a for a in action_history if a["action"] == "trigger_retraining"]
        recent_data_retrained = any(
            "last_" in str(a.get("parameters", {}).get("data_slice", "")).lower()
            for a in retraining_calls
        )
        
        deploy_called = any(a["action"] == "deploy_model" for a in action_history)
        
        if recent_data_retrained and deploy_called:
            score += 0.5
            fix_applied = True
        
        if final_accuracy >= target_accuracy:
            score += 0.3
        
        return min(1.0, score)


class GPUNaNFailureGrader(TaskGrader):
    """Grader for GPU NaN failure task."""
    
    def __init__(self):
        super().__init__("gpu_nan_failure")
    
    def grade(
        self,
        action_history: List[Dict[str, Any]],
        final_accuracy: float,
        root_cause_identified: bool,
        fix_applied: bool,
        target_accuracy: float,
    ) -> float:
        """
        Grade GPU NaN failure task.
        
        Root cause: GPU driver fault causing NaN loss spikes.
        Correct fix: Check GPU health → Restart training node → Resume from checkpoint → Deploy
        
        This is the hardest task — requires specific sequence of actions.
        """
        score = 0.0
        
        # Check if GPU health was inspected
        gpu_health_checked = any(a["action"] == "check_gpu_health" for a in action_history)
        if gpu_health_checked:
            root_cause_identified = True
            score += 0.2
        
        # Check if training was restarted
        restart_called = any(a["action"] == "restart_training_node" for a in action_history)
        
        # Check if checkpoint was resumed
        resume_calls = [a for a in action_history if a["action"] == "resume_from_checkpoint"]
        checkpoint_resumed = len(resume_calls) > 0
        
        # Check if model was deployed
        deploy_called = any(a["action"] == "deploy_model" for a in action_history)
        
        # All critical steps must be taken
        if restart_called and checkpoint_resumed and deploy_called:
            score += 0.5
            fix_applied = True
        elif restart_called and checkpoint_resumed:
            score += 0.3  # Partial credit for recovery steps
        
        if final_accuracy >= target_accuracy:
            score += 0.3
        
        return min(1.0, score)


def get_grader(task_id: str) -> TaskGrader:
    """Get grader for a task."""
    graders = {
        "schema_drift": SchemaDriftGrader(),
        "concept_drift": ConceptDriftGrader(),
        "gpu_nan_failure": GPUNaNFailureGrader(),
    }
    if task_id not in graders:
        raise ValueError(f"Unknown task: {task_id}")
    return graders[task_id]
