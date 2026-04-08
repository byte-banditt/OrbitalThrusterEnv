#!/usr/bin/env python3
"""Validation script to check if all components are working."""

import sys
import os

def check_imports():
    """Check if all modules can be imported."""
    print("Checking imports...")
    try:
        # Add server to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
        
        from models import MLOpsObservation, MLOpsAction, StepResponse, MLOpsState
        print("✓ models.py")
        
        from tasks import get_task, list_tasks
        print("✓ tasks.py")
        
        from graders import get_grader
        print("✓ graders.py")
        
        from env import MLOpsEnvironment
        print("✓ env.py")
        
        from main import app
        print("✓ main.py (FastAPI app)")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def check_tasks():
    """Check if all tasks are defined."""
    print("\nChecking tasks...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
        from tasks import list_tasks
        
        tasks = list_tasks()
        print(f"✓ Found {len(tasks)} tasks:")
        for task in tasks:
            print(f"  - {task.id} ({task.difficulty}): {task.name}")
        return len(tasks) == 3
    except Exception as e:
        print(f"✗ Task check failed: {e}")
        return False


def check_environment():
    """Check if environment can be initialized."""
    print("\nChecking environment...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
        from env import MLOpsEnvironment
        
        for task_id in ["schema_drift", "concept_drift", "gpu_nan_failure"]:
            env = MLOpsEnvironment(task_id)
            obs = env.reset()
            print(f"✓ {task_id}: initialized successfully")
            print(f"  - Status: {obs.pipeline_status}")
            print(f"  - Accuracy: {obs.model_accuracy:.2f}")
            print(f"  - Available actions: {len(obs.available_actions)}")
        
        return True
    except Exception as e:
        print(f"✗ Environment check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_graders():
    """Check if graders work."""
    print("\nChecking graders...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
        from graders import get_grader
        
        for task_id in ["schema_drift", "concept_drift", "gpu_nan_failure"]:
            grader = get_grader(task_id)
            score = grader.grade(
                action_history=[],
                final_accuracy=0.5,
                root_cause_identified=False,
                fix_applied=False,
                target_accuracy=0.75,
            )
            print(f"✓ {task_id}: grader score = {score:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Grader check failed: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("MLOps Pipeline Debugger - Validation Script")
    print("=" * 60)
    
    checks = [
        check_imports(),
        check_tasks(),
        check_environment(),
        check_graders(),
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✓ All checks passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some checks failed!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
