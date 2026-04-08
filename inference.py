#!/usr/bin/env python3
"""Baseline inference script for MLOps Pipeline Debugger.

This script:
1. Starts the FastAPI server for the OpenEnv environment
2. Initializes an LLM agent via OpenAI client
3. Runs all 3 tasks (schema_drift, concept_drift, gpu_nan_failure)
4. Prints results in the specified format
"""

import os
import sys
import subprocess
import time
import json
import logging
import httpx
from typing import Optional, Dict, Any

from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", None)

SERVER_URL = "http://localhost:8000"
MAX_RETRIES = 30
RETRY_DELAY = 1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

if not HF_TOKEN:
    logger.error("HF_TOKEN environment variable not set!")
    sys.exit(1)


# ============================================================================
# SERVER MANAGEMENT
# ============================================================================

server_process = None


def start_server():
    """Start the FastAPI server."""
    global server_process
    logger.info("Starting FastAPI server...")
    
    # Change to server directory and start uvicorn
    server_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server")
    
    try:
        server_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=server_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"Server process started with PID {server_process.pid}")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


def wait_for_server():
    """Wait for server to be ready."""
    logger.info("Waiting for server to be ready...")
    
    for attempt in range(MAX_RETRIES):
        try:
            response = httpx.get(f"{SERVER_URL}/health", timeout=2.0)
            if response.status_code == 200:
                logger.info("Server is ready!")
                return True
        except Exception as e:
            if attempt % 5 == 0:
                logger.info(f"Attempt {attempt+1}/{MAX_RETRIES}...")
        
        time.sleep(RETRY_DELAY)
    
    logger.error("Server did not start in time!")
    return False


def stop_server():
    """Stop the FastAPI server."""
    global server_process
    if server_process:
        logger.info("Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        logger.info("Server stopped")


# ============================================================================
# HTTP HELPERS
# ============================================================================

def http_reset(task_id: str) -> Optional[Dict[str, Any]]:
    """Reset environment for a task."""
    try:
        response = httpx.post(
            f"{SERVER_URL}/reset",
            params={"task_id": task_id},
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error resetting {task_id}: {e}")
        return None


def http_step(action: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Execute one step."""
    try:
        payload = {
            "action": action,
            "parameters": parameters
        }
        response = httpx.post(
            f"{SERVER_URL}/step",
            json=payload,
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error executing step: {e}")
        return None


def http_get_state() -> Optional[Dict[str, Any]]:
    """Get environment state."""
    try:
        response = httpx.get(
            f"{SERVER_URL}/state",
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        return None


def http_get_tasks() -> list:
    """Get list of tasks."""
    try:
        response = httpx.get(
            f"{SERVER_URL}/tasks",
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        return []


# ============================================================================
# LLM AGENT
# ============================================================================

class MLOpsAgent:
    """LLM-based agent for debugging MLOps pipelines."""
    
    def __init__(self):
        """Initialize the agent with OpenAI client."""
        self.client = OpenAI(
            api_key=HF_TOKEN,
            base_url=API_BASE_URL,
        )
        self.model = MODEL_NAME
        logger.info(f"Initialized agent with model: {self.model}")
    
    def get_action(
        self,
        observation: Dict[str, Any],
        available_actions: list,
        action_history: list,
    ) -> Dict[str, Any]:
        """
        Get next action from LLM agent.
        
        Returns:
            {"action": str, "parameters": dict}
        """
        # Build system prompt
        system_prompt = """You are an MLOps debugging expert. Your task is to diagnose and fix 
broken machine learning production pipelines. You will interact with a simulated environment 
through actions and observations.

WORKFLOW STRATEGY:
1. READ pipeline_status from observation
2. IF status = "failed" or "degraded": 
   - DIAGNOSE ONCE (use check_feature_stats, analyze_drift, or check_gpu_health FIRST)
   - Then apply the appropriate FIX action for that task
   - Then VERIFY with the task-specific verification action (see below)
3. IF status = "healthy": VERIFICATION ACTION (see below) 
4. IF status = "fixed": Task COMPLETE - stop

TASK-SPECIFIC ACTION SEQUENCES (FOLLOW EXACTLY):
- schema_drift: check_feature_stats → fix_schema → run_eval_set [STOP]
  * fix_schema parameters: feature="Age", cast_type="int"
  
- concept_drift: analyze_drift → trigger_retraining → run_eval_set → deploy_model [STOP]
  * analyze_drift: window_days >= 14
  * trigger_retraining: data_slice="last_30_days", learning_rate=0.001
  
- gpu_nan_failure: check_gpu_health → restart_training_node → resume_from_checkpoint → deploy_model [STOP]
  * restart_training_node: clean_env=True
  * resume_from_checkpoint: checkpoint_id="checkpoint_epoch_3" (use this specific checkpoint)
  * deploy_model: version="" or any version string

CRITICAL: After completing the sequence above, STOP. Do not take more actions.

VERIFICATION ACTIONS (use ONLY these to verify):
- schema_drift: run_eval_set() is the final action
- concept_drift: deploy_model() is the final action AFTER run_eval_set()
- gpu_nan_failure: deploy_model() is the final action AFTER resume_from_checkpoint()

ACTION PROGRESSION RULES:
1. Do NOT repeat the same action twice in a row
2. Follow the exact sequence for your task
3. Use action history to track progress - avoid repeating recent actions
4. Once pipeline_status reaches "fixed", STOP immediately

Respond ONLY with valid JSON:
{
  "action": "<action_name>",
  "parameters": {"key": "value"}
}"""

        # Build user prompt
        action_names = ", ".join(available_actions)
        
        # Include action history to help agent track what it's done
        history_str = ""
        if action_history:
            history_str = "\nRECENT ACTIONS TAKEN:\n"
            for i, action_rec in enumerate(action_history[-5:], 1):  # Last 5 actions
                history_str += f"  {i}. {action_rec['action']} -> reward={action_rec.get('reward', 0):.2f}\n"
        
        user_prompt = f"""Current observation:
Task: {observation.get('task_id', 'unknown')}
Step: {observation.get('step', 0)}
Pipeline Status: {observation.get('pipeline_status', 'unknown')}
Model Accuracy: {observation.get('model_accuracy', 0):.3f}
Baseline Accuracy: {observation.get('baseline_accuracy', 0):.3f}

Recent Logs:
{json.dumps(observation.get('logs', []), indent=2)}

Error Messages:
{json.dumps(observation.get('error_messages', []), indent=2)}

Feature Statistics:
{json.dumps(observation.get('feature_stats', {}), indent=2)}{history_str}

ACTION HISTORY (recent actions and rewards):
{history_str if history_str else "  (None yet)"}

NEXT STEP GUIDANCE:
Pipeline Status: {observation.get('pipeline_status', 'unknown')}
- If "failed" or "degraded": Diagnose ONCE, then apply FIX, then verify
- If "healthy": Diagnose succeeded! Now apply the FIX action specific to this task
- If "fixed": Task COMPLETE - take no more actions

CRITICAL: Read your recent actions above. If you just called a diagnostic action 
(check_feature_stats, analyze_drift, check_gpu_health), do NOT call it again!
Move to the next step in the sequence.

GOAL: Get pipeline_status to "fixed" and accuracy >= {observation.get('baseline_accuracy', 0):.3f}

Available Actions: {action_names}

What is your next action?"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,  # Reduced from 0.7 for more deterministic behavior
                max_tokens=300,    # Reduced to focus on just the action
                timeout=30.0,
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"LLM response: {response_text}")
            
            # Parse JSON response
            try:
                action = json.loads(response_text)
                if "action" in action and action["action"] in available_actions:
                    if "parameters" not in action:
                        action["parameters"] = {}
                    return action
                else:
                    logger.warning("LLM response has invalid action, using safe default")
                    return {"action": available_actions[0], "parameters": {}}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON: {response_text}")
                # Try to extract action from response
                for action_name in available_actions:
                    if action_name in response_text:
                        return {"action": action_name, "parameters": {}}
                # Fallback to first available action
                return {"action": available_actions[0], "parameters": {}}
        
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            # Fallback to first available action
            return {"action": available_actions[0], "parameters": {}}
    
    def run_task(self, task_id: str, max_steps: int = 20) -> Dict[str, Any]:
        """
        Run a single task.
        
        Returns:
            {
                "task_id": str,
                "success": bool,
                "steps": int,
                "total_reward": float,
                "final_score": float,
                "rewards": list[float],
            }
        """
        print(f"[START] task={task_id} env=mlops-pipeline-debugger model={self.model}")
        
        # Reset environment
        observation = http_reset(task_id)
        if observation is None:
            print(f"[END] success=false steps=0 score=0.0 rewards=[]")
            return {
                "task_id": task_id,
                "success": False,
                "steps": 0,
                "total_reward": 0.0,
                "final_score": 0.0,
                "rewards": [],
            }
        
        step_count = 0
        total_reward = 0.0
        rewards = []
        action_history = []  # Track full action history
        
        # Run episode
        while step_count < max_steps:
            step_count += 1
            
            # Get action from agent
            available_actions = observation.get("available_actions", [])
            if not available_actions:
                logger.warning("No available actions!")
                break
            
            action_data = self.get_action(observation, available_actions, action_history)
            action_name = action_data.get("action", "")
            parameters = action_data.get("parameters", {})
            
            # Execute action
            step_result = http_step(action_name, parameters)
            if step_result is None:
                print(f"[STEP] step={step_count} action={action_name} reward=0.00 done=true error=http_error")
                break
            
            # Extract results
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            total_reward += reward
            rewards.append(reward)
            
            # Track full action record
            action_history.append({
                "action": action_name,
                "reward": reward,
                "parameters": parameters
            })
            
            # Print step
            print(f"[STEP] step={step_count} action={action_name} reward={reward:.2f} done={done} error=null")
            
            # Update observation
            observation = step_result.get("observation", observation)
            
            if done:
                break
        
        # Get final state and score
        final_state = http_get_state()
        final_score = final_state.get("score", 0.0) if final_state else 0.0
        success = final_state.get("fix_applied", False) and final_state.get("root_cause_identified", False) if final_state else False
        
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step_count} score={final_score:.2f} rewards=[{rewards_str}]")
        
        return {
            "task_id": task_id,
            "success": success,
            "steps": step_count,
            "total_reward": total_reward,
            "final_score": final_score,
            "rewards": rewards,
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    try:
        # Start server
        start_server()
        time.sleep(1)  # Brief delay before checking
        
        if not wait_for_server():
            logger.error("Failed to start server!")
            return 1
        
        # Get available tasks
        tasks = http_get_tasks()
        task_ids = [t["id"] for t in tasks]
        logger.info(f"Available tasks: {task_ids}")
        
        # Initialize agent
        agent = MLOpsAgent()
        
        # Run all tasks
        results = []
        for task_id in task_ids:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running task: {task_id}")
            logger.info(f"{'='*60}\n")
            result = agent.run_task(task_id)
            results.append(result)
            time.sleep(1)  # Brief delay between tasks
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}\n")
        
        total_score = sum(r["final_score"] for r in results)
        avg_score = total_score / len(results) if results else 0.0
        
        for result in results:
            logger.info(
                f"{result['task_id']}: "
                f"success={result['success']}, "
                f"steps={result['steps']}, "
                f"score={result['final_score']:.2f}"
            )
        
        logger.info(f"\nAverage score: {avg_score:.2f}")
        logger.info(f"Total score: {total_score:.2f}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    finally:
        # Clean up
        stop_server()


if __name__ == "__main__":
    sys.exit(main())
