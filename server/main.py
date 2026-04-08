"""FastAPI server for MLOps Pipeline Debugger."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional
from env import MLOpsEnvironment
from models import (
    MLOpsObservation,
    MLOpsAction,
    StepResponse,
    HealthResponse,
    TaskDescription,
)
from tasks import list_tasks

# ============================================================================
# SETUP
# ============================================================================

app = FastAPI(
    title="MLOps Pipeline Debugger",
    description="An RL environment where AI agents debug broken ML production pipelines",
    version="1.0.0",
)

# Add CORS middleware for browser/external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global environment instance
current_env: Optional[MLOpsEnvironment] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/tasks")
async def get_tasks() -> list[TaskDescription]:
    """List all available tasks."""
    tasks = list_tasks()
    return [
        TaskDescription(
            id=task.id,
            description=task.description,
            difficulty=task.difficulty,
            max_steps=task.max_steps,
            scenario=task.scenario,
        )
        for task in tasks
    ]


@app.post("/reset", response_model=MLOpsObservation)
async def reset(task_id: str) -> MLOpsObservation:
    """
    Reset the environment and start a new episode.
    
    Args:
        task_id: One of "schema_drift", "concept_drift", "gpu_nan_failure"
    
    Returns:
        Initial observation
    """
    global current_env
    
    try:
        logger.info(f"Resetting environment for task: {task_id}")
        current_env = MLOpsEnvironment(task_id)
        observation = current_env.reset()
        return observation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error resetting environment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(action: MLOpsAction) -> StepResponse:
    """
    Execute one step of the environment.
    
    Args:
        action: MLOpsAction describing the action and parameters
    
    Returns:
        Observation, reward, done flag, and info dict
    """
    global current_env
    
    if current_env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        logger.info(f"Executing action: {action.action} with params: {action.parameters}")
        observation, reward, done, info = current_env.step(action)
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing step: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Get full internal state of the environment."""
    global current_env
    
    if current_env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        state = current_env.get_state()
        return state
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup."""
    logger.info("MLOps Pipeline Debugger server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown."""
    logger.info("MLOps Pipeline Debugger server shutting down")


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "MLOps Pipeline Debugger",
        "version": "1.0.0",
        "description": "An RL environment for debugging ML production pipelines",
        "endpoints": {
            "GET /health": "Health check",
            "GET /tasks": "List available tasks",
            "POST /reset": "Reset environment with task_id query param",
            "POST /step": "Execute one step",
            "GET /state": "Get full environment state",
            "GET /docs": "FastAPI interactive docs (Swagger UI)",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
