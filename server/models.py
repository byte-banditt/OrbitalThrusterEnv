"""Pydantic models for MLOps Pipeline Debugger environment."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class MLOpsObservation(BaseModel):
    """What the agent sees at each step."""
    task_id: str = Field(..., description="Identifier of the current task")
    step: int = Field(..., description="Current step number")
    pipeline_status: str = Field(
        ..., 
        description="Pipeline health status: 'healthy', 'degraded', 'failed', or 'fixed'"
    )
    model_accuracy: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Current model accuracy [0.0-1.0]"
    )
    baseline_accuracy: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Target accuracy to restore"
    )
    available_actions: List[str] = Field(
        ..., 
        description="List of actions the agent can take"
    )
    logs: List[str] = Field(
        default_factory=list, 
        description="Last 5 log lines from the pipeline"
    )
    feature_stats: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Feature statistics including type, mean, null percentage"
    )
    error_messages: List[str] = Field(
        default_factory=list, 
        description="Active error messages"
    )
    hint: str = Field(
        default="", 
        description="Optional hint for the agent"
    )


class MLOpsAction(BaseModel):
    """Action the agent wants to take."""
    action: str = Field(..., description="Action name from available_actions")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Action parameters (e.g., {'feature': 'Age', 'cast_type': 'int'})"
    )


class StepResponse(BaseModel):
    """Response from /step endpoint."""
    observation: MLOpsObservation
    reward: float = Field(ge=-1.0, le=1.0, description="Reward for this step")
    done: bool = Field(description="Whether episode is complete")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional info")


class MLOpsState(BaseModel):
    """Full internal state of the environment."""
    task_id: str
    step: int
    root_cause_identified: bool
    fix_applied: bool
    pipeline_status: str
    model_accuracy: float
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str


class TaskDescription(BaseModel):
    """Description of an available task."""
    id: str
    description: str
    difficulty: str
    max_steps: int
    scenario: str
