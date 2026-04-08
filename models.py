from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, field_validator


ActionType = Literal[
    "fire_pitch_pos_small",
    "fire_pitch_neg_small",
    "fire_roll_pos_small",
    "fire_roll_neg_small",
    "fire_yaw_pos_small",
    "fire_yaw_neg_small",
    "fire_pitch_pos_large",
    "fire_pitch_neg_large",
    "fire_roll_pos_large",
    "fire_roll_neg_large",
    "fire_yaw_pos_large",
    "fire_yaw_neg_large",
    "idle",
]


class AttitudeVector(BaseModel):
    pitch: float = Field(..., description="Pitch angle or rate")
    roll: float = Field(..., description="Roll angle or rate")
    yaw: float = Field(..., description="Yaw angle or rate")


class OrbitalThrusterAction(Action):
    action_type: ActionType
    reason: str = Field(default="", max_length=240, description="Optional control rationale")


class OrbitalThrusterObservation(Observation):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    mission_phase: str
    current_attitude_deg: AttitudeVector
    current_angular_velocity_dps: AttitudeVector
    target_attitude_deg: AttitudeVector
    attitude_error_deg: AttitudeVector
    fuel_remaining: float = Field(..., ge=0.0)
    fuel_used: float = Field(..., ge=0.0)
    step_budget: int = Field(..., ge=1)
    steps_used: int = Field(..., ge=0)
    time_elapsed: float = Field(..., ge=0.0)
    disturbance_level: float = Field(..., ge=0.0)
    last_action: Optional[ActionType] = None
    reward_so_far: float = 0.0
    success: bool = False
    last_feedback: Optional[str] = None
    done: bool = False
    reward: float = 0.0


class EnvState(State):
    task_id: str = ""
    steps_used: int = Field(default=0, ge=0)
    fuel_remaining: float = Field(default=0.0, ge=0.0)
    fuel_used: float = Field(default=0.0, ge=0.0)
    cumulative_reward: float = 0.0
    best_tracking_window: int = Field(default=0, ge=0)
    done: bool = False

    @field_validator("fuel_used")
    @classmethod
    def fuel_usage_nonnegative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("fuel_used must be non-negative")
        return value
