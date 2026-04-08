import math
from abc import ABC
from dataclasses import dataclass
from typing import Iterable

AXES = ("pitch", "roll", "yaw")


@dataclass(frozen=True)
class TimedTarget:
    start_step: int
    phase: str
    attitude_deg: tuple[float, float, float]


@dataclass(frozen=True)
class ControlProfile:
    small_impulse_dps: float
    large_impulse_dps: float
    small_fuel_cost: float
    large_fuel_cost: float


@dataclass(frozen=True)
class DisturbanceProfile:
    seed: int
    bias_dps2: tuple[float, float, float]
    min_amplitude_dps2: float
    max_amplitude_dps2: float
    frequency_scale: float
    reported_level: float


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    difficulty: str
    title: str
    description: str
    mission_summary: str
    initial_attitude_deg: tuple[float, float, float]
    initial_rates_dps: tuple[float, float, float]
    target_schedule: tuple[TimedTarget, ...]
    step_budget: int
    time_step_seconds: float
    inertia: tuple[float, float, float]
    damping: tuple[float, float, float]
    control_profile: ControlProfile
    disturbance_profile: DisturbanceProfile
    fuel_capacity: float
    fuel_reserve_success: float
    final_tolerance_deg: float
    angular_rate_tolerance_dps: float
    hold_streak_success: int
    fuel_budget: float
    mean_error_success: float
    on_target_fraction_success: float
    overshoot_budget: float
    early_success_allowed: bool
    pointing_scale_deg: float
    fuel_penalty_coeff: float
    stability_penalty_coeff: float
    overshoot_penalty_coeff: float
    hold_bonus_coeff: float


class MissionTask(ABC):
    def __init__(self, config: TaskConfig):
        self.config = config

    @property
    def task_id(self) -> str:
        return self.config.task_id

    @property
    def difficulty(self) -> str:
        return self.config.difficulty

    def target_for_step(self, step: int) -> tuple[float, float, float]:
        target = self.config.target_schedule[0]
        for candidate in self.config.target_schedule:
            if step >= candidate.start_step:
                target = candidate
            else:
                break
        return target.attitude_deg

    def phase_for_step(self, step: int) -> str:
        target = self.config.target_schedule[0]
        for candidate in self.config.target_schedule:
            if step >= candidate.start_step:
                target = candidate
            else:
                break
        return target.phase

    def target_switch_step(self) -> int:
        if len(self.config.target_schedule) < 2:
            return 0
        return self.config.target_schedule[1].start_step

    def inertia_map(self) -> dict[str, float]:
        return dict(zip(AXES, self.config.inertia))

    def damping_map(self) -> dict[str, float]:
        return dict(zip(AXES, self.config.damping))

    def initial_attitude_map(self) -> dict[str, float]:
        return dict(zip(AXES, self.config.initial_attitude_deg))

    def initial_rates_map(self) -> dict[str, float]:
        return dict(zip(AXES, self.config.initial_rates_dps))

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.config.task_id,
            "difficulty": self.config.difficulty,
            "title": self.config.title,
            "description": self.config.description,
            "step_budget": self.config.step_budget,
            "fuel_capacity": self.config.fuel_capacity,
            "mission_summary": self.config.mission_summary,
        }


def vector_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(component * component for component in values))
