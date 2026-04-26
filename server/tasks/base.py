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
    instruction: str = ""
    deadline_step: int = 0
    milestone: str = ""
    recommended_modes: tuple[str, ...] = ()
    fuel_reserve_target: float = 0.0
    completion_tolerance_deg: float | None = None
    completion_rate_tolerance_dps: float | None = None
    completion_hold_steps: int = 0


@dataclass(frozen=True)
class MissionAnomaly:
    start_step: int
    end_step: int
    flag: str
    note: str
    disturbance_scale: float = 1.0
    rate_bias_dps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    recommended_modes: tuple[str, ...] = ("recover", "safe_hold")


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
    mission_brief: str
    mission_summary: str
    initial_attitude_deg: tuple[float, float, float]
    initial_rates_dps: tuple[float, float, float]
    target_schedule: tuple[TimedTarget, ...]
    anomaly_schedule: tuple[MissionAnomaly, ...]
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
    required_milestones: int
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

    def directive_for_step(self, step: int) -> TimedTarget:
        target = self.config.target_schedule[0]
        for candidate in self.config.target_schedule:
            if step >= candidate.start_step:
                target = candidate
            else:
                break
        return target

    def target_for_step(self, step: int) -> tuple[float, float, float]:
        return self.directive_for_step(step).attitude_deg

    def phase_for_step(self, step: int) -> str:
        return self.directive_for_step(step).phase

    def pending_directives_count(self, step: int) -> int:
        return sum(1 for directive in self.config.target_schedule if directive.start_step > step)

    def anomalies_for_step(self, step: int) -> tuple[MissionAnomaly, ...]:
        return tuple(
            anomaly
            for anomaly in self.config.anomaly_schedule
            if anomaly.start_step <= step <= anomaly.end_step
        )

    def anomaly_flags_for_step(self, step: int) -> list[str]:
        return [anomaly.flag for anomaly in self.anomalies_for_step(step)]

    def anomaly_rate_bias_for_step(self, step: int) -> dict[str, float]:
        total = {axis: 0.0 for axis in AXES}
        for anomaly in self.anomalies_for_step(step):
            for axis, value in zip(AXES, anomaly.rate_bias_dps):
                total[axis] += value
        return total

    def disturbance_scale_for_step(self, step: int) -> float:
        scale = 1.0
        for anomaly in self.anomalies_for_step(step):
            scale *= anomaly.disturbance_scale
        return scale

    def recommended_modes_for_step(self, step: int) -> tuple[str, ...]:
        directive = self.directive_for_step(step)
        modes = list(directive.recommended_modes)
        for anomaly in self.anomalies_for_step(step):
            for mode in anomaly.recommended_modes:
                if mode not in modes:
                    modes.append(mode)
        return tuple(modes)

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
            "mission_brief": self.config.mission_brief,
            "mission_summary": self.config.mission_summary,
            "directive_count": len(self.config.target_schedule),
            "anomaly_flags": [anomaly.flag for anomaly in self.config.anomaly_schedule],
        }


def vector_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(component * component for component in values))
