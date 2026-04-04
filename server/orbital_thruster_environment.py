from __future__ import annotations

from collections import deque
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from models import AttitudeVector, EnvState, OrbitalThrusterAction, OrbitalThrusterObservation
except ModuleNotFoundError:
    from ..models import AttitudeVector, EnvState, OrbitalThrusterAction, OrbitalThrusterObservation

from server.dynamics import make_disturbance_function, propagate, signed_angle_error, vector_magnitude
from server.reward import RewardScorer
from server.tasks import get_task
from server.tasks.base import AXES, MissionTask


class OrbitalThrusterEnvironment(Environment[OrbitalThrusterAction, OrbitalThrusterObservation, EnvState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, generation: int = 0):
        super().__init__()
        self._generation = generation
        self._scorer = RewardScorer()
        self._task: MissionTask | None = None
        self._disturbance_coefficients: dict[str, list[tuple[float, float, float]]] = {}
        self._history_errors: list[float] = []
        self._on_target_steps = 0
        self._on_target_streak = 0
        self._best_tracking_window = 0
        self._overshoot_total = 0.0
        self._last_action: str | None = None
        self._last_feedback = "Reset required."
        self._done = False
        self._time_elapsed = 0.0
        self._rates = {axis: 0.0 for axis in AXES}
        self._attitude = {axis: 0.0 for axis in AXES}
        self._target = {axis: 0.0 for axis in AXES}
        self._errors = {axis: 0.0 for axis in AXES}
        self._disturbance_level = 0.0
        self._fuel_remaining = 0.0
        self._fuel_used = 0.0
        self._reward_so_far = 0.0
        self._recent_error_window: deque[float] = deque(maxlen=25)
        self._state = EnvState(episode_id=str(uuid4()), step_count=0)

    def reset(self, seed: int | None = None, episode_id: str | None = None, task_id: str = "detumble_satellite", **_: Any) -> OrbitalThrusterObservation:
        del seed
        self._task = get_task(task_id)
        self._disturbance_coefficients = make_disturbance_function(self._task)
        self._history_errors = []
        self._on_target_steps = 0
        self._on_target_streak = 0
        self._best_tracking_window = 0
        self._overshoot_total = 0.0
        self._last_action = None
        self._last_feedback = "Episode reset. Awaiting first control action."
        self._done = False
        self._time_elapsed = 0.0
        self._attitude = self._task.initial_attitude_map()
        self._rates = self._task.initial_rates_map()
        self._target = dict(zip(AXES, self._task.target_for_step(0)))
        self._errors = {
            axis: signed_angle_error(self._target[axis], self._attitude[axis])
            for axis in AXES
        }
        self._disturbance_level = self._task.config.disturbance_profile.reported_level
        self._fuel_remaining = self._task.config.fuel_capacity
        self._fuel_used = 0.0
        self._reward_so_far = 0.0
        self._recent_error_window = deque(maxlen=25)
        self._recent_error_window.append(self._error_norm())
        self._history_errors.append(self._error_norm())
        self._state = EnvState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            steps_used=0,
            fuel_remaining=self._fuel_remaining,
            fuel_used=self._fuel_used,
            cumulative_reward=0.0,
            best_tracking_window=0,
            done=False,
        )
        return self._make_observation(reward=0.0, success=False)

    def step(self, action: OrbitalThrusterAction, timeout_s: float | None = None, **_: Any) -> OrbitalThrusterObservation:
        del timeout_s
        if self._task is None:
            self.reset()
        if self._done:
            return self._make_observation(reward=0.0, success=self._success())

        assert self._task is not None
        previous_errors = dict(self._errors)
        self._state.step_count += 1
        step_number = self._state.step_count
        self._target = dict(zip(AXES, self._task.target_for_step(step_number)))

        dynamics_result = propagate(
            self._task,
            {
                "attitude": self._attitude,
                "rates": self._rates,
                "fuel_remaining": self._fuel_remaining,
            },
            action.action_type,
            step_number,
            self._disturbance_coefficients,
        )

        self._attitude = dynamics_result["attitude"]
        self._rates = dynamics_result["rates"]
        self._fuel_remaining = dynamics_result["fuel_remaining"]
        self._fuel_used += dynamics_result["fuel_used_step"]
        self._disturbance_level = dynamics_result["disturbance_level"]
        self._errors = {
            axis: signed_angle_error(self._target[axis], self._attitude[axis])
            for axis in AXES
        }

        overshoot_increment = 0.0
        if self._task.difficulty in {"medium", "hard"}:
            for axis in AXES:
                crossed = previous_errors[axis] != 0 and self._errors[axis] != 0 and (previous_errors[axis] > 0) != (self._errors[axis] > 0)
                if crossed and abs(previous_errors[axis]) > 0.75 and abs(self._rates[axis]) > self._task.config.angular_rate_tolerance_dps:
                    overshoot_increment += min(1.5, abs(self._rates[axis]))
        self._overshoot_total += overshoot_increment

        error_norm = self._error_norm()
        rate_norm = vector_magnitude(self._rates)
        if error_norm <= 1.0:
            self._on_target_steps += 1
            self._on_target_streak += 1
        else:
            self._on_target_streak = 0
        self._best_tracking_window = max(self._best_tracking_window, self._on_target_streak)
        self._history_errors.append(error_norm)
        self._recent_error_window.append(error_norm)

        reward, reward_terms = self._scorer.compute(
            self._task,
            {
                "error_norm": error_norm,
                "rate_norm": rate_norm,
                "fuel_used_step": dynamics_result["fuel_used_step"],
                "overshoot_increment": overshoot_increment,
                "on_target_streak": self._on_target_streak,
            },
        )
        self._reward_so_far = round(self._reward_so_far + reward, 6)
        self._time_elapsed = step_number * self._task.config.time_step_seconds
        self._last_action = action.action_type

        episode_score = self._episode_score(error_norm, rate_norm)
        success = self._success(error_norm=error_norm, rate_norm=rate_norm)
        self._last_feedback = self._scorer.feedback(
            self._task,
            {
                "error_norm": error_norm,
                "rate_norm": rate_norm,
                "fuel_remaining": self._fuel_remaining,
                "hold_streak": self._on_target_streak,
                "overshoot_total": self._overshoot_total,
            },
            reward_terms,
            episode_score,
            success,
        )

        if self._task.config.early_success_allowed and success:
            self._done = True
        elif step_number >= self._task.config.step_budget:
            self._done = True
        else:
            self._done = False

        self._state.task_id = self._task.task_id
        self._state.steps_used = step_number
        self._state.fuel_remaining = self._fuel_remaining
        self._state.fuel_used = self._fuel_used
        self._state.cumulative_reward = self._reward_so_far
        self._state.best_tracking_window = self._best_tracking_window
        self._state.done = self._done

        if self._done:
            success = self._success(error_norm=error_norm, rate_norm=rate_norm)

        return self._make_observation(reward=reward, success=success)

    @property
    def state(self) -> EnvState:
        return self._state

    def close(self) -> None:
        return None

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="OrbitalThrusterEnv",
            description="Fuel-aware spacecraft attitude-control benchmark with deterministic disturbance scenarios.",
            version="1.0.0",
            author="OpenAI Codex",
        )

    def _make_observation(self, reward: float, success: bool) -> OrbitalThrusterObservation:
        assert self._task is not None
        return OrbitalThrusterObservation(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            mission_phase=self._task.phase_for_step(self._state.step_count),
            current_attitude_deg=AttitudeVector(**self._attitude),
            current_angular_velocity_dps=AttitudeVector(**self._rates),
            target_attitude_deg=AttitudeVector(**self._target),
            attitude_error_deg=AttitudeVector(**self._errors),
            fuel_remaining=round(self._fuel_remaining, 6),
            fuel_used=round(self._fuel_used, 6),
            step_budget=self._task.config.step_budget,
            steps_used=self._state.step_count,
            time_elapsed=round(self._time_elapsed, 6),
            disturbance_level=self._disturbance_level,
            last_action=self._last_action,
            reward_so_far=round(self._reward_so_far, 6),
            last_feedback=self._last_feedback,
            done=self._done,
            reward=round(reward, 6),
            success=success,
        )

    def _success(self, error_norm: float | None = None, rate_norm: float | None = None) -> bool:
        assert self._task is not None
        current_rate_norm = vector_magnitude(self._rates) if rate_norm is None else rate_norm
        return self._scorer.is_success(
            self._task,
            {
                "final_max_axis_error": max(abs(value) for value in self._errors.values()),
                "final_rate": current_rate_norm,
                "fuel_remaining": self._fuel_remaining,
                "fuel_used": self._fuel_used,
                "hold_streak": self._on_target_streak,
                "mean_error": sum(self._history_errors) / max(len(self._history_errors), 1),
                "on_target_fraction": self._on_target_steps / max(self._state.step_count, 1),
                "overshoot_total": self._overshoot_total,
            },
        )

    def _episode_score(self, error_norm: float, rate_norm: float) -> float:
        assert self._task is not None
        return self._scorer.score_episode(
            self._task,
            {
                "mean_error": sum(self._history_errors) / max(len(self._history_errors), 1),
                "on_target_fraction": self._on_target_steps / max(self._state.step_count, 1),
                "fuel_used": self._fuel_used,
                "final_error": error_norm,
                "final_rate": rate_norm,
                "final_max_axis_error": max(abs(value) for value in self._errors.values()),
                "overshoot_total": self._overshoot_total,
                "hold_streak": self._on_target_streak,
                "fuel_remaining": self._fuel_remaining,
            },
        )

    def _error_norm(self) -> float:
        return vector_magnitude(self._errors)

