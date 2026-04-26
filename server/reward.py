from typing import Any

from server.dynamics import vector_magnitude
from server.tasks.base import MissionTask


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class RewardScorer:
    def compute(self, task: MissionTask, telemetry: dict[str, Any]) -> tuple[float, dict[str, float]]:
        config = task.config
        error_norm = float(telemetry["error_norm"])
        rate_norm = float(telemetry["rate_norm"])
        fuel_used_step = float(telemetry["fuel_used_step"])
        overshoot_increment = float(telemetry["overshoot_increment"])
        on_target_streak = int(telemetry["on_target_streak"])
        control_mode = str(telemetry.get("control_mode", "hold"))
        recommended_modes = tuple(str(mode) for mode in telemetry.get("recommended_modes", ()))
        directive_completed = bool(telemetry.get("directive_completed", False))
        anomaly_flags = list(telemetry.get("anomaly_flags", ()))
        error_improvement = max(0.0, float(telemetry.get("error_improvement", 0.0)))
        rate_improvement = max(0.0, float(telemetry.get("rate_improvement", 0.0)))
        stall_steps = int(telemetry.get("stall_steps", 0))
        fuel_remaining = float(telemetry.get("fuel_remaining", 0.0))
        fuel_reserve_target = float(telemetry.get("fuel_reserve_target", 0.0))

        if error_norm <= 1.0:
            pointing = 1.0
        else:
            pointing = clamp(1.0 - ((error_norm - 1.0) / max(config.pointing_scale_deg - 1.0, 1.0)), 0.0, 1.0)

        fuel_penalty = config.fuel_penalty_coeff * fuel_used_step
        stability_penalty = config.stability_penalty_coeff * rate_norm if error_norm < config.pointing_scale_deg * 0.55 else 0.0
        overshoot_penalty = config.overshoot_penalty_coeff * overshoot_increment
        hold_bonus = min(0.18, config.hold_bonus_coeff * on_target_streak) if error_norm <= 1.0 else 0.0

        physical_tracking_reward = clamp(pointing + hold_bonus - stability_penalty - overshoot_penalty, -1.0, 1.0)

        reserve_gap = max(0.0, fuel_reserve_target - fuel_remaining)
        reserve_penalty = min(0.28, reserve_gap * 0.012)
        fuel_discipline_reward = clamp(0.10 - fuel_penalty - reserve_penalty, -0.4, 0.18)

        milestone_completion_reward = 0.35 if directive_completed else 0.0
        if recommended_modes:
            control_mode_reward = 0.12 if control_mode in recommended_modes else -0.08
        else:
            control_mode_reward = 0.0

        anomaly_recovery_reward = 0.0
        if anomaly_flags:
            anomaly_recovery_reward += min(0.18, (0.22 * error_improvement) + (0.40 * rate_improvement))
            if control_mode in {"recover", "safe_hold"}:
                anomaly_recovery_reward += 0.08

        anti_stall_penalty = min(0.24, 0.03 * stall_steps)

        reward = clamp(
            physical_tracking_reward
            + fuel_discipline_reward
            + milestone_completion_reward
            + control_mode_reward
            + anomaly_recovery_reward
            - anti_stall_penalty,
            -1.0,
            1.0,
        )
        return reward, {
            "physical_tracking_reward": round(physical_tracking_reward, 6),
            "fuel_discipline_reward": round(fuel_discipline_reward, 6),
            "milestone_completion_reward": round(milestone_completion_reward, 6),
            "control_mode_reward": round(control_mode_reward, 6),
            "anomaly_recovery_reward": round(anomaly_recovery_reward, 6),
            "anti_stall_penalty": round(anti_stall_penalty, 6),
            "pointing": round(pointing, 6),
            "fuel_penalty": round(fuel_penalty, 6),
            "stability_penalty": round(stability_penalty, 6),
            "overshoot_penalty": round(overshoot_penalty, 6),
            "hold_bonus": round(hold_bonus, 6),
        }

    def score_episode(self, task: MissionTask, telemetry: dict[str, Any]) -> float:
        config = task.config
        mean_error = float(telemetry["mean_error"])
        on_target_fraction = float(telemetry["on_target_fraction"])
        fuel_used = float(telemetry["fuel_used"])
        final_error = float(telemetry["final_error"])
        final_rate = float(telemetry["final_rate"])
        overshoot_total = float(telemetry["overshoot_total"])
        hold_streak = int(telemetry["hold_streak"])
        milestones_completed_count = int(telemetry.get("milestones_completed_count", 0))

        accuracy_score = clamp(1.0 - (mean_error / max(config.mean_error_success * 2.5, 1.0)), 0.0, 1.0)
        fuel_score = clamp(1.0 - (fuel_used / max(config.fuel_budget * 1.6, 1.0)), 0.0, 1.0)
        final_score = clamp(1.0 - (final_error / max(config.final_tolerance_deg * 2.5, 1.0)), 0.0, 1.0)
        stability_score = clamp(1.0 - (final_rate / max(config.angular_rate_tolerance_dps * 5.0, 0.5)), 0.0, 1.0)
        fraction_score = clamp(on_target_fraction / max(config.on_target_fraction_success, 0.1), 0.0, 1.0)
        overshoot_score = clamp(1.0 - (overshoot_total / max(config.overshoot_budget * 2.0, 1.0)), 0.0, 1.0)
        hold_score = clamp(hold_streak / max(config.hold_streak_success, 1), 0.0, 1.0)
        milestone_score = 1.0 if config.required_milestones <= 0 else clamp(milestones_completed_count / config.required_milestones, 0.0, 1.0)

        if task.difficulty == "hard":
            total = (
                0.28 * accuracy_score
                + 0.20 * fraction_score
                + 0.16 * fuel_score
                + 0.14 * final_score
                + 0.07 * stability_score
                + 0.05 * hold_score
                + 0.10 * milestone_score
            )
        elif task.difficulty == "medium":
            total = (
                0.30 * accuracy_score
                + 0.20 * final_score
                + 0.18 * fuel_score
                + 0.14 * overshoot_score
                + 0.10 * stability_score
                + 0.08 * hold_score
            )
        else:
            total = (
                0.34 * accuracy_score
                + 0.24 * final_score
                + 0.16 * stability_score
                + 0.14 * fuel_score
                + 0.12 * hold_score
            )
        return round(clamp(total, 0.0, 1.0), 6)

    def is_success(self, task: MissionTask, telemetry: dict[str, Any]) -> bool:
        config = task.config
        final_max_axis_error = float(telemetry["final_max_axis_error"])
        final_rate = float(telemetry["final_rate"])
        fuel_remaining = float(telemetry["fuel_remaining"])
        fuel_used = float(telemetry["fuel_used"])
        hold_streak = int(telemetry["hold_streak"])
        mean_error = float(telemetry["mean_error"])
        on_target_fraction = float(telemetry["on_target_fraction"])
        overshoot_total = float(telemetry["overshoot_total"])
        milestones_completed_count = int(telemetry.get("milestones_completed_count", 0))

        base_conditions = (
            final_max_axis_error <= config.final_tolerance_deg
            and final_rate <= config.angular_rate_tolerance_dps
            and fuel_remaining >= config.fuel_reserve_success
        )

        if task.difficulty == "hard":
            return (
                base_conditions
                and mean_error <= config.mean_error_success
                and on_target_fraction >= config.on_target_fraction_success
                and fuel_used <= config.fuel_budget
                and hold_streak >= config.hold_streak_success
                and (
                    config.required_milestones <= 0
                    or milestones_completed_count >= config.required_milestones
                )
            )

        if task.difficulty == "medium":
            return (
                base_conditions
                and overshoot_total <= config.overshoot_budget
                and hold_streak >= config.hold_streak_success
            )

        return base_conditions and hold_streak >= config.hold_streak_success

    def feedback(self, task: MissionTask, telemetry: dict[str, Any], reward_terms: dict[str, float], episode_score: float, success: bool) -> str:
        error_norm = telemetry["error_norm"]
        rate_norm = telemetry["rate_norm"]
        fuel_remaining = telemetry["fuel_remaining"]
        hold_streak = telemetry["hold_streak"]
        overshoot_total = telemetry["overshoot_total"]
        anomaly_flags = telemetry.get("anomaly_flags", [])
        return (
            f"error_norm={error_norm:.3f}deg rate_norm={rate_norm:.3f}deg/s "
            f"fuel_remaining={fuel_remaining:.2f} hold_streak={hold_streak} "
            f"overshoot_total={overshoot_total:.2f} anomaly_flags={','.join(anomaly_flags) or 'none'} "
            f"tracking={reward_terms['physical_tracking_reward']:.3f} "
            f"fuel_discipline={reward_terms['fuel_discipline_reward']:.3f} "
            f"milestone={reward_terms['milestone_completion_reward']:.3f} "
            f"mode={reward_terms['control_mode_reward']:.3f} "
            f"recovery={reward_terms['anomaly_recovery_reward']:.3f} "
            f"stall={reward_terms['anti_stall_penalty']:.3f} "
            f"episode_score={episode_score:.3f} success={success}"
        )
