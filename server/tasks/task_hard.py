from server.tasks.base import ControlProfile, DisturbanceProfile, MissionTask, TaskConfig, TimedTarget


class HardTask(MissionTask):
    def __init__(self) -> None:
        super().__init__(
            TaskConfig(
                task_id="long_horizon_precision_hold",
                difficulty="hard",
                title="Long-Horizon Precision Hold",
                description="Maintain tight pointing throughout a disturbance-heavy compressed long-duration mission segment.",
                mission_brief="Hold a precision pointing mode for a long mission segment while budgeting every pulse against a hard fuel reserve.",
                mission_summary="The spacecraft must hold a precision pointing mode against persistent disturbance torques for a long horizon while conserving fuel.",
                initial_attitude_deg=(12.4, -8.1, 20.7),
                initial_rates_dps=(0.06, -0.05, 0.04),
                target_schedule=(
                    TimedTarget(
                        start_step=0,
                        phase="precision_hold",
                        attitude_deg=(12.0, -8.0, 20.0),
                        instruction="Stay inside the fine-pointing envelope for most of the mission, trimming disturbances while protecting the fuel reserve.",
                        deadline_step=480,
                        milestone="science_hold_complete",
                        recommended_modes=("hold", "trim", "safe_hold"),
                        fuel_reserve_target=14.0,
                        completion_tolerance_deg=1.0,
                        completion_rate_tolerance_dps=0.08,
                        completion_hold_steps=24,
                    ),
                ),
                anomaly_schedule=(),
                step_budget=480,
                time_step_seconds=90.0,
                inertia=(3.4, 3.1, 3.8),
                damping=(0.06, 0.06, 0.05),
                control_profile=ControlProfile(
                    small_impulse_dps=0.18,
                    large_impulse_dps=0.36,
                    small_fuel_cost=0.32,
                    large_fuel_cost=0.68,
                ),
                disturbance_profile=DisturbanceProfile(
                    seed=71,
                    bias_dps2=(0.0010, -0.0007, 0.0008),
                    min_amplitude_dps2=0.0008,
                    max_amplitude_dps2=0.0042,
                    frequency_scale=1.25,
                    reported_level=0.27,
                ),
                fuel_capacity=85.0,
                fuel_reserve_success=8.0,
                final_tolerance_deg=1.0,
                angular_rate_tolerance_dps=0.08,
                hold_streak_success=24,
                fuel_budget=32.0,
                mean_error_success=1.0,
                on_target_fraction_success=0.72,
                overshoot_budget=8.0,
                early_success_allowed=False,
                required_milestones=0,
                pointing_scale_deg=7.0,
                fuel_penalty_coeff=0.055,
                stability_penalty_coeff=0.07,
                overshoot_penalty_coeff=0.10,
                hold_bonus_coeff=0.008,
            )
        )
