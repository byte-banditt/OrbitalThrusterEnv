from server.tasks.base import ControlProfile, DisturbanceProfile, MissionTask, TaskConfig, TimedTarget


class MediumTask(MissionTask):
    def __init__(self) -> None:
        super().__init__(
            TaskConfig(
                task_id="retarget_180_flip",
                difficulty="medium",
                title="Retarget High-Gain Antenna",
                description="Execute a near-180 degree slew, brake cleanly, and settle without overshoot.",
                mission_summary="The spacecraft must repoint a high-gain antenna to a new ground station geometry.",
                initial_attitude_deg=(0.0, -1.0, 2.0),
                initial_rates_dps=(0.02, -0.03, 0.01),
                target_schedule=(
                    TimedTarget(start_step=0, phase="tracking-old-target", attitude_deg=(0.0, 0.0, 0.0)),
                    TimedTarget(start_step=18, phase="slew-to-new-target", attitude_deg=(178.0, 14.0, -9.0)),
                ),
                step_budget=180,
                time_step_seconds=1.8,
                inertia=(2.6, 2.4, 2.9),
                damping=(0.16, 0.15, 0.13),
                control_profile=ControlProfile(
                    small_impulse_dps=0.48,
                    large_impulse_dps=0.95,
                    small_fuel_cost=0.75,
                    large_fuel_cost=1.45,
                ),
                disturbance_profile=DisturbanceProfile(
                    seed=29,
                    bias_dps2=(0.0025, -0.0015, 0.0012),
                    min_amplitude_dps2=0.0010,
                    max_amplitude_dps2=0.0030,
                    frequency_scale=0.85,
                    reported_level=0.16,
                ),
                fuel_capacity=120.0,
                fuel_reserve_success=18.0,
                final_tolerance_deg=1.5,
                angular_rate_tolerance_dps=0.12,
                hold_streak_success=16,
                fuel_budget=78.0,
                mean_error_success=5.0,
                on_target_fraction_success=0.20,
                overshoot_budget=3.5,
                early_success_allowed=True,
                pointing_scale_deg=22.0,
                fuel_penalty_coeff=0.035,
                stability_penalty_coeff=0.05,
                overshoot_penalty_coeff=0.08,
                hold_bonus_coeff=0.010,
            )
        )
