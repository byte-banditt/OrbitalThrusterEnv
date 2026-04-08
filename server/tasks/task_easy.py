from server.tasks.base import ControlProfile, DisturbanceProfile, MissionTask, TaskConfig, TimedTarget


class EasyTask(MissionTask):
    def __init__(self) -> None:
        super().__init__(
            TaskConfig(
                task_id="detumble_satellite",
                difficulty="easy",
                title="Detumble Deployed Spacecraft",
                description="Cancel deployment spin and settle on the initial communications attitude.",
                mission_summary="A newly released spacecraft must damp residual rates without wasting fuel.",
                initial_attitude_deg=(7.0, -4.0, 11.0),
                initial_rates_dps=(0.65, -0.45, 0.55),
                target_schedule=(TimedTarget(start_step=0, phase="detumble", attitude_deg=(0.0, 0.0, 0.0)),),
                step_budget=96,
                time_step_seconds=2.0,
                inertia=(2.4, 2.1, 2.7),
                damping=(0.22, 0.20, 0.18),
                control_profile=ControlProfile(
                    small_impulse_dps=0.42,
                    large_impulse_dps=0.78,
                    small_fuel_cost=0.55,
                    large_fuel_cost=1.0,
                ),
                disturbance_profile=DisturbanceProfile(
                    seed=11,
                    bias_dps2=(0.0015, -0.0010, 0.0010),
                    min_amplitude_dps2=0.0005,
                    max_amplitude_dps2=0.0015,
                    frequency_scale=0.55,
                    reported_level=0.08,
                ),
                fuel_capacity=100.0,
                fuel_reserve_success=40.0,
                final_tolerance_deg=2.0,
                angular_rate_tolerance_dps=0.15,
                hold_streak_success=10,
                fuel_budget=35.0,
                mean_error_success=2.5,
                on_target_fraction_success=0.25,
                overshoot_budget=6.0,
                early_success_allowed=True,
                pointing_scale_deg=14.0,
                fuel_penalty_coeff=0.025,
                stability_penalty_coeff=0.04,
                overshoot_penalty_coeff=0.0,
                hold_bonus_coeff=0.012,
            )
        )
