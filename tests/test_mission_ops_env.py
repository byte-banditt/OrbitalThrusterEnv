import asyncio

from models import OrbitalThrusterAction
from server.orbital_thruster_environment import OrbitalThrusterEnvironment
from server.reward import RewardScorer
from server.tasks import get_task


FLAGSHIP_TASK_ID = "mission_ops_long_horizon"


def test_flagship_task_has_long_horizon_directives_and_anomalies() -> None:
    task = get_task(FLAGSHIP_TASK_ID)

    first_directive = task.directive_for_step(0)
    final_directive = task.directive_for_step(260)

    assert task.difficulty == "hard"
    assert "long-horizon" in task.config.mission_brief.lower()
    assert first_directive.phase == "detumble_window"
    assert final_directive.phase == "precision_hold"
    assert first_directive.instruction
    assert first_directive.deadline_step > first_directive.start_step
    assert "gyro_bias_spike" in task.anomaly_flags_for_step(180)


def test_reward_scorer_emits_rubric_columns() -> None:
    task = get_task(FLAGSHIP_TASK_ID)
    reward, columns = RewardScorer().compute(
        task,
        {
            "error_norm": 0.6,
            "rate_norm": 0.05,
            "fuel_used_step": 0.2,
            "overshoot_increment": 0.0,
            "on_target_streak": 5,
            "control_mode": "recover",
            "recommended_modes": ("recover", "safe_hold"),
            "directive_completed": True,
            "anomaly_flags": ["gyro_bias_spike"],
            "error_improvement": 0.4,
            "rate_improvement": 0.08,
            "stall_steps": 0,
            "fuel_remaining": 48.0,
            "fuel_reserve_target": 18.0,
        },
    )

    assert reward <= 1.0
    assert reward >= -1.0
    assert columns["physical_tracking_reward"] > 0.0
    assert columns["milestone_completion_reward"] > 0.0
    assert columns["control_mode_reward"] > 0.0
    assert columns["anomaly_recovery_reward"] > 0.0
    assert "anti_stall_penalty" in columns


def test_reset_and_step_surface_mission_fields() -> None:
    env = OrbitalThrusterEnvironment()
    observation = env.reset(task_id=FLAGSHIP_TASK_ID)

    assert observation.mission_brief
    assert observation.active_directive
    assert observation.pending_directives_count > 0
    assert observation.milestones_completed == []
    assert observation.fuel_reserve_target > 0.0
    assert observation.phase_deadline_step > 0
    assert "physical_tracking_reward" in observation.reward_breakdown
    assert "milestones_completed_count" in observation.episode_metrics

    next_observation = env.step(
        OrbitalThrusterAction(
            action_type="idle",
            control_mode="safe_hold",
            reason="validation probe",
        )
    )

    assert next_observation.last_action == "idle"
    assert next_observation.active_directive
    assert "control_mode_reward" in next_observation.reward_breakdown
    assert isinstance(env.state.reward_columns, dict)


def test_root_and_tasks_surface_flagship_task() -> None:
    from server.app import get_tasks, root

    root_payload = asyncio.run(root())
    tasks_payload = asyncio.run(get_tasks())

    assert root_payload["name"] == "orbital-thruster-env"
    assert any(task["id"] == FLAGSHIP_TASK_ID for task in root_payload["tasks"])
    assert any(task["id"] == FLAGSHIP_TASK_ID for task in tasks_payload)
