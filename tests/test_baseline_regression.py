from inference import deterministic_controller, tuned_mission_controller
from models import OrbitalThrusterAction
from server.orbital_thruster_environment import OrbitalThrusterEnvironment


def _rollout(task_id: str, controller, seed: int = 0) -> tuple[bool, float]:
    del seed
    env = OrbitalThrusterEnvironment()
    observation = env.reset(task_id=task_id)
    while not observation.done:
        action_payload = controller(observation.model_dump())
        observation = env.step(OrbitalThrusterAction(**action_payload))
    return observation.success, observation.reward_so_far


def _random_controller(_observation: dict[str, object]) -> dict[str, str]:
    return {
        "action_type": "idle",
        "control_mode": "safe_hold",
        "reason": "random baseline stub",
    }


def test_baseline_relationships_support_theme_story() -> None:
    easy_success, _ = _rollout("detumble_satellite", deterministic_controller)
    medium_success, _ = _rollout("retarget_180_flip", tuned_mission_controller)
    flagship_success, _ = _rollout("mission_ops_long_horizon", tuned_mission_controller)
    random_success, _ = _rollout("mission_ops_long_horizon", _random_controller)

    assert easy_success is True
    assert medium_success is True
    assert flagship_success is False
    assert random_success is False
