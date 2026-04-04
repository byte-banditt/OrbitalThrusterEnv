from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

try:
    from .models import EnvState, OrbitalThrusterAction, OrbitalThrusterObservation
except ImportError:
    from models import EnvState, OrbitalThrusterAction, OrbitalThrusterObservation


class OrbitalThrusterEnv(EnvClient[OrbitalThrusterAction, OrbitalThrusterObservation, EnvState]):
    def _step_payload(self, action: OrbitalThrusterAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[OrbitalThrusterObservation]:
        observation_data = dict(payload.get("observation", {}))
        observation_data.setdefault("reward", payload.get("reward", 0.0) or 0.0)
        observation_data.setdefault("done", payload.get("done", False))
        observation = OrbitalThrusterObservation(**observation_data)
        return StepResult(
            observation=observation,
            reward=float(payload.get("reward") or 0.0),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict[str, Any]) -> EnvState:
        return EnvState(**payload)
