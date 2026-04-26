import sys
import types

from fastapi import FastAPI
from pydantic import BaseModel


def _install_openenv_stubs() -> None:
    if "openenv.core.env_server.types" in sys.modules:
        return

    mod_openenv = types.ModuleType("openenv")
    mod_core = types.ModuleType("openenv.core")
    mod_env_server = types.ModuleType("openenv.core.env_server")
    mod_interfaces = types.ModuleType("openenv.core.env_server.interfaces")
    mod_types = types.ModuleType("openenv.core.env_server.types")
    mod_client_types = types.ModuleType("openenv.core.client_types")
    mod_env_client = types.ModuleType("openenv.core.env_client")

    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        pass

    class State(BaseModel):
        episode_id: str = ""
        step_count: int = 0

    class EnvironmentMetadata(BaseModel):
        name: str = ""
        description: str = ""
        version: str = ""
        author: str = ""

    class Environment:
        SUPPORTS_CONCURRENT_SESSIONS = False

        def __init__(self) -> None:
            pass

        def __class_getitem__(cls, _item):
            return cls

    class StepResult(BaseModel):
        observation: object
        reward: float
        done: bool

    class EnvClient:
        def __class_getitem__(cls, _item):
            return cls

    def create_app(*_args, **_kwargs) -> FastAPI:
        app = FastAPI()

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "healthy"}

        @app.get("/state")
        async def state() -> dict[str, str]:
            return {"episode_id": "stubbed"}

        return app

    mod_types.Action = Action
    mod_types.Observation = Observation
    mod_types.State = State
    mod_types.EnvironmentMetadata = EnvironmentMetadata
    mod_interfaces.Environment = Environment
    mod_env_server.create_app = create_app
    mod_client_types.StepResult = StepResult
    mod_env_client.EnvClient = EnvClient

    sys.modules["openenv"] = mod_openenv
    sys.modules["openenv.core"] = mod_core
    sys.modules["openenv.core.env_server"] = mod_env_server
    sys.modules["openenv.core.env_server.interfaces"] = mod_interfaces
    sys.modules["openenv.core.env_server.types"] = mod_types
    sys.modules["openenv.core.client_types"] = mod_client_types
    sys.modules["openenv.core.env_client"] = mod_env_client


_install_openenv_stubs()
