import logging

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from openenv.core.env_server import create_app

try:
    from models import OrbitalThrusterAction, OrbitalThrusterObservation
except ModuleNotFoundError:
    from ..models import OrbitalThrusterAction, OrbitalThrusterObservation

from server.orbital_thruster_environment import OrbitalThrusterEnvironment
from server.tasks import list_tasks


logging.basicConfig(level=logging.INFO)

ENV_GENERATION = 0
ACTIVE_ENV = OrbitalThrusterEnvironment(generation=ENV_GENERATION)


def build_environment() -> OrbitalThrusterEnvironment:
    return ACTIVE_ENV


app = create_app(
    build_environment,
    OrbitalThrusterAction,
    OrbitalThrusterObservation,
    env_name="orbital_thruster_env",
    max_concurrent_envs=1,
)
app.router.routes = [
    route
    for route in app.router.routes
    if not (getattr(route, "path", None) == "/state" and "GET" in getattr(route, "methods", set()))
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    logging.info("%s %s -> %s", request.method, request.url.path, response.status_code)
    return response


@app.get("/")
async def root():
    return {
        "name": "orbital-thruster-env",
        "version": "1.0.0",
        "status": "running",
        "description": "OpenEnv mission-operations benchmark for long-horizon spacecraft control with fuel, directives, milestones, and anomaly recovery",
        "tasks": list_tasks(),
    }


@app.get("/tasks")
async def get_tasks():
    return list_tasks()


@app.get("/state")
async def get_state():
    return build_environment().state.model_dump()


@app.post("/reset_hard")
async def reset_hard():
    global ENV_GENERATION, ACTIVE_ENV
    ENV_GENERATION += 1
    ACTIVE_ENV = OrbitalThrusterEnvironment(generation=ENV_GENERATION)
    return {"status": "reset", "generation": ENV_GENERATION}


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
