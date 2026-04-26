from server.tasks.task_easy import EasyTask
from server.tasks.task_flagship import FlagshipMissionTask
from server.tasks.task_hard import HardTask
from server.tasks.task_medium import MediumTask


TASK_REGISTRY = {
    "detumble_satellite": EasyTask(),
    "retarget_180_flip": MediumTask(),
    "long_horizon_precision_hold": HardTask(),
    "mission_ops_long_horizon": FlagshipMissionTask(),
}


def get_task(task_id: str):
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_id: {task_id}")
    return TASK_REGISTRY[task_id]


def list_tasks() -> list[dict[str, object]]:
    return [task.as_dict() for task in TASK_REGISTRY.values()]
