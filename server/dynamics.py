import math
import random
from typing import Any

from server.tasks.base import AXES, MissionTask


def wrap_angle_deg(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def signed_angle_error(target: float, current: float) -> float:
    return wrap_angle_deg(target - current)


def vector_magnitude(values: dict[str, float]) -> float:
    return math.sqrt(sum(component * component for component in values.values()))


def make_disturbance_function(task: MissionTask) -> dict[str, list[tuple[float, float, float]]]:
    profile = task.config.disturbance_profile
    coefficients: dict[str, list[tuple[float, float, float]]] = {}
    for index, axis in enumerate(AXES):
        rng = random.Random((profile.seed * 97) + (index * 13))
        components: list[tuple[float, float, float]] = []
        for component_index in range(3):
            amplitude = rng.uniform(profile.min_amplitude_dps2, profile.max_amplitude_dps2)
            frequency = rng.uniform(0.04, 0.14) * profile.frequency_scale * (1.0 + component_index * 0.15)
            phase = rng.uniform(-math.pi, math.pi)
            components.append((amplitude, frequency, phase))
        coefficients[axis] = components
    return coefficients


def disturbance_vector(task: MissionTask, step_number: int, coefficients: dict[str, list[tuple[float, float, float]]]) -> dict[str, float]:
    profile = task.config.disturbance_profile
    vector: dict[str, float] = {}
    for axis_index, axis in enumerate(AXES):
        bias = profile.bias_dps2[axis_index]
        value = bias
        for amplitude, frequency, phase in coefficients[axis]:
            value += amplitude * math.sin((step_number + 1) * frequency + phase)
            value += 0.5 * amplitude * math.cos((step_number + 1) * frequency * 0.7 + phase * 0.5)
        vector[axis] = value
    return vector


def parse_action(action_type: str, control_small: float, control_large: float, fuel_small: float, fuel_large: float) -> tuple[str | None, float, float]:
    if action_type == "idle":
        return None, 0.0, 0.0

    parts = action_type.split("_")
    axis = parts[1]
    sign = 1.0 if parts[2] == "pos" else -1.0
    magnitude = control_small if parts[3] == "small" else control_large
    fuel = fuel_small if parts[3] == "small" else fuel_large
    return axis, sign * magnitude, fuel


def propagate(task: MissionTask, snapshot: dict[str, Any], action_type: str, step_number: int, coefficients: dict[str, list[tuple[float, float, float]]]) -> dict[str, Any]:
    config = task.config
    current_attitude = dict(snapshot["attitude"])
    current_rates = dict(snapshot["rates"])
    fuel_remaining = float(snapshot["fuel_remaining"])

    axis, pulse_delta, requested_fuel = parse_action(
        action_type,
        config.control_profile.small_impulse_dps,
        config.control_profile.large_impulse_dps,
        config.control_profile.small_fuel_cost,
        config.control_profile.large_fuel_cost,
    )

    actual_fuel = min(fuel_remaining, requested_fuel)
    fuel_ratio = 0.0 if requested_fuel <= 0 else actual_fuel / requested_fuel
    disturbance = disturbance_vector(task, step_number, coefficients)
    inertia = task.inertia_map()
    damping = task.damping_map()
    next_rates: dict[str, float] = {}
    next_attitude: dict[str, float] = {}
    pulse_vector = {axis_name: 0.0 for axis_name in AXES}

    if axis is not None:
        pulse_vector[axis] = pulse_delta * fuel_ratio

    for axis_name in AXES:
        damping_factor = max(0.0, 1.0 - (damping[axis_name] * config.time_step_seconds / inertia[axis_name]))
        rate = current_rates[axis_name] * damping_factor
        rate += pulse_vector[axis_name] / inertia[axis_name]
        rate += disturbance[axis_name] * config.time_step_seconds
        angle = wrap_angle_deg(current_attitude[axis_name] + (rate * config.time_step_seconds))
        next_rates[axis_name] = rate
        next_attitude[axis_name] = angle

    return {
        "attitude": next_attitude,
        "rates": next_rates,
        "fuel_remaining": max(0.0, fuel_remaining - actual_fuel),
        "fuel_used_step": actual_fuel,
        "pulse_vector": pulse_vector,
        "disturbance_vector": disturbance,
        "disturbance_level": config.disturbance_profile.reported_level,
    }
