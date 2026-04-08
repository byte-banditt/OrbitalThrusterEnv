# OrbitalThrusterEnv

OrbitalThrusterEnv is an OpenEnv benchmark for satellite attitude control under mission-operations constraints. The agent must stabilize, reorient, and hold a spacecraft on target using a limited reaction-control-system fuel supply while deterministic environmental disturbances push the vehicle off attitude.

The environment is designed to reward planning instead of brute-force firing. Easy episodes focus on detumbling, medium episodes require a controlled 180-degree retargeting maneuver without overshoot, and hard episodes compress a long-horizon precision-pointing task into a validator-safe simulation with persistent seeded disturbances.

## Environment Summary

| Parameter | Value |
| --- | --- |
| Name | `orbital-thruster-env` |
| Version | `1.0.0` |
| Runtime | FastAPI + OpenEnv |
| Step horizon | 96 / 180 / 480 depending on task |
| Action space | 13 discrete actions |
| Reward range | `[-1.0, 1.0]` per step |
| Success mode | Explicit task thresholds plus dense reward |

## Tasks

### `detumble_satellite`
- Difficulty: `easy`
- Goal: eliminate deployment spin and settle near the target attitude.
- Success: final pointing error within 2 degrees on every axis, low angular rate, and positive fuel reserve.

### `retarget_180_flip`
- Difficulty: `medium`
- Goal: execute a large slew to a new target attitude and settle without excessive overshoot.
- Success: final pointing error within 1.5 degrees, bounded overshoot, and a sustained settle streak.

### `long_horizon_precision_hold`
- Difficulty: `hard`
- Goal: maintain precision pointing throughout a disturbance-heavy compressed long-duration mission segment.
- Success: low mean pointing error, high fraction of time within 1 degree, and fuel usage below budget.

## Observation Highlights

Each observation contains current attitude, angular velocity, target attitude, signed error, fuel state, mission phase, disturbance level, cumulative reward, and the last feedback string.

## Action Space

The agent may choose one action per step:
- `fire_pitch_pos_small`, `fire_pitch_neg_small`
- `fire_roll_pos_small`, `fire_roll_neg_small`
- `fire_yaw_pos_small`, `fire_yaw_neg_small`
- `fire_pitch_pos_large`, `fire_pitch_neg_large`
- `fire_roll_pos_large`, `fire_roll_neg_large`
- `fire_yaw_pos_large`, `fire_yaw_neg_large`
- `idle`

## Reward Design

Per-step reward combines a pointing term, a fuel penalty, a stability penalty near target, an overshoot penalty on harder tasks, and a hold bonus for consecutive on-target steps.

## API

The environment exposes:
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /health`
- `GET /`
- `GET /tasks`
- `POST /reset_hard`
- `WS /ws`

## Local Usage

```powershell
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
python validate.py
```

## Docker

```powershell
docker build -t orbital-thruster-env .
docker run -p 7860:7860 orbital-thruster-env
```

## Inference

Set `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`, then run:

```powershell
python inference.py
```
