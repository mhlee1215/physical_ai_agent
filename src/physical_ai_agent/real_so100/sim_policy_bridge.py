from __future__ import annotations

import math
from typing import Mapping


JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
GRIPPER_CLOSED_RAD = math.radians(-10.0)
GRIPPER_OPEN_RAD = math.radians(100.0)
STS3215_POSITION_RESOLUTION = 4095.0


def raw_hardware_positions_to_lerobot_positions(
    raw_positions: Mapping[str, float],
    calibration: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Apply LeRobot's calibrated SO101 degree/percent normalization."""
    normalized: dict[str, float] = {}
    for joint in JOINT_ORDER:
        if joint not in raw_positions:
            raise ValueError(f"raw positions are missing joint {joint!r}")
        if joint not in calibration:
            raise ValueError(f"calibration is missing joint {joint!r}")
        raw = float(raw_positions[joint])
        lower = float(calibration[joint]["range_min"])
        upper = float(calibration[joint]["range_max"])
        if not all(math.isfinite(value) for value in (raw, lower, upper)) or lower >= upper:
            raise ValueError(f"invalid raw/calibration values for {joint}")
        if joint == "gripper":
            normalized[joint] = (raw - lower) * 100.0 / (upper - lower)
        else:
            midpoint = (lower + upper) / 2.0
            normalized[joint] = (raw - midpoint) * 360.0 / STS3215_POSITION_RESOLUTION
    return normalized


def raw_hardware_positions_to_sim_qpos(
    raw_positions: Mapping[str, float],
    calibration: Mapping[str, Mapping[str, float]],
) -> list[float]:
    return hardware_positions_to_sim_qpos(
        raw_hardware_positions_to_lerobot_positions(raw_positions, calibration)
    )


def hardware_positions_to_sim_qpos(positions: Mapping[str, float]) -> list[float]:
    values = []
    for joint in JOINT_ORDER:
        value = float(positions[joint])
        if joint == "gripper":
            values.append(GRIPPER_CLOSED_RAD + (value / 100.0) * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD))
        else:
            values.append(math.radians(value))
    return values


def sim_qpos_to_hardware_positions(qpos: list[float]) -> dict[str, float]:
    if len(qpos) != len(JOINT_ORDER):
        raise ValueError(f"expected {len(JOINT_ORDER)} qpos values, got {len(qpos)}")
    result = {joint: math.degrees(float(qpos[index])) for index, joint in enumerate(JOINT_ORDER[:-1])}
    gripper = (float(qpos[-1]) - GRIPPER_CLOSED_RAD) / (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD) * 100.0
    result["gripper"] = min(100.0, max(0.0, gripper))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("converted hardware action contains a non-finite value")
    return result


def hardware_position_limits_from_calibration(
    calibration: Mapping[str, Mapping[str, float]],
) -> dict[str, tuple[float, float]]:
    """Return the normalized position limits accepted by LeRobot's SO follower."""
    limits: dict[str, tuple[float, float]] = {}
    for joint in JOINT_ORDER:
        if joint not in calibration:
            raise ValueError(f"calibration is missing joint {joint!r}")
        lower_raw = float(calibration[joint]["range_min"])
        upper_raw = float(calibration[joint]["range_max"])
        if not math.isfinite(lower_raw) or not math.isfinite(upper_raw) or lower_raw >= upper_raw:
            raise ValueError(f"invalid calibration range for {joint}: {lower_raw}..{upper_raw}")
        if joint == "gripper":
            limits[joint] = (0.0, 100.0)
            continue
        half_range_degrees = (upper_raw - lower_raw) * 180.0 / STS3215_POSITION_RESOLUTION
        limits[joint] = (-half_range_degrees, half_range_degrees)
    return limits


def clamp_hardware_positions(
    positions: Mapping[str, float],
    limits: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Clamp normalized degree/percent targets to the active calibration."""
    clamped: dict[str, float] = {}
    changes: dict[str, dict[str, float]] = {}
    for joint in JOINT_ORDER:
        value = float(positions[joint])
        lower, upper = limits[joint]
        safe = min(float(upper), max(float(lower), value))
        clamped[joint] = safe
        if safe != value:
            changes[joint] = {"requested": value, "clamped": safe, "lower": lower, "upper": upper}
    return clamped, changes
