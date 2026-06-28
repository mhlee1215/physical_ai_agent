from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


SO101_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class VisualServoError:
    wrist_dx_norm: float
    wrist_dy_norm: float
    edge_angle_error: float
    stop_prob: float = 0.0


@dataclass(frozen=True)
class VisualServoGains:
    pan: float = 0.04
    lift: float = 0.03
    flex: float = 0.03
    wrist_roll: float = 0.05
    sign_x: float = 1.0
    sign_y_lift: float = 1.0
    sign_y_flex: float = -1.0
    sign_angle: float = 1.0
    max_abs_delta: float = 0.08


@dataclass(frozen=True)
class VisualServoStopThresholds:
    stop_prob: float = 0.7
    dx: float = 0.08
    dy: float = 0.08
    angle: float = 0.12


def should_stop_visual_servo(error: VisualServoError, thresholds: VisualServoStopThresholds | None = None) -> bool:
    thresholds = thresholds or VisualServoStopThresholds()
    return (
        float(error.stop_prob) >= float(thresholds.stop_prob)
        and abs(float(error.wrist_dx_norm)) <= float(thresholds.dx)
        and abs(float(error.wrist_dy_norm)) <= float(thresholds.dy)
        and abs(float(error.edge_angle_error)) <= float(thresholds.angle)
    )


def visual_servo_delta_q(error: VisualServoError, gains: VisualServoGains | None = None) -> list[float]:
    gains = gains or VisualServoGains()
    delta = [0.0] * len(SO101_JOINT_ORDER)
    delta[0] = float(gains.pan) * float(gains.sign_x) * float(error.wrist_dx_norm)
    delta[1] = float(gains.lift) * float(gains.sign_y_lift) * float(error.wrist_dy_norm)
    delta[3] = float(gains.flex) * float(gains.sign_y_flex) * float(error.wrist_dy_norm)
    delta[4] = float(gains.wrist_roll) * float(gains.sign_angle) * float(error.edge_angle_error)
    limit = float(gains.max_abs_delta)
    return [max(-limit, min(limit, value)) for value in delta]


def apply_delta_q(qpos: Sequence[float], delta_q: Sequence[float], *, low: Sequence[float], high: Sequence[float]) -> list[float]:
    return [
        max(float(lo), min(float(hi), float(q) + float(delta)))
        for q, delta, lo, hi in zip(qpos, delta_q, low, high, strict=True)
    ]
