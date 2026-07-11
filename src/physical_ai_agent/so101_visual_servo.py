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
    target_area: int = 0


@dataclass(frozen=True)
class VisualServoGains:
    pan: float = 0.05
    lift: float = 0.04
    flex: float = 0.04
    wrist_roll: float = 0.06
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


LINEAR_CONTROLLER_WEIGHTS = {
    # ponytail: fitted from current SO101 teacher delta-q data; regenerate when camera/teacher contract changes.
    "camera1": (
        (0.029602, -0.023812, 0.024927, -0.008872, 0.008874, -0.017129),
        (-0.036962, 0.034991, -0.027431, 0.017487, -0.001784, 0.036099),
        (-0.001061, -0.000975, 0.003920, 0.001026, 0.002473, 0.002959),
        (-0.002481, 0.026135, -0.023080, 0.011261, 0.002068, 0.023747),
    ),
    "camera2": (
        (-0.002393, -0.014045, 0.008506, -0.006574, 0.002295, -0.018743),
        (0.016237, -0.024622, 0.023713, -0.010273, 0.001901, -0.019410),
        (0.000354, -0.000631, -0.000072, -0.000789, -0.002967, -0.001341),
        (-0.001289, 0.021761, -0.018974, 0.009394, 0.000960, 0.020146),
    ),
}

STATE_LINEAR_CONTROLLER_WEIGHTS = {
    # Feature order: dx, dy, angle, q0, q1, q2, q3, q4, q5, bias.
    "camera1": (
        (0.028326, -0.009540, 0.014103, -0.001565, 0.012690, -0.001909),
        (-0.032743, 0.065175, -0.051363, 0.031355, 0.004239, 0.066576),
        (-0.002409, 0.000072, 0.001152, 0.000421, -0.001119, 0.001772),
        (0.008648, -0.001026, 0.001154, -0.000369, -0.002028, -0.000369),
        (0.031538, -0.021770, 0.036499, -0.015217, 0.056056, -0.034533),
        (0.017722, -0.021774, 0.034730, -0.010000, 0.032136, -0.022408),
        (-0.008645, 0.010705, -0.011969, 0.015316, -0.015565, 0.012661),
        (0.001973, -0.001513, 0.000668, -0.000762, 0.014503, -0.003312),
        (-0.013134, -0.016047, 0.010338, -0.006657, -0.027850, -0.003323),
        (0.023019, 0.035063, -0.025288, 0.000887, 0.049469, 0.015018),
    ),
    "camera2": (
        (0.000875, -0.011794, 0.007612, -0.005355, 0.003641, -0.014917),
        (0.016567, -0.021363, 0.020252, -0.009117, 0.001070, -0.017082),
        (0.000655, -0.000541, 0.000145, -0.000443, -0.001372, -0.001049),
        (0.006152, 0.001929, -0.002174, 0.000708, -0.000225, 0.001128),
        (-0.000878, 0.014644, -0.004381, 0.003388, 0.000677, 0.009321),
        (-0.002049, 0.003772, 0.004744, 0.001796, 0.000869, 0.004200),
        (-0.002999, 0.010446, -0.008400, 0.012146, -0.002061, 0.010599),
        (-0.000511, 0.001196, -0.000776, 0.000593, 0.007362, 0.001376),
        (-0.002227, -0.035362, 0.030177, -0.015418, -0.003027, -0.026137),
        (0.007688, 0.053644, -0.048613, 0.013112, 0.007213, 0.039602),
    ),
}


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


def visual_servo_delta_q_for_camera(
    error: VisualServoError,
    camera: str,
    *,
    qpos: Sequence[float] | None = None,
    max_abs_delta: float = 0.08,
    scale: float = 1.0,
) -> list[float]:
    state_source = qpos if qpos is not None else []
    state = [float(value) for value in state_source[: len(SO101_JOINT_ORDER)]]
    weights = STATE_LINEAR_CONTROLLER_WEIGHTS.get(camera) if len(state) == len(SO101_JOINT_ORDER) else None
    features = (
        (float(error.wrist_dx_norm), float(error.wrist_dy_norm), float(error.edge_angle_error), *state, 1.0)
        if weights is not None
        else (float(error.wrist_dx_norm), float(error.wrist_dy_norm), float(error.edge_angle_error), 1.0)
    )
    weights = weights or LINEAR_CONTROLLER_WEIGHTS.get(camera)
    if weights is None:
        return visual_servo_delta_q(controller_error_for_camera(error, camera))
    delta = [
        float(scale) * sum(float(features[row]) * float(weights[row][joint]) for row in range(len(features)))
        for joint in range(len(SO101_JOINT_ORDER))
    ]
    if camera == "camera1":
        # ponytail: keep coarse egocentric pan monotonic in image x; teacher-fit bias can push past the target.
        delta[0] = 0.08 * float(error.wrist_dx_norm)
        if len(state) == len(SO101_JOINT_ORDER):
            # ponytail: camera1 is only a coarse approach view; keep it from driving past the wrist-camera handoff posture.
            target = [state[index] + delta[index] for index in range(len(SO101_JOINT_ORDER))]
            if state[0] < -1.0 and abs(float(error.wrist_dx_norm)) < 0.08:
                target[0] = max(-1.85, min(target[0], state[0] - 0.02))
            target[1] = max(-1.0, min(0.9, target[1]))
            target[2] = max(-0.85, min(1.0, target[2]))
            target[4] = max(-1.2, min(0.3, target[4]))
            if target[0] < -1.0 or state[0] < -1.0:
                target[4] = min(target[4], -0.4)
            delta = [target[index] - state[index] for index in range(len(SO101_JOINT_ORDER))]
    # ponytail: image-error servo steers pose only; gripper open/close belongs to grip primitives.
    delta[5] = 0.0
    limit = float(max_abs_delta)
    return [max(-limit, min(limit, value)) for value in delta]


def controller_error_for_camera(error: VisualServoError, camera: str) -> VisualServoError:
    if camera in {"camera1", "camera2"}:
        # ponytail: rendered policy cameras have image-y down; lift/flex correction uses robot-y up.
        return VisualServoError(
            wrist_dx_norm=float(error.wrist_dx_norm),
            wrist_dy_norm=-float(error.wrist_dy_norm),
            edge_angle_error=0.0 if camera == "camera1" else float(error.edge_angle_error),
            stop_prob=float(error.stop_prob),
        )
    return error


def select_visual_servo_camera(
    *,
    camera1_error: Sequence[float],
    camera2_error: Sequence[float],
    camera1_visible_prob: float,
    camera2_visible_prob: float,
    visible_threshold: float = 0.5,
    wrist_handoff_prob: float = 0.7,
    camera1_near_threshold: float = 0.3,
) -> dict[str, object]:
    """Pick exactly one camera for this visual-servo command.

    camera1 is the egocentric approach view. camera2 is the wrist refinement
    view, so it should not take over until camera1 says the target is near.
    """
    c1_prob = float(camera1_visible_prob)
    c2_prob = float(camera2_visible_prob)
    c1_near = max(abs(float(camera1_error[0])), abs(float(camera1_error[1]))) <= float(camera1_near_threshold)
    if c2_prob >= float(wrist_handoff_prob) and (c1_near or c1_prob < float(visible_threshold)):
        camera = "camera2"
        reason = "wrist_handoff" if c1_near else "egocentric_not_visible"
    elif c1_prob >= float(visible_threshold):
        camera = "camera1"
        reason = "egocentric_approach"
    elif c2_prob >= float(visible_threshold):
        camera = "camera2"
        reason = "wrist_fallback"
    else:
        camera = "camera1" if c1_prob >= c2_prob else "camera2"
        reason = "max_visible_prob"
    return {
        "servo_camera": camera,
        "reason": reason,
        "camera1_near": bool(c1_near),
        "camera1_visible_prob": c1_prob,
        "camera2_visible_prob": c2_prob,
        "visible_threshold": float(visible_threshold),
        "wrist_handoff_prob": float(wrist_handoff_prob),
        "camera1_near_threshold": float(camera1_near_threshold),
    }


def apply_delta_q(qpos: Sequence[float], delta_q: Sequence[float], *, low: Sequence[float], high: Sequence[float]) -> list[float]:
    return [
        max(float(lo), min(float(hi), float(q) + float(delta)))
        for q, delta, lo, hi in zip(qpos, delta_q, low, high, strict=True)
    ]
