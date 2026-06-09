from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SO100_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_MAX_ABS_ACTION = {
    "shoulder_pan": 2.0,
    "shoulder_lift": 2.0,
    "elbow_flex": 2.0,
    "wrist_flex": 2.0,
    "wrist_roll": 2.0,
    "gripper": 2.0,
}


@dataclass(frozen=True)
class SO100JointSafetyCheck:
    joint: str
    current_raw: float
    action_value: float
    max_abs_action: float
    proposed_raw_if_delta: float
    range_min: float | None
    range_max: float | None
    passed: bool
    reasons: list[str]


@dataclass(frozen=True)
class SO100ActionSafetyReport:
    status: str
    candidate_safe: bool
    execution_allowed: bool
    action_dim: int
    expected_action_dim: int
    action_semantics: str
    human_confirmed: bool
    send_action_called: bool
    policy_actions_executed: bool
    checks: list[SO100JointSafetyCheck]
    blockers: list[str]
    notes: list[str]


def evaluate_so100_action_safety(
    *,
    action: list[float],
    current_state: dict[str, Any],
    calibration: dict[str, Any] | None = None,
    human_confirmed: bool = False,
    max_abs_action: dict[str, float] | None = None,
    require_known_action_semantics: bool = True,
) -> SO100ActionSafetyReport:
    max_abs_action = max_abs_action or DEFAULT_MAX_ABS_ACTION
    blockers: list[str] = []
    checks: list[SO100JointSafetyCheck] = []

    if len(action) != len(SO100_JOINT_ORDER):
        blockers.append(f"Expected {len(SO100_JOINT_ORDER)} action values, got {len(action)}.")

    for index, joint in enumerate(SO100_JOINT_ORDER):
        reasons: list[str] = []
        value = float(action[index]) if index < len(action) else math.nan
        current = _float_or_nan(current_state.get(joint))
        limit = float(max_abs_action[joint])
        joint_calibration = (calibration or {}).get(joint, {})
        range_min = _optional_float(joint_calibration.get("range_min"))
        range_max = _optional_float(joint_calibration.get("range_max"))
        proposed = current + value if math.isfinite(current) and math.isfinite(value) else math.nan

        if not math.isfinite(value):
            reasons.append("action_not_finite")
        if abs(value) > limit:
            reasons.append("action_delta_too_large")
        if not math.isfinite(current):
            reasons.append("missing_current_state")
        if range_min is not None and range_max is not None and math.isfinite(proposed):
            if proposed < range_min or proposed > range_max:
                reasons.append("proposed_raw_delta_outside_calibrated_range")

        checks.append(
            SO100JointSafetyCheck(
                joint=joint,
                current_raw=current,
                action_value=value,
                max_abs_action=limit,
                proposed_raw_if_delta=proposed,
                range_min=range_min,
                range_max=range_max,
                passed=not reasons,
                reasons=reasons,
            )
        )

    if require_known_action_semantics:
        blockers.append(
            "SmolVLA raw action semantics are not yet mapped to SO-100 command units; "
            "execution requires an explicit action adapter."
        )
    if not human_confirmed:
        blockers.append("Human confirmation is required before any real SO-100 action.")

    candidate_safe = not any(check.reasons for check in checks) and len(action) == len(SO100_JOINT_ORDER)
    execution_allowed = candidate_safe and not blockers
    status = "passed" if candidate_safe else "blocked"
    return SO100ActionSafetyReport(
        status=status,
        candidate_safe=candidate_safe,
        execution_allowed=execution_allowed,
        action_dim=len(action),
        expected_action_dim=len(SO100_JOINT_ORDER),
        action_semantics="unknown_smolvla_raw_action_candidate_delta",
        human_confirmed=human_confirmed,
        send_action_called=False,
        policy_actions_executed=False,
        checks=checks,
        blockers=blockers,
        notes=[
            "This gate is no-actuation only.",
            "proposed_raw_if_delta is diagnostic; it is not a command target.",
        ],
    )


def load_action_payload(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    action = payload.get("raw_action")
    if not isinstance(action, list):
        raise ValueError(f"raw_action list not found in {path}")
    return [float(value) for value in action]


def load_action_chunk_payload(path: Path, *, action_steps: int | None = None) -> list[list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunk = payload.get("raw_action_chunk")
    if chunk is None:
        return [load_action_payload(path)]
    if not isinstance(chunk, list):
        raise ValueError(f"raw_action_chunk list not found in {path}")
    if action_steps is not None and action_steps < 1:
        raise ValueError(f"action_steps must be positive, got {action_steps}")
    selected = chunk[:action_steps] if action_steps is not None else chunk
    actions: list[list[float]] = []
    for index, action in enumerate(selected):
        if not isinstance(action, list):
            raise ValueError(f"raw_action_chunk[{index}] is not a list in {path}")
        actions.append([float(value) for value in action])
    if not actions:
        raise ValueError(f"raw_action_chunk is empty in {path}")
    return actions


def load_episode_state(path: Path, frame_index: int) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["frame_index"]) == frame_index:
            state = record.get("observation", {}).get("state", {})
            if not isinstance(state, dict):
                raise ValueError(f"state dict not found for frame_index={frame_index} in {path}")
            return state
    raise ValueError(f"frame_index={frame_index} not found in {path}")


def load_calibration(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_safety_report(report: SO100ActionSafetyReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
