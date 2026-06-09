from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER

DEFAULT_DELTA_SCALE_RAW_TICKS = {
    "shoulder_pan": 1.0,
    "shoulder_lift": 1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_roll": 1.0,
    "gripper": 1.0,
}

DEFAULT_MAX_DELTA_RAW_TICKS = {
    "shoulder_pan": 2.0,
    "shoulder_lift": 2.0,
    "elbow_flex": 2.0,
    "wrist_flex": 2.0,
    "wrist_roll": 2.0,
    "gripper": 2.0,
}


@dataclass(frozen=True)
class SO100JointCommandPlan:
    joint: str
    current_raw: float
    raw_action_value: float
    scaled_delta_raw: float
    clipped_delta_raw: float
    target_raw: float
    range_min: float | None
    range_max: float | None
    clipped_by_delta_limit: bool
    clipped_by_calibrated_range: bool


@dataclass(frozen=True)
class SO100CommandPlan:
    status: str
    ready_for_execution: bool
    action_adapter: str
    action_dim: int
    expected_action_dim: int
    command_units: str
    human_confirmed: bool
    send_action_called: bool
    policy_actions_executed: bool
    joint_plans: list[SO100JointCommandPlan]
    blockers: list[str]
    notes: list[str]


@dataclass(frozen=True)
class SO100CommandChunkPlan:
    status: str
    ready_for_execution: bool
    action_adapter: str
    action_chunk_steps: int
    action_dim: int
    expected_action_dim: int
    command_units: str
    human_confirmed: bool
    send_action_called: bool
    policy_actions_executed: bool
    step_plans: list[SO100CommandPlan]
    blockers: list[str]
    notes: list[str]


def build_so100_command_plan(
    *,
    action: list[float],
    current_state: dict[str, Any],
    calibration: dict[str, Any] | None = None,
    human_confirmed: bool = False,
    adapter_semantics_confirmed: bool = False,
    delta_scale_raw_ticks: dict[str, float] | None = None,
    max_delta_raw_ticks: dict[str, float] | None = None,
) -> SO100CommandPlan:
    delta_scale_raw_ticks = delta_scale_raw_ticks or DEFAULT_DELTA_SCALE_RAW_TICKS
    max_delta_raw_ticks = max_delta_raw_ticks or DEFAULT_MAX_DELTA_RAW_TICKS
    blockers: list[str] = []
    joint_plans: list[SO100JointCommandPlan] = []

    if len(action) != len(SO100_JOINT_ORDER):
        blockers.append(f"Expected {len(SO100_JOINT_ORDER)} action values, got {len(action)}.")

    for index, joint in enumerate(SO100_JOINT_ORDER):
        value = float(action[index]) if index < len(action) else math.nan
        current = _float_or_nan(current_state.get(joint))
        scale = float(delta_scale_raw_ticks[joint])
        max_delta = abs(float(max_delta_raw_ticks[joint]))
        scaled = value * scale if math.isfinite(value) else math.nan
        clipped_delta = _clip(scaled, -max_delta, max_delta) if math.isfinite(scaled) else math.nan
        clipped_by_delta = math.isfinite(scaled) and clipped_delta != scaled
        target = current + clipped_delta if math.isfinite(current) and math.isfinite(clipped_delta) else math.nan

        joint_calibration = (calibration or {}).get(joint, {})
        range_min = _optional_float(joint_calibration.get("range_min"))
        range_max = _optional_float(joint_calibration.get("range_max"))
        clipped_by_range = False
        if range_min is not None and range_max is not None and math.isfinite(target):
            bounded = _clip(target, range_min, range_max)
            clipped_by_range = bounded != target
            target = bounded

        joint_plans.append(
            SO100JointCommandPlan(
                joint=joint,
                current_raw=current,
                raw_action_value=value,
                scaled_delta_raw=scaled,
                clipped_delta_raw=clipped_delta,
                target_raw=target,
                range_min=range_min,
                range_max=range_max,
                clipped_by_delta_limit=clipped_by_delta,
                clipped_by_calibrated_range=clipped_by_range,
            )
        )

    if not adapter_semantics_confirmed:
        blockers.append(
            "Adapter semantics are not confirmed: current mapping treats SmolVLA values as tiny raw-tick deltas."
        )
    if not human_confirmed:
        blockers.append("Human confirmation is required before any real SO-100 command plan can execute.")
    if any(not math.isfinite(plan.target_raw) for plan in joint_plans):
        blockers.append("Command plan contains non-finite target values.")

    ready = not blockers
    return SO100CommandPlan(
        status="passed" if len(action) == len(SO100_JOINT_ORDER) else "blocked",
        ready_for_execution=ready,
        action_adapter="smolvla_raw_to_conservative_raw_tick_delta_v0",
        action_dim=len(action),
        expected_action_dim=len(SO100_JOINT_ORDER),
        command_units="feetech_raw_ticks",
        human_confirmed=human_confirmed,
        send_action_called=False,
        policy_actions_executed=False,
        joint_plans=joint_plans,
        blockers=blockers,
        notes=[
            "This command plan is a no-actuation artifact.",
            "Do not execute this plan until adapter semantics are validated on a non-contact micro-step.",
        ],
    )


def build_so100_command_chunk_plan(
    *,
    action_chunk: list[list[float]],
    current_state: dict[str, Any],
    calibration: dict[str, Any] | None = None,
    human_confirmed: bool = False,
    adapter_semantics_confirmed: bool = False,
    delta_scale_raw_ticks: dict[str, float] | None = None,
    max_delta_raw_ticks: dict[str, float] | None = None,
) -> SO100CommandChunkPlan:
    if not action_chunk:
        return SO100CommandChunkPlan(
            status="blocked",
            ready_for_execution=False,
            action_adapter="smolvla_raw_to_conservative_raw_tick_delta_v0",
            action_chunk_steps=0,
            action_dim=0,
            expected_action_dim=len(SO100_JOINT_ORDER),
            command_units="feetech_raw_ticks",
            human_confirmed=human_confirmed,
            send_action_called=False,
            policy_actions_executed=False,
            step_plans=[],
            blockers=["No action chunk steps were provided."],
            notes=["This command chunk plan is a no-actuation artifact."],
        )

    simulated_state = dict(current_state)
    step_plans: list[SO100CommandPlan] = []
    blockers: list[str] = []
    for action in action_chunk:
        plan = build_so100_command_plan(
            action=action,
            current_state=simulated_state,
            calibration=calibration,
            human_confirmed=human_confirmed,
            adapter_semantics_confirmed=adapter_semantics_confirmed,
            delta_scale_raw_ticks=delta_scale_raw_ticks,
            max_delta_raw_ticks=max_delta_raw_ticks,
        )
        step_plans.append(plan)
        for blocker in plan.blockers:
            if blocker not in blockers:
                blockers.append(blocker)
        for joint_plan in plan.joint_plans:
            simulated_state[joint_plan.joint] = joint_plan.target_raw

    wrong_dim = [index for index, plan in enumerate(step_plans) if plan.action_dim != len(SO100_JOINT_ORDER)]
    if wrong_dim:
        blockers.append(f"Chunk contains wrong-dimension actions at step indexes: {wrong_dim}.")

    ready = bool(step_plans) and not blockers and all(plan.ready_for_execution for plan in step_plans)
    return SO100CommandChunkPlan(
        status="passed" if not wrong_dim else "blocked",
        ready_for_execution=ready,
        action_adapter="smolvla_raw_to_conservative_raw_tick_delta_v0",
        action_chunk_steps=len(action_chunk),
        action_dim=step_plans[0].action_dim if step_plans else 0,
        expected_action_dim=len(SO100_JOINT_ORDER),
        command_units="feetech_raw_ticks",
        human_confirmed=human_confirmed,
        send_action_called=False,
        policy_actions_executed=False,
        step_plans=step_plans,
        blockers=blockers,
        notes=[
            "This command chunk plan is a no-actuation artifact.",
            "Each SmolVLA chunk step is converted into a conservative sequential raw-tick delta target.",
            "Do not execute this chunk until adapter semantics are validated on a non-contact micro-step.",
        ],
    )


def write_command_plan(plan: SO100CommandPlan | SO100CommandChunkPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(plan), indent=2, sort_keys=True), encoding="utf-8")


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
