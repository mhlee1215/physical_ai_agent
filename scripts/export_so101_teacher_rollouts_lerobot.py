#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.so101_resolution_contract import require_so101_image_resolution
from physical_ai_agent.so101_workspace_spawn_catalog import (
    WorkspaceCellQuotaScheduler,
    WorkspaceSpawnCandidate,
    load_workspace_spawn_catalog,
)
from physical_ai_agent.sim.so101_camera_input import EGOCENTRIC_CAMERA1_POSE, _make_camera, postprocess_camera_frame
from physical_ai_agent.sim.so101_live_viewer import _cartesian_error_controller_action
from train_so101_wrist_ego_picklift_policy import sweep_until_visible
from train_so101_wrist_ego_visual_servo import (
    WristEgoServoConfig,
    _current_qpos,
    _grasp_candidate_specs,
    _make_policy_renderers,
    _make_teacher_renderers,
    _restore_sim_state,
    _set_qpos,
    _snapshot_sim_state,
    make_high_contrast_picklift_env,
    make_teacher_targets,
    object_visible_to_teacher,
)

try:
    from filter_so101_lerobot_visual_alignment import _angle_diff, _image_alignment_score, _jaw_axis_angle
except ModuleNotFoundError:  # pragma: no cover
    from scripts.filter_so101_lerobot_visual_alignment import _angle_diff, _image_alignment_score, _jaw_axis_angle


def _clear_episode_buffer_robust(dataset: Any, *, attempts: int = 5) -> None:
    last_error: Exception | None = None
    for index in range(max(1, int(attempts))):
        try:
            dataset.clear_episode_buffer()
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05 * float(index + 1))
    if last_error is not None:
        raise last_error


def _json_safe_sim_snapshot(env: Any) -> dict[str, list[float]]:
    snapshot = _snapshot_sim_state(env)
    return {key: [float(value) for value in values] for key, values in snapshot.items()}


TASK = "Grasp the visible cube and lift it up."
SKILL_TASKS = {
    "pick_cube": TASK,
    "move_over_cube": "Move the gripper over the visible cube.",
    "pick_from_top_cube": "From above the visible cube, grasp it and lift it up.",
    "move_over_cube_edge": "Move the gripper above one visible cube edge.",
    "align_fixed_jaw_cube_edge": "Align the gripper jaws around one visible cube edge.",
    "move_and_align_cube_edge": "Move the gripper above one visible cube edge and align the jaws around it.",
    "grip_from_edge_cube": "Close the gripper on the cube edge and lift.",
    "grip_from_above_edge_cube": "Move down from above the cube edge, close the gripper, and lift.",
    "grip_the_cube_v1": "Grip the cube and lift.",
    "grip_the_cube_near_v1": "Grip the nearby cube and lift.",
    "grip_the_cube_continuous_v1": "Grip the cube and lift.",
}
COLOR_SHAPE_SKILL_TASK_TEMPLATES = {
    "pick_cube": "grip the {color} {shape} and lift",
    "move_over_cube": "Move the gripper over the visible {color} {shape}.",
    "pick_from_top_cube": "grip the {color} {shape} and lift",
    "move_over_cube_edge": "Move the gripper above one visible {color} {shape} edge.",
    "align_fixed_jaw_cube_edge": "Align the gripper jaws around one visible {color} {shape} edge.",
    "move_and_align_cube_edge": "Move above one visible {color} {shape} edge and align the gripper jaws around it.",
    "grip_from_edge_cube": "grip the {color} {shape} and lift",
    "grip_from_above_edge_cube": "grip the {color} {shape} and lift",
    "grip_the_cube_v1": "grip the {color} {shape} and lift",
    "grip_the_cube_near_v1": "grip the {color} {shape} and lift",
    "grip_the_cube_continuous_v1": "grip the {color} {shape} and lift",
}

GRIP_THE_CUBE_V1_CAMERA2_TOP_CONTACT_LIMITS = {
    "pre_close_image_alignment_error_deg": 12.0,
    "close_25_image_alignment_error_deg": 12.0,
    "close_50_image_alignment_error_deg": 12.0,
    "close_75_image_alignment_error_deg": 25.0,
}
GRIP_THE_CUBE_V1_MAX_WRIST_ROLL_STEP_RAD = 0.12
GRIP_THE_CUBE_V1_CLOSE_TRACE_FRACTIONS = (0.25, 0.50, 0.75)
# Contact can rotate the wrist between the coarse checkpoints. v1_5 keeps the
# correction closed-loop at every close control step; the local roll sweep is
# still bounded around the realized previous roll.
# The jaw mask becomes unreliable once contact occludes the top face. v1_5
# therefore performs one camera2 alignment before closing and holds that roll
# throughout contact instead of chasing a changing image edge.
GRIP_THE_CUBE_V1_REFINE_EVERY_CLOSE_STEP = False
FIXED_JAW_SKILL_MODES = {
    "move_over_cube_edge",
    "align_fixed_jaw_cube_edge",
    "move_and_align_cube_edge",
    "grip_from_edge_cube",
    "grip_from_above_edge_cube",
    "grip_the_cube_v1",
    "grip_the_cube_near_v1",
    "grip_the_cube_continuous_v1",
}
FULL_GRIP_SKILL_MODES = {
    "grip_the_cube_v1",
    "grip_the_cube_near_v1",
    "grip_the_cube_continuous_v1",
}

# Both solvers are attempted through the bridge band. Outside it, avoid the
# expensive second IK family while preserving the verified near/mid paths.
CONTINUOUS_TEACHER_NEAR_OVERLAP_MAX_M = 0.22
CONTINUOUS_TEACHER_MID_OVERLAP_MIN_M = 0.18
STATE_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _wrist_roll_safe_cosine_steps(
    start_roll: float,
    target_roll: float,
    *,
    requested_steps: int,
    max_step_rad: float = GRIP_THE_CUBE_V1_MAX_WRIST_ROLL_STEP_RAD,
) -> int:
    steps = max(1, int(requested_steps))
    delta = abs(float(target_roll) - float(start_roll))
    if delta <= 0.0:
        return steps
    limit = float(max_step_rad) * 0.98
    while True:
        alpha = 0.5 - 0.5 * np.cos(
            np.pi * np.arange(steps + 1, dtype=float) / float(steps)
        )
        if delta * float(np.max(np.diff(alpha))) <= limit:
            return steps
        steps += 1


def _make_grip_the_cube_wrist_safe_phases(
    phases: list[tuple[str, np.ndarray | None, np.ndarray | None, int]],
) -> tuple[
    list[tuple[str, np.ndarray | None, np.ndarray | None, int]],
    dict[str, dict[str, int]],
]:
    safe_phases: list[
        tuple[str, np.ndarray | None, np.ndarray | None, int]
    ] = []
    changed: dict[str, dict[str, int]] = {}
    for phase, start, target, requested_steps in phases:
        safe_steps = int(requested_steps)
        if (
            start is not None
            and target is not None
            and len(start) > 4
            and len(target) > 4
        ):
            safe_steps = _wrist_roll_safe_cosine_steps(
                float(start[4]),
                float(target[4]),
                requested_steps=safe_steps,
            )
        if safe_steps != int(requested_steps):
            changed[phase] = {
                "requested_steps": int(requested_steps),
                "safe_steps": safe_steps,
            }
        safe_phases.append((phase, start, target, safe_steps))
    return safe_phases, changed


def _close_alignment_limits(
    *,
    mode: str,
    pre_close: float | None,
    close_25: float | None,
    close_50: float | None,
    close_75: float | None,
) -> dict[str, float] | None:
    values = {
        "pre_close_image_alignment_error_deg": pre_close,
        "close_25_image_alignment_error_deg": close_25,
        "close_50_image_alignment_error_deg": close_50,
        "close_75_image_alignment_error_deg": close_75,
    }
    supplied = {key: float(value) for key, value in values.items() if value is not None}
    if not supplied:
        return None
    if mode == "geometry_only":
        raise ValueError("geometry_only does not accept camera2 image alignment limits")
    required = {
        "pre_close_image_alignment_error_deg",
        "close_25_image_alignment_error_deg",
        "close_50_image_alignment_error_deg",
    }
    if mode == "strict_image_trace":
        required.add("close_75_image_alignment_error_deg")
    missing = sorted(required - supplied.keys())
    if missing:
        raise ValueError(f"missing camera2 close alignment limits: {missing}")
    if mode == "preclose_and_early_trace" and "close_75_image_alignment_error_deg" in supplied:
        raise ValueError("preclose_and_early_trace must not set a close-75 limit")
    if any(value <= 0.0 or value > 90.0 for value in supplied.values()):
        raise ValueError("camera2 close alignment limits must be in (0, 90]")
    return supplied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export privileged SO101 teacher PickLift rollouts to a local LeRobotDataset."
    )
    parser.add_argument("--root", type=Path, default=Path("_workspace/so101_lerobot/teacher_picklift_smolvla"))
    parser.add_argument("--repo-id", default="physical-ai-agent/so101-picklift-teacher")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--data-files-size-in-mb", type=int, default=10000)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--teacher-style", choices=["legacy", "staged"], default="staged")
    parser.add_argument("--approach-steps", type=int, default=34)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--close-steps", type=int, default=42)
    parser.add_argument(
        "--trajectory-variant",
        choices=["standard", "roll_first", "direct_align", "auto"],
        default="standard",
        help=(
            "Deterministic intermediate path for fixed-jaw grip trajectories. "
            "auto is reserved for grip_the_cube_continuous_v1 and selects the "
            "verified near/mid path from the accepted solver profile."
        ),
    )
    parser.add_argument(
        "--grip-the-cube-start-profile",
        choices=["mixed", "home", "mid", "correction"],
        default="mixed",
        help="Start profile for grip_the_cube_v1; home is the stable constructive-generation mode.",
    )
    parser.add_argument(
        "--close-alignment-gate-mode",
        choices=["strict_image_trace", "preclose_and_early_trace", "geometry_only"],
        default="strict_image_trace",
        help="Gate close alignment by camera2, early camera2 trace, or authoritative simulator geometry.",
    )
    parser.add_argument("--pre-close-image-alignment-max-deg", type=float)
    parser.add_argument("--close-25-image-alignment-max-deg", type=float)
    parser.add_argument("--close-50-image-alignment-max-deg", type=float)
    parser.add_argument("--close-75-image-alignment-max-deg", type=float)
    parser.add_argument("--lift-steps", type=int, default=58)
    parser.add_argument(
        "--lift-target-height",
        type=float,
        default=0.05,
        help="Stop the lift phase after the grasped object reaches this height in meters.",
    )
    parser.add_argument(
        "--lift-success-height",
        type=float,
        help=(
            "Minimum grasped-object height that must remain after terminal hold. "
            "Defaults to --lift-target-height for backward compatibility."
        ),
    )
    parser.add_argument(
        "--lift-controller-z-error",
        type=float,
        default=0.12,
        help="Per-step Cartesian z error passed to the lift controller in meters.",
    )
    parser.add_argument("--start-mode", choices=["home", "near-gripper"], default="home")
    parser.add_argument("--near-gripper-joint-std", type=float, default=0.025)
    parser.add_argument(
        "--skill-mode",
        choices=sorted(SKILL_TASKS),
        default="pick_cube",
        help="Export the full pick skill or one of its agentic primitive segments.",
    )
    parser.add_argument(
        "--random-start-joint-std",
        type=float,
        default=0.55,
        help="Joint-space std used for move_over_cube random starts.",
    )
    parser.add_argument(
        "--move-success-tcp-dist",
        type=float,
        default=0.085,
        help="Max TCP-to-object distance accepted for move_over_cube success.",
    )
    parser.add_argument(
        "--move-target-z-offset",
        type=float,
        default=0.075,
        help="Approximate Cartesian z offset from grasp prepose for move_over_cube/pick_from_top starts.",
    )
    parser.add_argument(
        "--closed-gripper-prob",
        type=float,
        default=0.45,
        help="Probability that move_over_cube is exported with the gripper closed.",
    )
    parser.add_argument(
        "--move-gripper-profile",
        choices=["binary", "balanced", "closed"],
        default="balanced",
        help="Gripper qpos sampling for move_over_cube. balanced cycles through closed/intermediate/open gaps.",
    )
    parser.add_argument(
        "--move-min-actual-z",
        type=float,
        default=0.0,
        help="Reject move_over_cube episodes whose final TCP/cube world-z offset is smaller than this.",
    )
    parser.add_argument(
        "--terminal-hold-steps",
        type=int,
        default=0,
        help="Append this many final hold frames for fixed-jaw primitive datasets.",
    )
    parser.add_argument(
        "--move-and-align-near-target-correction-ratio",
        type=float,
        default=0.0,
        help="For move_and_align_cube_edge, export this fraction as near-target correction trajectories.",
    )
    parser.add_argument("--edge-contact-xy-success-threshold", type=float, default=0.012)
    parser.add_argument("--edge-contact-parallel-success-threshold-deg", type=float, default=8.0)
    parser.add_argument(
        "--near-target-joint-std",
        type=float,
        default=0.075,
        help="Joint perturbation std for generated near-target correction trajectories.",
    )
    parser.add_argument(
        "--near-target-xy-std",
        type=float,
        default=0.025,
        help="Cartesian XY perturbation std for generated near-target correction trajectories.",
    )
    parser.add_argument(
        "--pick-start-joint-std",
        type=float,
        default=0.035,
        help="Joint jitter around the elevated top pose for pick_from_top_cube starts.",
    )
    parser.add_argument(
        "--pick-correction-steps",
        type=int,
        default=18,
        help="Approach/correction steps from elevated top pose to grasp prepose before closing.",
    )
    parser.add_argument(
        "--pick-start-min-abs-y",
        type=float,
        default=0.018,
        help="Minimum absolute world-y offset between pick_from_top start TCP and cube.",
    )
    parser.add_argument(
        "--pick-start-max-abs-y",
        type=float,
        default=0.055,
        help="Maximum absolute world-y offset target between pick_from_top start TCP and cube.",
    )
    parser.add_argument(
        "--pick-start-min-actual-abs-y",
        type=float,
        default=0.015,
        help="Reject pick_from_top episodes whose actual start TCP/cube world-y offset is smaller than this.",
    )
    parser.add_argument(
        "--pick-start-min-actual-z",
        type=float,
        default=0.0,
        help="Reject pick_from_top episodes whose actual start TCP/cube world-z offset is smaller than this.",
    )
    parser.add_argument(
        "--above-edge-start-joint-std",
        type=float,
        default=0.0,
        help="Joint jitter around q_above for grip_from_above_edge_cube starts.",
    )
    parser.add_argument(
        "--above-edge-start-xy-std",
        type=float,
        default=0.0,
        help="Cartesian XY jitter around q_above for grip_from_above_edge_cube starts.",
    )
    parser.add_argument(
        "--above-edge-start-z-std",
        type=float,
        default=0.0,
        help="Cartesian Z jitter around q_above for grip_from_above_edge_cube starts.",
    )
    parser.add_argument(
        "--above-edge-start-min-actual-z",
        type=float,
        default=0.0,
        help="Reject grip_from_above_edge_cube episodes whose actual start TCP/cube world-z offset is smaller than this.",
    )
    parser.add_argument(
        "--above-edge-trajectory-variants",
        default="standard",
        help=(
            "Comma-separated grip_from_above_edge_cube trajectory variants. "
            "Supported: standard, two_stage_xy_z, roll_first, near_miss_correction."
        ),
    )
    parser.add_argument(
        "--above-edge-start-gripper-profile",
        choices=["open", "balanced"],
        default="open",
        help="Start/open-phase gripper profile for grip_from_above_edge_cube.",
    )
    parser.add_argument(
        "--above-edge-terminal-hold-jitter",
        type=int,
        default=0,
        help="If >0, cycle terminal hold length by +/- this many frames for grip_from_above_edge_cube.",
    )
    parser.add_argument(
        "--max-attempt-multiplier",
        type=int,
        default=8,
        help="Maximum candidate seeds to try, as episodes * multiplier.",
    )
    parser.add_argument(
        "--grid-balance-size",
        type=int,
        default=0,
        help="If >0, only save episodes whose start camera centroid falls in a requested grid bin.",
    )
    parser.add_argument(
        "--grid-balance-target-per-bin",
        type=int,
        default=0,
        help="Required saved episodes per requested grid bin. Requires --grid-balance-bins.",
    )
    parser.add_argument(
        "--grid-balance-bins",
        default="",
        help="Comma-separated camera1 grid bins to balance, for example '5,6,7,9,10,11'.",
    )
    parser.add_argument(
        "--grid-balance-spawn-lookup",
        action="store_true",
        help="Precompute world-XY -> camera1 grid-bin candidates and sample those instead of seed rejection.",
    )
    parser.add_argument(
        "--grid-balance-teacher-feasible-lookup",
        action="store_true",
        help="Filter spawn lookup candidates to coordinates that pass the fixed-jaw teacher policy-view filter.",
    )
    parser.add_argument("--grid-lookup-max-candidates-per-bin", type=int, default=0)
    parser.add_argument("--grid-lookup-x-min", type=float, default=-0.10)
    parser.add_argument("--grid-lookup-x-max", type=float, default=0.55)
    parser.add_argument("--grid-lookup-y-min", type=float, default=-0.45)
    parser.add_argument("--grid-lookup-y-max", type=float, default=0.45)
    parser.add_argument("--grid-lookup-resolution", type=int, default=21)
    parser.add_argument(
        "--grid-lookup-start-index",
        type=int,
        default=0,
        help=(
            "Start each requested bin at this ordered lookup candidate index. "
            "Use a nonzero value to create a spawn-disjoint validation split."
        ),
    )
    parser.add_argument(
        "--grid-lookup-cache",
        type=Path,
        help="JSON cache for the deterministic camera1 world-XY -> grid-bin lookup.",
    )
    parser.add_argument(
        "--grid-lookup-preserve-order",
        action="store_true",
        help="Keep cached spawn candidates in manifest order instead of sorting by center distance.",
    )
    parser.add_argument(
        "--deterministic-camera-bin-lookup",
        action="store_true",
        help=(
            "Use only the ordered camera-bin lookup candidates and deterministic fixed-jaw IK; "
            "do not fall back to random seed rejection."
        ),
    )
    parser.add_argument(
        "--workspace-spawn-catalog",
        type=Path,
        help=(
            "Typed seed-free catalog of workspace XY, candidate-specific cube yaw, "
            "and sampling metadata."
        ),
    )
    parser.add_argument("--workspace-spawn-start-index", type=int, default=0)
    parser.add_argument("--workspace-spawn-candidate-count", type=int, default=0)
    parser.add_argument(
        "--workspace-spawn-forbidden-report",
        type=Path,
        action="append",
        default=[],
        help=(
            "Completed shard export report whose accepted workspace positions must "
            "remain at least the catalog minimum spacing away. May be repeated."
        ),
    )
    parser.add_argument(
        "--target-object-color",
        choices=["red", "orange", "yellow", "green", "blue", "purple", "black", "white"],
        help="Only export episodes whose target object has this color.",
    )
    parser.add_argument("--spawn-center-x", type=float, default=0.15)
    parser.add_argument("--spawn-center-y", type=float, default=0.0)
    parser.add_argument("--spawn-min-radius", type=float, default=0.10)
    parser.add_argument("--spawn-max-radius", type=float, default=0.30)
    parser.add_argument("--spawn-angle-half-range-deg", type=float, default=90.0)
    parser.add_argument(
        "--target-object-yaw-deg",
        type=float,
        help="Override the settled target cube yaw after reset and before frame 0.",
    )
    parser.add_argument("--object-half-sizes", default="0.0125,0.015,0.0175")
    parser.add_argument(
        "--initial-qpos",
        help="Exact 6D MuJoCo start qpos, comma-separated. The first recorded frame preserves it.",
    )
    parser.add_argument(
        "--initial-qpos-mode",
        choices=["exact", "reset_only"],
        default="exact",
        help=(
            "Use initial qpos as the exact first frame, or only as the reset/IK reference "
            "while allowing a configured correction start profile."
        ),
    )
    parser.add_argument(
        "--camera-rig-config",
        type=Path,
        help="Config-defined physical camera rig used for both camera geometry and policy images.",
    )
    parser.add_argument(
        "--min-gripper-floor-clearance-m",
        type=float,
        default=0.0,
        help="Reject a teacher trajectory if any gripper collision geom is closer to the floor.",
    )
    parser.add_argument(
        "--require-initial-target-visible",
        action="store_true",
        help=(
            "Reject the placement before sweeping unless the target is visible in at "
            "least one selected policy camera at the exact first-frame pose."
        ),
    )
    parser.add_argument(
        "--initial-target-visibility-cameras",
        default="camera1,camera2",
        help="Comma-separated policy cameras used by the initial visibility gate.",
    )
    parser.add_argument(
        "--initial-target-min-area-pixels",
        type=int,
        default=20,
        help="Minimum segmented target area in a selected first-frame camera.",
    )
    parser.add_argument("--no-camera3-duplicate", action="store_true")
    parser.add_argument(
        "--capture-render-replay",
        action="store_true",
        help="Capture exact pre-action MuJoCo/geom/camera state for renderer-independent replay.",
    )
    args = parser.parse_args()

    report = export_teacher_rollouts(
        root=args.root,
        repo_id=args.repo_id,
        episodes=args.episodes,
        seed=args.seed,
        fps=args.fps,
        width=args.width,
        height=args.height,
        data_files_size_in_mb=args.data_files_size_in_mb,
        use_videos=args.use_videos,
        overwrite=args.overwrite,
        teacher_style=args.teacher_style,
        approach_steps=args.approach_steps,
        settle_steps=args.settle_steps,
        close_steps=args.close_steps,
        trajectory_variant=args.trajectory_variant,
        grip_the_cube_start_profile=args.grip_the_cube_start_profile,
        close_alignment_gate_mode=args.close_alignment_gate_mode,
        close_alignment_limits=_close_alignment_limits(
            mode=args.close_alignment_gate_mode,
            pre_close=args.pre_close_image_alignment_max_deg,
            close_25=args.close_25_image_alignment_max_deg,
            close_50=args.close_50_image_alignment_max_deg,
            close_75=args.close_75_image_alignment_max_deg,
        ),
        lift_steps=args.lift_steps,
        lift_target_height=args.lift_target_height,
        lift_success_height=args.lift_success_height,
        lift_controller_z_error=args.lift_controller_z_error,
        start_mode=args.start_mode,
        near_gripper_joint_std=args.near_gripper_joint_std,
        skill_mode=args.skill_mode,
        random_start_joint_std=args.random_start_joint_std,
        move_success_tcp_dist=args.move_success_tcp_dist,
        move_target_z_offset=args.move_target_z_offset,
        closed_gripper_prob=args.closed_gripper_prob,
        move_gripper_profile=args.move_gripper_profile,
        move_min_actual_z=args.move_min_actual_z,
        terminal_hold_steps=args.terminal_hold_steps,
        move_and_align_near_target_correction_ratio=args.move_and_align_near_target_correction_ratio,
        edge_contact_xy_success_threshold=args.edge_contact_xy_success_threshold,
        edge_contact_parallel_success_threshold_deg=args.edge_contact_parallel_success_threshold_deg,
        near_target_joint_std=args.near_target_joint_std,
        near_target_xy_std=args.near_target_xy_std,
        pick_start_joint_std=args.pick_start_joint_std,
        pick_correction_steps=args.pick_correction_steps,
        pick_start_min_abs_y=args.pick_start_min_abs_y,
        pick_start_max_abs_y=args.pick_start_max_abs_y,
        pick_start_min_actual_abs_y=args.pick_start_min_actual_abs_y,
        pick_start_min_actual_z=args.pick_start_min_actual_z,
        above_edge_start_joint_std=args.above_edge_start_joint_std,
        above_edge_start_xy_std=args.above_edge_start_xy_std,
        above_edge_start_z_std=args.above_edge_start_z_std,
        above_edge_start_min_actual_z=args.above_edge_start_min_actual_z,
        above_edge_trajectory_variants=args.above_edge_trajectory_variants,
        above_edge_start_gripper_profile=args.above_edge_start_gripper_profile,
        above_edge_terminal_hold_jitter=args.above_edge_terminal_hold_jitter,
        max_attempt_multiplier=args.max_attempt_multiplier,
        grid_balance_size=args.grid_balance_size,
        grid_balance_target_per_bin=args.grid_balance_target_per_bin,
        grid_balance_bins=args.grid_balance_bins,
        grid_balance_spawn_lookup=args.grid_balance_spawn_lookup,
        grid_balance_teacher_feasible_lookup=args.grid_balance_teacher_feasible_lookup,
        grid_lookup_max_candidates_per_bin=args.grid_lookup_max_candidates_per_bin,
        grid_lookup_x_min=args.grid_lookup_x_min,
        grid_lookup_x_max=args.grid_lookup_x_max,
        grid_lookup_y_min=args.grid_lookup_y_min,
        grid_lookup_y_max=args.grid_lookup_y_max,
        grid_lookup_resolution=args.grid_lookup_resolution,
        grid_lookup_start_index=args.grid_lookup_start_index,
        grid_lookup_cache=args.grid_lookup_cache,
        grid_lookup_preserve_order=args.grid_lookup_preserve_order,
        deterministic_camera_bin_lookup=args.deterministic_camera_bin_lookup,
        workspace_spawn_catalog_path=args.workspace_spawn_catalog,
        workspace_spawn_start_index=args.workspace_spawn_start_index,
        workspace_spawn_candidate_count=args.workspace_spawn_candidate_count,
        workspace_spawn_forbidden_reports=args.workspace_spawn_forbidden_report,
        target_object_color=args.target_object_color,
        spawn_center=(args.spawn_center_x, args.spawn_center_y),
        spawn_min_radius=args.spawn_min_radius,
        spawn_max_radius=args.spawn_max_radius,
        spawn_angle_half_range_deg=args.spawn_angle_half_range_deg,
        target_object_yaw_deg=args.target_object_yaw_deg,
        object_half_sizes=_parse_float_list(args.object_half_sizes),
        initial_qpos=(None if args.initial_qpos is None else _parse_float_list(args.initial_qpos)),
        initial_qpos_mode=args.initial_qpos_mode,
        camera_rig_config_path=args.camera_rig_config,
        min_gripper_floor_clearance_m=args.min_gripper_floor_clearance_m,
        require_initial_target_visible=args.require_initial_target_visible,
        initial_target_visibility_cameras=tuple(
            value.strip()
            for value in args.initial_target_visibility_cameras.split(",")
            if value.strip()
        ),
        initial_target_min_area_pixels=args.initial_target_min_area_pixels,
        include_camera3_duplicate=not args.no_camera3_duplicate,
        capture_render_replay=args.capture_render_replay,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if int(report.get("exported_episodes", 0)) != int(report.get("requested_episodes", 0)):
        raise SystemExit(2)


def export_teacher_rollouts(
    *,
    root: Path,
    repo_id: str,
    episodes: int,
    seed: int,
    fps: int,
    width: int,
    height: int,
    data_files_size_in_mb: int,
    use_videos: bool,
    overwrite: bool,
    teacher_style: str = "staged",
    approach_steps: int = 34,
    settle_steps: int = 10,
    close_steps: int = 42,
    trajectory_variant: str = "standard",
    grip_the_cube_start_profile: str = "mixed",
    close_alignment_gate_mode: str = "strict_image_trace",
    close_alignment_limits: dict[str, float] | None = None,
    lift_steps: int = 58,
    lift_target_height: float = 0.05,
    lift_success_height: float | None = None,
    lift_controller_z_error: float = 0.12,
    start_mode: str = "home",
    near_gripper_joint_std: float = 0.025,
    skill_mode: str = "pick_cube",
    random_start_joint_std: float = 0.55,
    move_success_tcp_dist: float = 0.085,
    move_target_z_offset: float = 0.075,
    closed_gripper_prob: float = 0.45,
    move_gripper_profile: str = "balanced",
    move_min_actual_z: float = 0.0,
    terminal_hold_steps: int = 0,
    move_and_align_near_target_correction_ratio: float = 0.0,
    edge_contact_xy_success_threshold: float = 0.012,
    edge_contact_parallel_success_threshold_deg: float = 8.0,
    near_target_joint_std: float = 0.075,
    near_target_xy_std: float = 0.025,
    pick_start_joint_std: float = 0.035,
    pick_correction_steps: int = 18,
    pick_start_min_abs_y: float = 0.018,
    pick_start_max_abs_y: float = 0.055,
    pick_start_min_actual_abs_y: float = 0.015,
    pick_start_min_actual_z: float = 0.0,
    above_edge_start_joint_std: float = 0.0,
    above_edge_start_xy_std: float = 0.0,
    above_edge_start_z_std: float = 0.0,
    above_edge_start_min_actual_z: float = 0.0,
    above_edge_trajectory_variants: str = "standard",
    above_edge_start_gripper_profile: str = "open",
    above_edge_terminal_hold_jitter: int = 0,
    max_attempt_multiplier: int = 8,
    grid_balance_size: int = 0,
    grid_balance_target_per_bin: int = 0,
    grid_balance_bins: str = "",
    grid_balance_spawn_lookup: bool = False,
    grid_balance_teacher_feasible_lookup: bool = False,
    grid_lookup_max_candidates_per_bin: int = 0,
    grid_lookup_x_min: float = -0.10,
    grid_lookup_x_max: float = 0.55,
    grid_lookup_y_min: float = -0.45,
    grid_lookup_y_max: float = 0.45,
    grid_lookup_resolution: int = 21,
    grid_lookup_start_index: int = 0,
    grid_lookup_cache: Path | None = None,
    grid_lookup_preserve_order: bool = False,
    deterministic_camera_bin_lookup: bool = False,
    workspace_spawn_catalog_path: Path | None = None,
    workspace_spawn_start_index: int = 0,
    workspace_spawn_candidate_count: int = 0,
    workspace_spawn_forbidden_reports: list[Path] | None = None,
    target_object_color: str | None = None,
    spawn_center: tuple[float, float] = (0.15, 0.0),
    spawn_min_radius: float = 0.10,
    spawn_max_radius: float = 0.30,
    spawn_angle_half_range_deg: float = 90.0,
    target_object_yaw_deg: float | None = None,
    object_half_sizes: tuple[float, ...] = (0.0125, 0.015, 0.0175),
    initial_qpos: tuple[float, ...] | None = None,
    initial_qpos_mode: str = "exact",
    camera_rig_config_path: Path | None = None,
    min_gripper_floor_clearance_m: float = 0.0,
    require_initial_target_visible: bool = False,
    initial_target_visibility_cameras: tuple[str, ...] = ("camera1", "camera2"),
    initial_target_min_area_pixels: int = 20,
    include_camera3_duplicate: bool = True,
    capture_render_replay: bool = False,
) -> dict[str, Any]:
    import shutil

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    export_started = time.perf_counter()
    require_so101_image_resolution(height=height, width=width, context="SO101 LeRobot teacher export")
    if initial_qpos is not None and len(initial_qpos) != 6:
        raise ValueError(f"--initial-qpos requires 6 values, got {len(initial_qpos)}")
    if initial_qpos_mode not in {"exact", "reset_only"}:
        raise ValueError("--initial-qpos-mode must be exact or reset_only")
    if float(min_gripper_floor_clearance_m) < 0.0:
        raise ValueError("--min-gripper-floor-clearance-m must be >= 0")
    allowed_visibility_cameras = {"camera1", "camera2"}
    requested_visibility_cameras = tuple(initial_target_visibility_cameras)
    if not requested_visibility_cameras:
        raise ValueError("--initial-target-visibility-cameras must not be empty")
    if len(requested_visibility_cameras) != len(set(requested_visibility_cameras)):
        raise ValueError("--initial-target-visibility-cameras must not contain duplicates")
    unknown_visibility_cameras = set(requested_visibility_cameras) - allowed_visibility_cameras
    if unknown_visibility_cameras:
        raise ValueError(
            "--initial-target-visibility-cameras contains unsupported values: "
            f"{sorted(unknown_visibility_cameras)}"
        )
    if int(initial_target_min_area_pixels) <= 0:
        raise ValueError("--initial-target-min-area-pixels must be > 0")
    resolved_lift_success_height = (
        float(lift_target_height)
        if lift_success_height is None
        else float(lift_success_height)
    )
    if resolved_lift_success_height <= 0.0:
        raise ValueError("--lift-success-height must be > 0")
    if resolved_lift_success_height > float(lift_target_height):
        raise ValueError("--lift-success-height cannot exceed --lift-target-height")
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists; pass --overwrite or choose a new --root")
        shutil.rmtree(root)

    features = _lerobot_features(
        height=height,
        width=width,
        use_videos=use_videos,
        include_camera3_duplicate=include_camera3_duplicate,
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="so101",
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    if hasattr(dataset.meta, "update_chunk_settings"):
        dataset.meta.update_chunk_settings(data_files_size_in_mb=int(data_files_size_in_mb))

    if skill_mode not in SKILL_TASKS:
        raise ValueError(f"unknown skill_mode: {skill_mode}")
    task_template = COLOR_SHAPE_SKILL_TASK_TEMPLATES[skill_mode]
    balance_bins = _parse_grid_balance_bins(grid_balance_bins)
    balance_enabled = int(grid_balance_size) > 0 or int(grid_balance_target_per_bin) > 0 or bool(balance_bins)
    workspace_candidates: list[WorkspaceSpawnCandidate] = []
    workspace_catalog = None
    workspace_quota_scheduler = None
    if workspace_spawn_catalog_path is not None:
        if balance_enabled or grid_balance_spawn_lookup or deterministic_camera_bin_lookup:
            raise ValueError(
                "--workspace-spawn-catalog cannot be combined with camera-grid balancing"
            )
        if target_object_yaw_deg is not None:
            raise ValueError(
                "--workspace-spawn-catalog supplies object yaw; "
                "--target-object-yaw-deg must be omitted"
            )
        if workspace_spawn_start_index < 0:
            raise ValueError("--workspace-spawn-start-index must be >= 0")
        if workspace_spawn_candidate_count <= 0:
            raise ValueError("--workspace-spawn-candidate-count must be > 0")
        workspace_catalog = load_workspace_spawn_catalog(workspace_spawn_catalog_path)
        end = workspace_spawn_start_index + workspace_spawn_candidate_count
        workspace_candidates = list(
            workspace_catalog.candidates[workspace_spawn_start_index:end]
        )
        if len(workspace_candidates) != workspace_spawn_candidate_count:
            raise ValueError(
                "workspace spawn candidate range exceeds the catalog: "
                f"start={workspace_spawn_start_index} "
                f"count={workspace_spawn_candidate_count}"
            )
        if episodes > len(workspace_candidates):
            raise ValueError(
                "requested episodes exceed workspace candidate range: "
                f"episodes={episodes} candidates={len(workspace_candidates)}"
            )
        if workspace_catalog.enforce_cell_local_quota:
            accepted_spacing = (
                0.0
                if workspace_catalog.continuous_distribution is None
                else workspace_catalog.continuous_distribution.minimum_spacing_m
            )
            forbidden_positions = _workspace_positions_from_export_reports(
                workspace_spawn_forbidden_reports or []
            )
            workspace_quota_scheduler = WorkspaceCellQuotaScheduler(
                workspace_candidates,
                accepted_minimum_spacing_m=accepted_spacing,
                forbidden_positions=forbidden_positions,
            )
            quota_target = int(
                workspace_quota_scheduler.summary()["target_total"]
            )
            if episodes != quota_target:
                raise ValueError(
                    "cell-local workspace shard requires episodes to match its "
                    f"primary quota: episodes={episodes} quota={quota_target}"
                )
    if balance_enabled:
        if int(grid_balance_size) <= 0:
            raise ValueError("--grid-balance-size must be >0 when grid balancing is enabled")
        if int(grid_balance_target_per_bin) <= 0:
            raise ValueError("--grid-balance-target-per-bin must be >0 when grid balancing is enabled")
        if not balance_bins:
            raise ValueError("--grid-balance-bins is required when grid balancing is enabled")
        episodes = int(grid_balance_target_per_bin) * len(balance_bins)
    balance_counts = {int(bin_id): 0 for bin_id in balance_bins}
    if deterministic_camera_bin_lookup and not balance_enabled:
        raise ValueError("--deterministic-camera-bin-lookup requires grid balance bins and target counts")
    if deterministic_camera_bin_lookup and not grid_balance_spawn_lookup:
        grid_balance_spawn_lookup = True

    config = WristEgoServoConfig(width=width, height=height)
    camera_rig_config = None
    camera_rig_config_sha256 = None
    camera_rig_config_declared = None
    if camera_rig_config_path is not None:
        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            load_so101_camera_rig_render_config,
        )

        camera_rig_config_declared = str(camera_rig_config_path)
        camera_rig_config_path = camera_rig_config_path.expanduser().resolve()
        camera_rig_config = load_so101_camera_rig_render_config(camera_rig_config_path)
        camera_rig_config_sha256 = hashlib.sha256(camera_rig_config_path.read_bytes()).hexdigest()
    env = make_high_contrast_picklift_env(
        target_object_color=target_object_color,
        object_half_sizes=object_half_sizes,
        spawn_center=spawn_center,
        spawn_min_radius=spawn_min_radius,
        spawn_max_radius=spawn_max_radius,
        spawn_angle_half_range_deg=spawn_angle_half_range_deg,
        camera_rig_preset=(None if camera_rig_config is None else camera_rig_config.preset),
        camera_rig_config=camera_rig_config,
    )
    policy_renderers = _make_policy_renderers(env, config)
    teacher_renderers = _make_teacher_renderers(env, config)
    action_space_low = np.asarray(env.action_space.low, dtype=np.float32).copy()
    action_space_high = np.asarray(env.action_space.high, dtype=np.float32).copy()
    allow_diagonal_fixed_jaw = skill_mode not in FULL_GRIP_SKILL_MODES
    spawn_lookup: dict[int, list[list[float]]] = {}
    if int(grid_lookup_start_index) < 0:
        raise ValueError("--grid-lookup-start-index must be >= 0")
    spawn_lookup_next = {int(bin_id): int(grid_lookup_start_index) for bin_id in balance_bins}
    lookup_cache_kind = "none"
    if balance_enabled and grid_balance_spawn_lookup:
        env.reset(seed=seed)
        lookup_started = time.perf_counter()
        lookup_cache_kind = "generated_camera1_spawn"
        if grid_lookup_cache and grid_lookup_cache.exists():
            cache = json.loads(grid_lookup_cache.read_text(encoding="utf-8"))
            _validate_camera1_spawn_lookup_cache(
                cache,
                grid_size=int(grid_balance_size),
                resolution=int(grid_lookup_resolution),
                x_range=(float(grid_lookup_x_min), float(grid_lookup_x_max)),
                y_range=(float(grid_lookup_y_min), float(grid_lookup_y_max)),
                target_object_yaw_deg=target_object_yaw_deg,
            )
            lookup_cache_kind = str(cache.get("candidate_kind", "generated_camera1_spawn"))
            spawn_lookup = {
                int(key): [[float(item) for item in value] for value in values]
                for key, values in cache["lookup"].items()
            }
            print(f"[so101-lerobot] loaded camera1 spawn lookup cache {grid_lookup_cache}", flush=True)
        else:
            print(
                f"[so101-lerobot] building camera1 spawn lookup "
                f"resolution={int(grid_lookup_resolution)} bins={balance_bins}",
                flush=True,
            )
            spawn_lookup = _build_camera1_spawn_lookup(
                env,
                policy_renderers,
                grid_size=int(grid_balance_size),
                x_min=float(grid_lookup_x_min),
                x_max=float(grid_lookup_x_max),
                y_min=float(grid_lookup_y_min),
                y_max=float(grid_lookup_y_max),
                resolution=int(grid_lookup_resolution),
                target_object_yaw_deg=target_object_yaw_deg,
            )
            if grid_lookup_cache:
                grid_lookup_cache.parent.mkdir(parents=True, exist_ok=True)
                grid_lookup_cache.write_text(
                    json.dumps(
                        _camera1_spawn_lookup_cache_payload(
                            spawn_lookup,
                            grid_size=int(grid_balance_size),
                            resolution=int(grid_lookup_resolution),
                            x_range=(float(grid_lookup_x_min), float(grid_lookup_x_max)),
                            y_range=(float(grid_lookup_y_min), float(grid_lookup_y_max)),
                            target_object_yaw_deg=target_object_yaw_deg,
                        ),
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
        lookup_build_seconds = time.perf_counter() - lookup_started
        if not grid_lookup_preserve_order:
            for candidates_xy in spawn_lookup.values():
                candidates_xy.sort(
                    key=lambda xy: (float(xy[0]) - float(spawn_center[0])) ** 2
                    + (float(xy[1]) - float(spawn_center[1])) ** 2
                )
        spawn_lookup = {int(bin_id): spawn_lookup.get(int(bin_id), []) for bin_id in balance_bins}
        if grid_balance_teacher_feasible_lookup and lookup_cache_kind != "trajectory_feasible":
            print("[so101-lerobot] filtering spawn lookup by teacher feasibility", flush=True)
            spawn_lookup = _filter_spawn_lookup_for_teacher_feasibility(
                env,
                policy_renderers,
                config=config,
                spawn_lookup=spawn_lookup,
                seed=seed,
                move_target_z_offset=move_target_z_offset,
                edge_contact_xy_success_threshold=edge_contact_xy_success_threshold,
                edge_contact_parallel_success_threshold_deg=edge_contact_parallel_success_threshold_deg,
                max_candidates_per_bin=int(grid_lookup_max_candidates_per_bin),
                allow_diagonal_fixed_jaw=allow_diagonal_fixed_jaw,
                min_gripper_floor_clearance_m=float(min_gripper_floor_clearance_m),
                close_steps=max(1, int(close_steps)),
                target_object_yaw_deg=target_object_yaw_deg,
            )
        missing = [int(bin_id) for bin_id in balance_bins if not spawn_lookup.get(int(bin_id))]
        if missing:
            raise RuntimeError(f"camera1 spawn lookup has no candidates for bins: {missing}")
        print(
            "[so101-lerobot] camera1 spawn lookup "
            + json.dumps({str(key): len(value) for key, value in sorted(spawn_lookup.items())}, sort_keys=True),
            flush=True,
        )
    else:
        lookup_build_seconds = 0.0
    exported = 0
    attempted = 0
    attempted_episode_seeds: set[int] = set()
    skipped = []
    episode_summaries = []
    initial_visibility_rejections = 0
    render_replay_captures: list[dict[str, Any]] = []
    try:
        candidate_seed = seed
        attempt_limit = int(episodes) * int(max_attempt_multiplier)
        if workspace_candidates:
            attempt_limit = min(attempt_limit, len(workspace_candidates))
        while exported < episodes and attempted < attempt_limit:
            attempted += 1
            if attempted % 50 == 0:
                print(
                    f"[so101-lerobot] attempts={attempted} exported={exported}/{episodes} "
                    f"grid_counts={json.dumps({str(k): int(v) for k, v in sorted(balance_counts.items())}, sort_keys=True)}",
                    flush=True,
                )
            desired_grid_bin = None
            forced_spawn_xy = None
            workspace_candidate = None
            forced_spawn_yaw_deg = target_object_yaw_deg
            if workspace_quota_scheduler is not None:
                workspace_candidate = workspace_quota_scheduler.next_candidate()
                forced_spawn_xy = list(workspace_candidate.world_xy_m)
                forced_spawn_yaw_deg = float(workspace_candidate.object_yaw_deg)
            elif workspace_candidates:
                workspace_candidate = workspace_candidates[attempted - 1]
                forced_spawn_xy = list(workspace_candidate.world_xy_m)
                forced_spawn_yaw_deg = float(workspace_candidate.object_yaw_deg)
            if spawn_lookup:
                remaining = [bin_id for bin_id in balance_bins if balance_counts[int(bin_id)] < int(grid_balance_target_per_bin)]
                if not remaining:
                    break
                min_count = min(balance_counts[int(bin_id)] for bin_id in remaining)
                least_filled = [int(bin_id) for bin_id in remaining if balance_counts[int(bin_id)] == min_count]
                desired_grid_bin = least_filled[(attempted - 1) % len(least_filled)]
                candidates_xy = spawn_lookup[desired_grid_bin]
                next_index = spawn_lookup_next[desired_grid_bin]
                forced_spawn_xy = _take_unique_spawn_candidate(
                    candidates_xy,
                    next_index=next_index,
                    bin_id=desired_grid_bin,
                    accepted=balance_counts[desired_grid_bin],
                    target=int(grid_balance_target_per_bin),
                )
                spawn_lookup_next[desired_grid_bin] = next_index + 1
            episode_seed = candidate_seed
            candidate_seed += 1
            if deterministic_camera_bin_lookup and forced_spawn_xy is not None:
                # The lookup index, not the rejection attempt count, defines
                # the environment seed. Re-running the same bin sequence is
                # therefore bit-for-bit reproducible.
                episode_seed = int(seed) + int(desired_grid_bin) * 100000 + int(next_index)
            if forced_spawn_xy is not None and len(forced_spawn_xy) >= 3:
                episode_seed = int(forced_spawn_xy[2])
            if episode_seed in attempted_episode_seeds:
                raise RuntimeError(
                    f"duplicate episode seed detected before simulation: seed={episode_seed}. "
                    "Seed reuse is forbidden for SO101 dataset generation."
                )
            attempted_episode_seeds.add(episode_seed)
            env.reset(seed=episode_seed)
            if forced_spawn_xy is not None:
                _set_target_object_xy(env, forced_spawn_xy)
            if forced_spawn_yaw_deg is not None:
                _set_target_object_yaw(env, float(forced_spawn_yaw_deg))
            if initial_qpos is not None:
                requested_initial_qpos = np.asarray(initial_qpos, dtype=np.float32)
                clipped_initial_qpos = np.clip(
                    requested_initial_qpos, env.action_space.low, env.action_space.high
                ).astype(np.float32)
                if not np.allclose(requested_initial_qpos, clipped_initial_qpos, atol=1e-7):
                    raise ValueError(
                        "initial qpos is outside the simulator action contract: "
                        f"requested={requested_initial_qpos.tolist()} "
                        f"clipped={clipped_initial_qpos.tolist()}"
                    )
                _set_qpos(env, clipped_initial_qpos)
            reset_home_qpos = _current_qpos(env).astype(np.float32)
            target_object = _target_object_metadata(env)
            episode_task = _format_skill_task(skill_mode, target_object)
            if target_object_color and target_object["color"] != target_object_color:
                skipped.append(
                    {
                        "seed": episode_seed,
                        "reason": "target_object_color_mismatch",
                        "object_color": target_object["color"],
                        "required_object_color": target_object_color,
                    }
                )
                continue
            if require_initial_target_visible:
                initial_visibility = _policy_camera_visibility(
                    env,
                    policy_renderers,
                    minimum_area=int(initial_target_min_area_pixels),
                )
                visible_in_selected_camera = any(
                    bool(initial_visibility[camera_key]["visible"])
                    for camera_key in requested_visibility_cameras
                )
                if not visible_in_selected_camera:
                    initial_visibility_rejections += 1
                    skipped.append(
                        {
                            "seed": episode_seed,
                            "reason": "initial_target_not_visible",
                            "selected_cameras": list(requested_visibility_cameras),
                            "minimum_area_pixels": int(initial_target_min_area_pixels),
                            "visibility": initial_visibility,
                            "forced_spawn_xy": forced_spawn_xy,
                        }
                    )
                    continue
            teacher_visible = object_visible_to_teacher(env, teacher_renderers, config=config)
            visible, search_steps = sweep_until_visible(env, policy_renderers, max_sweeps=config.max_sweeps)
            teacher_visible = teacher_visible or object_visible_to_teacher(env, teacher_renderers, config=config)
            if not visible:
                skipped.append({"seed": episode_seed, "reason": "not_visible_after_sweep"})
                continue
            if balance_enabled and not _grid_balance_needs_teacher_candidate_for_start(
                skill_mode=skill_mode,
                episode_index=exported,
                move_and_align_near_target_correction_ratio=move_and_align_near_target_correction_ratio,
            ):
                grid_bin = _camera1_grid_bin_at_qpos(
                    env,
                    policy_renderers,
                    qpos=reset_home_qpos,
                    grid_size=int(grid_balance_size),
                )
                if grid_bin not in balance_counts:
                    skipped.append(
                        {
                            "seed": episode_seed,
                            "reason": "grid_balance_bin_not_requested_pre_teacher",
                            "grid_bin": grid_bin,
                            "desired_grid_bin": desired_grid_bin,
                            "forced_spawn_xy": forced_spawn_xy,
                            "requested_bins": sorted(balance_counts),
                        }
                    )
                    continue
                if balance_counts[grid_bin] >= int(grid_balance_target_per_bin):
                    skipped.append(
                        {
                            "seed": episode_seed,
                            "reason": "grid_balance_bin_full_pre_teacher",
                            "grid_bin": grid_bin,
                            "desired_grid_bin": desired_grid_bin,
                            "forced_spawn_xy": forced_spawn_xy,
                            "target_per_bin": int(grid_balance_target_per_bin),
                        }
                    )
                    continue
            if skill_mode in FULL_GRIP_SKILL_MODES:
                candidates = _make_full_grip_teacher_targets_for_skill(
                    env,
                    skill_mode=skill_mode,
                    min_floor_clearance_m=float(min_gripper_floor_clearance_m),
                )
            elif skill_mode in FIXED_JAW_SKILL_MODES:
                candidates = _make_fast_fixed_jaw_teacher_targets(
                    env,
                    allow_diagonal=allow_diagonal_fixed_jaw,
                    min_floor_clearance_m=float(min_gripper_floor_clearance_m),
                )
            else:
                candidates = make_teacher_targets(env)
            if skill_mode in {"move_over_cube", "pick_from_top_cube", *FIXED_JAW_SKILL_MODES} and skill_mode not in FULL_GRIP_SKILL_MODES:
                candidates = [
                    candidate
                    for candidate in candidates
                    if str(candidate["meta"].get("mode")) == "overhead"
                ]
            if skill_mode in FIXED_JAW_SKILL_MODES:
                candidates = _filter_fixed_jaw_move_candidates_in_policy_view(
                    env,
                    renderers=policy_renderers,
                    candidates=candidates,
                    move_target_z_offset=move_target_z_offset,
                )
            if not candidates:
                skipped.append({"seed": episode_seed, "reason": "no_successful_teacher_candidate"})
                continue
            ranked_candidates = sorted(candidates, key=lambda item: float(item["meta"].get("score", -1e9)), reverse=True)
            candidate_failures: list[dict[str, Any]] = []
            summary = None
            candidate_start_snapshot = _snapshot_sim_state(env)
            for candidate_rank, best in enumerate(ranked_candidates):
                # A failed candidate can leave the cube lifted, rotated, or in
                # contact. Every candidate must start from the same settled
                # episode state so the accepted trajectory and its replay
                # snapshot are independent of retry order.
                _restore_sim_state(env, candidate_start_snapshot)
                candidate_meta = dict(best["meta"])
                candidate_meta["teacher_candidate_rank"] = int(candidate_rank)
                candidate_meta["teacher_candidate_count"] = int(len(ranked_candidates))
                summary = _write_teacher_episode(
                    dataset=dataset,
                    env=env,
                    renderers=policy_renderers,
                    q_open=np.asarray(best["q_open"], dtype=np.float32),
                    q_lift=np.asarray(best["q_lift"], dtype=np.float32),
                    seed=episode_seed,
                    search_steps=search_steps,
                    teacher_visible=teacher_visible,
                    best_meta=candidate_meta,
                    teacher_style=teacher_style,
                    approach_steps=approach_steps,
                    settle_steps=settle_steps,
                    close_steps=close_steps,
                    trajectory_variant=trajectory_variant,
                    grip_the_cube_start_profile=grip_the_cube_start_profile,
                    close_alignment_gate_mode=close_alignment_gate_mode,
                    close_alignment_limits=close_alignment_limits,
                    lift_steps=lift_steps,
                    lift_target_height=lift_target_height,
                    lift_success_height=resolved_lift_success_height,
                    lift_controller_z_error=lift_controller_z_error,
                    start_mode=start_mode,
                    near_gripper_joint_std=near_gripper_joint_std,
                    skill_mode=skill_mode,
                    task=episode_task,
                    episode_index=exported,
                    random_start_joint_std=random_start_joint_std,
                    move_success_tcp_dist=move_success_tcp_dist,
                    move_target_z_offset=move_target_z_offset,
                    closed_gripper_prob=closed_gripper_prob,
                    move_gripper_profile=move_gripper_profile,
                    move_min_actual_z=move_min_actual_z,
                    terminal_hold_steps=terminal_hold_steps,
                    move_and_align_near_target_correction_ratio=move_and_align_near_target_correction_ratio,
                    edge_contact_xy_success_threshold=edge_contact_xy_success_threshold,
                    edge_contact_parallel_success_threshold_deg=edge_contact_parallel_success_threshold_deg,
                    near_target_joint_std=near_target_joint_std,
                    near_target_xy_std=near_target_xy_std,
                    pick_start_joint_std=pick_start_joint_std,
                    pick_correction_steps=pick_correction_steps,
                    pick_start_min_abs_y=pick_start_min_abs_y,
                    pick_start_max_abs_y=pick_start_max_abs_y,
                    pick_start_min_actual_abs_y=pick_start_min_actual_abs_y,
                    pick_start_min_actual_z=pick_start_min_actual_z,
                    above_edge_start_joint_std=above_edge_start_joint_std,
                    above_edge_start_xy_std=above_edge_start_xy_std,
                    above_edge_start_z_std=above_edge_start_z_std,
                    above_edge_start_min_actual_z=above_edge_start_min_actual_z,
                    above_edge_trajectory_variants=above_edge_trajectory_variants,
                    above_edge_start_gripper_profile=above_edge_start_gripper_profile,
                    above_edge_terminal_hold_jitter=above_edge_terminal_hold_jitter,
                    include_camera3_duplicate=include_camera3_duplicate,
                    capture_render_replay=capture_render_replay,
                    capture_fps=fps,
                    reset_home_qpos=reset_home_qpos,
                    exact_start_pose=_uses_exact_initial_qpos(
                        initial_qpos,
                        mode=initial_qpos_mode,
                    ),
                    min_gripper_floor_clearance_m=min_gripper_floor_clearance_m,
                )
                if summary["success"]:
                    break
                summary.pop("_render_replay_capture", None)
                candidate_failures.append(
                    {
                        "candidate_rank": int(candidate_rank),
                        "candidate_mode": candidate_meta.get("candidate_mode"),
                        "reason": summary.get("reason", "teacher_replay_failed"),
                        "final_info": summary.get("final_info"),
                        "pre_close_static_edge_error": summary.get("pre_close_static_edge_error"),
                        "camera2_top_contact_close_alignment_gate": (
                            summary.get("best_meta", {}).get("camera2_top_contact_close_alignment_gate")
                            if isinstance(summary.get("best_meta"), dict)
                            else None
                        ),
                        "jaw_vertical_angle_deg": summary.get("best_meta", {}).get("jaw_vertical_angle_deg"),
                        "gripper_floor_clearance_gate": summary.get("gripper_floor_clearance_gate"),
                    }
                )
            assert summary is not None
            if candidate_failures:
                summary["failed_teacher_candidates"] = candidate_failures
            summary["task"] = episode_task
            summary["task_template"] = task_template
            summary["target_object"] = target_object
            summary["object_color"] = target_object["color"]
            summary["object_shape"] = target_object["shape"]
            if forced_spawn_xy is not None:
                summary["forced_spawn_xy"] = [float(forced_spawn_xy[0]), float(forced_spawn_xy[1])]
                summary["desired_grid_bin"] = desired_grid_bin
            if workspace_candidate is not None:
                summary["workspace_spawn"] = {
                    **workspace_candidate.model_dump(mode="json"),
                    "catalog": str(workspace_spawn_catalog_path),
                }
                if workspace_candidate.camera1_grid_bin is not None:
                    summary["camera1_grid_bin"] = int(
                        workspace_candidate.camera1_grid_bin
                    )
            if summary["success"]:
                if balance_enabled:
                    use_declared_spawn_bin = bool(
                        str(grip_the_cube_start_profile) == "correction"
                        and bool(deterministic_camera_bin_lookup)
                        and desired_grid_bin is not None
                        and forced_spawn_xy is not None
                    )
                    grid_bin = (
                        int(desired_grid_bin)
                        if use_declared_spawn_bin
                        else _summary_start_grid_bin(summary, grid_size=int(grid_balance_size))
                    )
                    summary["grid_balance_bin"] = grid_bin
                    summary["grid_balance_bin_source"] = (
                        "declared_camera1_spawn_catalog"
                        if use_declared_spawn_bin
                        else "start_camera1_centroid"
                    )
                    if grid_bin not in balance_counts:
                        _clear_episode_buffer_robust(dataset)
                        skipped.append(
                            {
                                "seed": episode_seed,
                                "reason": "grid_balance_bin_not_requested",
                                "grid_bin": grid_bin,
                                "requested_bins": sorted(balance_counts),
                            }
                        )
                        continue
                    if balance_counts[grid_bin] >= int(grid_balance_target_per_bin):
                        _clear_episode_buffer_robust(dataset)
                        skipped.append(
                            {
                                "seed": episode_seed,
                                "reason": "grid_balance_bin_full",
                                "grid_bin": grid_bin,
                                "target_per_bin": int(grid_balance_target_per_bin),
                            }
                        )
                        continue
                    balance_counts[grid_bin] += 1
                replay_capture = summary.pop("_render_replay_capture", None)
                if replay_capture is not None:
                    replay_capture["episode_index"] = int(exported)
                    for frame in replay_capture["frames"]:
                        frame["episode_index"] = int(exported)
                    replay_capture["target_slot_index"] = int(
                        (target_object or {}).get("target_slot_index", -1)
                    )
                    target_slot = int(replay_capture["target_slot_index"])
                    if target_slot >= 0 and replay_capture["frames"]:
                        target_joint = env.unwrapped.model.joint(f"pick_slot_{target_slot}_joint")
                        target_qpos_adr = int(np.asarray(target_joint.qposadr).item())
                        replay_capture["initial_object_z"] = float(
                            replay_capture["frames"][0]["qpos"][target_qpos_adr + 2]
                        )
                    render_replay_captures.append(replay_capture)
                dataset.save_episode()
                if workspace_quota_scheduler is not None:
                    workspace_quota_scheduler.record_success(workspace_candidate)
                exported += 1
                episode_summaries.append(summary)
                print(
                    f"[so101-lerobot] exported {exported}/{episodes} "
                    f"seed={summary['seed']} frames={summary['frames']} "
                    f"mode={summary['best_meta'].get('mode')} "
                    f"grid_bin={summary.get('grid_balance_bin')}",
                    flush=True,
                )
            else:
                _clear_episode_buffer_robust(dataset)
                skipped.append({"seed": episode_seed, "reason": "teacher_replay_failed", **summary})
                print(
                    "[so101-lerobot] rejected "
                    f"seed={episode_seed} reason={summary.get('reason')} "
                    f"grasped={summary.get('final_info', {}).get('is_grasped')} "
                    f"lift={summary.get('final_info', {}).get('lift_height')} "
                    f"candidate_reasons={[row.get('reason') for row in candidate_failures]}",
                    flush=True,
                )
        if workspace_quota_scheduler is not None and not workspace_quota_scheduler.complete:
            raise RuntimeError(
                "workspace cell-local quota was not completed: "
                + json.dumps(workspace_quota_scheduler.summary(), sort_keys=True)
            )
        if capture_render_replay:
            from physical_ai_agent.so101_render_replay import write_captured_render_replay_sidecar

            write_captured_render_replay_sidecar(
                root,
                model=env.unwrapped.model,
                episode_captures=render_replay_captures,
                environment={
                    "factory": "make_high_contrast_picklift_env",
                    "target_object_color": target_object_color,
                    "object_half_sizes": [float(value) for value in object_half_sizes],
                    "spawn_center": [float(spawn_center[0]), float(spawn_center[1])],
                    "spawn_min_radius": float(spawn_min_radius),
                    "spawn_max_radius": float(spawn_max_radius),
                    "spawn_angle_half_range_deg": float(spawn_angle_half_range_deg),
                    "target_object_yaw_deg": (
                        None
                        if target_object_yaw_deg is None
                        else float(target_object_yaw_deg)
                    ),
                    "n_distractors": 0,
                    "action_repeat": 1,
                    "camera_rig_config": (
                        camera_rig_config_declared
                    ),
                    "camera_rig_sha256": camera_rig_config_sha256,
                    "object_pool_order": [
                        {
                            "slot": int(index),
                            "color": str(target_object_color),
                            "half_size": float(half_size),
                        }
                        for index, half_size in enumerate(object_half_sizes)
                    ],
                },
            )
    finally:
        for renderer in [*policy_renderers.values(), *teacher_renderers.values()]:
            renderer.close()
        env.close()

    exported_seeds = [int(summary["seed"]) for summary in episode_summaries]
    if len(exported_seeds) != len(set(exported_seeds)):
        raise RuntimeError(
            "duplicate seeds detected in exported episodes; refusing to finalize the dataset"
        )
    dataset.finalize()
    audit = audit_lerobot_dataset(
        root=root,
        repo_id=repo_id,
        features=features,
        action_space_low=action_space_low,
        action_space_high=action_space_high,
    )
    report = {
        "operation": "export_so101_teacher_rollouts_lerobot",
        "root": str(root),
        "repo_id": repo_id,
        "task": task_template,
        "task_template": task_template,
        "task_generation": "episode-specific color/shape prompt from target object metadata",
        "skill_mode": skill_mode,
        "requested_episodes": episodes,
        "exported_episodes": exported,
        "attempted_seeds": attempted,
        "seed_uniqueness": {
            "required": True,
            "attempted_unique_seeds": len(attempted_episode_seeds),
            "exported_unique_seeds": len(set(exported_seeds)),
            "duplicate_exported_seeds": 0,
            "passed": len(exported_seeds) == len(set(exported_seeds)),
        },
        "generation_strategy": (
            "weighted_workspace_catalog_fixed_jaw_ik"
            if workspace_candidates
            else (
                "deterministic_camera1_bin_lookup_fixed_jaw_ik"
                if deterministic_camera_bin_lookup
                else "seed_rejection_with_fixed_jaw_ik"
            )
        ),
        "generation_timing": {
            "total_seconds": float(time.perf_counter() - export_started),
            "lookup_build_seconds": float(lookup_build_seconds),
            "seconds_per_exported_episode": (
                float(time.perf_counter() - export_started) / float(exported) if exported else None
            ),
        },
        "initial_target_visibility_gate": {
            "enabled": bool(require_initial_target_visible),
            "camera_keys": list(requested_visibility_cameras),
            "mode": "any",
            "minimum_area_pixels": int(initial_target_min_area_pixels),
            "rejected_candidates": int(initial_visibility_rejections),
            "passed_exported_episodes": int(exported),
        },
        "grid_balance": {
            "enabled": bool(balance_enabled),
            "grid_size": int(grid_balance_size),
            "target_per_bin": int(grid_balance_target_per_bin),
            "requested_bins": sorted(balance_counts),
            "accepted_counts": {str(key): int(value) for key, value in sorted(balance_counts.items())},
            "spawn_lookup": {
                "enabled": bool(spawn_lookup),
                "resolution": int(grid_lookup_resolution),
                "start_index": int(grid_lookup_start_index),
                "x_range": [float(grid_lookup_x_min), float(grid_lookup_x_max)],
                "y_range": [float(grid_lookup_y_min), float(grid_lookup_y_max)],
                "candidate_counts": {str(key): len(value) for key, value in sorted(spawn_lookup.items())},
                "teacher_feasible_filter": bool(grid_balance_teacher_feasible_lookup),
                "cache_kind": lookup_cache_kind,
                "max_candidates_per_bin": int(grid_lookup_max_candidates_per_bin),
                "cache": str(grid_lookup_cache) if grid_lookup_cache else None,
                "deterministic": bool(deterministic_camera_bin_lookup),
            },
        },
        "workspace_spawn": {
            "enabled": bool(workspace_candidates),
            "catalog": (
                None
                if workspace_spawn_catalog_path is None
                else str(workspace_spawn_catalog_path)
            ),
            "catalog_id": (
                None if workspace_catalog is None else workspace_catalog.catalog_id
            ),
            "start_index": int(workspace_spawn_start_index),
            "candidate_count": int(len(workspace_candidates)),
            "forbidden_reports": [
                str(path) for path in (workspace_spawn_forbidden_reports or [])
            ],
            "source_cell_count": (
                None if workspace_catalog is None else workspace_catalog.source_cell_count
            ),
            "distance_decay_rate_per_m": (
                None
                if workspace_catalog is None
                else workspace_catalog.distance_decay_rate_per_m
            ),
            "cell_local_quota": (
                None
                if workspace_quota_scheduler is None
                else workspace_quota_scheduler.summary()
            ),
        },
        "fps": fps,
        "use_videos": use_videos,
        "teacher_style": teacher_style,
        "teacher_timing": {
            "approach_steps": int(approach_steps),
            "settle_steps": int(settle_steps),
            "close_steps": int(close_steps),
            "close_alignment_gate_mode": str(close_alignment_gate_mode),
            "close_alignment_limits": dict(close_alignment_limits or {}),
            "trajectory_variant": str(trajectory_variant),
            "grip_the_cube_start_profile": str(grip_the_cube_start_profile),
            "lift_steps": int(lift_steps),
            "start_mode": str(start_mode),
            "near_gripper_joint_std": float(near_gripper_joint_std),
            "random_start_joint_std": float(random_start_joint_std),
            "move_success_tcp_dist": float(move_success_tcp_dist),
            "move_target_z_offset": float(move_target_z_offset),
            "closed_gripper_prob": float(closed_gripper_prob),
            "move_gripper_profile": str(move_gripper_profile),
            "move_min_actual_z": float(move_min_actual_z),
            "terminal_hold_steps": int(terminal_hold_steps),
            "move_and_align_near_target_correction_ratio": float(move_and_align_near_target_correction_ratio),
            "edge_contact_xy_success_threshold": float(edge_contact_xy_success_threshold),
            "edge_contact_parallel_success_threshold_deg": float(edge_contact_parallel_success_threshold_deg),
            "near_target_joint_std": float(near_target_joint_std),
            "near_target_xy_std": float(near_target_xy_std),
            "pick_start_joint_std": float(pick_start_joint_std),
            "pick_correction_steps": int(pick_correction_steps),
            "pick_start_min_abs_y": float(pick_start_min_abs_y),
            "pick_start_max_abs_y": float(pick_start_max_abs_y),
            "pick_start_min_actual_abs_y": float(pick_start_min_actual_abs_y),
            "pick_start_min_actual_z": float(pick_start_min_actual_z),
            "above_edge_start_joint_std": float(above_edge_start_joint_std),
            "above_edge_start_xy_std": float(above_edge_start_xy_std),
            "above_edge_start_z_std": float(above_edge_start_z_std),
            "above_edge_start_min_actual_z": float(above_edge_start_min_actual_z),
            "above_edge_trajectory_variants": str(above_edge_trajectory_variants),
            "above_edge_start_gripper_profile": str(above_edge_start_gripper_profile),
            "above_edge_terminal_hold_jitter": int(above_edge_terminal_hold_jitter),
            "target_object_color": target_object_color,
            "spawn_center": [float(spawn_center[0]), float(spawn_center[1])],
            "spawn_min_radius": float(spawn_min_radius),
            "spawn_max_radius": float(spawn_max_radius),
            "spawn_angle_half_range_deg": float(spawn_angle_half_range_deg),
            "target_object_yaw_deg": (
                None if target_object_yaw_deg is None else float(target_object_yaw_deg)
            ),
            "object_half_sizes": [float(value) for value in object_half_sizes],
            "initial_qpos": (
                None if initial_qpos is None else [float(value) for value in initial_qpos]
            ),
            "initial_qpos_mode": str(initial_qpos_mode),
            "camera_rig_config": (
                camera_rig_config_declared
            ),
            "camera_rig_sha256": camera_rig_config_sha256,
            "min_gripper_floor_clearance_m": float(min_gripper_floor_clearance_m),
        },
        "camera3_duplicate": {
            "enabled": bool(include_camera3_duplicate),
            "source": "wrist_cam",
            "reason": "lerobot/smolvla_base expects camera2 to carry the eye-in-hand/wrist view; camera3 duplicates camera2 when requested.",
        },
        "feature_mapping": {
            "observation.images.camera1": "egocentric_cam",
            "observation.images.camera2": "wrist_cam",
            **({"observation.images.camera3": "wrist_cam duplicate"} if include_camera3_duplicate else {}),
            "observation.state": "SO101 qpos/control state",
            "action": "SO101 qpos target action",
            "task": "episode-specific color/shape prompt",
        },
        "official_camera_contract": {
            "dataset": "SO101 egocentric+wrist visual-student dataset aligned to the local real-hardware policy cameras",
            "dataset_features": ["observation.images.egocentric_cam", "observation.images.wrist_cam"],
            "rename_map": {
                "observation.images.egocentric_cam": "observation.images.camera1",
                "observation.images.wrist_cam": "observation.images.camera2",
            },
            "camera1_pose": EGOCENTRIC_CAMERA1_POSE,
            "local_verification": "Student inputs use egocentric_cam and wrist_cam; top_down is debug-only and must not be fed to SmolVLA.",
        },
        "action_normalization": {
            "producer": "raw SO101 qpos target in simulator action-space units",
            "expected_smolvla_mode": "MEAN_STD from LeRobotDataset stats",
            "manual_scaling_applied": False,
        },
        "dataset_generation_augmentation": {
            "kind": "teacher_trajectory_generation",
            "terminal_hold_included": int(terminal_hold_steps) > 0,
            "terminal_hold_steps": int(terminal_hold_steps),
            "lift_target_height": float(lift_target_height),
            "lift_success_height": float(resolved_lift_success_height),
            "lift_controller_z_error": float(lift_controller_z_error),
            "near_target_correction_included": bool(
                float(move_and_align_near_target_correction_ratio) > 0.0
                or str(grip_the_cube_start_profile) == "correction"
            ),
            "near_target_correction_ratio": float(move_and_align_near_target_correction_ratio),
            "grip_the_cube_start_profile": str(grip_the_cube_start_profile),
            "near_target_joint_std": float(near_target_joint_std),
            "near_target_xy_std": float(near_target_xy_std),
            "note": "This is dataset generation augmentation, distinct from train-time image/state augmentation.",
        },
        "episodes": episode_summaries,
        "skipped": skipped,
        "audit": audit,
    }
    report_path = root / "so101_lerobot_export_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _uses_exact_initial_qpos(
    initial_qpos: tuple[float, ...] | None,
    *,
    mode: str,
) -> bool:
    if mode not in {"exact", "reset_only"}:
        raise ValueError("initial qpos mode must be exact or reset_only")
    return initial_qpos is not None and mode == "exact"


def _write_teacher_episode(
    *,
    dataset: Any,
    env: Any,
    renderers: dict[str, Any],
    q_open: np.ndarray,
    q_lift: np.ndarray,
    seed: int,
    search_steps: int,
    teacher_visible: bool,
    best_meta: dict[str, Any],
    teacher_style: str,
    approach_steps: int,
    settle_steps: int,
    close_steps: int,
    trajectory_variant: str,
    grip_the_cube_start_profile: str,
    close_alignment_gate_mode: str,
    close_alignment_limits: dict[str, float] | None,
    lift_steps: int,
    lift_target_height: float,
    lift_success_height: float,
    lift_controller_z_error: float,
    start_mode: str,
    near_gripper_joint_std: float,
    skill_mode: str,
    task: str,
    episode_index: int,
    random_start_joint_std: float,
    move_success_tcp_dist: float,
    move_target_z_offset: float,
    closed_gripper_prob: float,
    move_gripper_profile: str,
    move_min_actual_z: float,
    terminal_hold_steps: int,
    move_and_align_near_target_correction_ratio: float,
    edge_contact_xy_success_threshold: float,
    edge_contact_parallel_success_threshold_deg: float,
    near_target_joint_std: float,
    near_target_xy_std: float,
    pick_start_joint_std: float,
    pick_correction_steps: int,
    pick_start_min_abs_y: float,
    pick_start_max_abs_y: float,
    pick_start_min_actual_abs_y: float,
    pick_start_min_actual_z: float,
    above_edge_start_joint_std: float,
    above_edge_start_xy_std: float,
    above_edge_start_z_std: float,
    above_edge_start_min_actual_z: float,
    above_edge_trajectory_variants: str,
    above_edge_start_gripper_profile: str,
    above_edge_terminal_hold_jitter: int,
    include_camera3_duplicate: bool,
    capture_render_replay: bool = False,
    capture_fps: int = 12,
    reset_home_qpos: np.ndarray | None = None,
    exact_start_pose: bool = False,
    min_gripper_floor_clearance_m: float = 0.0,
) -> dict[str, Any]:
    if teacher_style == "legacy":
        return _write_legacy_teacher_episode(
            dataset=dataset,
            env=env,
            renderers=renderers,
            q_open=q_open,
            q_lift=q_lift,
            seed=seed,
            search_steps=search_steps,
            teacher_visible=teacher_visible,
            best_meta=best_meta,
            task=task,
            include_camera3_duplicate=include_camera3_duplicate,
            reset_home_qpos=reset_home_qpos,
        )

    if skill_mode == "move_over_cube":
        return _write_move_over_cube_episode(
            dataset=dataset,
            env=env,
            renderers=renderers,
            q_open=q_open,
            seed=seed,
            search_steps=search_steps,
            teacher_visible=teacher_visible,
            best_meta=best_meta,
            approach_steps=approach_steps,
            settle_steps=settle_steps,
            episode_index=episode_index,
            random_start_joint_std=random_start_joint_std,
            move_success_tcp_dist=move_success_tcp_dist,
            move_target_z_offset=move_target_z_offset,
            closed_gripper_prob=closed_gripper_prob,
            move_gripper_profile=move_gripper_profile,
            move_min_actual_z=move_min_actual_z,
            task=task,
            include_camera3_duplicate=include_camera3_duplicate,
        )

    if skill_mode == "pick_from_top_cube":
        return _write_pick_from_top_cube_episode(
            dataset=dataset,
            env=env,
            renderers=renderers,
            q_open=q_open,
            seed=seed,
            search_steps=search_steps,
            teacher_visible=teacher_visible,
            best_meta=best_meta,
            close_steps=close_steps,
            lift_steps=lift_steps,
            episode_index=episode_index,
            move_target_z_offset=move_target_z_offset,
            pick_start_joint_std=pick_start_joint_std,
            pick_correction_steps=pick_correction_steps,
            pick_start_min_abs_y=pick_start_min_abs_y,
            pick_start_max_abs_y=pick_start_max_abs_y,
            pick_start_min_actual_abs_y=pick_start_min_actual_abs_y,
            pick_start_min_actual_z=pick_start_min_actual_z,
            task=task,
            include_camera3_duplicate=include_camera3_duplicate,
        )

    if skill_mode in FIXED_JAW_SKILL_MODES:
        return _write_fixed_jaw_edge_episode(
            dataset=dataset,
            env=env,
            renderers=renderers,
            q_open=q_open,
            seed=seed,
            search_steps=search_steps,
            teacher_visible=teacher_visible,
            best_meta=best_meta,
            skill_mode=skill_mode,
            approach_steps=approach_steps,
            settle_steps=settle_steps,
            close_steps=close_steps,
            close_alignment_gate_mode=close_alignment_gate_mode,
            close_alignment_limits=close_alignment_limits,
            trajectory_variant=trajectory_variant,
            grip_the_cube_start_profile=grip_the_cube_start_profile,
            lift_steps=lift_steps,
            lift_target_height=lift_target_height,
            lift_success_height=lift_success_height,
            lift_controller_z_error=lift_controller_z_error,
            episode_index=episode_index,
            random_start_joint_std=random_start_joint_std,
            move_target_z_offset=move_target_z_offset,
            terminal_hold_steps=terminal_hold_steps,
            move_and_align_near_target_correction_ratio=move_and_align_near_target_correction_ratio,
            edge_contact_xy_success_threshold=edge_contact_xy_success_threshold,
            edge_contact_parallel_success_threshold_deg=edge_contact_parallel_success_threshold_deg,
            near_target_joint_std=near_target_joint_std,
            near_target_xy_std=near_target_xy_std,
            above_edge_start_joint_std=above_edge_start_joint_std,
            above_edge_start_xy_std=above_edge_start_xy_std,
            above_edge_start_z_std=above_edge_start_z_std,
            above_edge_start_min_actual_z=above_edge_start_min_actual_z,
            above_edge_trajectory_variants=above_edge_trajectory_variants,
            above_edge_start_gripper_profile=above_edge_start_gripper_profile,
            above_edge_terminal_hold_jitter=above_edge_terminal_hold_jitter,
            task=task,
            include_camera3_duplicate=include_camera3_duplicate,
            capture_render_replay=capture_render_replay,
            capture_fps=capture_fps,
            reset_home_qpos=reset_home_qpos,
            exact_start_pose=exact_start_pose,
            min_gripper_floor_clearance_m=min_gripper_floor_clearance_m,
        )

    return _write_staged_teacher_episode(
        dataset=dataset,
        env=env,
        renderers=renderers,
        q_open=q_open,
        seed=seed,
        search_steps=search_steps,
        teacher_visible=teacher_visible,
        best_meta=best_meta,
        approach_steps=approach_steps,
        settle_steps=settle_steps,
        close_steps=close_steps,
        lift_steps=lift_steps,
        start_mode=start_mode,
        near_gripper_joint_std=near_gripper_joint_std,
        task=task,
        include_camera3_duplicate=include_camera3_duplicate,
    )


def _target_object_metadata(env: Any) -> dict[str, Any]:
    unwrapped = env.unwrapped
    objects = list(getattr(getattr(unwrapped, "config", None), "objects", []) or [])
    target_index = int(getattr(unwrapped, "_target_slot_idx", 0) or 0)
    obj = objects[target_index] if 0 <= target_index < len(objects) else None
    color = str(getattr(obj, "color", "") or "").strip().lower()
    shape = _object_shape_name(obj)

    if not color:
        description = str(getattr(unwrapped, "_task_description", "") or "").lower()
        for candidate in ("red", "blue", "green", "yellow", "orange", "purple", "black", "white"):
            if candidate in description:
                color = candidate
                break
    if not color:
        color = "visible"

    return {
        "target_slot_index": target_index,
        "color": color,
        "shape": shape,
        "description": f"{color} {shape}".strip(),
        "source": "env.unwrapped.config.objects[target_slot_index]",
    }


def _object_shape_name(obj: Any) -> str:
    if obj is None:
        return "object"
    class_name = type(obj).__name__.lower()
    if "cube" in class_name:
        return "cube"
    if "cylinder" in class_name:
        return "cylinder"
    if "sphere" in class_name or "ball" in class_name:
        return "sphere"
    shape = str(getattr(obj, "shape", "") or "").strip().lower()
    return shape or "object"


def _format_skill_task(skill_mode: str, target_object: dict[str, Any]) -> str:
    template = COLOR_SHAPE_SKILL_TASK_TEMPLATES[skill_mode]
    return template.format(
        color=target_object["color"],
        shape=target_object["shape"],
    )


def _take_unique_spawn_candidate(
    candidates: list[list[float]],
    *,
    next_index: int,
    bin_id: int,
    accepted: int,
    target: int,
) -> list[float]:
    if next_index >= len(candidates):
        raise RuntimeError(
            "spawn lookup exhausted before reaching the requested unique episode count: "
            f"bin={bin_id} accepted={accepted} target={target} candidates={len(candidates)}. "
            "Seed reuse is forbidden; generate additional unique lookup candidates."
        )
    return candidates[next_index]


def _parse_grid_balance_bins(raw: str) -> list[int]:
    if not raw.strip():
        return []
    bins = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        bins.append(int(text))
    return sorted(set(bins))


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one float")
    return values


def _workspace_positions_from_export_reports(
    report_paths: list[Path],
) -> list[tuple[float, float]]:
    positions: list[tuple[float, float]] = []
    for report_path in report_paths:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(
                f"workspace forbidden report has no episode list: {report_path}"
            )
        for episode_index, episode in enumerate(episodes):
            workspace_spawn = episode.get("workspace_spawn")
            world_xy = (
                None
                if not isinstance(workspace_spawn, dict)
                else workspace_spawn.get("world_xy_m")
            )
            if not isinstance(world_xy, list) or len(world_xy) != 2:
                raise ValueError(
                    "workspace forbidden report episode has no world_xy_m: "
                    f"{report_path} episode={episode_index}"
                )
            positions.append((float(world_xy[0]), float(world_xy[1])))
    return positions


def _set_target_object_xy(env: Any, xy: list[float] | tuple[float, float] | np.ndarray) -> None:
    import mujoco

    unwrapped = env.unwrapped
    slot = unwrapped._slots[int(unwrapped._target_slot_idx)]
    addr = int(slot.qpos_addr)
    unwrapped.data.qpos[addr : addr + 2] = np.asarray(xy, dtype=float)[:2]
    unwrapped.data.qpos[addr + 2] = float(slot.spawn_z)
    unwrapped.data.qvel[:] = 0.0
    mujoco.mj_forward(unwrapped.model, unwrapped.data)
    if hasattr(unwrapped, "_refresh_reset_reference_state"):
        unwrapped._refresh_reset_reference_state()


def _set_target_object_yaw(env: Any, yaw_degrees: float) -> None:
    """Set the active cube yaw without changing its settled XYZ position."""
    import mujoco

    unwrapped = env.unwrapped
    slot = unwrapped._slots[int(unwrapped._target_slot_idx)]
    addr = int(slot.qpos_addr)
    half_yaw = 0.5 * math.radians(float(yaw_degrees))
    unwrapped.data.qpos[addr + 3 : addr + 7] = np.asarray(
        [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=float
    )
    unwrapped.data.qvel[:] = 0.0
    mujoco.mj_forward(unwrapped.model, unwrapped.data)
    if hasattr(unwrapped, "_refresh_reset_reference_state"):
        unwrapped._refresh_reset_reference_state()


def _build_camera1_spawn_lookup(
    env: Any,
    renderers: dict[str, Any],
    *,
    grid_size: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: int,
    target_object_yaw_deg: float | None = None,
) -> dict[int, list[list[float]]]:
    resolution = max(2, int(resolution))
    lookup: dict[int, list[list[float]]] = {}
    snapshot = _snapshot_sim_state(env)
    try:
        for x in np.linspace(float(x_min), float(x_max), resolution):
            for y in np.linspace(float(y_min), float(y_max), resolution):
                _set_target_object_xy(env, [float(x), float(y)])
                if target_object_yaw_deg is not None:
                    _set_target_object_yaw(env, float(target_object_yaw_deg))
                visibility = _object_visibility_in_camera(env, renderers["egocentric_cam"], "egocentric_cam")
                centroid = visibility.get("normalized_centroid")
                if not visibility.get("visible") or centroid is None:
                    continue
                bx = min(grid_size - 1, max(0, int(float(centroid[0]) * grid_size)))
                by = min(grid_size - 1, max(0, int(float(centroid[1]) * grid_size)))
                lookup.setdefault(int(by * grid_size + bx), []).append([float(x), float(y)])
    finally:
        _restore_sim_state(env, snapshot)
    return lookup


def _camera1_spawn_lookup_cache_payload(
    lookup: dict[int, list[list[float]]],
    *,
    grid_size: int,
    resolution: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    target_object_yaw_deg: float | None = None,
) -> dict[str, Any]:
    return {
        "format": "so101_camera1_spawn_lookup_v1",
        "grid_size": int(grid_size),
        "resolution": int(resolution),
        "x_range": [float(value) for value in x_range],
        "y_range": [float(value) for value in y_range],
        "target_object_yaw_deg": (
            None if target_object_yaw_deg is None else float(target_object_yaw_deg)
        ),
        "lookup": {
            str(int(bin_id)): [[float(x), float(y)] for x, y in values]
            for bin_id, values in sorted(lookup.items())
        },
    }


def _validate_camera1_spawn_lookup_cache(
    payload: dict[str, Any],
    *,
    grid_size: int,
    resolution: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    target_object_yaw_deg: float | None = None,
) -> None:
    if payload.get("format") not in {
        "so101_camera1_spawn_lookup_v1",
        "so101_spawn_catalog_v1",
    }:
        raise ValueError("unsupported camera1 spawn lookup cache format")
    expected = {
        "grid_size": int(grid_size),
        "resolution": int(resolution),
        "x_range": [float(value) for value in x_range],
        "y_range": [float(value) for value in y_range],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"camera1 spawn lookup cache mismatch for {key}: {payload.get(key)!r} != {value!r}")
    payload_yaw = payload.get("target_object_yaw_deg")
    if target_object_yaw_deg is not None and (
        payload_yaw is None
        or not math.isclose(float(payload_yaw), float(target_object_yaw_deg), abs_tol=1e-9)
    ):
        raise ValueError(
            "camera1 spawn lookup cache mismatch for target_object_yaw_deg: "
            f"{payload_yaw!r} != {float(target_object_yaw_deg)!r}"
        )
    if not isinstance(payload.get("lookup"), dict):
        raise ValueError("camera1 spawn lookup cache has no lookup mapping")


def _filter_spawn_lookup_for_teacher_feasibility(
    env: Any,
    renderers: dict[str, Any],
    *,
    config: WristEgoServoConfig,
    spawn_lookup: dict[int, list[list[float]]],
    seed: int,
    move_target_z_offset: float,
    edge_contact_xy_success_threshold: float,
    edge_contact_parallel_success_threshold_deg: float,
    max_candidates_per_bin: int,
    allow_diagonal_fixed_jaw: bool,
    min_gripper_floor_clearance_m: float,
    close_steps: int,
    target_object_yaw_deg: float | None = None,
) -> dict[int, list[list[float]]]:
    filtered_lookup: dict[int, list[list[float]]] = {}
    max_candidates_per_bin = max(0, int(max_candidates_per_bin))
    for bin_id, candidates_xy in sorted(spawn_lookup.items()):
        accepted: list[list[float]] = []
        for index, xy in enumerate(candidates_xy):
            candidate_seed = int(seed) + int(bin_id) * 1000 + index
            env.reset(seed=candidate_seed)
            _set_target_object_xy(env, xy)
            if target_object_yaw_deg is not None:
                _set_target_object_yaw(env, float(target_object_yaw_deg))
            visible, _search_steps = sweep_until_visible(env, renderers, max_sweeps=config.max_sweeps)
            if not visible:
                continue
            candidates = _filter_fixed_jaw_move_candidates_in_policy_view(
                env,
                renderers=renderers,
                candidates=_make_fast_fixed_jaw_teacher_targets(
                    env,
                    allow_diagonal=allow_diagonal_fixed_jaw,
                    min_floor_clearance_m=float(min_gripper_floor_clearance_m),
                ),
                move_target_z_offset=move_target_z_offset,
            )
            if not _has_success_contract_fixed_jaw_candidate(
                env,
                candidates,
                edge_contact_xy_success_threshold=edge_contact_xy_success_threshold,
                edge_contact_parallel_success_threshold_deg=edge_contact_parallel_success_threshold_deg,
                min_gripper_floor_clearance_m=float(min_gripper_floor_clearance_m),
                close_steps=max(1, int(close_steps)),
            ):
                continue
            accepted.append([float(xy[0]), float(xy[1])])
            if max_candidates_per_bin and len(accepted) >= max_candidates_per_bin:
                break
        filtered_lookup[int(bin_id)] = accepted
        print(
            f"[so101-lerobot] teacher-feasible lookup bin={int(bin_id)} "
            f"{len(accepted)}/{len(candidates_xy)}",
            flush=True,
        )
    return filtered_lookup


def _has_success_contract_fixed_jaw_candidate(
    env: Any,
    candidates: list[dict[str, Any]],
    *,
    edge_contact_xy_success_threshold: float,
    edge_contact_parallel_success_threshold_deg: float,
    min_gripper_floor_clearance_m: float = 0.0,
    close_steps: int = 42,
) -> bool:
    snapshot = _snapshot_sim_state(env)
    try:
        for candidate in candidates:
            meta = dict(candidate.get("meta") or {})
            if _candidate_cube_normal_parallel_error_deg(meta) > float(edge_contact_parallel_success_threshold_deg):
                continue
            q_edge = _make_fixed_jaw_edge_qpos(env, np.asarray(candidate["q_open"], dtype=np.float32), meta)
            q_edge[-1] = _open_gripper_value(env)
            if float(min_gripper_floor_clearance_m) > 0.0:
                q_edge, floor_meta = _raise_edge_pose_for_floor_clearance(
                    env,
                    q_edge,
                    required_clearance_m=float(min_gripper_floor_clearance_m),
                    close_steps=max(1, int(close_steps)),
                )
                if not bool(floor_meta.get("passed_preflight", False)):
                    continue
            _set_qpos(env, q_edge)
            if _jaw_capture_geometry_passes(
                _jaw_cube_capture_geometry(env),
                max_centerline_error_m=float(edge_contact_xy_success_threshold),
            ):
                return True
    finally:
        _restore_sim_state(env, snapshot)
    return False


def _jaw_line_cube_face_normal_error_deg(
    jaw_axis_xy: np.ndarray | list[float] | tuple[float, ...],
    cube_face_normal_xy: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Return the unoriented angle between the jaw line and cube-face normal.

    The jaw line is the line joining the two finger pads.  The cube vector is
    the normal of the contacted face, translated so it passes through the
    cube center.  Lines are unoriented, so reversing either vector is still
    parallel and must produce the same (zero) error.
    """
    jaw = np.asarray(jaw_axis_xy, dtype=float).reshape(-1)[:2]
    normal = np.asarray(cube_face_normal_xy, dtype=float).reshape(-1)[:2]
    jaw_norm = float(np.linalg.norm(jaw))
    normal_norm = float(np.linalg.norm(normal))
    if jaw_norm <= 1e-8 or normal_norm <= 1e-8:
        return 180.0
    dot = abs(float(np.dot(jaw / jaw_norm, normal / normal_norm)))
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def _candidate_cube_normal_parallel_error_deg(meta: dict[str, Any]) -> float:
    """Read the authoritative geometry metric, with legacy-report fallback."""
    value = meta.get("cube_face_normal_parallel_error_deg")
    if value is None:
        value = meta.get("cube_centerline_parallel_error_deg")
    if value is None:
        value = meta.get("finger_axis_parallel_angle_deg", 180.0)
    return float(value)


def _cube_local_axis_to_world_xy(
    local_axis: np.ndarray | list[float] | tuple[float, ...],
    object_rotation: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    local = np.asarray(local_axis, dtype=float).reshape(3)
    rotation = np.asarray(object_rotation, dtype=float).reshape(3, 3)
    world = rotation @ local
    world[2] = 0.0
    norm = float(np.linalg.norm(world[:2]))
    if norm <= 1e-8:
        raise ValueError("cube face normal has no usable world-XY projection")
    return world / norm


def _spec_with_rotated_cube_face_normal(env: Any, spec: dict[str, Any]) -> dict[str, Any]:
    obj_geom_id = int(env.unwrapped._obj_geom_id)
    local_axis = np.asarray(spec["axis"], dtype=float).copy()
    world_axis = _cube_local_axis_to_world_xy(
        local_axis,
        np.asarray(env.unwrapped.data.geom_xmat[obj_geom_id], dtype=float),
    )
    rotated = dict(spec)
    rotated["cube_face_local_axis"] = local_axis
    rotated["axis"] = world_axis
    return rotated


def _current_jaw_cube_face_normal_error_deg(env: Any, meta: dict[str, Any]) -> float:
    import mujoco

    local_axis = meta.get("cube_face_local_axis")
    if local_axis is None:
        return 180.0
    model = env.unwrapped.model
    data = env.unwrapped.data
    obj_geom_id = int(env.unwrapped._obj_geom_id)
    static_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
    moving_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
    jaw_axis = np.asarray(data.geom_xpos[moving_pad] - data.geom_xpos[static_pad], dtype=float)
    try:
        face_normal = _cube_local_axis_to_world_xy(
            local_axis,
            np.asarray(data.geom_xmat[obj_geom_id], dtype=float),
        )
    except ValueError:
        return 180.0
    return _jaw_line_cube_face_normal_error_deg(jaw_axis, face_normal)


def _realized_robot_joint_qpos(env: Any) -> np.ndarray:
    """Read realized robot joints from MuJoCo, rather than actuator targets."""
    model = env.unwrapped.model
    data = env.unwrapped.data
    return np.asarray(
        [data.qpos[model.jnt_qposadr[joint_id]] for joint_id in env.unwrapped._joint_ids],
        dtype=np.float32,
    )


def _summary_start_grid_bin(summary: dict[str, Any], *, grid_size: int) -> int | None:
    visibility = (
        summary.get("start_policy_camera_visibility", {})
        .get("camera1", {})
    )
    centroid = visibility.get("normalized_centroid")
    if not visibility.get("visible") or centroid is None:
        return None
    x = min(grid_size - 1, max(0, int(float(centroid[0]) * grid_size)))
    y = min(grid_size - 1, max(0, int(float(centroid[1]) * grid_size)))
    return int(y * grid_size + x)


def _grid_balance_needs_teacher_candidate_for_start(
    *,
    skill_mode: str,
    episode_index: int,
    move_and_align_near_target_correction_ratio: float,
) -> bool:
    if skill_mode != "move_and_align_cube_edge":
        return skill_mode in {
            "align_fixed_jaw_cube_edge",
            "grip_from_edge_cube",
            *FULL_GRIP_SKILL_MODES,
        }
    ratio = float(np.clip(move_and_align_near_target_correction_ratio, 0.0, 1.0))
    if ratio <= 0.0:
        return False
    return ratio >= 1.0 or (int(episode_index) % max(1, int(round(1.0 / ratio)))) == 0


def _camera1_grid_bin_at_qpos(
    env: Any,
    renderers: dict[str, Any],
    *,
    qpos: np.ndarray,
    grid_size: int,
) -> int | None:
    snapshot = _snapshot_sim_state(env)
    try:
        _set_qpos(env, qpos)
        visibility = _object_visibility_in_camera(env, renderers["egocentric_cam"], "egocentric_cam")
    finally:
        _restore_sim_state(env, snapshot)
    centroid = visibility.get("normalized_centroid")
    if not visibility.get("visible") or centroid is None:
        return None
    x = min(grid_size - 1, max(0, int(float(centroid[0]) * grid_size)))
    y = min(grid_size - 1, max(0, int(float(centroid[1]) * grid_size)))
    return int(y * grid_size + x)


def _make_fast_fixed_jaw_teacher_targets(
    env: Any,
    *,
    allow_diagonal: bool = True,
    min_floor_clearance_m: float = 0.0,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    initial_snapshot = _snapshot_sim_state(env)
    base_specs = [
        spec
        for spec in _grasp_candidate_specs(env)
        if (allow_diagonal or not str(spec.get("mode", "")).startswith("diag_"))
        and (allow_diagonal or str(spec.get("grasp_mode")) in {"front", "overhead"})
        and (
            float(min_floor_clearance_m) <= 0.0
            or str(spec.get("mode", "")).endswith("overhead_biased")
        )
    ]
    # The jaw line is unoriented for the parallelism gate, but the physical
    # gripper is not symmetric: choosing the opposite contacted face swaps
    # which side carries the moving-jaw body. Generate both directions
    # constructively so the floor-clearance gate can select a feasible pose
    # instead of relying on low-probability post-export rejection.
    specs: list[dict[str, Any]] = []
    for base_spec in base_specs:
        height_offsets = (
            (0.0, 0.004, 0.008, 0.010, 0.012, 0.014)
            if float(min_floor_clearance_m) > 0.0
            else (0.0,)
        )
        for height_offset in height_offsets:
            for direction, suffix in ((1.0, "positive"), (-1.0, "negative")):
                spec = dict(base_spec)
                spec["axis"] = direction * np.asarray(base_spec["axis"], dtype=float)
                spec["z_offset"] = float(base_spec["z_offset"]) + float(height_offset)
                spec["floor_safe_contact_z_offset_m"] = float(height_offset)
                height_suffix = f"z{int(round(1000.0 * float(spec['z_offset']))):03d}"
                spec["mode"] = f"{base_spec['mode']}_{height_suffix}_{suffix}"
                spec["candidate_index"] = len(specs)
                spec["contact_direction"] = suffix
                specs.append(spec)
    try:
        for raw_spec in specs:
            # Every candidate must solve from the same episode state. Leaving
            # the previous IK result in MuJoCo makes candidate quality depend
            # on iteration order and can collapse cube-yaw diversity.
            _restore_sim_state(env, initial_snapshot)
            try:
                spec = _spec_with_rotated_cube_face_normal(env, raw_spec)
                q_open, solve_meta = _solve_fixed_jaw_edge_qpos_variant(
                    env,
                    spec,
                    min_floor_clearance_m=float(min_floor_clearance_m),
                )
            except Exception:
                continue
            meta = {
                "mode": str(spec["grasp_mode"]),
                "candidate_mode": str(spec["mode"]),
                "axis": [float(value) for value in np.asarray(spec["axis"], dtype=float)],
                "cube_face_local_axis": [
                    float(value) for value in np.asarray(spec["cube_face_local_axis"], dtype=float)
                ],
                "gap": float(spec["gap"]),
                "z_offset": float(spec["z_offset"]),
                "floor_safe_contact_z_offset_m": float(
                    spec.get("floor_safe_contact_z_offset_m", 0.0)
                ),
                "open_value": float(spec["open_value"]),
                "success_step": None,
                "score": (
                    -float(solve_meta["cost"])
                    - 0.12 * float(solve_meta.get("cube_face_normal_parallel_error_deg", 0.0))
                    - 2.0 * float(spec["z_offset"])
                    - 0.25 * float(spec["gap"])
                    - 0.0005 * float(spec["candidate_index"])
                ),
                "candidate_index": int(spec["candidate_index"]),
                "candidate_attempts": len(specs),
                "contact_direction": str(spec["contact_direction"]),
                "mode_successes": None,
                "fast_preview_candidate": True,
                "fast_preview_source": "fixed_jaw_edge_ik",
                "fixed_jaw_solver": True,
                **solve_meta,
            }
            candidates.append({"q_open": q_open.astype(float), "q_lift": q_open.astype(float), "meta": meta})
    finally:
        _restore_sim_state(env, initial_snapshot)
    return candidates


def _near_range_candidate_specs(env: Any) -> list[dict[str, Any]]:
    """Construct contact-centric candidates for cubes close to the arm base.

    The ordinary fixed-jaw solver constrains both pads while the gripper is
    fully open.  Near the base, the moving SO101 jaw sweeps substantially in
    Z while closing, so that open-pose constraint rejects otherwise valid
    folded-arm grasps.  Near-range candidates are therefore defined at the
    physical contact width and validated over the complete close sweep.
    """
    specs: list[dict[str, Any]] = []
    open_value = _open_gripper_value(env)
    for axis_name, local_axis in (
        ("front_back", np.asarray([1.0, 0.0, 0.0], dtype=float)),
        ("left_right", np.asarray([0.0, 1.0, 0.0], dtype=float)),
    ):
        for direction, direction_name in ((1.0, "positive"), (-1.0, "negative")):
            for z_offset in (0.004, 0.006, 0.008, 0.010):
                specs.append(
                    {
                        "candidate_index": len(specs),
                        "mode": (
                            f"near_contact_{axis_name}_"
                            f"z{int(round(1000.0 * z_offset)):03d}_{direction_name}"
                        ),
                        "grasp_mode": "near_contact",
                        "axis": direction * local_axis,
                        "gap": 0.095,
                        "z_offset": float(z_offset),
                        "open_value": float(open_value),
                        "contact_direction": direction_name,
                    }
                )
    return specs


def _make_near_range_fixed_jaw_teacher_targets(
    env: Any,
    *,
    min_floor_clearance_m: float = 0.0,
) -> list[dict[str, Any]]:
    """Return executable folded-arm grasp candidates for the near workspace."""
    candidates: list[dict[str, Any]] = []
    initial_snapshot = _snapshot_sim_state(env)
    specs = _near_range_candidate_specs(env)
    try:
        for raw_spec in specs:
            _restore_sim_state(env, initial_snapshot)
            try:
                spec = _spec_with_rotated_cube_face_normal(env, raw_spec)
                q_open, solve_meta = _solve_near_range_fixed_jaw_qpos_variant(
                    env,
                    spec,
                    min_floor_clearance_m=float(min_floor_clearance_m),
                )
            except Exception:
                continue
            parallel_error = float(
                solve_meta.get("cube_face_normal_parallel_error_deg", 180.0)
            )
            static_error = float(solve_meta.get("contact_static_target_error_m", 1.0))
            moving_error = float(solve_meta.get("contact_moving_target_error_m", 1.0))
            sweep_clearance = solve_meta.get("ik_close_sweep_floor_clearance_m")
            if parallel_error > 3.0 or max(static_error, moving_error) > 0.004:
                continue
            if (
                float(min_floor_clearance_m) > 0.0
                and (sweep_clearance is None or float(sweep_clearance) < float(min_floor_clearance_m))
            ):
                continue
            meta = {
                "mode": "near_contact",
                "candidate_mode": str(spec["mode"]),
                "axis": [float(value) for value in np.asarray(spec["axis"], dtype=float)],
                "cube_face_local_axis": [
                    float(value) for value in np.asarray(spec["cube_face_local_axis"], dtype=float)
                ],
                "gap": float(spec["gap"]),
                "z_offset": float(spec["z_offset"]),
                "open_value": float(spec["open_value"]),
                "success_step": None,
                "score": (
                    -float(solve_meta["cost"])
                    - 0.25 * parallel_error
                    - 40.0 * (static_error + moving_error)
                    - 0.001 * float(spec["candidate_index"])
                ),
                "candidate_index": int(spec["candidate_index"]),
                "candidate_attempts": len(specs),
                "contact_direction": str(spec["contact_direction"]),
                "mode_successes": None,
                "fast_preview_candidate": True,
                "fast_preview_source": "near_contact_folded_arm_ik",
                "fixed_jaw_solver": True,
                "solver_profile": "near_contact",
                "teacher_range": "near",
                **solve_meta,
            }
            candidates.append(
                {
                    "q_open": q_open.astype(float),
                    "q_lift": q_open.astype(float),
                    "meta": meta,
                }
            )
    finally:
        _restore_sim_state(env, initial_snapshot)
    return candidates


def _target_radius_from_shoulder_pan_axis(env: Any) -> float:
    """Measure target XY distance from the physical shoulder-pan axis."""
    import mujoco

    model = env.unwrapped.model
    data = env.unwrapped.data
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan"
    )
    if joint_id < 0:
        raise RuntimeError("SO101 model has no shoulder_pan joint")
    body_id = int(model.jnt_bodyid[joint_id])
    joint_world = (
        np.asarray(data.xpos[body_id], dtype=float)
        + np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        @ np.asarray(model.jnt_pos[joint_id], dtype=float)
    )
    target = np.asarray(env.unwrapped._get_target_pose(), dtype=float)
    return float(np.linalg.norm(target[:2] - joint_world[:2]))


def _continuous_teacher_solver_profiles(radius_m: float) -> tuple[str, ...]:
    """Return ordered IK families for a requested continuous-range grasp."""
    radius = float(radius_m)
    if radius < CONTINUOUS_TEACHER_MID_OVERLAP_MIN_M:
        return ("near_contact",)
    if radius > CONTINUOUS_TEACHER_NEAR_OVERLAP_MAX_M:
        return ("mid_fixed_jaw",)
    if radius <= 0.20:
        return ("near_contact", "mid_fixed_jaw")
    return ("mid_fixed_jaw", "near_contact")


def _make_continuous_range_fixed_jaw_teacher_targets(
    env: Any,
    *,
    min_floor_clearance_m: float = 0.0,
) -> list[dict[str, Any]]:
    """Build executable candidates continuously from near to maximum reach.

    The two existing IK formulations encode genuinely different gripper
    geometry. The bridge band therefore tries both and records which contract
    produced each candidate rather than pretending one formulation is valid
    everywhere.
    """
    radius_m = _target_radius_from_shoulder_pan_axis(env)
    profiles = _continuous_teacher_solver_profiles(radius_m)
    candidates: list[dict[str, Any]] = []
    for priority, profile in enumerate(profiles):
        if profile == "near_contact":
            generated = _make_near_range_fixed_jaw_teacher_targets(
                env,
                min_floor_clearance_m=float(min_floor_clearance_m),
            )
        elif profile == "mid_fixed_jaw":
            generated = _make_fast_fixed_jaw_teacher_targets(
                env,
                allow_diagonal=False,
                min_floor_clearance_m=float(min_floor_clearance_m),
            )
        else:  # pragma: no cover - profiles are defined above.
            raise ValueError(f"unknown continuous teacher solver profile: {profile}")
        for candidate in generated:
            row = dict(candidate)
            meta = dict(row.get("meta") or {})
            raw_score = float(meta.get("score", -1e9))
            meta.update(
                {
                    "solver_profile": profile,
                    "teacher_range": "continuous",
                    "target_radius_from_base_m": float(radius_m),
                    "continuous_solver_priority": int(priority),
                    "score_before_continuous_priority": raw_score,
                    # Candidate replay still decides success. This offset only
                    # avoids paying for the fallback family first.
                    "score": raw_score + 1000.0 * float(len(profiles) - priority),
                }
            )
            row["meta"] = meta
            candidates.append(row)
    return candidates


def _make_full_grip_teacher_targets_for_skill(
    env: Any,
    *,
    skill_mode: str,
    min_floor_clearance_m: float = 0.0,
) -> list[dict[str, Any]]:
    """Single factory shared by dataset export and workspace probes."""
    if skill_mode == "grip_the_cube_near_v1":
        return _make_near_range_fixed_jaw_teacher_targets(
            env,
            min_floor_clearance_m=float(min_floor_clearance_m),
        )
    if skill_mode == "grip_the_cube_continuous_v1":
        return _make_continuous_range_fixed_jaw_teacher_targets(
            env,
            min_floor_clearance_m=float(min_floor_clearance_m),
        )
    if skill_mode == "grip_the_cube_v1":
        return _make_fast_fixed_jaw_teacher_targets(
            env,
            allow_diagonal=False,
            min_floor_clearance_m=float(min_floor_clearance_m),
        )
    raise ValueError(f"not a full-grip teacher skill mode: {skill_mode}")


def _uses_near_contact_success_contract(
    skill_mode: str,
    best_meta: dict[str, Any],
) -> bool:
    if skill_mode == "grip_the_cube_near_v1":
        return True
    return bool(
        skill_mode == "grip_the_cube_continuous_v1"
        and str(best_meta.get("solver_profile")) == "near_contact"
    )


def _resolve_full_grip_trajectory_variant(
    *,
    skill_mode: str,
    requested_variant: str,
    best_meta: dict[str, Any],
) -> str:
    requested = str(requested_variant)
    if requested != "auto":
        return requested
    if skill_mode != "grip_the_cube_continuous_v1":
        raise ValueError("trajectory_variant=auto requires grip_the_cube_continuous_v1")
    if _uses_near_contact_success_contract(skill_mode, best_meta):
        return "direct_align"
    return "standard"


def _filter_fixed_jaw_move_candidates_in_policy_view(
    env: Any,
    *,
    renderers: dict[str, Any],
    candidates: list[dict[str, Any]],
    move_target_z_offset: float,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    snapshot = _snapshot_sim_state(env)
    try:
        for candidate in candidates:
            meta = dict(candidate["meta"])
            q_edge = _make_fixed_jaw_edge_qpos(env, np.asarray(candidate["q_open"], dtype=np.float32), meta)
            q_above = _make_fixed_jaw_above_qpos(env, q_edge, meta, move_target_z_offset=move_target_z_offset)
            q_above[-1] = _open_gripper_value(env)
            _set_qpos(env, q_above)
            visibility = _policy_camera_visibility(env, renderers)
            wrist = visibility["camera2"]
            if not bool(wrist["visible"]) or not bool(wrist["centered"]):
                continue
            center_distance = float(wrist["center_distance"] or 0.0)
            _set_qpos(env, q_edge)
            alignment = _camera2_top_contact_alignment(env, renderers, best_meta=meta)
            meta["preselected_policy_camera_visibility"] = visibility
            meta["camera2_top_contact_alignment"] = alignment
            alignment_error = alignment.get("image_alignment_error_deg")
            if alignment_error is None:
                alignment_penalty = 90.0
            else:
                alignment_penalty = float(alignment_error)
            meta["score"] = float(meta.get("score", 0.0)) - center_distance - 2.5 * alignment_penalty
            selected = dict(candidate)
            selected["meta"] = meta
            ranked.append((float(meta["score"]), selected))
    finally:
        _restore_sim_state(env, snapshot)
    return [candidate for _score, candidate in sorted(ranked, key=lambda item: item[0], reverse=True)]


def _camera2_top_contact_alignment(
    env: Any,
    renderers: dict[str, Any],
    *,
    best_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image = _render_camera(env, renderers["wrist_cam"], "wrist_cam")
    score = _image_alignment_score(image, edge_mode="top-contact")
    projected_jaw_angle = _projected_jaw_line_angle(
        env,
        camera_name="wrist_cam",
        width=int(image.shape[1]),
        height=int(image.shape[0]),
    )
    projected_contact_normal_angle = _projected_cube_contact_normal_angle(
        env,
        best_meta=best_meta,
        camera_name="wrist_cam",
        width=int(image.shape[1]),
        height=int(image.shape[0]),
    )
    cube_edge_angle = score.get("cube_top_contact_edge_angle_deg")
    alignment_error = score.get("image_alignment_error_deg")
    jaw_angle = score.get("jaw_angle_deg")
    if projected_jaw_angle is not None and projected_contact_normal_angle is not None:
        jaw_angle = float(projected_jaw_angle)
        alignment_error = float(_angle_diff(float(projected_contact_normal_angle), jaw_angle))
    return {
        "reason": score.get("reason"),
        "image_alignment_error_deg": alignment_error,
        "cube_top_contact_edge_angle_deg": cube_edge_angle,
        "cube_contact_normal_angle_deg": projected_contact_normal_angle,
        "jaw_angle_deg": jaw_angle,
        "mask_jaw_angle_deg": score.get("jaw_angle_deg"),
        "jaw_angle_source": "projected_finger_pad_centers" if projected_jaw_angle is not None else "rgb_mask_fallback",
        "contact_edge_distance_px": score.get("contact_edge_distance_px"),
    }


def _camera2_locked_top_contact_alignment(
    env: Any,
    renderers: dict[str, Any],
    *,
    reference_contact_normal_angle_deg: float | None,
) -> dict[str, Any]:
    if reference_contact_normal_angle_deg is None:
        return _camera2_top_contact_alignment(env, renderers)
    image = _render_camera(env, renderers["wrist_cam"], "wrist_cam")
    jaw_angle = _projected_jaw_line_angle(
        env,
        camera_name="wrist_cam",
        width=int(image.shape[1]),
        height=int(image.shape[0]),
    )
    if jaw_angle is None:
        return {
            "reason": "missing_projected_jaw_line_locked_top_contact",
            "image_alignment_error_deg": None,
            "cube_contact_normal_angle_deg": float(reference_contact_normal_angle_deg),
            "jaw_angle_deg": None,
            "contact_edge_distance_px": None,
            "locked_top_contact_edge": True,
        }
    return {
        "reason": "ok",
        "image_alignment_error_deg": float(
            _angle_diff(float(reference_contact_normal_angle_deg), float(jaw_angle))
        ),
        "cube_contact_normal_angle_deg": float(reference_contact_normal_angle_deg),
        "jaw_angle_deg": float(jaw_angle),
        "jaw_angle_source": "projected_finger_pad_centers",
        "contact_edge_distance_px": None,
        "locked_top_contact_edge": True,
    }


def _projected_jaw_line_angle(
    env: Any,
    *,
    camera_name: str,
    width: int,
    height: int,
) -> float | None:
    """Return the camera-image angle of the line connecting both finger pads."""
    import mujoco

    model = env.unwrapped.model
    data = env.unwrapped.data
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    static_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
    moving_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
    if min(camera_id, static_id, moving_id) < 0:
        return None
    points = [
        _project_world_point_to_camera(
            np.asarray(data.geom_xpos[geom_id], dtype=float),
            camera_position=np.asarray(data.cam_xpos[camera_id], dtype=float),
            camera_rotation=np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3),
            fovy_degrees=float(model.cam_fovy[camera_id]),
            width=width,
            height=height,
        )
        for geom_id in (static_id, moving_id)
    ]
    if any(point is None for point in points):
        return None
    static_xy, moving_xy = points
    delta = np.asarray(moving_xy, dtype=float) - np.asarray(static_xy, dtype=float)
    if float(np.linalg.norm(delta)) < 1e-6:
        return None
    return float((math.degrees(math.atan2(float(delta[1]), float(delta[0]))) + 180.0) % 180.0)


def _projected_cube_contact_normal_angle(
    env: Any,
    *,
    best_meta: dict[str, Any] | None,
    camera_name: str,
    width: int,
    height: int,
) -> float | None:
    import mujoco

    if not best_meta or best_meta.get("cube_face_normal_xy") is None:
        return None
    normal_xy = np.asarray(best_meta["cube_face_normal_xy"], dtype=float)
    if normal_xy.shape != (2,) or float(np.linalg.norm(normal_xy)) < 1e-8:
        return None
    normal_xy = normal_xy / float(np.linalg.norm(normal_xy))
    model = env.unwrapped.model
    data = env.unwrapped.data
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    object_geom_id = int(env.unwrapped._obj_geom_id)
    if camera_id < 0 or object_geom_id < 0:
        return None
    center = np.asarray(data.geom_xpos[object_geom_id], dtype=float)
    half_length = 0.03
    direction = np.asarray([normal_xy[0], normal_xy[1], 0.0], dtype=float)
    projected = [
        _project_world_point_to_camera(
            center + sign * half_length * direction,
            camera_position=np.asarray(data.cam_xpos[camera_id], dtype=float),
            camera_rotation=np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3),
            fovy_degrees=float(model.cam_fovy[camera_id]),
            width=width,
            height=height,
        )
        for sign in (-1.0, 1.0)
    ]
    if any(point is None for point in projected):
        return None
    start, end = projected
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    if float(np.linalg.norm(delta)) < 1e-6:
        return None
    return float((math.degrees(math.atan2(float(delta[1]), float(delta[0]))) + 180.0) % 180.0)


def _project_world_point_to_camera(
    point_world: np.ndarray,
    *,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    fovy_degrees: float,
    width: int,
    height: int,
) -> np.ndarray | None:
    relative = np.asarray(point_world, dtype=float) - np.asarray(camera_position, dtype=float)
    rotation = np.asarray(camera_rotation, dtype=float).reshape(3, 3)
    camera_x = float(np.dot(relative, rotation[:, 0]))
    camera_y = float(np.dot(relative, rotation[:, 1]))
    camera_z = float(np.dot(relative, -rotation[:, 2]))
    if camera_z <= 1e-6:
        return None
    focal = 0.5 * float(height) / math.tan(0.5 * math.radians(float(fovy_degrees)))
    return np.asarray(
        [
            0.5 * float(width) + focal * camera_x / camera_z,
            0.5 * float(height) - focal * camera_y / camera_z,
        ],
        dtype=float,
    )


def _solve_fixed_jaw_edge_qpos_variant(
    env: Any,
    spec: dict[str, Any],
    *,
    min_floor_clearance_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    import mujoco
    from scipy.optimize import least_squares

    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data
    joint_addrs = [model.jnt_qposadr[jid] for jid in unwrapped._joint_ids]
    low = np.asarray(env.action_space.low, dtype=float)
    high = np.asarray(env.action_space.high, dtype=float)
    static_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
    moving_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
    obj_geom_id = int(unwrapped._obj_geom_id)
    obj_pos = np.asarray(data.geom_xpos[obj_geom_id], dtype=float).copy()
    cube_half_extent = float(max(model.geom_size[obj_geom_id][0], model.geom_size[obj_geom_id][1]))
    q_seed = np.asarray([data.qpos[addr] for addr in joint_addrs], dtype=float)
    axis = np.asarray(spec["axis"], dtype=float)
    axis[2] = 0.0
    axis = axis / max(1e-6, float(np.linalg.norm(axis)))
    gap = float(spec["gap"])
    z_offset = float(spec["z_offset"])
    open_value = float(spec["open_value"])
    wrist_roll_target = spec.get("wrist_roll_target")
    desired_static = obj_pos - axis * (cube_half_extent + 0.002) + np.asarray([0.0, 0.0, z_offset])
    desired_moving = desired_static + axis * gap
    desired_contact_moving = (
        obj_pos
        + axis * (cube_half_extent + 0.002)
        + np.asarray([0.0, 0.0, z_offset])
    )
    desired_center = 0.5 * (desired_static + desired_moving)
    desired_contact_center = 0.5 * (desired_static + desired_contact_moving)
    desired_axis_xy = axis[:2] / max(1e-6, float(np.linalg.norm(axis[:2])))
    floor_clearance_geoms = (
        _gripper_floor_clearance_geoms(env)
        if float(min_floor_clearance_m) > 0.0
        else None
    )

    def set_qpos(qpos: np.ndarray) -> None:
        for addr, value in zip(joint_addrs, qpos):
            data.qpos[addr] = value
        data.ctrl[unwrapped._actuator_ids] = np.clip(qpos, low, high)
        mujoco.mj_forward(model, data)

    target_contact_span = float(
        np.linalg.norm(desired_contact_moving - desired_static)
    )
    contact_gripper_value = float(low[-1])
    contact_span_error = float("inf")
    for gripper_value in np.linspace(open_value, float(low[-1]), 65):
        probe_qpos = np.concatenate(
            [q_seed[:5], np.asarray([float(gripper_value)])]
        )
        set_qpos(probe_qpos)
        pad_span = float(
            np.linalg.norm(data.geom_xpos[moving_pad] - data.geom_xpos[static_pad])
        )
        error = abs(pad_span - target_contact_span)
        if error < contact_span_error:
            contact_span_error = error
            contact_gripper_value = float(gripper_value)

    fixed_wrist_roll = (
        None
        if wrist_roll_target is None
        else float(np.clip(float(wrist_roll_target), low[4], high[4]))
    )

    def expand_solver_qpos(solver_qpos: np.ndarray) -> np.ndarray:
        values = np.asarray(solver_qpos, dtype=float)
        if fixed_wrist_roll is None:
            return values
        return np.concatenate([values[:4], np.asarray([fixed_wrist_roll])])

    def residual(solver_qpos: np.ndarray) -> np.ndarray:
        arm_qpos = expand_solver_qpos(solver_qpos)
        qpos = np.concatenate([arm_qpos, np.asarray([open_value])])
        set_qpos(qpos)
        static_pos = np.asarray(data.geom_xpos[static_pad], dtype=float).copy()
        moving_pos = np.asarray(data.geom_xpos[moving_pad], dtype=float).copy()
        center = 0.5 * (static_pos + moving_pos)
        finger_axis = moving_pos - static_pos
        finger_axis_xy = finger_axis[:2] / max(1e-6, float(np.linalg.norm(finger_axis[:2])))
        parallel_error = finger_axis_xy - desired_axis_xy
        contact_qpos = np.concatenate(
            [arm_qpos, np.asarray([contact_gripper_value])]
        )
        set_qpos(contact_qpos)
        contact_static = np.asarray(data.geom_xpos[static_pad], dtype=float).copy()
        contact_moving = np.asarray(data.geom_xpos[moving_pad], dtype=float).copy()
        contact_center = 0.5 * (contact_static + contact_moving)
        contact_axis = contact_moving - contact_static
        contact_axis_xy = contact_axis[:2] / max(
            1e-6,
            float(np.linalg.norm(contact_axis[:2])),
        )
        contact_parallel_error = contact_axis_xy - desired_axis_xy
        terms = [
            (static_pos - desired_static) * 28.0,
            (moving_pos - desired_moving) * 10.0,
            (center - desired_center) * 6.0,
            parallel_error * 18.0,
            # The contacted-face normal lies in the table plane. Matching
            # only its XY projection admits a severely tilted jaw line,
            # which can put the moving-jaw body through the floor.
            np.asarray([finger_axis[2] * 48.0]),
            # The open-jaw geometry alone does not guarantee that the moving
            # pad crosses the cube's opposite face while closing. Constrain
            # the kinematic contact-width pose as well so arbitrary cube yaw
            # produces an executable close path rather than a visual-only
            # parallel pose.
            (contact_static - desired_static) * 28.0,
            (contact_moving - desired_contact_moving) * 28.0,
            (contact_center - desired_contact_center) * 8.0,
            contact_parallel_error * 22.0,
            np.asarray([contact_axis[2] * 36.0]),
            (arm_qpos - q_seed[:5]) * 0.025,
        ]
        if floor_clearance_geoms is not None:
            floor_geom, gripper_geoms = floor_clearance_geoms
            clearance_penalties = []
            # The SO101 moving jaw rotates. Its lowest point occurs midway
            # through closing, not necessarily at either endpoint, so both
            # endpoint-only checks can pass while the real close trajectory
            # clips the table.
            for gripper_qpos in np.linspace(open_value, float(low[-1]), 9):
                sweep_qpos = np.concatenate(
                    [arm_qpos, np.asarray([float(gripper_qpos)])]
                )
                set_qpos(sweep_qpos)
                for geom_id in gripper_geoms:
                    distance = float(
                        mujoco.mj_geomDistance(
                            model, data, floor_geom, geom_id, 1.0, None
                        )
                    )
                    deficit = max(
                        0.0,
                        float(min_floor_clearance_m) + 0.001 - distance,
                    )
                    clearance_penalties.append(deficit * 80.0)
            terms.append(np.asarray(clearance_penalties, dtype=float))
        return np.concatenate(terms)

    base_starts = [
        q_seed[:5],
        # Canonical floor-safe overhead postures found by the same constrained
        # solver. Keeping both mirrored variants as deterministic warm starts
        # avoids repeating a large global search for nearby spawn positions.
        np.asarray([-0.52, 0.90, -0.90, 1.658, -1.45]),
        np.asarray([0.52, 0.90, -0.90, 1.658, 1.45]),
        np.asarray([-0.5, 0.4, 0.1, 0.5, -1.3]),
        np.asarray([0.0, 0.55, -0.25, 0.85, 1.2]),
        np.asarray([0.6, 0.2, 0.2, 0.6, -1.0]),
        np.asarray([-0.8, 0.2, 0.2, 0.6, -1.0]),
        np.asarray([0.0, -0.15, 0.85, -0.75, 0.0]),
    ]
    roll_targets = [
        float(np.arctan2(axis[1], axis[0])),
        float(np.arctan2(axis[1], axis[0]) + np.pi / 2.0),
        float(np.arctan2(axis[1], axis[0]) - np.pi / 2.0),
        float(np.arctan2(axis[1], axis[0]) + np.pi),
    ]
    starts = list(base_starts[:3])
    for base in base_starts[:3]:
        for roll_target in roll_targets:
            candidate = np.asarray(base, dtype=float).copy()
            candidate[4] = roll_target
            starts.append(candidate)
    if fixed_wrist_roll is not None:
        targeted_seed = q_seed[:5].copy()
        targeted_seed[4] = fixed_wrist_roll
        starts.insert(0, targeted_seed)
        for start in starts:
            start[4] = fixed_wrist_roll
    if float(min_floor_clearance_m) > 0.0:
        from scipy.stats import qmc

        # A fixed low-discrepancy multistart covers the non-convex five-joint
        # IK space reproducibly. It is only enabled for the strict floor gate;
        # ordinary exports keep the cheaper legacy starts.
        sampler = qmc.Sobol(
            d=5,
            scramble=True,
            seed=10_000 + int(spec.get("candidate_index", 0)),
        )
        unit_starts = sampler.random_base2(m=3)
        sampled_starts = qmc.scale(unit_starts, low[:5], high[:5]).astype(float)
        if fixed_wrist_roll is not None:
            sampled_starts[:, 4] = fixed_wrist_roll
        starts.extend(sampled_starts)
    best: tuple[float, np.ndarray] | None = None
    for start in starts:
        solver_start = np.asarray(start, dtype=float)
        solver_low = low[:5]
        solver_high = high[:5]
        if fixed_wrist_roll is not None:
            solver_start = solver_start[:4]
            solver_low = low[:4]
            solver_high = high[:4]
        result = least_squares(
            residual,
            np.clip(solver_start, solver_low, solver_high),
            bounds=(solver_low, solver_high),
            max_nfev=(
                45
                if fixed_wrist_roll is not None
                else (80 if float(min_floor_clearance_m) > 0.0 else 35)
            ),
        )
        cost = float(np.linalg.norm(residual(result.x)))
        candidate = np.concatenate(
            [expand_solver_qpos(result.x), np.asarray([open_value])]
        )
        if best is None or cost < best[0]:
            best = (cost, candidate)
        if best is not None and best[0] < (0.35 if float(min_floor_clearance_m) > 0.0 else 1.25):
            break
    assert best is not None
    qpos = np.clip(best[1], low, high)
    set_qpos(qpos)
    static_delta = np.asarray(data.geom_xpos[static_pad] - data.geom_xpos[obj_geom_id], dtype=float)
    moving_delta = np.asarray(data.geom_xpos[moving_pad] - data.geom_xpos[obj_geom_id], dtype=float)
    finger_axis = np.asarray(data.geom_xpos[moving_pad] - data.geom_xpos[static_pad], dtype=float)
    finger_axis_xy = finger_axis[:2] / max(1e-6, float(np.linalg.norm(finger_axis[:2])))
    jaw_vertical_angle_deg = float(
        np.degrees(
            np.arctan2(
                abs(float(finger_axis[2])),
                max(1e-8, float(np.linalg.norm(finger_axis[:2]))),
            )
        )
    )
    axis_parallel_dot = float(np.clip(np.dot(finger_axis_xy, desired_axis_xy), -1.0, 1.0))
    cube_face_normal_parallel_error_deg = _jaw_line_cube_face_normal_error_deg(
        finger_axis_xy,
        desired_axis_xy,
    )
    target_delta = desired_static - obj_pos
    contact_qpos = qpos.copy()
    contact_qpos[-1] = contact_gripper_value
    set_qpos(contact_qpos)
    contact_static_delta = np.asarray(
        data.geom_xpos[static_pad] - data.geom_xpos[obj_geom_id],
        dtype=float,
    )
    contact_moving_delta = np.asarray(
        data.geom_xpos[moving_pad] - data.geom_xpos[obj_geom_id],
        dtype=float,
    )
    contact_axis = np.asarray(
        data.geom_xpos[moving_pad] - data.geom_xpos[static_pad],
        dtype=float,
    )
    contact_axis_xy = contact_axis[:2] / max(
        1e-6,
        float(np.linalg.norm(contact_axis[:2])),
    )
    contact_axis_parallel_error_deg = _jaw_line_cube_face_normal_error_deg(
        contact_axis_xy,
        desired_axis_xy,
    )
    set_qpos(qpos)
    open_floor_clearance_m = None
    closed_floor_clearance_m = None
    sweep_floor_clearance_m = None
    if floor_clearance_geoms is not None:
        open_floor_clearance_m, _open_floor_geom = _minimum_gripper_floor_clearance(
            env, floor_clearance_geoms
        )
        qpos_closed = qpos.copy()
        qpos_closed[-1] = float(low[-1])
        set_qpos(qpos_closed)
        closed_floor_clearance_m, _closed_floor_geom = _minimum_gripper_floor_clearance(
            env, floor_clearance_geoms
        )
        sweep_floor_clearance_m = float("inf")
        for gripper_qpos in np.linspace(open_value, float(low[-1]), 33):
            qpos_sweep = qpos.copy()
            qpos_sweep[-1] = float(gripper_qpos)
            set_qpos(qpos_sweep)
            clearance, _clearance_geom = _minimum_gripper_floor_clearance(
                env, floor_clearance_geoms
            )
            sweep_floor_clearance_m = min(
                float(sweep_floor_clearance_m), float(clearance)
            )
        set_qpos(qpos)
    return qpos.astype(np.float32), {
        "cost": float(best[0]),
        "static_edge_xy_error": float(np.linalg.norm((static_delta - target_delta)[:2])),
        "finger_axis_parallel_dot": axis_parallel_dot,
        # Authoritative acceptance metric: jaw line || contacted-face normal
        # through the cube center. Keep the old key as a compatibility alias.
        "cube_face_normal_parallel_error_deg": cube_face_normal_parallel_error_deg,
        "cube_centerline_parallel_error_deg": cube_face_normal_parallel_error_deg,
        "finger_axis_parallel_angle_deg": cube_face_normal_parallel_error_deg,
        "parallel_geometry_contract": "jaw_line_vs_contact_face_normal_through_cube_center",
        "cube_face_normal_xy": [float(value) for value in desired_axis_xy],
        "jaw_line_xy": [float(value) for value in finger_axis_xy],
        "static_delta_x": float(static_delta[0]),
        "static_delta_y": float(static_delta[1]),
        "static_delta_z": float(static_delta[2]),
        "moving_delta_x": float(moving_delta[0]),
        "moving_delta_y": float(moving_delta[1]),
        "moving_delta_z": float(moving_delta[2]),
        "finger_axis_x": float(finger_axis[0]),
        "finger_axis_y": float(finger_axis[1]),
        "finger_axis_z": float(finger_axis[2]),
        "jaw_vertical_angle_deg": jaw_vertical_angle_deg,
        "target_delta_x": float(target_delta[0]),
        "target_delta_y": float(target_delta[1]),
        "target_delta_z": float(target_delta[2]),
        "contact_gripper_value": float(contact_gripper_value),
        "contact_target_span_m": float(target_contact_span),
        "contact_span_sampling_error_m": float(contact_span_error),
        "contact_static_target_error_m": float(
            np.linalg.norm(contact_static_delta - target_delta)
        ),
        "contact_moving_target_error_m": float(
            np.linalg.norm(
                contact_moving_delta - (desired_contact_moving - obj_pos)
            )
        ),
        "contact_axis_parallel_error_deg": float(
            contact_axis_parallel_error_deg
        ),
        "ik_open_floor_clearance_m": open_floor_clearance_m,
        "ik_closed_floor_clearance_m": closed_floor_clearance_m,
        "ik_close_sweep_floor_clearance_m": sweep_floor_clearance_m,
    }


def _solve_near_range_fixed_jaw_qpos_variant(
    env: Any,
    spec: dict[str, Any],
    *,
    min_floor_clearance_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a folded-arm grasp from the geometry at jaw contact.

    Unlike the mid-range solver, the moving pad is not forced onto an
    artificial target while the jaw is fully open.  Its full open-to-close
    sweep is still checked for floor clearance before a candidate is exposed
    to the episode writer.
    """
    import mujoco
    from scipy.optimize import least_squares

    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data
    joint_addrs = [model.jnt_qposadr[jid] for jid in unwrapped._joint_ids]
    low = np.asarray(env.action_space.low, dtype=float)
    high = np.asarray(env.action_space.high, dtype=float)
    static_pad = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad"
    )
    moving_pad = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad"
    )
    obj_geom_id = int(unwrapped._obj_geom_id)
    obj_pos = np.asarray(data.geom_xpos[obj_geom_id], dtype=float).copy()
    cube_half_extent = float(
        max(model.geom_size[obj_geom_id][0], model.geom_size[obj_geom_id][1])
    )
    q_seed = np.asarray([data.qpos[addr] for addr in joint_addrs], dtype=float)
    axis = np.asarray(spec["axis"], dtype=float)
    axis[2] = 0.0
    axis = axis / max(1e-8, float(np.linalg.norm(axis)))
    desired_axis_xy = axis[:2].copy()
    z_offset = float(spec["z_offset"])
    open_value = float(spec["open_value"])
    contact_margin_m = 0.002
    desired_static = (
        obj_pos
        - axis * (cube_half_extent + contact_margin_m)
        + np.asarray([0.0, 0.0, z_offset])
    )
    desired_moving = (
        obj_pos
        + axis * (cube_half_extent + contact_margin_m)
        + np.asarray([0.0, 0.0, z_offset])
    )
    desired_center = 0.5 * (desired_static + desired_moving)

    def set_qpos(qpos: np.ndarray) -> None:
        clipped = np.clip(np.asarray(qpos, dtype=float), low, high)
        for addr, value in zip(joint_addrs, clipped):
            data.qpos[addr] = value
        data.ctrl[unwrapped._actuator_ids] = clipped
        mujoco.mj_forward(model, data)

    target_contact_span = float(np.linalg.norm(desired_moving - desired_static))
    contact_gripper_value = float(low[-1])
    contact_span_error = float("inf")
    for gripper_value in np.linspace(open_value, float(low[-1]), 97):
        probe = np.concatenate([q_seed[:5], np.asarray([float(gripper_value)])])
        set_qpos(probe)
        span = float(
            np.linalg.norm(data.geom_xpos[moving_pad] - data.geom_xpos[static_pad])
        )
        error = abs(span - target_contact_span)
        if error < contact_span_error:
            contact_span_error = error
            contact_gripper_value = float(gripper_value)

    floor_clearance_geoms = (
        _gripper_floor_clearance_geoms(env)
        if float(min_floor_clearance_m) > 0.0
        else None
    )

    def residual(arm_qpos: np.ndarray) -> np.ndarray:
        contact_qpos = np.concatenate(
            [np.asarray(arm_qpos, dtype=float), np.asarray([contact_gripper_value])]
        )
        set_qpos(contact_qpos)
        static_pos = np.asarray(data.geom_xpos[static_pad], dtype=float).copy()
        moving_pos = np.asarray(data.geom_xpos[moving_pad], dtype=float).copy()
        center = 0.5 * (static_pos + moving_pos)
        jaw_axis = moving_pos - static_pos
        jaw_axis_xy = jaw_axis[:2] / max(
            1e-8, float(np.linalg.norm(jaw_axis[:2]))
        )
        terms = [
            (static_pos - desired_static) * 36.0,
            (moving_pos - desired_moving) * 36.0,
            (center - desired_center) * 12.0,
            (jaw_axis_xy - desired_axis_xy) * 24.0,
            np.asarray([float(jaw_axis[2]) * 48.0]),
            (np.asarray(arm_qpos, dtype=float) - q_seed[:5]) * 0.003,
        ]
        if floor_clearance_geoms is not None:
            floor_geom, gripper_geoms = floor_clearance_geoms
            clearance_penalties: list[float] = []
            for gripper_qpos in np.linspace(open_value, float(low[-1]), 7):
                sweep_qpos = np.concatenate(
                    [np.asarray(arm_qpos, dtype=float), np.asarray([float(gripper_qpos)])]
                )
                set_qpos(sweep_qpos)
                for geom_id in gripper_geoms:
                    distance = float(
                        mujoco.mj_geomDistance(
                            model, data, floor_geom, geom_id, 1.0, None
                        )
                    )
                    clearance_penalties.append(
                        max(
                            0.0,
                            float(min_floor_clearance_m) + 0.0005 - distance,
                        )
                        * 120.0
                    )
            terms.append(np.asarray(clearance_penalties, dtype=float))
        return np.concatenate(terms)

    first_joint_id = int(unwrapped._joint_ids[0])
    base_body_id = int(model.jnt_bodyid[first_joint_id])
    base_xy = np.asarray(data.xpos[base_body_id][:2], dtype=float)
    cube_bearing = float(
        math.atan2(float(obj_pos[1] - base_xy[1]), float(obj_pos[0] - base_xy[0]))
    )
    desired_roll = float(math.atan2(float(axis[1]), float(axis[0])))

    def wrapped(value: float) -> float:
        return float((value + math.pi) % (2.0 * math.pi) - math.pi)

    folded_profiles = (
        (-0.86, 1.18, 1.24),
        (-0.70, 1.08, 1.20),
        (-0.55, 0.95, 1.18),
        (-0.35, 0.75, 1.20),
        (-0.15, 0.55, 1.25),
    )
    # Seed the two folded branches that were independently verified at the
    # 10 cm front-boundary probe before adding the broader interpolated set.
    starts: list[np.ndarray] = [
        np.asarray(
            [
                cube_bearing + 0.052,
                -0.6965,
                1.0827,
                1.1873,
                wrapped(desired_roll + 0.106),
            ],
            dtype=float,
        ),
        np.asarray(
            [
                cube_bearing - 0.062,
                -0.8481,
                1.1808,
                1.2351,
                wrapped(desired_roll - 0.010),
            ],
            dtype=float,
        ),
    ]
    for profile_index, (lift, elbow, wrist_flex) in enumerate(folded_profiles):
        roll = desired_roll + (math.pi if profile_index % 2 else 0.0)
        starts.append(
            np.asarray(
                [cube_bearing, lift, elbow, wrist_flex, wrapped(roll)],
                dtype=float,
            )
        )
    for roll_offset in (0.0, math.pi):
        seed_start = q_seed[:5].copy()
        seed_start[0] = cube_bearing
        seed_start[4] = wrapped(desired_roll + roll_offset)
        starts.append(seed_start)

    best: tuple[float, np.ndarray] | None = None
    for start in starts:
        result = least_squares(
            residual,
            np.clip(start, low[:5], high[:5]),
            bounds=(low[:5], high[:5]),
            max_nfev=110,
        )
        cost = float(np.linalg.norm(residual(result.x)))
        if best is None or cost < best[0]:
            best = (cost, np.asarray(result.x, dtype=float).copy())
        if best[0] < 0.12:
            break
    if best is None:
        raise RuntimeError("near-range contact IK produced no candidate")

    arm_qpos = np.clip(best[1], low[:5], high[:5])
    contact_qpos = np.concatenate(
        [arm_qpos, np.asarray([contact_gripper_value])]
    )
    set_qpos(contact_qpos)
    contact_static = np.asarray(data.geom_xpos[static_pad], dtype=float).copy()
    contact_moving = np.asarray(data.geom_xpos[moving_pad], dtype=float).copy()
    contact_axis = contact_moving - contact_static
    contact_axis_xy = contact_axis[:2] / max(
        1e-8, float(np.linalg.norm(contact_axis[:2]))
    )
    parallel_error_deg = _jaw_line_cube_face_normal_error_deg(
        contact_axis_xy, desired_axis_xy
    )
    static_error_m = float(np.linalg.norm(contact_static - desired_static))
    moving_error_m = float(np.linalg.norm(contact_moving - desired_moving))

    open_qpos = np.concatenate([arm_qpos, np.asarray([open_value])])
    set_qpos(open_qpos)
    open_static = np.asarray(data.geom_xpos[static_pad], dtype=float).copy()
    open_moving = np.asarray(data.geom_xpos[moving_pad], dtype=float).copy()
    open_axis = open_moving - open_static
    open_axis_xy = open_axis[:2] / max(1e-8, float(np.linalg.norm(open_axis[:2])))
    open_floor_clearance_m = None
    closed_floor_clearance_m = None
    sweep_floor_clearance_m = None
    if floor_clearance_geoms is not None:
        open_floor_clearance_m, _ = _minimum_gripper_floor_clearance(
            env, floor_clearance_geoms
        )
        sweep_floor_clearance_m = float("inf")
        for gripper_qpos in np.linspace(open_value, float(low[-1]), 33):
            sweep_qpos = np.concatenate(
                [arm_qpos, np.asarray([float(gripper_qpos)])]
            )
            set_qpos(sweep_qpos)
            clearance, _ = _minimum_gripper_floor_clearance(
                env, floor_clearance_geoms
            )
            sweep_floor_clearance_m = min(
                float(sweep_floor_clearance_m), float(clearance)
            )
        closed_qpos = np.concatenate([arm_qpos, np.asarray([float(low[-1])])])
        set_qpos(closed_qpos)
        closed_floor_clearance_m, _ = _minimum_gripper_floor_clearance(
            env, floor_clearance_geoms
        )
    set_qpos(open_qpos)

    return open_qpos.astype(np.float32), {
        "cost": float(best[0]),
        "solver_profile": "near_contact",
        "solver_constraint_pose": "contact_width",
        "near_solver_warm_start_count": len(starts),
        "contact_margin_m": float(contact_margin_m),
        "contact_gripper_value": float(contact_gripper_value),
        "contact_target_span_m": float(target_contact_span),
        "contact_span_sampling_error_m": float(contact_span_error),
        "contact_static_target_error_m": static_error_m,
        "contact_moving_target_error_m": moving_error_m,
        "cube_face_normal_parallel_error_deg": float(parallel_error_deg),
        "cube_centerline_parallel_error_deg": float(parallel_error_deg),
        "finger_axis_parallel_angle_deg": float(parallel_error_deg),
        "parallel_geometry_contract": (
            "jaw_line_vs_contact_face_normal_through_cube_center"
        ),
        "cube_face_normal_xy": [float(value) for value in desired_axis_xy],
        "jaw_line_xy": [float(value) for value in open_axis_xy],
        "contact_jaw_line_xy": [float(value) for value in contact_axis_xy],
        "jaw_vertical_angle_deg": float(
            np.degrees(
                np.arctan2(
                    abs(float(contact_axis[2])),
                    max(1e-8, float(np.linalg.norm(contact_axis[:2]))),
                )
            )
        ),
        "static_delta_x": float(open_static[0] - obj_pos[0]),
        "static_delta_y": float(open_static[1] - obj_pos[1]),
        "static_delta_z": float(open_static[2] - obj_pos[2]),
        "moving_delta_x": float(open_moving[0] - obj_pos[0]),
        "moving_delta_y": float(open_moving[1] - obj_pos[1]),
        "moving_delta_z": float(open_moving[2] - obj_pos[2]),
        "target_delta_x": float(desired_static[0] - obj_pos[0]),
        "target_delta_y": float(desired_static[1] - obj_pos[1]),
        "target_delta_z": float(desired_static[2] - obj_pos[2]),
        "ik_open_floor_clearance_m": open_floor_clearance_m,
        "ik_closed_floor_clearance_m": closed_floor_clearance_m,
        "ik_close_sweep_floor_clearance_m": sweep_floor_clearance_m,
    }


def _write_legacy_teacher_episode(
    *,
    dataset: Any,
    env: Any,
    renderers: dict[str, Any],
    q_open: np.ndarray,
    q_lift: np.ndarray,
    seed: int,
    search_steps: int,
    teacher_visible: bool,
    best_meta: dict[str, Any],
    task: str,
    include_camera3_duplicate: bool,
) -> dict[str, Any]:
    q_close = q_open.copy()
    q_close[-1] = float(env.action_space.low[-1])
    info: dict[str, Any] = {}
    frames = 0
    success_step = None
    for step in range(180):
        if step < 58:
            action = q_open.copy()
        elif step < 118:
            action = q_close.copy()
        else:
            action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=np.float32)
            action[-1] = q_close[-1]
            q_lift = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
        dataset.add_frame(
            _make_lerobot_frame(
                env=env,
                renderers=renderers,
                action=np.asarray(action, dtype=np.float32),
                task=task,
                include_camera3_duplicate=include_camera3_duplicate,
            )
        )
        frames += 1
        _obs, _reward, terminated, truncated, info = env.step(np.asarray(action, dtype=float))
        if bool(info.get("success", False)) and success_step is None:
            success_step = step + 1
        if bool(info.get("success", False)) or terminated or truncated:
            break
    return {
        "seed": seed,
        "frames": frames,
        "success": bool(info.get("success", False)),
        "success_step": success_step,
        "search_steps": search_steps,
        "teacher_visible_in_any_camera": bool(teacher_visible),
        "best_meta": best_meta,
        "final_info": {
            "is_grasped": bool(info.get("is_grasped", False)),
            "lift_height": float(info.get("lift_height", 0.0)),
            "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        },
        "q_lift": [float(value) for value in q_lift],
        "teacher_style": "legacy",
        "phase_counts": {"approach": 58, "close": 60, "lift": max(0, frames - 118)},
    }


def _write_staged_teacher_episode(
    *,
    dataset: Any,
    env: Any,
    renderers: dict[str, Any],
    q_open: np.ndarray,
    seed: int,
    search_steps: int,
    teacher_visible: bool,
    best_meta: dict[str, Any],
    approach_steps: int,
    settle_steps: int,
    close_steps: int,
    lift_steps: int,
    start_mode: str,
    near_gripper_joint_std: float,
    task: str,
    include_camera3_duplicate: bool,
) -> dict[str, Any]:
    q_start = _current_qpos(env).astype(np.float32)
    q_start[-1] = _open_gripper_value(env)
    q_open = np.clip(q_open.astype(np.float32), env.action_space.low, env.action_space.high)
    q_open[-1] = _open_gripper_value(env)
    if start_mode == "near-gripper":
        q_start = _make_near_gripper_qpos(
            env,
            q_open,
            seed=seed,
            joint_std=near_gripper_joint_std,
        )
        _set_qpos(env, q_start)
    elif start_mode != "home":
        raise ValueError(f"unknown start_mode: {start_mode}")
    q_close = q_open.copy()
    q_close[-1] = float(env.action_space.low[-1])
    info: dict[str, Any] = env.unwrapped._get_info()
    frames = 0
    success_step = None
    q_lift = q_close.copy()
    phase_counts = {"approach": 0, "settle": 0, "close": 0, "lift": 0}
    action_deltas: list[float] = []
    previous_action: np.ndarray | None = None

    def add_step(action: np.ndarray, phase: str) -> bool:
        nonlocal frames, info, success_step, previous_action
        action = np.clip(np.asarray(action, dtype=np.float32), env.action_space.low, env.action_space.high)
        dataset.add_frame(
            _make_lerobot_frame(
                env=env,
                renderers=renderers,
                action=action,
                task=task,
                include_camera3_duplicate=include_camera3_duplicate,
            )
        )
        frames += 1
        phase_counts[phase] += 1
        if previous_action is not None:
            action_deltas.append(float(np.linalg.norm(action[:5] - previous_action[:5])))
        previous_action = action.copy()
        _obs, _reward, terminated, truncated, info = env.step(np.asarray(action, dtype=float))
        if bool(info.get("success", False)) and success_step is None:
            success_step = frames
        return bool(terminated) or bool(truncated)

    approach_steps = max(1, int(approach_steps))
    for index in range(approach_steps):
        alpha = (index + 1) / float(approach_steps)
        alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
        action = (1.0 - alpha) * q_start + alpha * q_open
        action[-1] = _open_gripper_value(env)
        if add_step(action, "approach"):
            break

    if not bool(info.get("success", False)):
        for _ in range(max(0, int(settle_steps))):
            if add_step(q_open, "settle"):
                break

    if not bool(info.get("success", False)):
        for _ in range(max(1, int(close_steps))):
            if add_step(q_close, "close"):
                break

    if not bool(info.get("success", False)):
        for _ in range(max(1, int(lift_steps))):
            action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=np.float32)
            action[-1] = q_close[-1]
            q_lift = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            if add_step(q_lift, "lift"):
                break

    return {
        "seed": seed,
        "frames": frames,
        "success": bool(info.get("success", False)),
        "success_step": success_step,
        "search_steps": search_steps,
        "teacher_visible_in_any_camera": bool(teacher_visible),
        "best_meta": best_meta,
        "final_info": {
            "is_grasped": bool(info.get("is_grasped", False)),
            "lift_height": float(info.get("lift_height", 0.0)),
            "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        },
        "q_start": [float(value) for value in q_start],
        "q_open": [float(value) for value in q_open],
        "q_lift": [float(value) for value in q_lift],
        "teacher_style": "staged",
        "start_mode": start_mode,
        "phase_counts": phase_counts,
        "mean_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "max_action_delta": float(np.max(action_deltas)) if action_deltas else 0.0,
    }


def _write_move_over_cube_episode(
    *,
    dataset: Any,
    env: Any,
    renderers: dict[str, Any],
    q_open: np.ndarray,
    seed: int,
    search_steps: int,
    teacher_visible: bool,
    best_meta: dict[str, Any],
    approach_steps: int,
    settle_steps: int,
    episode_index: int,
    random_start_joint_std: float,
    move_success_tcp_dist: float,
    move_target_z_offset: float,
    closed_gripper_prob: float,
    move_gripper_profile: str,
    move_min_actual_z: float,
    task: str,
    include_camera3_duplicate: bool,
) -> dict[str, Any]:
    q_open = np.clip(q_open.astype(np.float32), env.action_space.low, env.action_space.high)
    q_open[-1] = _open_gripper_value(env)
    q_above = _offset_qpos_by_cartesian(env, q_open, np.asarray([0.0, 0.0, float(move_target_z_offset)]))
    rng = np.random.default_rng(int(seed) + 4242)
    gripper_value = _sample_move_gripper_value(
        env,
        rng=rng,
        episode_index=episode_index,
        profile=move_gripper_profile,
        closed_gripper_prob=closed_gripper_prob,
    )
    q_above[-1] = gripper_value
    q_above_delta = _tcp_to_object_delta_for_qpos(env, q_above)
    q_above_z_offset = float(q_above_delta[2])
    if q_above_z_offset < float(move_min_actual_z):
        return {
            "seed": seed,
            "frames": 0,
            "success": False,
            "success_step": None,
            "search_steps": search_steps,
            "teacher_visible_in_any_camera": bool(teacher_visible),
            "best_meta": best_meta,
            "final_info": dict(env.unwrapped._get_info()),
            "q_start": [float(value) for value in q_above],
            "q_above": [float(value) for value in q_above],
            "q_open": [float(value) for value in q_open],
            "q_above_tcp_to_obj_delta": [float(value) for value in q_above_delta],
            "q_above_z_offset": q_above_z_offset,
            "move_min_actual_z": float(move_min_actual_z),
            "gripper_value": float(gripper_value),
            "gripper_closed": bool(gripper_value <= float(env.action_space.low[-1]) + 1e-5),
            "gripper_profile": str(move_gripper_profile),
            "gripper_bucket": None,
            "teacher_style": "staged_skill",
            "skill_mode": "move_over_cube",
            "phase_counts": {"move": 0, "settle": 0},
            "mean_action_delta": 0.0,
            "max_action_delta": 0.0,
        }
    q_start = _make_random_start_qpos(env, q_above, seed=seed, joint_std=random_start_joint_std)
    q_start[-1] = gripper_value
    _set_qpos(env, q_start)
    info: dict[str, Any] = env.unwrapped._get_info()
    frames = 0
    phase_counts = {"move": 0, "settle": 0}
    action_deltas: list[float] = []
    previous_action: np.ndarray | None = None

    def add_step(action: np.ndarray, phase: str) -> None:
        nonlocal frames, info, previous_action
        action = np.clip(np.asarray(action, dtype=np.float32), env.action_space.low, env.action_space.high)
        action[-1] = gripper_value
        dataset.add_frame(
            _make_lerobot_frame(
                env=env,
                renderers=renderers,
                action=action,
                task=task,
                include_camera3_duplicate=include_camera3_duplicate,
            )
        )
        frames += 1
        phase_counts[phase] += 1
        if previous_action is not None:
            action_deltas.append(float(np.linalg.norm(action[:5] - previous_action[:5])))
        previous_action = action.copy()
        _obs, _reward, _terminated, _truncated, info = env.step(np.asarray(action, dtype=float))

    approach_steps = max(1, int(approach_steps))
    for index in range(approach_steps):
        alpha = (index + 1) / float(approach_steps)
        alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
        add_step((1.0 - alpha) * q_start + alpha * q_above, "move")
    for _ in range(max(0, int(settle_steps))):
        add_step(q_above, "settle")

    tcp_to_obj_dist = float(info.get("tcp_to_obj_dist", 1.0))
    final_tcp_to_obj_delta = _tcp_to_object_delta(env)
    final_z_offset = float(final_tcp_to_obj_delta[2])
    success = tcp_to_obj_dist <= float(move_success_tcp_dist) and final_z_offset >= float(move_min_actual_z)
    return {
        "seed": seed,
        "frames": frames,
        "success": success,
        "success_step": frames if success else None,
        "search_steps": search_steps,
        "teacher_visible_in_any_camera": bool(teacher_visible),
        "best_meta": best_meta,
        "final_info": {
            "is_grasped": bool(info.get("is_grasped", False)),
            "lift_height": float(info.get("lift_height", 0.0)),
            "tcp_to_obj_dist": tcp_to_obj_dist,
        },
        "q_start": [float(value) for value in q_start],
        "q_above": [float(value) for value in q_above],
        "q_open": [float(value) for value in q_open],
        "q_above_tcp_to_obj_delta": [float(value) for value in q_above_delta],
        "q_above_z_offset": q_above_z_offset,
        "final_tcp_to_obj_delta": [float(value) for value in final_tcp_to_obj_delta],
        "final_z_offset": final_z_offset,
        "move_min_actual_z": float(move_min_actual_z),
        "gripper_value": float(gripper_value),
        "gripper_closed": bool(gripper_value <= float(env.action_space.low[-1]) + 1e-5),
        "gripper_profile": str(move_gripper_profile),
        "gripper_bucket": int(episode_index % 5) if move_gripper_profile == "balanced" else None,
        "teacher_style": "staged_skill",
        "skill_mode": "move_over_cube",
        "phase_counts": phase_counts,
        "mean_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "max_action_delta": float(np.max(action_deltas)) if action_deltas else 0.0,
    }


def _write_pick_from_top_cube_episode(
    *,
    dataset: Any,
    env: Any,
    renderers: dict[str, Any],
    q_open: np.ndarray,
    seed: int,
    search_steps: int,
    teacher_visible: bool,
    best_meta: dict[str, Any],
    close_steps: int,
    lift_steps: int,
    episode_index: int,
    move_target_z_offset: float,
    pick_start_joint_std: float,
    pick_correction_steps: int,
    pick_start_min_abs_y: float,
    pick_start_max_abs_y: float,
    pick_start_min_actual_abs_y: float,
    pick_start_min_actual_z: float,
    task: str,
    include_camera3_duplicate: bool,
) -> dict[str, Any]:
    q_open = np.clip(q_open.astype(np.float32), env.action_space.low, env.action_space.high)
    q_open[-1] = _open_gripper_value(env)
    q_above = _offset_qpos_by_cartesian(env, q_open, np.asarray([0.0, 0.0, float(move_target_z_offset)]))
    q_start = _make_near_gripper_qpos(env, q_above, seed=seed + 313, joint_std=pick_start_joint_std)
    q_start, start_target_y_offset = _balance_pick_start_y_offset(
        env,
        q_start,
        episode_index=episode_index,
        min_abs_y=pick_start_min_abs_y,
        max_abs_y=pick_start_max_abs_y,
    )
    q_start[-1] = float(env.action_space.low[-1])
    _set_qpos(env, q_start)
    start_tcp_to_obj_delta = _tcp_to_object_delta(env)
    start_abs_y_offset = float(abs(start_tcp_to_obj_delta[1]))
    start_z_offset = float(start_tcp_to_obj_delta[2])
    if start_abs_y_offset < float(pick_start_min_actual_abs_y) or start_z_offset < float(pick_start_min_actual_z):
        return {
            "seed": seed,
            "frames": 0,
            "success": False,
            "success_step": None,
            "search_steps": search_steps,
            "teacher_visible_in_any_camera": bool(teacher_visible),
            "best_meta": best_meta,
            "final_info": dict(env.unwrapped._get_info()),
            "q_start": [float(value) for value in q_start],
            "q_above": [float(value) for value in q_above],
            "q_open": [float(value) for value in q_open],
            "q_lift": [float(value) for value in q_open],
            "start_target_y_offset": float(start_target_y_offset),
            "start_tcp_to_obj_delta": [float(value) for value in start_tcp_to_obj_delta],
            "start_abs_y_offset": start_abs_y_offset,
            "start_z_offset": start_z_offset,
            "pick_start_min_actual_abs_y": float(pick_start_min_actual_abs_y),
            "pick_start_min_actual_z": float(pick_start_min_actual_z),
            "teacher_style": "staged_skill",
            "skill_mode": "pick_from_top_cube",
            "phase_counts": {"correct": 0, "close": 0, "lift": 0},
            "mean_action_delta": 0.0,
            "max_action_delta": 0.0,
        }
    q_close = q_open.copy()
    q_close[-1] = float(env.action_space.low[-1])
    info: dict[str, Any] = env.unwrapped._get_info()
    frames = 0
    success_step = None
    q_lift = q_close.copy()
    phase_counts = {"correct": 0, "close": 0, "lift": 0}
    action_deltas: list[float] = []
    previous_action: np.ndarray | None = None

    def add_step(action: np.ndarray, phase: str) -> bool:
        nonlocal frames, info, success_step, previous_action
        action = np.clip(np.asarray(action, dtype=np.float32), env.action_space.low, env.action_space.high)
        dataset.add_frame(
            _make_lerobot_frame(
                env=env,
                renderers=renderers,
                action=action,
                task=task,
                include_camera3_duplicate=include_camera3_duplicate,
            )
        )
        frames += 1
        phase_counts[phase] += 1
        if previous_action is not None:
            action_deltas.append(float(np.linalg.norm(action[:5] - previous_action[:5])))
        previous_action = action.copy()
        _obs, _reward, terminated, truncated, info = env.step(np.asarray(action, dtype=float))
        if bool(info.get("success", False)) and success_step is None:
            success_step = frames
        return bool(info.get("success", False)) or bool(terminated) or bool(truncated)

    correction_steps = max(0, int(pick_correction_steps))
    for index in range(correction_steps):
        alpha = (index + 1) / float(max(1, correction_steps))
        alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
        action = (1.0 - alpha) * q_start + alpha * q_open
        action[-1] = _open_gripper_value(env)
        if add_step(action, "correct"):
            break

    if not bool(info.get("success", False)):
        for _ in range(max(1, int(close_steps))):
            if add_step(q_close, "close"):
                break
    if not bool(info.get("success", False)):
        for _ in range(max(1, int(lift_steps))):
            action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=np.float32)
            action[-1] = q_close[-1]
            q_lift = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            if add_step(q_lift, "lift"):
                break

    return {
        "seed": seed,
        "frames": frames,
        "success": bool(info.get("success", False)),
        "success_step": success_step,
        "search_steps": search_steps,
        "teacher_visible_in_any_camera": bool(teacher_visible),
        "best_meta": best_meta,
        "final_info": {
            "is_grasped": bool(info.get("is_grasped", False)),
            "lift_height": float(info.get("lift_height", 0.0)),
            "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        },
        "q_start": [float(value) for value in q_start],
        "q_above": [float(value) for value in q_above],
        "q_open": [float(value) for value in q_open],
        "q_lift": [float(value) for value in q_lift],
        "start_target_y_offset": float(start_target_y_offset),
        "start_tcp_to_obj_delta": [float(value) for value in start_tcp_to_obj_delta],
        "start_abs_y_offset": start_abs_y_offset,
        "start_z_offset": start_z_offset,
        "pick_start_min_actual_abs_y": float(pick_start_min_actual_abs_y),
        "pick_start_min_actual_z": float(pick_start_min_actual_z),
        "teacher_style": "staged_skill",
        "skill_mode": "pick_from_top_cube",
        "phase_counts": phase_counts,
        "mean_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "max_action_delta": float(np.max(action_deltas)) if action_deltas else 0.0,
    }


def _fixed_jaw_lift_target_reached(info: dict[str, Any], *, target_height: float) -> bool:
    return bool(
        info.get("is_grasped", False)
        and float(info.get("lift_height", 0.0)) >= float(target_height)
    )


def _fixed_jaw_pick_success(
    info: dict[str, Any],
    *,
    lift_target_reached: bool,
    terminal_min_height: float,
) -> bool:
    return bool(
        info.get("is_grasped", False)
        and float(info.get("lift_height", 0.0)) >= float(terminal_min_height)
        and lift_target_reached
    )


def _fixed_jaw_terminal_event_stops_episode(
    phase: str,
    *,
    terminated: bool,
    truncated: bool,
) -> bool:
    if truncated:
        return True
    return bool(terminated) and phase not in {"lift", "terminal_hold"}


def _grip_the_cube_correction_phases(
    *,
    q_start: np.ndarray,
    q_above: np.ndarray,
    q_edge: np.ndarray,
    q_close: np.ndarray,
    approach_steps: int,
    settle_steps: int,
    close_steps: int,
    lift_steps: int,
) -> list[tuple[str, np.ndarray, np.ndarray | None, int]]:
    """Build a local correction path around the executable grasp prepose."""
    return [
        ("near_target_correct", q_start, q_above, max(1, int(approach_steps))),
        ("gripper_descend", q_above, q_edge, max(1, int(approach_steps))),
        ("settle_aligned", q_edge, q_edge, max(0, int(settle_steps))),
        ("close", q_edge, q_close, max(1, int(close_steps))),
        ("lift", q_close, None, max(1, int(lift_steps))),
    ]


def _retain_visible_correction_start(
    env: Any,
    renderers: dict[str, Any],
    *,
    q_start: np.ndarray,
    q_reference: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Keep the largest deterministic perturbation visible to either policy camera."""
    snapshot = _snapshot_sim_state(env)
    try:
        start = np.asarray(q_start, dtype=np.float32)
        reference = np.asarray(q_reference, dtype=np.float32)
        for scale in _correction_visibility_scales():
            candidate = np.clip(
                reference + float(scale) * (start - reference),
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)
            candidate[-1] = _open_gripper_value(env)
            _set_qpos(env, candidate)
            visibility = _policy_camera_visibility(env, renderers)
            if bool(visibility["camera1"]["visible"] or visibility["camera2"]["visible"]):
                return candidate, float(scale)
        return reference.copy(), 0.0
    finally:
        _restore_sim_state(env, snapshot)


def _correction_visibility_scales() -> tuple[float, ...]:
    """Try the mirrored local correction before reducing its magnitude."""
    return (1.0, -1.0, 0.75, -0.75, 0.5, -0.5, 0.25, -0.25, 0.0)


def _write_fixed_jaw_edge_episode(
    *,
    dataset: Any,
    env: Any,
    renderers: dict[str, Any],
    q_open: np.ndarray,
    seed: int,
    search_steps: int,
    teacher_visible: bool,
    best_meta: dict[str, Any],
    skill_mode: str,
    approach_steps: int,
    settle_steps: int,
    close_steps: int,
    close_alignment_gate_mode: str,
    close_alignment_limits: dict[str, float] | None,
    trajectory_variant: str,
    grip_the_cube_start_profile: str,
    lift_steps: int,
    lift_target_height: float,
    lift_success_height: float,
    lift_controller_z_error: float,
    episode_index: int,
    random_start_joint_std: float,
    move_target_z_offset: float,
    terminal_hold_steps: int,
    move_and_align_near_target_correction_ratio: float,
    edge_contact_xy_success_threshold: float,
    edge_contact_parallel_success_threshold_deg: float,
    near_target_joint_std: float,
    near_target_xy_std: float,
    above_edge_start_joint_std: float,
    above_edge_start_xy_std: float,
    above_edge_start_z_std: float,
    above_edge_start_min_actual_z: float,
    above_edge_trajectory_variants: str,
    above_edge_start_gripper_profile: str,
    above_edge_terminal_hold_jitter: int,
    task: str,
    include_camera3_duplicate: bool,
    capture_render_replay: bool = False,
    capture_fps: int = 12,
    reset_home_qpos: np.ndarray | None,
    exact_start_pose: bool,
    min_gripper_floor_clearance_m: float,
    record_dataset_frames: bool = True,
) -> dict[str, Any]:
    uses_near_contact_contract = _uses_near_contact_success_contract(
        skill_mode,
        best_meta,
    )
    q_edge = _make_fixed_jaw_edge_qpos(env, q_open, best_meta)
    q_edge, floor_pre_refine_meta = _raise_edge_pose_for_floor_clearance(
        env,
        q_edge,
        required_clearance_m=float(min_gripper_floor_clearance_m),
        close_steps=max(1, int(close_steps)),
    )
    if (
        float(min_gripper_floor_clearance_m) > 0.0
        and not bool(floor_pre_refine_meta.get("passed_preflight", False))
    ):
        best_meta = dict(best_meta)
        best_meta["floor_clearance_constructive_refine"] = {
            "before_camera2_refine": floor_pre_refine_meta,
            "after_camera2_refine": None,
        }
        samples = list(floor_pre_refine_meta.get("samples", []))
        best_sample = max(
            samples,
            key=lambda row: float(row.get("minimum_clearance_m", float("-inf"))),
            default={},
        )
        return {
            "seed": int(seed),
            "success": False,
            "reason": "gripper_floor_clearance_preflight_failed",
            "frames": 0,
            "best_meta": best_meta,
            "gripper_floor_clearance_gate": {
                "required_min_clearance_m": float(min_gripper_floor_clearance_m),
                "minimum_clearance_m": best_sample.get("minimum_clearance_m"),
                "minimum_frame_index": None,
                "minimum_geom": best_sample.get("minimum_geom"),
                "passed": False,
                "stage": "dynamic_close_preflight_before_camera2",
            },
        }
    if close_alignment_gate_mode == "geometry_only":
        # The fixed-jaw IK already solved the authoritative geometry contract.
        # Do not let a perspective/occlusion-sensitive camera2 refinement
        # rotate that solution away from the cube-face normal before closing.
        close_stable_refine_meta = {
            "reason": "geometry_authoritative_no_image_refine",
            "camera2_role": "diagnostic_only",
        }
    else:
        q_edge, close_stable_refine_meta = _refine_close_stable_fixed_jaw_qpos_for_camera2_top_contact(
            env,
            renderers,
            q_edge=q_edge,
            best_meta=best_meta,
            close_steps=max(1, int(close_steps)),
            close_alignment_gate_mode=close_alignment_gate_mode,
            close_alignment_limits=close_alignment_limits,
            min_floor_clearance_m=float(min_gripper_floor_clearance_m),
        )
    q_edge, floor_post_refine_meta = _raise_edge_pose_for_floor_clearance(
        env,
        q_edge,
        required_clearance_m=float(min_gripper_floor_clearance_m),
        close_steps=max(1, int(close_steps)),
    )
    if close_alignment_gate_mode != "geometry_only":
        close_stable_refine_meta = dict(close_stable_refine_meta)
        close_stable_refine_meta["close_probe"] = _probe_grip_the_cube_v1_close_trace_gate(
            env,
            renderers,
            q_edge=q_edge,
            best_meta=best_meta,
            close_steps=max(1, int(close_steps)),
            close_alignment_gate_mode=close_alignment_gate_mode,
            close_alignment_limits=close_alignment_limits,
            min_floor_clearance_m=float(min_gripper_floor_clearance_m),
        )
    best_meta = dict(best_meta)
    best_meta["floor_clearance_constructive_refine"] = {
        "before_camera2_refine": floor_pre_refine_meta,
        "after_camera2_refine": floor_post_refine_meta,
    }
    best_meta["camera2_top_contact_close_stable_refine"] = close_stable_refine_meta
    best_meta["camera2_top_contact_roll_refine"] = close_stable_refine_meta.get("roll_refine", {})
    if (
        float(min_gripper_floor_clearance_m) > 0.0
        and not bool(floor_post_refine_meta.get("passed_preflight", False))
    ):
        samples = list(floor_post_refine_meta.get("samples", []))
        best_sample = max(
            samples,
            key=lambda row: float(row.get("minimum_clearance_m", float("-inf"))),
            default={},
        )
        return {
            "seed": int(seed),
            "success": False,
            "reason": "gripper_floor_clearance_preflight_failed",
            "frames": 0,
            "best_meta": best_meta,
            "gripper_floor_clearance_gate": {
                "required_min_clearance_m": float(min_gripper_floor_clearance_m),
                "minimum_clearance_m": best_sample.get("minimum_clearance_m"),
                "minimum_frame_index": None,
                "minimum_geom": best_sample.get("minimum_geom"),
                "passed": False,
                "stage": "dynamic_close_preflight",
            },
        }
    if not _close_probe_allows_full_episode(
        mode=close_alignment_gate_mode,
        refine_meta=close_stable_refine_meta,
    ):
        probe = close_stable_refine_meta.get("close_probe", {})
        best_meta["camera2_top_contact_close_alignment_gate"] = probe.get("gate", {})
        return {
            "seed": int(seed),
            "success": False,
            "reason": "camera2_close_probe_failed",
            "frames": 0,
            "best_meta": best_meta,
        }
    q_above = _make_fixed_jaw_above_qpos(env, q_edge, best_meta, move_target_z_offset=move_target_z_offset)
    q_above = q_above.copy()
    q_above[4] = q_edge[4]
    q_edge[-1] = _open_gripper_value(env)
    q_above[-1] = _open_gripper_value(env)

    if skill_mode == "move_over_cube_edge":
        q_start = _make_home_closed_start_qpos(env, reset_home_qpos)
        q_above = q_above.copy()
        q_above[-1] = float(env.action_space.low[-1])
        phases = [("move", q_start, q_above, max(1, int(approach_steps))), ("settle", q_above, q_above, max(0, int(settle_steps)))]
        success_kind = "edge_above"
    elif skill_mode == "align_fixed_jaw_cube_edge":
        q_above = q_above.copy()
        q_above[-1] = float(env.action_space.low[-1])
        q_start = q_above.copy()
        q_edge = q_edge.copy()
        q_edge[-1] = _open_gripper_value(env)
        phases = [("align", q_start, q_edge, max(1, int(approach_steps))), ("settle", q_edge, q_edge, max(0, int(settle_steps)))]
        success_kind = "edge_contact"
    elif skill_mode == "move_and_align_cube_edge":
        near_target_ratio = float(np.clip(move_and_align_near_target_correction_ratio, 0.0, 1.0))
        use_near_target = near_target_ratio > 0.0 and (
            near_target_ratio >= 1.0 or (int(episode_index) % max(1, int(round(1.0 / near_target_ratio)))) == 0
        )
        if use_near_target:
            q_start = _make_near_target_fixed_jaw_correction_qpos(
                env,
                q_edge=q_edge,
                seed=seed,
                episode_index=episode_index,
                joint_std=near_target_joint_std,
                xy_std=near_target_xy_std,
            )
            trajectory_variant = "near_target_correction"
        else:
            q_start = _make_home_closed_start_qpos(env, reset_home_qpos)
            trajectory_variant = "generated_teacher"
        q_edge = q_edge.copy()
        q_edge[-1] = _open_gripper_value(env)
        phase_steps = max(1, int(approach_steps)) if use_near_target else max(1, int(approach_steps)) * 2
        phases = [
            ("move_align", q_start, q_edge, phase_steps),
            ("settle", q_edge, q_edge, max(0, int(settle_steps))),
        ]
        success_kind = "edge_contact_parallel"
    elif skill_mode == "grip_from_edge_cube":
        q_start = q_edge.copy()
        q_close = q_edge.copy()
        q_close[-1] = float(env.action_space.low[-1])
        phases = [
            ("settle", q_start, q_start, max(0, int(settle_steps))),
            ("close", q_start, q_close, max(1, int(close_steps))),
            ("lift", q_close, None, max(1, int(lift_steps))),
        ]
        success_kind = "pick_success"
    elif skill_mode == "grip_from_above_edge_cube":
        variants = _parse_above_edge_variants(above_edge_trajectory_variants)
        selected_variant = variants[int(episode_index) % len(variants)]
        q_start, above_edge_start_meta = _make_above_edge_perturbed_start_qpos(
            env,
            q_above=q_above,
            seed=seed,
            episode_index=episode_index,
            joint_std=above_edge_start_joint_std,
            xy_std=above_edge_start_xy_std,
            z_std=above_edge_start_z_std,
            min_actual_z=above_edge_start_min_actual_z,
        )
        open_phase_gripper_value = _above_edge_open_phase_gripper_value(
            env,
            episode_index=episode_index,
            profile=above_edge_start_gripper_profile,
        )
        q_start[-1] = open_phase_gripper_value
        q_edge = q_edge.copy()
        q_edge[-1] = open_phase_gripper_value
        q_close = q_edge.copy()
        q_close[-1] = float(env.action_space.low[-1])
        if selected_variant == "two_stage_xy_z":
            q_mid = q_start.copy()
            q_mid[:2] = q_edge[:2]
            q_mid[4] = q_edge[4]
            phases = [
                ("xy_roll_correct", q_start, q_mid, max(1, int(approach_steps) // 2)),
                ("descend_align", q_mid, q_edge, max(1, int(approach_steps) - max(1, int(approach_steps) // 2))),
                ("settle", q_edge, q_edge, max(0, int(settle_steps))),
                ("close", q_edge, q_close, max(1, int(close_steps))),
                ("lift", q_close, None, max(1, int(lift_steps))),
            ]
        elif selected_variant == "roll_first":
            q_mid = q_start.copy()
            q_mid[4] = q_edge[4]
            phases = [
                ("roll_correct", q_start, q_mid, max(1, int(approach_steps) // 3)),
                ("descend_align", q_mid, q_edge, max(1, int(approach_steps) - max(1, int(approach_steps) // 3))),
                ("settle", q_edge, q_edge, max(0, int(settle_steps))),
                ("close", q_edge, q_close, max(1, int(close_steps))),
                ("lift", q_close, None, max(1, int(lift_steps))),
            ]
        elif selected_variant == "near_miss_correction":
            q_start = _make_near_target_fixed_jaw_correction_qpos(
                env,
                q_edge=q_edge,
                seed=seed,
                episode_index=episode_index,
                joint_std=near_target_joint_std,
                xy_std=near_target_xy_std,
            )
            q_start = _offset_qpos_by_cartesian(
                env,
                q_start,
                np.asarray([0.0, 0.0, max(0.0, float(above_edge_start_min_actual_z))], dtype=float),
                steps=8,
            )
            q_start[-1] = open_phase_gripper_value
            phases = [
                ("near_miss_correct", q_start, q_edge, max(1, int(approach_steps))),
                ("settle", q_edge, q_edge, max(0, int(settle_steps))),
                ("close", q_edge, q_close, max(1, int(close_steps))),
                ("lift", q_close, None, max(1, int(lift_steps))),
            ]
        else:
            phases = [
                ("descend_align", q_start, q_edge, max(1, int(approach_steps))),
                ("settle", q_edge, q_edge, max(0, int(settle_steps))),
                ("close", q_edge, q_close, max(1, int(close_steps))),
                ("lift", q_close, None, max(1, int(lift_steps))),
            ]
        success_kind = "pick_success"
        trajectory_variant = f"above_edge_{selected_variant}"
    elif skill_mode in FULL_GRIP_SKILL_MODES:
        requested_path_variant = _resolve_full_grip_trajectory_variant(
            skill_mode=skill_mode,
            requested_variant=str(trajectory_variant),
            best_meta=best_meta,
        )
        if requested_path_variant not in {"standard", "roll_first", "direct_align"}:
            raise ValueError(
                f"unknown {skill_mode} trajectory variant: {requested_path_variant}"
            )
        home_start = _current_qpos(env).astype(np.float32) if reset_home_qpos is None else np.asarray(reset_home_qpos, dtype=np.float32).copy()
        home_start = np.clip(home_start, env.action_space.low, env.action_space.high).astype(np.float32)
        hardware_start = home_start.copy()
        if not exact_start_pose:
            home_start[-1] = _open_gripper_value(env)
        q_edge = q_edge.copy()
        q_edge[-1] = _open_gripper_value(env)
        q_above = q_above.copy()
        q_above[-1] = _open_gripper_value(env)
        q_above_misaligned = _make_roll_misaligned_fixed_jaw_qpos(
            env,
            q_edge=q_edge,
            q_above=q_above,
            seed=seed,
            episode_index=episode_index,
        )
        q_close = q_edge.copy()
        q_close[-1] = float(env.action_space.low[-1])
        best_meta = dict(best_meta)
        best_meta["camera2_top_contact_close_roll_refine"] = {
            "reason": "skipped",
            "replacement": "per_close_step_camera2_top_contact_wrist_roll_refine",
            "initial_wrist_roll": float(q_close[4]),
        }
        use_correction_start = grip_the_cube_start_profile == "correction"
        use_home_start = grip_the_cube_start_profile == "home" or (
            grip_the_cube_start_profile == "mixed" and int(seed) % 2 == 0
        )
        if use_correction_start:
            q_start = _make_near_target_fixed_jaw_correction_qpos(
                env,
                q_edge=q_above,
                seed=seed,
                episode_index=episode_index,
                joint_std=near_target_joint_std,
                xy_std=near_target_xy_std,
            )
            q_start[-1] = _open_gripper_value(env)
            q_start, correction_visibility_scale = _retain_visible_correction_start(
                env,
                renderers,
                q_start=q_start,
                q_reference=q_above,
            )
            start_variant = "near_target_correction"
            move_steps = max(1, int(approach_steps))
        elif use_home_start:
            q_start = home_start
            start_variant = "home_start"
            move_steps = max(1, int(approach_steps))
        else:
            rng = np.random.default_rng(int(seed) + 61001)
            q_mid = (0.55 * home_start + 0.45 * q_above_misaligned).astype(np.float32)
            jitter = rng.normal(0.0, max(0.0, float(near_target_joint_std)), size=q_mid.shape).astype(np.float32)
            jitter[-1] = 0.0
            q_start = np.clip(q_mid + jitter, env.action_space.low, env.action_space.high).astype(np.float32)
            q_start[-1] = _open_gripper_value(env)
            start_variant = "mid_start"
            move_steps = max(1, int(approach_steps))
        roll_align_steps = max(1, int(approach_steps) // 2)
        descend_steps = max(1, int(approach_steps) // 2)
        if use_correction_start:
            phases = _grip_the_cube_correction_phases(
                q_start=q_start,
                q_above=q_above,
                q_edge=q_edge,
                q_close=q_close,
                approach_steps=move_steps,
                settle_steps=settle_steps,
                close_steps=close_steps,
                lift_steps=lift_steps,
            )
        elif requested_path_variant == "roll_first":
            q_roll_first = q_start.copy()
            q_roll_first[4] = q_above[4]
            q_roll_first[-1] = _open_gripper_value(env)
            phases = [
                ("roll_align_first", q_start, q_roll_first, roll_align_steps),
                ("move_to_cube", q_roll_first, q_above, move_steps),
                ("gripper_descend", q_above, q_edge, descend_steps),
                ("settle_aligned", q_edge, q_edge, max(0, int(settle_steps))),
                ("close", q_edge, q_close, max(1, int(close_steps))),
                ("lift", q_close, None, max(1, int(lift_steps))),
            ]
        elif requested_path_variant == "direct_align":
            phases = [
                ("move_and_align", q_start, q_above, move_steps + roll_align_steps),
                ("gripper_descend", q_above, q_edge, descend_steps),
                ("settle_aligned", q_edge, q_edge, max(0, int(settle_steps))),
                ("close", q_edge, q_close, max(1, int(close_steps))),
                ("lift", q_close, None, max(1, int(lift_steps))),
            ]
        else:
            phases = [
                ("move_to_cube", q_start, q_above_misaligned, move_steps),
                ("roll_align_with_cube_edge", q_above_misaligned, q_above, roll_align_steps),
                ("gripper_descend", q_above, q_edge, descend_steps),
                ("settle_aligned", q_edge, q_edge, max(0, int(settle_steps))),
                ("close", q_edge, q_close, max(1, int(close_steps))),
                ("lift", q_close, None, max(1, int(lift_steps))),
            ]
        if exact_start_pose:
            opened_start = hardware_start.copy()
            opened_start[-1] = _open_gripper_value(env)
            q_start = hardware_start
            phases.insert(
                0,
                (
                    "open_from_hardware_start",
                    hardware_start,
                    opened_start,
                    max(1, int(settle_steps)),
                ),
            )
            first_motion_phase = 1
            phase_name, _old_start, phase_target, phase_steps = phases[first_motion_phase]
            phases[first_motion_phase] = (
                phase_name,
                opened_start,
                phase_target,
                phase_steps,
            )
            start_variant = "exact_hardware_start"
        trajectory_variant = f"{skill_mode}_{start_variant}_{requested_path_variant}"
        success_kind = "pick_success"
    else:
        raise ValueError(f"unknown fixed jaw skill mode: {skill_mode}")
    effective_terminal_hold_steps = int(terminal_hold_steps)
    if skill_mode == "grip_from_above_edge_cube" and int(above_edge_terminal_hold_jitter) > 0:
        span = int(above_edge_terminal_hold_jitter)
        offsets = list(range(-span, span + 1))
        effective_terminal_hold_steps = max(0, int(terminal_hold_steps) + offsets[int(episode_index) % len(offsets)])
    if int(effective_terminal_hold_steps) > 0 and skill_mode != "grip_from_edge_cube":
        phases.append(("terminal_hold", None, None, int(effective_terminal_hold_steps)))
    if skill_mode in FULL_GRIP_SKILL_MODES:
        phases, wrist_safe_phase_changes = _make_grip_the_cube_wrist_safe_phases(
            phases
        )
        best_meta = dict(best_meta)
        best_meta["wrist_roll_safe_phase_changes"] = wrist_safe_phase_changes
    if "trajectory_variant" not in locals():
        trajectory_variant = "generated_teacher"

    _set_qpos(env, q_start)
    if skill_mode == "grip_from_above_edge_cube" and float(above_edge_start_min_actual_z) > 0.0:
        start_delta = _tcp_to_object_delta(env)
        if float(start_delta[2]) < float(above_edge_start_min_actual_z):
            return {
                "seed": seed,
                "success": False,
                "reason": "above_edge_start_min_actual_z_failed",
                "frames": 0,
                "best_meta": dict(best_meta),
                "q_start": [float(value) for value in q_start],
                "trajectory_variant": trajectory_variant,
                "dataset_generation_augmentation": {
                    "above_edge_start": True,
                    "above_edge_start_joint_std": float(above_edge_start_joint_std),
                    "above_edge_start_xy_std": float(above_edge_start_xy_std),
                    "above_edge_start_z_std": float(above_edge_start_z_std),
                    "above_edge_start_min_actual_z": float(above_edge_start_min_actual_z),
                    "above_edge_start_meta": locals().get("above_edge_start_meta"),
                    "actual_start_tcp_to_obj_delta": [float(value) for value in start_delta],
                },
            }
    start_sim_snapshot = _json_safe_sim_snapshot(env)
    start_target_pose = np.asarray(env.unwrapped._get_target_pose(), dtype=float).copy()
    start_static_edge_error = _static_finger_edge_error(env, best_meta)
    start_policy_camera_visibility = _policy_camera_visibility(env, renderers)
    info: dict[str, Any] = env.unwrapped._get_info()
    frames = 0
    success_step = None
    phase_counts: dict[str, int] = {phase[0]: 0 for phase in phases}
    action_deltas: list[float] = []
    wrist_roll_deltas: list[float] = []
    previous_action: np.ndarray | None = None
    episode_frames: list[dict[str, Any]] = []
    q_lift = q_start.copy()
    lift_target_reached = False
    pre_close_static_edge_error: dict[str, float] | None = None
    pre_close_jaw_capture_geometry: dict[str, Any] | None = None
    pre_close_cube_face_normal_parallel_error_deg: float | None = None
    pre_close_policy_camera_visibility: dict[str, Any] | None = None
    pre_close_camera2_top_contact_alignment: dict[str, Any] | None = None
    pre_close_target_pose: np.ndarray | None = None
    pre_close_qpos: np.ndarray | None = None
    near_contact_alignment_sample: dict[str, Any] | None = None
    close_visual_alignment_trace: list[dict[str, Any]] = []
    close_trace_targets = {
        max(0, int(max(1, int(close_steps)) * fraction) - 1): fraction
        for fraction in GRIP_THE_CUBE_V1_CLOSE_TRACE_FRACTIONS
    }
    previous_close_wrist_roll: float | None = None
    render_replay_frames: list[dict[str, Any]] = []
    floor_clearance_geoms = _gripper_floor_clearance_geoms(env)
    minimum_floor_clearance_m = float("inf")
    minimum_floor_clearance_frame: int | None = None
    minimum_floor_clearance_geom: str | None = None

    def add_step(action: np.ndarray, phase: str) -> tuple[bool, bool]:
        nonlocal frames, info, success_step, previous_action
        nonlocal minimum_floor_clearance_m
        nonlocal minimum_floor_clearance_frame, minimum_floor_clearance_geom
        action = np.clip(np.asarray(action, dtype=np.float32), env.action_space.low, env.action_space.high)
        floor_clearance_m, floor_clearance_geom = _minimum_gripper_floor_clearance(
            env, floor_clearance_geoms
        )
        if floor_clearance_m < minimum_floor_clearance_m:
            minimum_floor_clearance_m = float(floor_clearance_m)
            minimum_floor_clearance_frame = int(frames)
            minimum_floor_clearance_geom = floor_clearance_geom
        if capture_render_replay:
            from physical_ai_agent.so101_render_replay import capture_render_replay_frame

            render_replay_frames.append(
                capture_render_replay_frame(
                    env,
                    renderers,
                    episode_index=episode_index,
                    frame_index=frames,
                    timestamp=float(frames) / float(capture_fps),
                )
            )
        if record_dataset_frames:
            episode_frames.append(
                _make_lerobot_frame(
                    env=env,
                    renderers=renderers,
                    action=action,
                    task=task,
                    include_camera3_duplicate=include_camera3_duplicate,
                )
            )
        frames += 1
        phase_counts[phase] += 1
        if previous_action is not None:
            action_deltas.append(float(np.linalg.norm(action[:5] - previous_action[:5])))
            wrist_roll_deltas.append(float(abs(float(action[4]) - float(previous_action[4]))))
        previous_action = action.copy()
        _obs, _reward, terminated, truncated, info = env.step(np.asarray(action, dtype=float))
        if bool(info.get("success", False)) and success_step is None:
            success_step = frames
        return bool(terminated), bool(truncated)

    stopped = False
    for phase_index, (phase, start, target, steps) in enumerate(phases):
        for index in range(max(0, int(steps))):
            if phase == "close" and index == 0 and pre_close_static_edge_error is None:
                pre_close_qpos = _current_qpos(env).astype(np.float32)
                pre_close_target_pose = np.asarray(
                    env.unwrapped._get_target_pose(),
                    dtype=float,
                ).copy()
                pre_close_static_edge_error = _static_finger_edge_error(env, best_meta)
                pre_close_jaw_capture_geometry = _jaw_cube_capture_geometry(env)
                pre_close_cube_face_normal_parallel_error_deg = (
                    _current_jaw_cube_face_normal_error_deg(env, best_meta)
                )
                pre_close_policy_camera_visibility = _policy_camera_visibility(env, renderers)
                if skill_mode in FULL_GRIP_SKILL_MODES:
                    pre_close_camera2_top_contact_alignment = _camera2_top_contact_alignment(
                        env, renderers, best_meta=best_meta
                    )
            if phase == "lift":
                action = np.asarray(
                    _cartesian_error_controller_action(
                        env,
                        np.asarray([0.0, 0.0, float(lift_controller_z_error)]),
                    ),
                    dtype=np.float32,
                )
                action[-1] = float(env.action_space.low[-1])
                if skill_mode in FULL_GRIP_SKILL_MODES and previous_close_wrist_roll is not None:
                    action[4] = float(previous_close_wrist_roll)
                q_lift = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            elif phase == "terminal_hold":
                action = np.asarray(q_lift, dtype=np.float32)
                if skill_mode in FULL_GRIP_SKILL_MODES and previous_close_wrist_roll is not None:
                    action = action.copy()
                    action[4] = float(previous_close_wrist_roll)
            else:
                alpha = (index + 1) / float(max(1, int(steps)))
                alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
                action = (1.0 - alpha) * start + alpha * target
            if skill_mode in FULL_GRIP_SKILL_MODES and phase == "close":
                action = np.asarray(action, dtype=np.float32).copy()
                base_close_step_roll = float(action[4])
                if previous_close_wrist_roll is None:
                    previous_close_wrist_roll = base_close_step_roll
                action[4] = float(previous_close_wrist_roll)
                close_refine_meta = {
                    "reason": "held_aligned_edge_wrist_roll",
                    "wrist_roll": float(previous_close_wrist_roll),
                    "base_wrist_roll": base_close_step_roll,
                    "previous_wrist_roll": float(previous_close_wrist_roll),
                }
                close_trace_entry = {
                    "close_index": int(index),
                    "close_fraction": float((index + 1) / float(max(1, int(steps)))),
                    "planned": close_refine_meta,
                    "refined_this_step": False,
                }
                if index in close_trace_targets:
                    close_trace_entry["checkpoint_fraction"] = float(close_trace_targets[index])
                close_visual_alignment_trace.append(close_trace_entry)
            terminated, truncated = add_step(action, phase)
            if uses_near_contact_contract and phase == "close":
                realized_qpos = _realized_robot_joint_qpos(env)
                target_contact_gripper = float(best_meta["contact_gripper_value"])
                contact_sample = {
                    "close_index": int(index),
                    "close_fraction": float((index + 1) / float(max(1, int(steps)))),
                    "realized_gripper_value": float(realized_qpos[-1]),
                    "target_contact_gripper_value": target_contact_gripper,
                    "gripper_value_error": float(
                        abs(float(realized_qpos[-1]) - target_contact_gripper)
                    ),
                    "parallel_error_deg": float(
                        _current_jaw_cube_face_normal_error_deg(env, best_meta)
                    ),
                    "capture_geometry": _jaw_cube_capture_geometry(env),
                }
                if (
                    near_contact_alignment_sample is None
                    or float(contact_sample["gripper_value_error"])
                    < float(near_contact_alignment_sample["gripper_value_error"])
                ):
                    near_contact_alignment_sample = contact_sample
            if phase == "lift" and _fixed_jaw_lift_target_reached(
                info,
                target_height=float(lift_target_height),
            ):
                lift_target_reached = True
                break
            if _fixed_jaw_terminal_event_stops_episode(
                phase,
                terminated=terminated,
                truncated=truncated,
            ):
                stopped = True
                break
            if skill_mode in FULL_GRIP_SKILL_MODES and phase == "close":
                # Keep commanding the pre-close aligned wrist roll through
                # contact. Updating this from realized qpos lets contact drift
                # become the next target, which breaks close75 alignment.
                pass
            if (
                skill_mode in FULL_GRIP_SKILL_MODES
                and phase == "close"
                and close_visual_alignment_trace
                and (
                    "checkpoint_fraction" in close_visual_alignment_trace[-1]
                )
            ):
                reference_contact_normal_angle = None
                if pre_close_camera2_top_contact_alignment is not None:
                    reference_contact_normal_angle = pre_close_camera2_top_contact_alignment.get(
                        "cube_contact_normal_angle_deg"
                    )
                actual_alignment = _camera2_locked_top_contact_alignment(
                    env,
                    renderers,
                    reference_contact_normal_angle_deg=(
                        None
                        if reference_contact_normal_angle is None
                        else float(reference_contact_normal_angle)
                    ),
                )
                actual_edge_error = _static_finger_edge_error(env, best_meta)
                close_visual_alignment_trace[-1]["actual_after_step"] = {
                    **actual_alignment,
                    "static_edge_xy_error": float(actual_edge_error["xy_error"]),
                    "wrist_roll": float(_current_qpos(env)[4]),
                }
        if stopped:
            remaining_phases = [item[0] for item in phases[phase_index + 1 :]]
            if "terminal_hold" not in remaining_phases:
                break

    final_static_edge_error = _static_finger_edge_error(env, best_meta)
    final_tcp_to_obj_delta = _tcp_to_object_delta(env)
    final_policy_camera_visibility = _policy_camera_visibility(env, renderers)
    close_trace_gate: dict[str, Any] | None = None
    if skill_mode in FULL_GRIP_SKILL_MODES:
        close_trace_gate = _grip_the_cube_v1_close_trace_gate(
            pre_close_camera2_top_contact_alignment,
            close_visual_alignment_trace,
            mode=close_alignment_gate_mode,
            limits=close_alignment_limits,
        )
        best_meta = dict(best_meta)
        best_meta["camera2_top_contact_close_alignment_gate"] = close_trace_gate
        best_meta["wrist_roll_delta_gate"] = {
            "max_wrist_roll_delta_rad": float(max(wrist_roll_deltas) if wrist_roll_deltas else 0.0),
            "limit_rad": float(GRIP_THE_CUBE_V1_MAX_WRIST_ROLL_STEP_RAD),
            "passed": bool((max(wrist_roll_deltas) if wrist_roll_deltas else 0.0) <= GRIP_THE_CUBE_V1_MAX_WRIST_ROLL_STEP_RAD),
        }
    if success_kind == "pick_success":
        task_success = _fixed_jaw_pick_success(
            info,
            lift_target_reached=lift_target_reached,
            terminal_min_height=float(lift_success_height),
        )
        if skill_mode in FULL_GRIP_SKILL_MODES and not uses_near_contact_contract:
            task_success = bool(
                task_success
                and pre_close_jaw_capture_geometry is not None
                and _jaw_capture_geometry_passes(
                    pre_close_jaw_capture_geometry,
                    max_centerline_error_m=float(
                        edge_contact_xy_success_threshold
                    ),
                )
                and pre_close_cube_face_normal_parallel_error_deg is not None
                and pre_close_cube_face_normal_parallel_error_deg
                <= float(edge_contact_parallel_success_threshold_deg)
                and close_trace_gate is not None
                and bool(close_trace_gate.get("passed", False))
                and bool(best_meta.get("wrist_roll_delta_gate", {}).get("passed", True))
            )
        elif uses_near_contact_contract:
            task_success = bool(
                task_success
                and pre_close_jaw_capture_geometry is not None
                and _jaw_capture_geometry_passes(
                    pre_close_jaw_capture_geometry,
                    max_centerline_error_m=float(
                        edge_contact_xy_success_threshold
                    ),
                )
                and near_contact_alignment_sample is not None
                and float(near_contact_alignment_sample["parallel_error_deg"])
                <= float(edge_contact_parallel_success_threshold_deg)
                and _jaw_capture_geometry_passes(
                    near_contact_alignment_sample["capture_geometry"],
                    max_centerline_error_m=float(
                        edge_contact_xy_success_threshold
                    ),
                )
                and close_trace_gate is not None
                and bool(close_trace_gate.get("passed", False))
                and bool(best_meta.get("wrist_roll_delta_gate", {}).get("passed", True))
            )
    elif success_kind == "edge_above":
        start_camera1 = start_policy_camera_visibility["camera1"]
        wrist = final_policy_camera_visibility["camera2"]
        task_success = bool(
            final_static_edge_error["xy_error"] <= 0.025
            and final_tcp_to_obj_delta[2] >= 0.035
            and start_camera1["visible"]
            and start_camera1["centered"]
            and wrist["visible"]
            and wrist["centered"]
        )
    elif success_kind == "edge_contact_parallel":
        task_success = bool(
            final_static_edge_error["xy_error"] <= float(edge_contact_xy_success_threshold)
            and _candidate_cube_normal_parallel_error_deg(best_meta)
            <= float(edge_contact_parallel_success_threshold_deg)
        )
    else:
        task_success = bool(final_static_edge_error["xy_error"] <= 0.015)
    success = task_success
    floor_clearance_passed = bool(
        float(min_gripper_floor_clearance_m) <= 0.0
        or minimum_floor_clearance_m >= float(min_gripper_floor_clearance_m)
    )
    success = bool(success and floor_clearance_passed)
    failure_reason = None
    if not success:
        if not floor_clearance_passed:
            failure_reason = "gripper_floor_clearance_gate_failed"
        elif skill_mode in FULL_GRIP_SKILL_MODES and close_trace_gate is not None and not bool(close_trace_gate.get("passed", False)):
            failure_reason = "close_alignment_gate_failed"
        elif skill_mode in FULL_GRIP_SKILL_MODES and not bool(best_meta.get("wrist_roll_delta_gate", {}).get("passed", True)):
            failure_reason = "wrist_roll_delta_gate_failed"
        elif uses_near_contact_contract and (
            near_contact_alignment_sample is None
            or float(near_contact_alignment_sample["parallel_error_deg"])
            > float(edge_contact_parallel_success_threshold_deg)
            or not _jaw_capture_geometry_passes(
                near_contact_alignment_sample["capture_geometry"],
                max_centerline_error_m=float(edge_contact_xy_success_threshold),
            )
        ):
            failure_reason = "near_contact_alignment_gate_failed"
        else:
            failure_reason = "teacher_replay_failed"
    if success and record_dataset_frames:
        for frame in episode_frames:
            dataset.add_frame(frame)

    result = {
        "seed": seed,
        "frames": frames,
        "success": success,
        "reason": "ok" if success else failure_reason,
        "success_step": success_step if task_success and success_kind == "pick_success" else (frames if task_success else None),
        "task_success": task_success,
        "search_steps": search_steps,
        "teacher_visible_in_any_camera": bool(teacher_visible),
        "best_meta": best_meta,
        "final_info": {
            "is_grasped": bool(info.get("is_grasped", False)),
            "lift_height": float(info.get("lift_height", 0.0)),
            "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        },
        "q_start": [float(value) for value in q_start],
        "sim_snapshot": start_sim_snapshot,
        "q_edge": [float(value) for value in q_edge],
        "q_above": [float(value) for value in q_above],
        "q_above_misaligned": [float(value) for value in locals().get("q_above_misaligned", q_above)],
        "q_lift": [float(value) for value in q_lift],
        "start_target_pose": [float(value) for value in start_target_pose],
        "pre_close_target_pose": (
            None
            if pre_close_target_pose is None
            else [float(value) for value in pre_close_target_pose]
        ),
        "pre_close_qpos": (
            None
            if pre_close_qpos is None
            else [float(value) for value in pre_close_qpos]
        ),
        "pre_close_q_edge_error_l2": (
            None
            if pre_close_qpos is None
            else float(np.linalg.norm(pre_close_qpos - q_edge))
        ),
        "lift_target_height": float(lift_target_height),
        "lift_success_height": float(lift_success_height),
        "lift_controller_z_error": float(lift_controller_z_error),
        "lift_target_reached": bool(lift_target_reached),
        "start_static_edge_error": start_static_edge_error,
        "pre_close_static_edge_error": pre_close_static_edge_error,
        "pre_close_jaw_capture_geometry": pre_close_jaw_capture_geometry,
        "pre_close_cube_face_normal_parallel_error_deg": (
            None
            if pre_close_cube_face_normal_parallel_error_deg is None
            else float(pre_close_cube_face_normal_parallel_error_deg)
        ),
        "pre_close_camera2_top_contact_alignment": pre_close_camera2_top_contact_alignment,
        "near_contact_alignment_sample": near_contact_alignment_sample,
        "camera2_top_contact_close_alignment_trace": close_visual_alignment_trace,
        "final_static_edge_error": final_static_edge_error,
        "start_policy_camera_visibility": start_policy_camera_visibility,
        "pre_close_policy_camera_visibility": pre_close_policy_camera_visibility,
        "final_policy_camera_visibility": final_policy_camera_visibility,
        "wrist_roll_start": float(q_start[4]) if len(q_start) > 4 else None,
        "wrist_roll_edge": float(q_edge[4]) if len(q_edge) > 4 else None,
        "wrist_roll_delta_to_edge": float(abs(float(q_start[4]) - float(q_edge[4]))) if len(q_start) > 4 and len(q_edge) > 4 else None,
        "max_wrist_roll_delta_rad": float(max(wrist_roll_deltas) if wrist_roll_deltas else 0.0),
        "final_tcp_to_obj_delta": [float(value) for value in final_tcp_to_obj_delta],
        "teacher_style": "staged_fixed_jaw_skill",
        "skill_mode": skill_mode,
        "trajectory_variant": trajectory_variant,
        "dataset_generation_augmentation": {
            "terminal_hold_steps": int(effective_terminal_hold_steps),
            "near_target_correction": "near_target_correction" in trajectory_variant,
            "near_target_joint_std": float(near_target_joint_std),
            "near_target_xy_std": float(near_target_xy_std),
            "correction_visibility_scale": locals().get("correction_visibility_scale"),
            "above_edge_start": skill_mode == "grip_from_above_edge_cube",
            "above_edge_start_joint_std": float(above_edge_start_joint_std),
            "above_edge_start_xy_std": float(above_edge_start_xy_std),
            "above_edge_start_z_std": float(above_edge_start_z_std),
            "above_edge_start_min_actual_z": float(above_edge_start_min_actual_z),
            "above_edge_trajectory_variants": _parse_above_edge_variants(above_edge_trajectory_variants),
            "above_edge_selected_variant": trajectory_variant,
            "above_edge_start_gripper_profile": str(above_edge_start_gripper_profile),
            "above_edge_open_phase_gripper_value": float(locals().get("open_phase_gripper_value", _open_gripper_value(env))),
            "above_edge_terminal_hold_jitter": int(above_edge_terminal_hold_jitter),
            "above_edge_start_meta": locals().get("above_edge_start_meta"),
        },
        "fixed_jaw_reference": "static_finger_pad",
        "phase_counts": phase_counts,
        "gripper_floor_clearance_gate": {
            "required_min_clearance_m": float(min_gripper_floor_clearance_m),
            "minimum_clearance_m": float(minimum_floor_clearance_m),
            "minimum_frame_index": minimum_floor_clearance_frame,
            "minimum_geom": minimum_floor_clearance_geom,
            "passed": floor_clearance_passed,
        },
        "mean_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "max_action_delta": float(np.max(action_deltas)) if action_deltas else 0.0,
    }
    if capture_render_replay:
        result["_render_replay_capture"] = {
            "episode_index": int(episode_index),
            "seed": int(seed),
            "frames": render_replay_frames,
            "initial_object_z": float("nan"),
        }
    return result


def _grip_the_cube_v1_close_trace_gate(
    pre_close_alignment: dict[str, Any] | None,
    close_trace: list[dict[str, Any]],
    *,
    mode: str = "strict_image_trace",
    limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    if mode == "geometry_only":
        # Camera2 remains recorded for debugging, but jaw occlusion and
        # perspective projection must not reject a geometrically valid pose.
        return {
            "passed": True,
            "reason": "camera2_diagnostic_only",
            "mode": mode,
            "limits": {},
            "values": {},
            "failures": {},
            "trace_steps": len(close_trace),
        }
    checkpoints: dict[str, float | None] = {}
    if pre_close_alignment is None:
        checkpoints["pre_close_image_alignment_error_deg"] = None
    else:
        error = pre_close_alignment.get("image_alignment_error_deg")
        checkpoints["pre_close_image_alignment_error_deg"] = None if error is None else float(error)

    for fraction in GRIP_THE_CUBE_V1_CLOSE_TRACE_FRACTIONS:
        checkpoint_key = f"close_{int(fraction * 100)}_image_alignment_error_deg"
        selected: dict[str, Any] | None = None
        for entry in close_trace:
            if abs(float(entry.get("checkpoint_fraction", -1.0)) - float(fraction)) < 1e-6:
                selected = entry
                break
        if selected is None and close_trace:
            selected = min(
                close_trace,
                key=lambda item: abs(float(item.get("close_fraction", 0.0)) - float(fraction)),
            )
        actual = (selected or {}).get("actual_after_step", {})
        planned = (selected or {}).get("planned", {})
        error = actual.get("image_alignment_error_deg", planned.get("image_alignment_error_deg"))
        checkpoints[checkpoint_key] = None if error is None else float(error)

    resolved_limits = dict(limits or GRIP_THE_CUBE_V1_CAMERA2_TOP_CONTACT_LIMITS)
    if mode == "preclose_and_early_trace":
        if limits is None:
            # After contact, the jaw mask can disappear behind the cube. Keep the
            # camera2 pre-close and early-contact checks strict, while treating the
            # late image angle as diagnostic rather than a rejection criterion.
            resolved_limits.pop("close_75_image_alignment_error_deg", None)
            resolved_limits["pre_close_image_alignment_error_deg"] = 8.0
            resolved_limits["close_25_image_alignment_error_deg"] = 8.0
            resolved_limits["close_50_image_alignment_error_deg"] = 8.0
    elif mode != "strict_image_trace":
        raise ValueError(f"unknown close alignment gate mode: {mode}")
    failures = {
        key: {"value": checkpoints.get(key), "limit": limit}
        for key, limit in resolved_limits.items()
        if checkpoints.get(key) is None or float(checkpoints[key]) > float(limit)
    }
    return {
        "passed": not failures,
        "reason": "ok" if not failures else "threshold_exceeded",
        "mode": mode,
        "limits": resolved_limits,
        "values": checkpoints,
        "failures": failures,
        "trace_steps": len(close_trace),
    }


def _make_fixed_jaw_edge_qpos(env: Any, q_open: np.ndarray, best_meta: dict[str, Any]) -> np.ndarray:
    q_open = np.clip(np.asarray(q_open, dtype=np.float32), env.action_space.low, env.action_space.high)
    q_open[-1] = _open_gripper_value(env)
    if bool(best_meta.get("fixed_jaw_solver", False)):
        return q_open.astype(np.float32)
    snapshot = _snapshot_sim_state(env)
    try:
        _set_qpos(env, q_open)
        model = env.unwrapped.model
        data = env.unwrapped.data
        obj_geom_id = int(env.unwrapped._obj_geom_id)
        static_geom_id = model.geom("static_finger_pad").id
        current_delta = np.asarray(data.geom_xpos[static_geom_id] - data.geom_xpos[obj_geom_id], dtype=float)
        target_delta = _fixed_jaw_target_delta(env, best_meta, z_value=float(current_delta[2]))
        offset = target_delta - current_delta
    finally:
        _restore_sim_state(env, snapshot)
    q_edge = _offset_qpos_by_cartesian(env, q_open, offset, steps=40)
    q_edge[-1] = _open_gripper_value(env)
    return np.clip(q_edge, env.action_space.low, env.action_space.high).astype(np.float32)


def _refine_close_stable_fixed_jaw_qpos_for_camera2_top_contact(
    env: Any,
    renderers: dict[str, Any],
    *,
    q_edge: np.ndarray,
    best_meta: dict[str, Any],
    close_steps: int,
    close_alignment_gate_mode: str,
    close_alignment_limits: dict[str, float] | None,
    min_floor_clearance_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    current = np.clip(np.asarray(q_edge, dtype=np.float32), low, high)
    current[-1] = _open_gripper_value(env)
    rounds: list[dict[str, Any]] = []
    best: tuple[float, np.ndarray, dict[str, Any]] | None = None
    initial_probe = _probe_grip_the_cube_v1_close_trace_gate(
        env,
        renderers,
        q_edge=current,
        best_meta=best_meta,
        close_steps=max(1, int(close_steps)),
        close_alignment_gate_mode=close_alignment_gate_mode,
        close_alignment_limits=close_alignment_limits,
        min_floor_clearance_m=float(min_floor_clearance_m),
    )
    initial_snapshot = _snapshot_sim_state(env)
    try:
        _set_qpos(env, current)
        initial_edge_error = _static_finger_edge_error(env, best_meta)
        initial_geometry_error_deg = _current_jaw_cube_face_normal_error_deg(
            env, best_meta
        )
    finally:
        _restore_sim_state(env, initial_snapshot)
    initial_floor_clearance_m, initial_floor_geom = _dynamic_close_floor_clearance(
        env,
        current,
        close_steps=max(1, int(close_steps)),
    )
    if (
        bool(initial_probe.get("gate", {}).get("passed", False))
        and float(initial_edge_error["xy_error"]) <= 0.012
        and float(initial_geometry_error_deg) <= 3.0
        and (
            float(min_floor_clearance_m) <= 0.0
            or float(initial_floor_clearance_m) >= float(min_floor_clearance_m)
        )
    ):
        return current, {
            "reason": "existing_pose_passes_all_gates",
            "close_probe": initial_probe,
            "initial_static_edge_error": initial_edge_error,
            "initial_geometry_parallel_error_deg": float(initial_geometry_error_deg),
            "initial_floor_clearance_m": float(initial_floor_clearance_m),
            "initial_floor_clearance_geom": initial_floor_geom,
            "rounds": [],
            "roll_refine": {"reason": "not_needed"},
        }
    snapshot = _snapshot_sim_state(env)
    try:
        for round_index in range(2):
            _set_qpos(env, current)
            before_edge_error = _static_finger_edge_error(env, best_meta)
            contact_offset = np.asarray(
                [
                    float(before_edge_error["target_delta_x"]) - float(before_edge_error["static_delta_x"]),
                    float(before_edge_error["target_delta_y"]) - float(before_edge_error["static_delta_y"]),
                    0.0,
                ],
                dtype=float,
            )
            if float(np.linalg.norm(contact_offset[:2])) > 0.001:
                current = _offset_qpos_by_cartesian(env, current, contact_offset, steps=14)
                current[-1] = _open_gripper_value(env)
                _set_qpos(env, current)
            current, roll_meta = _refine_wrist_roll_for_camera2_top_contact(
                env,
                renderers,
                q_edge=current,
                best_meta=best_meta,
                min_floor_clearance_m=float(min_floor_clearance_m),
            )
            current[-1] = _open_gripper_value(env)
            _set_qpos(env, current)
            after_edge_error = _static_finger_edge_error(env, best_meta)
            probe = _probe_grip_the_cube_v1_close_trace_gate(
                env,
                renderers,
                q_edge=current,
                best_meta=best_meta,
                close_steps=close_steps,
                close_alignment_gate_mode=close_alignment_gate_mode,
                close_alignment_limits=close_alignment_limits,
                min_floor_clearance_m=float(min_floor_clearance_m),
            )
            objective = _close_trace_probe_objective(probe) + 350.0 * float(after_edge_error["xy_error"])
            round_meta = {
                "round": int(round_index),
                "contact_offset": [float(value) for value in contact_offset],
                "before_static_edge_xy_error": float(before_edge_error["xy_error"]),
                "after_static_edge_xy_error": float(after_edge_error["xy_error"]),
                "roll_refine": roll_meta,
                "close_probe": probe,
                "objective": float(objective),
                "q_edge": [float(value) for value in current],
            }
            rounds.append(round_meta)
            if best is None or objective < best[0]:
                best = (float(objective), current.copy(), round_meta)
            if bool(probe.get("gate", {}).get("passed", False)) and float(after_edge_error["xy_error"]) <= 0.012:
                break
    finally:
        _restore_sim_state(env, snapshot)
    if best is None:
        return current.astype(np.float32), {"reason": "no_close_stable_candidate", "rounds": rounds}
    refined = np.clip(best[1], low, high).astype(np.float32)
    refined[-1] = _open_gripper_value(env)
    selected_probe = best[2].get("close_probe", {})
    selected_gate = selected_probe.get("gate", {}) if isinstance(selected_probe, dict) else {}
    selected_passed = bool(selected_gate.get("passed", False)) if isinstance(selected_gate, dict) else False
    return refined, {
        "reason": "ok" if selected_passed else "best_effort",
        "selected_round": int(best[2]["round"]),
        "objective": float(best[0]),
        "promoted_close_wrist_roll": None,
        "post_promote_static_edge_xy_error": None,
        "close_roll_contract": "fixed_preclose_roll_matches_exported_teacher",
        "roll_refine": best[2].get("roll_refine", {}),
        "close_probe": selected_probe,
        "rounds": rounds,
    }


def _close_trace_probe_objective(probe: dict[str, Any]) -> float:
    gate = probe.get("gate", {})
    values = gate.get("values", {}) if isinstance(gate, dict) else {}
    objective = 0.0
    limits = gate.get("limits", GRIP_THE_CUBE_V1_CAMERA2_TOP_CONTACT_LIMITS) if isinstance(gate, dict) else {}
    for key, limit in limits.items():
        value = values.get(key)
        if value is None:
            objective += 180.0
            continue
        objective += float(value)
        objective += 5.0 * max(0.0, float(value) - float(limit))
    required_floor = float(probe.get("required_floor_clearance_m", 0.0))
    minimum_floor = probe.get("minimum_floor_clearance_m")
    if required_floor > 0.0:
        if minimum_floor is None:
            objective += 1_000.0
        else:
            objective += 10_000.0 * max(
                0.0,
                required_floor - float(minimum_floor),
            )
    return float(objective)


def _close_probe_allows_full_episode(*, mode: str, refine_meta: dict[str, Any]) -> bool:
    if mode == "geometry_only":
        return True
    probe = refine_meta.get("close_probe", {})
    gate = probe.get("gate", {}) if isinstance(probe, dict) else {}
    return bool(gate.get("passed", False)) if isinstance(gate, dict) else False


def _probe_grip_the_cube_v1_close_trace_gate(
    env: Any,
    renderers: dict[str, Any],
    *,
    q_edge: np.ndarray,
    best_meta: dict[str, Any],
    close_steps: int,
    close_alignment_gate_mode: str = "strict_image_trace",
    close_alignment_limits: dict[str, float] | None = None,
    min_floor_clearance_m: float = 0.0,
) -> dict[str, Any]:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    start = np.clip(np.asarray(q_edge, dtype=np.float32), low, high)
    start[-1] = _open_gripper_value(env)
    q_close = start.copy()
    q_close[-1] = float(low[-1])
    trace_targets = {
        max(0, int(max(1, int(close_steps)) * fraction) - 1): fraction
        for fraction in GRIP_THE_CUBE_V1_CLOSE_TRACE_FRACTIONS
    }
    trace: list[dict[str, Any]] = []
    locked_wrist_roll = float(start[4])
    pre_close_alignment: dict[str, Any] | None = None
    minimum_floor_clearance_m: float | None = None
    minimum_floor_geom: str | None = None
    floor_clearance_geoms = (
        _gripper_floor_clearance_geoms(env)
        if float(min_floor_clearance_m) > 0.0
        else None
    )
    snapshot = _snapshot_sim_state(env)
    try:
        _set_qpos(env, start)
        pre_close_alignment = _camera2_top_contact_alignment(env, renderers, best_meta=best_meta)
        reference_contact_normal_angle = pre_close_alignment.get("cube_contact_normal_angle_deg")
        for index in range(max(1, int(close_steps))):
            alpha = (index + 1) / float(max(1, int(close_steps)))
            alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
            action = ((1.0 - alpha) * start + alpha * q_close).astype(np.float32)
            base_roll = float(action[4])
            action = action.copy()
            action[4] = locked_wrist_roll
            planned = {
                "reason": "held_aligned_edge_wrist_roll",
                "wrist_roll": locked_wrist_roll,
                "base_wrist_roll": base_roll,
                "previous_wrist_roll": locked_wrist_roll,
            }
            _obs, _reward, terminated, truncated, _info = env.step(np.asarray(action, dtype=float))
            if floor_clearance_geoms is not None:
                floor_clearance, floor_geom = _minimum_gripper_floor_clearance(
                    env,
                    floor_clearance_geoms,
                )
                if (
                    minimum_floor_clearance_m is None
                    or float(floor_clearance) < minimum_floor_clearance_m
                ):
                    minimum_floor_clearance_m = float(floor_clearance)
                    minimum_floor_geom = floor_geom
            entry = {
                "close_index": int(index),
                "close_fraction": float((index + 1) / float(max(1, int(close_steps)))),
                "planned": planned,
                "refined_this_step": False,
            }
            if index in trace_targets:
                entry["checkpoint_fraction"] = float(trace_targets[index])
                actual_alignment = _camera2_locked_top_contact_alignment(
                    env,
                    renderers,
                    reference_contact_normal_angle_deg=(
                        None
                        if reference_contact_normal_angle is None
                        else float(reference_contact_normal_angle)
                    ),
                )
                actual_edge_error = _static_finger_edge_error(env, best_meta)
                entry["actual_after_step"] = {
                    **actual_alignment,
                    "static_edge_xy_error": float(actual_edge_error["xy_error"]),
                    "wrist_roll": float(_current_qpos(env)[4]),
                }
            trace.append(entry)
            if bool(terminated) or bool(truncated):
                break
    finally:
        _restore_sim_state(env, snapshot)
    gate = _grip_the_cube_v1_close_trace_gate(
        pre_close_alignment,
        trace,
        mode=close_alignment_gate_mode,
        limits=close_alignment_limits,
    )
    if (
        float(min_floor_clearance_m) > 0.0
        and (
            minimum_floor_clearance_m is None
            or minimum_floor_clearance_m < float(min_floor_clearance_m)
        )
    ):
        gate = dict(gate)
        gate["passed"] = False
        gate["reason"] = "floor_clearance_below_threshold"
        gate["failures"] = [
            *list(gate.get("failures", [])),
            "minimum_floor_clearance_m",
        ]
    return {
        "pre_close_alignment": pre_close_alignment,
        "trace": trace,
        "gate": gate,
        "required_floor_clearance_m": float(min_floor_clearance_m),
        "minimum_floor_clearance_m": minimum_floor_clearance_m,
        "minimum_floor_geom": minimum_floor_geom,
    }


def _refine_wrist_roll_for_camera2_top_contact(
    env: Any,
    renderers: dict[str, Any],
    *,
    q_edge: np.ndarray,
    best_meta: dict[str, Any],
    min_floor_clearance_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    base = np.clip(np.asarray(q_edge, dtype=np.float32), low, high)
    center = float(base[4])
    span = min(1.2, float(high[4] - low[4]))
    offsets = np.linspace(-span, span, 17, dtype=np.float32)
    candidates = [center, *[float(center + value) for value in offsets]]
    best: tuple[float, np.ndarray, dict[str, Any]] | None = None
    snapshot = _snapshot_sim_state(env)
    try:
        for roll in candidates:
            candidate = base.copy()
            candidate[4] = float(np.clip(roll, low[4], high[4]))
            candidate[-1] = _open_gripper_value(env)
            _set_qpos(env, candidate)
            alignment = _camera2_top_contact_alignment(env, renderers, best_meta=best_meta)
            alignment_error = alignment.get("image_alignment_error_deg")
            if alignment_error is None:
                continue
            parallel_error = _current_jaw_cube_face_normal_error_deg(env, best_meta)
            if parallel_error > 3.0:
                continue
            floor_clearance_m: float | None = None
            if float(min_floor_clearance_m) > 0.0:
                floor_clearance_m, _floor_geom = _dynamic_close_floor_clearance(
                    env,
                    candidate,
                    close_steps=17,
                )
                if floor_clearance_m < float(min_floor_clearance_m):
                    continue
            edge_error = _static_finger_edge_error(env, best_meta)
            xy_error = float(edge_error["xy_error"])
            objective = float(alignment_error) + 350.0 * xy_error
            meta = {
                **alignment,
                "objective": float(objective),
                "static_edge_xy_error": xy_error,
                "geometry_parallel_error_deg": float(parallel_error),
                "wrist_roll": float(candidate[4]),
                "minimum_floor_clearance_m": floor_clearance_m,
                "wrist_roll_target": float(roll),
                "constrained_ik": None,
            }
            if best is None or objective < best[0]:
                best = (objective, candidate.copy(), meta)
    finally:
        _restore_sim_state(env, snapshot)
    if best is None:
        return base.astype(np.float32), {"reason": "no_camera2_top_contact_candidate", "wrist_roll": center}
    refined = np.clip(best[1], low, high).astype(np.float32)
    refined[-1] = _open_gripper_value(env)
    return refined, {"reason": "ok", **best[2], "initial_wrist_roll": center}


def _refine_close_step_wrist_roll_for_camera2_top_contact(
    env: Any,
    renderers: dict[str, Any],
    *,
    action: np.ndarray,
    best_meta: dict[str, Any],
    previous_roll: float | None,
    reference_contact_normal_angle_deg: float | None,
    min_floor_clearance_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    base = np.clip(np.asarray(action, dtype=np.float32), low, high)
    base_roll = float(base[4])
    centers = [base_roll]
    if previous_roll is not None:
        centers.append(float(previous_roll))
    span = min(0.45 if previous_roll is None else 0.28, float(high[4] - low[4]))
    offsets = np.linspace(-span, span, 9 if previous_roll is not None else 13, dtype=np.float32)
    rolls: list[float] = []
    primary_center = float(previous_roll) if previous_roll is not None else base_roll
    for center in [base_roll, primary_center]:
        rolls.append(float(np.clip(center, low[4], high[4])))
    for offset in offsets:
        rolls.append(float(np.clip(primary_center + float(offset), low[4], high[4])))
    rolls = sorted(set(round(value, 6) for value in rolls))

    best: tuple[float, np.ndarray, dict[str, Any]] | None = None
    floor_clearance_geoms = (
        _gripper_floor_clearance_geoms(env)
        if float(min_floor_clearance_m) > 0.0
        else None
    )
    snapshot = _snapshot_sim_state(env)
    try:
        for roll in rolls:
            candidate = base.copy()
            candidate[4] = float(roll)
            # Preserve the close trajectory's gripper command; only wrist roll is
            # corrected so the teacher still learns the intended close profile.
            candidate[-1] = float(base[-1])
            _set_qpos(env, candidate)
            floor_clearance_m: float | None = None
            if floor_clearance_geoms is not None:
                floor_clearance_m, _floor_geom = _minimum_gripper_floor_clearance(
                    env,
                    floor_clearance_geoms,
                )
                if floor_clearance_m < float(min_floor_clearance_m):
                    continue
            alignment = _camera2_locked_top_contact_alignment(
                env,
                renderers,
                reference_contact_normal_angle_deg=reference_contact_normal_angle_deg,
            )
            alignment_error = alignment.get("image_alignment_error_deg")
            if alignment_error is None:
                continue
            parallel_error = _current_jaw_cube_face_normal_error_deg(env, best_meta)
            if parallel_error > 3.0:
                continue
            edge_error = _static_finger_edge_error(env, best_meta)
            xy_error = float(edge_error["xy_error"])
            smooth_from_base = abs(float(candidate[4]) - base_roll)
            smooth_from_previous = 0.0 if previous_roll is None else abs(float(candidate[4]) - float(previous_roll))
            objective = (
                float(alignment_error)
                + 350.0 * xy_error
                + 0.20 * smooth_from_base
                + 0.45 * smooth_from_previous
            )
            meta = {
                **alignment,
                "objective": float(objective),
                "static_edge_xy_error": xy_error,
                "geometry_parallel_error_deg": float(parallel_error),
                "wrist_roll": float(candidate[4]),
                "base_wrist_roll": base_roll,
                "previous_wrist_roll": None if previous_roll is None else float(previous_roll),
                "smooth_from_base": float(smooth_from_base),
                "smooth_from_previous": float(smooth_from_previous),
                "floor_clearance_m": floor_clearance_m,
            }
            if best is None or objective < best[0]:
                best = (float(objective), candidate.copy(), meta)
    finally:
        _restore_sim_state(env, snapshot)

    if best is None:
        return base.astype(np.float32), {
            "reason": "no_close_step_top_contact_candidate",
            "wrist_roll": base_roll,
            "base_wrist_roll": base_roll,
            "previous_wrist_roll": None if previous_roll is None else float(previous_roll),
        }
    refined = np.clip(best[1], low, high).astype(np.float32)
    refined[-1] = float(base[-1])
    return refined, {"reason": "ok", **best[2]}


def _make_fixed_jaw_above_qpos(
    env: Any,
    q_edge: np.ndarray,
    best_meta: dict[str, Any],
    *,
    move_target_z_offset: float,
) -> np.ndarray:
    if str(best_meta.get("solver_profile", "")) == "near_contact":
        q_above = _offset_qpos_by_cartesian(
            env,
            q_edge,
            np.asarray([0.0, 0.0, float(move_target_z_offset)]),
            steps=20,
        )
        q_above[-1] = _open_gripper_value(env)
        return np.clip(
            q_above, env.action_space.low, env.action_space.high
        ).astype(np.float32)
    if bool(best_meta.get("fixed_jaw_solver", False)) and "open_value" in best_meta:
        spec = {
            "grasp_mode": str(best_meta.get("mode", "overhead")),
            "mode": str(
                best_meta.get(
                    "candidate_mode",
                    best_meta.get("mode", "overhead"),
                )
            ),
            "axis": list(best_meta.get("axis", [0.0, 1.0, 0.0])),
            "gap": float(best_meta.get("gap", 0.034)),
            "z_offset": (
                float(best_meta.get("z_offset", 0.0))
                + float(move_target_z_offset)
            ),
            "open_value": float(
                best_meta.get("open_value", _open_gripper_value(env))
            ),
            "candidate_index": int(best_meta.get("candidate_index", 0)),
        }
        try:
            q_above, _solve_meta = _solve_fixed_jaw_edge_qpos_variant(env, spec)
            q_above[-1] = _open_gripper_value(env)
            return np.clip(
                q_above,
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)
        except Exception:
            pass
    q_above = _offset_qpos_by_cartesian(
        env,
        q_edge,
        np.asarray([0.0, 0.0, float(move_target_z_offset)]),
    )
    q_above[-1] = _open_gripper_value(env)
    return np.clip(q_above, env.action_space.low, env.action_space.high).astype(np.float32)


def _fixed_jaw_target_delta(env: Any, best_meta: dict[str, Any], *, z_value: float) -> np.ndarray:
    model = env.unwrapped.model
    obj_geom_id = int(env.unwrapped._obj_geom_id)
    cube_half_extent = float(max(model.geom_size[obj_geom_id][0], model.geom_size[obj_geom_id][1]))
    axis = np.asarray(best_meta.get("axis", [0.0, 1.0, 0.0]), dtype=float)
    axis[2] = 0.0
    norm = float(np.linalg.norm(axis[:2]))
    if norm < 1e-6:
        axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
    else:
        axis = axis / norm
    target_delta = -axis * (cube_half_extent + 0.002)
    target_delta[2] = float(z_value)
    return target_delta


def _static_finger_edge_error(env: Any, best_meta: dict[str, Any]) -> dict[str, float]:
    model = env.unwrapped.model
    data = env.unwrapped.data
    obj_geom_id = int(env.unwrapped._obj_geom_id)
    static_geom_id = model.geom("static_finger_pad").id
    current_delta = np.asarray(data.geom_xpos[static_geom_id] - data.geom_xpos[obj_geom_id], dtype=float)
    target_delta = _fixed_jaw_target_delta(env, best_meta, z_value=float(current_delta[2]))
    delta = current_delta - target_delta
    return {
        "xy_error": float(np.linalg.norm(delta[:2])),
        "z_error": float(delta[2]),
        "static_delta_x": float(current_delta[0]),
        "static_delta_y": float(current_delta[1]),
        "static_delta_z": float(current_delta[2]),
        "target_delta_x": float(target_delta[0]),
        "target_delta_y": float(target_delta[1]),
        "target_delta_z": float(target_delta[2]),
    }


def _jaw_cube_capture_geometry(env: Any) -> dict[str, Any]:
    """Measure whether the cube center lies in the open jaw capture corridor."""
    model = env.unwrapped.model
    data = env.unwrapped.data
    static_geom_id = model.geom("static_finger_pad").id
    moving_geom_id = model.geom("moving_finger_pad").id
    obj_geom_id = int(env.unwrapped._obj_geom_id)
    static = np.asarray(data.geom_xpos[static_geom_id], dtype=float)
    moving = np.asarray(data.geom_xpos[moving_geom_id], dtype=float)
    center = np.asarray(data.geom_xpos[obj_geom_id], dtype=float)
    return _jaw_capture_geometry_from_points(static, moving, center)


def _jaw_capture_geometry_from_points(
    static: np.ndarray,
    moving: np.ndarray,
    center: np.ndarray,
) -> dict[str, Any]:
    static = np.asarray(static, dtype=float).reshape(3)
    moving = np.asarray(moving, dtype=float).reshape(3)
    center = np.asarray(center, dtype=float).reshape(3)
    jaw_xy = (moving - static)[:2]
    span_xy = float(np.linalg.norm(jaw_xy))
    if span_xy <= 1e-8:
        return {
            "valid": False,
            "reason": "degenerate_jaw_line",
            "jaw_span_xy_m": span_xy,
        }
    axis_xy = jaw_xy / span_xy
    center_from_static_xy = center[:2] - static[:2]
    projection_m = float(np.dot(center_from_static_xy, axis_xy))
    perpendicular = center_from_static_xy - projection_m * axis_xy
    return {
        "valid": True,
        "reason": "ok",
        "jaw_span_xy_m": span_xy,
        "cube_projection_from_static_m": projection_m,
        "cube_projection_fraction": projection_m / span_xy,
        "cube_center_to_jaw_line_xy_m": float(np.linalg.norm(perpendicular)),
        "cube_center_between_jaws": bool(0.0 <= projection_m <= span_xy),
        "cube_center_minus_jaw_midpoint_z_m": float(
            center[2] - 0.5 * (static[2] + moving[2])
        ),
    }


def _jaw_capture_geometry_passes(
    geometry: dict[str, Any],
    *,
    max_centerline_error_m: float,
) -> bool:
    return bool(
        geometry.get("valid")
        and geometry.get("cube_center_between_jaws")
        and float(geometry.get("cube_center_to_jaw_line_xy_m", float("inf")))
        <= float(max_centerline_error_m)
    )


def _open_gripper_value(env: Any) -> float:
    return float(env.action_space.high[-1])


def _sample_move_gripper_value(
    env: Any,
    *,
    rng: np.random.Generator,
    episode_index: int,
    profile: str,
    closed_gripper_prob: float,
) -> float:
    low = float(env.action_space.low[-1])
    high = _open_gripper_value(env)
    if profile == "closed":
        return low
    if profile == "balanced":
        buckets = np.linspace(low, high, num=5, dtype=np.float32)
        return float(buckets[int(episode_index) % len(buckets)])
    return low if rng.random() < float(np.clip(closed_gripper_prob, 0.0, 1.0)) else high


def _parse_above_edge_variants(raw: str) -> list[str]:
    supported = {"standard", "two_stage_xy_z", "roll_first", "near_miss_correction"}
    variants = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    if not variants:
        return ["standard"]
    unknown = sorted(set(variants) - supported)
    if unknown:
        raise ValueError(f"unknown above-edge trajectory variants: {unknown}; supported={sorted(supported)}")
    return variants


def _above_edge_open_phase_gripper_value(env: Any, *, episode_index: int, profile: str) -> float:
    low = float(env.action_space.low[-1])
    high = _open_gripper_value(env)
    if str(profile) == "balanced":
        buckets = np.asarray([high, high * 0.75 + low * 0.25, high * 0.5 + low * 0.5], dtype=np.float32)
        return float(buckets[int(episode_index) % len(buckets)])
    return high


def _balance_pick_start_y_offset(
    env: Any,
    qpos: np.ndarray,
    *,
    episode_index: int,
    min_abs_y: float,
    max_abs_y: float,
) -> tuple[np.ndarray, float]:
    low = max(0.0, float(min_abs_y))
    high = max(low, float(max_abs_y))
    buckets = np.linspace(low, high, num=5, dtype=np.float32)
    target_abs_y = float(buckets[int(episode_index) % len(buckets)])
    sign = -1.0 if ((int(episode_index) // len(buckets)) % 2) else 1.0
    target_y_offset = sign * target_abs_y
    snapshot = _snapshot_sim_state(env)
    try:
        qpos = np.clip(np.asarray(qpos, dtype=np.float32), env.action_space.low, env.action_space.high)
        _set_qpos(env, qpos)
        current_y_offset = float(_tcp_to_object_delta(env)[1])
        adjusted = _offset_qpos_by_cartesian(env, qpos, np.asarray([0.0, target_y_offset - current_y_offset, 0.0]))
        adjusted[-1] = _open_gripper_value(env)
        return adjusted, target_y_offset
    finally:
        _restore_sim_state(env, snapshot)


def _tcp_to_object_delta(env: Any) -> np.ndarray:
    model = env.unwrapped.model
    data = env.unwrapped.data
    site_id = model.site("gripperframe").id
    obj_geom_id = int(env.unwrapped._obj_geom_id)
    return np.asarray(data.site_xpos[site_id], dtype=float) - np.asarray(data.geom_xpos[obj_geom_id], dtype=float)


def _tcp_to_object_delta_for_qpos(env: Any, qpos: np.ndarray) -> np.ndarray:
    snapshot = _snapshot_sim_state(env)
    try:
        _set_qpos(env, np.clip(np.asarray(qpos, dtype=np.float32), env.action_space.low, env.action_space.high))
        return _tcp_to_object_delta(env)
    finally:
        _restore_sim_state(env, snapshot)


def _make_near_gripper_qpos(env: Any, q_open: np.ndarray, *, seed: int, joint_std: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 9117)
    target = np.asarray(q_open, dtype=np.float32).copy()
    jitter = rng.normal(0.0, max(0.0, float(joint_std)), size=target.shape).astype(np.float32)
    if jitter.shape[0] >= 6:
        jitter[-1] = 0.0
    target = target + jitter
    target[-1] = _open_gripper_value(env)
    return np.clip(target, env.action_space.low, env.action_space.high).astype(np.float32)


def _make_roll_misaligned_fixed_jaw_qpos(
    env: Any,
    *,
    q_edge: np.ndarray,
    q_above: np.ndarray,
    seed: int,
    episode_index: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 61403)
    target = np.asarray(q_above, dtype=np.float32).copy()
    edge = np.asarray(q_edge, dtype=np.float32)
    if target.shape[0] > 4 and edge.shape[0] > 4:
        roll_offsets = np.asarray([0.45, -0.45, 0.62, -0.62, 0.32, -0.32], dtype=np.float32)
        offset = float(roll_offsets[int(seed) % len(roll_offsets)] + rng.normal(0.0, 0.04))
        target[4] = float(edge[4] + offset)
    target[-1] = _open_gripper_value(env)
    return np.clip(target, env.action_space.low, env.action_space.high).astype(np.float32)


def _make_near_target_fixed_jaw_correction_qpos(
    env: Any,
    *,
    q_edge: np.ndarray,
    seed: int,
    episode_index: int,
    joint_std: float,
    xy_std: float,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 73091 + int(episode_index) * 17)
    target = np.asarray(q_edge, dtype=np.float32).copy()
    jitter = rng.normal(0.0, max(0.0, float(joint_std)), size=target.shape).astype(np.float32)
    if jitter.shape[0] >= 6:
        jitter[-1] = 0.0
    if jitter.shape[0] > 4:
        roll_offsets = np.asarray([0.28, -0.28, 0.42, -0.42, 0.18, -0.18], dtype=np.float32)
        jitter[4] += float(roll_offsets[int(episode_index) % len(roll_offsets)])
    start = np.clip(target + jitter, env.action_space.low, env.action_space.high).astype(np.float32)
    if float(xy_std) > 0.0:
        xy_offset = rng.normal(0.0, float(xy_std), size=2)
        start = _offset_qpos_by_cartesian(env, start, np.asarray([xy_offset[0], xy_offset[1], 0.0], dtype=float), steps=8)
    start[-1] = float(env.action_space.low[-1])
    return np.clip(start, env.action_space.low, env.action_space.high).astype(np.float32)


def _make_above_edge_perturbed_start_qpos(
    env: Any,
    *,
    q_above: np.ndarray,
    seed: int,
    episode_index: int,
    joint_std: float,
    xy_std: float,
    z_std: float,
    min_actual_z: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed) + 95101 + int(episode_index) * 31)
    start = np.asarray(q_above, dtype=np.float32).copy()
    joint_noise = rng.normal(0.0, max(0.0, float(joint_std)), size=start.shape).astype(np.float32)
    if joint_noise.shape[0] >= 6:
        joint_noise[-1] = 0.0
    start = np.clip(start + joint_noise, env.action_space.low, env.action_space.high).astype(np.float32)

    cartesian_offset = np.asarray(
        [
            rng.normal(0.0, max(0.0, float(xy_std))),
            rng.normal(0.0, max(0.0, float(xy_std))),
            rng.normal(0.0, max(0.0, float(z_std))),
        ],
        dtype=float,
    )
    if np.any(np.abs(cartesian_offset) > 0.0):
        start = _offset_qpos_by_cartesian(env, start, cartesian_offset, steps=8)

    delta = _tcp_to_object_delta_for_qpos(env, start)
    min_z = max(0.0, float(min_actual_z))
    z_correction = 0.0
    for _ in range(3):
        if float(delta[2]) >= min_z:
            break
        correction = float(min_z - float(delta[2]) + 0.004)
        z_correction += correction
        start = _offset_qpos_by_cartesian(env, start, np.asarray([0.0, 0.0, correction], dtype=float), steps=12)
        delta = _tcp_to_object_delta_for_qpos(env, start)

    start[-1] = _open_gripper_value(env)
    delta = _tcp_to_object_delta_for_qpos(env, start)
    return np.clip(start, env.action_space.low, env.action_space.high).astype(np.float32), {
        "joint_std": float(joint_std),
        "xy_std": float(xy_std),
        "z_std": float(z_std),
        "min_actual_z": float(min_actual_z),
        "cartesian_offset": [float(value) for value in cartesian_offset],
        "z_correction": float(z_correction),
        "start_tcp_to_obj_delta": [float(value) for value in delta],
    }


def _make_home_closed_start_qpos(env: Any, reset_home_qpos: np.ndarray | None) -> np.ndarray:
    if reset_home_qpos is None:
        qpos = _current_qpos(env).astype(np.float32)
    else:
        qpos = np.asarray(reset_home_qpos, dtype=np.float32).copy()
    qpos = np.clip(qpos, env.action_space.low, env.action_space.high).astype(np.float32)
    qpos[-1] = float(env.action_space.low[-1])
    return qpos


def _offset_qpos_by_cartesian(env: Any, qpos: np.ndarray, offset: np.ndarray, *, steps: int = 10) -> np.ndarray:
    snapshot = _snapshot_sim_state(env)
    try:
        target = np.clip(np.asarray(qpos, dtype=np.float32), env.action_space.low, env.action_space.high)
        gripper_value = float(target[-1])
        _set_qpos(env, target)
        per_step_offset = np.asarray(offset, dtype=float) / float(max(1, int(steps)))
        action = target.copy()
        for _ in range(max(1, int(steps))):
            action = np.asarray(_cartesian_error_controller_action(env, per_step_offset), dtype=np.float32)
            action[-1] = gripper_value
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            _obs, _reward, terminated, truncated, _info = env.step(np.asarray(action, dtype=float))
            if terminated or truncated:
                break
        result = _current_qpos(env).astype(np.float32)
        result[-1] = gripper_value
        return np.clip(result, env.action_space.low, env.action_space.high).astype(np.float32)
    finally:
        _restore_sim_state(env, snapshot)


def _make_random_start_qpos(env: Any, q_open: np.ndarray, *, seed: int, joint_std: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 27183)
    home = _current_qpos(env).astype(np.float32)
    home[-1] = _open_gripper_value(env)
    jitter = rng.normal(0.0, max(0.0, float(joint_std)), size=home.shape).astype(np.float32)
    jitter[-1] = 0.0
    # Blend a home-relative random pose with the target so starts are varied but still reachable.
    target = 0.65 * (home + jitter) + 0.35 * np.asarray(q_open, dtype=np.float32)
    target[-1] = _open_gripper_value(env)
    return np.clip(target, env.action_space.low, env.action_space.high).astype(np.float32)


def _make_lerobot_frame(
    *,
    env: Any,
    renderers: dict[str, Any],
    action: np.ndarray,
    task: str,
    include_camera3_duplicate: bool,
) -> dict[str, Any]:
    wrist = _render_camera(env, renderers["wrist_cam"], "wrist_cam")
    ego = _render_camera(env, renderers["egocentric_cam"], "egocentric_cam")
    frame = {
        "observation.images.camera1": ego,
        "observation.images.camera2": wrist,
        "observation.state": _current_qpos(env).astype(np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "task": task,
    }
    if include_camera3_duplicate:
        frame["observation.images.camera3"] = wrist.copy()
    return frame


def _render_camera(env: Any, renderer: Any, camera_name: str) -> np.ndarray:
    renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, camera_name))
    rotation_degrees = None
    rig_config = getattr(env.unwrapped, "_so101_camera_rig_config", None)
    if rig_config is not None:
        camera_config = rig_config.camera1 if camera_name == "egocentric_cam" else rig_config.camera2
        rotation_degrees = int(camera_config.pixel_postprocess_rotation_degrees)
    pixels = renderer.render()
    if camera_name == "egocentric_cam" and rotation_degrees is not None:
        pixels = postprocess_camera_frame(
            camera_name,
            pixels,
            egocentric_rotation_degrees=rotation_degrees,
        )
    else:
        pixels = postprocess_camera_frame(camera_name, pixels)
    return pixels.astype(np.uint8)


def _gripper_floor_clearance_geoms(env: Any) -> tuple[int, tuple[int, ...]]:
    import mujoco

    model = env.unwrapped.model
    floor_geom = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    )
    if floor_geom < 0:
        raise ValueError("gripper floor-clearance gate requires a named floor geom")
    gripper_body_ids = {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in ("gripper", "moving_jaw_so101_v1")
    }
    gripper_body_ids.discard(-1)
    collision_geoms = tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in gripper_body_ids
        and int(model.geom_contype[geom_id]) != 0
    )
    if not collision_geoms:
        raise ValueError("gripper floor-clearance gate found no gripper collision geoms")
    return floor_geom, collision_geoms


def _minimum_gripper_floor_clearance(
    env: Any,
    clearance_geoms: tuple[int, tuple[int, ...]],
) -> tuple[float, str]:
    import mujoco

    floor_geom, gripper_geoms = clearance_geoms
    model = env.unwrapped.model
    data = env.unwrapped.data
    minimum = float("inf")
    minimum_name = ""
    for geom_id in gripper_geoms:
        distance = float(
            mujoco.mj_geomDistance(model, data, floor_geom, geom_id, 1.0, None)
        )
        if distance < minimum:
            minimum = distance
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            minimum_name = geom_name or f"unnamed_geom_{geom_id}"
    return minimum, minimum_name


def _raise_edge_pose_for_floor_clearance(
    env: Any,
    q_edge: np.ndarray,
    *,
    required_clearance_m: float,
    close_steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Constructively lift a grasp pose before export instead of rejection sampling."""
    target = np.clip(
        np.asarray(q_edge, dtype=np.float32),
        env.action_space.low,
        env.action_space.high,
    ).astype(np.float32)
    target[-1] = _open_gripper_value(env)
    if float(required_clearance_m) <= 0.0:
        return target, {
            "enabled": False,
            "required_clearance_m": float(required_clearance_m),
            "z_correction_m": 0.0,
        }

    best_target = target.copy()
    best_clearance, best_geom = _dynamic_close_floor_clearance(
        env,
        best_target,
        close_steps=close_steps,
    )
    correction = 0.0
    samples: list[dict[str, Any]] = [
        {
            "round": 0,
            "minimum_clearance_m": float(best_clearance),
            "minimum_geom": best_geom,
            "accepted": True,
        }
    ]
    for round_index in range(1, 4):
        # Keep the correction small enough to preserve side contact on the
        # 30 mm cube. The hard trajectory gate still enforces the exact 10 mm
        # requirement; this is only a constructive search margin.
        deficit = float(required_clearance_m) + 0.0002 - best_clearance
        if deficit <= 0.0:
            break
        raised = _offset_qpos_by_cartesian(
            env,
            best_target,
            np.asarray([0.0, 0.0, deficit], dtype=float),
            steps=14,
        )
        raised[-1] = _open_gripper_value(env)
        if float(np.linalg.norm(raised[:5] - best_target[:5])) < 1e-6:
            break
        clearance, geom_name = _dynamic_close_floor_clearance(
            env,
            raised,
            close_steps=close_steps,
        )
        accepted = bool(clearance > best_clearance + 0.00025)
        samples.append(
            {
                "round": int(round_index),
                "minimum_clearance_m": float(clearance),
                "minimum_geom": geom_name,
                "accepted": accepted,
                "requested_z_correction_m": float(deficit),
            }
        )
        if not accepted:
            break
        best_target = raised.astype(np.float32)
        best_clearance = float(clearance)
        best_geom = geom_name
        correction += float(deficit)
    return best_target, {
        "enabled": True,
        "required_clearance_m": float(required_clearance_m),
        "safety_margin_m": 0.0002,
        "z_correction_m": float(correction),
        "samples": samples,
        "passed_preflight": bool(
            best_clearance >= float(required_clearance_m)
        ),
    }


def _dynamic_close_floor_clearance(
    env: Any,
    q_edge: np.ndarray,
    *,
    close_steps: int,
) -> tuple[float, str]:
    clearance_geoms = _gripper_floor_clearance_geoms(env)
    start = np.clip(np.asarray(q_edge, dtype=np.float32), env.action_space.low, env.action_space.high)
    start[-1] = _open_gripper_value(env)
    closed = start.copy()
    closed[-1] = float(env.action_space.low[-1])
    minimum = float("inf")
    minimum_geom = ""
    snapshot = _snapshot_sim_state(env)
    try:
        _set_qpos(env, start)
        for index in range(max(1, int(close_steps))):
            value, geom_name = _minimum_gripper_floor_clearance(env, clearance_geoms)
            if value < minimum:
                minimum = float(value)
                minimum_geom = geom_name
            alpha = (index + 1) / float(max(1, int(close_steps)))
            alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
            action = ((1.0 - alpha) * start + alpha * closed).astype(np.float32)
            _obs, _reward, terminated, truncated, _info = env.step(np.asarray(action, dtype=float))
            if bool(terminated) or bool(truncated):
                break
        value, geom_name = _minimum_gripper_floor_clearance(env, clearance_geoms)
        if value < minimum:
            minimum = float(value)
            minimum_geom = geom_name
    finally:
        _restore_sim_state(env, snapshot)
    return float(minimum), minimum_geom


def _policy_camera_visibility(
    env: Any,
    renderers: dict[str, Any],
    *,
    minimum_area: int = 20,
) -> dict[str, dict[str, Any]]:
    return {
        "camera1": _object_visibility_in_camera(
            env,
            renderers["egocentric_cam"],
            "egocentric_cam",
            minimum_area=minimum_area,
        ),
        "camera2": _object_visibility_in_camera(
            env,
            renderers["wrist_cam"],
            "wrist_cam",
            minimum_area=minimum_area,
        ),
    }


def _object_visibility_in_camera(
    env: Any,
    renderer: Any,
    camera_name: str,
    *,
    minimum_area: int = 20,
) -> dict[str, Any]:
    import mujoco

    renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, camera_name))
    renderer.enable_segmentation_rendering()
    try:
        segmentation = np.asarray(renderer.render()).copy()
    finally:
        renderer.disable_segmentation_rendering()

    rig_config = getattr(env.unwrapped, "_so101_camera_rig_config", None)
    if camera_name == "egocentric_cam" and rig_config is not None:
        segmentation = postprocess_camera_frame(
            camera_name,
            segmentation,
            egocentric_rotation_degrees=int(
                rig_config.camera1.pixel_postprocess_rotation_degrees
            ),
        )
    else:
        segmentation = postprocess_camera_frame(camera_name, segmentation)

    target_slot = env.unwrapped._slots[int(env.unwrapped._target_slot_idx)]
    detection = _target_geom_visibility_from_segmentation(
        segmentation,
        target_geom_id=int(target_slot.geom_id),
        geom_object_type=int(mujoco.mjtObj.mjOBJ_GEOM),
        minimum_area=int(minimum_area),
    )
    height, width = segmentation.shape[:2]
    if not detection["visible"]:
        return {
            "camera_name": camera_name,
            "visible": False,
            "centered": False,
            "centroid": None,
            "normalized_centroid": None,
            "area": 0,
            "bbox": None,
            "center_distance": None,
        }
    u, v = [float(value) for value in detection["centroid"]]
    norm_u = u / float(max(1, width - 1))
    norm_v = v / float(max(1, height - 1))
    centered = bool(0.12 <= norm_u <= 0.88 and 0.12 <= norm_v <= 0.88)
    return {
        "camera_name": camera_name,
        "visible": True,
        "centered": centered,
        "centroid": [u, v],
        "normalized_centroid": [float(norm_u), float(norm_v)],
        "area": int(detection.get("area", 0)),
        "bbox": detection.get("bbox"),
        "center_distance": float(np.linalg.norm(np.asarray([norm_u - 0.5, norm_v - 0.5], dtype=float))),
    }


def _target_geom_visibility_from_segmentation(
    segmentation: np.ndarray,
    *,
    target_geom_id: int,
    geom_object_type: int,
    minimum_area: int = 20,
) -> dict[str, Any]:
    pixels = np.asarray(segmentation)
    if pixels.ndim != 3 or pixels.shape[2] < 2:
        raise ValueError(
            "MuJoCo segmentation must have shape [height, width, >=2], "
            f"got {pixels.shape}"
        )
    target_mask = (
        (pixels[..., 0] == int(target_geom_id))
        & (pixels[..., 1] == int(geom_object_type))
    )
    ys, xs = np.nonzero(target_mask)
    area = int(len(xs))
    if area < int(minimum_area):
        return {
            "visible": False,
            "centroid": None,
            "area": area,
            "bbox": None,
        }
    return {
        "visible": True,
        "centroid": [float(xs.mean()), float(ys.mean())],
        "area": area,
        "bbox": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ],
    }


def _lerobot_features(
    *,
    height: int,
    width: int,
    use_videos: bool,
    include_camera3_duplicate: bool,
) -> dict[str, dict[str, Any]]:
    image_dtype = "video" if use_videos else "image"
    image_feature = {
        "dtype": image_dtype,
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }
    features = {
        "observation.images.camera1": dict(image_feature),
        "observation.images.camera2": dict(image_feature),
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": STATE_NAMES,
        },
    }
    if include_camera3_duplicate:
        features["observation.images.camera3"] = dict(image_feature)
    return features


def audit_lerobot_dataset(
    *,
    root: Path,
    repo_id: str,
    features: dict[str, dict[str, Any]],
    action_space_low: np.ndarray,
    action_space_high: np.ndarray,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # A strict teacher gate can legitimately reject every candidate. In that
    # case LeRobot has not written tasks.parquet yet; constructing a dataset
    # object would fall through to an HF lookup and turn a valid empty export
    # into a misleading network 404.
    if not (root / "meta" / "tasks.parquet").exists():
        return {
            "status": "no_episodes",
            "dataset_len": 0,
            "num_episodes": 0,
            "fps": None,
            "features": features,
            "sample_keys": [],
            "missing_required_keys": [
                "observation.images.camera1",
                "observation.images.camera2",
                "observation.state",
                "action",
                "task",
            ],
            "stats_path": str(root / "meta" / "stats.json"),
            "stats_keys": [],
            "action_min": [],
            "action_max": [],
            "state_min": [],
            "state_max": [],
            "requested_action_space_low": [float(value) for value in action_space_low],
            "requested_action_space_high": [float(value) for value in action_space_high],
        }

    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    sample = dataset[0]
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    action_values = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
    state_values = np.asarray(dataset.hf_dataset["observation.state"], dtype=np.float32)
    action_low = action_values.min(axis=0)
    action_high = action_values.max(axis=0)
    state_low = state_values.min(axis=0)
    state_high = state_values.max(axis=0)
    required_keys = {
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.state",
        "action",
        "task",
    }
    if "observation.images.camera3" in features:
        required_keys.add("observation.images.camera3")
    sample_keys = set(sample.keys())
    audit = {
        "status": "passed",
        "dataset_len": len(dataset),
        "num_episodes": dataset.num_episodes,
        "fps": dataset.fps,
        "features": dataset.features,
        "sample_keys": sorted(sample_keys),
        "missing_required_keys": sorted(required_keys - sample_keys),
        "sample_shapes": {
            key: list(np.asarray(sample[key]).shape)
            for key in required_keys
            if key in sample and key != "task"
        },
        "task_sample": sample.get("task"),
        "stats_path": str(stats_path),
        "stats_keys": sorted(stats.keys()),
        "action_min": [float(value) for value in action_low],
        "action_max": [float(value) for value in action_high],
        "action_space_low": [float(value) for value in action_space_low],
        "action_space_high": [float(value) for value in action_space_high],
        "action_within_space": bool(
            np.all(action_low >= action_space_low - 1e-5)
            and np.all(action_high <= action_space_high + 1e-5)
        ),
        "state_min": [float(value) for value in state_low],
        "state_max": [float(value) for value in state_high],
        "state_within_space": bool(
            np.all(state_low >= action_space_low - 1e-5)
            and np.all(state_high <= action_space_high + 1e-5)
        ),
        "declared_features": features,
    }
    if audit["missing_required_keys"]:
        audit["status"] = "failed"
    if list(action_values.shape[1:]) != [6] or list(state_values.shape[1:]) != [6]:
        audit["status"] = "failed"
    if "action" not in stats or "observation.state" not in stats:
        audit["status"] = "failed"
    if not audit["action_within_space"] or not audit["state_within_space"]:
        audit["status"] = "failed"
    audit_path = root / "so101_lerobot_audit.json"
    audit["audit_path"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


if __name__ == "__main__":
    main()
