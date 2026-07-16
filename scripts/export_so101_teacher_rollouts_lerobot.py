#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_so101_picklift_image_policy import detect_colored_object
from physical_ai_agent.so101_resolution_contract import require_so101_image_resolution
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
}
STATE_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


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
        choices=["standard", "roll_first", "direct_align"],
        default="standard",
        help="Deterministic intermediate path for fixed-jaw grip trajectories.",
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
    parser.add_argument("--lift-steps", type=int, default=58)
    parser.add_argument(
        "--lift-target-height",
        type=float,
        default=0.05,
        help="Stop the lift phase after the grasped object reaches this height in meters.",
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
        "--target-object-color",
        choices=["red", "orange", "yellow", "green", "blue", "purple", "black", "white"],
        help="Only export episodes whose target object has this color.",
    )
    parser.add_argument("--spawn-center-x", type=float, default=0.15)
    parser.add_argument("--spawn-center-y", type=float, default=0.0)
    parser.add_argument("--spawn-min-radius", type=float, default=0.10)
    parser.add_argument("--spawn-max-radius", type=float, default=0.30)
    parser.add_argument("--spawn-angle-half-range-deg", type=float, default=90.0)
    parser.add_argument("--object-half-sizes", default="0.0125,0.015,0.0175")
    parser.add_argument("--no-camera3-duplicate", action="store_true")
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
        lift_steps=args.lift_steps,
        lift_target_height=args.lift_target_height,
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
        target_object_color=args.target_object_color,
        spawn_center=(args.spawn_center_x, args.spawn_center_y),
        spawn_min_radius=args.spawn_min_radius,
        spawn_max_radius=args.spawn_max_radius,
        spawn_angle_half_range_deg=args.spawn_angle_half_range_deg,
        object_half_sizes=_parse_float_list(args.object_half_sizes),
        include_camera3_duplicate=not args.no_camera3_duplicate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


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
    lift_steps: int = 58,
    lift_target_height: float = 0.05,
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
    target_object_color: str | None = None,
    spawn_center: tuple[float, float] = (0.15, 0.0),
    spawn_min_radius: float = 0.10,
    spawn_max_radius: float = 0.30,
    spawn_angle_half_range_deg: float = 90.0,
    object_half_sizes: tuple[float, ...] = (0.0125, 0.015, 0.0175),
    include_camera3_duplicate: bool = True,
) -> dict[str, Any]:
    import shutil

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    export_started = time.perf_counter()
    require_so101_image_resolution(height=height, width=width, context="SO101 LeRobot teacher export")
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
    env = make_high_contrast_picklift_env(
        target_object_color=target_object_color,
        object_half_sizes=object_half_sizes,
        spawn_center=spawn_center,
        spawn_min_radius=spawn_min_radius,
        spawn_max_radius=spawn_max_radius,
        spawn_angle_half_range_deg=spawn_angle_half_range_deg,
    )
    policy_renderers = _make_policy_renderers(env, config)
    teacher_renderers = _make_teacher_renderers(env, config)
    action_space_low = np.asarray(env.action_space.low, dtype=np.float32).copy()
    action_space_high = np.asarray(env.action_space.high, dtype=np.float32).copy()
    allow_diagonal_fixed_jaw = skill_mode != "grip_the_cube_v1"
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
    try:
        candidate_seed = seed
        while exported < episodes and attempted < episodes * max_attempt_multiplier:
            attempted += 1
            if attempted % 50 == 0:
                print(
                    f"[so101-lerobot] attempts={attempted} exported={exported}/{episodes} "
                    f"grid_counts={json.dumps({str(k): int(v) for k, v in sorted(balance_counts.items())}, sort_keys=True)}",
                    flush=True,
                )
            desired_grid_bin = None
            forced_spawn_xy = None
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
            if skill_mode in FIXED_JAW_SKILL_MODES:
                candidates = _make_fast_fixed_jaw_teacher_targets(
                    env,
                    allow_diagonal=allow_diagonal_fixed_jaw,
                )
            else:
                candidates = make_teacher_targets(env)
            if skill_mode in {"move_over_cube", "pick_from_top_cube", *FIXED_JAW_SKILL_MODES} and skill_mode != "grip_the_cube_v1":
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
            for candidate_rank, best in enumerate(ranked_candidates):
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
                    lift_steps=lift_steps,
                    lift_target_height=lift_target_height,
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
                    reset_home_qpos=reset_home_qpos,
                )
                if summary["success"]:
                    break
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
            if summary["success"]:
                if balance_enabled:
                    use_source_manifest_bin = bool(
                        str(grip_the_cube_start_profile) == "correction"
                        and lookup_cache_kind == "source_episode_manifest"
                        and desired_grid_bin is not None
                        and forced_spawn_xy is not None
                        and len(forced_spawn_xy) >= 3
                    )
                    grid_bin = (
                        int(desired_grid_bin)
                        if use_source_manifest_bin
                        else _summary_start_grid_bin(summary, grid_size=int(grid_balance_size))
                    )
                    summary["grid_balance_bin"] = grid_bin
                    summary["grid_balance_bin_source"] = (
                        "source_episode_manifest" if use_source_manifest_bin else "start_camera1_centroid"
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
                dataset.save_episode()
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
            "deterministic_camera1_bin_lookup_fixed_jaw_ik"
            if deterministic_camera_bin_lookup
            else "seed_rejection_with_fixed_jaw_ik"
        ),
        "generation_timing": {
            "total_seconds": float(time.perf_counter() - export_started),
            "lookup_build_seconds": float(lookup_build_seconds),
            "seconds_per_exported_episode": (
                float(time.perf_counter() - export_started) / float(exported) if exported else None
            ),
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
        "fps": fps,
        "use_videos": use_videos,
        "teacher_style": teacher_style,
        "teacher_timing": {
            "approach_steps": int(approach_steps),
            "settle_steps": int(settle_steps),
            "close_steps": int(close_steps),
            "close_alignment_gate_mode": str(close_alignment_gate_mode),
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
            "object_half_sizes": [float(value) for value in object_half_sizes],
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
    lift_steps: int,
    lift_target_height: float,
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
    reset_home_qpos: np.ndarray | None = None,
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
            trajectory_variant=trajectory_variant,
            grip_the_cube_start_profile=grip_the_cube_start_profile,
            lift_steps=lift_steps,
            lift_target_height=lift_target_height,
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
            reset_home_qpos=reset_home_qpos,
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
) -> dict[int, list[list[float]]]:
    resolution = max(2, int(resolution))
    lookup: dict[int, list[list[float]]] = {}
    snapshot = _snapshot_sim_state(env)
    try:
        for x in np.linspace(float(x_min), float(x_max), resolution):
            for y in np.linspace(float(y_min), float(y_max), resolution):
                _set_target_object_xy(env, [float(x), float(y)])
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
) -> dict[str, Any]:
    return {
        "format": "so101_camera1_spawn_lookup_v1",
        "grid_size": int(grid_size),
        "resolution": int(resolution),
        "x_range": [float(value) for value in x_range],
        "y_range": [float(value) for value in y_range],
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
) -> None:
    if payload.get("format") != "so101_camera1_spawn_lookup_v1":
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
) -> dict[int, list[list[float]]]:
    filtered_lookup: dict[int, list[list[float]]] = {}
    max_candidates_per_bin = max(0, int(max_candidates_per_bin))
    for bin_id, candidates_xy in sorted(spawn_lookup.items()):
        accepted: list[list[float]] = []
        for index, xy in enumerate(candidates_xy):
            candidate_seed = int(seed) + int(bin_id) * 1000 + index
            env.reset(seed=candidate_seed)
            _set_target_object_xy(env, xy)
            visible, _search_steps = sweep_until_visible(env, renderers, max_sweeps=config.max_sweeps)
            if not visible:
                continue
            candidates = _filter_fixed_jaw_move_candidates_in_policy_view(
                env,
                renderers=renderers,
                candidates=_make_fast_fixed_jaw_teacher_targets(
                    env,
                    allow_diagonal=allow_diagonal_fixed_jaw,
                ),
                move_target_z_offset=move_target_z_offset,
            )
            if not _has_success_contract_fixed_jaw_candidate(
                env,
                candidates,
                edge_contact_xy_success_threshold=edge_contact_xy_success_threshold,
                edge_contact_parallel_success_threshold_deg=edge_contact_parallel_success_threshold_deg,
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
) -> bool:
    snapshot = _snapshot_sim_state(env)
    try:
        for candidate in candidates:
            meta = dict(candidate.get("meta") or {})
            if _candidate_cube_normal_parallel_error_deg(meta) > float(edge_contact_parallel_success_threshold_deg):
                continue
            q_edge = _make_fixed_jaw_edge_qpos(env, np.asarray(candidate["q_open"], dtype=np.float32), meta)
            q_edge[-1] = _open_gripper_value(env)
            _set_qpos(env, q_edge)
            if float(_static_finger_edge_error(env, meta)["xy_error"]) <= float(edge_contact_xy_success_threshold):
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
        return skill_mode in {"align_fixed_jaw_cube_edge", "grip_from_edge_cube", "grip_the_cube_v1"}
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


def _make_fast_fixed_jaw_teacher_targets(env: Any, *, allow_diagonal: bool = True) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    specs = [
        spec
        for spec in _grasp_candidate_specs(env)
        if (allow_diagonal or not str(spec.get("mode", "")).startswith("diag_"))
        and (allow_diagonal or str(spec.get("grasp_mode")) in {"front", "overhead"})
    ]
    for raw_spec in specs:
        try:
            spec = _spec_with_rotated_cube_face_normal(env, raw_spec)
            q_open, solve_meta = _solve_fixed_jaw_edge_qpos_variant(env, spec)
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
            "mode_successes": None,
            "fast_preview_candidate": True,
            "fast_preview_source": "fixed_jaw_edge_ik",
            "fixed_jaw_solver": True,
            **solve_meta,
        }
        candidates.append({"q_open": q_open.astype(float), "q_lift": q_open.astype(float), "meta": meta})
    return candidates


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
            alignment = _camera2_top_contact_alignment(env, renderers)
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


def _camera2_top_contact_alignment(env: Any, renderers: dict[str, Any]) -> dict[str, Any]:
    image = _render_camera(env, renderers["wrist_cam"], "wrist_cam")
    score = _image_alignment_score(image, edge_mode="top-contact")
    return {
        "reason": score.get("reason"),
        "image_alignment_error_deg": score.get("image_alignment_error_deg"),
        "cube_top_contact_edge_angle_deg": score.get("cube_top_contact_edge_angle_deg"),
        "jaw_angle_deg": score.get("jaw_angle_deg"),
        "contact_edge_distance_px": score.get("contact_edge_distance_px"),
    }


def _camera2_locked_top_contact_alignment(
    env: Any,
    renderers: dict[str, Any],
    *,
    reference_edge_angle_deg: float | None,
) -> dict[str, Any]:
    if reference_edge_angle_deg is None:
        return _camera2_top_contact_alignment(env, renderers)
    import cv2

    image = _render_camera(env, renderers["wrist_cam"], "wrist_cam")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yellow = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 45) & (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 55)).astype(
        "uint8"
    )
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    jaw_angle = _jaw_axis_angle(yellow)
    if jaw_angle is None:
        return {
            "reason": "missing_jaw_mask_locked_top_contact",
            "image_alignment_error_deg": None,
            "cube_top_contact_edge_angle_deg": float(reference_edge_angle_deg),
            "jaw_angle_deg": None,
            "contact_edge_distance_px": None,
            "locked_top_contact_edge": True,
        }
    return {
        "reason": "ok",
        "image_alignment_error_deg": float(_angle_diff(float(reference_edge_angle_deg), float(jaw_angle))),
        "cube_top_contact_edge_angle_deg": float(reference_edge_angle_deg),
        "jaw_angle_deg": float(jaw_angle),
        "contact_edge_distance_px": None,
        "locked_top_contact_edge": True,
    }


def _solve_fixed_jaw_edge_qpos_variant(env: Any, spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
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
    desired_static = obj_pos - axis * (cube_half_extent + 0.002) + np.asarray([0.0, 0.0, z_offset])
    desired_moving = desired_static + axis * gap
    desired_center = 0.5 * (desired_static + desired_moving)
    desired_axis_xy = axis[:2] / max(1e-6, float(np.linalg.norm(axis[:2])))

    def set_qpos(qpos: np.ndarray) -> None:
        for addr, value in zip(joint_addrs, qpos):
            data.qpos[addr] = value
        data.ctrl[unwrapped._actuator_ids] = np.clip(qpos, low, high)
        mujoco.mj_forward(model, data)

    def residual(arm_qpos: np.ndarray) -> np.ndarray:
        qpos = np.concatenate([arm_qpos, np.asarray([open_value])])
        set_qpos(qpos)
        static_pos = data.geom_xpos[static_pad]
        moving_pos = data.geom_xpos[moving_pad]
        center = 0.5 * (static_pos + moving_pos)
        finger_axis = moving_pos - static_pos
        finger_axis_xy = finger_axis[:2] / max(1e-6, float(np.linalg.norm(finger_axis[:2])))
        parallel_error = finger_axis_xy - desired_axis_xy
        return np.concatenate(
            [
                (static_pos - desired_static) * 28.0,
                (moving_pos - desired_moving) * 18.0,
                (center - desired_center) * 6.0,
                parallel_error * 18.0,
                (arm_qpos - q_seed[:5]) * 0.025,
            ]
        )

    base_starts = [
        q_seed[:5],
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
    best: tuple[float, np.ndarray] | None = None
    for start in starts:
        result = least_squares(
            residual,
            np.clip(start, low[:5], high[:5]),
            bounds=(low[:5], high[:5]),
            max_nfev=35,
        )
        cost = float(np.linalg.norm(residual(result.x)))
        candidate = np.concatenate([result.x, np.asarray([open_value])])
        if best is None or cost < best[0]:
            best = (cost, candidate)
        if best is not None and best[0] < 1.25:
            break
    assert best is not None
    qpos = np.clip(best[1], low, high)
    set_qpos(qpos)
    static_delta = np.asarray(data.geom_xpos[static_pad] - data.geom_xpos[obj_geom_id], dtype=float)
    moving_delta = np.asarray(data.geom_xpos[moving_pad] - data.geom_xpos[obj_geom_id], dtype=float)
    finger_axis = np.asarray(data.geom_xpos[moving_pad] - data.geom_xpos[static_pad], dtype=float)
    finger_axis_xy = finger_axis[:2] / max(1e-6, float(np.linalg.norm(finger_axis[:2])))
    axis_parallel_dot = float(np.clip(np.dot(finger_axis_xy, desired_axis_xy), -1.0, 1.0))
    cube_face_normal_parallel_error_deg = _jaw_line_cube_face_normal_error_deg(
        finger_axis_xy,
        desired_axis_xy,
    )
    target_delta = desired_static - obj_pos
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
        "target_delta_x": float(target_delta[0]),
        "target_delta_y": float(target_delta[1]),
        "target_delta_z": float(target_delta[2]),
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
    trajectory_variant: str,
    grip_the_cube_start_profile: str,
    lift_steps: int,
    lift_target_height: float,
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
    reset_home_qpos: np.ndarray | None,
) -> dict[str, Any]:
    q_edge = _make_fixed_jaw_edge_qpos(env, q_open, best_meta)
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
        )
    best_meta = dict(best_meta)
    best_meta["camera2_top_contact_close_stable_refine"] = close_stable_refine_meta
    best_meta["camera2_top_contact_roll_refine"] = close_stable_refine_meta.get("roll_refine", {})
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
    elif skill_mode == "grip_the_cube_v1":
        requested_path_variant = str(trajectory_variant)
        if requested_path_variant not in {"standard", "roll_first", "direct_align"}:
            raise ValueError(f"unknown grip_the_cube_v1 trajectory variant: {requested_path_variant}")
        home_start = _current_qpos(env).astype(np.float32) if reset_home_qpos is None else np.asarray(reset_home_qpos, dtype=np.float32).copy()
        home_start = np.clip(home_start, env.action_space.low, env.action_space.high).astype(np.float32)
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
        trajectory_variant = f"grip_the_cube_v1_{start_variant}_{requested_path_variant}"
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
    pre_close_cube_face_normal_parallel_error_deg: float | None = None
    pre_close_policy_camera_visibility: dict[str, Any] | None = None
    pre_close_camera2_top_contact_alignment: dict[str, Any] | None = None
    close_visual_alignment_trace: list[dict[str, Any]] = []
    close_trace_targets = {
        max(0, int(max(1, int(close_steps)) * fraction) - 1): fraction
        for fraction in GRIP_THE_CUBE_V1_CLOSE_TRACE_FRACTIONS
    }
    previous_close_wrist_roll: float | None = None

    def add_step(action: np.ndarray, phase: str) -> tuple[bool, bool]:
        nonlocal frames, info, success_step, previous_action
        action = np.clip(np.asarray(action, dtype=np.float32), env.action_space.low, env.action_space.high)
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
                pre_close_static_edge_error = _static_finger_edge_error(env, best_meta)
                pre_close_cube_face_normal_parallel_error_deg = (
                    _current_jaw_cube_face_normal_error_deg(env, best_meta)
                )
                pre_close_policy_camera_visibility = _policy_camera_visibility(env, renderers)
                if skill_mode == "grip_the_cube_v1":
                    pre_close_camera2_top_contact_alignment = _camera2_top_contact_alignment(env, renderers)
            if phase == "lift":
                action = np.asarray(
                    _cartesian_error_controller_action(
                        env,
                        np.asarray([0.0, 0.0, float(lift_controller_z_error)]),
                    ),
                    dtype=np.float32,
                )
                action[-1] = float(env.action_space.low[-1])
                if skill_mode == "grip_the_cube_v1" and previous_close_wrist_roll is not None:
                    action[4] = float(previous_close_wrist_roll)
                q_lift = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            elif phase == "terminal_hold":
                action = np.asarray(q_lift, dtype=np.float32)
                if skill_mode == "grip_the_cube_v1" and previous_close_wrist_roll is not None:
                    action = action.copy()
                    action[4] = float(previous_close_wrist_roll)
            else:
                alpha = (index + 1) / float(max(1, int(steps)))
                alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
                action = (1.0 - alpha) * start + alpha * target
            if skill_mode == "grip_the_cube_v1" and phase == "close":
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
            if skill_mode == "grip_the_cube_v1" and phase == "close":
                # Keep commanding the pre-close aligned wrist roll through
                # contact. Updating this from realized qpos lets contact drift
                # become the next target, which breaks close75 alignment.
                pass
            if (
                skill_mode == "grip_the_cube_v1"
                and phase == "close"
                and close_visual_alignment_trace
                and (
                    "checkpoint_fraction" in close_visual_alignment_trace[-1]
                )
            ):
                reference_edge_angle = None
                if pre_close_camera2_top_contact_alignment is not None:
                    reference_edge_angle = pre_close_camera2_top_contact_alignment.get("cube_top_contact_edge_angle_deg")
                actual_alignment = _camera2_locked_top_contact_alignment(
                    env,
                    renderers,
                    reference_edge_angle_deg=None if reference_edge_angle is None else float(reference_edge_angle),
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
    if skill_mode == "grip_the_cube_v1":
        close_trace_gate = _grip_the_cube_v1_close_trace_gate(
            pre_close_camera2_top_contact_alignment,
            close_visual_alignment_trace,
            mode=close_alignment_gate_mode,
        )
        best_meta = dict(best_meta)
        best_meta["camera2_top_contact_close_alignment_gate"] = close_trace_gate
        best_meta["wrist_roll_delta_gate"] = {
            "max_wrist_roll_delta_rad": float(max(wrist_roll_deltas) if wrist_roll_deltas else 0.0),
            "limit_rad": float(GRIP_THE_CUBE_V1_MAX_WRIST_ROLL_STEP_RAD),
            "passed": bool((max(wrist_roll_deltas) if wrist_roll_deltas else 0.0) <= GRIP_THE_CUBE_V1_MAX_WRIST_ROLL_STEP_RAD),
        }
    if success_kind == "pick_success":
        task_success = bool(
            info.get("is_grasped", False)
            and float(info.get("lift_height", 0.0)) >= float(lift_target_height)
            and lift_target_reached
        )
        if skill_mode == "grip_the_cube_v1":
            pre_close_error = pre_close_static_edge_error or final_static_edge_error
            task_success = bool(
                task_success
                and pre_close_error["xy_error"] <= float(edge_contact_xy_success_threshold)
                and pre_close_cube_face_normal_parallel_error_deg is not None
                and pre_close_cube_face_normal_parallel_error_deg
                <= float(edge_contact_parallel_success_threshold_deg)
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
    failure_reason = None
    if not success:
        if skill_mode == "grip_the_cube_v1" and close_trace_gate is not None and not bool(close_trace_gate.get("passed", False)):
            failure_reason = "close_alignment_gate_failed"
        elif skill_mode == "grip_the_cube_v1" and not bool(best_meta.get("wrist_roll_delta_gate", {}).get("passed", True)):
            failure_reason = "wrist_roll_delta_gate_failed"
        else:
            failure_reason = "teacher_replay_failed"
    if success:
        for frame in episode_frames:
            dataset.add_frame(frame)

    return {
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
        "lift_target_height": float(lift_target_height),
        "lift_controller_z_error": float(lift_controller_z_error),
        "lift_target_reached": bool(lift_target_reached),
        "start_static_edge_error": start_static_edge_error,
        "pre_close_static_edge_error": pre_close_static_edge_error,
        "pre_close_cube_face_normal_parallel_error_deg": (
            None
            if pre_close_cube_face_normal_parallel_error_deg is None
            else float(pre_close_cube_face_normal_parallel_error_deg)
        ),
        "pre_close_camera2_top_contact_alignment": pre_close_camera2_top_contact_alignment,
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
        "mean_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "max_action_delta": float(np.max(action_deltas)) if action_deltas else 0.0,
    }


def _grip_the_cube_v1_close_trace_gate(
    pre_close_alignment: dict[str, Any] | None,
    close_trace: list[dict[str, Any]],
    *,
    mode: str = "strict_image_trace",
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

    limits = dict(GRIP_THE_CUBE_V1_CAMERA2_TOP_CONTACT_LIMITS)
    if mode == "preclose_and_early_trace":
        # After contact, the jaw mask can disappear behind the cube. Keep the
        # camera2 pre-close and early-contact checks strict, while treating the
        # late image angle as diagnostic rather than a rejection criterion.
        limits.pop("close_75_image_alignment_error_deg", None)
        limits["pre_close_image_alignment_error_deg"] = 8.0
        limits["close_25_image_alignment_error_deg"] = 8.0
        limits["close_50_image_alignment_error_deg"] = 8.0
    elif mode != "strict_image_trace":
        raise ValueError(f"unknown close alignment gate mode: {mode}")
    failures = {
        key: {"value": checkpoints.get(key), "limit": limit}
        for key, limit in limits.items()
        if checkpoints.get(key) is None or float(checkpoints[key]) > float(limit)
    }
    return {
        "passed": not failures,
        "reason": "ok" if not failures else "threshold_exceeded",
        "mode": mode,
        "limits": limits,
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
) -> tuple[np.ndarray, dict[str, Any]]:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    current = np.clip(np.asarray(q_edge, dtype=np.float32), low, high)
    current[-1] = _open_gripper_value(env)
    rounds: list[dict[str, Any]] = []
    best: tuple[float, np.ndarray, dict[str, Any]] | None = None
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
    promoted_close_roll: float | None = None
    post_promote_static_edge_xy_error: float | None = None
    if isinstance(selected_probe, dict):
        for entry in selected_probe.get("trace", []):
            planned = entry.get("planned", {}) if isinstance(entry, dict) else {}
            if planned.get("wrist_roll") is not None:
                promoted_close_roll = float(planned["wrist_roll"])
                break
    if promoted_close_roll is not None:
        refined[4] = float(np.clip(promoted_close_roll, low[4], high[4]))
        edge_error: dict[str, float] | None = None
        for _ in range(4):
            _set_qpos(env, refined)
            edge_error = _static_finger_edge_error(env, best_meta)
            contact_offset = np.asarray(
                [
                    float(edge_error["target_delta_x"]) - float(edge_error["static_delta_x"]),
                    float(edge_error["target_delta_y"]) - float(edge_error["static_delta_y"]),
                    0.0,
                ],
                dtype=float,
            )
            if float(np.linalg.norm(contact_offset[:2])) <= 0.001:
                break
            refined = _offset_qpos_by_cartesian(env, refined, contact_offset, steps=18)
            refined[-1] = _open_gripper_value(env)
            refined[4] = float(np.clip(promoted_close_roll, low[4], high[4]))
            _set_qpos(env, refined)
            edge_error = _static_finger_edge_error(env, best_meta)
        post_promote_static_edge_xy_error = None if edge_error is None else float(edge_error["xy_error"])
    return refined, {
        "reason": "ok" if selected_passed else "best_effort",
        "selected_round": int(best[2]["round"]),
        "objective": float(best[0]),
        "promoted_close_wrist_roll": promoted_close_roll,
        "post_promote_static_edge_xy_error": post_promote_static_edge_xy_error,
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
    return float(objective)


def _probe_grip_the_cube_v1_close_trace_gate(
    env: Any,
    renderers: dict[str, Any],
    *,
    q_edge: np.ndarray,
    best_meta: dict[str, Any],
    close_steps: int,
    close_alignment_gate_mode: str = "strict_image_trace",
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
    previous_roll: float | None = None
    pre_close_alignment: dict[str, Any] | None = None
    snapshot = _snapshot_sim_state(env)
    try:
        _set_qpos(env, start)
        pre_close_alignment = _camera2_top_contact_alignment(env, renderers)
        reference_edge_angle = pre_close_alignment.get("cube_top_contact_edge_angle_deg")
        for index in range(max(1, int(close_steps))):
            alpha = (index + 1) / float(max(1, int(close_steps)))
            alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
            action = ((1.0 - alpha) * start + alpha * q_close).astype(np.float32)
            should_refine = GRIP_THE_CUBE_V1_REFINE_EVERY_CLOSE_STEP or previous_roll is None
            if should_refine:
                action, planned = _refine_close_step_wrist_roll_for_camera2_top_contact(
                    env,
                    renderers,
                    action=action,
                    best_meta=best_meta,
                    previous_roll=previous_roll,
                    reference_edge_angle_deg=None if reference_edge_angle is None else float(reference_edge_angle),
                )
                previous_roll = float(action[4])
            else:
                base_roll = float(action[4])
                action = action.copy()
                action[4] = float(previous_roll)
                planned = {
                    "reason": "held_previous_close_wrist_roll",
                    "wrist_roll": float(previous_roll),
                    "base_wrist_roll": base_roll,
                    "previous_wrist_roll": float(previous_roll),
                }
            _obs, _reward, terminated, truncated, _info = env.step(np.asarray(action, dtype=float))
            entry = {
                "close_index": int(index),
                "close_fraction": float((index + 1) / float(max(1, int(close_steps)))),
                "planned": planned,
                "refined_this_step": bool(should_refine),
            }
            if index in trace_targets:
                entry["checkpoint_fraction"] = float(trace_targets[index])
                actual_alignment = _camera2_locked_top_contact_alignment(
                    env,
                    renderers,
                    reference_edge_angle_deg=None if reference_edge_angle is None else float(reference_edge_angle),
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
    )
    return {
        "pre_close_alignment": pre_close_alignment,
        "trace": trace,
        "gate": gate,
    }


def _refine_wrist_roll_for_camera2_top_contact(
    env: Any,
    renderers: dict[str, Any],
    *,
    q_edge: np.ndarray,
    best_meta: dict[str, Any],
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
            alignment = _camera2_top_contact_alignment(env, renderers)
            alignment_error = alignment.get("image_alignment_error_deg")
            if alignment_error is None:
                continue
            edge_error = _static_finger_edge_error(env, best_meta)
            xy_error = float(edge_error["xy_error"])
            objective = float(alignment_error) + 350.0 * xy_error
            meta = {
                **alignment,
                "objective": float(objective),
                "static_edge_xy_error": xy_error,
                "wrist_roll": float(candidate[4]),
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
    reference_edge_angle_deg: float | None,
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
    snapshot = _snapshot_sim_state(env)
    try:
        for roll in rolls:
            candidate = base.copy()
            candidate[4] = float(roll)
            # Preserve the close trajectory's gripper command; only wrist roll is
            # corrected so the teacher still learns the intended close profile.
            candidate[-1] = float(base[-1])
            _set_qpos(env, candidate)
            alignment = _camera2_locked_top_contact_alignment(
                env,
                renderers,
                reference_edge_angle_deg=reference_edge_angle_deg,
            )
            alignment_error = alignment.get("image_alignment_error_deg")
            if alignment_error is None:
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
                "wrist_roll": float(candidate[4]),
                "base_wrist_roll": base_roll,
                "previous_wrist_roll": None if previous_roll is None else float(previous_roll),
                "smooth_from_base": float(smooth_from_base),
                "smooth_from_previous": float(smooth_from_previous),
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
    if bool(best_meta.get("fixed_jaw_solver", False)) and "open_value" in best_meta:
        spec = {
            "grasp_mode": str(best_meta.get("mode", "overhead")),
            "mode": str(best_meta.get("candidate_mode", best_meta.get("mode", "overhead"))),
            "axis": list(best_meta.get("axis", [0.0, 1.0, 0.0])),
            "gap": float(best_meta.get("gap", 0.034)),
            "z_offset": float(best_meta.get("z_offset", 0.0)) + float(move_target_z_offset),
            "open_value": float(best_meta.get("open_value", _open_gripper_value(env))),
            "candidate_index": int(best_meta.get("candidate_index", 0)),
        }
        try:
            q_above, _solve_meta = _solve_fixed_jaw_edge_qpos_variant(env, spec)
            q_above[-1] = _open_gripper_value(env)
            return np.clip(q_above, env.action_space.low, env.action_space.high).astype(np.float32)
        except Exception:
            pass
    q_above = _offset_qpos_by_cartesian(env, q_edge, np.asarray([0.0, 0.0, float(move_target_z_offset)]))
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
    return postprocess_camera_frame(camera_name, renderer.render()).astype(np.uint8)


def _policy_camera_visibility(env: Any, renderers: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "camera1": _object_visibility_in_camera(env, renderers["egocentric_cam"], "egocentric_cam"),
        "camera2": _object_visibility_in_camera(env, renderers["wrist_cam"], "wrist_cam"),
    }


def _object_visibility_in_camera(env: Any, renderer: Any, camera_name: str) -> dict[str, Any]:
    image = _render_camera(env, renderer, camera_name)
    detection = detect_colored_object(image)
    height, width = image.shape[:2]
    if detection is None or int(detection.get("area", 0)) < 20:
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
