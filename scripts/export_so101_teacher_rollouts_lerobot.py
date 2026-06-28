#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_so101_picklift_image_policy import detect_colored_object
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


TASK = "Grasp the visible cube and lift it up."
SKILL_TASKS = {
    "pick_cube": TASK,
    "move_over_cube": "Move the gripper over the visible cube.",
    "pick_from_top_cube": "From above the visible cube, grasp it and lift it up.",
    "move_over_cube_edge": "Move the gripper above one visible cube edge.",
    "align_fixed_jaw_cube_edge": "Align the gripper jaws around one visible cube edge.",
    "move_and_align_cube_edge": "Move the gripper above one visible cube edge and align the jaws around it.",
    "grip_from_edge_cube": "Close the gripper on the cube edge and lift.",
}
COLOR_SHAPE_SKILL_TASK_TEMPLATES = {
    "pick_cube": "Grasp the visible {color} {shape} and lift it up.",
    "move_over_cube": "Move the gripper over the visible {color} {shape}.",
    "pick_from_top_cube": "From above the visible {color} {shape}, grasp it and lift it up.",
    "move_over_cube_edge": "Move the gripper above one visible {color} {shape} edge.",
    "align_fixed_jaw_cube_edge": "Align the gripper jaws around one visible {color} {shape} edge.",
    "move_and_align_cube_edge": "Move above one visible {color} {shape} edge and align the gripper jaws around it.",
    "grip_from_edge_cube": "Close the gripper on the {color} {shape} edge and lift.",
}
FIXED_JAW_SKILL_MODES = {
    "move_over_cube_edge",
    "align_fixed_jaw_cube_edge",
    "move_and_align_cube_edge",
    "grip_from_edge_cube",
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
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--teacher-style", choices=["legacy", "staged"], default="staged")
    parser.add_argument("--approach-steps", type=int, default=34)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--close-steps", type=int, default=42)
    parser.add_argument("--lift-steps", type=int, default=58)
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
        use_videos=args.use_videos,
        overwrite=args.overwrite,
        teacher_style=args.teacher_style,
        approach_steps=args.approach_steps,
        settle_steps=args.settle_steps,
        close_steps=args.close_steps,
        lift_steps=args.lift_steps,
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
    use_videos: bool,
    overwrite: bool,
    teacher_style: str = "staged",
    approach_steps: int = 34,
    settle_steps: int = 10,
    close_steps: int = 42,
    lift_steps: int = 58,
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
    spawn_lookup: dict[int, list[list[float]]] = {}
    spawn_lookup_next = {int(bin_id): 0 for bin_id in balance_bins}
    if balance_enabled and grid_balance_spawn_lookup:
        env.reset(seed=seed)
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
        for candidates_xy in spawn_lookup.values():
            candidates_xy.sort(
                key=lambda xy: (float(xy[0]) - float(spawn_center[0])) ** 2
                + (float(xy[1]) - float(spawn_center[1])) ** 2
            )
        if grid_balance_teacher_feasible_lookup:
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
            )
        missing = [int(bin_id) for bin_id in balance_bins if not spawn_lookup.get(int(bin_id))]
        if missing:
            raise RuntimeError(f"camera1 spawn lookup has no candidates for bins: {missing}")
        print(
            "[so101-lerobot] camera1 spawn lookup "
            + json.dumps({str(key): len(value) for key, value in sorted(spawn_lookup.items())}, sort_keys=True),
            flush=True,
        )
    exported = 0
    attempted = 0
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
                forced_spawn_xy = candidates_xy[next_index % len(candidates_xy)]
                spawn_lookup_next[desired_grid_bin] = next_index + 1
            episode_seed = candidate_seed
            candidate_seed += 1
            if forced_spawn_xy is not None and len(forced_spawn_xy) >= 3:
                episode_seed = int(forced_spawn_xy[2])
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
                candidates = _make_fast_fixed_jaw_teacher_targets(env)
            else:
                candidates = make_teacher_targets(env)
            if skill_mode in {"move_over_cube", "pick_from_top_cube", *FIXED_JAW_SKILL_MODES}:
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
            best = max(candidates, key=lambda item: float(item["meta"].get("score", -1e9)))
            summary = _write_teacher_episode(
                dataset=dataset,
                env=env,
                renderers=policy_renderers,
                q_open=np.asarray(best["q_open"], dtype=np.float32),
                q_lift=np.asarray(best["q_lift"], dtype=np.float32),
                seed=episode_seed,
                search_steps=search_steps,
                teacher_visible=teacher_visible,
                best_meta=dict(best["meta"]),
                teacher_style=teacher_style,
                approach_steps=approach_steps,
                settle_steps=settle_steps,
                close_steps=close_steps,
                lift_steps=lift_steps,
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
                include_camera3_duplicate=include_camera3_duplicate,
                reset_home_qpos=reset_home_qpos,
            )
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
                    grid_bin = _summary_start_grid_bin(summary, grid_size=int(grid_balance_size))
                    summary["grid_balance_bin"] = grid_bin
                    if grid_bin not in balance_counts:
                        dataset.clear_episode_buffer()
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
                        dataset.clear_episode_buffer()
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
                dataset.clear_episode_buffer()
                skipped.append({"seed": episode_seed, "reason": "teacher_replay_failed", **summary})
    finally:
        for renderer in [*policy_renderers.values(), *teacher_renderers.values()]:
            renderer.close()
        env.close()

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
        "grid_balance": {
            "enabled": bool(balance_enabled),
            "grid_size": int(grid_balance_size),
            "target_per_bin": int(grid_balance_target_per_bin),
            "requested_bins": sorted(balance_counts),
            "accepted_counts": {str(key): int(value) for key, value in sorted(balance_counts.items())},
            "spawn_lookup": {
                "enabled": bool(spawn_lookup),
                "resolution": int(grid_lookup_resolution),
                "x_range": [float(grid_lookup_x_min), float(grid_lookup_x_max)],
                "y_range": [float(grid_lookup_y_min), float(grid_lookup_y_max)],
                "candidate_counts": {str(key): len(value) for key, value in sorted(spawn_lookup.items())},
                "teacher_feasible_filter": bool(grid_balance_teacher_feasible_lookup),
                "max_candidates_per_bin": int(grid_lookup_max_candidates_per_bin),
            },
        },
        "fps": fps,
        "use_videos": use_videos,
        "teacher_style": teacher_style,
        "teacher_timing": {
            "approach_steps": int(approach_steps),
            "settle_steps": int(settle_steps),
            "close_steps": int(close_steps),
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
            "near_target_correction_included": float(move_and_align_near_target_correction_ratio) > 0.0,
            "near_target_correction_ratio": float(move_and_align_near_target_correction_ratio),
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
    lift_steps: int,
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
            lift_steps=lift_steps,
            episode_index=episode_index,
            random_start_joint_std=random_start_joint_std,
            move_target_z_offset=move_target_z_offset,
            terminal_hold_steps=terminal_hold_steps,
            move_and_align_near_target_correction_ratio=move_and_align_near_target_correction_ratio,
            edge_contact_xy_success_threshold=edge_contact_xy_success_threshold,
            edge_contact_parallel_success_threshold_deg=edge_contact_parallel_success_threshold_deg,
            near_target_joint_std=near_target_joint_std,
            near_target_xy_std=near_target_xy_std,
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
                candidates=_make_fast_fixed_jaw_teacher_targets(env),
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
            if float(meta.get("finger_axis_parallel_angle_deg", 180.0)) > float(edge_contact_parallel_success_threshold_deg):
                continue
            q_edge = _make_fixed_jaw_edge_qpos(env, np.asarray(candidate["q_open"], dtype=np.float32), meta)
            q_edge[-1] = _open_gripper_value(env)
            _set_qpos(env, q_edge)
            if float(_static_finger_edge_error(env, meta)["xy_error"]) <= float(edge_contact_xy_success_threshold):
                return True
    finally:
        _restore_sim_state(env, snapshot)
    return False


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
        return skill_mode in {"align_fixed_jaw_cube_edge", "grip_from_edge_cube"}
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


def _make_fast_fixed_jaw_teacher_targets(env: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    specs = [
        spec
        for spec in _grasp_candidate_specs(env)
        if str(spec.get("grasp_mode")) == "overhead"
    ]
    for spec in specs:
        try:
            q_open, solve_meta = _solve_fixed_jaw_edge_qpos_variant(env, spec)
        except Exception:
            continue
        meta = {
            "mode": str(spec["grasp_mode"]),
            "candidate_mode": str(spec["mode"]),
            "axis": [float(value) for value in np.asarray(spec["axis"], dtype=float)],
            "gap": float(spec["gap"]),
            "z_offset": float(spec["z_offset"]),
            "open_value": float(spec["open_value"]),
            "success_step": None,
            "score": (
                -float(solve_meta["cost"])
                - 0.12 * float(solve_meta.get("finger_axis_parallel_angle_deg", 0.0))
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
            q_above[-1] = float(env.action_space.low[-1])
            _set_qpos(env, q_above)
            visibility = _policy_camera_visibility(env, renderers)
            wrist = visibility["camera2"]
            if not bool(wrist["visible"]) or not bool(wrist["centered"]):
                continue
            center_distance = float(wrist["center_distance"] or 0.0)
            meta["preselected_policy_camera_visibility"] = visibility
            meta["score"] = float(meta.get("score", 0.0)) - center_distance
            selected = dict(candidate)
            selected["meta"] = meta
            ranked.append((float(meta["score"]), selected))
    finally:
        _restore_sim_state(env, snapshot)
    return [candidate for _score, candidate in sorted(ranked, key=lambda item: item[0], reverse=True)]


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
    starts = list(base_starts)
    for base in base_starts:
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
            max_nfev=140,
        )
        cost = float(np.linalg.norm(residual(result.x)))
        candidate = np.concatenate([result.x, np.asarray([open_value])])
        if best is None or cost < best[0]:
            best = (cost, candidate)
    assert best is not None
    qpos = np.clip(best[1], low, high)
    set_qpos(qpos)
    static_delta = np.asarray(data.geom_xpos[static_pad] - data.geom_xpos[obj_geom_id], dtype=float)
    moving_delta = np.asarray(data.geom_xpos[moving_pad] - data.geom_xpos[obj_geom_id], dtype=float)
    finger_axis = np.asarray(data.geom_xpos[moving_pad] - data.geom_xpos[static_pad], dtype=float)
    finger_axis_xy = finger_axis[:2] / max(1e-6, float(np.linalg.norm(finger_axis[:2])))
    axis_parallel_dot = float(np.clip(np.dot(finger_axis_xy, desired_axis_xy), -1.0, 1.0))
    axis_parallel_angle_deg = float(np.degrees(np.arccos(axis_parallel_dot)))
    target_delta = desired_static - obj_pos
    return qpos.astype(np.float32), {
        "cost": float(best[0]),
        "static_edge_xy_error": float(np.linalg.norm((static_delta - target_delta)[:2])),
        "finger_axis_parallel_dot": axis_parallel_dot,
        "finger_axis_parallel_angle_deg": axis_parallel_angle_deg,
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
        return bool(info.get("success", False)) or bool(terminated) or bool(truncated)

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
    lift_steps: int,
    episode_index: int,
    random_start_joint_std: float,
    move_target_z_offset: float,
    terminal_hold_steps: int,
    move_and_align_near_target_correction_ratio: float,
    edge_contact_xy_success_threshold: float,
    edge_contact_parallel_success_threshold_deg: float,
    near_target_joint_std: float,
    near_target_xy_std: float,
    task: str,
    include_camera3_duplicate: bool,
    reset_home_qpos: np.ndarray | None,
) -> dict[str, Any]:
    q_edge = _make_fixed_jaw_edge_qpos(env, q_open, best_meta)
    q_above = _make_fixed_jaw_above_qpos(env, q_edge, best_meta, move_target_z_offset=move_target_z_offset)
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
    else:
        raise ValueError(f"unknown fixed jaw skill mode: {skill_mode}")
    if int(terminal_hold_steps) > 0 and skill_mode != "grip_from_edge_cube":
        hold_target = np.asarray(phases[-1][2] if phases[-1][2] is not None else phases[-1][1], dtype=np.float32)
        phases.append(("terminal_hold", hold_target, hold_target, int(terminal_hold_steps)))
    if "trajectory_variant" not in locals():
        trajectory_variant = "generated_teacher"

    _set_qpos(env, q_start)
    start_static_edge_error = _static_finger_edge_error(env, best_meta)
    start_policy_camera_visibility = _policy_camera_visibility(env, renderers)
    info: dict[str, Any] = env.unwrapped._get_info()
    frames = 0
    success_step = None
    phase_counts: dict[str, int] = {phase[0]: 0 for phase in phases}
    action_deltas: list[float] = []
    previous_action: np.ndarray | None = None
    q_lift = q_start.copy()

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

    for phase, start, target, steps in phases:
        for index in range(max(0, int(steps))):
            if target is None:
                action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=np.float32)
                action[-1] = float(env.action_space.low[-1])
                q_lift = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            else:
                alpha = (index + 1) / float(max(1, int(steps)))
                alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
                action = (1.0 - alpha) * start + alpha * target
            if add_step(action, phase):
                break
        if bool(info.get("success", False)):
            break

    final_static_edge_error = _static_finger_edge_error(env, best_meta)
    final_tcp_to_obj_delta = _tcp_to_object_delta(env)
    final_policy_camera_visibility = _policy_camera_visibility(env, renderers)
    if success_kind == "pick_success":
        task_success = bool(info.get("success", False))
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
            and float(best_meta.get("finger_axis_parallel_angle_deg", 180.0))
            <= float(edge_contact_parallel_success_threshold_deg)
        )
    else:
        task_success = bool(final_static_edge_error["xy_error"] <= 0.015)
    success = task_success

    return {
        "seed": seed,
        "frames": frames,
        "success": success,
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
        "q_edge": [float(value) for value in q_edge],
        "q_above": [float(value) for value in q_above],
        "q_lift": [float(value) for value in q_lift],
        "start_static_edge_error": start_static_edge_error,
        "final_static_edge_error": final_static_edge_error,
        "start_policy_camera_visibility": start_policy_camera_visibility,
        "final_policy_camera_visibility": final_policy_camera_visibility,
        "wrist_roll_start": float(q_start[4]) if len(q_start) > 4 else None,
        "wrist_roll_edge": float(q_edge[4]) if len(q_edge) > 4 else None,
        "wrist_roll_delta_to_edge": float(abs(float(q_start[4]) - float(q_edge[4]))) if len(q_start) > 4 and len(q_edge) > 4 else None,
        "final_tcp_to_obj_delta": [float(value) for value in final_tcp_to_obj_delta],
        "teacher_style": "staged_fixed_jaw_skill",
        "skill_mode": skill_mode,
        "trajectory_variant": trajectory_variant,
        "dataset_generation_augmentation": {
            "terminal_hold_steps": int(terminal_hold_steps),
            "near_target_correction": trajectory_variant == "near_target_correction",
            "near_target_joint_std": float(near_target_joint_std),
            "near_target_xy_std": float(near_target_xy_std),
        },
        "fixed_jaw_reference": "static_finger_pad",
        "phase_counts": phase_counts,
        "mean_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "max_action_delta": float(np.max(action_deltas)) if action_deltas else 0.0,
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
        offset = float(roll_offsets[int(episode_index) % len(roll_offsets)] + rng.normal(0.0, 0.04))
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
    if detection is None:
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
