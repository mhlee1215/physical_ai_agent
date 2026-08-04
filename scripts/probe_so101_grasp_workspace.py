#!/usr/bin/env python3
# ruff: noqa: E402
"""Map SO101 base-relative positions that support a real grasp-and-lift rollout."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import export_so101_teacher_rollouts_lerobot as teacher_exporter
from train_so101_wrist_ego_visual_servo import (
    WristEgoServoConfig,
    _make_policy_renderers,
    make_high_contrast_picklift_env,
)

from physical_ai_agent.sim.so101_camera_rig_render_config import (
    load_so101_camera_rig_render_config,
)
from physical_ai_agent.so101_workspace_probe import (
    WorkspaceProbeConfig,
    annotate_physical_outcomes,
    base_relative_to_world_xy,
    grid_points,
    physical_outcome_metrics,
    successful_workspace_cells,
    summarize_workspace_records,
)

_WORKER_RUNTIME: dict[str, Any] = {}


class _NoFrameDataset:
    def add_frame(self, _frame: dict[str, Any]) -> None:
        raise AssertionError("workspace probe must not materialize LeRobot frames")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Run at most this many unfinished points; 0 runs the complete grid.",
    )
    parser.add_argument(
        "--physical-only",
        action="store_true",
        help="Skip the camera-aware dataset-contract replay.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent MuJoCo worker processes. Each worker owns its env/renderers.",
    )
    parser.add_argument(
        "--point-id",
        action="append",
        default=[],
        help="Probe only this exact grid point id; repeat for multiple points.",
    )
    parser.add_argument(
        "--base-xy",
        action="append",
        default=[],
        help="Probe an exact base-relative x,y[,seed] position; repeat as needed.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> tuple[WorkspaceProbeConfig, str]:
    raw = path.read_bytes()
    return WorkspaceProbeConfig.model_validate_json(raw), hashlib.sha256(raw).hexdigest()


def _exact_points(
    raw_points: list[str], *, config: WorkspaceProbeConfig
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, value in enumerate(raw_points):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) not in {2, 3}:
            raise ValueError(f"--base-xy must be x,y[,seed], got {value!r}")
        x_m, y_m = (float(part) for part in parts[:2])
        if config.object_yaw_mode == "fixed":
            yaw_deg = float(config.object_yaw_degrees[0])
        else:
            yaw_deg = math.degrees(math.atan2(y_m, x_m)) + float(
                config.radial_yaw_offsets_degrees[0]
            )
        points.append(
            {
                "point_id": f"exact_{index:03d}",
                "x_index": -1,
                "y_index": -1,
                "yaw_index": 0,
                "base_x_m": x_m,
                "base_y_m": y_m,
                "yaw_deg": yaw_deg,
                "seed_override": None if len(parts) == 2 else int(parts[2]),
            }
        )
    return points


def _base_frame(env: Any) -> tuple[tuple[float, float], np.ndarray, dict[str, Any]]:
    import mujoco

    model = env.unwrapped.model
    data = env.unwrapped.data
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan"
    )
    if joint_id < 0:
        raise RuntimeError("SO101 model has no shoulder_pan joint")
    body_id = int(model.jnt_bodyid[joint_id])
    local_joint_pos = np.asarray(model.jnt_pos[joint_id], dtype=float)
    joint_world = (
        np.asarray(data.xpos[body_id], dtype=float)
        + np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        @ local_joint_pos
    )
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_body_id < 0:
        raise RuntimeError("SO101 model has no base body")
    base_rotation = np.asarray(data.xmat[base_body_id], dtype=float).reshape(3, 3)
    rotation_xy = base_rotation[:2, :2]
    return (
        (float(joint_world[0]), float(joint_world[1])),
        rotation_xy,
        {
            "reference": "shoulder_pan_axis_projection",
            "world_xyz_m": [float(value) for value in joint_world],
            "base_world_rotation_xy": rotation_xy.tolist(),
        },
    )


def _make_runtime(
    config: WorkspaceProbeConfig,
) -> tuple[
    Any,
    dict[str, Any],
    tuple[float, float],
    np.ndarray,
    dict[str, Any],
    str,
]:
    rig_path = (REPO_ROOT / config.camera_rig_config).resolve()
    rig = load_so101_camera_rig_render_config(rig_path)
    env = make_high_contrast_picklift_env(
        target_object_color=config.object_color,
        object_half_sizes=(config.object_half_size_m,),
        spawn_center=(0.15, 0.0),
        spawn_min_radius=0.1,
        spawn_max_radius=0.3,
        spawn_angle_half_range_deg=180.0,
        camera_rig_preset=rig.preset,
        camera_rig_config=rig,
    )
    renderers = _make_policy_renderers(
        env, WristEgoServoConfig(width=256, height=256)
    )
    env.reset(seed=config.seed_base)
    teacher_exporter._set_qpos(
        env, np.asarray(config.home_qpos, dtype=np.float32)
    )
    base_xy, base_rotation_xy, base_contract = _base_frame(env)
    return (
        env,
        renderers,
        base_xy,
        base_rotation_xy,
        base_contract,
        hashlib.sha256(rig_path.read_bytes()).hexdigest(),
    )


def _close_runtime(env: Any, renderers: dict[str, Any]) -> None:
    for renderer in renderers.values():
        renderer.close()
    env.close()


def _worker_init(config_json: str, evaluate_dataset_contract: bool) -> None:
    config = WorkspaceProbeConfig.model_validate_json(config_json)
    (
        env,
        renderers,
        base_xy,
        base_rotation_xy,
        _base_contract,
        _rig_sha256,
    ) = _make_runtime(config)
    _WORKER_RUNTIME.update(
        {
            "config": config,
            "env": env,
            "renderers": renderers,
            "base_xy": base_xy,
            "base_rotation_xy": base_rotation_xy,
            "evaluate_dataset_contract": evaluate_dataset_contract,
        }
    )


def _worker_probe(payload: tuple[dict[str, Any], int]) -> dict[str, Any]:
    point, point_index = payload
    return _probe_point(
        env=_WORKER_RUNTIME["env"],
        renderers=_WORKER_RUNTIME["renderers"],
        config=_WORKER_RUNTIME["config"],
        point=point,
        point_index=point_index,
        base_xy=_WORKER_RUNTIME["base_xy"],
        base_rotation_xy=_WORKER_RUNTIME["base_rotation_xy"],
        evaluate_dataset_contract=bool(
            _WORKER_RUNTIME["evaluate_dataset_contract"]
        ),
    )


def _run_teacher_candidate_set(
    *,
    env: Any,
    renderers: dict[str, Any],
    candidates: list[dict[str, Any]],
    config: WorkspaceProbeConfig,
    seed: int,
    point_index: int,
    reset_home_qpos: np.ndarray,
    close_alignment_gate_mode: str,
    close_alignment_limits: dict[str, float] | None,
) -> dict[str, Any]:
    snapshot = teacher_exporter._snapshot_sim_state(env)
    failures: list[dict[str, Any]] = []
    try:
        ranked = sorted(
            candidates,
            key=lambda item: float(item["meta"].get("score", -1e9)),
            reverse=True,
        )
        for rank, candidate in enumerate(ranked):
            teacher_exporter._restore_sim_state(env, snapshot)
            meta = dict(candidate["meta"])
            meta["teacher_candidate_rank"] = rank
            meta["teacher_candidate_count"] = len(ranked)
            result = teacher_exporter._write_fixed_jaw_edge_episode(
                dataset=_NoFrameDataset(),
                env=env,
                renderers=renderers,
                q_open=np.asarray(candidate["q_open"], dtype=np.float32),
                seed=seed,
                search_steps=0,
                teacher_visible=True,
                best_meta=meta,
                skill_mode=config.teacher.skill_mode,
                approach_steps=config.teacher.approach_steps,
                settle_steps=config.teacher.settle_steps,
                close_steps=config.teacher.close_steps,
                close_alignment_gate_mode=close_alignment_gate_mode,
                close_alignment_limits=close_alignment_limits,
                trajectory_variant=config.teacher.trajectory_variant,
                grip_the_cube_start_profile="home",
                lift_steps=config.teacher.lift_steps,
                lift_target_height=config.teacher.lift_target_height_m,
                lift_success_height=(
                    config.teacher.operational_lift_height_m
                ),
                lift_controller_z_error=config.teacher.lift_controller_z_error_m,
                episode_index=point_index,
                random_start_joint_std=0.0,
                move_target_z_offset=config.teacher.move_target_z_offset_m,
                terminal_hold_steps=config.teacher.terminal_hold_steps,
                move_and_align_near_target_correction_ratio=0.0,
                edge_contact_xy_success_threshold=(
                    config.teacher.edge_contact_xy_threshold_m
                ),
                edge_contact_parallel_success_threshold_deg=(
                    config.teacher.edge_parallel_threshold_deg
                ),
                near_target_joint_std=0.0,
                near_target_xy_std=0.0,
                above_edge_start_joint_std=0.0,
                above_edge_start_xy_std=0.0,
                above_edge_start_z_std=0.0,
                above_edge_start_min_actual_z=0.0,
                above_edge_trajectory_variants="standard",
                above_edge_start_gripper_profile="open",
                above_edge_terminal_hold_jitter=0,
                task=config.teacher.task_prompt,
                include_camera3_duplicate=False,
                capture_render_replay=False,
                capture_fps=12,
                reset_home_qpos=reset_home_qpos,
                exact_start_pose=True,
                min_gripper_floor_clearance_m=(
                    config.teacher.min_gripper_floor_clearance_m
                ),
                record_dataset_frames=False,
            )
            if bool(result.get("success")):
                result["candidate_rank"] = rank
                result["candidate_count"] = len(ranked)
                result["candidate_failures"] = failures
                return result
            failures.append(
                {
                    "rank": rank,
                    "candidate_mode": meta.get("candidate_mode"),
                    "reason": result.get("reason"),
                    "task_success": result.get("task_success"),
                    "lift_target_reached": result.get("lift_target_reached"),
                    "final_info": result.get("final_info"),
                    "pre_close_static_edge_error": result.get(
                        "pre_close_static_edge_error"
                    ),
                    "pre_close_jaw_capture_geometry": result.get(
                        "pre_close_jaw_capture_geometry"
                    ),
                    "pre_close_cube_face_normal_parallel_error_deg": result.get(
                        "pre_close_cube_face_normal_parallel_error_deg"
                    ),
                    "q_edge": result.get("q_edge"),
                    "q_above": result.get("q_above"),
                    "start_target_pose": result.get("start_target_pose"),
                    "pre_close_target_pose": result.get("pre_close_target_pose"),
                    "pre_close_qpos": result.get("pre_close_qpos"),
                    "pre_close_q_edge_error_l2": result.get(
                        "pre_close_q_edge_error_l2"
                    ),
                    "close_alignment_gate": (
                        result.get("best_meta") or {}
                    ).get("camera2_top_contact_close_alignment_gate"),
                    "wrist_roll_delta_gate": (
                        result.get("best_meta") or {}
                    ).get("wrist_roll_delta_gate"),
                    "gripper_floor_clearance_gate": result.get(
                        "gripper_floor_clearance_gate"
                    ),
                    "floor_clearance_constructive_refine": (
                        result.get("best_meta") or {}
                    ).get("floor_clearance_constructive_refine"),
                }
            )
    finally:
        teacher_exporter._restore_sim_state(env, snapshot)
    return {
        "success": False,
        "reason": (
            Counter(str(row.get("reason") or "unknown") for row in failures)
            .most_common(1)[0][0]
            if failures
            else "no_candidate"
        ),
        "candidate_count": len(candidates),
        "candidate_failures": failures,
    }


def _probe_point(
    *,
    env: Any,
    renderers: dict[str, Any],
    config: WorkspaceProbeConfig,
    point: dict[str, Any],
    point_index: int,
    base_xy: tuple[float, float],
    base_rotation_xy: np.ndarray,
    evaluate_dataset_contract: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed = int(point.get("seed_override") or (config.seed_base + point_index))
    world_xy = base_relative_to_world_xy(
        base_xy,
        base_rotation_xy,
        (float(point["base_x_m"]), float(point["base_y_m"])),
    )
    env.reset(seed=seed)
    teacher_exporter._set_target_object_xy(env, world_xy)
    teacher_exporter._set_target_object_yaw(env, float(point["yaw_deg"]))
    home_qpos = np.asarray(config.home_qpos, dtype=np.float32)
    teacher_exporter._set_qpos(env, home_qpos)
    start_snapshot = teacher_exporter._snapshot_sim_state(env)
    start_visibility = teacher_exporter._policy_camera_visibility(env, renderers)

    raw_candidates = teacher_exporter._make_full_grip_teacher_targets_for_skill(
        env,
        skill_mode=config.teacher.skill_mode,
        min_floor_clearance_m=config.teacher.min_gripper_floor_clearance_m,
    )
    preflight_passed = teacher_exporter._has_success_contract_fixed_jaw_candidate(
        env,
        raw_candidates,
        edge_contact_xy_success_threshold=(
            config.teacher.edge_contact_xy_threshold_m
        ),
        edge_contact_parallel_success_threshold_deg=(
            config.teacher.edge_parallel_threshold_deg
        ),
        min_gripper_floor_clearance_m=(
            config.teacher.min_gripper_floor_clearance_m
        ),
        close_steps=config.teacher.close_steps,
    )

    physical_result: dict[str, Any]
    if preflight_passed:
        teacher_exporter._restore_sim_state(env, start_snapshot)
        physical_result = _run_teacher_candidate_set(
            env=env,
            renderers=renderers,
            candidates=raw_candidates,
            config=config,
            seed=seed,
            point_index=point_index,
            reset_home_qpos=home_qpos,
            close_alignment_gate_mode="geometry_only",
            close_alignment_limits=None,
        )
    else:
        physical_result = {"success": False, "reason": "geometry_preflight_failed"}

    physical_metrics = physical_outcome_metrics(
        physical_result,
        target_lift_height_m=config.teacher.lift_target_height_m,
        operational_lift_height_m=config.teacher.operational_lift_height_m,
    )
    dataset_result: dict[str, Any] = {
        "success": False,
        "reason": "physical_workspace_failed",
        "candidate_count": 0,
    }
    if evaluate_dataset_contract and bool(physical_metrics["target_lift_success"]):
        teacher_exporter._restore_sim_state(env, start_snapshot)
        policy_candidates = (
            teacher_exporter._filter_fixed_jaw_move_candidates_in_policy_view(
                env,
                renderers=renderers,
                candidates=raw_candidates,
                move_target_z_offset=config.teacher.move_target_z_offset_m,
            )
        )
        if not bool(start_visibility["camera1"].get("visible")):
            dataset_result = {
                "success": False,
                "reason": "camera1_not_visible_at_hardware_home",
                "candidate_count": len(policy_candidates),
            }
        elif not policy_candidates:
            dataset_result = {
                "success": False,
                "reason": "no_policy_view_candidate",
                "candidate_count": 0,
            }
        else:
            teacher_exporter._restore_sim_state(env, start_snapshot)
            dataset_result = _run_teacher_candidate_set(
                env=env,
                renderers=renderers,
                candidates=policy_candidates,
                config=config,
                seed=seed,
                point_index=point_index,
                reset_home_qpos=home_qpos,
                close_alignment_gate_mode=config.teacher.camera2_gate_mode,
                close_alignment_limits=config.teacher.camera2_limits(),
            )

    radius = math.hypot(float(point["base_x_m"]), float(point["base_y_m"]))
    angle = math.degrees(
        math.atan2(float(point["base_y_m"]), float(point["base_x_m"]))
    )
    record = {
        **point,
        "seed": seed,
        "world_x_m": world_xy[0],
        "world_y_m": world_xy[1],
        "radius_from_base_m": radius,
        "angle_from_base_deg": angle,
        "initial_camera1_visible": bool(
            start_visibility["camera1"].get("visible")
        ),
        "initial_camera1_centroid": start_visibility["camera1"].get(
            "normalized_centroid"
        ),
        "initial_camera2_visible": bool(
            start_visibility["camera2"].get("visible")
        ),
        "raw_candidate_count": len(raw_candidates),
        "preflight_passed": bool(preflight_passed),
        "physical_teacher_contract_success": bool(
            physical_metrics["teacher_geometry_contract_success"]
        ),
        "physical_grasp_success": bool(physical_metrics["grasp_success"]),
        "physical_operational_lift_success": bool(
            physical_metrics["operational_lift_success"]
        ),
        "physical_target_lift_success": bool(
            physical_metrics["target_lift_success"]
        ),
        "physical_max_grasped_lift_height_m": physical_metrics[
            "max_grasped_lift_height_m"
        ],
        "physical_success": bool(physical_metrics["target_lift_success"]),
        "physical_failure_reason": (
            None
            if bool(physical_metrics["target_lift_success"])
            else physical_result.get("reason")
        ),
        "physical_result": physical_result,
        "dataset_contract_success": bool(dataset_result.get("success")),
        "dataset_contract_failure_reason": (
            None
            if bool(dataset_result.get("success"))
            else dataset_result.get("reason")
        ),
        "dataset_contract_result": dataset_result,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return record


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _persist_records(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_plot(
    records: list[dict[str, Any]],
    *,
    output: Path,
    config: WorkspaceProbeConfig,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    panels = (
        ("Physical outcome", "physical_success"),
        ("Dataset contract", "dataset_contract_success"),
    )
    x_values = [float(row["base_x_m"]) for row in records]
    y_values = [float(row["base_y_m"]) for row in records]
    x_margin = 0.02
    y_margin = 0.02
    x_limits = (min(x_values) - x_margin, max(x_values) + x_margin)
    y_limits = (min(y_values) - y_margin, max(y_values) + y_margin)
    for axis, (title, success_key) in zip(axes, panels, strict=True):
        for row in records:
            success = bool(row.get(success_key))
            preflight = bool(row.get("preflight_passed"))
            if success:
                color = "#16803c"
            elif success_key == "physical_success" and bool(
                row.get("physical_operational_lift_success")
            ):
                color = "#0891b2"
            elif success_key == "physical_success" and bool(
                row.get("physical_grasp_success")
            ):
                color = "#7c3aed"
            else:
                color = "#e59f23" if preflight else "#cf3e48"
            axis.scatter(
                float(row["base_x_m"]),
                float(row["base_y_m"]),
                c=color,
                marker="s",
                s=44,
                edgecolors="white",
                linewidths=0.25,
            )
            if not bool(row.get("initial_camera1_visible")):
                axis.scatter(
                    float(row["base_x_m"]),
                    float(row["base_y_m"]),
                    c="#111827",
                    marker="x",
                    s=20,
                    linewidths=0.8,
                )
        axis.scatter(0.0, 0.0, c="#2563eb", marker="*", s=180, label="base")
        for radius in (0.1, 0.2, 0.3, 0.4):
            axis.add_patch(
                plt.Circle(
                    (0.0, 0.0),
                    radius,
                    fill=False,
                    color="#94a3b8",
                    linewidth=0.6,
                    linestyle="--",
                )
            )
        axis.set_title(title)
        axis.set_xlabel("base-forward X (m)")
        axis.set_ylabel("base-left Y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.grid(alpha=0.18)
    figure.suptitle(
        "SO101 base-relative grasp workspace\n"
        "green=target lift, cyan=operational lift, purple=grasp only, "
        "amber=preflight only, red=no preflight, x=camera1 invisible"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    config, config_sha256 = _load_config(config_path)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT / "_workspace" / "so101_workspace_probes" / config.name
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "points.jsonl"
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != config_sha256:
            raise RuntimeError(
                "existing workspace probe used a different config; choose a new output directory"
            )

    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")
    (
        env,
        renderers,
        base_xy,
        base_rotation_xy,
        base_contract,
        camera_rig_sha256,
    ) = _make_runtime(config)
    all_points: list[dict[str, Any]] = []
    configured_points: list[dict[str, Any]] = []
    exact_points: list[dict[str, Any]] = []
    try:
        configured_points = grid_points(config)
        exact_points = _exact_points(args.base_xy, config=config)
        all_points = configured_points + exact_points
        manifest = {
            "config": str(config_path),
            "config_sha256": config_sha256,
            "camera_rig_sha256": camera_rig_sha256,
            "base_contract": base_contract,
            "total_requested_points": len(configured_points),
            "exact_requested_points": len(exact_points),
            "evaluate_dataset_contract": bool(
                config.evaluate_dataset_contract and not args.physical_only
            ),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        records = [
            annotate_physical_outcomes(
                row,
                target_lift_height_m=config.teacher.lift_target_height_m,
                operational_lift_height_m=(
                    config.teacher.operational_lift_height_m
                ),
            )
            for row in _load_records(records_path)
        ]
        if records:
            _persist_records(records_path, records)
        completed = {str(row["point_id"]) for row in records}
        requested_ids = set(args.point_id)
        if exact_points:
            requested_ids.update(str(point["point_id"]) for point in exact_points)
        known_ids = {str(point["point_id"]) for point in all_points}
        unknown_ids = sorted(requested_ids - known_ids)
        if unknown_ids:
            raise ValueError(f"unknown --point-id values: {unknown_ids}")
        pending = [
            point
            for point in all_points
            if point["point_id"] not in completed
            and (not requested_ids or point["point_id"] in requested_ids)
        ]
        if int(args.max_points) > 0:
            pending = pending[: int(args.max_points)]
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "base_contract": base_contract,
                    "completed": len(records),
                    "pending_this_run": len(pending),
                    "total": len(all_points),
                },
                indent=2,
            ),
            flush=True,
        )

        indexed_pending = [(point, all_points.index(point)) for point in pending]
        evaluate_dataset_contract = bool(
            config.evaluate_dataset_contract and not args.physical_only
        )

        def record_result(
            stream: Any, record: dict[str, Any], run_index: int
        ) -> None:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            records.append(record)
            print(
                f"[{run_index}/{len(pending)}] {record['point_id']} "
                f"xy=({record['base_x_m']:+.3f},{record['base_y_m']:+.3f}) "
                f"preflight={record['preflight_passed']} "
                f"physical={record['physical_success']} "
                f"dataset={record['dataset_contract_success']} "
                f"elapsed={record['elapsed_seconds']:.2f}s",
                flush=True,
            )

        with records_path.open("a", encoding="utf-8") as stream:
            if int(args.workers) == 1:
                for run_index, (point, point_index) in enumerate(
                    indexed_pending, start=1
                ):
                    record = _probe_point(
                        env=env,
                        renderers=renderers,
                        config=config,
                        point=point,
                        point_index=point_index,
                        base_xy=base_xy,
                        base_rotation_xy=base_rotation_xy,
                        evaluate_dataset_contract=evaluate_dataset_contract,
                    )
                    record_result(stream, record, run_index)
            else:
                # The parent runtime is no longer needed. Closing it before
                # spawning keeps renderer/GL state process-local.
                _close_runtime(env, renderers)
                env = None
                renderers = {}
                context = multiprocessing.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=int(args.workers),
                    mp_context=context,
                    initializer=_worker_init,
                    initargs=(
                        config.model_dump_json(),
                        evaluate_dataset_contract,
                    ),
                ) as pool:
                    futures = [
                        pool.submit(_worker_probe, payload)
                        for payload in indexed_pending
                    ]
                    for run_index, future in enumerate(
                        concurrent.futures.as_completed(futures), start=1
                    ):
                        record_result(stream, future.result(), run_index)
    finally:
        if env is not None:
            _close_runtime(env, renderers)

    summary = {
        **summarize_workspace_records(records),
        "config": str(config_path),
        "config_sha256": config_sha256,
        "base_contract": manifest["base_contract"],
        "completed_points": len(records),
        "total_requested_points": len(configured_points),
        "exact_requested_points": len(exact_points),
        "complete": (
            not exact_points
            and len(records) == len(configured_points)
        ),
        "estimated_physical_area_m2": sum(
            float(row.get("point_cell_area_m2", 0.0))
            for row in records
            if bool(row.get("physical_success"))
        ),
        "estimated_dataset_contract_area_m2": sum(
            float(row.get("point_cell_area_m2", 0.0))
            for row in records
            if bool(row.get("dataset_contract_success"))
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    successful_cells = successful_workspace_cells(records)
    camera_bin_counts = Counter(
        str(cell["camera1_grid_bin"])
        for cell in successful_cells
        if cell["camera1_grid_bin"] is not None
    )
    workspace_catalog = {
        "format": "so101_grasp_workspace_catalog_v1",
        "catalog_id": f"{config.name}_strict_dataset_contract",
        "source_config": str(config_path),
        "source_config_sha256": config_sha256,
        "base_contract": manifest["base_contract"],
        "camera_rig_config": config.camera_rig_config,
        "home_qpos": [float(value) for value in config.home_qpos],
        "object_half_size_m": float(config.object_half_size_m),
        "object_color": config.object_color,
        "success_contract": {
            "grasp_required": True,
            "lift_height_m": float(config.teacher.lift_target_height_m),
            "terminal_hold_steps": int(config.teacher.terminal_hold_steps),
            "minimum_gripper_floor_clearance_m": float(
                config.teacher.min_gripper_floor_clearance_m
            ),
            "dataset_contract_required": True,
        },
        "sampling_contract": {
            "strategy": "sample_cells_by_uniform_area_weight",
            "position": "probed_cell_center",
            "object_yaw": "use_each_cell_object_yaw_deg",
            "camera_grid_size": 4,
            "note": (
                "Do not drop object_yaw_deg: feasibility is coupled to the "
                "contacted cube face at each base-relative position."
            ),
        },
        "cell_count": len(successful_cells),
        "total_estimated_area_m2": sum(
            float(cell["cell_area_m2"]) for cell in successful_cells
        ),
        "camera1_bin_counts": dict(sorted(camera_bin_counts.items())),
        "cells": successful_cells,
    }
    (output_dir / "successful_workspace_catalog.json").write_text(
        json.dumps(workspace_catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_plot(
        records,
        output=output_dir / "workspace_map.png",
        config=config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
