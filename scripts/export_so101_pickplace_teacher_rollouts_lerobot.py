#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from export_so101_teacher_rollouts_lerobot import (
    _lerobot_features,
    _open_gripper_value,
    _render_camera,
    audit_lerobot_dataset,
)
from physical_ai_agent.sim.so101_live_viewer import _cartesian_error_controller_action
from train_so101_wrist_ego_picklift_policy import sweep_until_visible
from train_so101_wrist_ego_visual_servo import (
    WristEgoServoConfig,
    _current_qpos,
    _grasp_candidate_specs,
    _make_policy_renderers,
    _make_teacher_renderers,
    _restore_sim_state,
    _snapshot_sim_state,
    _solve_pregrasp_qpos_variant,
    object_visible_to_teacher,
)


TASK = "Pick up the small red cube and place it on the blue circle."

OBJECT_SHAPE_PRESETS: dict[str, tuple[float, float, float]] = {
    "cube_small": (0.0125, 0.0125, 0.0125),
    "cube_medium": (0.0150, 0.0150, 0.0150),
    "long_bar": (0.0260, 0.0090, 0.0090),
    "wide_flat": (0.0200, 0.0160, 0.0090),
    "thin_flat": (0.0180, 0.0120, 0.0080),
    "tall_block": (0.0100, 0.0100, 0.0200),
}
OBJECT_PHYSICS_PRESETS: dict[str, dict[str, Any]] = {
    "cube_small": {"mass": 0.0040, "object_friction": (1.05, 0.010, 0.0002)},
    "cube_medium": {"mass": 0.0055, "object_friction": (1.05, 0.010, 0.0002)},
    "long_bar": {"mass": 0.0045, "object_friction": (1.35, 0.018, 0.0004)},
    "wide_flat": {"mass": 0.0040, "object_friction": (1.35, 0.018, 0.0004)},
    "thin_flat": {"mass": 0.0035, "object_friction": (1.45, 0.020, 0.0004)},
    "tall_block": {"mass": 0.0045, "object_friction": (1.20, 0.014, 0.0003)},
}
FINGER_PAD_FRICTION = (1.60, 0.020, 0.0005)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export privileged SO101 pick-and-place teacher rollouts to a local LeRobotDataset."
    )
    parser.add_argument("--root", type=Path, default=Path("_workspace/so101_lerobot/teacher_pickplace_smolvla"))
    parser.add_argument("--repo-id", default="physical-ai-agent/so101-pickplace-teacher")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=99000)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--approach-steps", type=int, default=34)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--close-steps", type=int, default=42)
    parser.add_argument("--lift-steps", type=int, default=70)
    parser.add_argument("--transport-steps", type=int, default=160)
    parser.add_argument("--lower-steps", type=int, default=70)
    parser.add_argument("--release-steps", type=int, default=24)
    parser.add_argument("--settle-after-release-steps", type=int, default=56)
    parser.add_argument(
        "--recovery-steps",
        type=int,
        default=0,
        help="Optional off-nominal recovery frames before the nominal approach. Default 0 preserves legacy data.",
    )
    parser.add_argument("--recovery-joint-std", type=float, default=0.04)
    parser.add_argument(
        "--approach-mode",
        choices=["joint", "side-slide"],
        default="joint",
        help="joint interpolates home to pregrasp; side-slide adds a low lateral slide into the final grasp pose.",
    )
    parser.add_argument("--side-approach-retreat", type=float, default=0.075)
    parser.add_argument("--min-success-frames", type=int, default=0)
    parser.add_argument("--max-success-dist", type=float, default=0.035)
    parser.add_argument("--max-attempts", type=int, default=0, help="0 means episodes * 10.")
    parser.add_argument(
        "--grasp-filter",
        choices=["all", "side-low", "left-right", "left-right-side-low"],
        default="all",
        help="Restrict teacher grasp candidates. left-right keeps the jaw axis left/right instead of vertical-looking grips.",
    )
    parser.add_argument("--allow-sticky-grasp", action="store_true")
    parser.add_argument("--no-camera3-duplicate", action="store_true")
    parser.add_argument(
        "--object-shapes",
        default="cube_small",
        help=(
            "Comma-separated object shape presets. Aliases: cube_only, diverse_boxes. "
            f"Available presets: {','.join(OBJECT_SHAPE_PRESETS)}"
        ),
    )
    args = parser.parse_args()

    report = export_pickplace_teacher_rollouts(
        root=args.root,
        repo_id=args.repo_id,
        episodes=args.episodes,
        seed=args.seed,
        fps=args.fps,
        width=args.width,
        height=args.height,
        use_videos=args.use_videos,
        overwrite=args.overwrite,
        approach_steps=args.approach_steps,
        settle_steps=args.settle_steps,
        close_steps=args.close_steps,
        lift_steps=args.lift_steps,
        transport_steps=args.transport_steps,
        lower_steps=args.lower_steps,
        release_steps=args.release_steps,
        settle_after_release_steps=args.settle_after_release_steps,
        recovery_steps=args.recovery_steps,
        recovery_joint_std=args.recovery_joint_std,
        approach_mode=args.approach_mode,
        side_approach_retreat=args.side_approach_retreat,
        min_success_frames=args.min_success_frames,
        max_success_dist=args.max_success_dist,
        max_attempts=args.max_attempts,
        grasp_filter=args.grasp_filter,
        sticky_grasp=args.allow_sticky_grasp,
        include_camera3_duplicate=not args.no_camera3_duplicate,
        object_shapes=args.object_shapes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def export_pickplace_teacher_rollouts(
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
    approach_steps: int,
    settle_steps: int,
    close_steps: int,
    lift_steps: int,
    transport_steps: int,
    lower_steps: int,
    release_steps: int,
    settle_after_release_steps: int,
    recovery_steps: int,
    recovery_joint_std: float,
    approach_mode: str,
    side_approach_retreat: float,
    min_success_frames: int,
    max_success_dist: float,
    max_attempts: int,
    grasp_filter: str,
    sticky_grasp: bool,
    include_camera3_duplicate: bool = True,
    object_shapes: str = "cube_small",
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

    config = WristEgoServoConfig(width=width, height=height)
    env = _make_pickplace_env()
    _install_cube_pose_alias(env)
    policy_renderers = _make_policy_renderers(env, config)
    _add_top_down_policy_renderer(env, policy_renderers, config)
    teacher_renderers = _make_teacher_renderers(env, config)
    action_space_low = np.asarray(env.action_space.low, dtype=np.float32).copy()
    action_space_high = np.asarray(env.action_space.high, dtype=np.float32).copy()
    shape_sequence = _resolve_object_shape_sequence(object_shapes)
    exported = 0
    attempted = 0
    skipped: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []

    try:
        candidate_seed = seed
        attempt_limit = int(max_attempts) if int(max_attempts) > 0 else episodes * 10
        while exported < episodes and attempted < attempt_limit:
            attempted += 1
            shape = shape_sequence[(candidate_seed - seed) % len(shape_sequence)]
            _apply_object_shape(env, shape)
            env.reset(seed=candidate_seed)
            _apply_object_shape(env, shape)
            _install_cube_pose_alias(env)
            candidate_seed += 1
            teacher_visible = object_visible_to_teacher(env, teacher_renderers, config=config)
            visible, search_steps = sweep_until_visible(env, policy_renderers, max_sweeps=config.max_sweeps)
            teacher_visible = teacher_visible or object_visible_to_teacher(env, teacher_renderers, config=config)
            if not visible:
                skipped.append({"seed": candidate_seed - 1, "reason": "not_visible_after_sweep"})
                print(
                    f"[so101-pickplace] skipped seed={candidate_seed - 1} "
                    f"shape={shape['name']} reason=not_visible_after_sweep",
                    flush=True,
                )
                continue
            candidates = _make_pickplace_grasp_targets(env, grasp_filter=grasp_filter)
            if not candidates:
                skipped.append({"seed": candidate_seed - 1, "reason": "no_graspable_teacher_candidate"})
                print(
                    f"[so101-pickplace] skipped seed={candidate_seed - 1} "
                    f"shape={shape['name']} reason=no_graspable_teacher_candidate",
                    flush=True,
                )
                continue
            best = max(candidates, key=lambda item: float(item["meta"].get("score", -1e9)))
            summary = _write_pickplace_episode(
                dataset=dataset,
                env=env,
                renderers=policy_renderers,
                q_open=np.asarray(best["q_open"], dtype=np.float32),
                seed=candidate_seed - 1,
                search_steps=search_steps,
                teacher_visible=teacher_visible,
                best_meta=dict(best["meta"]),
                approach_steps=approach_steps,
                settle_steps=settle_steps,
                close_steps=close_steps,
                lift_steps=lift_steps,
                transport_steps=transport_steps,
                lower_steps=lower_steps,
                release_steps=release_steps,
                settle_after_release_steps=settle_after_release_steps,
                recovery_steps=recovery_steps,
                recovery_joint_std=recovery_joint_std,
                approach_mode=approach_mode,
                side_approach_retreat=side_approach_retreat,
                min_success_frames=min_success_frames,
                max_success_dist=max_success_dist,
                sticky_grasp=sticky_grasp,
                include_camera3_duplicate=include_camera3_duplicate,
                object_shape=shape,
            )
            if summary["success"]:
                dataset.save_episode()
                exported += 1
                episode_summaries.append(summary)
                print(
                    f"[so101-pickplace] exported {exported}/{episodes} "
                    f"seed={summary['seed']} frames={summary['frames']} "
                    f"placed_dist={summary['final_info'].get('obj_to_target_dist'):.4f} "
                    f"mode={summary['best_meta'].get('mode')} "
                    f"shape={summary['object_shape'].get('name')}",
                    flush=True,
                )
            else:
                dataset.clear_episode_buffer()
                skipped.append({"seed": candidate_seed - 1, "reason": "teacher_replay_failed", **summary})
                print(
                    f"[so101-pickplace] skipped seed={candidate_seed - 1} "
                    f"shape={summary['object_shape'].get('name')} reason=teacher_replay_failed "
                    f"raw_success={summary['raw_success']} "
                    f"placed_dist={summary['final_info'].get('obj_to_target_dist'):.4f} "
                    f"mode={summary['best_meta'].get('candidate_mode')}",
                    flush=True,
                )
    finally:
        for renderer in [*policy_renderers.values(), *teacher_renderers.values()]:
            renderer.close()
        env.close()

    dataset.finalize()
    if exported > 0:
        audit = audit_lerobot_dataset(
            root=root,
            repo_id=repo_id,
            features=features,
            action_space_low=action_space_low,
            action_space_high=action_space_high,
        )
    else:
        audit = {"status": "skipped_empty_dataset", "dataset_len": 0, "num_episodes": 0}
    report = {
        "operation": "export_so101_pickplace_teacher_rollouts_lerobot",
        "root": str(root),
        "repo_id": repo_id,
        "task": TASK,
        "requested_episodes": episodes,
        "exported_episodes": exported,
        "attempted_seeds": attempted,
        "max_attempts": int(max_attempts) if int(max_attempts) > 0 else int(episodes * 10),
        "fps": fps,
        "width": width,
        "height": height,
        "use_videos": use_videos,
        "teacher_timing": {
            "approach_steps": int(approach_steps),
            "settle_steps": int(settle_steps),
            "close_steps": int(close_steps),
            "lift_steps": int(lift_steps),
            "transport_steps": int(transport_steps),
            "lower_steps": int(lower_steps),
            "release_steps": int(release_steps),
            "settle_after_release_steps": int(settle_after_release_steps),
            "recovery_steps": int(recovery_steps),
            "recovery_joint_std": float(recovery_joint_std),
            "approach_mode": str(approach_mode),
            "side_approach_retreat": float(side_approach_retreat),
            "min_success_frames": int(min_success_frames),
            "max_success_dist": float(max_success_dist),
        },
        "grasp_filter": str(grasp_filter),
        "object_shapes": [_object_shape_report(shape) for shape in shape_sequence],
        "teacher_privilege": {
            "sticky_grasp_enabled": bool(sticky_grasp),
            "recovery_or_off_nominal_states": int(recovery_steps) > 0,
            "reason": "Default is false. When true, cube pose is privilegedly attached after contact and should not be used for visual-policy training data.",
        },
        "camera3_duplicate": {
            "enabled": bool(include_camera3_duplicate),
            "source": "wrist_cam",
            "reason": "lerobot/svla_so100_pickplace uses observation.images.top and observation.images.wrist; SmolVLA preprocessing maps top->camera1 and wrist->camera2. camera3 duplicates wrist only when a 3-camera base config is required.",
        },
        "feature_mapping": {
            "observation.images.camera1": "top_down",
            "observation.images.camera2": "wrist_cam",
            **({"observation.images.camera3": "wrist_cam duplicate"} if include_camera3_duplicate else {}),
            "observation.state": "SO101 qpos/control state",
            "action": "SO101 qpos target action",
            "task": TASK,
        },
        "official_camera_contract": {
            "dataset": "lerobot/svla_so100_pickplace",
            "dataset_features": ["observation.images.top", "observation.images.wrist"],
            "rename_map": {
                "observation.images.top": "observation.images.camera1",
                "observation.images.wrist": "observation.images.camera2",
            },
            "local_verification": "HF cached SmolVLA SO101 artifact policy_preprocessor records the same rename map.",
        },
        "episodes": episode_summaries,
        "skipped": skipped,
        "audit": audit,
    }
    report_path = root / "so101_lerobot_export_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _make_pickplace_env() -> Any:
    from so101_nexus_core.config import PickAndPlaceConfig
    from so101_nexus_mujoco.pick_and_place import PickAndPlaceEnv

    config = PickAndPlaceConfig(
        cube_colors="red",
        target_colors="blue",
        cube_half_size=0.0125,
        cube_mass=0.01,
        target_disc_radius=0.05,
        min_cube_target_separation=0.05,
        goal_thresh=0.035,
        spawn_min_radius=0.10,
        spawn_max_radius=0.24,
        spawn_angle_half_range_deg=75.0,
        max_episode_steps=512,
    )
    return PickAndPlaceEnv(config=config, render_mode=None, robot_init_qpos_noise=0.02)


def _resolve_object_shape_sequence(value: str) -> list[dict[str, Any]]:
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    if not names:
        names = ["cube_small"]
    if names == ["cube_only"]:
        names = ["cube_small"]
    if names == ["diverse_boxes"]:
        names = ["cube_small", "long_bar", "wide_flat", "thin_flat", "tall_block"]
    shapes = []
    for name in names:
        if name not in OBJECT_SHAPE_PRESETS:
            raise ValueError(f"unknown object shape preset {name!r}; choose from {sorted(OBJECT_SHAPE_PRESETS)}")
        half_extents = np.asarray(OBJECT_SHAPE_PRESETS[name], dtype=float)
        shapes.append(
            {
                "name": name,
                "geom_type": "box",
                "half_extents": half_extents,
                **OBJECT_PHYSICS_PRESETS.get(name, {}),
            }
        )
    return shapes


def _apply_object_shape(env: Any, shape: dict[str, Any]) -> None:
    import mujoco

    half_extents = np.asarray(shape["half_extents"], dtype=float)
    if half_extents.shape != (3,):
        raise ValueError(f"object half_extents must be length 3, got {half_extents}")
    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data
    geom_id = int(unwrapped._obj_geom_id)
    model.geom_type[geom_id] = int(mujoco.mjtGeom.mjGEOM_BOX)
    model.geom_size[geom_id, :3] = half_extents
    body_id = int(model.geom_bodyid[geom_id])
    mass = shape.get("mass")
    unwrapped.cube_half_size = float(half_extents[2])
    if hasattr(unwrapped, "config"):
        unwrapped.config.cube_half_size = float(half_extents[2])
    try:
        mujoco.mj_setConst(model, data)
    except Exception:
        pass
    object_friction = shape.get("object_friction")
    if object_friction is not None:
        model.geom_friction[geom_id, :3] = np.asarray(object_friction, dtype=float)
    if mass is not None:
        _set_box_body_mass_and_inertia(model, body_id=body_id, half_extents=half_extents, mass=float(mass))
    _set_finger_pad_friction(model)
    addr = int(unwrapped._cube_qpos_addr)
    data.qpos[addr + 2] = float(half_extents[2])
    data.qvel[:] = 0.0
    unwrapped._initial_obj_z = float(half_extents[2])
    mujoco.mj_forward(model, data)


def _set_box_body_mass_and_inertia(model: Any, *, body_id: int, half_extents: np.ndarray, mass: float) -> None:
    model.body_mass[body_id] = float(max(1e-5, mass))
    full_extents = np.asarray(half_extents, dtype=float) * 2.0
    x, y, z = [float(value) for value in full_extents]
    inertia = np.asarray(
        [
            mass * (y * y + z * z) / 12.0,
            mass * (x * x + z * z) / 12.0,
            mass * (x * x + y * y) / 12.0,
        ],
        dtype=float,
    )
    model.body_inertia[body_id, :3] = np.maximum(inertia, 1e-8)


def _set_finger_pad_friction(model: Any) -> None:
    import mujoco

    for geom_name in ("static_finger_pad", "moving_finger_pad"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            model.geom_friction[geom_id, :3] = np.asarray(FINGER_PAD_FRICTION, dtype=float)


def _object_shape_report(shape: dict[str, Any]) -> dict[str, Any]:
    half_extents = np.asarray(shape["half_extents"], dtype=float)
    return {
        "name": str(shape["name"]),
        "geom_type": str(shape.get("geom_type", "box")),
        "half_extents_m": [float(value) for value in half_extents],
        "full_extents_m": [float(value * 2.0) for value in half_extents],
        "mass_kg": float(shape["mass"]) if "mass" in shape else None,
        "object_friction": [float(value) for value in shape["object_friction"]]
        if "object_friction" in shape
        else None,
        "finger_pad_friction": [float(value) for value in FINGER_PAD_FRICTION],
    }


def _install_cube_pose_alias(env: Any) -> None:
    # Existing grasp IK helpers use the PickLift `_get_target_pose` name.
    env.unwrapped._get_target_pose = env.unwrapped._get_cube_pose


def _make_pickplace_grasp_targets(env: Any, *, grasp_filter: str = "all") -> list[dict[str, Any]]:
    import mujoco

    snapshot = _snapshot_sim_state(env)
    successes_by_mode: dict[str, tuple[float, np.ndarray, dict[str, Any]]] = {}
    specs = [*_grasp_candidate_specs(env), *_flat_friendly_pickplace_grasp_specs(env, start_index=1000)]
    for spec in specs:
        if not _candidate_matches_grasp_filter(spec, grasp_filter):
            continue
        _restore_sim_state(env, snapshot)
        try:
            q_open = _solve_pregrasp_qpos_variant(env, spec)
            pad_meta = _finger_pad_axis_meta(env, q_open)
            q_close = q_open.copy()
            q_close[-1] = float(env.action_space.low[-1])
            info: dict[str, Any] = {}
            grasped_steps = 0
            max_lift_height = 0.0
            preclose_contact = False
            for step in range(150):
                if step < 42:
                    action = q_open
                elif step < 86:
                    action = q_close
                else:
                    action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=float)
                    action[-1] = q_close[-1]
                _obs, _reward, terminated, truncated, info = env.step(action)
                if step < 42 and _finger_object_contact(env):
                    preclose_contact = True
                    break
                if bool(info.get("is_grasped", False)):
                    grasped_steps += 1
                max_lift_height = max(max_lift_height, float(info.get("lift_height", 0.0)))
                if terminated or truncated:
                    break
            if not preclose_contact and grasped_steps > 0 and max_lift_height > 0.035:
                score = (
                    max_lift_height * 8.0
                    - float(info.get("tcp_to_obj_dist", 1.0))
                    - 0.0005 * float(spec["candidate_index"])
                    - 3.0 * max(0.0, float(spec["z_offset"]) - float(env.unwrapped.cube_half_size) * 0.25)
                )
                meta = {
                    "mode": str(spec["grasp_mode"]),
                    "candidate_mode": str(spec["mode"]),
                    "axis": [float(value) for value in spec["axis"]],
                    "gap": float(spec["gap"]),
                    "z_offset": float(spec["z_offset"]),
                    "score": score,
                    "candidate_index": int(spec["candidate_index"]),
                    "candidate_attempts": len(specs),
                    "grasped_steps": int(grasped_steps),
                    "max_lift_height": float(max_lift_height),
                    **pad_meta,
                }
                grasp_mode = str(spec["grasp_mode"])
                if str(spec["mode"]).startswith("left_right"):
                    grasp_mode = "left_right"
                current = successes_by_mode.get(grasp_mode)
                candidate = (score, q_open.astype(float), meta)
                if current is None or score > current[0]:
                    successes_by_mode[grasp_mode] = candidate
        except Exception:
            continue
    _restore_sim_state(env, snapshot)
    return [
        {"q_open": q_open, "meta": meta}
        for _score, q_open, meta in sorted(successes_by_mode.values(), key=lambda item: -item[0])
    ]


def _candidate_matches_grasp_filter(spec: dict[str, Any], grasp_filter: str) -> bool:
    mode = str(spec["mode"])
    if grasp_filter == "all":
        return True
    if grasp_filter == "side-low":
        return mode.endswith("_side")
    if grasp_filter == "left-right":
        return mode.startswith("left_right")
    if grasp_filter == "left-right-side-low":
        return mode == "left_right_side"
    raise ValueError(f"unknown grasp_filter: {grasp_filter}")


def _flat_friendly_pickplace_grasp_specs(env: Any, *, start_index: int) -> list[dict[str, Any]]:
    axes = [
        ("front", "front_back", np.asarray([1.0, 0.0, 0.0], dtype=float)),
        ("front", "left_right", np.asarray([0.0, 1.0, 0.0], dtype=float)),
        ("diagonal", "diag_front", np.asarray([1.0, -1.0, 0.0], dtype=float)),
        ("diagonal", "diag_back", np.asarray([1.0, 1.0, 0.0], dtype=float)),
    ]
    specs = []
    index = int(start_index)
    for grasp_mode, name, axis in axes:
        axis = axis / max(1e-6, float(np.linalg.norm(axis)))
        for gap, z_offset in (
            (0.044, 0.0015),
            (0.052, 0.0025),
            (0.060, 0.0035),
            (0.068, 0.0045),
            (0.076, 0.0025),
            (0.084, 0.0035),
        ):
            specs.append(
                {
                    "candidate_index": index,
                    "mode": f"{name}_flat_low_{gap:.3f}_{z_offset:.4f}",
                    "grasp_mode": grasp_mode,
                    "axis": axis,
                    "gap": gap,
                    "z_offset": z_offset,
                    "open_value": _open_gripper_value(env),
                }
            )
            index += 1
    return specs


def _finger_pad_axis_meta(env: Any, qpos: np.ndarray) -> dict[str, Any]:
    import mujoco

    snapshot = _snapshot_sim_state(env)
    try:
        unwrapped = env.unwrapped
        model = unwrapped.model
        data = unwrapped.data
        low = np.asarray(env.action_space.low, dtype=float)
        high = np.asarray(env.action_space.high, dtype=float)
        for joint_id, value in zip(unwrapped._joint_ids, np.clip(qpos, low, high)):
            addr = model.jnt_qposadr[joint_id]
            data.qpos[addr] = value
        data.ctrl[unwrapped._actuator_ids] = np.clip(qpos, low, high)
        mujoco.mj_forward(model, data)
        static_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
        moving_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
        static_pos = np.asarray(data.geom_xpos[static_pad], dtype=float)
        moving_pos = np.asarray(data.geom_xpos[moving_pad], dtype=float)
        delta = moving_pos - static_pos
        xy_norm = float(np.linalg.norm(delta[:2]))
        total_norm = float(np.linalg.norm(delta))
        horizontal_ratio = xy_norm / max(1e-6, total_norm)
        return {
            "finger_pad_delta_world": [float(value) for value in delta],
            "finger_pad_axis_xy_norm": xy_norm,
            "finger_pad_axis_z_abs": float(abs(delta[2])),
            "finger_pad_axis_horizontal_ratio": horizontal_ratio,
        }
    finally:
        _restore_sim_state(env, snapshot)


def _write_pickplace_episode(
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
    transport_steps: int,
    lower_steps: int,
    release_steps: int,
    settle_after_release_steps: int,
    recovery_steps: int,
    recovery_joint_std: float,
    approach_mode: str,
    side_approach_retreat: float,
    min_success_frames: int,
    max_success_dist: float,
    sticky_grasp: bool,
    include_camera3_duplicate: bool,
    object_shape: dict[str, Any],
) -> dict[str, Any]:
    import mujoco

    q_start = _current_qpos(env).astype(np.float32)
    q_start[-1] = _open_gripper_value(env)
    q_open = np.clip(q_open.astype(np.float32), env.action_space.low, env.action_space.high)
    q_open[-1] = _open_gripper_value(env)
    q_close = q_open.copy()
    q_close[-1] = float(env.action_space.low[-1])
    q_release = q_close.copy()
    q_release[-1] = _open_gripper_value(env)
    target_pos = np.asarray(env.unwrapped._get_target_pos(), dtype=float).copy()
    cube_start = np.asarray(env.unwrapped._get_cube_pose()[:3], dtype=float).copy()
    info: dict[str, Any] = env.unwrapped._get_info()
    frames = 0
    success_step = None
    preclose_contact = False
    preclose_contact_step = None
    phase_counts = {
        "recovery": 0,
        "approach": 0,
        "settle": 0,
        "close": 0,
        "lift": 0,
        "transport": 0,
        "lower": 0,
        "release": 0,
        "settle_after_release": 0,
    }
    action_deltas: list[float] = []
    previous_action: np.ndarray | None = None
    sticky_active = False
    sticky_offset: np.ndarray | None = None
    recovery_target = _make_recovery_qpos(env, q_start, seed=seed, joint_std=float(recovery_joint_std))

    def apply_sticky_grasp(phase: str) -> None:
        nonlocal info, sticky_active, sticky_offset
        if not sticky_grasp:
            return
        if phase in {"release", "settle_after_release"}:
            sticky_active = False
            sticky_offset = None
            return
        tcp = np.asarray(env.unwrapped._get_tcp_pose()[:3], dtype=float)
        cube_pos = np.asarray(env.unwrapped._get_cube_pose()[:3], dtype=float)
        if not sticky_active and bool(info.get("is_grasped", False)):
            sticky_active = True
            sticky_offset = cube_pos - tcp
        if not sticky_active or sticky_offset is None:
            return
        addr = int(env.unwrapped._cube_qpos_addr)
        env.unwrapped.data.qpos[addr : addr + 3] = tcp + sticky_offset
        env.unwrapped.data.qvel[:] = 0.0
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        info = env.unwrapped._get_info()

    def tcp_target_for_cube_target(cube_target: np.ndarray, grasp_cube_offset: np.ndarray | None) -> np.ndarray:
        if sticky_grasp and sticky_offset is not None:
            return np.asarray(cube_target, dtype=float) - sticky_offset
        if grasp_cube_offset is not None:
            return np.asarray(cube_target, dtype=float) - grasp_cube_offset
        return np.asarray(cube_target, dtype=float)

    def add_step(action: np.ndarray, phase: str) -> bool:
        nonlocal frames, info, success_step, previous_action, preclose_contact, preclose_contact_step
        action = np.clip(np.asarray(action, dtype=np.float32), env.action_space.low, env.action_space.high)
        dataset.add_frame(
            _make_pickplace_frame(
                env=env,
                renderers=renderers,
                action=action,
                include_camera3_duplicate=include_camera3_duplicate,
            )
        )
        frames += 1
        phase_counts[phase] += 1
        if previous_action is not None:
            action_deltas.append(float(np.linalg.norm(action[:5] - previous_action[:5])))
        previous_action = action.copy()
        _obs, _reward, terminated, truncated, info = env.step(np.asarray(action, dtype=float))
        apply_sticky_grasp(phase)
        if phase in {"approach", "settle"} and _finger_object_contact(env):
            preclose_contact = True
            if preclose_contact_step is None:
                preclose_contact_step = frames
        if bool(info.get("success", False)) and success_step is None:
            success_step = frames
        return bool(terminated) or bool(truncated)

    q_side_pre = None
    if approach_mode == "side-slide":
        q_side_pre = _solve_side_slide_preapproach_qpos(
            env,
            q_open=q_open,
            cube_start=cube_start,
            retreat=float(side_approach_retreat),
        )
    elif approach_mode != "joint":
        raise ValueError(f"unknown approach_mode: {approach_mode}")

    if q_side_pre is None:
        for index in range(max(0, int(recovery_steps))):
            alpha = (index + 1) / float(max(1, int(recovery_steps)))
            if alpha <= 0.5:
                beta = alpha / 0.5
                action = (1.0 - beta) * q_start + beta * recovery_target
            else:
                beta = (alpha - 0.5) / 0.5
                action = (1.0 - beta) * recovery_target + beta * q_start
            action[-1] = _open_gripper_value(env)
            if add_step(action, "recovery"):
                break
        for index in range(max(1, int(approach_steps))):
            alpha = (index + 1) / float(max(1, int(approach_steps)))
            alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
            action = (1.0 - alpha) * q_start + alpha * q_open
            action[-1] = _open_gripper_value(env)
            if add_step(action, "approach"):
                break
    else:
        for index in range(max(0, int(recovery_steps))):
            alpha = (index + 1) / float(max(1, int(recovery_steps)))
            if alpha <= 0.5:
                beta = alpha / 0.5
                action = (1.0 - beta) * q_start + beta * recovery_target
            else:
                beta = (alpha - 0.5) / 0.5
                action = (1.0 - beta) * recovery_target + beta * q_start
            action[-1] = _open_gripper_value(env)
            if add_step(action, "recovery"):
                break
        pre_steps = max(1, int(round(int(approach_steps) * 0.45)))
        slide_steps = max(1, int(approach_steps) - pre_steps)
        for index in range(pre_steps):
            alpha = (index + 1) / float(pre_steps)
            alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
            action = (1.0 - alpha) * q_start + alpha * q_side_pre
            action[-1] = _open_gripper_value(env)
            if add_step(action, "approach"):
                break
        for index in range(slide_steps):
            alpha = (index + 1) / float(slide_steps)
            alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
            action = (1.0 - alpha) * q_side_pre + alpha * q_open
            action[-1] = _open_gripper_value(env)
            if add_step(action, "approach"):
                break

    for _ in range(max(0, int(settle_steps))):
        if add_step(q_open, "settle"):
            break

    for _ in range(max(1, int(close_steps))):
        if add_step(q_close, "close"):
            break

    grasp_cube_offset = None
    if bool(info.get("is_grasped", False)):
        grasp_cube_offset = (
            np.asarray(env.unwrapped._get_cube_pose()[:3], dtype=float)
            - np.asarray(env.unwrapped._get_tcp_pose()[:3], dtype=float)
        )

    for _ in range(max(1, int(lift_steps))):
        cube = np.asarray(env.unwrapped._get_cube_pose()[:3], dtype=float)
        desired_cube = cube.copy()
        desired_cube[2] = max(desired_cube[2], float(env.unwrapped.cube_half_size) + 0.040)
        desired_tcp = tcp_target_for_cube_target(desired_cube, grasp_cube_offset)
        action = _closed_cartesian_target_action(env, desired_tcp, q_close[-1], gain=0.35)
        if add_step(action, "lift"):
            break

    transport_cube_start = np.asarray(env.unwrapped._get_cube_pose()[:3], dtype=float).copy()
    transport_cube_goal = target_pos + np.asarray([0.0, 0.0, float(env.unwrapped.cube_half_size) + 0.040], dtype=float)
    for _ in range(max(1, int(transport_steps))):
        alpha = (phase_counts["transport"] + 1) / float(max(1, int(transport_steps)))
        alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
        desired_cube = (1.0 - alpha) * transport_cube_start + alpha * transport_cube_goal
        desired_tcp = tcp_target_for_cube_target(desired_cube, grasp_cube_offset)
        action = _closed_cartesian_target_action(env, desired_tcp, q_close[-1], gain=0.28)
        if add_step(action, "transport"):
            break

    lower_cube_start = np.asarray(env.unwrapped._get_cube_pose()[:3], dtype=float).copy()
    lower_cube_goal = target_pos + np.asarray([0.0, 0.0, float(env.unwrapped.cube_half_size)], dtype=float)
    for _ in range(max(1, int(lower_steps))):
        alpha = (phase_counts["lower"] + 1) / float(max(1, int(lower_steps)))
        alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
        desired_cube = (1.0 - alpha) * lower_cube_start + alpha * lower_cube_goal
        desired_tcp = tcp_target_for_cube_target(desired_cube, grasp_cube_offset)
        action = _closed_cartesian_target_action(env, desired_tcp, q_close[-1], gain=0.24)
        if add_step(action, "lower"):
            break

    q_release = _current_qpos(env).astype(np.float32)
    q_release[-1] = _open_gripper_value(env)
    for _ in range(max(1, int(release_steps))):
        if add_step(q_release, "release"):
            break

    for _ in range(max(1, int(settle_after_release_steps))):
        action = _current_qpos(env).astype(np.float32)
        action[-1] = _open_gripper_value(env)
        if add_step(action, "settle_after_release"):
            break
        if bool(info.get("success", False)):
            break

    final_cube = np.asarray(env.unwrapped._get_cube_pose()[:3], dtype=float).copy()
    raw_success = bool(info.get("success", False))
    final_obj_to_target_dist = float(info.get("obj_to_target_dist", 0.0))
    quality_pass = (
        raw_success
        and frames >= max(0, int(min_success_frames))
        and final_obj_to_target_dist <= float(max_success_dist)
        and not preclose_contact
    )
    return {
        "seed": seed,
        "frames": frames,
        "success": bool(quality_pass),
        "raw_success": bool(raw_success),
        "preclose_contact": bool(preclose_contact),
        "preclose_contact_step": preclose_contact_step,
        "quality_pass": bool(quality_pass),
        "quality_thresholds": {
            "min_success_frames": int(min_success_frames),
            "max_success_dist": float(max_success_dist),
        },
        "success_step": success_step,
        "search_steps": search_steps,
        "teacher_visible_in_any_camera": bool(teacher_visible),
        "best_meta": best_meta,
        "object_shape": _object_shape_report(object_shape),
        "final_info": {
            "success": bool(info.get("success", False)),
            "is_grasped": bool(info.get("is_grasped", False)),
            "is_obj_placed": bool(info.get("is_obj_placed", False)),
            "obj_to_target_dist": final_obj_to_target_dist,
            "lift_height": float(info.get("lift_height", 0.0)),
            "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        },
        "cube_start": [float(value) for value in cube_start],
        "target_pos": [float(value) for value in target_pos],
        "final_cube": [float(value) for value in final_cube],
        "q_start": [float(value) for value in q_start],
        "q_recovery": [float(value) for value in recovery_target] if int(recovery_steps) > 0 else None,
        "q_open": [float(value) for value in q_open],
        "q_side_pre": [float(value) for value in q_side_pre] if q_side_pre is not None else None,
        "approach_mode": approach_mode,
        "phase_counts": phase_counts,
        "mean_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "max_action_delta": float(np.max(action_deltas)) if action_deltas else 0.0,
    }


def _make_recovery_qpos(env: Any, q_start: np.ndarray, *, seed: int, joint_std: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 1701)
    offset = rng.normal(0.0, max(0.0, float(joint_std)), size=q_start.shape).astype(np.float32)
    if offset.shape[0] >= 6:
        offset[-1] = 0.0
    target = np.asarray(q_start, dtype=np.float32) + offset
    target[-1] = _open_gripper_value(env)
    return np.clip(target, env.action_space.low, env.action_space.high).astype(np.float32)


def _closed_cartesian_action(env: Any, error: np.ndarray, gripper_value: float) -> np.ndarray:
    action = np.asarray(_cartesian_error_controller_action(env, np.asarray(error, dtype=float)), dtype=np.float32)
    action[-1] = float(gripper_value)
    return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


def _solve_side_slide_preapproach_qpos(
    env: Any,
    *,
    q_open: np.ndarray,
    cube_start: np.ndarray,
    retreat: float,
) -> np.ndarray:
    """Solve a low pregrasp offset toward the robot so the final segment slides in laterally."""

    unwrapped = env.unwrapped
    obj_xy = np.asarray(cube_start[:2], dtype=float)
    radial = obj_xy / max(1e-6, float(np.linalg.norm(obj_xy)))
    fake_obj = np.asarray(cube_start, dtype=float).copy()
    fake_obj[:2] = fake_obj[:2] - radial * max(0.0, float(retreat))
    old_get_target_pose = unwrapped._get_target_pose
    try:
        unwrapped._get_target_pose = lambda: np.concatenate([fake_obj, np.asarray([1.0, 0.0, 0.0, 0.0])])
        spec = {
            "candidate_index": 6,
            "mode": "left_right_side_preapproach",
            "grasp_mode": "front",
            "axis": np.asarray([0.0, 1.0, 0.0], dtype=float),
            "gap": 0.055,
            "z_offset": 0.006,
            "open_value": _open_gripper_value(env),
        }
        q_pre = _solve_pregrasp_qpos_variant(env, spec).astype(np.float32)
        q_pre[-1] = _open_gripper_value(env)
        return np.clip(q_pre, env.action_space.low, env.action_space.high).astype(np.float32)
    except Exception:
        return np.asarray(q_open, dtype=np.float32).copy()
    finally:
        unwrapped._get_target_pose = old_get_target_pose


def _finger_object_contact(env: Any) -> bool:
    import mujoco

    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data
    obj_geom = int(unwrapped._obj_geom_id)
    finger_geoms = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad"),
    }
    finger_geoms.discard(-1)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if obj_geom in pair and pair.intersection(finger_geoms):
            return True
    return False


def _closed_cartesian_target_action(env: Any, target_tcp: np.ndarray, gripper_value: float, *, gain: float) -> np.ndarray:
    tcp = np.asarray(env.unwrapped._get_tcp_pose()[:3], dtype=float)
    return _closed_cartesian_action(env, float(gain) * (np.asarray(target_tcp, dtype=float) - tcp), gripper_value)


def _make_pickplace_frame(
    *,
    env: Any,
    renderers: dict[str, Any],
    action: np.ndarray,
    include_camera3_duplicate: bool,
) -> dict[str, Any]:
    top = _render_camera(env, renderers["top_down"], "top_down")
    wrist = _render_camera(env, renderers["wrist_cam"], "wrist_cam")
    frame = {
        "observation.images.camera1": top,
        "observation.images.camera2": wrist,
        "observation.state": _current_qpos(env).astype(np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "task": TASK,
    }
    if include_camera3_duplicate:
        frame["observation.images.camera3"] = wrist.copy()
    return frame


def _add_top_down_policy_renderer(env: Any, renderers: dict[str, Any], config: WristEgoServoConfig) -> None:
    if "top_down" in renderers:
        return
    import mujoco

    renderers["top_down"] = mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width)


if __name__ == "__main__":
    main()
