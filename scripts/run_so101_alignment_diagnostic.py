#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_METHODS = {"teacher_replay", "gt_waypoint_controller", "policy"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare teacher replay, an oracle waypoint controller, and a policy from the same "
            "held-out SO101 alignment-entry reset pool."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report = run_alignment_diagnostic(_load_json(_repo_path(args.config)))
    print(json.dumps(report, indent=2, sort_keys=True))


def run_alignment_diagnostic(config: dict[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    _validate_config(config)
    training_config_path = _repo_path(config["training_config_path"])
    training_config = _load_json(training_config_path)
    closed_loop = training_config["training_config"]["closed_loop"]
    test_case = _single_loop_test(closed_loop)
    start_report_path = _repo_path(test_case["start_report_path"])
    start_report = _load_json(start_report_path)
    validation_root = _repo_path(test_case["start_dataset"]["root"])
    output_dir = _repo_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_episodes = list(start_report["episodes"])[: int(config["episode_count"])]
    episode_indices = [
        int(row.get("source_validation_episode_index", row.get("episode_index", index)))
        for index, row in enumerate(selected_episodes)
    ]
    dataset_rows = _load_episode_rows(validation_root, episode_indices)
    env_config = dict(test_case["env_config"])

    reset_pool, reset_audit = _build_alignment_reset_pool(
        selected_episodes=selected_episodes,
        dataset_rows=dataset_rows,
        env_config=env_config,
        alignment_window=config["alignment_window"],
    )
    reset_pool_path = output_dir / "alignment_reset_pool.json"
    reset_pool_report = {
        "operation": "build_so101_alignment_reset_pool",
        "source_start_report": str(start_report_path),
        "source_validation_root": str(validation_root),
        "episodes": reset_pool,
        "audit": reset_audit,
    }
    reset_pool_path.write_text(
        json.dumps(reset_pool_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    method_rows: dict[str, list[dict[str, Any]]] = {}
    if "teacher_replay" in config["methods"]:
        method_rows["teacher_replay"] = _run_action_method(
            method="teacher_replay",
            reset_pool=reset_pool,
            env_config=env_config,
            success_config=config["success"],
            controller_config=None,
        )
    if "gt_waypoint_controller" in config["methods"]:
        method_rows["gt_waypoint_controller"] = _run_action_method(
            method="gt_waypoint_controller",
            reset_pool=reset_pool,
            env_config=env_config,
            success_config=config["success"],
            controller_config=config["gt_controller"],
        )
    if "policy" in config["methods"]:
        method_rows["policy"] = _run_policy_method(
            reset_pool=reset_pool,
            env_config=env_config,
            success_config=config["success"],
            checkpoint_path=_repo_path(config["checkpoint_path"]),
            policy_config=config["policy"],
            renderer_config=closed_loop["observation_renderer"],
            output_dir=output_dir,
            media_config=config["media"],
        )

    summaries = {method: _summarize_method(rows) for method, rows in method_rows.items()}
    report = {
        "operation": "run_so101_alignment_diagnostic",
        "diagnostic_id": config["diagnostic_id"],
        "training_config_path": str(training_config_path),
        "checkpoint_path": str(_repo_path(config["checkpoint_path"])),
        "source_validation_root": str(validation_root),
        "source_start_report": str(start_report_path),
        "reset_pool_path": str(reset_pool_path),
        "reset_contract": {
            "source": "held_out_validation_teacher_replay",
            "episode_indices": episode_indices,
            "alignment_window": config["alignment_window"],
            "policy_inputs": [
                "observation.images.camera1=egocentric_cam",
                "observation.images.camera2=wrist_cam",
                "observation.state=SO101 motor state",
                f"task={config['policy']['task_prompt']}",
            ],
            "training_overlap": False,
        },
        "success_contract": config["success"],
        "summaries": summaries,
        "episodes": method_rows,
        "duration_s": round(perf_counter() - started, 4),
    }
    report_path = output_dir / "alignment_diagnostic_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "diagnostic_id",
        "training_config_path",
        "checkpoint_path",
        "output_dir",
        "episode_count",
        "methods",
        "alignment_window",
        "success",
        "gt_controller",
        "policy",
        "media",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"alignment diagnostic config is missing: {missing}")
    methods = set(config["methods"])
    unknown = sorted(methods - SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"unsupported diagnostic methods: {unknown}")
    if int(config["episode_count"]) < 1:
        raise ValueError("episode_count must be positive")
    if not config["alignment_window"].get("phases"):
        raise ValueError("alignment_window.phases must not be empty")
    policy_required = {
        "device",
        "n_action_steps",
        "num_steps",
        "task_prompt",
        "torch_seed",
        "use_exact_loop_renderer",
        "reuse_rendered_policy_inputs",
    }
    missing_policy = sorted(policy_required - set(config["policy"]))
    if missing_policy:
        raise ValueError(f"alignment diagnostic policy config is missing: {missing_policy}")


def _single_loop_test(closed_loop: dict[str, Any]) -> dict[str, Any]:
    cases = list(closed_loop.get("test_cases") or [])
    if len(cases) != 1:
        raise ValueError(f"alignment diagnostic requires exactly one loop test, found {len(cases)}")
    return dict(cases[0])


def alignment_frame_window(
    phase_counts: dict[str, Any],
    *,
    phases_before: list[str],
    phases: list[str],
) -> tuple[int, int]:
    missing = [phase for phase in [*phases_before, *phases] if phase not in phase_counts]
    if missing:
        raise ValueError(f"episode phase_counts is missing phases: {missing}")
    start = sum(int(phase_counts[phase]) for phase in phases_before)
    end = start + sum(int(phase_counts[phase]) for phase in phases)
    if end <= start:
        raise ValueError(f"invalid alignment frame window: start={start} end={end}")
    return start, end


def alignment_success(
    *,
    static_edge_xy_error_m: float,
    jaw_face_parallel_error_deg: float,
    success_config: dict[str, Any],
) -> bool:
    return float(static_edge_xy_error_m) <= float(
        success_config["static_edge_xy_error_max_m"]
    ) and float(jaw_face_parallel_error_deg) <= float(
        success_config["jaw_face_parallel_error_max_deg"]
    )


def _load_episode_rows(
    dataset_root: Path,
    episode_indices: list[int],
) -> dict[int, list[dict[str, Any]]]:
    import pandas as pd

    files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no LeRobot parquet files under {dataset_root / 'data'}")
    selected = set(int(value) for value in episode_indices)
    parts = []
    columns = ["episode_index", "frame_index", "observation.state", "action"]
    for path in files:
        part = pd.read_parquet(path, columns=columns)
        part = part[part["episode_index"].isin(selected)]
        if not part.empty:
            parts.append(part)
    if not parts:
        raise ValueError(
            f"none of the requested episodes exist in {dataset_root}: {sorted(selected)}"
        )
    table = pd.concat(parts, ignore_index=True)
    result: dict[int, list[dict[str, Any]]] = {}
    for episode_index, episode in table.groupby("episode_index"):
        ordered = episode.sort_values("frame_index")
        frames = [int(value) for value in ordered["frame_index"]]
        if frames != list(range(len(frames))):
            raise ValueError(f"episode {episode_index} frame indices are not contiguous")
        result[int(episode_index)] = ordered.to_dict(orient="records")
    missing = sorted(selected - set(result))
    if missing:
        raise ValueError(f"validation dataset is missing requested episodes: {missing}")
    return result


def _build_alignment_reset_pool(
    *,
    selected_episodes: list[dict[str, Any]],
    dataset_rows: dict[int, list[dict[str, Any]]],
    env_config: dict[str, Any],
    alignment_window: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from evaluate_so101_picklift_smolvla_policy import _make_eval_env
    from train_so101_wrist_ego_visual_servo import _restore_sim_state

    env = _make_eval_env(
        "grip_the_cube_v1",
        target_object_color=str(env_config["target_object_color"]),
        env_config=env_config,
    )
    pool: list[dict[str, Any]] = []
    qpos_rmses: list[float] = []
    try:
        for pool_index, source in enumerate(selected_episodes):
            episode_index = int(
                source.get(
                    "source_validation_episode_index",
                    source.get("episode_index", pool_index),
                )
            )
            rows = dataset_rows[episode_index]
            start_frame, end_frame = alignment_frame_window(
                source["phase_counts"],
                phases_before=list(alignment_window["phases_before"]),
                phases=list(alignment_window["phases"]),
            )
            if end_frame >= len(rows):
                raise ValueError(
                    f"episode {episode_index} alignment end {end_frame} exceeds {len(rows)} frames"
                )
            env.reset(seed=int(source["seed"]))
            _restore_sim_state(env, _snapshot_arrays(source["sim_snapshot"]))
            for frame in range(start_frame):
                env.step(np.asarray(rows[frame]["action"], dtype=float))
            replay_snapshot = _snapshot_sim_state(env)
            initial_object_position = _object_position(env)
            expected_state = np.asarray(rows[start_frame]["observation.state"], dtype=float)[:6]
            actual_state = np.asarray(replay_snapshot["qpos"], dtype=float)[:6]
            replay_rmse = float(np.sqrt(np.mean(np.square(actual_state - expected_state))))
            replay_tolerance = float(alignment_window["entry_replay_qpos_rmse_max"])
            if replay_rmse > replay_tolerance:
                raise RuntimeError(
                    "teacher replay did not reproduce validation alignment entry: "
                    f"episode={episode_index} frame={start_frame} qpos_rmse={replay_rmse:.8f} "
                    f"max={replay_tolerance:.8f}"
                )
            qpos_rmses.append(replay_rmse)
            pool.append(
                {
                    "pool_index": pool_index,
                    "source_validation_episode_index": episode_index,
                    "seed": int(source["seed"]),
                    "grid_balance_bin": int(source["grid_balance_bin"]),
                    "task": str(source["task"]),
                    "alignment_start_frame": start_frame,
                    "alignment_end_frame_exclusive": end_frame,
                    "alignment_steps": end_frame - start_frame,
                    "entry_replay_qpos_rmse": replay_rmse,
                    "sim_snapshot": replay_snapshot,
                    "initial_object_position": initial_object_position.tolist(),
                    "q_edge": [float(value) for value in source["q_edge"][:6]],
                    "best_meta": source["best_meta"],
                    "teacher_actions": [
                        [float(value) for value in rows[frame]["action"][:6]]
                        for frame in range(start_frame, end_frame)
                    ],
                }
            )
    finally:
        env.close()
    return pool, {
        "episode_count": len(pool),
        "bins": _count_values([row["grid_balance_bin"] for row in pool]),
        "max_entry_replay_qpos_rmse": max(qpos_rmses, default=math.nan),
        "mean_entry_replay_qpos_rmse": float(np.mean(qpos_rmses)) if qpos_rmses else math.nan,
        "qpos_rmse_max_contract": float(alignment_window["entry_replay_qpos_rmse_max"]),
        "passed": bool(
            qpos_rmses and max(qpos_rmses) <= float(alignment_window["entry_replay_qpos_rmse_max"])
        ),
    }


def _run_action_method(
    *,
    method: str,
    reset_pool: list[dict[str, Any]],
    env_config: dict[str, Any],
    success_config: dict[str, Any],
    controller_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    from evaluate_so101_picklift_smolvla_policy import _make_eval_env
    from train_so101_wrist_ego_visual_servo import _current_qpos, _restore_sim_state

    env = _make_eval_env(
        "grip_the_cube_v1",
        target_object_color=str(env_config["target_object_color"]),
        env_config=env_config,
    )
    rows: list[dict[str, Any]] = []
    try:
        for reset in reset_pool:
            env.reset(seed=int(reset["seed"]))
            _restore_sim_state(env, _snapshot_arrays(reset["sim_snapshot"]))
            if method == "teacher_replay":
                actions = reset["teacher_actions"]
            elif method == "gt_waypoint_controller":
                assert controller_config is not None
                actions = []
                max_delta = np.asarray(controller_config["max_delta_per_joint"], dtype=float)
                if max_delta.shape != (6,):
                    raise ValueError("gt_controller.max_delta_per_joint must contain six values")
                held_gripper = float(_current_qpos(env)[-1])
                target = np.asarray(reset["q_edge"], dtype=float)
                action_delta_max = 0.0
                action_delta_values: list[float] = []
                for _ in range(int(reset["alignment_steps"])):
                    current = np.asarray(_current_qpos(env), dtype=float)[:6]
                    delta = np.clip(target - current, -max_delta, max_delta)
                    action = current + delta
                    if bool(controller_config["hold_gripper_from_reset"]):
                        action[-1] = held_gripper
                    action_delta = float(np.max(np.abs(action - current)))
                    action_delta_max = max(action_delta_max, action_delta)
                    action_delta_values.append(action_delta)
                    actions.append(action.tolist())
                    env.step(action)
                row = _measure_alignment(
                    env=env,
                    reset=reset,
                    method=method,
                    success_config=success_config,
                )
                row["max_action_delta"] = action_delta_max
                row["mean_action_delta"] = float(np.mean(action_delta_values))
                rows.append(row)
                continue
            else:
                raise ValueError(f"unsupported action method: {method}")
            action_delta_max = 0.0
            action_delta_values = []
            for action in actions:
                current = np.asarray(_current_qpos(env), dtype=float)[:6]
                action_delta = float(np.max(np.abs(np.asarray(action, dtype=float) - current)))
                action_delta_max = max(action_delta_max, action_delta)
                action_delta_values.append(action_delta)
                env.step(np.asarray(action, dtype=float))
            row = _measure_alignment(
                env=env,
                reset=reset,
                method=method,
                success_config=success_config,
            )
            row["max_action_delta"] = action_delta_max
            row["mean_action_delta"] = float(np.mean(action_delta_values))
            rows.append(row)
    finally:
        env.close()
    return rows


def _run_policy_method(
    *,
    reset_pool: list[dict[str, Any]],
    env_config: dict[str, Any],
    success_config: dict[str, Any],
    checkpoint_path: Path,
    policy_config: dict[str, Any],
    renderer_config: dict[str, Any],
    output_dir: Path,
    media_config: dict[str, Any],
) -> list[dict[str, Any]]:
    import torch
    from evaluate_so101_picklift_smolvla_policy import (
        _load_policy_processors,
        _make_eval_env,
        _make_live_observation_renderer,
        _predict_action_chunk_with_processors,
        _render_policy_camera_pair,
        _render_policy_cameras,
    )
    from train_so101_wrist_ego_visual_servo import (
        WristEgoServoConfig,
        _current_qpos,
        _make_policy_renderers,
        _restore_sim_state,
    )

    from physical_ai_agent.policies.smolvla_real import _load_pretrained_policy

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"policy checkpoint does not exist: {checkpoint_path}")
    policy = _load_pretrained_policy(
        model_id=str(checkpoint_path),
        local_files_only=True,
        device=str(policy_config["device"]),
    )
    _set_policy_rollout_config(
        policy,
        n_action_steps=int(policy_config["n_action_steps"]),
        num_steps=int(policy_config["num_steps"]),
    )
    preprocessor, postprocessor = _load_policy_processors(policy, str(checkpoint_path))
    if preprocessor is None or postprocessor is None:
        raise RuntimeError(f"checkpoint is missing saved policy processors: {checkpoint_path}")

    env = _make_eval_env(
        "grip_the_cube_v1",
        target_object_color=str(env_config["target_object_color"]),
        env_config=env_config,
    )
    renderers = _make_policy_renderers(env, WristEgoServoConfig(width=256, height=256))
    live_renderer = None
    if bool(policy_config["use_exact_loop_renderer"]):
        live_renderer = _make_live_observation_renderer(
            output_dir=output_dir / "policy",
            width=256,
            height=256,
            config=renderer_config,
        )
    rows: list[dict[str, Any]] = []
    try:
        for reset in reset_pool:
            episode = int(reset["pool_index"])
            seed = int(reset["seed"])
            torch.manual_seed(int(policy_config["torch_seed"]) + episode)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(policy_config["torch_seed"]) + episode)
            env.reset(seed=seed)
            _restore_sim_state(env, _snapshot_arrays(reset["sim_snapshot"]))
            if hasattr(policy, "reset"):
                policy.reset()
            chunk: np.ndarray | None = None
            chunk_index = 0
            action_delta_max = 0.0
            action_delta_values: list[float] = []
            query_frames = []
            query_steps: list[int] = []
            render_cache_hits = 0
            n_action_steps = int(policy_config["n_action_steps"])
            for step in range(int(reset["alignment_steps"])):
                if chunk is None or chunk_index >= min(n_action_steps, len(chunk)):
                    camera_pixels = None
                    if bool(policy_config["reuse_rendered_policy_inputs"]):
                        camera_pixels = _load_cached_policy_cameras(
                            output_dir=output_dir,
                            episode=episode,
                            seed=seed,
                            step=step,
                        )
                        if camera_pixels is not None:
                            render_cache_hits += 1
                    if camera_pixels is None and live_renderer is not None:
                        camera_pixels, _record = live_renderer.render(
                            env=env,
                            mujoco_renderers=renderers,
                            episode=episode,
                            seed=seed,
                            step=step,
                        )
                    elif camera_pixels is None:
                        camera_pixels = _render_policy_cameras(env, renderers)
                    query_frames.append(
                        _annotate_query_frame(
                            _render_policy_camera_pair(camera_pixels),
                            episode=episode,
                            step=step,
                            prompt=str(policy_config["task_prompt"]),
                        )
                    )
                    query_steps.append(step)
                    chunk = _predict_action_chunk_with_processors(
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        qpos=np.asarray(_current_qpos(env), dtype=float)[:6],
                        camera_pixels=camera_pixels,
                        task_prompt=str(policy_config["task_prompt"]),
                    )
                    chunk_index = 0
                action = np.asarray(chunk[chunk_index], dtype=float)
                current = np.asarray(_current_qpos(env), dtype=float)[:6]
                action_delta = float(np.max(np.abs(action - current)))
                action_delta_max = max(action_delta_max, action_delta)
                action_delta_values.append(action_delta)
                env.step(action)
                chunk_index += 1
            row = _measure_alignment(
                env=env,
                reset=reset,
                method="policy",
                success_config=success_config,
            )
            row["policy_query_steps"] = query_steps
            row["policy_render_cache_hits"] = render_cache_hits
            row["max_action_delta"] = action_delta_max
            row["mean_action_delta"] = float(np.mean(action_delta_values))
            if bool(media_config["write_policy_query_gif"]) and query_frames:
                gif_path = (
                    output_dir / "policy" / "media" / f"policy_query_episode_{episode:03d}.gif"
                )
                _write_gif(query_frames, gif_path, fps=int(media_config["gif_fps"]))
                row["policy_query_gif"] = str(gif_path)
            rows.append(row)
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()
    return rows


def _measure_alignment(
    *,
    env: Any,
    reset: dict[str, Any],
    method: str,
    success_config: dict[str, Any],
) -> dict[str, Any]:
    from export_so101_teacher_rollouts_lerobot import (
        _current_jaw_cube_face_normal_error_deg,
        _static_finger_edge_error,
    )
    from train_so101_wrist_ego_visual_servo import _current_qpos

    qpos = np.asarray(_current_qpos(env), dtype=float)[:6]
    q_edge = np.asarray(reset["q_edge"], dtype=float)
    static_error = _static_finger_edge_error(env, reset["best_meta"])
    angle_error = float(_current_jaw_cube_face_normal_error_deg(env, reset["best_meta"]))
    xy_error = float(static_error["xy_error"])
    final_object_position = _object_position(env)
    initial_object_position = np.asarray(reset["initial_object_position"], dtype=float)
    return {
        "method": method,
        "pool_index": int(reset["pool_index"]),
        "source_validation_episode_index": int(reset["source_validation_episode_index"]),
        "seed": int(reset["seed"]),
        "grid_balance_bin": int(reset["grid_balance_bin"]),
        "alignment_steps": int(reset["alignment_steps"]),
        "static_edge_xy_error_m": xy_error,
        "jaw_face_parallel_error_deg": angle_error,
        "q_edge_rmse": float(np.sqrt(np.mean(np.square(qpos - q_edge)))),
        "object_displacement_m": float(
            np.linalg.norm(final_object_position - initial_object_position)
        ),
        "initial_object_position": initial_object_position.tolist(),
        "final_object_position": final_object_position.tolist(),
        "final_qpos": qpos.tolist(),
        "target_q_edge": q_edge.tolist(),
        "success": alignment_success(
            static_edge_xy_error_m=xy_error,
            jaw_face_parallel_error_deg=angle_error,
            success_config=success_config,
        ),
    }


def _summarize_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "success_rate": 0.0,
        }
    return {
        "episodes": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "static_edge_xy_error_mean_m": float(
            np.mean([float(row["static_edge_xy_error_m"]) for row in rows])
        ),
        "static_edge_xy_error_max_m": float(
            np.max([float(row["static_edge_xy_error_m"]) for row in rows])
        ),
        "jaw_face_parallel_error_mean_deg": float(
            np.mean([float(row["jaw_face_parallel_error_deg"]) for row in rows])
        ),
        "jaw_face_parallel_error_max_deg": float(
            np.max([float(row["jaw_face_parallel_error_deg"]) for row in rows])
        ),
        "q_edge_rmse_mean": float(np.mean([float(row["q_edge_rmse"]) for row in rows])),
        "object_displacement_mean_m": float(
            np.mean([float(row["object_displacement_m"]) for row in rows])
        ),
        "object_displacement_max_m": float(
            np.max([float(row["object_displacement_m"]) for row in rows])
        ),
        "max_action_delta": float(np.max([float(row["max_action_delta"]) for row in rows])),
        "mean_action_delta": float(np.mean([float(row["mean_action_delta"]) for row in rows])),
    }


def _set_policy_rollout_config(policy: Any, *, n_action_steps: int, num_steps: int) -> None:
    config = getattr(policy, "config", None)
    if config is None:
        raise ValueError("loaded policy has no config")
    chunk_size = int(getattr(config, "chunk_size", n_action_steps))
    if n_action_steps < 1 or n_action_steps > chunk_size:
        raise ValueError(
            f"policy n_action_steps must be in [1, {chunk_size}], got {n_action_steps}"
        )
    if num_steps < 1:
        raise ValueError("policy num_steps must be positive")
    config.n_action_steps = int(n_action_steps)
    config.num_steps = int(num_steps)
    if hasattr(policy, "reset"):
        policy.reset()


def _snapshot_arrays(snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(snapshot[key], dtype=float) for key in ("qpos", "qvel", "ctrl")}


def _snapshot_sim_state(env: Any) -> dict[str, list[float]]:
    data = env.unwrapped.data
    return {
        key: [float(value) for value in np.asarray(getattr(data, key), dtype=float)]
        for key in ("qpos", "qvel", "ctrl")
    }


def _object_position(env: Any) -> np.ndarray:
    geom_id = int(env.unwrapped._obj_geom_id)
    return np.asarray(env.unwrapped.data.geom_xpos[geom_id], dtype=float).copy()


def _load_cached_policy_cameras(
    *,
    output_dir: Path,
    episode: int,
    seed: int,
    step: int,
) -> dict[str, np.ndarray] | None:
    from PIL import Image

    frame_dir = (
        output_dir
        / "policy"
        / "photoreal_policy_inputs"
        / f"episode_{episode:03d}_seed_{seed}"
        / f"step_{step:04d}"
    )
    paths = {
        "egocentric_cam": frame_dir / "camera1.png",
        "wrist_cam": frame_dir / "camera2.png",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    images = {
        camera: np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
        for camera, path in paths.items()
    }
    shapes = {camera: image.shape for camera, image in images.items()}
    if set(shapes.values()) != {(256, 256, 3)}:
        raise ValueError(f"cached policy cameras must be 256x256 RGB: {shapes}")
    return images


def _annotate_query_frame(
    frame: np.ndarray,
    *,
    episode: int,
    step: int,
    prompt: str,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    canvas = Image.new("RGB", (image.width, image.height + 48), color=(12, 18, 28))
    canvas.paste(image, (0, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f"episode {episode:03d} | inference step {step:03d}", fill=(96, 235, 150))
    draw.text((8, 24), prompt, fill=(235, 240, 248))
    draw.rectangle((0, 47, canvas.width - 1, canvas.height - 1), outline=(45, 210, 110), width=4)
    return np.asarray(canvas, dtype=np.uint8)


def _write_gif(frames: list[np.ndarray], path: Path, *, fps: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
    duration_ms = max(1, int(round(1000 / max(1, fps))))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def _count_values(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    main()
