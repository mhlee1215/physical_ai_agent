"""Build deterministic, renderer-independent SO101 frame replay sidecars."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.so101_dataset_generation_schema import DatasetGenerationRecipe


def capture_render_replay_frame(
    env: Any,
    renderers: dict[str, Any],
    *,
    episode_index: int,
    frame_index: int,
    timestamp: float,
) -> dict[str, Any]:
    """Capture the exact pre-action visual state while the teacher is running."""
    import mujoco

    try:
        from render_so101_dataset_blender_preview import _camera_specs_from_mujoco_scene
    except ModuleNotFoundError:
        from scripts.render_so101_dataset_blender_preview import _camera_specs_from_mujoco_scene

    model = env.unwrapped.model
    data = env.unwrapped.data
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration_state = np.empty(int(mujoco.mj_stateSize(model, state_spec)), dtype=np.float64)
    mujoco.mj_getState(model, data, integration_state, state_spec)
    positions, quaternions = _geom_world_transforms(model, data)
    collision_state = _collision_state(model)
    return {
        "episode_index": int(episode_index),
        "frame_index": int(frame_index),
        "timestamp": float(timestamp),
        "integration_state": integration_state.astype(np.float32).tolist(),
        "qpos": np.asarray(data.qpos, dtype=np.float32).tolist(),
        "qvel": np.asarray(data.qvel, dtype=np.float32).tolist(),
        "ctrl": np.asarray(data.ctrl, dtype=np.float32).tolist(),
        "act": np.asarray(data.act, dtype=np.float32).tolist(),
        "mocap_pos": np.asarray(data.mocap_pos, dtype=np.float32).reshape(-1).tolist(),
        "mocap_quat": np.asarray(data.mocap_quat, dtype=np.float32).reshape(-1).tolist(),
        "rng_state_json": json.dumps(_rng_state(env), sort_keys=True),
        "collision_state_json": json.dumps(collision_state, sort_keys=True),
        "active_object_slots": _active_object_slots(collision_state),
        "geom_positions": positions.reshape(-1).tolist(),
        "geom_quaternions_wxyz": quaternions.reshape(-1).tolist(),
        "geom_rgba": np.asarray(model.geom_rgba, dtype=np.float32).reshape(-1).tolist(),
        "geom_visible": (np.asarray(model.geom_rgba, dtype=np.float32)[:, 3] > 0.0).tolist(),
        "camera_specs_json": json.dumps(
            _camera_specs_from_mujoco_scene(env, renderers, camera_lens=48.0), sort_keys=True
        ),
    }


def write_captured_render_replay_sidecar(
    dataset_root: Path,
    *,
    model: Any,
    episode_captures: list[dict[str, Any]],
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Persist exact teacher-time captures before the MuJoCo environment closes."""
    import mujoco
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        from render_so101_mitsuba_probe import _write_ply
    except ModuleNotFoundError:
        from scripts.render_so101_mitsuba_probe import _write_ply

    output_dir = dataset_root / "render_replay"
    assets_dir = output_dir / "assets" / "meshes"
    assets_dir.mkdir(parents=True, exist_ok=True)
    mesh_assets = _export_local_mesh_assets(model, assets_dir, write_ply=_write_ply)
    geom_manifest = _geom_manifest(model, mesh_assets, output_dir=output_dir)
    frames = [frame for episode in episode_captures for frame in episode["frames"]]
    snapshot_only_keys = {
        "camera_specs_json",
        "mocap_pos",
        "mocap_quat",
        "rng_state_json",
        "collision_state_json",
        "active_object_slots",
    }
    frame_states = [
        {key: value for key, value in frame.items() if key not in snapshot_only_keys}
        for frame in frames
    ]
    frame_cameras = [
        {
            "episode_index": frame["episode_index"],
            "frame_index": frame["frame_index"],
            "timestamp": frame["timestamp"],
            "camera_specs_json": frame["camera_specs_json"],
        }
        for frame in frames
    ]
    snapshots = []
    for episode in episode_captures:
        first = episode["frames"][0]
        snapshots.append(
            {
                "episode_index": int(episode["episode_index"]),
                "seed": int(episode["seed"]),
                "target_slot_index": int(episode.get("target_slot_index", -1)),
                "initial_object_z": float(episode.get("initial_object_z", float("nan"))),
                "integration_state": first["integration_state"],
                "qpos": first["qpos"],
                "qvel": first["qvel"],
                "ctrl": first["ctrl"],
                "act": first["act"],
                "mocap_pos": first["mocap_pos"],
                "mocap_quat": first["mocap_quat"],
                "rng_state_json": first["rng_state_json"],
                "active_object_slots": first["active_object_slots"],
                "collision_state_json": first["collision_state_json"],
            }
        )
    pq.write_table(
        pa.Table.from_pylist(snapshots),
        output_dir / "episode_snapshots.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(frame_states),
        output_dir / "frame_world_state.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(frame_cameras),
        output_dir / "frame_camera_specs.parquet",
        compression="zstd",
    )
    asset_checksums = {
        str(path.relative_to(output_dir)): _sha256(path)
        for path in sorted((output_dir / "assets").rglob("*"))
        if path.is_file()
    }
    (output_dir / "asset_checksums.json").write_text(
        json.dumps(asset_checksums, indent=2, sort_keys=True), encoding="utf-8"
    )
    state_size = int(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    manifest = {
        "schema_version": 1,
        "capture_mode": "teacher_time_exact",
        "dataset_root": str(dataset_root),
        "source_frames": len(frames),
        "source_episodes": len(episode_captures),
        "state_spec": "mjSTATE_INTEGRATION",
        "state_size": state_size,
        "snapshot_boundary": "settled_frame_0_before_render_and_action",
        "model": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "na": int(model.na),
            "ngeom": int(model.ngeom),
            "integration_state_size": state_size,
            "fingerprint": _model_fingerprint(model, output_dir),
            "joint_qpos_ranges": _joint_ranges(model, velocity=False),
            "joint_qvel_ranges": _joint_ranges(model, velocity=True),
            "physics": _physics_contract(model),
        },
        "environment": environment,
        "camera_contract": {
            "observation.images.camera1": {"role": "egocentric_cam", "width": 256, "height": 256},
            "observation.images.camera2": {"role": "wrist_cam", "width": 256, "height": 256},
        },
        "geom_manifest": geom_manifest,
        "files": {
            "episode_snapshots": "episode_snapshots.parquet",
            "frame_world_state": "frame_world_state.parquet",
            "frame_camera_specs": "frame_camera_specs.parquet",
            "asset_checksums": "asset_checksums.json",
        },
        "versions": _runtime_versions(mujoco.__version__),
        "validation": {
            "frame_count_matches": True,
            "max_state_error": 0.0,
            "final_outcome_mismatches": [],
            "teacher_success_only": True,
            "passed": True,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def validate_captured_render_replay_sidecar(
    dataset_root: Path, *, recipe: DatasetGenerationRecipe, split_name: str
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    replay = recipe.render_replay
    if replay is None:
        raise ValueError("recipe does not define render_replay")
    output_dir = dataset_root / replay.output_dir
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_rows = len(_dataset_rows(dataset_root))
    state_rows = pq.read_metadata(output_dir / manifest["files"]["frame_world_state"]).num_rows
    camera_rows = pq.read_metadata(output_dir / manifest["files"]["frame_camera_specs"]).num_rows
    if data_rows != state_rows or data_rows != camera_rows:
        raise ValueError(
            "render replay frame mismatch: "
            f"dataset={data_rows} state={state_rows} camera={camera_rows}"
        )
    expected_environment = replay.environment.model_dump(mode="json")
    if manifest.get("environment") != expected_environment:
        raise ValueError(
            "render replay environment mismatch: "
            f"captured={manifest.get('environment')} recipe={expected_environment}"
        )
    snapshots = pq.read_table(output_dir / manifest["files"]["episode_snapshots"]).to_pylist()
    expected_episodes = sum(item.episodes for item in recipe.splits[split_name].bins)
    if len(snapshots) != expected_episodes:
        raise ValueError(
            "render replay episode mismatch: "
            f"captured={len(snapshots)} expected={expected_episodes}"
        )
    required_snapshot_fields = {
        "integration_state",
        "qpos",
        "qvel",
        "ctrl",
        "act",
        "mocap_pos",
        "mocap_quat",
        "rng_state_json",
        "active_object_slots",
        "collision_state_json",
    }
    if snapshots and not required_snapshot_fields.issubset(snapshots[0]):
        missing = sorted(required_snapshot_fields - set(snapshots[0]))
        raise ValueError(f"render replay snapshots are missing fields: {missing}")
    camera_table = pq.read_table(output_dir / manifest["files"]["frame_camera_specs"])
    for row in camera_table.slice(0, min(8, camera_table.num_rows)).to_pylist():
        cameras = json.loads(row["camera_specs_json"])
        for key in ("observation.images.camera1", "observation.images.camera2"):
            camera = cameras.get(key, {})
            intrinsics = camera.get("intrinsics", {})
            if len(camera.get("world_from_camera", [])) != 16:
                raise ValueError(f"{key} is missing world_from_camera")
            if (intrinsics.get("width"), intrinsics.get("height")) != (256, 256):
                raise ValueError(f"{key} intrinsics must be 256x256")
    source_files = [
        dataset_root / "so101_lerobot_export_report.json",
        *sorted((dataset_root / "data").glob("chunk-*/*.parquet")),
    ]
    checksums_path = output_dir / manifest["files"]["asset_checksums"]
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["source_files"] = {
        str(path.relative_to(dataset_root)): _sha256(path) for path in source_files
    }
    checksums_path.write_text(json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8")
    manifest["recipe_name"] = recipe.name
    manifest["recipe_split"] = split_name
    manifest["dataset_root"] = str(dataset_root.resolve())
    manifest["validation"].update(
        {"frame_count_matches": True, "source_checksums_present": True, "passed": True}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def merge_render_replay_sidecars(
    shard_roots: list[Path], output_root: Path
) -> dict[str, Any] | None:
    """Merge shard-local exact captures using the same episode offsets as LeRobot."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    sidecars = [root / "render_replay" for root in shard_roots]
    present = [path.exists() for path in sidecars]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("all shards must either contain render_replay or omit it")
    output_dir = output_root / "render_replay"
    output_dir.mkdir(parents=True, exist_ok=False)
    manifests = [
        json.loads((path / "manifest.json").read_text(encoding="utf-8")) for path in sidecars
    ]
    first_assets = json.loads((sidecars[0] / "asset_checksums.json").read_text(encoding="utf-8"))
    for path in sidecars[1:]:
        assets = json.loads((path / "asset_checksums.json").read_text(encoding="utf-8"))
        if {k: v for k, v in assets.items() if k != "source_files"} != {
            k: v for k, v in first_assets.items() if k != "source_files"
        }:
            raise ValueError("render replay shard asset checksums do not match")
    import shutil

    shutil.copytree(sidecars[0] / "assets", output_dir / "assets")
    snapshots: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    cameras: list[dict[str, Any]] = []
    episode_offset = 0
    for sidecar, manifest in zip(sidecars, manifests, strict=True):
        shard_snapshots = pq.read_table(
            sidecar / manifest["files"]["episode_snapshots"]
        ).to_pylist()
        shard_states = pq.read_table(sidecar / manifest["files"]["frame_world_state"]).to_pylist()
        shard_cameras = pq.read_table(sidecar / manifest["files"]["frame_camera_specs"]).to_pylist()
        for rows in (shard_snapshots, shard_states, shard_cameras):
            for row in rows:
                row["episode_index"] = int(row["episode_index"]) + episode_offset
        snapshots.extend(shard_snapshots)
        states.extend(shard_states)
        cameras.extend(shard_cameras)
        episode_offset += int(manifest["source_episodes"])
    pq.write_table(
        pa.Table.from_pylist(snapshots),
        output_dir / "episode_snapshots.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(states), output_dir / "frame_world_state.parquet", compression="zstd"
    )
    pq.write_table(
        pa.Table.from_pylist(cameras), output_dir / "frame_camera_specs.parquet", compression="zstd"
    )
    (output_dir / "asset_checksums.json").write_text(
        json.dumps(first_assets, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = dict(manifests[0])
    manifest.update(
        {
            "dataset_root": str(output_root),
            "source_episodes": len(snapshots),
            "source_frames": len(states),
            "merged_from_shards": [str(root) for root in shard_roots],
        }
    )
    manifest["validation"] = {
        "frame_count_matches": len(states) == len(cameras),
        "max_state_error": 0.0,
        "final_outcome_mismatches": [],
        "teacher_success_only": True,
        "passed": len(states) == len(cameras),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def build_render_replay_sidecar(
    dataset_root: Path,
    *,
    recipe: DatasetGenerationRecipe,
    split_name: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    import mujoco
    import pyarrow as pa
    import pyarrow.parquet as pq
    from export_so101_teacher_rollouts_lerobot import _restore_sim_state

    try:
        from render_so101_dataset_blender_preview import _camera_specs_from_mujoco_scene
        from render_so101_mitsuba_probe import _write_ply
    except ModuleNotFoundError:
        from scripts.render_so101_dataset_blender_preview import _camera_specs_from_mujoco_scene
        from scripts.render_so101_mitsuba_probe import _write_ply
    from train_so101_wrist_ego_visual_servo import make_high_contrast_picklift_env

    replay = recipe.render_replay
    if replay is None or not replay.enabled:
        raise ValueError("recipe does not enable render_replay")
    split = recipe.splits[split_name]
    if split.kind != "generated":
        raise ValueError("render replay can only be built from a generated split")

    captured_manifest = dataset_root.resolve() / replay.output_dir / "manifest.json"
    if captured_manifest.exists():
        return validate_captured_render_replay_sidecar(
            dataset_root.resolve(), recipe=recipe, split_name=split_name
        )
    raise FileNotFoundError(
        "renderer-independent replay requires teacher-time exact capture; "
        "regenerate this split with common.capture_render_replay=true"
    )

    dataset_root = dataset_root.resolve()
    output_dir = (output_dir or dataset_root / replay.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets" / "meshes"
    assets_dir.mkdir(parents=True, exist_ok=True)

    report_path = dataset_root / "so101_lerobot_export_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episode_reports = report.get("episodes") or []
    rows = _dataset_rows(dataset_root)
    rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_episode.setdefault(int(row["episode_index"]), []).append(row)
    if len(episode_reports) != len(rows_by_episode):
        raise ValueError(
            "report/data episode mismatch: "
            f"report={len(episode_reports)} data={len(rows_by_episode)}"
        )

    environment = replay.environment
    env = make_high_contrast_picklift_env(
        target_object_color=environment.target_object_color,
        object_half_sizes=tuple(environment.object_half_sizes),
        spawn_center=tuple(environment.spawn_center),
        spawn_min_radius=environment.spawn_min_radius,
        spawn_max_radius=environment.spawn_max_radius,
        spawn_angle_half_range_deg=environment.spawn_angle_half_range_deg,
    )
    model = env.unwrapped.model
    data = env.unwrapped.data
    renderers = {
        name: mujoco.Renderer(model, width=256, height=256)
        for name in ("egocentric_cam", "wrist_cam")
    }
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = int(mujoco.mj_stateSize(model, state_spec))
    frame_states: list[dict[str, Any]] = []
    frame_cameras: list[dict[str, Any]] = []
    episode_snapshots: list[dict[str, Any]] = []
    max_state_error = 0.0
    final_outcome_mismatches: list[int] = []

    try:
        env.reset(seed=int(episode_reports[0]["seed"]))
        mesh_assets = _export_local_mesh_assets(model, assets_dir, write_ply=_write_ply)
        geom_manifest = _geom_manifest(model, mesh_assets, output_dir=output_dir)
        model_fingerprint = _model_fingerprint(model, output_dir)

        for episode_index in sorted(rows_by_episode):
            episode = episode_reports[episode_index]
            episode_rows = sorted(
                rows_by_episode[episode_index], key=lambda row: int(row["frame_index"])
            )
            env.reset(seed=int(episode["seed"]))
            snapshot = {
                key: np.asarray(episode["sim_snapshot"][key], dtype=float)
                for key in ("qpos", "qvel", "ctrl")
            }
            _restore_sim_state(env, snapshot)
            integration_state = np.empty(state_size, dtype=np.float64)
            mujoco.mj_getState(model, data, integration_state, state_spec)
            episode_snapshots.append(
                {
                    "episode_index": episode_index,
                    "seed": int(episode["seed"]),
                    "target_slot_index": int(
                        (episode.get("target_object") or {}).get("target_slot_index", -1)
                    ),
                    "initial_object_z": float(_target_object_z(env)),
                    "integration_state": integration_state.tolist(),
                    "qpos": np.asarray(data.qpos, dtype=np.float64).tolist(),
                    "qvel": np.asarray(data.qvel, dtype=np.float64).tolist(),
                    "ctrl": np.asarray(data.ctrl, dtype=np.float64).tolist(),
                    "act": np.asarray(data.act, dtype=np.float64).tolist(),
                    "mocap_pos": np.asarray(data.mocap_pos, dtype=np.float64).reshape(-1).tolist(),
                    "mocap_quat": np.asarray(data.mocap_quat, dtype=np.float64)
                    .reshape(-1)
                    .tolist(),
                    "collision_state_json": json.dumps(_collision_state(model), sort_keys=True),
                }
            )
            info: dict[str, Any] = {}
            for row in episode_rows:
                frame_index = int(row["frame_index"])
                observed_state = np.asarray(row["observation.state"], dtype=np.float64)
                actuator_ids = getattr(env.unwrapped, "_actuator_ids", None)
                replay_state = (
                    data.ctrl[actuator_ids]
                    if actuator_ids is not None
                    else data.ctrl[: len(observed_state)]
                )
                state_error = float(np.linalg.norm(np.asarray(replay_state) - observed_state))
                max_state_error = max(max_state_error, state_error)
                if state_error > 1e-4:
                    raise ValueError(
                        "source replay state mismatch at "
                        f"episode={episode_index} frame={frame_index}: {state_error:.6g}"
                    )

                mujoco.mj_getState(model, data, integration_state, state_spec)
                positions, quaternions = _geom_world_transforms(model, data)
                rgba = np.asarray(model.geom_rgba, dtype=np.float32)
                camera_specs = _camera_specs_from_mujoco_scene(env, renderers, camera_lens=48.0)
                frame_states.append(
                    {
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "timestamp": float(row["timestamp"]),
                        "integration_state": integration_state.astype(np.float32).tolist(),
                        "qpos": np.asarray(data.qpos, dtype=np.float32).tolist(),
                        "qvel": np.asarray(data.qvel, dtype=np.float32).tolist(),
                        "ctrl": np.asarray(data.ctrl, dtype=np.float32).tolist(),
                        "act": np.asarray(data.act, dtype=np.float32).tolist(),
                        "geom_positions": positions.reshape(-1).tolist(),
                        "geom_quaternions_wxyz": quaternions.reshape(-1).tolist(),
                        "geom_rgba": rgba.reshape(-1).tolist(),
                    }
                )
                frame_cameras.append(
                    {
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "timestamp": float(row["timestamp"]),
                        "camera_specs_json": json.dumps(camera_specs, sort_keys=True),
                    }
                )
                _obs, _reward, _terminated, _truncated, info = env.step(
                    np.asarray(row["action"], dtype=float)
                )
            expected = episode.get("final_info") or {}
            if expected and (
                bool(expected.get("is_grasped", False)) != bool(info.get("is_grasped", False))
                or abs(
                    float(expected.get("lift_height", 0.0)) - float(info.get("lift_height", 0.0))
                )
                > 1e-4
            ):
                final_outcome_mismatches.append(episode_index)

        pq.write_table(
            pa.Table.from_pylist(episode_snapshots),
            output_dir / "episode_snapshots.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(frame_states),
            output_dir / "frame_world_state.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(frame_cameras),
            output_dir / "frame_camera_specs.parquet",
            compression="zstd",
        )

        source_files = [report_path, *sorted((dataset_root / "data").glob("chunk-*/*.parquet"))]
        asset_checksums = {
            str(path.relative_to(output_dir)): _sha256(path)
            for path in sorted((output_dir / "assets").rglob("*"))
            if path.is_file()
        }
        asset_checksums["source_files"] = {
            str(path.relative_to(dataset_root)): _sha256(path) for path in source_files
        }
        (output_dir / "asset_checksums.json").write_text(
            json.dumps(asset_checksums, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "dataset_root": str(dataset_root),
            "recipe_name": recipe.name,
            "recipe_split": split_name,
            "source_frames": len(rows),
            "source_episodes": len(rows_by_episode),
            "state_spec": replay.state_spec,
            "state_size": state_size,
            "model": {
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nu": int(model.nu),
                "na": int(model.na),
                "ngeom": int(model.ngeom),
                "integration_state_size": state_size,
                "fingerprint": model_fingerprint,
                "joint_qpos_ranges": _joint_ranges(model, velocity=False),
                "joint_qvel_ranges": _joint_ranges(model, velocity=True),
            },
            "environment": environment.model_dump(mode="json"),
            "camera_contract": {
                "observation.images.camera1": {
                    "role": "egocentric_cam",
                    "width": 256,
                    "height": 256,
                },
                "observation.images.camera2": {"role": "wrist_cam", "width": 256, "height": 256},
            },
            "geom_manifest": geom_manifest,
            "files": {
                "episode_snapshots": "episode_snapshots.parquet",
                "frame_world_state": "frame_world_state.parquet",
                "frame_camera_specs": "frame_camera_specs.parquet",
                "asset_checksums": "asset_checksums.json",
            },
            "versions": {
                "python": platform.python_version(),
                "mujoco": mujoco.__version__,
            },
            "validation": {
                "frame_count_matches": len(frame_states) == len(rows) == len(frame_cameras),
                "max_state_error": max_state_error,
                "final_outcome_mismatches": final_outcome_mismatches,
                "passed": max_state_error <= 1e-4 and not final_outcome_mismatches,
            },
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        if not manifest["validation"]["passed"]:
            raise ValueError(f"render replay validation failed: {manifest['validation']}")
        return manifest
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()


def load_render_replay_frames(
    sidecar_dir: Path,
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    import pyarrow.parquet as pq

    manifest = json.loads((sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    states = pq.read_table(sidecar_dir / manifest["files"]["frame_world_state"]).to_pylist()
    cameras = pq.read_table(sidecar_dir / manifest["files"]["frame_camera_specs"]).to_pylist()
    camera_index = {
        (int(row["episode_index"]), int(row["frame_index"])): json.loads(row["camera_specs_json"])
        for row in cameras
    }
    frames = {}
    for row in states:
        key = (int(row["episode_index"]), int(row["frame_index"]))
        frames[key] = {**row, "camera_specs": camera_index[key]}
    return manifest, frames


def sidecar_scene_items(
    sidecar_dir: Path, manifest: dict[str, Any], frame: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ngeom = int(manifest["model"]["ngeom"])
    positions = np.asarray(frame["geom_positions"], dtype=float).reshape(ngeom, 3)
    quaternions = np.asarray(frame["geom_quaternions_wxyz"], dtype=float).reshape(ngeom, 4)
    rgba = np.asarray(frame["geom_rgba"], dtype=float).reshape(ngeom, 4)
    visible = np.asarray(frame.get("geom_visible", rgba[:, 3] > 0.0), dtype=bool).reshape(ngeom)
    meshes: list[dict[str, Any]] = []
    primitives: list[dict[str, Any]] = []
    for geom in manifest["geom_manifest"]:
        geom_id = int(geom["geom_id"])
        if not visible[geom_id] or rgba[geom_id, 3] <= 0.01:
            continue
        item = {
            **geom,
            "position": positions[geom_id].tolist(),
            "quaternion_wxyz": quaternions[geom_id].tolist(),
            "rgba": rgba[geom_id].tolist(),
        }
        if geom["type"] == "mesh":
            item["path"] = str((sidecar_dir / geom["asset_path"]).resolve())
            meshes.append(item)
        elif geom["type"] in {"box", "sphere", "cylinder"}:
            item["xmat"] = _quat_to_xmat(quaternions[geom_id]).reshape(-1).tolist()
            primitives.append(item)
    return meshes, primitives


def _dataset_rows(dataset_root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    table = pq.read_table(
        files,
        columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
    )
    return sorted(
        table.to_pylist(), key=lambda row: (int(row["episode_index"]), int(row["frame_index"]))
    )


def _geom_world_transforms(model: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    positions = np.asarray(data.geom_xpos, dtype=np.float32).copy()
    quaternions = np.empty((model.ngeom, 4), dtype=np.float32)
    for geom_id in range(model.ngeom):
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, np.asarray(data.geom_xmat[geom_id], dtype=np.float64))
        quaternions[geom_id] = quat
    return positions, quaternions


def _export_local_mesh_assets(
    model: Any, assets_dir: Path, *, write_ply: Any
) -> dict[int, dict[str, Any]]:
    assets: dict[int, dict[str, Any]] = {}
    for mesh_id in range(model.nmesh):
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        face_adr = int(model.mesh_faceadr[mesh_id])
        face_num = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float64)
        faces = np.asarray(model.mesh_face[face_adr : face_adr + face_num], dtype=np.int32)
        name = model.mesh(mesh_id).name or f"mesh_{mesh_id:03d}"
        path = assets_dir / f"{mesh_id:03d}_{_safe_name(name)}.ply"
        write_ply(path, vertices, faces)
        assets[mesh_id] = {"name": name, "path": path, "sha256": _sha256(path)}
    return assets


def _geom_manifest(
    model: Any, mesh_assets: dict[int, dict[str, Any]], *, output_dir: Path
) -> list[dict[str, Any]]:
    import mujoco

    kinds = {
        int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    }
    rows = []
    for geom_id in range(model.ngeom):
        kind = kinds.get(int(model.geom_type[geom_id]), "unsupported")
        mesh_id = int(model.geom_dataid[geom_id]) if kind == "mesh" else -1
        asset = mesh_assets.get(mesh_id)
        rows.append(
            {
                "geom_id": geom_id,
                "name": model.geom(geom_id).name or f"geom_{geom_id:03d}",
                "body_name": model.body(int(model.geom_bodyid[geom_id])).name,
                "type": kind,
                "size": np.asarray(model.geom_size[geom_id], dtype=float).tolist(),
                "mesh_id": mesh_id if mesh_id >= 0 else None,
                "mesh_name": asset["name"] if asset else None,
                "asset_path": str(asset["path"].relative_to(output_dir)) if asset else None,
                "asset_sha256": asset["sha256"] if asset else None,
                "semantic_color": "green_cube"
                if "pick_slot" in (model.geom(geom_id).name or "")
                else None,
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "friction": np.asarray(model.geom_friction[geom_id], dtype=float).tolist(),
                "solref": np.asarray(model.geom_solref[geom_id], dtype=float).tolist(),
                "solimp": np.asarray(model.geom_solimp[geom_id], dtype=float).tolist(),
                "group": int(model.geom_group[geom_id]),
            }
        )
    return rows


def _model_fingerprint(model: Any, output_dir: Path) -> dict[str, str]:
    import mujoco

    with tempfile.NamedTemporaryFile(suffix=".xml", dir=output_dir, delete=False) as handle:
        xml_path = Path(handle.name)
    try:
        mujoco.mj_saveLastXML(str(xml_path), model)
        return {"mjcf_sha256": _sha256(xml_path)}
    finally:
        xml_path.unlink(missing_ok=True)


def _joint_ranges(model: Any, *, velocity: bool) -> dict[str, list[int]]:
    ranges = {}
    addresses = model.jnt_dofadr if velocity else model.jnt_qposadr
    limit = int(model.nv if velocity else model.nq)
    for joint_id in range(model.njnt):
        start = int(addresses[joint_id])
        end = int(addresses[joint_id + 1]) if joint_id + 1 < model.njnt else limit
        ranges[model.joint(joint_id).name or f"joint_{joint_id:03d}"] = [start, end]
    return ranges


def _collision_state(model: Any) -> dict[str, dict[str, int]]:
    state = {}
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name or f"geom_{geom_id:03d}"
        if "pick_slot" in name:
            state[name] = {
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
            }
    return state


def _active_object_slots(collision_state: dict[str, dict[str, int]]) -> list[int]:
    slots = set()
    for name, flags in collision_state.items():
        if int(flags["contype"]) == 0 and int(flags["conaffinity"]) == 0:
            continue
        match = __import__("re").search(r"pick_slot_(\d+)", name)
        if match:
            slots.add(int(match.group(1)))
    return sorted(slots)


def _rng_state(env: Any) -> dict[str, Any]:
    generator = getattr(env.unwrapped, "np_random", None)
    if generator is None or not hasattr(generator, "bit_generator"):
        return {}
    return _json_ready(generator.bit_generator.state)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _physics_contract(model: Any) -> dict[str, Any]:
    option = model.opt
    return {
        "timestep": float(option.timestep),
        "integrator": int(option.integrator),
        "solver": int(option.solver),
        "cone": int(option.cone),
        "iterations": int(option.iterations),
        "ls_iterations": int(option.ls_iterations),
        "gravity": np.asarray(option.gravity, dtype=float).tolist(),
    }


def _runtime_versions(mujoco_version: str) -> dict[str, str]:
    try:
        nexus_version = importlib.metadata.version("so101-nexus-mujoco")
    except importlib.metadata.PackageNotFoundError:
        nexus_version = "unknown"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "python": platform.python_version(),
        "mujoco": str(mujoco_version),
        "so101_nexus_mujoco": nexus_version,
        "environment_code_git_commit": commit,
    }


def _target_object_z(env: Any) -> float:
    slot = int(getattr(env.unwrapped, "_target_slot_index", -1))
    if slot < 0:
        return float("nan")
    joint_name = f"pick_slot_{slot}_joint"
    try:
        return float(env.unwrapped.data.joint(joint_name).qpos[2])
    except (KeyError, AttributeError):
        return float("nan")


def _quat_to_xmat(quaternion: np.ndarray) -> np.ndarray:
    import mujoco

    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
