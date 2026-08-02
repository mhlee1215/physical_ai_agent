#!/usr/bin/env python3
"""Materialize contiguous teacher phases as a standalone LeRobot dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

IMAGE_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)
NUMERIC_KEYS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
PHOTOREAL_LEROBOT_FORMAT = "so101_photoreal_lerobot_v1"
PHASE_SUBSET_LEROBOT_FORMAT = "so101_phase_subset_lerobot_v1"


@dataclass(frozen=True)
class SourceSpec:
    root: Path
    phase_order: tuple[str, ...]
    phases: tuple[str, ...]
    trajectory_variants: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedEpisode:
    output_episode_index: int
    source: SourceSpec
    source_episode_index: int
    source_file_index: int
    start_frame: int
    end_frame_exclusive: int
    source_episode: dict[str, Any]

    @property
    def length(self) -> int:
        return self.end_frame_exclusive - self.start_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--reconstruct-sim-snapshots", action="store_true")
    parser.add_argument("--entry-replay-qpos-rmse-max", type=float, default=0.001)
    parser.add_argument("--target-object-color", default="green")
    parser.add_argument("--object-half-sizes", default="0.015")
    parser.add_argument("--camera-rig-config", type=Path)
    parser.add_argument("--spawn-center", default="0.15,0.0")
    parser.add_argument("--spawn-min-radius", type=float, default=0.1)
    parser.add_argument("--spawn-max-radius", type=float, default=0.3)
    parser.add_argument("--spawn-angle-half-range-deg", type=float, default=90.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_specs = tuple(_parse_source_spec(value) for value in args.source_spec)
    env_config = None
    if args.reconstruct_sim_snapshots:
        if args.camera_rig_config is None:
            parser.error("--reconstruct-sim-snapshots requires --camera-rig-config")
        env_config = {
            "target_object_color": args.target_object_color,
            "object_half_sizes": _float_list(args.object_half_sizes),
            "camera_rig_config": str(args.camera_rig_config),
            "spawn_center": _float_list(args.spawn_center),
            "spawn_min_radius": args.spawn_min_radius,
            "spawn_max_radius": args.spawn_max_radius,
            "spawn_angle_half_range_deg": args.spawn_angle_half_range_deg,
        }
    report = materialize_phase_dataset(
        source_specs=source_specs,
        output_root=args.output_root,
        repo_id=args.repo_id,
        phase_id=args.phase_id,
        prompt=args.prompt,
        reconstruct_sim_snapshots=args.reconstruct_sim_snapshots,
        entry_replay_qpos_rmse_max=args.entry_replay_qpos_rmse_max,
        env_config=env_config,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "operation": report["operation"],
                "root": report["root"],
                "phase_id": report["phase_id"],
                "task": report["task"],
                "exported_episodes": report["exported_episodes"],
                "frames": sum(
                    int(episode["frames"]) for episode in report["episodes"]
                ),
                "sim_snapshot_reconstruction": report[
                    "sim_snapshot_reconstruction"
                ],
                "audit_status": report["audit"]["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def phase_frame_window(
    phase_counts: dict[str, Any],
    *,
    phase_order: tuple[str, ...],
    phases: tuple[str, ...],
) -> tuple[int, int]:
    missing = [phase for phase in phase_order if phase not in phase_counts]
    extra = [phase for phase in phase_counts if phase not in phase_order]
    if missing or extra:
        raise ValueError(
            "source phase metadata does not match the declared phase order: "
            f"missing={missing} extra={extra}"
        )
    selected_indices = [phase_order.index(phase) for phase in phases]
    if selected_indices != list(
        range(selected_indices[0], selected_indices[0] + len(selected_indices))
    ):
        raise ValueError("selected phases must form one contiguous trajectory window")
    start = sum(int(phase_counts[phase]) for phase in phase_order[: selected_indices[0]])
    length = sum(int(phase_counts[phase]) for phase in phases)
    if length <= 0:
        raise ValueError("selected phase window is empty")
    return start, start + length


def materialize_phase_dataset(
    *,
    source_specs: tuple[SourceSpec, ...],
    output_root: Path,
    repo_id: str,
    phase_id: str,
    prompt: str,
    reconstruct_sim_snapshots: bool,
    entry_replay_qpos_rmse_max: float,
    env_config: dict[str, Any] | None,
    overwrite: bool,
) -> dict[str, Any]:
    if not source_specs:
        raise ValueError("at least one source dataset is required")
    if not prompt.strip():
        raise ValueError("phase prompt must not be empty")
    source_specs = tuple(
        SourceSpec(
            root=source.root.resolve(),
            phase_order=source.phase_order,
            phases=source.phases,
            trajectory_variants=source.trajectory_variants,
        )
        for source in source_specs
    )
    output_root = output_root.resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; phase datasets are append-only")
        shutil.rmtree(output_root)

    plans, source_contexts = _plan_episodes(source_specs)
    total_frames = sum(plan.length for plan in plans)
    if not plans or total_frames <= 0:
        raise ValueError("phase selection produced no frames")
    _validate_source_features(source_contexts)

    output_root.mkdir(parents=True)
    (output_root / "data" / "chunk-000").mkdir(parents=True)
    (output_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    sample_indices = set(
        np.round(np.linspace(0, total_frames - 1, min(100, total_frames)))
        .astype(int)
        .tolist()
    )
    image_samples: dict[str, list[bytes]] = {key: [] for key in IMAGE_KEYS}
    episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
    episode_metadata: list[dict[str, Any]] = []
    output_report_episodes: list[dict[str, Any]] = []
    replay_errors: list[float] = []
    source_digest = hashlib.sha256()
    output_file_index = 0
    global_index = 0
    env = _make_replay_env(env_config) if reconstruct_sim_snapshots else None
    policy_renderers = _make_replay_policy_renderers(env) if env is not None else None

    try:
        for source_spec in source_specs:
            context = source_contexts[source_spec.root]
            source_plans = [plan for plan in plans if plan.source == source_spec]
            plans_by_file: dict[int, list[PlannedEpisode]] = {}
            for plan in source_plans:
                plans_by_file.setdefault(plan.source_file_index, []).append(plan)

            for source_file_index, file_plans in sorted(plans_by_file.items()):
                source_path = _source_data_path(
                    source_spec.root,
                    context["info"],
                    source_file_index,
                )
                source_table = pq.read_table(source_path)
                output_tables: list[pa.Table] = []
                for plan in sorted(file_plans, key=lambda item: item.source_episode_index):
                    episode_table = _episode_table(
                        source_table,
                        episode_index=plan.source_episode_index,
                    )
                    if episode_table.num_rows != int(plan.source_episode["frames"]):
                        raise ValueError(
                            "source episode row count disagrees with its export report: "
                            f"root={source_spec.root} episode={plan.source_episode_index} "
                            f"rows={episode_table.num_rows} "
                            f"report={plan.source_episode['frames']}"
                        )
                    selected = episode_table.slice(plan.start_frame, plan.length)
                    source_digest.update(_action_state_digest(selected))
                    transformed = rewrite_phase_table(
                        selected,
                        output_episode_index=plan.output_episode_index,
                        output_global_start=global_index,
                        fps=int(context["info"]["fps"]),
                        prompt=prompt,
                    )
                    output_tables.append(transformed)
                    _collect_image_samples(
                        transformed,
                        global_start=global_index,
                        wanted_indices=sample_indices,
                        samples=image_samples,
                    )
                    stats = _numeric_episode_stats(transformed, context["info"]["features"])
                    episode_stats.append(stats)

                    replay_snapshot = None
                    replay_rmse = None
                    replay_visibility = None
                    if env is not None:
                        replay_snapshot, replay_rmse, replay_visibility = _replay_phase_start(
                            env,
                            policy_renderers=policy_renderers,
                            source_episode=plan.source_episode,
                            episode_table=episode_table,
                            start_frame=plan.start_frame,
                            expected_state=np.asarray(
                                selected["observation.state"][0].as_py(),
                                dtype=float,
                            ),
                            tolerance=entry_replay_qpos_rmse_max,
                        )
                        replay_errors.append(replay_rmse)
                    elif isinstance(plan.source_episode.get("sim_snapshot"), dict):
                        if plan.start_frame == 0:
                            replay_snapshot = plan.source_episode["sim_snapshot"]

                    episode_metadata.append(
                        _episode_metadata_row(
                            episode_index=plan.output_episode_index,
                            length=plan.length,
                            prompt=prompt,
                            output_file_index=output_file_index,
                            dataset_from_index=global_index,
                            stats=stats,
                        )
                    )
                    output_report_episodes.append(
                        _phase_report_episode(
                            plan,
                            phase_id=phase_id,
                            prompt=prompt,
                            replay_snapshot=replay_snapshot,
                            replay_rmse=replay_rmse,
                            replay_visibility=replay_visibility,
                            selected=selected,
                        )
                    )
                    global_index += plan.length

                output_path = (
                    output_root
                    / "data"
                    / "chunk-000"
                    / f"file-{output_file_index:03d}.parquet"
                )
                pq.write_table(
                    pa.concat_tables(output_tables),
                    output_path,
                    compression="snappy",
                )
                output_file_index += 1
    finally:
        if policy_renderers is not None:
            for renderer in policy_renderers.values():
                renderer.close()
        if env is not None:
            env.close()

    features = source_contexts[source_specs[0].root]["info"]["features"]
    aggregate = _aggregate_numeric_stats(episode_stats)
    aggregate.update(_sampled_image_stats(image_samples))
    _write_metadata(
        output_root=output_root,
        source_info=source_contexts[source_specs[0].root]["info"],
        total_episodes=len(plans),
        total_frames=total_frames,
        prompt=prompt,
        episode_metadata=episode_metadata,
        stats=aggregate,
    )
    audit = _audit_materialized_dataset(
        output_root=output_root,
        repo_id=repo_id,
        expected_episodes=len(plans),
        expected_frames=total_frames,
        expected_prompt=prompt,
        expected_features=features,
    )
    report = _write_reports(
        output_root=output_root,
        repo_id=repo_id,
        phase_id=phase_id,
        prompt=prompt,
        source_specs=source_specs,
        total_frames=total_frames,
        episodes=output_report_episodes,
        audit=audit,
        source_action_state_sha256=source_digest.hexdigest(),
        replay_errors=replay_errors,
        replay_tolerance=entry_replay_qpos_rmse_max,
        fps=int(source_contexts[source_specs[0].root]["info"]["fps"]),
    )
    return report


def rewrite_phase_table(
    selected: pa.Table,
    *,
    output_episode_index: int,
    output_global_start: int,
    fps: int,
    prompt: str,
) -> pa.Table:
    del prompt
    length = selected.num_rows
    updated = selected
    replacements = {
        "timestamp": pa.array(
            np.arange(length, dtype=np.float32) / float(fps),
            type=selected.schema.field("timestamp").type,
        ),
        "frame_index": pa.array(
            np.arange(length, dtype=np.int64),
            type=selected.schema.field("frame_index").type,
        ),
        "episode_index": pa.array(
            np.full(length, output_episode_index, dtype=np.int64),
            type=selected.schema.field("episode_index").type,
        ),
        "index": pa.array(
            np.arange(output_global_start, output_global_start + length, dtype=np.int64),
            type=selected.schema.field("index").type,
        ),
        "task_index": pa.array(
            np.zeros(length, dtype=np.int64),
            type=selected.schema.field("task_index").type,
        ),
    }
    for key, values in replacements.items():
        field_index = updated.schema.get_field_index(key)
        updated = updated.set_column(field_index, key, values)
    for image_key in IMAGE_KEYS:
        if image_key not in updated.column_names:
            continue
        field_index = updated.schema.get_field_index(image_key)
        values = updated.column(image_key).combine_chunks()
        paths = pa.array(
            [
                (
                    f"images/{image_key.replace('.', '_')}/"
                    f"episode_{output_episode_index:06d}_frame_{frame:06d}.png"
                )
                for frame in range(length)
            ],
            type=updated.schema.field(image_key).type.field("path").type,
        )
        image_values = pa.StructArray.from_arrays(
            [values.field("bytes"), paths],
            fields=list(updated.schema.field(image_key).type),
        )
        updated = updated.set_column(field_index, image_key, image_values)
    return updated


def _plan_episodes(
    source_specs: tuple[SourceSpec, ...],
) -> tuple[list[PlannedEpisode], dict[Path, dict[str, Any]]]:
    plans: list[PlannedEpisode] = []
    contexts: dict[Path, dict[str, Any]] = {}
    selected_source_episodes: set[tuple[Path, int]] = set()
    output_episode_index = 0
    for source in source_specs:
        root = source.root.resolve()
        report = _read_json(root / "so101_lerobot_export_report.json")
        info = _read_json(root / "meta" / "info.json")
        episode_files = _episode_file_indices(root)
        episodes = list(report.get("episodes") or [])
        if len(episodes) != int(info["total_episodes"]):
            raise ValueError(
                f"source report episode count mismatch at {root}: "
                f"{len(episodes)} != {info['total_episodes']}"
            )
        contexts[root] = {"report": report, "info": info}
        matched_episodes = 0
        for source_episode_index, episode in enumerate(episodes):
            if not _episode_matches_source(episode, source):
                continue
            source_key = (root, source_episode_index)
            if source_key in selected_source_episodes:
                raise ValueError(
                    "phase_subset source filters selected the same episode more than once: "
                    f"root={root} episode={source_episode_index}"
                )
            selected_source_episodes.add(source_key)
            matched_episodes += 1
            start, end = phase_frame_window(
                dict(episode.get("phase_counts") or {}),
                phase_order=source.phase_order,
                phases=source.phases,
            )
            if end > int(episode["frames"]):
                raise ValueError(
                    f"phase window exceeds source episode {source_episode_index}: "
                    f"{start}:{end} > {episode['frames']}"
                )
            plans.append(
                PlannedEpisode(
                    output_episode_index=output_episode_index,
                    source=SourceSpec(
                        root=root,
                        phase_order=source.phase_order,
                        phases=source.phases,
                        trajectory_variants=source.trajectory_variants,
                    ),
                    source_episode_index=source_episode_index,
                    source_file_index=episode_files[source_episode_index],
                    start_frame=start,
                    end_frame_exclusive=end,
                    source_episode=episode,
                )
            )
            output_episode_index += 1
        if matched_episodes == 0:
            raise ValueError(
                "phase_subset source filter matched no episodes: "
                f"root={root} trajectory_variants={list(source.trajectory_variants)}"
            )
    normalized_contexts = {path.resolve(): value for path, value in contexts.items()}
    return plans, normalized_contexts


def _episode_file_indices(root: Path) -> dict[int, int]:
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"missing episode metadata under {root}")
    table = pa.concat_tables([pq.read_table(path) for path in paths])
    rows = table.select(["episode_index", "data/file_index"]).to_pylist()
    return {int(row["episode_index"]): int(row["data/file_index"]) for row in rows}


def _source_data_path(root: Path, info: dict[str, Any], file_index: int) -> Path:
    path = root / str(info["data_path"]).format(chunk_index=0, file_index=file_index)
    if not path.is_file():
        raise FileNotFoundError(f"missing source data file: {path}")
    return path


def _episode_table(table: pa.Table, *, episode_index: int) -> pa.Table:
    selected = table.filter(pc.equal(table["episode_index"], episode_index))
    frames = selected["frame_index"].to_pylist()
    if frames != list(range(len(frames))):
        raise ValueError(
            f"episode {episode_index} frame indices are not contiguous: "
            f"first={frames[:4]} last={frames[-4:]}"
        )
    return selected


def _numeric_episode_stats(
    table: pa.Table,
    features: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    from lerobot.datasets.compute_stats import get_feature_stats

    result: dict[str, dict[str, np.ndarray]] = {}
    for key in NUMERIC_KEYS:
        if key not in table.column_names or key not in features:
            continue
        values = np.asarray(table[key].to_pylist())
        result[key] = get_feature_stats(
            values,
            axis=0,
            keepdims=values.ndim == 1,
        )
    return result


def _aggregate_numeric_stats(
    episode_stats: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    from lerobot.datasets.compute_stats import aggregate_stats

    return aggregate_stats(episode_stats)


def _collect_image_samples(
    table: pa.Table,
    *,
    global_start: int,
    wanted_indices: set[int],
    samples: dict[str, list[bytes]],
) -> None:
    local_indices = [
        global_index - global_start
        for global_index in sorted(wanted_indices)
        if global_start <= global_index < global_start + table.num_rows
    ]
    if not local_indices:
        return
    for key in IMAGE_KEYS:
        if key not in table.column_names:
            continue
        values = table[key].combine_chunks()
        for local_index in local_indices:
            value = values[local_index].as_py()
            samples[key].append(bytes(value["bytes"]))


def _sampled_image_stats(
    samples: dict[str, list[bytes]],
) -> dict[str, dict[str, np.ndarray]]:
    from lerobot.datasets.compute_stats import get_feature_stats

    result: dict[str, dict[str, np.ndarray]] = {}
    for key, blobs in samples.items():
        if not blobs:
            continue
        images = np.stack(
            [
                np.transpose(
                    np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.uint8),
                    (2, 0, 1),
                )
                for blob in blobs
            ]
        )
        stats = get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
        result[key] = {
            name: value if name == "count" else np.squeeze(value / 255.0, axis=0)
            for name, value in stats.items()
        }
    return result


def _episode_metadata_row(
    *,
    episode_index: int,
    length: int,
    prompt: str,
    output_file_index: int,
    dataset_from_index: int,
    stats: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    from lerobot.datasets.utils import flatten_dict, serialize_dict

    row = {
        "episode_index": episode_index,
        "tasks": [prompt],
        "length": length,
        "data/chunk_index": 0,
        "data/file_index": output_file_index,
        "dataset_from_index": dataset_from_index,
        "dataset_to_index": dataset_from_index + length,
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index": 0,
    }
    row.update(flatten_dict({"stats": serialize_dict(stats)}))
    return row


def _write_metadata(
    *,
    output_root: Path,
    source_info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    prompt: str,
    episode_metadata: list[dict[str, Any]],
    stats: dict[str, dict[str, np.ndarray]],
) -> None:
    import pandas as pd
    from lerobot.datasets.io_utils import write_stats, write_tasks

    info = dict(source_info)
    info.update(
        {
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": 1,
            "splits": {"train": f"0:{total_episodes}"},
        }
    )
    (output_root / "meta" / "info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_tasks(
        pd.DataFrame(
            {"task_index": [0]},
            index=pd.Index([prompt], name="task"),
        ),
        output_root,
    )
    pq.write_table(
        pa.Table.from_pylist(episode_metadata),
        output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        compression="snappy",
    )
    write_stats(stats, output_root)


def _phase_report_episode(
    plan: PlannedEpisode,
    *,
    phase_id: str,
    prompt: str,
    replay_snapshot: dict[str, Any] | None,
    replay_rmse: float | None,
    replay_visibility: dict[str, dict[str, Any]] | None,
    selected: pa.Table,
) -> dict[str, Any]:
    source = plan.source_episode
    phase_counts = {
        phase: int(source["phase_counts"][phase]) for phase in plan.source.phases
    }
    row = {
        key: source[key]
        for key in (
            "seed",
            "forced_spawn_xy",
            "grid_balance_bin",
            "desired_grid_bin",
            "camera1_grid_bin",
            "workspace_spawn",
            "object_color",
            "object_shape",
            "target_object",
            "trajectory_variant",
            "teacher_style",
            "fixed_jaw_reference",
            "best_meta",
            "q_above",
            "q_above_misaligned",
            "q_edge",
            "q_lift",
        )
        if key in source
    }
    row.update(
        {
            "frames": plan.length,
            "phase_id": phase_id,
            "phase_counts": phase_counts,
            "task": prompt,
            "task_template": prompt,
            "success": bool(source.get("success", True)),
            "task_success": bool(source.get("task_success", True)),
            "q_start": selected["observation.state"][0].as_py(),
            "phase_end_observation_state": selected["observation.state"][
                selected.num_rows - 1
            ].as_py(),
            "start_policy_camera_visibility": (
                replay_visibility
                if replay_visibility is not None
                else _phase_start_policy_camera_visibility(selected)
            ),
            "source_provenance": {
                "dataset_root": str(plan.source.root),
                "episode_index": plan.source_episode_index,
                "frame_start": plan.start_frame,
                "frame_end_exclusive": plan.end_frame_exclusive,
                "source_task": source.get("task"),
                "source_phase_counts": source.get("phase_counts"),
            },
        }
    )
    if replay_snapshot is not None:
        row["sim_snapshot"] = replay_snapshot
    if replay_rmse is not None:
        row["phase_entry_replay_state_rmse"] = replay_rmse
        row["phase_entry_observation_state_contract"] = "simulation.ctrl"
    return row


def _phase_start_policy_camera_visibility(
    selected: pa.Table,
    *,
    min_area: int = 20,
) -> dict[str, dict[str, Any]]:
    return {
        camera_name: _green_object_visibility(
            selected[camera_key][0].as_py(),
            camera_name=(
                "egocentric_cam" if camera_name == "camera1" else "wrist_cam"
            ),
            min_area=min_area,
        )
        for camera_name, camera_key in (
            ("camera1", "observation.images.camera1"),
            ("camera2", "observation.images.camera2"),
        )
    }


def _green_object_visibility(
    image_value: object,
    *,
    camera_name: str,
    min_area: int,
) -> dict[str, Any]:
    blob = image_value.get("bytes") if isinstance(image_value, dict) else bytes(image_value)
    if not blob:
        raise ValueError(f"{camera_name} phase-start image has no encoded bytes")

    image = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.int16)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    mask = (green > 80) & (green > red + 25) & (green > blue + 20)
    ys, xs = np.where(mask)
    area = int(len(xs))
    visible = area >= min_area
    if visible:
        height, width = image.shape[:2]
        centroid = [float(xs.mean()), float(ys.mean())]
        normalized_centroid = [
            centroid[0] / max(1, width - 1),
            centroid[1] / max(1, height - 1),
        ]
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        center_distance = float(
            np.linalg.norm(np.asarray(normalized_centroid) - np.asarray([0.5, 0.5]))
        )
    else:
        centroid = None
        normalized_centroid = None
        bbox = None
        center_distance = None
    return {
        "area": area,
        "bbox": bbox,
        "camera_name": camera_name,
        "center_distance": center_distance,
        "centered": bool(center_distance is not None and center_distance <= 0.2),
        "centroid": centroid,
        "normalized_centroid": normalized_centroid,
        "visible": visible,
    }


def _make_replay_env(env_config: dict[str, Any] | None) -> Any:
    if env_config is None:
        raise ValueError("sim snapshot reconstruction requires env_config")
    try:
        from evaluate_so101_picklift_smolvla_policy import _make_eval_env
    except ModuleNotFoundError:  # pragma: no cover
        from scripts.evaluate_so101_picklift_smolvla_policy import _make_eval_env

    return _make_eval_env(
        "grip_the_cube_v1",
        target_object_color=str(env_config["target_object_color"]),
        env_config=env_config,
    )


def _make_replay_policy_renderers(env: Any) -> dict[str, Any]:
    import mujoco

    return {
        camera_name: mujoco.Renderer(env.unwrapped.model, height=256, width=256)
        for camera_name in ("egocentric_cam", "wrist_cam")
    }


def _replay_phase_start(
    env: Any,
    *,
    policy_renderers: dict[str, Any] | None,
    source_episode: dict[str, Any],
    episode_table: pa.Table,
    start_frame: int,
    expected_state: np.ndarray,
    tolerance: float,
) -> tuple[dict[str, list[float]], float, dict[str, dict[str, Any]] | None]:
    try:
        from train_so101_wrist_ego_visual_servo import (
            _restore_sim_state,
            _snapshot_sim_state,
        )
    except ModuleNotFoundError:  # pragma: no cover
        from scripts.train_so101_wrist_ego_visual_servo import (
            _restore_sim_state,
            _snapshot_sim_state,
        )

    env.reset(seed=int(source_episode["seed"]))
    snapshot = {
        key: np.asarray(source_episode["sim_snapshot"][key], dtype=float)
        for key in ("qpos", "qvel", "ctrl")
    }
    _restore_sim_state(env, snapshot)
    for frame in range(start_frame):
        env.step(np.asarray(episode_table["action"][frame].as_py(), dtype=float))
    replay = _snapshot_sim_state(env)
    rmse = _observation_state_replay_rmse(replay, expected_state)
    if rmse > tolerance:
        raise RuntimeError(
            "teacher replay failed to reconstruct the phase entry state: "
            f"seed={source_episode['seed']} start_frame={start_frame} "
            f"ctrl_state_rmse={rmse:.8f} max={tolerance:.8f}"
        )
    visibility = None
    if policy_renderers is not None:
        try:
            from export_so101_teacher_rollouts_lerobot import _policy_camera_visibility
        except ModuleNotFoundError:  # pragma: no cover
            from scripts.export_so101_teacher_rollouts_lerobot import (
                _policy_camera_visibility,
            )
        visibility = _policy_camera_visibility(
            env,
            policy_renderers,
            minimum_area=20,
        )
    return (
        {
            key: [float(value) for value in np.asarray(replay[key], dtype=float)]
            for key in ("qpos", "qvel", "ctrl")
        },
        rmse,
        visibility,
    )


def _observation_state_replay_rmse(
    replay: dict[str, Any],
    expected_state: np.ndarray,
) -> float:
    """Compare SO101 observation.state with its simulated motor-target contract."""
    actual_state = np.asarray(replay["ctrl"], dtype=float)[:6]
    expected = np.asarray(expected_state, dtype=float)[:6]
    return float(np.sqrt(np.mean(np.square(actual_state - expected))))


def _audit_materialized_dataset(
    *,
    output_root: Path,
    repo_id: str,
    expected_episodes: int,
    expected_frames: int,
    expected_prompt: str,
    expected_features: dict[str, Any],
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=output_root)
    sample = dataset[0]
    last = dataset[len(dataset) - 1]
    failures: list[str] = []
    if len(dataset) != expected_frames:
        failures.append(f"frames={len(dataset)} expected={expected_frames}")
    if dataset.num_episodes != expected_episodes:
        failures.append(
            f"episodes={dataset.num_episodes} expected={expected_episodes}"
        )
    if sample["task"] != expected_prompt or last["task"] != expected_prompt:
        failures.append("prompt mismatch")
    for key in ("observation.images.camera1", "observation.images.camera2"):
        if tuple(sample[key].shape) not in {(3, 256, 256), (256, 256, 3)}:
            failures.append(f"{key} shape={tuple(sample[key].shape)}")
    audit = {
        "operation": "materialize_so101_phase_dataset",
        "status": "passed" if not failures else "failed",
        "dataset_len": len(dataset),
        "num_episodes": dataset.num_episodes,
        "expected_prompt": expected_prompt,
        "sample_task": sample["task"],
        "last_task": last["task"],
        "sample_keys": sorted(sample),
        "declared_features": expected_features,
        "failures": failures,
    }
    audit_path = output_root / "so101_lerobot_audit.json"
    audit["audit_path"] = str(audit_path)
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"materialized phase dataset audit failed: {failures}")
    return audit


def _write_reports(
    *,
    output_root: Path,
    repo_id: str,
    phase_id: str,
    prompt: str,
    source_specs: tuple[SourceSpec, ...],
    total_frames: int,
    episodes: list[dict[str, Any]],
    audit: dict[str, Any],
    source_action_state_sha256: str,
    replay_errors: list[float],
    replay_tolerance: float,
    fps: int,
) -> dict[str, Any]:
    source_roots = list(dict.fromkeys(str(source.root) for source in source_specs))
    report = {
        "operation": "materialize_so101_phase_dataset",
        "root": str(output_root),
        "repo_id": repo_id,
        "phase_id": phase_id,
        "task": prompt,
        "task_template": prompt,
        "fps": fps,
        "requested_episodes": len(episodes),
        "exported_episodes": len(episodes),
        "episodes": episodes,
        "source_dataset_roots": source_roots,
        "phase_sources": [
            {
                "source_dataset_root": str(source.root),
                "phase_order": list(source.phase_order),
                "phases": list(source.phases),
                "trajectory_variants": list(source.trajectory_variants),
            }
            for source in source_specs
        ],
        "frame_copy_contract": {
            "images": "source PNG bytes preserved",
            "observation_state": "source values preserved",
            "action": "source values preserved",
            "episode_frame_timestamp_task_indices": "reindexed for standalone dataset",
            "source_action_state_sha256": source_action_state_sha256,
        },
        "sim_snapshot_reconstruction": {
            "enabled": bool(replay_errors),
            "episodes": len(replay_errors),
            "observation_state_contract": "simulation.ctrl",
            "state_rmse_max_contract": replay_tolerance,
            "state_rmse_mean": (
                float(np.mean(replay_errors)) if replay_errors else None
            ),
            "state_rmse_max": max(replay_errors, default=None),
            "passed": bool(replay_errors) and max(replay_errors) <= replay_tolerance,
        },
        "audit": audit,
    }
    report_path = output_root / "so101_lerobot_export_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    merge_report = {
        "operation": "materialize_so101_phase_dataset",
        "status": "passed",
        "output_root": str(output_root),
        "repo_id": repo_id,
        "phase_id": phase_id,
        "prompt": prompt,
        "source_dataset_roots": source_roots,
        "episodes": len(episodes),
        "frames": total_frames,
        "source_action_state_sha256": source_action_state_sha256,
    }
    (output_root / "so101_lerobot_merge_report.json").write_text(
        json.dumps(merge_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    source_manifest = _source_dataset_manifest(source_specs[0].root)
    camera_keys = list(source_manifest.get("camera_keys") or [])
    manifest = {
        **source_manifest,
        "operation": "phase_subset",
        "repo_id": repo_id,
        "source_dataset_roots": source_roots,
        "source_dataset_root": source_roots[0],
        "source_dataset_name": source_specs[0].root.name,
        "phase_id": phase_id,
        "phase_prompt": prompt,
        "episodes": len(episodes),
        "frames": total_frames,
        "replaced_frames": total_frames,
        "replaced_images": total_frames * len(camera_keys),
        "training_ready": True,
    }
    if source_manifest.get("format") != PHOTOREAL_LEROBOT_FORMAT:
        manifest["format"] = PHASE_SUBSET_LEROBOT_FORMAT
    _phase_subset_manifest_path(output_root, source_manifest).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _validate_source_features(contexts: dict[Path, dict[str, Any]]) -> None:
    expected = None
    expected_fps = None
    for root, context in contexts.items():
        info = context["info"]
        if expected is None:
            expected = info["features"]
            expected_fps = info["fps"]
            continue
        if info["features"] != expected:
            raise ValueError(f"source feature contract mismatch at {root}")
        if info["fps"] != expected_fps:
            raise ValueError(f"source fps mismatch at {root}")


def _action_state_digest(table: pa.Table) -> bytes:
    digest = hashlib.sha256()
    for key in ("observation.state", "action"):
        digest.update(np.asarray(table[key].to_pylist(), dtype=np.float32).tobytes())
    return digest.digest()


def _parse_source_spec(value: str) -> SourceSpec:
    payload = json.loads(value)
    return SourceSpec(
        root=Path(payload["source_dataset_root"]).resolve(),
        phase_order=tuple(str(item) for item in payload["phase_order"]),
        phases=tuple(str(item) for item in payload["phases"]),
        trajectory_variants=tuple(
            str(item) for item in payload.get("trajectory_variants", [])
        ),
    )


def _episode_matches_source(episode: dict[str, Any], source: SourceSpec) -> bool:
    if not source.trajectory_variants:
        return True
    return str(episode.get("trajectory_variant")) in source.trajectory_variants


def _source_dataset_manifest(root: Path) -> dict[str, Any]:
    for name in ("photoreal_lerobot_manifest.json", "phase_subset_manifest.json"):
        manifest_path = root / name
        if manifest_path.is_file():
            return _read_json(manifest_path)
    info = _read_json(root / "meta" / "info.json")
    camera_keys = [
        key
        for key, feature in dict(info.get("features") or {}).items()
        if str(feature.get("dtype")) in {"image", "video"}
    ]
    return {
        "operation": "phase_subset_source",
        "source_dataset_root": str(root),
        "source_dataset_name": root.name,
        "camera_keys": camera_keys,
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "training_ready": True,
    }


def _phase_subset_manifest_path(output_root: Path, source_manifest: dict[str, Any]) -> Path:
    filename = (
        "photoreal_lerobot_manifest.json"
        if source_manifest.get("format") == PHOTOREAL_LEROBOT_FORMAT
        else "phase_subset_manifest.json"
    )
    return output_root / filename


def _float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
