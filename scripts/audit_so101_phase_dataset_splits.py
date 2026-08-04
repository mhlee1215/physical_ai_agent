#!/usr/bin/env python3
"""Audit independent train/validation datasets materialized from teacher phases."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--expected-prompt", required=True)
    parser.add_argument("--expected-resolution", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    width, height = _resolution(args.expected_resolution)
    report = audit_phase_splits(
        train_root=args.train_root,
        validation_root=args.validation_root,
        expected_prompt=args.expected_prompt,
        expected_resolution=(width, height),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


def audit_phase_splits(
    *,
    train_root: Path,
    validation_root: Path,
    expected_prompt: str,
    expected_resolution: tuple[int, int],
) -> dict[str, Any]:
    train = _audit_root(
        train_root.resolve(),
        expected_prompt=expected_prompt,
        expected_resolution=expected_resolution,
    )
    validation = _audit_root(
        validation_root.resolve(),
        expected_prompt=expected_prompt,
        expected_resolution=expected_resolution,
    )
    source_overlap = sorted(
        train["source_episode_keys"] & validation["source_episode_keys"]
    )
    seed_spawn_overlap = sorted(
        train["seed_spawn_keys"] & validation["seed_spawn_keys"]
    )
    failures = [
        *(f"train: {item}" for item in train["failures"]),
        *(f"validation: {item}" for item in validation["failures"]),
    ]
    if source_overlap:
        failures.append(
            f"train/validation source episode overlap ({len(source_overlap)})"
        )
    if seed_spawn_overlap:
        failures.append(
            f"train/validation seed+spawn overlap ({len(seed_spawn_overlap)})"
        )
    return {
        "operation": "audit_so101_phase_dataset_splits",
        "status": "passed" if not failures else "failed",
        "expected_prompt": expected_prompt,
        "expected_resolution": list(expected_resolution),
        "train": _public_root_audit(train),
        "validation": _public_root_audit(validation),
        "source_episode_overlap_count": len(source_overlap),
        "seed_spawn_overlap_count": len(seed_spawn_overlap),
        "failures": failures,
    }


def _audit_root(
    root: Path,
    *,
    expected_prompt: str,
    expected_resolution: tuple[int, int],
) -> dict[str, Any]:
    failures: list[str] = []
    info = _read_json(root / "meta" / "info.json")
    report = _read_json(root / "so101_lerobot_export_report.json")
    tasks = _read_tables(root / "meta" / "tasks.parquet")
    episodes_meta = _read_tables(root / "meta" / "episodes")
    data = _read_tables(
        root / "data",
        columns=[
            "episode_index",
            "frame_index",
            "index",
            "task_index",
            "observation.state",
            "action",
        ],
    )

    expected_episodes = int(info["total_episodes"])
    expected_frames = int(info["total_frames"])
    if data.num_rows != expected_frames:
        failures.append(f"data rows={data.num_rows}, info frames={expected_frames}")
    if episodes_meta.num_rows != expected_episodes:
        failures.append(
            f"episode rows={episodes_meta.num_rows}, info episodes={expected_episodes}"
        )
    report_episodes = list(report.get("episodes") or [])
    if len(report_episodes) != expected_episodes:
        failures.append(
            f"report episodes={len(report_episodes)}, info episodes={expected_episodes}"
        )
    task_rows = tasks.to_pylist()
    if task_rows != [{"task_index": 0, "task": expected_prompt}]:
        failures.append(f"task metadata does not equal expected prompt: {task_rows}")
    if report.get("task") != expected_prompt:
        failures.append(f"report prompt={report.get('task')!r}")

    episode_indices = np.asarray(data["episode_index"].to_pylist(), dtype=np.int64)
    frame_indices = np.asarray(data["frame_index"].to_pylist(), dtype=np.int64)
    global_indices = np.asarray(data["index"].to_pylist(), dtype=np.int64)
    task_indices = np.asarray(data["task_index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(global_indices, np.arange(expected_frames)):
        failures.append("global indices are not contiguous")
    if np.any(task_indices != 0):
        failures.append("task_index contains values other than zero")
    for episode_index in range(expected_episodes):
        mask = episode_indices == episode_index
        frames = frame_indices[mask]
        if not np.array_equal(frames, np.arange(len(frames))):
            failures.append(f"episode {episode_index} frame indices are not contiguous")
            break

    source_episode_keys: set[str] = set()
    seed_spawn_keys: set[str] = set()
    trajectory_hashes: set[str] = set()
    for episode in report_episodes:
        provenance = dict(episode.get("source_provenance") or {})
        source_episode_keys.add(
            f"{Path(str(provenance.get('dataset_root'))).resolve()}:"
            f"{int(provenance.get('episode_index', -1))}"
        )
        seed_spawn_keys.add(
            json.dumps(
                {
                    "seed": episode.get("seed"),
                    "forced_spawn_xy": episode.get("forced_spawn_xy"),
                },
                sort_keys=True,
            )
        )
    for episode_index in range(expected_episodes):
        table = data.filter(pc.equal(data["episode_index"], episode_index))
        trajectory_hashes.add(_trajectory_hash(table))
    if len(trajectory_hashes) != expected_episodes:
        failures.append(
            f"unique trajectories={len(trajectory_hashes)}, episodes={expected_episodes}"
        )

    image_shapes: dict[str, list[int]] = {}
    first_data_path = sorted((root / "data").glob("chunk-*/*.parquet"))[0]
    for key in ("observation.images.camera1", "observation.images.camera2"):
        value = pq.read_table(first_data_path, columns=[key])[key][0].as_py()
        image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        image_shapes[key] = [image.width, image.height]
        if image.size != expected_resolution:
            failures.append(
                f"{key} resolution={image.size}, expected={expected_resolution}"
            )

    return {
        "root": str(root),
        "episodes": expected_episodes,
        "frames": expected_frames,
        "phase_id": report.get("phase_id"),
        "prompt": report.get("task"),
        "image_shapes": image_shapes,
        "unique_trajectory_hashes": len(trajectory_hashes),
        "source_episode_keys": source_episode_keys,
        "seed_spawn_keys": seed_spawn_keys,
        "failures": failures,
    }


def _public_root_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in audit.items()
        if key not in {"source_episode_keys", "seed_spawn_keys", "failures"}
    }


def _trajectory_hash(table: pa.Table) -> str:
    digest = hashlib.sha256()
    for key in ("observation.state", "action"):
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)
        digest.update(values.tobytes())
    return digest.hexdigest()


def _read_tables(path: Path, *, columns: list[str] | None = None) -> pa.Table:
    paths = [path] if path.is_file() else sorted(path.glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files under {path}")
    return pa.concat_tables([pq.read_table(item, columns=columns) for item in paths])


def _resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"invalid resolution: {value}")
    return int(parts[0]), int(parts[1])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
