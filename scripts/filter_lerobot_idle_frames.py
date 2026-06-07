#!/usr/bin/env python3
"""Filter idle frames from a single-file LeRobot dataset.

This is intended for ManiSkill3/STARE SFT parity experiments where paper
protocols filter near-idle actions before training. It rewrites the tabular
LeRobot metadata and reuses source videos through symlinks by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/chunk-000/file-000.parquet")
EPISODES_PATH = Path("meta/episodes/chunk-000/file-000.parquet")
TASKS_PATH = Path("meta/tasks.parquet")
INFO_PATH = Path("meta/info.json")
STATS_PATH = Path("meta/stats.json")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _vector_stats(values: list[np.ndarray]) -> dict:
    array = np.stack(values).astype(np.float64)
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def _scalar_stats(values: list[float | int]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": [float(array.min())],
        "max": [float(array.max())],
        "mean": [float(array.mean())],
        "std": [float(array.std())],
        "count": [int(array.shape[0])],
    }


def _keep_mask(states: list[np.ndarray], threshold: float) -> list[bool]:
    if not states:
        return []
    mask = [True]
    previous_kept = states[0]
    for state in states[1:]:
        keep = float(np.linalg.norm(state - previous_kept)) >= threshold
        mask.append(keep)
        if keep:
            previous_kept = state
    if not any(mask[1:]) and len(mask) > 1:
        mask[-1] = True
    return mask


def _symlink_or_copy_tree(source: Path, dest: Path, copy_videos: bool) -> None:
    if not source.exists():
        return
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if copy_videos:
        shutil.copytree(source, dest)
    else:
        dest.symlink_to(source, target_is_directory=True)


def filter_dataset(
    source_root: Path,
    dest_root: Path,
    threshold: float,
    state_key: str,
    copy_videos: bool,
) -> dict:
    if dest_root.exists():
        shutil.rmtree(dest_root)
    (dest_root / DATA_PATH.parent).mkdir(parents=True, exist_ok=True)
    (dest_root / EPISODES_PATH.parent).mkdir(parents=True, exist_ok=True)
    (dest_root / "meta").mkdir(parents=True, exist_ok=True)

    info = _load_json(source_root / INFO_PATH)
    source_stats = _load_json(source_root / STATS_PATH)
    data = pd.read_parquet(source_root / DATA_PATH)
    episodes = pd.read_parquet(source_root / EPISODES_PATH)

    rows: list[dict] = []
    episode_rows: list[dict] = []
    next_index = 0
    lengths: list[int] = []

    for episode_index in sorted(data["episode_index"].unique()):
        episode_data = data[data["episode_index"] == episode_index].reset_index(drop=True)
        source_episode = episodes[episodes["episode_index"] == episode_index].iloc[0].to_dict()
        states = [np.asarray(value, dtype=np.float32) for value in episode_data[state_key]]
        mask = _keep_mask(states, threshold)
        kept = episode_data.loc[mask].reset_index(drop=True)
        length = int(len(kept))
        lengths.append(length)

        from_index = next_index
        for frame_index, item in kept.iterrows():
            row = item.to_dict()
            row["frame_index"] = int(frame_index)
            row["episode_index"] = int(episode_index)
            row["index"] = int(next_index)
            row["timestamp"] = float(frame_index / info.get("fps", 30))
            rows.append(row)
            next_index += 1

        episode_row = source_episode
        episode_row["dataset_from_index"] = int(from_index)
        episode_row["dataset_to_index"] = int(next_index)
        episode_row["length"] = length
        video_prefix = "videos/observation.images.base_camera"
        if f"{video_prefix}/from_timestamp" in episode_row:
            episode_row[f"{video_prefix}/from_timestamp"] = 0.0
        if f"{video_prefix}/to_timestamp" in episode_row:
            episode_row[f"{video_prefix}/to_timestamp"] = float((length - 1) / info.get("fps", 30))

        for key in ("action", state_key):
            stats = _vector_stats([np.asarray(value, dtype=np.float32) for value in kept[key]])
            for stat_name, stat_value in stats.items():
                episode_row[f"stats/{key}/{stat_name}"] = stat_value
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            stats = _scalar_stats([item[key] for item in rows[from_index:next_index]])
            for stat_name, stat_value in stats.items():
                episode_row[f"stats/{key}/{stat_name}"] = stat_value

        episode_rows.append(episode_row)

    filtered = pd.DataFrame(rows)
    filtered.to_parquet(dest_root / DATA_PATH, index=False)
    pd.DataFrame(episode_rows).to_parquet(dest_root / EPISODES_PATH, index=False)
    shutil.copy2(source_root / TASKS_PATH, dest_root / TASKS_PATH)

    videos_source = source_root / "videos"
    videos_dest = dest_root / "videos"
    _symlink_or_copy_tree(videos_source, videos_dest, copy_videos)

    info["total_frames"] = int(len(filtered))
    info["total_episodes"] = int(len(episode_rows))
    info["splits"] = {"train": f"0:{len(episode_rows)}"}
    info["data_files_size_in_mb"] = max(1, int((dest_root / DATA_PATH).stat().st_size / 1_000_000))
    _write_json(dest_root / INFO_PATH, info)

    stats = dict(source_stats)
    for key in ("action", state_key):
        stats[key] = _vector_stats([np.asarray(value, dtype=np.float32) for value in filtered[key]])
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        stats[key] = _scalar_stats(filtered[key].tolist())
    _write_json(dest_root / STATS_PATH, stats)

    return {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "threshold": threshold,
        "episodes": int(len(episode_rows)),
        "frames": int(len(filtered)),
        "length_mean": float(np.mean(lengths)),
        "length_p50": float(np.percentile(lengths, 50)),
        "length_p90": float(np.percentile(lengths, 90)),
        "length_max": int(max(lengths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--dest-root", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--copy-videos", action="store_true")
    args = parser.parse_args()

    result = filter_dataset(
        source_root=args.source_root,
        dest_root=args.dest_root,
        threshold=args.threshold,
        state_key=args.state_key,
        copy_videos=args.copy_videos,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
