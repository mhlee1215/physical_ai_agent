#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a SO101 LeRobotDataset copy whose action column stores delta-q instead of absolute qpos."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--target-repo-id", required=True)
    parser.add_argument("--mode", choices=["state_to_action"], default="state_to_action")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--video-backend", default="torchcodec")
    args = parser.parse_args()

    report = convert_dataset(
        source_root=args.source_root,
        source_repo_id=args.source_repo_id,
        target_root=args.target_root,
        target_repo_id=args.target_repo_id,
        mode=args.mode,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
        video_backend=args.video_backend,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def convert_dataset(
    *,
    source_root: Path,
    source_repo_id: str,
    target_root: Path,
    target_repo_id: str,
    mode: str,
    overwrite: bool,
    progress_every: int,
    video_backend: str,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    started = perf_counter()
    if target_root.exists():
        if not overwrite:
            raise FileExistsError(f"{target_root} already exists; pass --overwrite or choose a new --target-root")
        shutil.rmtree(target_root)

    source = LeRobotDataset(source_repo_id, root=source_root, video_backend=video_backend)
    features = _target_features(source.features)
    target = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=int(source.fps),
        features=features,
        root=target_root,
        robot_type=getattr(source.meta, "robot_type", None) or "so101",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    delta_norms: list[float] = []
    arm_delta_norms: list[float] = []
    max_abs: list[np.ndarray] = []
    converted_frames = 0
    for episode in source.meta.episodes:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        for index in range(start, end):
            item = source[index]
            state = _to_numpy(item["observation.state"])
            action = _to_numpy(item["action"])
            delta = _delta_action(state=state, action=action, mode=mode)
            target.add_frame(_convert_frame(item, action=delta))
            delta_norms.append(float(np.linalg.norm(delta)))
            arm_delta_norms.append(float(np.linalg.norm(delta[:5])))
            max_abs.append(np.abs(delta))
            converted_frames += 1
            if progress_every > 0 and converted_frames % progress_every == 0:
                print(f"[delta] {converted_frames}/{len(source)} frames", flush=True)
        target.save_episode()

    consolidate = getattr(target, "consolidate", None)
    if callable(consolidate):
        consolidate()
    report = _report(
        source=source,
        target=target,
        source_root=source_root,
        source_repo_id=source_repo_id,
        target_root=target_root,
        target_repo_id=target_repo_id,
        mode=mode,
        delta_norms=delta_norms,
        arm_delta_norms=arm_delta_norms,
        max_abs=max_abs,
        duration_s=perf_counter() - started,
    )
    (target_root / "so101_delta_action_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _target_features(features: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
        "observation.state",
        "action",
    }
    return {key: dict(value) for key, value in features.items() if key in keep}


def _convert_frame(item: dict[str, Any], *, action: np.ndarray) -> dict[str, Any]:
    frame = {
        "observation.images.camera1": _image_hwc_uint8(item["observation.images.camera1"]),
        "observation.images.camera2": _image_hwc_uint8(item["observation.images.camera2"]),
        "observation.state": _to_numpy(item["observation.state"]).astype(np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "task": str(item["task"]),
    }
    if "observation.images.camera3" in item:
        frame["observation.images.camera3"] = _image_hwc_uint8(item["observation.images.camera3"])
    return frame


def _delta_action(*, state: np.ndarray, action: np.ndarray, mode: str) -> np.ndarray:
    if mode != "state_to_action":
        raise ValueError(f"unsupported delta mode: {mode}")
    return (np.asarray(action, dtype=np.float32) - np.asarray(state, dtype=np.float32)).astype(np.float32)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_hwc_uint8(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim != 3:
        raise ValueError(f"expected 3D image, got shape={array.shape}")
    if array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(np.clip(np.rint(array.astype(np.float32) * 255.0), 0, 255).astype(np.uint8))


def _report(
    *,
    source: Any,
    target: Any,
    source_root: Path,
    source_repo_id: str,
    target_root: Path,
    target_repo_id: str,
    mode: str,
    delta_norms: list[float],
    arm_delta_norms: list[float],
    max_abs: list[np.ndarray],
    duration_s: float,
) -> dict[str, Any]:
    delta_array = np.asarray(delta_norms, dtype=np.float64)
    arm_delta_array = np.asarray(arm_delta_norms, dtype=np.float64)
    max_abs_array = np.stack(max_abs, axis=0) if max_abs else np.zeros((0, 6), dtype=np.float32)
    stats_path = target_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    return {
        "operation": "convert_so101_lerobot_actions_to_delta",
        "action_mode": "delta_q",
        "delta_mode": mode,
        "source_root": str(source_root),
        "source_repo_id": source_repo_id,
        "target_root": str(target_root),
        "target_repo_id": target_repo_id,
        "num_frames": int(len(target)),
        "num_episodes": int(target.num_episodes),
        "source_num_frames": int(len(source)),
        "source_num_episodes": int(source.num_episodes),
        "fps": int(target.fps),
        "duration_s": round(float(duration_s), 4),
        "delta_norm": _array_summary(delta_array),
        "arm_delta_norm": _array_summary(arm_delta_array),
        "delta_abs_max_per_dim": [float(value) for value in max_abs_array.max(axis=0)] if len(max_abs_array) else [],
        "delta_near_zero_ratio_1e_3": float(np.mean(delta_array <= 1e-3)) if len(delta_array) else 0.0,
        "delta_near_zero_ratio_1e_2": float(np.mean(delta_array <= 1e-2)) if len(delta_array) else 0.0,
        "stats_path": str(stats_path),
        "stats_keys": sorted(stats),
    }


def _array_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"count": 0.0}
    return {
        "count": float(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q50": float(np.quantile(values, 0.5)),
        "q90": float(np.quantile(values, 0.9)),
        "q99": float(np.quantile(values, 0.99)),
    }


if __name__ == "__main__":
    main()
