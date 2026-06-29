#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local uint8 image cache for SO101 LeRobot datasets.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    report = build_cache(
        dataset_root=args.dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        cache_dir=args.cache_dir,
        video_backend=args.video_backend,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def build_cache(
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    cache_dir: Path,
    video_backend: str,
    overwrite: bool,
    progress_every: int,
) -> dict[str, Any]:
    started = perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _manifest_matches_request(
            manifest,
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
        ):
            raise RuntimeError(
                "Existing image cache manifest does not match the requested dataset. "
                f"cache_dir={cache_dir} requested_repo_id={dataset_repo_id} "
                f"cached_repo_id={manifest.get('dataset_repo_id')}"
            )
        return manifest

    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, video_backend=video_backend)
    image_keys = [key for key in dataset.features if key.startswith("observation.images.")]
    if not image_keys:
        raise RuntimeError("No observation.images.* keys found")

    sample = dataset[0]
    image_shape = tuple(int(dim) for dim in sample[image_keys[0]].shape)
    arrays = {
        key: np.lib.format.open_memmap(
            cache_dir / f"{key.replace('.', '_')}.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(dataset), *image_shape),
        )
        for key in image_keys
    }

    for index in range(len(dataset)):
        item = dataset[index]
        for key in image_keys:
            arrays[key][index] = _to_uint8_chw(item[key]).numpy()
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(f"[cache] {index + 1}/{len(dataset)} frames", flush=True)

    for array in arrays.values():
        array.flush()

    manifest = {
        "operation": "build_so101_predecoded_image_cache",
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "cache_dir": str(cache_dir),
        "image_keys": image_keys,
        "num_frames": int(len(dataset)),
        "num_episodes": int(dataset.num_episodes),
        "image_shape": list(image_shape),
        "dtype": "uint8",
        "duration_s": round(perf_counter() - started, 4),
        "files": {
            key: str(cache_dir / f"{key.replace('.', '_')}.npy")
            for key in image_keys
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _manifest_matches_request(
    manifest: dict[str, Any],
    *,
    dataset_root: Path,
    dataset_repo_id: str,
) -> bool:
    if manifest.get("dataset_repo_id") != dataset_repo_id:
        return False
    cached_root = manifest.get("dataset_root")
    if not isinstance(cached_root, str) or not cached_root:
        return False
    try:
        return Path(cached_root).expanduser().resolve() == dataset_root.expanduser().resolve()
    except OSError:
        return str(Path(cached_root)) == str(dataset_root)


def _to_uint8_chw(value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach().cpu()
    if tensor.dtype == torch.uint8:
        return tensor.contiguous()
    return tensor.to(torch.float32).mul(255.0).round().clamp(0, 255).to(torch.uint8).contiguous()


if __name__ == "__main__":
    main()
