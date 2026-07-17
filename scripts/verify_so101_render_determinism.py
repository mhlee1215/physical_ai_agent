#!/usr/bin/env python3
"""Compare repeated SO101 renders with a small GPU pixel tolerance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--max-channel-diff", type=int, default=1)
    parser.add_argument("--max-changed-pixels", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_render_determinism(
        reference_dir=args.reference_dir,
        candidate_dir=args.candidate_dir,
        max_channel_diff=args.max_channel_diff,
        max_changed_pixels=args.max_changed_pixels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


def verify_render_determinism(
    *,
    reference_dir: Path,
    candidate_dir: Path,
    max_channel_diff: int,
    max_changed_pixels: int,
) -> dict[str, object]:
    reference = _camera_images(reference_dir)
    candidate = _camera_images(candidate_dir)
    if set(reference) != set(candidate):
        raise ValueError(
            f"render camera mismatch: reference={sorted(reference)} candidate={sorted(candidate)}"
        )
    cameras = {}
    passed = True
    for camera_key in sorted(reference):
        first = np.asarray(Image.open(reference[camera_key]).convert("RGB"), dtype=np.int16)
        second = np.asarray(Image.open(candidate[camera_key]).convert("RGB"), dtype=np.int16)
        if first.shape != second.shape:
            raise ValueError(f"{camera_key} shape mismatch: {first.shape} != {second.shape}")
        difference = np.abs(first - second)
        changed_pixels = int(np.any(difference > 0, axis=2).sum())
        max_difference = int(difference.max(initial=0))
        camera_passed = max_difference <= max_channel_diff and changed_pixels <= max_changed_pixels
        passed = passed and camera_passed
        cameras[camera_key] = {
            "max_channel_diff": max_difference,
            "changed_pixels": changed_pixels,
            "total_pixels": int(first.shape[0] * first.shape[1]),
            "mean_absolute_diff": float(difference.mean()),
            "passed": camera_passed,
        }
    return {
        "schema_version": 1,
        "reference_dir": str(reference_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "tolerance": {
            "max_channel_diff": int(max_channel_diff),
            "max_changed_pixels": int(max_changed_pixels),
        },
        "cameras": cameras,
        "passed": passed,
    }


def _camera_images(root: Path) -> dict[str, Path]:
    frame_dir = root / "episode_0000_frame_0000"
    images = {}
    for path in sorted(frame_dir.glob("episode_0000_frame_0000_camera*.*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        images[path.stem.rsplit("_", 1)[-1]] = path
    if not images:
        raise FileNotFoundError(f"no determinism probe images under {frame_dir}")
    return images


if __name__ == "__main__":
    main()
