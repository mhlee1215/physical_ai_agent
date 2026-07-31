#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

TASK_PHASES = (
    "approach_down_to_cube_on_mat",
    "close_on_cube_on_mat",
    "hold_before_lift",
    "lift_from_mat",
    "post_lift_hold",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit myCobot dataset camera provenance and target visibility by task phase."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--expected-profile")
    parser.add_argument(
        "--legacy-profile",
        help="Explicit camera profile to assign to an older manifest without observation_camera.",
    )
    parser.add_argument("--samples-per-phase", type=int, default=3)
    parser.add_argument("--max-episodes-per-split", type=int, default=0)
    parser.add_argument("--min-target-pixels", type=int, default=64)
    parser.add_argument("--min-target-fraction", type=float, default=0.001)
    parser.add_argument("--min-phase-visible-rate", type=float, default=0.80)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_dataset(
        args.dataset_root,
        expected_profile=args.expected_profile,
        legacy_profile=args.legacy_profile,
        samples_per_phase=args.samples_per_phase,
        max_episodes_per_split=args.max_episodes_per_split,
        min_target_pixels=args.min_target_pixels,
        min_target_fraction=args.min_target_fraction,
        min_phase_visible_rate=args.min_phase_visible_rate,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    if args.require_pass and report["status"] != "passed":
        raise SystemExit(1)


def audit_dataset(
    dataset_root: Path,
    *,
    expected_profile: str | None = None,
    legacy_profile: str | None = None,
    samples_per_phase: int = 3,
    max_episodes_per_split: int = 0,
    min_target_pixels: int = 64,
    min_target_fraction: float = 0.001,
    min_phase_visible_rate: float = 0.80,
) -> dict[str, Any]:
    if samples_per_phase <= 0:
        raise ValueError("samples_per_phase must be positive")
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    camera_contract = manifest.get("observation_camera")
    if not isinstance(camera_contract, dict):
        if legacy_profile is None:
            errors.append("manifest is missing observation_camera")
            camera_contract = {"profile": None, "status": "missing"}
        else:
            warnings.append(
                "manifest is missing observation_camera; profile was supplied explicitly "
                f"as legacy_profile={legacy_profile!r}"
            )
            camera_contract = {
                "profile": legacy_profile,
                "status": "legacy_explicit_override",
            }
    profile = camera_contract.get("profile")
    if expected_profile is not None and profile != expected_profile:
        errors.append(
            f"camera profile mismatch: expected {expected_profile!r}, observed {profile!r}"
        )

    episode_paths = _episode_paths(dataset_root, max_episodes_per_split)
    samples: list[dict[str, Any]] = []
    for split, episode_path in episode_paths:
        rows = [
            json.loads(line)
            for line in episode_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for phase in TASK_PHASES:
            candidates = [
                row
                for row in rows
                if row.get("phase") == phase
                and row.get("observation", {}).get("images", {}).get("render")
            ]
            for index in _sample_indices(len(candidates), samples_per_phase):
                row = candidates[index]
                relative_image = row["observation"]["images"]["render"]
                image_path = dataset_root / relative_image
                try:
                    rgb = _read_rgb(image_path)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"could not read {relative_image}: {exc}")
                    continue
                target = _target_stats(
                    rgb,
                    min_target_pixels=min_target_pixels,
                    min_target_fraction=min_target_fraction,
                )
                samples.append(
                    {
                        "split": split,
                        "episode": str(episode_path.relative_to(dataset_root)),
                        "phase": phase,
                        "frame_index": int(row["frame_index"]),
                        "image": relative_image,
                        "resolution_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
                        **target,
                    }
                )

    contract_resolution = camera_contract.get("resolution_hw")
    if contract_resolution is not None:
        mismatches = [
            sample
            for sample in samples
            if sample["resolution_hw"] != list(contract_resolution)
        ]
        if mismatches:
            errors.append(
                f"{len(mismatches)} sampled images do not match contract resolution "
                f"{contract_resolution}"
            )

    phase_metrics: dict[str, Any] = {}
    for phase in TASK_PHASES:
        phase_samples = [sample for sample in samples if sample["phase"] == phase]
        if not phase_samples:
            errors.append(f"no rendered samples found for phase {phase!r}")
            continue
        fractions = [float(sample["target_fraction"]) for sample in phase_samples]
        visible_count = sum(bool(sample["target_visible"]) for sample in phase_samples)
        visible_rate = visible_count / len(phase_samples)
        phase_metrics[phase] = {
            "sample_count": len(phase_samples),
            "visible_count": visible_count,
            "visible_rate": visible_rate,
            "target_fraction_min": min(fractions),
            "target_fraction_median": statistics.median(fractions),
            "target_fraction_max": max(fractions),
        }
        if visible_rate < min_phase_visible_rate:
            errors.append(
                f"phase {phase!r} target visibility rate {visible_rate:.3f} is below "
                f"{min_phase_visible_rate:.3f}"
            )

    contract_payload = json.dumps(camera_contract, sort_keys=True, separators=(",", ":"))
    return {
        "status": "passed" if not errors else "failed",
        "dataset_root": str(dataset_root.resolve()),
        "dataset_id": manifest.get("dataset_id"),
        "camera_contract": camera_contract,
        "camera_contract_sha256": hashlib.sha256(
            contract_payload.encode("utf-8")
        ).hexdigest(),
        "thresholds": {
            "min_target_pixels": min_target_pixels,
            "min_target_fraction": min_target_fraction,
            "min_phase_visible_rate": min_phase_visible_rate,
        },
        "episode_files_audited": len(episode_paths),
        "sampled_images": len(samples),
        "phase_metrics": phase_metrics,
        "samples": samples,
        "warnings": warnings,
        "errors": errors,
    }


def _episode_paths(
    dataset_root: Path, max_episodes_per_split: int
) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for split in ("train", "validation"):
        split_paths = sorted((dataset_root / "splits" / split / "episodes").glob("*.jsonl"))
        if max_episodes_per_split > 0:
            split_paths = split_paths[:max_episodes_per_split]
        paths.extend((split, path) for path in split_paths)
    return paths


def _sample_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = min(length, count)
    if count == 1:
        return [length // 2]
    return sorted(
        {round(index * (length - 1) / (count - 1)) for index in range(count)}
    )


def _target_stats(
    rgb: np.ndarray,
    *,
    min_target_pixels: int,
    min_target_fraction: float,
) -> dict[str, Any]:
    red = rgb[:, :, 0].astype(np.float32)
    green = rgb[:, :, 1].astype(np.float32)
    blue = rgb[:, :, 2].astype(np.float32)
    mask = (red > 70.0) & (red > 1.35 * green) & (red > 1.35 * blue)
    ys, xs = np.nonzero(mask)
    target_pixels = int(mask.sum())
    target_fraction = target_pixels / float(mask.shape[0] * mask.shape[1])
    bbox = None
    if target_pixels:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    return {
        "target_pixels": target_pixels,
        "target_fraction": target_fraction,
        "target_bbox_xywh": bbox,
        "target_visible": (
            target_pixels >= min_target_pixels
            and target_fraction >= min_target_fraction
        ),
    }


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV returned no image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
