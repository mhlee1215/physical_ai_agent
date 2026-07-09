from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SO101_IMAGE_HEIGHT = 256
SO101_IMAGE_WIDTH = 256
SO101_IMAGE_CHANNELS = 3
SO101_IMAGE_SHAPE_HWC = [SO101_IMAGE_HEIGHT, SO101_IMAGE_WIDTH, SO101_IMAGE_CHANNELS]
SO101_CAMERA_FEATURE_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)


def require_so101_image_resolution(
    *,
    height: int,
    width: int,
    context: str,
) -> None:
    if int(height) != SO101_IMAGE_HEIGHT or int(width) != SO101_IMAGE_WIDTH:
        raise ValueError(
            f"{context}: SO101 visual inputs must be {SO101_IMAGE_WIDTH}x{SO101_IMAGE_HEIGHT}; "
            f"got {int(width)}x{int(height)}"
        )


def require_lerobot_dataset_256(root: str | Path, *, context: str) -> dict[str, Any]:
    root_path = Path(root)
    info_path = root_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{context}: missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{context}: {info_path} has no features object")
    bad: dict[str, Any] = {}
    for key in SO101_CAMERA_FEATURE_KEYS:
        shape = (features.get(key) or {}).get("shape")
        if list(shape or []) != SO101_IMAGE_SHAPE_HWC:
            bad[key] = shape
    if bad:
        raise ValueError(
            f"{context}: SO101 image feature contract failed for {root_path}: "
            f"expected {SO101_IMAGE_SHAPE_HWC}, got {bad}"
        )
    return info


def require_dataset_config_256(dataset_config: dict[str, Any] | None, *, repo_root: Path, context: str) -> None:
    if not isinstance(dataset_config, dict):
        return
    for label, entry in _dataset_entries(dataset_config):
        root = entry.get("root")
        if root is None:
            continue
        require_lerobot_dataset_256(_resolve_root(root, repo_root=repo_root), context=f"{context}:{label}")
    closed_loop = dataset_config.get("closed_loop")
    if isinstance(closed_loop, dict):
        for index, test_case in enumerate(closed_loop.get("test_cases") or []):
            if not isinstance(test_case, dict):
                continue
            start_dataset = test_case.get("start_dataset")
            if isinstance(start_dataset, dict) and start_dataset.get("root"):
                require_lerobot_dataset_256(
                    _resolve_root(start_dataset["root"], repo_root=repo_root),
                    context=f"{context}:closed_loop.test_cases[{index}].start_dataset",
                )
            start_report_path = test_case.get("start_report_path")
            if start_report_path:
                report_path = _resolve_root(start_report_path, repo_root=repo_root)
                if report_path.name == "so101_lerobot_export_report.json":
                    require_lerobot_dataset_256(report_path.parent, context=f"{context}:closed_loop.test_cases[{index}].start_report_path")


def _dataset_entries(dataset_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    train_dataset = dataset_config.get("train_dataset")
    if isinstance(train_dataset, dict):
        entries.append(("train_dataset", train_dataset))
    validation_dataset = dataset_config.get("validation_dataset")
    if isinstance(validation_dataset, dict):
        entries.append(("validation_dataset", validation_dataset))
    for field in ("train_datasets", "validation_datasets"):
        value = dataset_config.get(field)
        if isinstance(value, list):
            for index, entry in enumerate(value):
                if isinstance(entry, dict):
                    entries.append((f"{field}[{index}]", entry))
    return entries


def _resolve_root(value: Any, *, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path
