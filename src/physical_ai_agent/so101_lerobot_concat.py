from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Sequence

import torch


REQUIRED_SO101_IMAGE_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)
REQUIRED_SO101_FEATURE_KEYS = (
    *REQUIRED_SO101_IMAGE_KEYS,
    "observation.state",
    "action",
)


class LeRobotConcatDataset(torch.utils.data.ConcatDataset):
    """Virtual LeRobot dataset merge that preserves the first dataset metadata."""

    def __init__(self, datasets: Sequence[Any], names: Sequence[str] | None = None) -> None:
        if not datasets:
            raise ValueError("LeRobotConcatDataset requires at least one child dataset")
        super().__init__(list(datasets))
        self.source_lengths = [len(dataset) for dataset in datasets]
        empty_sources = [index for index, length in enumerate(self.source_lengths) if length <= 0]
        if empty_sources:
            raise ValueError(f"LeRobotConcatDataset child datasets must be non-empty: {empty_sources}")
        self.meta = datasets[0].meta
        self.repo_id = "+".join(str(getattr(dataset, "repo_id", f"dataset_{index}")) for index, dataset in enumerate(datasets))
        self.root = [str(getattr(dataset, "root", "")) for dataset in datasets]
        self.names = list(names or [f"dataset_{index}" for index in range(len(datasets))])
        self.disable_episode_aware_sampler = True
        self.requires_dataset_balanced_sampler = True

    def source_for_index(self, index: int) -> dict[str, Any]:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = bisect_right(self.cumulative_sizes, index)
        previous_size = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        return {
            "dataset_index": dataset_index,
            "dataset_name": self.names[dataset_index],
            "local_index": index - previous_size,
        }

    def balanced_sample_weights(self) -> torch.Tensor:
        """Return per-frame weights whose total mass is equal per child dataset."""

        weights = torch.empty(len(self), dtype=torch.double)
        start = 0
        source_count = len(self.source_lengths)
        for length in self.source_lengths:
            stop = start + length
            weights[start:stop] = 1.0 / (source_count * length)
            start = stop
        return weights

    def make_dataset_balanced_sampler(
        self,
        *,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.utils.data.WeightedRandomSampler:
        return torch.utils.data.WeightedRandomSampler(
            weights=self.balanced_sample_weights(),
            num_samples=int(num_samples or len(self)),
            replacement=True,
            generator=generator,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.datasets[0], name)


def validate_lerobot_dataset_infos(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate local LeRobot metadata before expensive model/training setup."""

    if not entries:
        raise ValueError("train_datasets must contain at least one dataset")
    infos = []
    for index, entry in enumerate(entries):
        label = str(entry.get("name") or entry.get("repo_id") or f"train_datasets[{index}]")
        root = Path(str(entry.get("root") or ""))
        if not root.exists():
            raise FileNotFoundError(f"{label}: dataset root not found: {root}")
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"{label}: LeRobot metadata not found: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        _validate_expected_count(label, "expected_episodes", entry.get("expected_episodes"), info.get("total_episodes"))
        _validate_expected_count(label, "expected_frames", entry.get("expected_frames"), info.get("total_frames"))
        _validate_required_features(label, info)
        infos.append({"label": label, "root": str(root), "info": info})
    _validate_compatible_infos(infos)
    return {
        "dataset_count": len(infos),
        "total_episodes": sum(int(item["info"].get("total_episodes") or 0) for item in infos),
        "total_frames": sum(int(item["info"].get("total_frames") or 0) for item in infos),
        "datasets": [
            {
                "name": item["label"],
                "root": item["root"],
                "episodes": int(item["info"].get("total_episodes") or 0),
                "frames": int(item["info"].get("total_frames") or 0),
            }
            for item in infos
        ],
    }


def validate_compatible_lerobot_datasets(datasets: Sequence[Any]) -> None:
    if not datasets:
        raise ValueError("train_datasets must contain at least one dataset")
    reference = datasets[0].meta
    for index, dataset in enumerate(datasets[1:], start=1):
        _validate_compatible_meta(reference, dataset.meta, label=f"dataset_{index}")


def _validate_expected_count(label: str, key: str, expected: Any, actual: Any) -> None:
    if expected is None:
        return
    if actual is None:
        raise ValueError(f"{label}: cannot validate {key}; metadata count is missing")
    if int(expected) != int(actual):
        raise ValueError(f"{label}: {key}={expected} does not match metadata count {actual}")


def _validate_required_features(label: str, info: dict[str, Any]) -> None:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{label}: metadata features must be an object")
    missing = [key for key in REQUIRED_SO101_FEATURE_KEYS if key not in features]
    if missing:
        raise ValueError(f"{label}: missing required features: {', '.join(missing)}")
    for key in REQUIRED_SO101_IMAGE_KEYS:
        shape = features[key].get("shape")
        if list(shape or []) != [256, 256, 3]:
            raise ValueError(f"{label}: {key} shape must be [256, 256, 3], got {shape}")


def _validate_compatible_infos(infos: Sequence[dict[str, Any]]) -> None:
    reference = infos[0]["info"]
    reference_features = _feature_signature(reference)
    reference_fps = reference.get("fps")
    for item in infos[1:]:
        info = item["info"]
        label = item["label"]
        if info.get("fps") != reference_fps:
            raise ValueError(f"{label}: fps={info.get('fps')} does not match reference fps={reference_fps}")
        if _feature_signature(info) != reference_features:
            raise ValueError(f"{label}: feature schema does not match the first train dataset")


def _feature_signature(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features") or {}
    return {
        key: {
            "dtype": value.get("dtype"),
            "shape": list(value.get("shape") or []),
            "names": value.get("names"),
        }
        for key, value in sorted(features.items())
        if isinstance(value, dict)
    }


def _validate_compatible_meta(reference: Any, candidate: Any, *, label: str) -> None:
    ref_features = getattr(reference, "features", None)
    cand_features = getattr(candidate, "features", None)
    if ref_features is not None and cand_features is not None:
        if dict(ref_features) != dict(cand_features):
            raise ValueError(f"{label}: meta.features does not match the first train dataset")
    ref_fps = getattr(reference, "fps", None)
    cand_fps = getattr(candidate, "fps", None)
    if ref_fps is not None and cand_fps is not None and ref_fps != cand_fps:
        raise ValueError(f"{label}: meta.fps={cand_fps} does not match first train dataset fps={ref_fps}")
