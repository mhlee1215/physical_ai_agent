from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
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


class _MetaWithAggregatedStats:
    """Forward dataset metadata while replacing stats with virtual-merge stats."""

    def __init__(self, base_meta: Any, stats: dict[str, dict[str, np.ndarray]]) -> None:
        self._base_meta = base_meta
        self.stats = stats

    def __getattr__(self, name: str) -> Any:
        if name == "_base_meta" or "_base_meta" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._base_meta, name)


class LeRobotConcatDataset(torch.utils.data.ConcatDataset):
    """Virtual LeRobot dataset merge with aggregate metadata statistics."""

    def __init__(self, datasets: Sequence[Any], names: Sequence[str] | None = None) -> None:
        if not datasets:
            raise ValueError("LeRobotConcatDataset requires at least one child dataset")
        super().__init__(list(datasets))
        self.source_lengths = [len(dataset) for dataset in datasets]
        empty_sources = [index for index, length in enumerate(self.source_lengths) if length <= 0]
        if empty_sources:
            raise ValueError(f"LeRobotConcatDataset child datasets must be non-empty: {empty_sources}")
        self.meta = _MetaWithAggregatedStats(
            datasets[0].meta,
            aggregate_lerobot_dataset_stats(datasets),
        )
        self.repo_id = "+".join(str(getattr(dataset, "repo_id", f"dataset_{index}")) for index, dataset in enumerate(datasets))
        self.root = [str(getattr(dataset, "root", "")) for dataset in datasets]
        self.names = list(names or [f"dataset_{index}" for index in range(len(datasets))])
        self.disable_episode_aware_sampler = True
        self.requires_grid_bin_balanced_sampler = all(
            bool(getattr(dataset, "requires_grid_bin_balanced_sampler", False))
            for dataset in datasets
        )
        self.requires_dataset_balanced_sampler = not self.requires_grid_bin_balanced_sampler

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

    def __getitem__(self, index: int) -> Any:
        item = super().__getitem__(index)
        if isinstance(item, dict):
            source = self.source_for_index(index)
            item = dict(item)
            item["dataset_index"] = int(source["dataset_index"])
            item["dataset_name"] = str(source["dataset_name"])
            item["dataset_local_index"] = int(source["local_index"])
        return item

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

    def make_grid_bin_balanced_sampler(
        self,
        *,
        num_samples: int | None = None,
        drop_n_last_frames: int = 0,
        generator: torch.Generator | None = None,
    ) -> torch.utils.data.WeightedRandomSampler:
        if not self.requires_grid_bin_balanced_sampler:
            raise ValueError("all child datasets must provide grid-bin sidecars for grid-balanced sampling")
        child_weights = [
            dataset.grid_bin_sample_weights(drop_n_last_frames=drop_n_last_frames)
            for dataset in self.datasets
        ]
        weights = torch.cat([
            weights / max(float(weights.sum()), 1.0)
            for weights in child_weights
        ]).to(dtype=torch.double)
        weights = weights / max(float(weights.sum()), 1.0)
        return torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=int(num_samples or len(self)),
            replacement=True,
            generator=generator,
        )

    def __getattr__(self, name: str) -> Any:
        if name == "datasets" or "datasets" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.datasets[0], name)


class GridBinBalancedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that samples frames uniformly across object-position bins."""

    def __init__(self, dataset: Any, sidecar_path: Path) -> None:
        self.dataset = dataset
        self.sidecar_path = Path(sidecar_path)
        if not self.sidecar_path.exists():
            raise FileNotFoundError(f"grid bin sidecar not found: {self.sidecar_path}")
        table = pd.read_parquet(self.sidecar_path)
        required = {"episode_index", "visible", "grid_bin"}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"grid bin sidecar missing columns: {missing}")
        self._episode_to_bin = {
            int(row.episode_index): int(row.grid_bin)
            for row in table.itertuples(index=False)
            if bool(row.visible) or int(row.grid_bin) == -1
        }
        self.disable_episode_aware_sampler = True
        self.requires_grid_bin_balanced_sampler = True

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    def __getattr__(self, name: str) -> Any:
        if name == "dataset" or "dataset" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def make_grid_bin_balanced_sampler(
        self,
        *,
        num_samples: int | None = None,
        drop_n_last_frames: int = 0,
        generator: torch.Generator | None = None,
    ) -> torch.utils.data.WeightedRandomSampler:
        weights = self.grid_bin_sample_weights(drop_n_last_frames=drop_n_last_frames)
        if float(weights.sum()) <= 0.0:
            raise ValueError(f"grid bin sidecar has no sampleable frames: {self.sidecar_path}")
        return torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=int(num_samples or len(self)),
            replacement=True,
            generator=generator,
        )

    def grid_bin_sample_weights(self, *, drop_n_last_frames: int = 0) -> torch.Tensor:
        weights = torch.zeros(len(self), dtype=torch.double)
        bin_to_indices: dict[int, list[int]] = {}
        for row in _iter_episode_rows(getattr(self.meta, "episodes")):
            episode = int(row["episode_index"])
            grid_bin = self._episode_to_bin.get(episode)
            if grid_bin is None:
                continue
            start = int(row["dataset_from_index"])
            stop = int(row["dataset_to_index"]) - max(0, int(drop_n_last_frames))
            for index in range(start, max(start, stop)):
                bin_to_indices.setdefault(grid_bin, []).append(index)
        occupied = [indices for indices in bin_to_indices.values() if indices]
        if not occupied:
            return weights
        bin_mass = 1.0 / len(occupied)
        for indices in occupied:
            sample_weight = bin_mass / len(indices)
            weights[indices] = sample_weight
        return weights


def _iter_episode_rows(episodes: Any) -> list[dict[str, Any]]:
    if hasattr(episodes, "itertuples"):
        return [row._asdict() for row in episodes.itertuples(index=False)]
    return [dict(episodes[index]) for index in range(len(episodes))]


def aggregate_lerobot_dataset_stats(datasets: Sequence[Any]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate child LeRobot metadata stats for virtual merged training.

    LeRobot normalizers consume ``dataset.meta.stats`` even when the samples are
    read from a virtual concat dataset. Using the first child stats is unsafe
    when a primitive has a near-constant action dimension, such as the closed
    gripper in move-only data.
    """

    if not datasets:
        raise ValueError("cannot aggregate stats for an empty dataset list")
    stats_list = []
    for index, dataset in enumerate(datasets):
        meta = getattr(dataset, "meta", None)
        stats = getattr(meta, "stats", None)
        if not isinstance(stats, dict) or not stats:
            raise ValueError(f"dataset_{index}: missing LeRobot metadata stats")
        stats_list.append(_stats_to_numpy(stats))
    try:
        from lerobot.datasets.compute_stats import aggregate_stats
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("LeRobot is required to aggregate virtual dataset stats") from exc
    return aggregate_stats(stats_list)


def _stats_to_numpy(stats: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    converted: dict[str, dict[str, np.ndarray]] = {}
    for feature_key, feature_stats in stats.items():
        if not isinstance(feature_stats, dict):
            continue
        converted[feature_key] = {}
        for stat_key, value in feature_stats.items():
            converted[feature_key][stat_key] = np.asarray(value)
    return converted


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
