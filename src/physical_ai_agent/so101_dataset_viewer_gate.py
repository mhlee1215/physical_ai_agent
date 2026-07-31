"""Completion gate for recipe-backed datasets exposed by the Robot Experiment Manager."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from physical_ai_agent.so101_dataset_registry import DatasetRegistryEntry


class DatasetViewerGateError(RuntimeError):
    """Raised when a training-ready dataset is not usable through the viewer API."""


@dataclass(frozen=True)
class DatasetViewerGateResult:
    base_url: str
    datasets: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"base_url": self.base_url, "datasets": list(self.datasets)}


def verify_dataset_viewer_api(
    base_url: str,
    entries: Iterable[DatasetRegistryEntry],
    *,
    timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 0.5,
) -> DatasetViewerGateResult:
    """Wait for the viewer and verify catalog plus one real frame for every split."""

    expected = tuple(entries)
    if not expected:
        raise DatasetViewerGateError("no dataset splits were selected for viewer verification")
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return _verify_once(base_url.rstrip("/"), expected)
        except (DatasetViewerGateError, OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(poll_interval_seconds)
    raise DatasetViewerGateError(
        f"dataset viewer did not satisfy the completion contract within "
        f"{timeout_seconds:g}s: {last_error}"
    ) from last_error


def _verify_once(
    base_url: str, entries: tuple[DatasetRegistryEntry, ...]
) -> DatasetViewerGateResult:
    catalog = _read_json(f"{base_url}/api/datasets")
    datasets = catalog.get("datasets") if isinstance(catalog, dict) else None
    if not isinstance(datasets, dict):
        raise DatasetViewerGateError("/api/datasets must return a datasets object")

    verified: list[dict[str, Any]] = []
    for entry in entries:
        summary = datasets.get(entry.catalog_name)
        if not isinstance(summary, dict):
            raise DatasetViewerGateError(
                f"/api/datasets is missing recipe split '{entry.catalog_name}'"
            )
        for field, expected in (("episodes", entry.episodes), ("frames", entry.frames)):
            if expected is not None and int(summary.get(field, -1)) != int(expected):
                raise DatasetViewerGateError(
                    f"{entry.catalog_name} {field} mismatch: "
                    f"viewer={summary.get(field)!r}, registry={expected}"
                )

        query = urllib.parse.urlencode(
            {"split": entry.catalog_name, "episode": 0, "frame": 0}
        )
        frame = _read_json(f"{base_url}/api/frame?{query}")
        if frame.get("split") != entry.catalog_name:
            raise DatasetViewerGateError(
                f"frame split mismatch for {entry.catalog_name}: {frame.get('split')!r}"
            )
        images = frame.get("images")
        if not isinstance(images, dict):
            raise DatasetViewerGateError(
                f"frame payload for {entry.catalog_name} is missing images"
            )
        for camera in ("observation.images.camera1", "observation.images.camera2"):
            image = images.get(camera)
            if not isinstance(image, str) or not image.startswith("data:image/"):
                raise DatasetViewerGateError(
                    f"frame payload for {entry.catalog_name} is missing {camera}"
                )
        prompt = frame.get("prompt") or frame.get("task")
        if not isinstance(prompt, str) or not prompt.strip():
            raise DatasetViewerGateError(
                f"frame payload for {entry.catalog_name} has no prompt"
            )
        verified.append(
            {
                "split": entry.catalog_name,
                "episodes": entry.episodes,
                "frames": entry.frames,
                "frame": 0,
                "prompt": prompt,
                "cameras": [
                    "observation.images.camera1",
                    "observation.images.camera2",
                ],
            }
        )
    return DatasetViewerGateResult(base_url=base_url, datasets=tuple(verified))


def _read_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise DatasetViewerGateError(f"{url} returned HTTP {response.status}")
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise DatasetViewerGateError(f"{url} did not return a JSON object")
    return payload
