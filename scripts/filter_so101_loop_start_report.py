#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def allowed_bins_from_sidecars(sidecar_paths: list[Path]) -> set[int]:
    allowed: set[int] = set()
    for sidecar_path in sidecar_paths:
        table = pd.read_parquet(sidecar_path)
        missing = {"visible", "grid_bin"} - set(table.columns)
        if missing:
            raise ValueError(f"{sidecar_path}: missing columns {sorted(missing)}")
        visible = table[(table["visible"] == True) & (table["grid_bin"] >= 0)]  # noqa: E712
        allowed.update(int(value) for value in visible["grid_bin"].unique())
    return allowed


def filter_loop_start_report(
    *,
    source_report: Path,
    output_report: Path,
    train_grid_bin_sidecars: list[Path] | None = None,
    allowed_grid_bins: list[int] | None = None,
    source_camera1_grid_bin_sidecar: Path | None = None,
    source_camera2_grid_bin_sidecar: Path | None = None,
    camera1_min_area: int,
    camera2_min_area: int,
    grid_size: int = 4,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    allowed_bins = set(int(value) for value in (allowed_grid_bins or []))
    if train_grid_bin_sidecars:
        allowed_bins.update(allowed_bins_from_sidecars(train_grid_bin_sidecars))
    if not allowed_bins:
        raise ValueError("no allowed train grid bins found")

    report = json.loads(source_report.read_text(encoding="utf-8"))
    episodes = report.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{source_report}: missing episodes list")
    source_camera1_bins = _sidecar_rows_by_episode(source_camera1_grid_bin_sidecar)
    source_camera2_bins = _sidecar_rows_by_episode(source_camera2_grid_bin_sidecar)

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source_index, episode in enumerate(episodes):
        decision = _loop_start_decision(
            episode,
            source_index=source_index,
            allowed_bins=allowed_bins,
            source_camera1_bin=source_camera1_bins.get(source_index),
            source_camera2_bin=source_camera2_bins.get(source_index),
            camera1_min_area=camera1_min_area,
            camera2_min_area=camera2_min_area,
            grid_size=grid_size,
        )
        if decision["accepted"]:
            copy = dict(episode)
            copy["loop_source_episode_index"] = int(source_index)
            copy["loop_camera1_grid_bin_0_based"] = int(decision["grid_bin"])
            copy["loop_filter_decision"] = decision
            selected.append(copy)
            if max_episodes is not None and len(selected) >= max_episodes:
                break
        else:
            rejected.append(decision)

    if not selected:
        raise ValueError(
            f"{source_report}: filter selected no episodes; allowed_bins={sorted(allowed_bins)} "
            f"camera1_min_area={camera1_min_area} camera2_min_area={camera2_min_area}"
        )

    filtered = dict(report)
    filtered["episodes"] = selected
    filtered["requested_episodes"] = len(selected)
    filtered["exported_episodes"] = len(selected)
    filtered["report_path"] = str(output_report)
    filtered["loop_filter"] = {
        "source_report": str(source_report),
        "train_grid_bin_sidecars": [str(path) for path in (train_grid_bin_sidecars or [])],
        "source_camera1_grid_bin_sidecar": str(source_camera1_grid_bin_sidecar) if source_camera1_grid_bin_sidecar else None,
        "source_camera2_grid_bin_sidecar": str(source_camera2_grid_bin_sidecar) if source_camera2_grid_bin_sidecar else None,
        "allowed_grid_bins_0_based": sorted(allowed_bins),
        "camera1_min_area": int(camera1_min_area),
        "camera2_min_area": int(camera2_min_area),
        "grid_size": int(grid_size),
        "selected_episodes": len(selected),
        "source_episodes": len(episodes),
        "rejected_episodes": len(episodes) - len(selected),
        "rejected_sample": rejected[:20],
        "contract": "closed-loop starts must use train-seen camera1 grid bins and visible policy-camera targets",
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(filtered, indent=2, sort_keys=True), encoding="utf-8")
    return filtered


def _loop_start_decision(
    episode: dict[str, Any],
    *,
    source_index: int,
    allowed_bins: set[int],
    source_camera1_bin: dict[str, Any] | None = None,
    source_camera2_bin: dict[str, Any] | None = None,
    camera1_min_area: int,
    camera2_min_area: int,
    grid_size: int,
) -> dict[str, Any]:
    # Closed-loop starts replay the first-frame q_start, so the visibility
    # contract must be checked against the actual start frame.  Older reports
    # may also contain a preselected candidate visibility under best_meta, but
    # that can describe a different preselection state and admit invisible
    # starts into loop tests.
    visibility = episode.get("start_policy_camera_visibility")
    if not isinstance(visibility, dict):
        visibility = (episode.get("best_meta", {}) or {}).get("preselected_policy_camera_visibility", {}) or {}
    camera1 = visibility.get("camera1") or {}
    camera2 = visibility.get("camera2") or {}
    camera1_area = int((source_camera1_bin or {}).get("area") or camera1.get("area") or 0)
    camera2_area = int((source_camera2_bin or {}).get("area") or camera2.get("area") or 0)
    grid_bin = (
        int(source_camera1_bin["grid_bin"])
        if source_camera1_bin is not None and int(source_camera1_bin.get("grid_bin", -1)) >= 0
        else _episode_camera1_grid_bin(episode, camera1=camera1, grid_size=grid_size)
    )
    camera1_visible = bool((source_camera1_bin or {}).get("visible", camera1.get("visible")))
    camera2_visible = bool((source_camera2_bin or {}).get("visible", camera2.get("visible")))
    reasons: list[str] = []
    if not camera1_visible or camera1_area < camera1_min_area:
        reasons.append("camera1_target_not_visible_enough")
    if not camera2_visible or camera2_area < camera2_min_area:
        reasons.append("camera2_target_not_visible_enough")
    if grid_bin is None or grid_bin not in allowed_bins:
        reasons.append("camera1_grid_bin_not_used_by_training")
    return {
        "accepted": not reasons,
        "source_index": int(source_index),
        "seed": episode.get("seed"),
        "grid_bin": grid_bin,
        "allowed_grid_bins_0_based": sorted(allowed_bins),
        "camera1_area": camera1_area,
        "camera2_area": camera2_area,
        "camera1_bbox": camera1.get("bbox"),
        "camera2_bbox": camera2.get("bbox"),
        "source_camera1_sidecar": _jsonable_sidecar_row(source_camera1_bin),
        "source_camera2_sidecar": _jsonable_sidecar_row(source_camera2_bin),
        "reasons": reasons,
    }


def _episode_camera1_grid_bin(episode: dict[str, Any], *, camera1: dict[str, Any], grid_size: int) -> int | None:
    explicit = episode.get("loop_camera1_grid_bin_0_based")
    if explicit is not None:
        return int(explicit)
    centroid = camera1.get("centroid")
    if isinstance(centroid, list | tuple) and len(centroid) >= 2:
        x = min(grid_size - 1, max(0, int(float(centroid[0]) / 256.0 * grid_size)))
        y = min(grid_size - 1, max(0, int(float(centroid[1]) / 256.0 * grid_size)))
        return y * grid_size + x
    bbox = camera1.get("bbox")
    if isinstance(bbox, list | tuple) and len(bbox) >= 4:
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        x = min(grid_size - 1, max(0, int(cx / 256.0 * grid_size)))
        y = min(grid_size - 1, max(0, int(cy / 256.0 * grid_size)))
        return y * grid_size + x
    return None


def _sidecar_rows_by_episode(sidecar_path: Path | None) -> dict[int, dict[str, Any]]:
    if sidecar_path is None:
        return {}
    table = pd.read_parquet(sidecar_path)
    missing = {"episode_index", "visible", "grid_bin", "area"} - set(table.columns)
    if missing:
        raise ValueError(f"{sidecar_path}: missing columns {sorted(missing)}")
    rows: dict[int, dict[str, Any]] = {}
    for row in table.to_dict(orient="records"):
        rows[int(row["episode_index"])] = row
    return rows


def _jsonable_sidecar_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            out[key] = None
        elif isinstance(value, float):
            out[key] = float(value)
        elif isinstance(value, int | bool | str):
            out[key] = value
        else:
            try:
                out[key] = int(value)
            except Exception:
                out[key] = str(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--train-grid-bin-sidecar", type=Path, action="append")
    parser.add_argument("--allowed-grid-bin", type=int, action="append")
    parser.add_argument("--source-camera1-grid-bin-sidecar", type=Path)
    parser.add_argument("--source-camera2-grid-bin-sidecar", type=Path)
    parser.add_argument("--camera1-min-area", type=int, default=60)
    parser.add_argument("--camera2-min-area", type=int, default=80)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    filtered = filter_loop_start_report(
        source_report=args.source_report,
        output_report=args.output_report,
        train_grid_bin_sidecars=args.train_grid_bin_sidecar,
        allowed_grid_bins=args.allowed_grid_bin,
        source_camera1_grid_bin_sidecar=args.source_camera1_grid_bin_sidecar,
        source_camera2_grid_bin_sidecar=args.source_camera2_grid_bin_sidecar,
        camera1_min_area=args.camera1_min_area,
        camera2_min_area=args.camera2_min_area,
        grid_size=args.grid_size,
        max_episodes=args.max_episodes,
    )
    loop_filter = filtered["loop_filter"]
    print(
        json.dumps(
            {
                "output_report": str(args.output_report),
                "selected_episodes": loop_filter["selected_episodes"],
                "source_episodes": loop_filter["source_episodes"],
                "allowed_grid_bins_0_based": loop_filter["allowed_grid_bins_0_based"],
                "camera1_min_area": loop_filter["camera1_min_area"],
                "camera2_min_area": loop_filter["camera2_min_area"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
