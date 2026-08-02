#!/usr/bin/env python3
"""Combine verified near, bridge, mid, and outer SO101 grasp evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from physical_ai_agent.so101_workspace_spawn_catalog import (
    load_workspace_spawn_catalog,
)


BAND_PROBABILITIES = {
    "near_10_18cm": 0.32,
    "bridge_18_22cm": 0.24,
    "mid_22_28cm": 0.34,
    "outer_28_30cm": 0.10,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrapped_distance(left: float, right: float, period: float) -> float:
    delta = abs((left - right) % period)
    return min(delta, period - delta)


def _select_spread(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Deterministically spread evidence across radius, angle, and cube face."""
    if count >= len(rows):
        return sorted(
            rows,
            key=lambda row: (
                float(row["radius_from_base_m"]),
                float(row["angle_from_base_deg"]),
                float(row["object_yaw_deg"]),
            ),
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["radius_from_base_m"]),
            float(row["angle_from_base_deg"]),
            float(row["object_yaw_deg"]),
        ),
    )
    selected = [ordered[0]]
    remaining = ordered[1:]
    radius_span = max(
        1e-9,
        max(float(row["radius_from_base_m"]) for row in rows)
        - min(float(row["radius_from_base_m"]) for row in rows),
    )
    angle_span = max(
        1e-9,
        max(float(row["angle_from_base_deg"]) for row in rows)
        - min(float(row["angle_from_base_deg"]) for row in rows),
    )

    def distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        dr = (
            float(left["radius_from_base_m"])
            - float(right["radius_from_base_m"])
        ) / radius_span
        da = (
            float(left["angle_from_base_deg"])
            - float(right["angle_from_base_deg"])
        ) / angle_span
        left_relative_yaw = (
            float(left["object_yaw_deg"])
            - float(left["angle_from_base_deg"])
        ) % 90.0
        right_relative_yaw = (
            float(right["object_yaw_deg"])
            - float(right["angle_from_base_deg"])
        ) % 90.0
        dy = _wrapped_distance(left_relative_yaw, right_relative_yaw, 90.0) / 45.0
        return math.sqrt(dr * dr + da * da + dy * dy)

    while len(selected) < count:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                min(distance(remaining[index], prior) for prior in selected),
                -index,
            ),
        )
        selected.append(remaining.pop(best_index))
    return sorted(
        selected,
        key=lambda row: (
            float(row["radius_from_base_m"]),
            float(row["angle_from_base_deg"]),
            float(row["object_yaw_deg"]),
        ),
    )


def _candidate_to_cell(candidate: Any) -> dict[str, Any]:
    return {
        "point_id": str(candidate.candidate_id),
        "world_xy_m": [float(value) for value in candidate.world_xy_m],
        "base_xy_m": [float(value) for value in candidate.base_xy_m],
        "radius_from_base_m": float(candidate.radius_from_base_m),
        "angle_from_base_deg": float(candidate.angle_from_base_deg),
        "object_yaw_deg": float(candidate.object_yaw_deg),
        "camera1_grid_bin": candidate.camera1_grid_bin,
    }


def _load_cells(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "so101_grasp_workspace_catalog_v1":
        raise ValueError(f"expected grasp workspace catalog: {path}")
    return payload, [dict(row) for row in payload.get("cells") or []]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--mid", type=Path, required=True)
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--near-cells", type=int, default=100)
    args = parser.parse_args()

    near_catalog = load_workspace_spawn_catalog(args.near)
    near_rows = [
        _candidate_to_cell(row)
        for row in near_catalog.candidates
        if float(row.radius_from_base_m) <= 0.1800001
        and row.camera1_grid_bin is not None
    ]
    near_rows = _select_spread(near_rows, int(args.near_cells))
    bridge_payload, bridge_rows = _load_cells(args.bridge)
    mid_payload, mid_rows = _load_cells(args.mid)
    outer_payload, outer_rows = _load_cells(args.outer)

    bands = {
        "near_10_18cm": near_rows,
        "bridge_18_22cm": [
            row
            for row in bridge_rows
            if 0.1800001 < float(row["radius_from_base_m"]) <= 0.2200001
        ],
        "mid_22_28cm": [
            row
            for row in mid_rows
            if 0.2200001 < float(row["radius_from_base_m"]) <= 0.2800001
        ],
        "outer_28_30cm": [
            row
            for row in outer_rows
            if 0.2800001 < float(row["radius_from_base_m"]) <= 0.3000001
        ],
    }
    empty = [name for name, rows in bands.items() if not rows]
    if empty:
        raise ValueError(f"continuous evidence bands are empty: {empty}")

    cells: list[dict[str, Any]] = []
    for band_name, rows in bands.items():
        per_cell_weight = BAND_PROBABILITIES[band_name] / len(rows)
        for index, source in enumerate(rows):
            world_xy = [float(value) for value in source["world_xy_m"]]
            cells.append(
                {
                    "point_id": f"{band_name}-{index:03d}",
                    "source_point_id": str(source["point_id"]),
                    "source_band": band_name,
                    "world_xy_m": world_xy,
                    "base_xy_m": [float(value) for value in source["base_xy_m"]],
                    "radius_from_base_m": float(source["radius_from_base_m"]),
                    "angle_from_base_deg": float(source["angle_from_base_deg"]),
                    "object_yaw_deg": float(source["object_yaw_deg"]),
                    "camera1_grid_bin": int(source["camera1_grid_bin"]),
                    "cell_area_m2": float(per_cell_weight),
                    "uniform_area_weight": float(per_cell_weight),
                }
            )

    base_contract = dict(mid_payload["base_contract"])
    origin = [float(value) for value in base_contract["world_xyz_m"][:2]]
    if any(
        not math.isclose(
            float(near_catalog.base_origin_world_xy_m[index]),
            origin[index],
            abs_tol=1e-9,
        )
        for index in range(2)
    ):
        raise ValueError("near and mid evidence use different base origins")
    for payload in (bridge_payload, outer_payload):
        if payload["camera_rig_config"] != mid_payload["camera_rig_config"]:
            raise ValueError("continuous evidence uses mismatched camera rigs")

    result = {
        "format": "so101_grasp_workspace_catalog_v1",
        "catalog_id": args.catalog_id,
        "description": (
            "Seed-free jointly teacher-feasible evidence for the unified "
            "10-30 cm continuous SO101 grasp teacher."
        ),
        "camera_rig_config": mid_payload["camera_rig_config"],
        "home_qpos": [float(value) for value in mid_payload["home_qpos"]],
        "object_color": str(mid_payload["object_color"]),
        "object_half_size_m": float(mid_payload["object_half_size_m"]),
        "base_contract": base_contract,
        "cell_count": len(cells),
        "cells": cells,
        "band_probabilities": BAND_PROBABILITIES,
        "band_cell_counts": {
            name: sum(row["source_band"] == name for row in cells)
            for name in bands
        },
        "source_catalogs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (args.near, args.bridge, args.mid, args.outer)
        ],
        "selection_contract": {
            "radius_min_m": 0.10,
            "radius_max_m": 0.30,
            "outer_boundary_30_5cm_excluded": True,
            "near_evidence_selection": "deterministic_farthest_point_radius_angle_relative_yaw",
            "source_frames_actions_states_or_seeds_reused": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "cell_count": len(cells),
                "band_cell_counts": result["band_cell_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
