#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


IK_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]


def build_visual_ik_forward_model(
    *,
    manifests: list[Path],
    output: Path,
    min_rows: int,
    ridge: float,
) -> dict[str, Any]:
    rows = []
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        rows.extend(_rows_from_manifest(manifest, payload))

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group_key"], []).append(row)

    models = {}
    for key, group_rows in sorted(groups.items()):
        if len(group_rows) < min_rows:
            continue
        models[key] = _fit_group(group_rows, ridge=ridge)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_visual_ik_forward_model",
        "manifests": [str(path) for path in manifests],
        "ik_joints": IK_JOINTS,
        "row_count": len(rows),
        "group_count": len(groups),
        "fitted_group_count": len(models),
        "min_rows": min_rows,
        "ridge": ridge,
        "models": models,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _rows_from_manifest(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for sample in payload.get("samples", []):
        q = sample.get("readback_raw") or sample.get("commanded_target_raw")
        if not isinstance(q, dict) or not all(joint in q for joint in IK_JOINTS):
            continue
        q_vec = [float(q[joint]) for joint in IK_JOINTS]
        for camera in sample.get("camera_records", []):
            camera_index = int(camera["camera_index"])
            for detection in camera.get("detections", []):
                family = detection.get("family")
                for marker in detection.get("markers", []):
                    marker_id = int(marker["id"])
                    rows.append(
                        {
                            "source_manifest": str(path),
                            "sample_index": int(sample["sample_index"]),
                            "pose_index": int(sample["pose_index"]),
                            "offset_index": int(sample["offset_index"]),
                            "camera_index": camera_index,
                            "family": family,
                            "marker_id": marker_id,
                            "group_key": f"camera_{camera_index}/{family}/id_{marker_id}",
                            "q_raw": q_vec,
                            "center_px": [float(marker["center_px"][0]), float(marker["center_px"][1])],
                            "min_side_px": float(marker.get("min_side_px", 0.0)),
                            "image": camera.get("raw_image"),
                            "overlay_image": camera.get("overlay_image"),
                        }
                    )
    return rows


def _fit_group(rows: list[dict[str, Any]], *, ridge: float) -> dict[str, Any]:
    q = np.asarray([row["q_raw"] for row in rows], dtype=np.float64)
    y = np.asarray([row["center_px"] for row in rows], dtype=np.float64)
    q_mean = q.mean(axis=0)
    q_std = q.std(axis=0)
    q_std[q_std < 1e-6] = 1.0
    x = _features((q - q_mean) / q_std)
    xtx = x.T @ x
    reg = np.eye(xtx.shape[0], dtype=np.float64) * ridge
    reg[0, 0] = 0.0
    coef = np.linalg.solve(xtx + reg, x.T @ y)
    pred = x @ coef
    err = pred - y
    rmse = np.sqrt(np.mean(err * err, axis=0))
    max_abs = np.max(np.abs(err), axis=0)
    return {
        "sample_count": len(rows),
        "camera_index": rows[0]["camera_index"],
        "family": rows[0]["family"],
        "marker_id": rows[0]["marker_id"],
        "feature_schema": _feature_schema(),
        "q_mean_raw": _round_list(q_mean),
        "q_std_raw": _round_list(q_std),
        "coef_px": [[round(float(value), 8) for value in row] for row in coef.tolist()],
        "train_rmse_px": _round_list(rmse),
        "train_max_abs_error_px": _round_list(max_abs),
        "q_bounds_raw": {
            joint: [round(float(q[:, index].min()), 4), round(float(q[:, index].max()), 4)]
            for index, joint in enumerate(IK_JOINTS)
        },
        "rows": rows,
    }


def _features(z: np.ndarray) -> np.ndarray:
    columns = [np.ones((z.shape[0], 1), dtype=np.float64), z]
    squares = z * z
    columns.append(squares)
    pairwise = []
    for left in range(z.shape[1]):
        for right in range(left + 1, z.shape[1]):
            pairwise.append((z[:, left] * z[:, right])[:, None])
    columns.extend(pairwise)
    return np.concatenate(columns, axis=1)


def _feature_schema() -> list[str]:
    names = ["bias"]
    names.extend(IK_JOINTS)
    names.extend([f"{joint}^2" for joint in IK_JOINTS])
    for left in range(len(IK_JOINTS)):
        for right in range(left + 1, len(IK_JOINTS)):
            names.append(f"{IK_JOINTS[left]}*{IK_JOINTS[right]}")
    return names


def _round_list(values: Any) -> list[float]:
    return [round(float(value), 4) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build q->pixel forward models for visual IK from SO-100 marker manifests.")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()
    print(
        json.dumps(
            build_visual_ik_forward_model(
                manifests=args.manifest,
                output=args.output,
                min_rows=args.min_rows,
                ridge=args.ridge,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
