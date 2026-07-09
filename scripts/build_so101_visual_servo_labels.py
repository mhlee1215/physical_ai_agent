#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build image-space visual-servo labels for SO101 LeRobot datasets without modifying the dataset."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--camera-key", action="append", dest="camera_keys")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--stop-action-norm", type=float, default=1e-4)
    args = parser.parse_args()
    report = build_visual_servo_labels(
        dataset_root=args.dataset_root,
        camera_keys=tuple(args.camera_keys or ("observation.images.camera1", "observation.images.camera2")),
        action_key=args.action_key,
        min_area=args.min_area,
        stop_action_norm=args.stop_action_norm,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def build_visual_servo_labels(
    *,
    dataset_root: Path,
    camera_keys: tuple[str, ...],
    action_key: str,
    min_area: int,
    stop_action_norm: float,
) -> dict[str, Any]:
    import pandas as pd

    data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise SystemExit(f"no parquet data files under {dataset_root / 'data'}")

    records: list[dict[str, Any]] = []
    visible = {camera_key: 0 for camera_key in camera_keys}
    stop = 0
    for path in data_files:
        columns = ["episode_index", "frame_index", "index", *camera_keys, action_key]
        for _, row in pd.read_parquet(path, columns=columns).iterrows():
            episode_index = int(row["episode_index"])
            frame_index = int(row["frame_index"])
            dataset_index = int(row["index"])
            action = np.asarray(row[action_key], dtype=np.float32)
            stop_label = bool(float(np.linalg.norm(action)) <= float(stop_action_norm))
            stop += int(stop_label)
            record = {
                "index": dataset_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "stop_label": stop_label,
                "action_norm": float(np.linalg.norm(action)),
            }
            for camera_key in camera_keys:
                prefix = _camera_prefix(camera_key)
                label = (
                    extract_egocentric_visual_servo_label(row[camera_key], min_area=min_area)
                    if prefix == "camera1"
                    else extract_visual_servo_label(row[camera_key], min_area=min_area)
                )
                visible[camera_key] += int(label["visible"])
                record.update(
                    {
                        f"{prefix}_visible": bool(label["visible"]),
                        f"{prefix}_dx_norm": float(label["dx_norm"]),
                        f"{prefix}_dy_norm": float(label["dy_norm"]),
                        f"{prefix}_edge_angle_error": float(label["edge_angle_error"]),
                        f"{prefix}_target_area": int(label["target_area"]),
                        f"{prefix}_target_angle_rad": float(label["target_angle_rad"]),
                    }
                )
            records.append(record)

    table = pd.DataFrame(records).sort_values(["episode_index", "frame_index", "index"])
    out_dir = dataset_root / "meta" / "visual_servo_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{'_'.join(_camera_prefix(camera_key) for camera_key in camera_keys)}_green_cube"
    parquet_path = out_dir / f"{stem}.parquet"
    report_path = out_dir / f"{stem}.json"
    table.to_parquet(parquet_path, index=False)
    report = {
        "dataset_root": str(dataset_root),
        "camera_keys": list(camera_keys),
        "action_key": action_key,
        "rows": int(len(table)),
        "visible_rows": {key: int(value) for key, value in visible.items()},
        "visible_rate": {key: float(value / max(1, len(table))) for key, value in visible.items()},
        "stop_rows": int(stop),
        "stop_rate": float(stop / max(1, len(table))),
        "min_area": int(min_area),
        "stop_action_norm": float(stop_action_norm),
        "parquet_path": str(parquet_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def extract_visual_servo_label(image_value: object, *, min_area: int) -> dict[str, Any]:
    image = _decode_rgb(image_value)
    mask = _green_mask(image)
    ys, xs = np.where(mask)
    if len(xs) < int(min_area):
        return {
            "visible": False,
            "dx_norm": 0.0,
            "dy_norm": 0.0,
            "edge_angle_error": 0.0,
            "target_area": 0,
            "target_angle_rad": 0.0,
        }
    height, width = image.shape[:2]
    cx = float(xs.mean())
    cy = float(ys.mean())
    dx = ((cx - ((width - 1) * 0.5)) / max(1.0, (width - 1) * 0.5))
    dy = ((cy - ((height - 1) * 0.5)) / max(1.0, (height - 1) * 0.5))
    angle = _mask_major_axis_angle(xs, ys)
    # Normalize the image-space jaw/edge angle to roughly [-1, 1].
    angle_error = float(np.clip(angle / (math.pi * 0.5), -1.0, 1.0))
    return {
        "visible": True,
        "dx_norm": float(dx),
        "dy_norm": float(dy),
        "edge_angle_error": angle_error,
        "target_area": int(len(xs)),
        "target_angle_rad": float(angle),
    }


def extract_egocentric_visual_servo_label(image_value: object, *, min_area: int) -> dict[str, Any]:
    label = extract_visual_servo_label(image_value, min_area=min_area)
    if not label["visible"]:
        return label
    image = _decode_rgb(image_value)
    green = _green_mask(image)
    yellow = _yellow_robot_mask(image)
    green_ys, green_xs = np.where(green)
    yellow_ys, yellow_xs = np.where(yellow)
    if len(yellow_xs) < int(min_area):
        return label
    target = np.asarray([float(green_xs.mean()), float(green_ys.mean())])
    robot_points = np.stack([yellow_xs.astype(np.float64), yellow_ys.astype(np.float64)], axis=1)
    distances = np.linalg.norm(robot_points - target[None, :], axis=1)
    nearest_count = max(1, min(200, len(robot_points)))
    gripper_proxy = robot_points[np.argpartition(distances, nearest_count - 1)[:nearest_count]].mean(axis=0)
    height, width = image.shape[:2]
    # ponytail: camera1 is not wrist-local; use target minus nearest visible robot pixels as a gripper proxy.
    label["dx_norm"] = float(np.clip((target[0] - gripper_proxy[0]) / max(1.0, (width - 1) * 0.5), -1.0, 1.0))
    label["dy_norm"] = float(np.clip((target[1] - gripper_proxy[1]) / max(1.0, (height - 1) * 0.5), -1.0, 1.0))
    label["gripper_proxy_x"] = float(gripper_proxy[0])
    label["gripper_proxy_y"] = float(gripper_proxy[1])
    return label


def extract_wrist_visual_servo_label(image_value: object, *, min_area: int) -> dict[str, Any]:
    label = extract_visual_servo_label(image_value, min_area=min_area)
    return {
        **label,
        "wrist_dx_norm": label["dx_norm"],
        "wrist_dy_norm": label["dy_norm"],
    }


def _camera_prefix(camera_key: str) -> str:
    return camera_key.rsplit(".", maxsplit=1)[-1]


def _decode_rgb(image_value: object) -> np.ndarray:
    if isinstance(image_value, dict):
        blob = image_value.get("bytes")
    else:
        blob = bytes(image_value)  # type: ignore[arg-type]
    if not blob:
        raise ValueError("empty image payload")
    return np.asarray(Image.open(BytesIO(blob)).convert("RGB"), dtype=np.int16)


def _green_mask(image: np.ndarray) -> np.ndarray:
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (green > 80) & (green > red + 25) & (green > blue + 20)


def _yellow_robot_mask(image: np.ndarray) -> np.ndarray:
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (red > 120) & (green > 90) & (blue < 100) & (red > green * 0.75) & (green > red * 0.45)


def _mask_major_axis_angle(xs: np.ndarray, ys: np.ndarray) -> float:
    points = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))]
    angle = math.atan2(float(axis[1]), float(axis[0]))
    while angle <= -math.pi * 0.5:
        angle += math.pi
    while angle > math.pi * 0.5:
        angle -= math.pi
    return angle

if __name__ == "__main__":
    main()
