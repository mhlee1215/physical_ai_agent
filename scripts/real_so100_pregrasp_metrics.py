#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def measure_green_contours(
    *,
    image_path: Path,
    output: Path | None = None,
    min_area_px: int = 300,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 45, 35], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_rows: list[dict[str, Any]] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area_px:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        contour_rows.append(
            {
                "area_px": area,
                "bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
                "center_px": [round(float(x + w / 2), 2), round(float(y + h / 2), 2)],
            }
        )
    contour_rows.sort(key=lambda row: row["area_px"], reverse=True)

    doll = _select_doll_candidate(contour_rows)
    robot_candidates = _select_robot_green_candidates(contour_rows, doll)
    gap = _gap_to_nearest_robot_candidate(doll, robot_candidates)
    lower_tip = _select_lower_tip_candidate(doll, robot_candidates)
    result = {
        "image_path": str(image_path),
        "image_shape": list(image.shape),
        "min_area_px": min_area_px,
        "green_contours": contour_rows,
        "doll_candidate": doll,
        "robot_green_candidates": robot_candidates,
        "nearest_robot_gap": gap,
        "lower_tip_candidate": lower_tip,
        "notes": [
            "Heuristic camera-2 image-space metric only; green robot parts can be confused with the object.",
            "Use as an agentic progress signal, not a contact or grasp-success oracle.",
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _select_doll_candidate(contours: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not contours:
        return None
    left_half = [row for row in contours if row["center_px"][0] < 800]
    if left_half:
        return max(left_half, key=lambda row: row["area_px"])
    return contours[0]


def _select_robot_green_candidates(
    contours: list[dict[str, Any]],
    doll: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if doll is None:
        return []
    doll_box = doll["bbox_xyxy"]
    return [
        row
        for row in contours
        if row is not doll and row["bbox_xyxy"][0] >= doll_box[0]
    ]


def _gap_to_nearest_robot_candidate(
    doll: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if doll is None or not candidates:
        return None
    doll_box = doll["bbox_xyxy"]
    rows = []
    for candidate in candidates:
        box = candidate["bbox_xyxy"]
        rows.append(
            {
                "candidate_bbox_xyxy": box,
                "signed_x_gap_px": int(box[0] - doll_box[2]),
                "signed_y_center_gap_px": round(float(candidate["center_px"][1] - doll["center_px"][1]), 2),
                "candidate_area_px": candidate["area_px"],
            }
        )
    return min(rows, key=lambda row: abs(row["signed_x_gap_px"]))


def _select_lower_tip_candidate(
    doll: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if doll is None:
        return None
    doll_box = doll["bbox_xyxy"]
    doll_center_y = float(doll["center_px"][1])
    rows = []
    for candidate in candidates:
        box = candidate["bbox_xyxy"]
        center_y = float(candidate["center_px"][1])
        if candidate["area_px"] > 8000:
            continue
        if not (doll_box[1] <= center_y <= doll_box[3] + 80):
            continue
        rows.append(
            {
                "candidate_bbox_xyxy": box,
                "candidate_center_px": candidate["center_px"],
                "candidate_area_px": candidate["area_px"],
                "signed_x_gap_px": int(box[0] - doll_box[2]),
                "signed_y_center_gap_px": round(center_y - doll_center_y, 2),
            }
        )
    if not rows:
        return None
    return min(rows, key=lambda row: (abs(row["signed_y_center_gap_px"]), abs(row["signed_x_gap_px"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure camera-2 green contour gaps for SO-100 pre-grasp.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-area-px", type=int, default=300)
    args = parser.parse_args()
    print(
        json.dumps(
            measure_green_contours(
                image_path=args.image,
                output=args.output,
                min_area_px=args.min_area_px,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
