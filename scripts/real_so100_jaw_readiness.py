#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess_jaw_readiness(
    *,
    image_path: Path,
    output: Path | None = None,
    min_object_area_px: int = 800,
    min_jaw_marker_area_px: int = 500,
    edge_margin_px: int = 8,
) -> dict[str, Any]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    green_candidates = _color_candidates(
        image,
        lower_hsv=[35, 45, 35],
        upper_hsv=[90, 255, 255],
        min_area_px=min_object_area_px,
    )
    blue_candidates = _color_candidates(
        image,
        lower_hsv=[90, 35, 20],
        upper_hsv=[135, 255, 255],
        min_area_px=min_jaw_marker_area_px,
    )
    object_candidate = _select_object_candidate(green_candidates)
    jaw_marker_candidate = _select_jaw_marker_candidate(blue_candidates)
    object_edge_clipped = (
        _bbox_touches_edge(object_candidate["bbox_xyxy"], list(image.shape), edge_margin_px)
        if object_candidate
        else None
    )
    blockers: list[str] = []
    if object_candidate is None:
        blockers.append("green object not visible")
    elif object_edge_clipped:
        blockers.append("green object touches image boundary")
    if jaw_marker_candidate is None:
        blockers.append("jaw marker not visible")
    result = {
        "status": "ready" if not blockers else "blocked",
        "image_path": str(image_path),
        "image_shape": list(image.shape),
        "min_object_area_px": min_object_area_px,
        "min_jaw_marker_area_px": min_jaw_marker_area_px,
        "edge_margin_px": edge_margin_px,
        "object_candidate": object_candidate,
        "jaw_marker_candidate": jaw_marker_candidate,
        "object_edge_clipped": object_edge_clipped,
        "green_candidates": green_candidates,
        "blue_candidates": blue_candidates,
        "blockers": blockers,
        "notes": [
            "Camera-0 readiness gate for object-and-jaw framing.",
            "This verifies framing only; it does not prove grasp geometry or contact.",
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _color_candidates(
    image: Any,
    *,
    lower_hsv: list[int],
    upper_hsv: list[int],
    min_area_px: int,
) -> list[dict[str, Any]]:
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8))
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area_px:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rows.append(
            {
                "area_px": area,
                "bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
                "center_px": [round(float(x + w / 2), 2), round(float(y + h / 2), 2)],
            }
        )
    return sorted(rows, key=lambda row: row["area_px"], reverse=True)


def _select_object_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["area_px"])


def _select_jaw_marker_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["area_px"])


def _bbox_touches_edge(bbox: list[int], image_shape: list[int], margin_px: int) -> bool:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    return x1 <= margin_px or y1 <= margin_px or x2 >= width - margin_px or y2 >= height - margin_px


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess SO-100 camera-0 object/jaw framing readiness.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-object-area-px", type=int, default=800)
    parser.add_argument("--min-jaw-marker-area-px", type=int, default=500)
    parser.add_argument("--edge-margin-px", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            assess_jaw_readiness(
                image_path=args.image,
                output=args.output,
                min_object_area_px=args.min_object_area_px,
                min_jaw_marker_area_px=args.min_jaw_marker_area_px,
                edge_margin_px=args.edge_margin_px,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
