#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess_grasp_outcome(
    *,
    before_image: Path,
    after_image: Path,
    output: Path | None = None,
    min_object_area_px: int = 800,
    stable_center_px: float = 8.0,
    stable_iou: float = 0.92,
    changed_pixel_threshold: int = 10,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    before = cv2.imread(str(before_image))
    after = cv2.imread(str(after_image))
    if before is None:
        raise ValueError(f"failed to read before image: {before_image}")
    if after is None:
        raise ValueError(f"failed to read after image: {after_image}")
    if before.shape != after.shape:
        raise ValueError(f"image shape mismatch: before={before.shape}, after={after.shape}")

    before_object = _largest_green_object(before, min_area_px=min_object_area_px)
    after_object = _largest_green_object(after, min_area_px=min_object_area_px)
    diff = cv2.absdiff(before, after)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    changed_mask = diff_gray > changed_pixel_threshold

    result: dict[str, Any] = {
        "before_image": str(before_image),
        "after_image": str(after_image),
        "image_shape": list(after.shape),
        "min_object_area_px": min_object_area_px,
        "stable_center_px": stable_center_px,
        "stable_iou": stable_iou,
        "changed_pixel_threshold": changed_pixel_threshold,
        "before_object": before_object,
        "after_object": after_object,
        "visual_motion": {
            "mean_absdiff": round(float(diff_gray.mean()), 4),
            "changed_pixel_ratio": round(float(changed_mask.mean()), 6),
        },
        "notes": [
            "Camera-2 image-space grasp outcome verifier.",
            "This is a conservative retry signal, not a physical force/contact oracle.",
        ],
    }
    result.update(_classify(before_object, after_object, changed_mask, stable_center_px, stable_iou))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _largest_green_object(image: Any, *, min_area_px: int) -> dict[str, Any] | None:
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 45, 35], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
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
    if not rows:
        return None
    return max(rows, key=lambda row: row["area_px"])


def _classify(
    before_object: dict[str, Any] | None,
    after_object: dict[str, Any] | None,
    changed_mask: Any,
    stable_center_px: float,
    stable_iou: float,
) -> dict[str, Any]:
    if before_object is None or after_object is None:
        return {
            "status": "blocked",
            "grasp_outcome": "object_not_visible",
            "object_stationary": None,
            "object_center_delta_px": None,
            "object_bbox_iou": None,
            "object_changed_pixel_ratio": None,
        }

    center_delta = _center_delta(before_object["center_px"], after_object["center_px"])
    iou = _bbox_iou(before_object["bbox_xyxy"], after_object["bbox_xyxy"])
    object_mask = _bbox_mask(changed_mask.shape, before_object["bbox_xyxy"]) | _bbox_mask(
        changed_mask.shape,
        after_object["bbox_xyxy"],
    )
    object_changed_ratio = float((changed_mask & object_mask).sum()) / max(float(object_mask.sum()), 1.0)
    object_stationary = bool(center_delta <= stable_center_px and iou >= stable_iou)
    if object_stationary:
        outcome = "grasp_failed_object_stationary"
    elif center_delta >= stable_center_px * 2.0 or iou < stable_iou:
        outcome = "object_moved_or_occluded_candidate"
    else:
        outcome = "ambiguous_object_motion"
    return {
        "status": "passed",
        "grasp_outcome": outcome,
        "object_stationary": object_stationary,
        "object_center_delta_px": round(center_delta, 3),
        "object_bbox_iou": round(iou, 4),
        "object_changed_pixel_ratio": round(object_changed_ratio, 6),
    }


def _center_delta(a: list[float], b: list[float]) -> float:
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _bbox_mask(shape: tuple[int, ...], bbox: list[int]) -> Any:
    import numpy as np

    x1, y1, x2, y2 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess SO-100 camera-2 before/after grasp outcome.")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-object-area-px", type=int, default=800)
    args = parser.parse_args()
    print(
        json.dumps(
            assess_grasp_outcome(
                before_image=args.before,
                after_image=args.after,
                output=args.output,
                min_object_area_px=args.min_object_area_px,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
