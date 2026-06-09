#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess_object_relocation(
    *,
    before_image: Path,
    after_image: Path,
    output: Path | None = None,
    target_direction: str = "right",
    min_delta_px: float = 40.0,
    min_object_area_px: int = 800,
    color_preset: str = "green",
) -> dict[str, Any]:
    import cv2

    before = cv2.imread(str(before_image))
    after = cv2.imread(str(after_image))
    if before is None:
        raise ValueError(f"failed to read before image: {before_image}")
    if after is None:
        raise ValueError(f"failed to read after image: {after_image}")
    if before.shape != after.shape:
        raise ValueError(f"image shape mismatch: before={before.shape}, after={after.shape}")

    before_object = _largest_colored_object(before, min_area_px=min_object_area_px, color_preset=color_preset)
    after_object = _largest_colored_object(after, min_area_px=min_object_area_px, color_preset=color_preset)
    result: dict[str, Any] = {
        "before_image": str(before_image),
        "after_image": str(after_image),
        "image_shape": list(after.shape),
        "target_direction": target_direction,
        "min_delta_px": min_delta_px,
        "min_object_area_px": min_object_area_px,
        "color_preset": color_preset,
        "before_object": before_object,
        "after_object": after_object,
        "notes": [
            "Image-space object relocation verifier for real SO-100 task outcomes.",
            "This is a verifier signal for the agentic layer; it is not a force/contact oracle.",
        ],
    }
    result.update(_classify(before_object, after_object, target_direction, min_delta_px))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _largest_colored_object(image: Any, *, min_area_px: int, color_preset: str) -> dict[str, Any] | None:
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color_preset == "green":
        lower = np.array([35, 45, 35], dtype=np.uint8)
        upper = np.array([90, 255, 255], dtype=np.uint8)
    else:
        raise ValueError(f"unsupported color_preset={color_preset!r}")
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
    target_direction: str,
    min_delta_px: float,
) -> dict[str, Any]:
    if before_object is None or after_object is None:
        return {
            "status": "blocked",
            "relocation_outcome": "object_not_visible",
            "task_success_candidate": False,
            "delta_px": None,
            "signed_goal_delta_px": None,
        }
    dx = round(float(after_object["center_px"][0]) - float(before_object["center_px"][0]), 3)
    dy = round(float(after_object["center_px"][1]) - float(before_object["center_px"][1]), 3)
    signed_goal_delta = _signed_goal_delta(dx=dx, dy=dy, target_direction=target_direction)
    task_success_candidate = signed_goal_delta >= float(min_delta_px)
    if task_success_candidate:
        outcome = f"object_moved_{target_direction}"
    elif abs(dx) < min_delta_px and abs(dy) < min_delta_px:
        outcome = "object_stationary_or_small_motion"
    else:
        outcome = "object_moved_wrong_direction"
    return {
        "status": "passed",
        "relocation_outcome": outcome,
        "task_success_candidate": bool(task_success_candidate),
        "delta_px": {"x": dx, "y": dy},
        "signed_goal_delta_px": round(float(signed_goal_delta), 3),
    }


def _signed_goal_delta(*, dx: float, dy: float, target_direction: str) -> float:
    if target_direction == "right":
        return dx
    if target_direction == "left":
        return -dx
    if target_direction == "down":
        return dy
    if target_direction == "up":
        return -dy
    raise ValueError(f"unsupported target_direction={target_direction!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess real SO-100 object relocation from before/after images.")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-direction", choices=["right", "left", "up", "down"], default="right")
    parser.add_argument("--min-delta-px", type=float, default=40.0)
    parser.add_argument("--min-object-area-px", type=int, default=800)
    parser.add_argument("--color-preset", choices=["green"], default="green")
    args = parser.parse_args()
    print(
        json.dumps(
            assess_object_relocation(
                before_image=args.before,
                after_image=args.after,
                output=args.output,
                target_direction=args.target_direction,
                min_delta_px=args.min_delta_px,
                min_object_area_px=args.min_object_area_px,
                color_preset=args.color_preset,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
