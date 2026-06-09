#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_reframe_overlay(
    *,
    image: Path,
    advice: Path,
    output: Path,
    action_index: int = 0,
) -> dict[str, Any]:
    payload = _load_json(advice)
    actions = payload.get("actions", [])
    if action_index >= len(actions):
        raise IndexError(f"action_index {action_index} outside actions length {len(actions)}")
    action = actions[action_index]
    nudge = action.get("image_space_nudge") or payload.get("jaw_object_image_space_nudge")
    if not nudge:
        raise ValueError("reframe advice does not include image_space_nudge")
    overlay = _draw_overlay(image=image, output=output, action=action, nudge=nudge)
    manifest = {
        "status": "passed",
        "operation": "real_so100_reframe_overlay",
        "image": str(image),
        "advice": str(advice),
        "output_image": str(output),
        "action_index": action_index,
        "camera": action.get("camera"),
        "agent_actionable": action.get("agent_actionable"),
        "external_setup_required": action.get("external_setup_required"),
        "diagnostic_summary": action.get("diagnostic_summary"),
        "nudge_instruction": nudge.get("instruction"),
        "recommended_shift_px": nudge.get("recommended_shift_px"),
        "current_center_px": nudge.get("current_center_px"),
        "desired_center_px": nudge.get("desired_center_px"),
        "current_bbox_xyxy": nudge.get("current_bbox_xyxy"),
        "desired_bbox_xyxy": overlay["desired_bbox_xyxy"],
        "target_margin_px": nudge.get("target_margin_px"),
        "purpose": "human-visible observation repair overlay; no robot motion",
    }
    manifest_path = output.with_suffix(".json")
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _draw_overlay(*, image: Path, output: Path, action: dict[str, Any], nudge: dict[str, Any]) -> dict[str, Any]:
    import cv2

    frame = cv2.imread(str(image))
    if frame is None:
        raise ValueError(f"failed to read image: {image}")
    current_bbox = [float(value) for value in nudge["current_bbox_xyxy"]]
    shift = [float(value) for value in nudge["recommended_shift_px"]]
    desired_bbox = [
        current_bbox[0] + shift[0],
        current_bbox[1] + shift[1],
        current_bbox[2] + shift[0],
        current_bbox[3] + shift[1],
    ]
    current_center = tuple(int(round(value)) for value in nudge["current_center_px"])
    desired_center = tuple(int(round(value)) for value in nudge["desired_center_px"])

    orange = (0, 140, 255)
    green = (70, 220, 70)
    blue = (255, 120, 0)
    white = (255, 255, 255)
    _rectangle(frame, current_bbox, orange, "current clipped target")
    _rectangle(frame, desired_bbox, green, "desired target margin")
    cv2.circle(frame, current_center, 10, orange, thickness=-1)
    cv2.circle(frame, desired_center, 10, green, thickness=-1)
    cv2.arrowedLine(frame, current_center, desired_center, blue, thickness=5, tipLength=0.25)
    label = str(action.get("diagnostic_summary") or nudge.get("instruction") or "external setup blocker")
    _put_label(frame, label, (24, 42), white, blue)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)
    return {"desired_bbox_xyxy": desired_bbox}


def _rectangle(frame: Any, bbox: list[float], color: tuple[int, int, int], label: str) -> None:
    import cv2

    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=4)
    cv2.putText(
        frame,
        label,
        (max(0, x1), max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def _put_label(
    frame: Any,
    text: str,
    origin: tuple[int, int],
    outline_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    import cv2

    compact = text if len(text) <= 110 else text[:107] + "..."
    cv2.putText(frame, compact, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.85, outline_color, 5, cv2.LINE_AA)
    cv2.putText(frame, compact, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2, cv2.LINE_AA)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human-visible SO-100 reframe overlay from advice JSON.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--advice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-index", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            build_reframe_overlay(
                image=args.image,
                advice=args.advice,
                output=args.output,
                action_index=args.action_index,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
