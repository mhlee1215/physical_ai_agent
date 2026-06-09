#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_reframe_advice(
    *,
    pregrasp_probe: Path,
    jaw_readiness: Path,
    output: Path | None = None,
    jaw_camera: str = "0",
    object_view_camera: str = "1",
) -> dict[str, Any]:
    pregrasp = _load_json(pregrasp_probe)
    jaw = _load_json(jaw_readiness)
    jaw_object = jaw.get("object_candidate")
    object_view = _assessment_for_camera(pregrasp, object_view_camera)
    jaw_assessment = _assessment_for_camera(pregrasp, jaw_camera)

    jaw_image_shape = jaw.get("image_shape") or (jaw_assessment or {}).get("image_shape")
    clipped_sides = _clipped_sides(
        bbox=(jaw_object or {}).get("bbox_xyxy"),
        image_shape=jaw_image_shape,
        edge_margin_px=int(jaw.get("edge_margin_px") or pregrasp.get("edge_margin_px") or 8),
    )
    target_margin_px = int(jaw.get("target_margin_px") or pregrasp.get("target_margin_px") or 32)
    nudge = _nudge_target(
        bbox=(jaw_object or {}).get("bbox_xyxy"),
        image_shape=jaw_image_shape,
        target_margin_px=target_margin_px,
    )
    actions = []
    if jaw.get("status") != "ready":
        actions.append(
            {
                "type": "repair_jaw_camera_framing",
                "camera": jaw_camera,
                "priority": 1,
                "reason": "; ".join(str(item) for item in jaw.get("blockers", [])) or "jaw camera not ready",
                "clipped_sides": clipped_sides,
                "agent_actionable": False,
                "external_setup_required": True,
                "diagnostic_summary": _diagnostic_summary(
                    clipped_sides=clipped_sides,
                    camera=jaw_camera,
                    nudge=nudge,
                ),
                "image_space_goal": _image_space_goal(nudge),
                "target_margin_px": target_margin_px,
                "image_space_nudge": nudge,
            }
        )
    if not object_view or object_view.get("usable_for_pregrasp") is not True:
        actions.append(
            {
                "type": "repair_object_view_camera",
                "camera": object_view_camera,
                "priority": 2,
                "reason": "object-view camera is not usable for pregrasp",
                "agent_actionable": False,
                "external_setup_required": True,
                "diagnostic_summary": f"camera {object_view_camera} does not provide a usable policy view; this is an external setup blocker, not an agent action",
            }
        )
    result = {
        "status": "ready" if actions else "passed",
        "operation": "real_so100_reframe_advisor",
        "pregrasp_probe": str(pregrasp_probe),
        "jaw_readiness": str(jaw_readiness),
        "jaw_camera": jaw_camera,
        "object_view_camera": object_view_camera,
        "jaw_status": jaw.get("status"),
        "object_view_usable": bool(object_view and object_view.get("usable_for_pregrasp")),
        "jaw_object_candidate": jaw_object,
        "jaw_object_clipped_sides": clipped_sides,
        "target_margin_px": target_margin_px,
        "jaw_object_image_space_nudge": nudge,
        "actions": actions,
        "notes": [
            "Observation repair advice only; it does not execute robot motion.",
            "Advice is image-space and camera-role based so it can be reused for other target objects and directions.",
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result["manifest_path"] = str(output)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _assessment_for_camera(pregrasp: dict[str, Any], camera: str) -> dict[str, Any] | None:
    for item in pregrasp.get("assessments", []):
        if str(item.get("camera")) == str(camera):
            return item
    return None


def _clipped_sides(*, bbox: Any, image_shape: Any, edge_margin_px: int) -> list[str]:
    if not bbox or not image_shape:
        return []
    x1, y1, x2, y2 = [float(value) for value in bbox]
    height = float(image_shape[0])
    width = float(image_shape[1])
    sides = []
    if x1 <= edge_margin_px:
        sides.append("left")
    if y1 <= edge_margin_px:
        sides.append("top")
    if x2 >= width - edge_margin_px:
        sides.append("right")
    if y2 >= height - edge_margin_px:
        sides.append("bottom")
    return sides


def _nudge_target(*, bbox: Any, image_shape: Any, target_margin_px: int) -> dict[str, Any] | None:
    if not bbox or not image_shape:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    height = float(image_shape[0])
    width = float(image_shape[1])
    dx = 0.0
    dy = 0.0
    if x1 < target_margin_px:
        dx = max(dx, target_margin_px - x1)
    if x2 > width - target_margin_px:
        dx = min(dx, width - target_margin_px - x2)
    if y1 < target_margin_px:
        dy = max(dy, target_margin_px - y1)
    if y2 > height - target_margin_px:
        dy = min(dy, height - target_margin_px - y2)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return {
        "target_margin_px": target_margin_px,
        "current_bbox_xyxy": [x1, y1, x2, y2],
        "current_center_px": [center_x, center_y],
        "recommended_shift_px": [dx, dy],
        "desired_center_px": [center_x + dx, center_y + dy],
        "image_shape": image_shape,
        "instruction": _nudge_instruction(dx=dx, dy=dy),
    }


def _nudge_instruction(*, dx: float, dy: float) -> str:
    parts = []
    if abs(dx) >= 1.0:
        side = "left" if dx > 0 else "right"
        parts.append(f"detected target bbox is about {abs(dx):.0f}px too far {side} for the policy-input margin")
    if abs(dy) >= 1.0:
        side = "high" if dy > 0 else "low"
        parts.append(f"detected target bbox is about {abs(dy):.0f}px too {side} for the policy-input margin")
    if not parts:
        return "target has the requested image-space margin"
    return "; ".join(parts)


def _image_space_goal(nudge: dict[str, Any] | None) -> str | None:
    if not nudge:
        return None
    current_bbox = nudge.get("current_bbox_xyxy")
    shift = nudge.get("recommended_shift_px")
    target_margin = nudge.get("target_margin_px")
    if not current_bbox or not shift:
        return None
    return (
        f"image-space diagnostic: current bbox {current_bbox} would need approximately {shift} px more "
        f"boundary margin to satisfy the {target_margin}px policy-input gate"
    )


def _diagnostic_summary(*, clipped_sides: list[str], camera: str, nudge: dict[str, Any] | None = None) -> str:
    if not clipped_sides:
        return f"camera {camera} does not show both target object and jaw marker clearly; external setup must change before the autonomous loop can continue"
    edge_text = ", ".join(clipped_sides)
    nudge_text = f" Image-space diagnostic: {nudge.get('instruction')}." if nudge else ""
    return (
        f"camera {camera} target detection is clipped on the {edge_text} edge; this is an external setup blocker, "
        f"not an autonomous agent action.{nudge_text}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build observation-repair advice for real SO-100 camera gates.")
    parser.add_argument("--pregrasp-probe", type=Path, required=True)
    parser.add_argument("--jaw-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jaw-camera", default="0")
    parser.add_argument("--object-view-camera", default="1")
    args = parser.parse_args()
    print(
        json.dumps(
            build_reframe_advice(
                pregrasp_probe=args.pregrasp_probe,
                jaw_readiness=args.jaw_readiness,
                output=args.output,
                jaw_camera=args.jaw_camera,
                object_view_camera=args.object_view_camera,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
