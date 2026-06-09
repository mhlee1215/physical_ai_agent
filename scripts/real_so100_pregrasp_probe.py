#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from physical_ai_agent.perception.green_object_verifier import (
    detect_green_object,
    image_paths_from_episode_record,
)


@dataclass(frozen=True)
class PregraspCameraAssessment:
    camera: str
    image_path: str
    object_visible: bool
    bbox_xyxy: list[int] | None
    center_px: list[float] | None
    area_px: int
    image_shape: list[int]
    edge_clipped: bool
    usable_for_pregrasp: bool
    notes: list[str]


def assess_episode_frame(
    *,
    episode: Path,
    frame_index: int,
    output: Path,
    min_area_px: int,
    edge_margin_px: int,
) -> dict[str, Any]:
    record = _load_episode_record(episode, frame_index)
    assessments = [
        assess_camera(
            camera=camera,
            image_path=image_path,
            min_area_px=min_area_px,
            edge_margin_px=edge_margin_px,
        )
        for camera, image_path in sorted(image_paths_from_episode_record(record).items())
    ]
    usable = [item.camera for item in assessments if item.usable_for_pregrasp]
    primary = max(
        (item for item in assessments if item.usable_for_pregrasp),
        key=lambda item: item.area_px,
        default=None,
    )
    result = {
        "status": "passed" if primary else "blocked",
        "episode": str(episode),
        "frame_index": frame_index,
        "task": record.get("task"),
        "state": record.get("observation", {}).get("state", {}),
        "min_area_px": min_area_px,
        "edge_margin_px": edge_margin_px,
        "assessments": [asdict(item) for item in assessments],
        "usable_cameras": usable,
        "primary_camera": primary.camera if primary else None,
        "notes": [
            "Pre-grasp probe only; it does not prove a grasp and does not send robot actions.",
            "Edge-clipped green detections are rejected because they are ambiguous for approach control.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def assess_camera(
    *,
    camera: str,
    image_path: Path,
    min_area_px: int,
    edge_margin_px: int,
) -> PregraspCameraAssessment:
    detection = detect_green_object(camera=camera, image_path=image_path, min_area_px=min_area_px)
    notes: list[str] = []
    edge_clipped = False
    if detection.bbox_xyxy and detection.image_shape:
        edge_clipped = _bbox_touches_edge(
            bbox=detection.bbox_xyxy,
            image_shape=detection.image_shape,
            margin_px=edge_margin_px,
        )
        if edge_clipped:
            notes.append("green detection touches image boundary")
    if not detection.visible:
        notes.append("green object not visible above area threshold")
    usable = bool(detection.visible and not edge_clipped)
    return PregraspCameraAssessment(
        camera=camera,
        image_path=str(image_path),
        object_visible=detection.visible,
        bbox_xyxy=detection.bbox_xyxy,
        center_px=detection.center_px,
        area_px=detection.area_px,
        image_shape=detection.image_shape,
        edge_clipped=edge_clipped,
        usable_for_pregrasp=usable,
        notes=notes,
    )


def _bbox_touches_edge(*, bbox: list[int], image_shape: list[int], margin_px: int) -> bool:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    return x1 <= margin_px or y1 <= margin_px or x2 >= width - margin_px or y2 >= height - margin_px


def _load_episode_record(path: Path, frame_index: int) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["frame_index"]) == frame_index:
            return record
    raise ValueError(f"frame_index={frame_index} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess real SO-100 camera frames for pre-grasp usability.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area-px", type=int, default=800)
    parser.add_argument("--edge-margin-px", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            assess_episode_frame(
                episode=args.episode,
                frame_index=args.frame_index,
                output=args.output,
                min_area_px=args.min_area_px,
                edge_margin_px=args.edge_margin_px,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
