from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GreenObjectDetection:
    camera: str
    image_path: str
    visible: bool
    bbox_xyxy: list[int] | None
    center_px: list[float] | None
    area_px: int
    image_shape: list[int]
    confidence: float


@dataclass(frozen=True)
class GreenObjectVerifierResult:
    status: str
    detections: list[GreenObjectDetection]
    visible_cameras: list[str]
    primary_camera: str | None
    object_visible: bool
    notes: list[str]


def verify_green_object_images(
    image_paths: dict[str, Path],
    *,
    min_area_px: int = 800,
) -> GreenObjectVerifierResult:
    detections = [
        detect_green_object(camera=camera, image_path=image_path, min_area_px=min_area_px)
        for camera, image_path in sorted(image_paths.items())
    ]
    visible = [detection.camera for detection in detections if detection.visible]
    primary = max(
        (detection for detection in detections if detection.visible),
        key=lambda detection: detection.area_px,
        default=None,
    )
    notes = [
        "Color-segmentation verifier only; use as a progress/retry signal, not final task success.",
        "Green threshold is tuned for the current Android figure and table lighting.",
    ]
    return GreenObjectVerifierResult(
        status="passed" if visible else "blocked",
        detections=detections,
        visible_cameras=visible,
        primary_camera=primary.camera if primary else None,
        object_visible=bool(visible),
        notes=notes,
    )


def detect_green_object(
    *,
    camera: str,
    image_path: Path,
    min_area_px: int = 800,
) -> GreenObjectDetection:
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        return GreenObjectDetection(
            camera=camera,
            image_path=str(image_path),
            visible=False,
            bbox_xyxy=None,
            center_px=None,
            area_px=0,
            image_shape=[],
            confidence=0.0,
        )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Broad green range for the Android figure under warm desk lighting.
    lower = np.array([35, 45, 35], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _empty_detection(camera, image_path, image.shape)

    contour = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(contour))
    if area < min_area_px:
        return _empty_detection(camera, image_path, image.shape, area_px=area)

    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"]:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx = float(x + w / 2)
        cy = float(y + h / 2)

    image_area = float(image.shape[0] * image.shape[1])
    confidence = min(1.0, area / max(float(min_area_px) * 20.0, 1.0))
    confidence = max(confidence, min(1.0, area / image_area * 40.0))
    return GreenObjectDetection(
        camera=camera,
        image_path=str(image_path),
        visible=True,
        bbox_xyxy=[int(x), int(y), int(x + w), int(y + h)],
        center_px=[round(cx, 2), round(cy, 2)],
        area_px=area,
        image_shape=list(image.shape),
        confidence=round(confidence, 4),
    )


def write_verifier_result(result: GreenObjectVerifierResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")


def _empty_detection(
    camera: str,
    image_path: Path,
    image_shape: tuple[int, ...],
    area_px: int = 0,
) -> GreenObjectDetection:
    return GreenObjectDetection(
        camera=camera,
        image_path=str(image_path),
        visible=False,
        bbox_xyxy=None,
        center_px=None,
        area_px=area_px,
        image_shape=list(image_shape),
        confidence=0.0,
    )


def image_paths_from_episode_record(record: dict[str, Any]) -> dict[str, Path]:
    images = record.get("observation", {}).get("images", {})
    return {str(camera): Path(path) for camera, path in images.items() if path}
