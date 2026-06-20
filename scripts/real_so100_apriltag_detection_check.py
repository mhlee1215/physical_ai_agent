#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


MARKER_DICTIONARIES = {
    "aruco4x4_50": "DICT_4X4_50",
    "aruco4x4_100": "DICT_4X4_100",
    "aruco5x5_100": "DICT_5X5_100",
    "aruco5x5_250": "DICT_5X5_250",
    "aruco6x6_250": "DICT_6X6_250",
    "tag36h11": "DICT_APRILTAG_36h11",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag16h5": "DICT_APRILTAG_16h5",
}


def check_apriltag_detection(
    *,
    camera_indexes: list[int],
    output_dir: Path,
    dictionaries: list[str],
    warmup_frames: int,
) -> dict[str, Any]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_fiducial_detection_check",
        "camera_indexes": camera_indexes,
        "dictionaries_checked": dictionaries,
        "warmup_frames": warmup_frames,
        "cameras": {},
        "ok": True,
    }

    for camera_index in camera_indexes:
        item = _check_camera(
            cv2=cv2,
            camera_index=camera_index,
            output_dir=output_dir,
            dictionaries=dictionaries,
            warmup_frames=warmup_frames,
        )
        if not item.get("opened") or item.get("error"):
            report["ok"] = False
        report["cameras"][str(camera_index)] = item

    report["total_detections"] = sum(
        detection.get("count", 0)
        for camera in report["cameras"].values()
        for detection in camera.get("detections", [])
    )
    report["detection_found"] = report["total_detections"] > 0
    report_path = output_dir / "detection_report.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _check_camera(
    *,
    cv2: Any,
    camera_index: int,
    output_dir: Path,
    dictionaries: list[str],
    warmup_frames: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    item: dict[str, Any] = {"opened": bool(cap.isOpened()), "detections": []}
    try:
        if not cap.isOpened():
            item["error"] = "camera did not open"
            return item
        frame = None
        ok = False
        for _ in range(max(1, warmup_frames)):
            ok, frame = cap.read()
        if not ok or frame is None:
            item["error"] = "frame read failed"
            return item

        raw_path = output_dir / f"camera_{camera_index}_raw.jpg"
        cv2.imwrite(str(raw_path), frame)
        item["raw_image"] = str(raw_path)
        item["shape"] = list(frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        overlay = frame.copy()

        for family_name in dictionaries:
            dictionary_name = MARKER_DICTIONARIES[family_name]
            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
            parameters = cv2.aruco.DetectorParameters()
            parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            corners, ids, rejected = detector.detectMarkers(gray)
            detection = {
                "family": family_name,
                "count": 0,
                "ids": [],
                "markers": [],
                "rejected_count": len(rejected) if rejected is not None else 0,
            }
            if ids is not None and len(ids):
                cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
                detection["count"] = int(len(ids))
                detection["ids"] = [int(marker_id[0]) for marker_id in ids]
                for marker_id, corner in zip(ids, corners):
                    points = corner.reshape(-1, 2)
                    center = points.mean(axis=0)
                    side_lengths = [
                        float(((points[(index + 1) % 4] - points[index]) ** 2).sum() ** 0.5)
                        for index in range(4)
                    ]
                    detection["markers"].append(
                        {
                            "id": int(marker_id[0]),
                            "center_px": [round(float(center[0]), 2), round(float(center[1]), 2)],
                            "corners_px": [
                                [round(float(x), 2), round(float(y), 2)]
                                for x, y in points
                            ],
                            "mean_side_px": round(sum(side_lengths) / len(side_lengths), 2),
                            "min_side_px": round(min(side_lengths), 2),
                        }
                    )
            item["detections"].append(detection)

        overlay_path = output_dir / f"camera_{camera_index}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        item["overlay_image"] = str(overlay_path)
        return item
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture camera frames and check ArUco/AprilTag fiducial detection.")
    parser.add_argument("--camera-index", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dictionary", choices=sorted(MARKER_DICTIONARIES), action="append", default=[])
    parser.add_argument("--family", choices=sorted(MARKER_DICTIONARIES), action="append", default=[], help="Alias for --dictionary.")
    parser.add_argument("--warmup-frames", type=int, default=8)
    args = parser.parse_args()
    dictionaries = args.dictionary or args.family or [
        "aruco4x4_50",
        "aruco4x4_100",
        "tag36h11",
        "tag25h9",
        "tag16h5",
    ]
    print(
        json.dumps(
            check_apriltag_detection(
                camera_indexes=args.camera_index,
                output_dir=args.output_dir,
                dictionaries=dictionaries,
                warmup_frames=args.warmup_frames,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
