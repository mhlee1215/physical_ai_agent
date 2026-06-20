#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

from scripts.real_so100_apriltag_detection_check import MARKER_DICTIONARIES
from scripts.real_so100_pose_calibration_recorder import JOINT_ORDER


DEFAULT_DICTIONARY = "tag36h11"


def build_body_map_model(
    *,
    session: Path,
    dictionary_name: str,
    primary_marker_ids: list[int],
    min_marker_side_px: float,
    min_dt_seconds: float,
    max_dt_seconds: float,
    min_joint_delta_raw: float,
    min_marker_delta_px: float,
    max_transition_gap: int,
) -> dict[str, Any]:
    samples = _read_jsonl(session / "aligned_samples.jsonl")
    pose_dir = session / "body_map"
    pose_dir.mkdir(parents=True, exist_ok=True)
    pose_path = pose_dir / "pose_samples.jsonl"
    transition_path = pose_dir / "transition_samples.jsonl"
    model_path = pose_dir / "body_map_knn_model.json"
    report_path = pose_dir / "body_map_report.html"
    pose_path.write_text("", encoding="utf-8")
    transition_path.write_text("", encoding="utf-8")

    pose_samples = [
        pose
        for pose in (
            _build_pose_sample(
                sample=sample,
                dictionary_name=dictionary_name,
                primary_marker_ids=primary_marker_ids,
                min_marker_side_px=min_marker_side_px,
            )
            for sample in samples
        )
        if pose is not None
    ]
    for pose in pose_samples:
        _append_jsonl(pose_path, pose)

    transitions = _build_transitions(
        poses=pose_samples,
        min_dt_seconds=min_dt_seconds,
        max_dt_seconds=max_dt_seconds,
        min_joint_delta_raw=min_joint_delta_raw,
        min_marker_delta_px=min_marker_delta_px,
        max_transition_gap=max_transition_gap,
    )
    for transition in transitions:
        _append_jsonl(transition_path, transition)

    model = _build_knn_model(
        transitions=transitions,
        dictionary_name=dictionary_name,
        primary_marker_ids=primary_marker_ids,
        min_marker_side_px=min_marker_side_px,
    )
    model["artifacts"] = {
        "pose_samples": str(pose_path),
        "transition_samples": str(transition_path),
        "model": str(model_path),
        "html": str(report_path),
    }
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
    _write_html_report(
        path=report_path,
        session=session,
        model=model,
        pose_samples=pose_samples,
        transitions=transitions,
    )
    return model


def _build_pose_sample(
    *,
    sample: dict[str, Any],
    dictionary_name: str,
    primary_marker_ids: list[int],
    min_marker_side_px: float,
) -> dict[str, Any] | None:
    detections_by_camera: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    for camera, image_path in sample.get("cameras", {}).items():
        detections = _detect_markers(Path(image_path), dictionary_name=dictionary_name)
        usable = [
            marker
            for marker in detections
            if marker["min_side_px"] >= min_marker_side_px
            and (not primary_marker_ids or marker["id"] in primary_marker_ids)
        ]
        detections_by_camera[str(camera)] = {
            "image": image_path,
            "markers": detections,
            "usable_markers": usable,
        }
        if usable:
            selected[str(camera)] = max(usable, key=lambda marker: marker["min_side_px"])
    if not selected:
        return None
    return {
        "sample_index": sample["sample_index"],
        "source_frame_index": sample.get("source_frame_index"),
        "t": sample["t"],
        "sync_error_ms": sample.get("sync_error_ms"),
        "joint_raw": sample.get("motor_state_nearest", {}),
        "selected_markers": selected,
        "detections_by_camera": detections_by_camera,
    }


def _detect_markers(image_path: Path, *, dictionary_name: str) -> list[dict[str, Any]]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICTIONARIES[dictionary_name]))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or not len(ids):
        return []
    markers = []
    for marker_id, corner in zip(ids, corners):
        points = corner.reshape(-1, 2)
        center = points.mean(axis=0)
        side_lengths = [
            float(((points[(index + 1) % 4] - points[index]) ** 2).sum() ** 0.5)
            for index in range(4)
        ]
        markers.append(
            {
                "id": int(marker_id[0]),
                "center_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
                "corners_px": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
                "mean_side_px": round(sum(side_lengths) / len(side_lengths), 3),
                "min_side_px": round(min(side_lengths), 3),
            }
        )
    return markers


def _build_transitions(
    *,
    poses: list[dict[str, Any]],
    min_dt_seconds: float,
    max_dt_seconds: float,
    min_joint_delta_raw: float,
    min_marker_delta_px: float,
    max_transition_gap: int,
) -> list[dict[str, Any]]:
    transitions = []
    for start_index, start in enumerate(poses):
        for end_index in range(start_index + 1, min(len(poses), start_index + max_transition_gap + 1)):
            end = poses[end_index]
            dt = float(end["t"]) - float(start["t"])
            if dt < min_dt_seconds:
                continue
            if dt > max_dt_seconds:
                break
            shared = _shared_marker_deltas(start, end)
            if not shared:
                continue
            joint_delta = _joint_delta(start.get("joint_raw", {}), end.get("joint_raw", {}))
            joint_norm = math.sqrt(sum(value * value for value in joint_delta.values()))
            marker_norm = math.sqrt(sum(delta * delta for item in shared.values() for delta in item["delta_px"]))
            if joint_norm < min_joint_delta_raw or marker_norm < min_marker_delta_px:
                continue
            transition = {
                "transition_index": len(transitions),
                "pose0_index": start["sample_index"],
                "pose1_index": end["sample_index"],
                "dt_seconds": round(dt, 4),
                "joint0_raw": start.get("joint_raw", {}),
                "joint1_raw": end.get("joint_raw", {}),
                "joint_delta_raw": joint_delta,
                "marker_deltas": shared,
                "joint_delta_norm_raw": round(joint_norm, 4),
                "marker_delta_norm_px": round(marker_norm, 4),
                "usable": True,
            }
            transitions.append(transition)
            break
    return transitions


def _shared_marker_deltas(start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for camera, marker0 in start.get("selected_markers", {}).items():
        marker1 = end.get("selected_markers", {}).get(camera)
        if marker1 is None:
            continue
        if marker0["id"] != marker1["id"]:
            continue
        center0 = marker0["center_px"]
        center1 = marker1["center_px"]
        key = f"camera_{camera}_id_{marker0['id']}"
        result[key] = {
            "camera": int(camera),
            "marker_id": int(marker0["id"]),
            "center0_px": center0,
            "center1_px": center1,
            "delta_px": [round(center1[0] - center0[0], 4), round(center1[1] - center0[1], 4)],
            "min_side0_px": marker0["min_side_px"],
            "min_side1_px": marker1["min_side_px"],
        }
    return result


def _joint_delta(start: dict[str, Any], end: dict[str, Any]) -> dict[str, float]:
    result = {}
    for joint in JOINT_ORDER:
        if joint in start and joint in end:
            result[joint] = round(float(end[joint]) - float(start[joint]), 4)
    return result


def _build_knn_model(
    *,
    transitions: list[dict[str, Any]],
    dictionary_name: str,
    primary_marker_ids: list[int],
    min_marker_side_px: float,
) -> dict[str, Any]:
    feature_rows = []
    for transition in transitions:
        feature_rows.append(
            {
                "transition_index": transition["transition_index"],
                "joint0_raw": transition["joint0_raw"],
                "joint_delta_raw": transition["joint_delta_raw"],
                "marker_deltas": transition["marker_deltas"],
            }
        )
    return {
        "operation": "real_so100_body_map_knn_model",
        "model_type": "transition_knn_replay",
        "dictionary": dictionary_name,
        "primary_marker_ids": primary_marker_ids,
        "min_marker_side_px": min_marker_side_px,
        "transition_count": len(transitions),
        "feature_schema": {
            "input": [
                "current_joint_raw",
                "desired_marker_delta_px_by_camera_and_marker_id",
            ],
            "output": "joint_delta_raw",
        },
        "inference_note": (
            "Use nearest transitions by current joint state and desired marker delta, "
            "then weighted-average joint_delta_raw and clamp inside calibration."
        ),
        "rows": feature_rows,
    }


def _write_html_report(
    *,
    path: Path,
    session: Path,
    model: dict[str, Any],
    pose_samples: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
) -> None:
    marker_counts: dict[str, int] = {}
    for pose in pose_samples:
        for camera, marker in pose.get("selected_markers", {}).items():
            key = f"camera {camera} id {marker['id']}"
            marker_counts[key] = marker_counts.get(key, 0) + 1
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{count}</td></tr>"
        for key, count in sorted(marker_counts.items())
    )
    examples = "".join(
        "<tr>"
        f"<td>{item['transition_index']}</td>"
        f"<td>{item['pose0_index']} -> {item['pose1_index']}</td>"
        f"<td>{html.escape(json.dumps(item['joint_delta_raw'], sort_keys=True))}</td>"
        f"<td>{html.escape(json.dumps(item['marker_deltas'], sort_keys=True))}</td>"
        "</tr>"
        for item in transitions[:30]
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SO-100 Body Map Model</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    .metric b {{ display: block; font-size: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; font-size: 12px; }}
    th {{ background: #f4f4f4; text-align: left; }}
    code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>SO-100 Body Map Model</h1>
  <div class="metrics">
    <div class="metric"><b>{len(pose_samples)}</b><span>pose samples with marker detections</span></div>
    <div class="metric"><b>{len(transitions)}</b><span>usable transitions</span></div>
    <div class="metric"><b>{html.escape(model['dictionary'])}</b><span>fiducial dictionary</span></div>
  </div>
  <p>Session: <code>{html.escape(str(session))}</code></p>
  <h2>Marker Visibility</h2>
  <table><tr><th>camera / marker</th><th>pose sample count</th></tr>{rows}</table>
  <h2>Transition Examples</h2>
  <table><tr><th>index</th><th>pose pair</th><th>joint delta raw</th><th>marker delta px</th></tr>{examples}</table>
</body>
</html>
""",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an empirical SO-100 body-map approximate IK dataset/model.")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--dictionary", choices=sorted(MARKER_DICTIONARIES), default=DEFAULT_DICTIONARY)
    parser.add_argument("--primary-marker-id", type=int, action="append", default=[])
    parser.add_argument("--min-marker-side-px", type=float, default=20.0)
    parser.add_argument("--min-dt-seconds", type=float, default=0.45)
    parser.add_argument("--max-dt-seconds", type=float, default=2.8)
    parser.add_argument("--min-joint-delta-raw", type=float, default=3.0)
    parser.add_argument("--min-marker-delta-px", type=float, default=1.5)
    parser.add_argument("--max-transition-gap", type=int, default=12)
    args = parser.parse_args()
    print(
        json.dumps(
            build_body_map_model(
                session=args.session,
                dictionary_name=args.dictionary,
                primary_marker_ids=args.primary_marker_id,
                min_marker_side_px=args.min_marker_side_px,
                min_dt_seconds=args.min_dt_seconds,
                max_dt_seconds=args.max_dt_seconds,
                min_joint_delta_raw=args.min_joint_delta_raw,
                min_marker_delta_px=args.min_marker_delta_px,
                max_transition_gap=args.max_transition_gap,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
