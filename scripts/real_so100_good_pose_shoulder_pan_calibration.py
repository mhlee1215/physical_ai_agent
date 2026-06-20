#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from scripts.real_so100_apriltag_detection_check import MARKER_DICTIONARIES
from scripts.real_so100_micro_step import _capture_frame, _make_so100_bus
from scripts.real_so100_move_to_home_pose import DEFAULT_HOME_POSE, move_to_home_pose


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION = Path("_workspace/real_so100/calibration/so100_local.json")
DEFAULT_OUTPUT_ROOT = Path("_workspace/real_so100/pose_calibration")
DEFAULT_POSES = [
    Path("_workspace/real_so100/pregrasp_pose_candidates/good_height_20260611_210705/good_height_pose.json"),
    Path(
        "_workspace/real_so100/pregrasp_pose_candidates/"
        "saved_no_contact_body_pose_after_elbow_20260611_212535/no_contact_body_side_pose_after_elbow.json"
    ),
    Path(
        "_workspace/real_so100/pregrasp_pose_candidates/"
        "saved_best_no_contact_body_pose_20260611_212731/best_no_contact_body_side_pose.json"
    ),
    Path(
        "_workspace/real_so100/pregrasp_pose_candidates/"
        "saved_best_no_contact_body_pose_2_20260611_213005/best_no_contact_body_side_pose_2.json"
    ),
    Path(
        "_workspace/real_so100/pregrasp_pose_candidates/"
        "saved_best_no_contact_body_pose_3_20260611_213211/best_no_contact_body_side_pose_3.json"
    ),
    Path(
        "_workspace/real_so100/pregrasp_pose_candidates/"
        "final_best_body_side_no_contact_20260611_213431/final_best_body_side_no_contact_pose.json"
    ),
]


def run_good_pose_shoulder_pan_calibration(
    *,
    port: str,
    calibration: Path,
    output_dir: Path,
    pose_paths: list[Path],
    camera_indexes: list[int],
    dictionaries: list[str],
    shoulder_pan_offsets: list[float],
    shoulder_lift_offset: float,
    pose_interpolation_steps: int,
    execute: bool,
    human_confirmed: bool,
    workspace_clear_confirmed: bool,
    max_step_delta_raw: float,
    settle_seconds: float,
    frame_delay_seconds: float,
    warmup_frames: int,
    return_home: bool,
) -> dict[str, Any]:
    import cv2

    if execute and not (human_confirmed and workspace_clear_confirmed):
        raise ValueError("--execute requires --human-confirmed and --workspace-clear-confirmed.")
    if not pose_paths:
        raise ValueError("At least one pose path is required.")
    if not camera_indexes:
        raise ValueError("At least one camera index is required.")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    overlays_dir = output_dir / "overlays"
    frames_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    calibration_payload = _load_calibration(calibration)
    source_poses = [_load_pose(path) for path in pose_paths]
    poses = _expand_interpolated_poses(source_poses, pose_interpolation_steps)
    report: dict[str, Any] = {
        "timestamp": _wall_time(),
        "operation": "real_so100_good_pose_shoulder_pan_keypoint_calibration",
        "port": port,
        "calibration": str(calibration),
        "output_dir": str(output_dir),
        "pose_paths": [str(path) for path in pose_paths],
        "source_pose_count": len(source_poses),
        "expanded_pose_count": len(poses),
        "pose_interpolation_steps": pose_interpolation_steps,
        "camera_indexes": camera_indexes,
        "dictionaries": dictionaries,
        "shoulder_pan_offsets": shoulder_pan_offsets,
        "shoulder_lift_offset_raw": shoulder_lift_offset,
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "workspace_clear_confirmed": workspace_clear_confirmed,
        "max_step_delta_raw": max_step_delta_raw,
        "settle_seconds": settle_seconds,
        "frame_delay_seconds": frame_delay_seconds,
        "warmup_frames": warmup_frames,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "post_task_torque_disabled": False,
        "samples": [],
        "status": "started",
    }
    _write_json(output_dir / "manifest_started.json", report)

    if not execute:
        report["status"] = "dry_run"
        report["planned_targets"] = _planned_targets(poses, shoulder_pan_offsets, shoulder_lift_offset, calibration_payload)
        _write_json(output_dir / "manifest.json", report)
        return report

    bus, _motors = _make_so100_bus(port)
    try:
        bus.connect(handshake=True)
        current = _read_positions(bus)
        for pose_index, pose in enumerate(poses):
            for offset_index, shoulder_offset in enumerate(shoulder_pan_offsets):
                target = dict(pose["state_raw"])
                target["shoulder_pan"] = _clip(
                    target["shoulder_pan"] + shoulder_offset,
                    calibration_payload["shoulder_pan"]["range_min"],
                    calibration_payload["shoulder_pan"]["range_max"],
                )
                target["shoulder_lift"] = _clip(
                    target["shoulder_lift"] + shoulder_lift_offset,
                    calibration_payload["shoulder_lift"]["range_min"],
                    calibration_payload["shoulder_lift"]["range_max"],
                )
                target = _clip_target(target, calibration_payload)
                step_targets = _interpolate_targets(current, target, max_step_delta_raw, calibration_payload)
                for step_target in step_targets:
                    bus.sync_write("Goal_Position", step_target, normalize=False, num_retry=3)
                    report["send_action_called"] = True
                    report["physical_robot_motion"] = True
                    time.sleep(settle_seconds)
                time.sleep(frame_delay_seconds)
                readback = _read_positions(bus)
                sample_dir = frames_dir / f"pose_{pose_index:02d}_offset_{offset_index:02d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                camera_records = []
                for camera_index in camera_indexes:
                    frame = _capture_stable_frame(camera_index=camera_index, warmup_frames=warmup_frames)
                    raw_path = sample_dir / f"camera_{camera_index}.jpg"
                    cv2.imwrite(str(raw_path), frame)
                    detections, overlay = _detect_markers(frame=frame, cv2=cv2, dictionaries=dictionaries)
                    overlay_path = overlays_dir / f"pose_{pose_index:02d}_offset_{offset_index:02d}_camera_{camera_index}.jpg"
                    cv2.imwrite(str(overlay_path), overlay)
                    camera_records.append(
                        {
                            "camera_index": camera_index,
                            "raw_image": str(raw_path),
                            "overlay_image": str(overlay_path),
                            "shape": list(frame.shape),
                            "detections": detections,
                            "detection_count": sum(item["count"] for item in detections),
                        }
                    )
                sample = {
                    "sample_index": len(report["samples"]),
                    "pose_index": pose_index,
                    "pose_label": pose["label"],
                    "pose_path": str(pose["path"]),
                    "offset_index": offset_index,
                    "shoulder_pan_offset_raw": shoulder_offset,
                    "commanded_target_raw": target,
                    "readback_raw": readback,
                    "camera_records": camera_records,
                }
                report["samples"].append(sample)
                current = readback

        model = _build_piecewise_marker_model(report)
        model_path = output_dir / "shoulder_pan_keypoint_interpolation_model.json"
        _write_json(model_path, model)
        html_path = output_dir / "shoulder_pan_keypoint_calibration_report.html"
        html_path.write_text(_render_html(report, model), encoding="utf-8")
        report["model_path"] = str(model_path)
        report["html_path"] = str(html_path)
        report["status"] = "passed"
    except BaseException as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["error"] = repr(exc)
    finally:
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        except Exception as exc:  # noqa: BLE001
            report["disconnect_error"] = repr(exc)

    if return_home and execute:
        home_report = move_to_home_pose(
            port=port,
            calibration=calibration,
            home_pose=DEFAULT_HOME_POSE,
            output=output_dir / "home_return_after_pose_calibration.json",
            execute=True,
            human_confirmed=human_confirmed,
            workspace_clear_confirmed=workspace_clear_confirmed,
            max_abs_delta_raw=max_step_delta_raw,
            step_settle_seconds=max(0.08, min(0.16, settle_seconds)),
            camera_index=camera_indexes[-1],
            visual_output_dir=output_dir / "home_return_visual",
            record_video=True,
            video_fps=8.0,
        )
        report["home_return_report"] = str(output_dir / "home_return_after_pose_calibration.json")
        report["home_return_status"] = home_report.get("status")
        report["post_task_torque_disabled"] = bool(home_report.get("post_task_torque_disabled"))

    _write_json(output_dir / "manifest.json", report)
    return report


def _load_pose(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload.get("state_raw")
    if not isinstance(state, dict):
        raise ValueError(f"pose missing state_raw: {path}")
    missing = [joint for joint in SO100_JOINT_ORDER if joint not in state]
    if missing:
        raise ValueError(f"pose {path} missing joints: {missing}")
    return {
        "path": path,
        "label": payload.get("pose_label") or payload.get("operation") or path.parent.name,
        "state_raw": {joint: float(state[joint]) for joint in SO100_JOINT_ORDER},
    }


def _expand_interpolated_poses(poses: list[dict[str, Any]], steps_per_segment: int) -> list[dict[str, Any]]:
    if steps_per_segment < 1:
        raise ValueError("pose_interpolation_steps must be >= 1")
    if len(poses) < 2 or steps_per_segment == 1:
        return poses
    expanded: list[dict[str, Any]] = []
    for left_index, (left, right) in enumerate(zip(poses, poses[1:])):
        if left_index == 0:
            expanded.append(left)
        for step in range(1, steps_per_segment + 1):
            alpha = step / steps_per_segment
            state = {
                joint: (left["state_raw"][joint] * (1.0 - alpha)) + (right["state_raw"][joint] * alpha)
                for joint in SO100_JOINT_ORDER
            }
            expanded.append(
                {
                    "path": left["path"],
                    "label": (
                        f"interp_{left_index:02d}_{step:02d}_"
                        f"{left['label']}__to__{right['label']}"
                    ),
                    "state_raw": state,
                    "source_left_path": str(left["path"]),
                    "source_right_path": str(right["path"]),
                    "interpolation_alpha": round(alpha, 6),
                }
            )
    return expanded


def _load_calibration(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        joint: {
            "range_min": float(payload[joint]["range_min"]),
            "range_max": float(payload[joint]["range_max"]),
        }
        for joint in SO100_JOINT_ORDER
    }


def _planned_targets(
    poses: list[dict[str, Any]],
    shoulder_offsets: list[float],
    shoulder_lift_offset: float,
    calibration: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    result = []
    for pose_index, pose in enumerate(poses):
        for offset_index, offset in enumerate(shoulder_offsets):
            target = dict(pose["state_raw"])
            target["shoulder_pan"] = _clip(
                target["shoulder_pan"] + offset,
                calibration["shoulder_pan"]["range_min"],
                calibration["shoulder_pan"]["range_max"],
            )
            target["shoulder_lift"] = _clip(
                target["shoulder_lift"] + shoulder_lift_offset,
                calibration["shoulder_lift"]["range_min"],
                calibration["shoulder_lift"]["range_max"],
            )
            result.append(
                {
                    "pose_index": pose_index,
                    "offset_index": offset_index,
                    "pose_label": pose["label"],
                    "pose_path": str(pose["path"]),
                    "target_raw": _clip_target(target, calibration),
                }
            )
    return result


def _read_positions(bus: Any) -> dict[str, float]:
    return {
        joint: float(value)
        for joint, value in bus.sync_read("Present_Position", normalize=False).items()
        if joint in SO100_JOINT_ORDER
    }


def _interpolate_targets(
    current: dict[str, float],
    target: dict[str, float],
    max_step_delta_raw: float,
    calibration: dict[str, dict[str, float]],
) -> list[dict[str, int]]:
    largest_delta = max(abs(target[joint] - current[joint]) for joint in SO100_JOINT_ORDER)
    steps = max(1, int((largest_delta / max_step_delta_raw) + 0.999))
    result = []
    for step_index in range(1, steps + 1):
        alpha = step_index / steps
        step_target = {}
        for joint in SO100_JOINT_ORDER:
            raw = current[joint] + ((target[joint] - current[joint]) * alpha)
            step_target[joint] = int(round(_clip(raw, calibration[joint]["range_min"], calibration[joint]["range_max"])))
        result.append(step_target)
    return result


def _clip_target(target: dict[str, float], calibration: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        joint: _clip(float(target[joint]), calibration[joint]["range_min"], calibration[joint]["range_max"])
        for joint in SO100_JOINT_ORDER
    }


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _capture_stable_frame(*, camera_index: int, warmup_frames: int):
    import cv2

    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"camera {camera_index} did not open")
        frame = None
        ok = False
        for _ in range(max(1, warmup_frames)):
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"camera {camera_index} frame read failed")
        return frame
    finally:
        cap.release()


def _detect_markers(*, frame: Any, cv2: Any, dictionaries: list[str]) -> tuple[list[dict[str, Any]], Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overlay = frame.copy()
    detections = []
    for family_name in dictionaries:
        dictionary_name = MARKER_DICTIONARIES[family_name]
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        item = {
            "family": family_name,
            "count": 0,
            "ids": [],
            "markers": [],
            "rejected_count": len(rejected) if rejected is not None else 0,
        }
        if ids is not None and len(ids):
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
            item["count"] = int(len(ids))
            item["ids"] = [int(marker_id[0]) for marker_id in ids]
            for marker_id, corner in zip(ids, corners):
                points = corner.reshape(-1, 2)
                center = points.mean(axis=0)
                side_lengths = [
                    float(((points[(index + 1) % 4] - points[index]) ** 2).sum() ** 0.5)
                    for index in range(4)
                ]
                item["markers"].append(
                    {
                        "id": int(marker_id[0]),
                        "center_px": [round(float(center[0]), 2), round(float(center[1]), 2)],
                        "corners_px": [[round(float(x), 2), round(float(y), 2)] for x, y in points],
                        "mean_side_px": round(sum(side_lengths) / len(side_lengths), 2),
                        "min_side_px": round(min(side_lengths), 2),
                    }
                )
        detections.append(item)
    return detections, overlay


def _build_piecewise_marker_model(report: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in report["samples"]:
        pan = float(sample["readback_raw"]["shoulder_pan"])
        for camera in sample["camera_records"]:
            for detection in camera["detections"]:
                for marker in detection["markers"]:
                    key = f"pose_{sample['pose_index']:02d}/camera_{camera['camera_index']}/{detection['family']}/id_{marker['id']}"
                    groups.setdefault(key, []).append(
                        {
                            "sample_index": sample["sample_index"],
                            "pose_index": sample["pose_index"],
                            "pose_label": sample["pose_label"],
                            "camera_index": camera["camera_index"],
                            "family": detection["family"],
                            "marker_id": marker["id"],
                            "shoulder_pan_raw": pan,
                            "center_px": marker["center_px"],
                            "min_side_px": marker["min_side_px"],
                            "image": camera["raw_image"],
                            "overlay_image": camera["overlay_image"],
                            "readback_raw": sample["readback_raw"],
                        }
                    )
    model_groups = {}
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda item: item["shoulder_pan_raw"])
        segments = []
        for left, right in zip(rows, rows[1:]):
            dx = right["shoulder_pan_raw"] - left["shoulder_pan_raw"]
            if abs(dx) < 1e-9:
                continue
            segments.append(
                {
                    "pan_range_raw": [left["shoulder_pan_raw"], right["shoulder_pan_raw"]],
                    "left_center_px": left["center_px"],
                    "right_center_px": right["center_px"],
                    "slope_px_per_raw": [
                        round((right["center_px"][0] - left["center_px"][0]) / dx, 6),
                        round((right["center_px"][1] - left["center_px"][1]) / dx, 6),
                    ],
                    "left_sample_index": left["sample_index"],
                    "right_sample_index": right["sample_index"],
                }
            )
        model_groups[key] = {
            "sample_count": len(rows),
            "rows": rows,
            "segments": segments,
            "usable_for_interpolation": len(segments) > 0,
        }
    return {
        "operation": "shoulder_pan_keypoint_piecewise_interpolation_model",
        "source_manifest": str(Path(report["output_dir"]) / "manifest.json"),
        "camera_indexes": report["camera_indexes"],
        "dictionaries": report["dictionaries"],
        "group_count": len(model_groups),
        "groups": model_groups,
    }


def _render_html(report: dict[str, Any], model: dict[str, Any]) -> str:
    rows = []
    for sample in report["samples"]:
        detections = sum(camera["detection_count"] for camera in sample["camera_records"])
        first_image = sample["camera_records"][0]["raw_image"] if sample["camera_records"] else ""
        rows.append(
            "<tr>"
            f"<td>{sample['sample_index']}</td>"
            f"<td>{html.escape(sample['pose_label'])}</td>"
            f"<td>{sample['shoulder_pan_offset_raw']}</td>"
            f"<td>{sample['readback_raw']['shoulder_pan']:.0f}</td>"
            f"<td>{detections}</td>"
            f"<td><img src='{html.escape(first_image)}' width='220'></td>"
            "</tr>"
        )
    group_rows = []
    for key, group in model["groups"].items():
        if not group["usable_for_interpolation"]:
            continue
        group_rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{group['sample_count']}</td>"
            f"<td>{len(group['segments'])}</td>"
            f"<td>{html.escape(json.dumps(group['segments'][:2]))}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>SO-100 Shoulder Pan Keypoint Calibration</title>",
            "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;}"
            "table{border-collapse:collapse;width:100%;}td,th{border:1px solid #ddd;padding:6px;vertical-align:top;}"
            "img{max-width:100%;height:auto;}code{white-space:pre-wrap;}</style>",
            "</head><body>",
            "<h1>SO-100 Shoulder Pan Keypoint Calibration</h1>",
            f"<p>Status: <b>{html.escape(report.get('status', 'unknown'))}</b></p>",
            f"<p>Samples: {len(report['samples'])}, model groups: {model['group_count']}</p>",
            "<h2>Samples</h2><table><tr><th>#</th><th>Pose</th><th>pan offset</th><th>pan readback</th><th>detections</th><th>image</th></tr>",
            *rows,
            "</table>",
            "<h2>Usable Interpolation Groups</h2><table><tr><th>group</th><th>samples</th><th>segments</th><th>first segments</th></tr>",
            *group_rows,
            "</table>",
            "</body></html>",
        ]
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _wall_time() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _parse_offsets(items: list[str]) -> list[float]:
    if not items:
        return [-120.0, -60.0, 0.0, 60.0, 120.0]
    values = []
    for item in items:
        for part in item.split(","):
            part = part.strip()
            if part:
                values.append(float(part))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Build keypoint interpolation data from saved good poses plus shoulder_pan offsets.")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pose", type=Path, action="append", default=[])
    parser.add_argument("--camera-index", type=int, action="append", default=[])
    parser.add_argument("--dictionary", choices=sorted(MARKER_DICTIONARIES), action="append", default=[])
    parser.add_argument("--shoulder-pan-offset", action="append", default=[])
    parser.add_argument(
        "--pose-interpolation-steps",
        type=int,
        default=1,
        help="Number of samples per segment between saved good poses; 1 keeps only the saved poses.",
    )
    parser.add_argument(
        "--shoulder-lift-offset",
        type=float,
        default=0.0,
        help="Raw offset added to saved shoulder_lift targets. Negative values raised the arm in the latest recovery run.",
    )
    parser.add_argument("--max-step-delta-raw", type=float, default=90.0)
    parser.add_argument("--settle-seconds", type=float, default=0.12)
    parser.add_argument("--frame-delay-seconds", type=float, default=0.35)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--workspace-clear-confirmed", action="store_true")
    parser.add_argument("--no-return-home", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / time.strftime("good_pose_shoulder_pan_%Y%m%d_%H%M%S"))
    pose_paths = args.pose or [path for path in DEFAULT_POSES if path.exists()]
    camera_indexes = args.camera_index or [1, 3]
    dictionaries = args.dictionary or ["tag36h11", "aruco4x4_50", "aruco5x5_100"]
    print(
        json.dumps(
            run_good_pose_shoulder_pan_calibration(
                port=args.port,
                calibration=args.calibration,
                output_dir=output_dir,
                pose_paths=pose_paths,
                camera_indexes=camera_indexes,
                dictionaries=dictionaries,
                shoulder_pan_offsets=_parse_offsets(args.shoulder_pan_offset),
                shoulder_lift_offset=args.shoulder_lift_offset,
                pose_interpolation_steps=args.pose_interpolation_steps,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                workspace_clear_confirmed=args.workspace_clear_confirmed,
                max_step_delta_raw=args.max_step_delta_raw,
                settle_seconds=args.settle_seconds,
                frame_delay_seconds=args.frame_delay_seconds,
                warmup_frames=args.warmup_frames,
                return_home=not args.no_return_home,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
