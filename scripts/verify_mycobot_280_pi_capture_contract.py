#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_mycobot_280_pi_adaptive_lerobot_dataset import (  # noqa: E402
    CAMERA_KEYS,
    JOINT_NAMES,
    extract_action_vector,
    extract_contact_count,
    extract_joint_vector,
    extract_object_position,
    load_camera_manifest,
    load_jsonl,
)

SUPPORTED_IMAGE_SUFFIXES = {".ppm", ".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class CaptureFrameCheck:
    frame_index: int
    status: str
    errors: list[str]
    timestamp: float | None
    object_position: list[float] | None
    contact_count: int | None
    camera_paths: dict[str, str]


@dataclass(frozen=True)
class CaptureContractReport:
    status: str
    trace_path: str
    camera_manifest: str
    frame_count: int
    passed_frame_count: int
    failed_frame_count: int
    joint_names: list[str]
    required_cameras: list[str]
    checks: list[CaptureFrameCheck]
    artifacts: dict[str, str]
    claim_boundary: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a myCobot 280 Pi adaptive-gripper ROS/Gazebo capture has the "
            "minimum trace, object-oracle, and real-camera fields needed before dataset export."
        )
    )
    parser.add_argument("--input-trace", type=Path, required=True)
    parser.add_argument("--camera-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/mycobot_280pi_capture_contract_verify"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify_mycobot_280_pi_capture_contract(
        input_trace=args.input_trace,
        camera_manifest=args.camera_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    raise SystemExit(0 if report.status == "passed" else 1)


def verify_mycobot_280_pi_capture_contract(
    *,
    input_trace: Path,
    camera_manifest: Path,
    output_dir: Path,
) -> CaptureContractReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_records = load_jsonl(input_trace)
    camera_records = load_camera_manifest(camera_manifest)
    checks: list[CaptureFrameCheck] = []

    if len(trace_records) != len(camera_records):
        frame_count = max(len(trace_records), len(camera_records))
        for frame_index in range(frame_count):
            errors = []
            if frame_index >= len(trace_records):
                errors.append("missing trace record")
            if frame_index >= len(camera_records):
                errors.append("missing camera manifest record")
            checks.append(
                CaptureFrameCheck(
                    frame_index=frame_index,
                    status="failed",
                    errors=errors,
                    timestamp=None,
                    object_position=None,
                    contact_count=None,
                    camera_paths={},
                )
            )
    else:
        previous_timestamp: float | None = None
        for frame_index, (trace, camera) in enumerate(zip(trace_records, camera_records, strict=True)):
            check = _verify_frame(
                frame_index=frame_index,
                trace=trace,
                camera=camera,
                previous_timestamp=previous_timestamp,
            )
            checks.append(check)
            if check.timestamp is not None:
                previous_timestamp = check.timestamp

    failed = [check for check in checks if check.status != "passed"]
    report = CaptureContractReport(
        status="passed" if not failed else "failed",
        trace_path=str(input_trace),
        camera_manifest=str(camera_manifest),
        frame_count=len(checks),
        passed_frame_count=len(checks) - len(failed),
        failed_frame_count=len(failed),
        joint_names=JOINT_NAMES,
        required_cameras=CAMERA_KEYS,
        checks=checks,
        artifacts={},
        claim_boundary=(
            "This verifies capture contract completeness only. It does not prove robot calibration, "
            "camera calibration, physics fidelity, or SmolVLA train/eval performance."
        ),
    )
    return replace(report, artifacts=_write_artifacts(report, output_dir))


def _verify_frame(
    *,
    frame_index: int,
    trace: dict[str, Any],
    camera: dict[str, Any],
    previous_timestamp: float | None,
) -> CaptureFrameCheck:
    errors: list[str] = []
    timestamp = _timestamp(trace, camera)
    if timestamp is None:
        errors.append("missing numeric timestamp")
    elif previous_timestamp is not None and timestamp < previous_timestamp:
        errors.append("timestamps must be monotonic")

    object_position: list[float] | None = None
    contact_count: int | None = None
    try:
        state = extract_joint_vector(trace)
        if len(state) != len(JOINT_NAMES):
            errors.append("joint vector has wrong length")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid joint_state: {exc}")
    try:
        action = extract_action_vector(trace, fallback=[0.0] * len(JOINT_NAMES))
        if len(action) != len(JOINT_NAMES):
            errors.append("action vector has wrong length")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid action: {exc}")
    try:
        object_position = extract_object_position(trace)
        if object_position is None:
            errors.append("missing object_pose/object_state position")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid object pose: {exc}")
    try:
        contact_count = extract_contact_count(trace)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid contact evidence: {exc}")
    if contact_count is None or contact_count < 0:
        errors.append("missing or negative contact evidence")

    camera_paths: dict[str, str] = {}
    for camera_key in CAMERA_KEYS:
        camera_path = _camera_path(camera, camera_key)
        if camera_path is None:
            errors.append(f"missing {camera_key} camera image path")
            continue
        camera_paths[camera_key] = str(camera_path)
        if not camera_path.exists() or not camera_path.is_file():
            errors.append(f"missing {camera_key} camera file: {camera_path}")
        if camera_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            errors.append(f"unsupported {camera_key} camera extension: {camera_path.suffix}")

    return CaptureFrameCheck(
        frame_index=frame_index,
        status="passed" if not errors else "failed",
        errors=errors,
        timestamp=timestamp,
        object_position=object_position,
        contact_count=contact_count,
        camera_paths=camera_paths,
    )


def _timestamp(trace: dict[str, Any], camera: dict[str, Any]) -> float | None:
    raw = trace.get("timestamp", camera.get("timestamp"))
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _camera_path(camera: dict[str, Any], camera_key: str) -> Path | None:
    raw_path = camera.get(camera_key) or camera.get(f"{camera_key}_image")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    manifest_root = Path(str(camera.get("manifest_root", "."))).expanduser()
    return manifest_root / path


def _write_artifacts(report: CaptureContractReport, output_dir: Path) -> dict[str, str]:
    json_path = output_dir / "mycobot_280_pi_capture_contract_report.json"
    md_path = output_dir / "mycobot_280_pi_capture_contract_report.md"
    json_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# myCobot 280 Pi Capture Contract Verify",
        "",
        f"Status: `{report.status}`",
        f"Trace: `{report.trace_path}`",
        f"Camera manifest: `{report.camera_manifest}`",
        f"Frames: `{report.frame_count}`",
        "",
        "| Frame | Status | Timestamp | Object Z | Contacts | Errors |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for check in report.checks:
        object_z = "" if check.object_position is None else f"{check.object_position[2]:.6f}"
        errors = "; ".join(check.errors)
        lines.append(
            f"| {check.frame_index} | `{check.status}` | "
            f"{'' if check.timestamp is None else f'{check.timestamp:.6f}'} | "
            f"{object_z} | {'' if check.contact_count is None else check.contact_count} | {errors} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


if __name__ == "__main__":
    main()
