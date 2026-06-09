#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _make_so100_bus(port: str):
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    return FeetechMotorsBus(port=port, motors=motors), motors


def _open_cameras(indexes: list[int]):
    import cv2

    captures = {}
    for index in indexes:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(f"camera index {index} did not open")
        captures[index] = cap
    return captures


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def record_observation(
    *,
    port: str,
    camera_indexes: list[int],
    output_dir: Path,
    duration_seconds: float,
    fps: float,
    task: str,
    calibration_file: Path | None,
    camera_roles: dict[str, str] | None = None,
    policy_camera_indexes: list[int] | None = None,
    observer_camera_indexes: list[int] | None = None,
    allow_camera_only_without_robot: bool = False,
) -> dict[str, Any]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "episode.jsonl"
    robot_states_path = output_dir / "robot_states.jsonl"
    episode_path.write_text("", encoding="utf-8")
    robot_states_path.write_text("", encoding="utf-8")

    bus, motors = _make_so100_bus(port)
    captures = {}
    robot_connected = False
    robot_connection_error = None
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest: dict[str, Any] = {
        "started_at": started_at,
        "robot_kind": "so100_follower",
        "port": port,
        "camera_indexes": camera_indexes,
        "camera_roles": camera_roles or {},
        "policy_camera_indexes": policy_camera_indexes or camera_indexes,
        "observer_camera_indexes": observer_camera_indexes or [],
        "duration_seconds": duration_seconds,
        "target_fps": fps,
        "task": task,
        "actuation_enabled": False,
        "policy_actions_executed": False,
        "send_action_called": False,
        "writes_intended": False,
        "disconnect_disable_torque": False,
        "allow_camera_only_without_robot": allow_camera_only_without_robot,
        "robot_connected": False,
        "robot_state_available": False,
        "calibration_file": str(calibration_file) if calibration_file else None,
        "motor_names": list(motors.keys()),
        "ok": False,
    }

    frame_index = 0
    try:
        try:
            bus.connect(handshake=True)
            robot_connected = True
            manifest["robot_connected"] = True
        except Exception as exc:  # noqa: BLE001 - preserve hardware failure details while allowing camera-only evidence.
            robot_connection_error = repr(exc)
            manifest["robot_connection_error"] = robot_connection_error
            if not allow_camera_only_without_robot:
                raise
        captures = _open_cameras(camera_indexes)
        period = 1.0 / fps
        deadline = time.monotonic() + duration_seconds
        next_sample = time.monotonic()

        while time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_sample:
                time.sleep(min(next_sample - now, 0.01))
                continue

            wall_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            state = bus.sync_read("Present_Position", normalize=False) if robot_connected else None
            if state is not None:
                manifest["robot_state_available"] = True
            camera_paths: dict[str, str] = {}
            camera_shapes: dict[str, list[int]] = {}

            for camera_index, cap in captures.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    camera_paths[str(camera_index)] = ""
                    continue
                frame_path = frames_dir / f"camera_{camera_index}_{frame_index:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                camera_paths[str(camera_index)] = str(frame_path)
                camera_shapes[str(camera_index)] = list(frame.shape)

            robot_record = {
                "frame_index": frame_index,
                "monotonic_time": now,
                "wall_time": wall_time,
                "positions_raw": state,
                "robot_connected": robot_connected,
                "robot_state_available": state is not None,
            }
            episode_record = {
                "episode_index": 0,
                "frame_index": frame_index,
                "monotonic_time": now,
                "wall_time": wall_time,
                "task": task,
                "observation": {
                    "state": state,
                    "state_available": state is not None,
                    "state_source": "live_so100_readback" if state is not None else "unavailable_camera_only",
                    "images": camera_paths,
                    "image_shapes": camera_shapes,
                    "camera_roles": camera_roles or {},
                    "policy_camera_indexes": policy_camera_indexes or camera_indexes,
                    "observer_camera_indexes": observer_camera_indexes or [],
                },
                "action": None,
                "actuation_enabled": False,
            }
            with robot_states_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(robot_record, sort_keys=True) + "\n")
            with episode_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(episode_record, sort_keys=True) + "\n")

            frame_index += 1
            next_sample += period

        manifest["ok"] = True
        manifest["frames_recorded"] = frame_index
        manifest["mode"] = "live_readback_and_camera" if robot_connected else "camera_only_without_robot"
        manifest["episode_jsonl"] = str(episode_path)
        manifest["robot_states_jsonl"] = str(robot_states_path)
        manifest["frames_dir"] = str(frames_dir)
    except Exception as exc:  # noqa: BLE001 - preserve hardware/camera failure details.
        manifest["error"] = repr(exc)
    finally:
        for cap in captures.values():
            cap.release()
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        except Exception as exc:  # noqa: BLE001
            manifest["disconnect_error"] = repr(exc)

    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parse_camera_roles(items: list[str]) -> dict[str, str]:
    roles = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"--camera-role must be INDEX:ROLE, got {item!r}")
        index, role = item.split(":", 1)
        roles[index.strip()] = role.strip()
    return roles


def main() -> None:
    parser = argparse.ArgumentParser(description="Observation-only SO-100 camera + joint-state recorder.")
    parser.add_argument("--port", required=True)
    parser.add_argument("--camera-index", type=int, action="append", required=True)
    parser.add_argument(
        "--camera-role",
        action="append",
        default=[],
        help="Map one camera index to a role, e.g. 0:wrist_cam, 1:egocentric_cam, 3:codex_observer.",
    )
    parser.add_argument("--policy-camera-index", type=int, action="append", default=[])
    parser.add_argument("--observer-camera-index", type=int, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--task", default="observe_green_doll")
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument(
        "--allow-camera-only-without-robot",
        action="store_true",
        help="If serial readback is unavailable, still save policy-camera frames with observation.state=null.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            record_observation(
                port=args.port,
                camera_indexes=args.camera_index,
                output_dir=args.output_dir,
                duration_seconds=args.duration_seconds,
                fps=args.fps,
                task=args.task,
                calibration_file=args.calibration_file,
                camera_roles=_parse_camera_roles(args.camera_role),
                policy_camera_indexes=args.policy_camera_index or args.camera_index,
                observer_camera_indexes=args.observer_camera_index,
                allow_camera_only_without_robot=args.allow_camera_only_without_robot,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
