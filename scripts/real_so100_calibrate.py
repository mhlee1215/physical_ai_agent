#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def run_calibration(
    port: str,
    robot_id: str,
    calibration_dir: Path,
    manifest: Path,
    operator: str,
) -> dict[str, Any]:
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    cfg = SO100FollowerConfig(
        id=robot_id,
        port=port,
        calibration_dir=calibration_dir,
        cameras={},
        disable_torque_on_disconnect=False,
        use_degrees=True,
    )
    robot = SO100Follower(cfg)

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "robot_kind": "so100_follower",
        "robot_id": robot_id,
        "port": port,
        "calibration_dir": str(calibration_dir),
        "calibration_file": str(robot.calibration_fpath),
        "operation": "device_calibration",
        "operator": operator,
        "policy_actions_executed": False,
        "send_action_called": False,
        "connect_method": "bus.connect(handshake=True)",
        "disconnect_disable_torque": False,
        "ok": False,
    }

    try:
        calibration_dir.mkdir(parents=True, exist_ok=True)
        robot.bus.connect(handshake=True)
        robot.calibrate()
        report["calibration_file_exists"] = robot.calibration_fpath.is_file()
        report["calibrated_motors"] = sorted(robot.calibration)
        report["ok"] = True
    except Exception as exc:  # noqa: BLE001 - preserve hardware failure detail.
        report["error"] = repr(exc)
    finally:
        try:
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=False)
        except Exception as exc:  # noqa: BLE001
            report["disconnect_error"] = repr(exc)

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive SO-100 calibration with manifest output.")
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", default="so100_local")
    parser.add_argument("--operator", default="user")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("_workspace/real_so100/calibration"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("_workspace/real_so100/calibration_manifest.json"),
    )
    args = parser.parse_args()
    run_calibration(args.port, args.robot_id, args.calibration_dir, args.manifest, args.operator)


if __name__ == "__main__":
    main()
