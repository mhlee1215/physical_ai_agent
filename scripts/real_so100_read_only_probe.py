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


def run_probe(port: str, output: Path) -> dict[str, Any]:
    bus, motors = _make_so100_bus(port)
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "port": port,
        "robot_kind": "so100_follower",
        "operation": "read_only_present_position",
        "writes_intended": False,
        "disconnect_disable_torque": False,
        "motor_names": list(motors.keys()),
        "ok": False,
    }

    try:
        bus.connect(handshake=True)
        report["positions_raw"] = bus.sync_read("Present_Position", normalize=False)
        report["ok"] = True
    except Exception as exc:  # noqa: BLE001 - hardware probe should preserve exact failure.
        report["error"] = repr(exc)
    finally:
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        except Exception as exc:  # noqa: BLE001
            report["disconnect_error"] = repr(exc)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only SO-100 follower present-position probe.")
    parser.add_argument("--port", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_workspace/real_so100/read_only_discovery_report.json"),
    )
    args = parser.parse_args()
    print(json.dumps(run_probe(args.port, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
