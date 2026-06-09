#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


SO100_MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}


def disable_so100_torque(port: str, output: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_disable_all_torque",
        "port": port,
        "writes_intended": True,
        "motor_ids": {name: motor.id for name, motor in SO100_MOTORS.items()},
        "ok": False,
        "attempts": [],
    }

    full_attempt = _disable_group(port=port, motors=SO100_MOTORS, handshake=True)
    report["attempts"].append(full_attempt)
    if full_attempt.get("ok"):
        report["ok"] = True
        report["torque_after"] = full_attempt.get("torque_after")
        _write_json(output, report)
        return report

    direct_results = {}
    for name, motor in SO100_MOTORS.items():
        direct_results[name] = _disable_group(port=port, motors={name: motor}, handshake=False)
    report["direct_motor_attempts"] = direct_results
    report["torque_after"] = {
        name: result.get("torque_after", {}).get(name)
        for name, result in direct_results.items()
    }
    report["ok"] = all(result.get("ok") for result in direct_results.values())
    _write_json(output, report)
    return report


def _disable_group(*, port: str, motors: dict[str, Motor], handshake: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "handshake": handshake,
        "motor_ids": {name: motor.id for name, motor in motors.items()},
        "ok": False,
    }
    bus = FeetechMotorsBus(port=port, motors=motors)
    try:
        bus.connect(handshake=handshake)
        result["connected"] = True
        result["torque_before"] = _read_torque(bus, motors)
        bus.disable_torque(num_retry=5)
        result["torque_after"] = _read_torque(bus, motors)
        result["ok"] = all(int(value) == 0 for value in result["torque_after"].values())
    except Exception as exc:  # noqa: BLE001 - hardware recovery report preserves exact failure.
        result["error"] = repr(exc)
    finally:
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        except Exception as exc:  # noqa: BLE001
            result["disconnect_error"] = repr(exc)
    return result


def _read_torque(bus: FeetechMotorsBus, motors: dict[str, Motor]) -> dict[str, int | str]:
    values: dict[str, int | str] = {}
    for name in motors:
        try:
            values[name] = int(bus.read("Torque_Enable", name, normalize=False))
        except Exception as exc:  # noqa: BLE001
            values[name] = f"error: {exc!r}"
    return values


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Disable torque on all SO-100 follower motors.")
    parser.add_argument("--port", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_workspace/real_so100/disable_torque_report.json"),
    )
    args = parser.parse_args()
    print(json.dumps(disable_so100_torque(args.port, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
