#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scripts.real_so100_micro_step import _make_so100_bus


REGISTERS = [
    "Present_Position",
    "Present_Velocity",
    "Present_Load",
    "Present_Voltage",
    "Present_Temperature",
    "Present_Current",
    "Moving",
    "Torque_Enable",
]


def read_motor_state_snapshot(*, port: str, calibration: Path | None, output: Path) -> dict[str, Any]:
    calibration_payload = _load_calibration(calibration) if calibration is not None else {}
    bus, motors = _make_so100_bus(port)
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_motor_state_snapshot",
        "port": port,
        "calibration": str(calibration) if calibration is not None else None,
        "writes_intended": False,
        "disconnect_disable_torque": False,
        "registers_requested": REGISTERS,
        "motor_ids": {name: motor.id for name, motor in motors.items()},
        "motors": {},
        "ok": False,
    }
    try:
        bus.connect(handshake=True)
        register_values: dict[str, dict[str, Any]] = {}
        for register in REGISTERS:
            register_values[register] = _read_register_for_all_motors(bus, register, motors)
        for name in motors:
            state = {register: register_values[register].get(name) for register in REGISTERS}
            _attach_calibration_position_metadata(state, name, calibration_payload)
            report["motors"][name] = state
        report["ok"] = True
    except Exception as exc:  # noqa: BLE001 - preserve hardware error exactly.
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


def _read_register_for_all_motors(bus: Any, register: str, motors: dict[str, Any]) -> dict[str, Any]:
    try:
        values = bus.sync_read(register, normalize=False)
        return {name: _jsonable(value) for name, value in values.items()}
    except Exception as sync_exc:  # noqa: BLE001
        values: dict[str, Any] = {}
        for name in motors:
            try:
                values[name] = _jsonable(bus.read(register, name, normalize=False))
            except Exception as read_exc:  # noqa: BLE001
                values[name] = {"error": repr(read_exc), "sync_error": repr(sync_exc)}
        return values


def _attach_calibration_position_metadata(
    state: dict[str, Any],
    name: str,
    calibration: dict[str, dict[str, float]],
) -> None:
    item = calibration.get(name)
    position = state.get("Present_Position")
    if item is None or not isinstance(position, (int, float)):
        return
    low = item["range_min"]
    high = item["range_max"]
    span = high - low
    state["calibration_range_min"] = low
    state["calibration_range_max"] = high
    state["position_fraction_in_calibration"] = None if span == 0 else round((float(position) - low) / span, 6)
    state["inside_calibration_range"] = bool(low <= float(position) <= high)


def _load_calibration(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, float]] = {}
    for name, item in payload.items():
        if isinstance(item, dict) and "range_min" in item and "range_max" in item:
            result[name] = {
                "range_min": float(item["range_min"]),
                "range_max": float(item["range_max"]),
            }
    return result


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read all relevant SO-100 follower motor state registers.")
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_workspace/real_so100/motor_state_snapshot.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            read_motor_state_snapshot(port=args.port, calibration=args.calibration, output=args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
