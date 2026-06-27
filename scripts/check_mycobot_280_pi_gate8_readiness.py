#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path

REQUIRED_MYCOBOT_MUJOCO_PATHS = [
    Path("xml/mycobot_280jn_mujoco.xml"),
]
REQUIRED_MYCOBOT_ROS_PATHS = [
    Path("mycobot_description/urdf/mycobot_280_pi/mycobot_280_pi.urdf"),
    Path("mycobot_description/urdf/adaptive_gripper/mycobot_adaptive_gripper.urdf"),
]
GATE_COMMANDS = {
    "gate7_static_contact": (
        "PYTHONPATH=src:. python3 scripts/mycobot_280_pi_adaptive_static_contact_smoke.py "
        "--asset-root {asset_root} --official-gripper-root {official_gripper_root}"
    ),
    "gate8_grasp_lift": (
        "PYTHONPATH=src:. python3 scripts/mycobot_280_pi_adaptive_grasp_lift_smoke.py "
        "--asset-root {asset_root} --official-gripper-root {official_gripper_root}"
    ),
    "gate8_teacher_dataset": (
        "PYTHONPATH=src:. python3 scripts/export_mycobot_280_pi_adaptive_teacher_dataset.py "
        "--asset-root {asset_root} --official-gripper-root {official_gripper_root} "
        "--episodes 10 --render-every 4"
    ),
}


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    status: str
    asset_root: str
    official_gripper_root: str
    checks: list[ReadinessCheck]
    next_commands: dict[str, str]
    claim_boundary: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight the runtime/assets needed for myCobot 280 Pi Gate 7/8 parity runs."
    )
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = check_readiness(
        asset_root=args.asset_root,
        official_gripper_root=args.official_gripper_root,
    )
    payload = json.dumps(asdict(report), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(0 if report.status == "passed" else 1)


def check_readiness(*, asset_root: Path, official_gripper_root: Path) -> ReadinessReport:
    checks: list[ReadinessCheck] = []
    checks.append(
        ReadinessCheck(
            name="python_package:mujoco",
            status="passed" if importlib.util.find_spec("mujoco") is not None else "failed",
            detail="mujoco import spec found" if importlib.util.find_spec("mujoco") is not None else "mujoco is not importable",
        )
    )
    checks.extend(_path_checks("mycobot_mujoco", asset_root, REQUIRED_MYCOBOT_MUJOCO_PATHS))
    checks.extend(_path_checks("mycobot_ros", official_gripper_root, REQUIRED_MYCOBOT_ROS_PATHS))
    failed = [check for check in checks if check.status != "passed"]
    commands = {
        name: template.format(asset_root=asset_root, official_gripper_root=official_gripper_root)
        for name, template in GATE_COMMANDS.items()
    }
    return ReadinessReport(
        status="passed" if not failed else "blocked",
        asset_root=str(asset_root),
        official_gripper_root=str(official_gripper_root),
        checks=checks,
        next_commands=commands,
        claim_boundary=(
            "This preflight only checks local runtime and asset availability. A passed preflight "
            "does not prove Gate 7/8 physics success; the smoke scripts must still be run and "
            "their rendered artifacts inspected."
        ),
    )


def _path_checks(label: str, root: Path, relative_paths: list[Path]) -> list[ReadinessCheck]:
    checks = [
        ReadinessCheck(
            name=f"path:{label}:root",
            status="passed" if root.exists() else "failed",
            detail=str(root),
        )
    ]
    for relative_path in relative_paths:
        path = root / relative_path
        checks.append(
            ReadinessCheck(
                name=f"path:{label}:{relative_path}",
                status="passed" if path.exists() else "failed",
                detail=str(path),
            )
        )
    return checks


if __name__ == "__main__":
    main()
