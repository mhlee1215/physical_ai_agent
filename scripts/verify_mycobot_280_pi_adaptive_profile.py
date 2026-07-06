#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_ai_agent.sim.mycobot_nexus_env import (
    MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
    MYCOBOT_320_MODEL_JOINT_NAMES,
    OFFICIAL_280_PI_URDF_RELATIVE_PATH,
    OFFICIAL_ADAPTIVE_GRIPPER_URDF_RELATIVE_PATH,
    OFFICIAL_320_GRIPPER_JOINT_NAMES,
    TCP_SITE,
    build_mycobot_nexus_scene_model,
)


@dataclass(frozen=True)
class ProfileCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class ProfileReport:
    status: str
    model_profile: str
    official_root: str
    arm_urdf: str
    gripper_urdf: str
    generated_scene: str
    passed_check_count: int
    failed_check_count: int
    checks: list[ProfileCheck]
    artifacts: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the dry myCobot 280 Pi + adaptive gripper profile by composing "
            "the official ROS1 arm and gripper URDFs into a MuJoCo XML scene."
        )
    )
    parser.add_argument(
        "--official-gripper-root",
        type=Path,
        required=True,
        help="Local clone of elephantrobotics/mycobot_ros.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/mycobot_280_pi_adaptive_verify"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify_280_pi_adaptive_profile(
        official_gripper_root=args.official_gripper_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    raise SystemExit(0 if report.status == "passed" else 1)


def verify_280_pi_adaptive_profile(
    *,
    official_gripper_root: Path,
    output_dir: Path,
) -> ProfileReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "mycobot_280_pi_adaptive_scene.xml"
    root = official_gripper_root.expanduser()
    arm_urdf = root / OFFICIAL_280_PI_URDF_RELATIVE_PATH
    gripper_urdf = root / OFFICIAL_ADAPTIVE_GRIPPER_URDF_RELATIVE_PATH
    checks: list[ProfileCheck] = []

    for label, path in (("arm_urdf", arm_urdf), ("adaptive_gripper_urdf", gripper_urdf)):
        checks.append(
            ProfileCheck(
                name=label,
                status="passed" if path.exists() else "failed",
                detail=str(path),
            )
        )

    if arm_urdf.exists() and gripper_urdf.exists():
        build_mycobot_nexus_scene_model(
            model_path=Path(""),
            scene_path=scene_path,
            official_gripper_root=root,
            model_profile=MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
        )
        scene = ET.parse(scene_path).getroot()
        names = {element.attrib.get("name") for element in scene.iter()}
        mesh_files = {
            Path(element.attrib["file"]).name
            for element in scene.findall(".//mesh")
            if "file" in element.attrib
        }
        checks.extend(_required_name_checks(names))
        checks.extend(_required_mesh_checks(mesh_files))
    else:
        checks.append(
            ProfileCheck(
                name="generated_scene",
                status="failed",
                detail="skipped because one or more source URDFs are missing",
            )
        )

    failed = [check for check in checks if check.status != "passed"]
    report = ProfileReport(
        status="passed" if not failed else "failed",
        model_profile=MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
        official_root=str(root),
        arm_urdf=str(arm_urdf),
        gripper_urdf=str(gripper_urdf),
        generated_scene=str(scene_path),
        passed_check_count=len(checks) - len(failed),
        failed_check_count=len(failed),
        checks=checks,
        artifacts={},
    )
    artifacts = _write_artifacts(report, output_dir)
    return replace(report, artifacts=artifacts)


def _required_name_checks(names: set[str | None]) -> list[ProfileCheck]:
    required: list[str | tuple[str, ...]] = [
        *MYCOBOT_320_MODEL_JOINT_NAMES[:-1],
        ("joint6output_to_joint6", "joint7_to_joint6"),
        "joint6_flange",
        "gripper_base",
        *OFFICIAL_320_GRIPPER_JOINT_NAMES,
        "left_finger_pad",
        "right_finger_pad",
        TCP_SITE,
        "task_cube",
        "nexus_work_mat",
    ]
    checks = []
    for item in required:
        aliases = item if isinstance(item, tuple) else (item,)
        detail = "|".join(aliases)
        checks.append(
            ProfileCheck(
                name=f"scene_name:{detail}",
                status="passed" if any(name in names for name in aliases) else "failed",
                detail=detail,
            )
        )
    return checks


def _required_mesh_checks(mesh_files: set[str]) -> list[ProfileCheck]:
    required = ["gripper_base.obj", "gripper_left1.obj", "gripper_right1.obj"]
    return [
        ProfileCheck(
            name=f"mesh:{name}",
            status="passed" if name in mesh_files else "failed",
            detail=name,
        )
        for name in required
    ]


def _write_artifacts(report: ProfileReport, output_dir: Path) -> dict[str, str]:
    json_path = output_dir / "mycobot_280_pi_adaptive_profile_report.json"
    md_path = output_dir / "mycobot_280_pi_adaptive_profile_report.md"
    json_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# myCobot 280 Pi Adaptive Profile Verify",
        "",
        f"Status: `{report.status}`",
        f"Model profile: `{report.model_profile}`",
        f"Official root: `{report.official_root}`",
        f"Generated scene: `{report.generated_scene}`",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in report.checks:
        lines.append(f"| `{check.name}` | `{check.status}` | `{check.detail}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


if __name__ == "__main__":
    main()
