#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_ai_agent.sim.mycobot_nexus_env import (
    MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
    OFFICIAL_320_ADAPTIVE_URDF_RELATIVE_PATH,
    OFFICIAL_320_GRIPPER_MIMIC,
    OFFICIAL_320_LINK_NAMES,
    build_mycobot_nexus_scene_model,
)


FLOAT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class JointComparison:
    name: str
    parent: str
    child: str
    status: str
    errors: list[str]
    expected: dict[str, Any]
    actual: dict[str, Any]


@dataclass(frozen=True)
class KinematicTreeReport:
    status: str
    model_profile: str
    source_urdf: str
    generated_scene: str
    compared_joint_count: int
    passed_joint_count: int
    failed_joint_count: int
    comparisons: list[JointComparison]
    artifacts: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the official ROS2 320 adaptive gripper URDF kinematic tree "
            "against the generated MuJoCo body tree."
        )
    )
    parser.add_argument(
        "--official-gripper-root",
        type=Path,
        required=True,
        help="Local clone of elephantrobotics/mycobot_ros2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/mycobot_320_adaptive_tree_verify"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify_adaptive_kinematic_tree(
        official_gripper_root=args.official_gripper_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(_report_to_jsonable(report), indent=2, sort_keys=True))
    raise SystemExit(0 if report.status == "passed" else 1)


def verify_adaptive_kinematic_tree(
    *,
    official_gripper_root: Path,
    output_dir: Path,
) -> KinematicTreeReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "mycobot_nexus_scene.xml"
    build_mycobot_nexus_scene_model(
        model_path=Path(""),
        scene_path=scene_path,
        official_gripper_root=official_gripper_root,
        model_profile=MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
    )

    urdf_path = official_gripper_root / OFFICIAL_320_ADAPTIVE_URDF_RELATIVE_PATH
    expected = _expected_urdf_joints(urdf_path)
    actual = _actual_mujoco_joints(scene_path)

    comparisons: list[JointComparison] = []
    for name, expected_joint in expected.items():
        actual_joint = actual.get(name) or actual.get(_edge_key(expected_joint))
        errors = _compare_joint(expected_joint, actual_joint)
        comparisons.append(
            JointComparison(
                name=name,
                parent=str(expected_joint["parent"]),
                child=str(expected_joint["child"]),
                status="passed" if not errors else "failed",
                errors=errors,
                expected=expected_joint,
                actual=actual_joint or {},
            )
        )

    missing_constants = sorted(
        name
        for name, multiplier in OFFICIAL_320_GRIPPER_MIMIC.items()
        if multiplier != 1.0 or name != "gripper_controller"
        if name not in expected
    )
    for name in missing_constants:
        comparisons.append(
            JointComparison(
                name=name,
                parent="",
                child="",
                status="failed",
                errors=["mimic constant has no matching upstream URDF joint"],
                expected={},
                actual={"mimic_multiplier": OFFICIAL_320_GRIPPER_MIMIC[name]},
            )
        )

    failed = [item for item in comparisons if item.status != "passed"]
    report = KinematicTreeReport(
        status="passed" if not failed else "failed",
        model_profile=MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
        source_urdf=str(urdf_path),
        generated_scene=str(scene_path),
        compared_joint_count=len(comparisons),
        passed_joint_count=len(comparisons) - len(failed),
        failed_joint_count=len(failed),
        comparisons=comparisons,
        artifacts={},
    )
    artifacts = _write_artifacts(report, output_dir)
    return replace(report, artifacts=artifacts)


def _expected_urdf_joints(urdf_path: Path) -> dict[str, dict[str, Any]]:
    root = ET.parse(urdf_path).getroot()
    link_set = set(OFFICIAL_320_LINK_NAMES)
    joints: dict[str, dict[str, Any]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        parent = _required_child(joint, "parent").attrib["link"]
        child = _required_child(joint, "child").attrib["link"]
        if parent not in link_set or child not in link_set:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        joints[name] = {
            "name": name,
            "type": joint.attrib.get("type", "fixed"),
            "parent": parent,
            "child": child,
            "xyz": _origin_tuple(origin, "xyz"),
            "rpy": _origin_tuple(origin, "rpy"),
            "axis": _float_tuple(axis.attrib.get("xyz", "0 0 1") if axis is not None else "0 0 1"),
            "lower": float(limit.attrib.get("lower", "-3.14159")) if limit is not None else None,
            "upper": float(limit.attrib.get("upper", "3.14159")) if limit is not None else None,
            "mimic_joint": mimic.attrib.get("joint") if mimic is not None else None,
            "mimic_multiplier": (
                float(mimic.attrib.get("multiplier", "1.0")) if mimic is not None else None
            ),
            "mimic_offset": float(mimic.attrib.get("offset", "0.0")) if mimic is not None else None,
        }
    return joints


def _actual_mujoco_joints(scene_path: Path) -> dict[str, dict[str, Any]]:
    root = ET.parse(scene_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"missing worldbody in {scene_path}")
    joints: dict[str, dict[str, Any]] = {}
    for body in worldbody.findall("body"):
        _collect_mujoco_body_joints(body, parent_name=None, joints=joints)
    return joints


def _collect_mujoco_body_joints(
    body: ET.Element,
    *,
    parent_name: str | None,
    joints: dict[str, dict[str, Any]],
) -> None:
    body_name = body.attrib.get("name", "")
    joint = body.find("joint")
    if parent_name is not None:
        if joint is None:
            joint_name = f"{parent_name}_to_{body_name}"
            joint_type = "fixed"
            axis = (0.0, 0.0, 1.0)
            lower = None
            upper = None
        else:
            joint_name = joint.attrib["name"]
            joint_type = (
                "revolute" if joint.attrib.get("type") == "hinge" else joint.attrib.get("type")
            )
            axis = _float_tuple(joint.attrib.get("axis", "0 0 1"))
            joint_range = _float_tuple(joint.attrib.get("range", "0 0"))
            lower = joint_range[0]
            upper = joint_range[1]
        joints[joint_name] = {
            "name": joint_name,
            "type": joint_type,
            "parent": parent_name,
            "child": body_name,
            "xyz": _float_tuple(body.attrib.get("pos", "0 0 0")),
            "rpy": _float_tuple(body.attrib.get("euler", "0 0 0")),
            "axis": axis,
            "lower": lower,
            "upper": upper,
            "mimic_multiplier": OFFICIAL_320_GRIPPER_MIMIC.get(joint_name),
        }
        if joint is None:
            joints[_edge_key(joints[joint_name])] = joints[joint_name]
    for child in body.findall("body"):
        _collect_mujoco_body_joints(child, parent_name=body_name, joints=joints)


def _compare_joint(
    expected: dict[str, Any],
    actual: dict[str, Any] | None,
) -> list[str]:
    if actual is None:
        return ["missing generated MuJoCo joint/body edge"]
    errors: list[str] = []
    for field in ("parent", "child", "type"):
        if expected[field] != actual.get(field):
            errors.append(f"{field}: expected {expected[field]!r}, got {actual.get(field)!r}")
    for field in ("xyz", "rpy"):
        if not _tuple_close(expected[field], actual.get(field)):
            errors.append(f"{field}: expected {expected[field]}, got {actual.get(field)}")
    if expected["type"] != "fixed":
        if not _tuple_close(expected["axis"], actual.get("axis")):
            errors.append(f"axis: expected {expected['axis']}, got {actual.get('axis')}")
        if not _float_close(expected["lower"], actual.get("lower")):
            errors.append(f"lower: expected {expected['lower']}, got {actual.get('lower')}")
        if not _float_close(expected["upper"], actual.get("upper")):
            errors.append(f"upper: expected {expected['upper']}, got {actual.get('upper')}")
    if expected["mimic_joint"] is not None:
        if expected["mimic_joint"] != "gripper_controller":
            errors.append(f"unsupported mimic source: {expected['mimic_joint']}")
        if expected["mimic_offset"] not in (0, 0.0):
            errors.append(f"unsupported nonzero mimic offset: {expected['mimic_offset']}")
        actual_multiplier = actual.get("mimic_multiplier")
        if not _float_close(expected["mimic_multiplier"], actual_multiplier):
            errors.append(
                "mimic_multiplier: expected "
                f"{expected['mimic_multiplier']}, got {actual_multiplier}"
            )
    return errors


def _write_artifacts(report: KinematicTreeReport, output_dir: Path) -> dict[str, str]:
    json_path = output_dir / "kinematic_tree_report.json"
    md_path = output_dir / "kinematic_tree_report.md"
    svg_path = output_dir / "kinematic_tree.svg"
    json_path.write_text(
        json.dumps(_report_to_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    svg_path.write_text(_render_svg(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "svg": str(svg_path),
    }


def _render_markdown(report: KinematicTreeReport) -> str:
    lines = [
        "# myCobot 320 Adaptive Kinematic Tree Gate",
        "",
        f"- status: `{report.status}`",
        f"- compared joints: `{report.compared_joint_count}`",
        f"- passed: `{report.passed_joint_count}`",
        f"- failed: `{report.failed_joint_count}`",
        "",
        "| joint | parent | child | status | errors |",
        "|---|---|---|---|---|",
    ]
    for item in report.comparisons:
        errors = "<br>".join(item.errors) if item.errors else ""
        lines.append(
            f"| `{item.name}` | `{item.parent}` | `{item.child}` | "
            f"`{item.status}` | {errors} |"
        )
    return "\n".join(lines) + "\n"


def _render_svg(report: KinematicTreeReport) -> str:
    rows = report.comparisons
    width = 1420
    row_height = 46
    height = 110 + row_height * max(1, len(rows))
    status_color = "#2e7d32" if report.status == "passed" else "#b3261e"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f8f5"/>',
        '<text x="32" y="42" font-family="Arial" font-size="26" '
        'font-weight="700" fill="#202124">myCobot 320 adaptive kinematic tree gate</text>',
        f'<rect x="32" y="58" width="130" height="32" rx="6" fill="{status_color}"/>',
        f'<text x="50" y="80" font-family="Arial" font-size="16" fill="#fff">'
        f'{report.status.upper()}</text>',
        f'<text x="190" y="80" font-family="Arial" font-size="16" fill="#4b4f52">'
        f'{report.passed_joint_count}/{report.compared_joint_count} joints passed</text>',
    ]
    y = 118
    for index, item in enumerate(rows):
        fill = "#ffffff" if index % 2 == 0 else "#eef1ec"
        color = "#2e7d32" if item.status == "passed" else "#b3261e"
        parts.append(f'<rect x="32" y="{y - 28}" width="1356" height="38" rx="4" fill="{fill}"/>')
        parts.append(_svg_text(48, y - 4, item.parent, size=14, weight="700"))
        parts.append(_svg_text(250, y - 4, "->", size=14))
        parts.append(_svg_text(292, y - 4, item.child, size=14, weight="700"))
        parts.append(_svg_text(516, y - 4, item.name, size=14))
        parts.append(f'<circle cx="865" cy="{y - 9}" r="9" fill="{color}"/>')
        parts.append(_svg_text(884, y - 4, item.status, size=14))
        error_text = "; ".join(item.errors[:2])
        parts.append(_svg_text(980, y - 4, error_text, size=12, fill="#5f6368"))
        y += row_height
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_text(
    x: int,
    y: int,
    text: str,
    *,
    size: int,
    weight: str = "400",
    fill: str = "#202124",
) -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escaped}</text>'
    )


def _required_child(element: ET.Element, tag: str) -> ET.Element:
    child = element.find(tag)
    if child is None:
        raise ValueError(f"missing {tag} in {element.attrib}")
    return child


def _edge_key(joint: dict[str, Any]) -> str:
    return f"{joint['parent']}->{joint['child']}"


def _float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(value) for value in raw.split())


def _origin_tuple(origin: ET.Element | None, field: str) -> tuple[float, ...]:
    return _float_tuple(origin.attrib.get(field, "0 0 0") if origin is not None else "0 0 0")


def _tuple_close(left: Any, right: Any) -> bool:
    if left is None or right is None or len(left) != len(right):
        return False
    return all(_float_close(a, b) for a, b in zip(left, right, strict=True))


def _float_close(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), abs_tol=FLOAT_TOLERANCE)


def _report_to_jsonable(report: KinematicTreeReport) -> dict[str, Any]:
    return {
        **asdict(report),
        "comparisons": [asdict(item) for item in report.comparisons],
    }


if __name__ == "__main__":
    main()
