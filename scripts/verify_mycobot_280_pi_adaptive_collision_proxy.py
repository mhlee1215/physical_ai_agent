#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_ai_agent.sim.mycobot_nexus_env import (
    ADAPTIVE_280_FINGER_PAD_FRICTION,
    ADAPTIVE_280_FINGER_PAD_SIZE,
    ADAPTIVE_280_LEFT_FINGER_PAD_POS,
    ADAPTIVE_280_RIGHT_FINGER_PAD_POS,
    ADAPTIVE_FINGER_PAD_CONDIM,
    ADAPTIVE_LEFT_FINGER_PAD_PARENT,
    ADAPTIVE_RIGHT_FINGER_PAD_PARENT,
    MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
    OFFICIAL_ADAPTIVE_GRIPPER_MESH_RELATIVE_PATH,
    OFFICIAL_ADAPTIVE_GRIPPER_URDF_RELATIVE_PATH,
    build_mycobot_nexus_scene_model,
    _convert_collada_mesh_to_obj,
)


FLOAT_TOLERANCE = 1e-6
PAD_MESH_CENTER_TOLERANCE = 0.018
PAD_TIP_EXTENSION_TOLERANCE = 0.006
PAD_SPECS = {
    "left_finger_pad": {
        "parent": ADAPTIVE_LEFT_FINGER_PAD_PARENT,
        "link": "gripper_left1",
        "pos": ADAPTIVE_280_LEFT_FINGER_PAD_POS,
        "selection": "280_pi_closed_fingertip_contact",
    },
    "right_finger_pad": {
        "parent": ADAPTIVE_RIGHT_FINGER_PAD_PARENT,
        "link": "gripper_right1",
        "pos": ADAPTIVE_280_RIGHT_FINGER_PAD_POS,
        "selection": "280_pi_closed_fingertip_contact",
    },
}


@dataclass(frozen=True)
class CollisionProxyComparison:
    pad_name: str
    status: str
    errors: list[str]
    expected: dict[str, Any]
    actual: dict[str, Any]


@dataclass(frozen=True)
class CollisionProxyReport:
    status: str
    model_profile: str
    source_urdf: str
    generated_scene: str
    compared_proxy_count: int
    passed_proxy_count: int
    failed_proxy_count: int
    comparisons: list[CollisionProxyComparison]
    artifacts: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify myCobot 280 Pi adaptive gripper collision proxy pads are "
            "attached to validated official finger link frames."
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
        default=Path("_workspace/mycobot_280_pi_adaptive_collision_proxy_verify"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify_280_pi_adaptive_collision_proxy(
        official_gripper_root=args.official_gripper_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(_report_to_jsonable(report), indent=2, sort_keys=True))
    raise SystemExit(0 if report.status == "passed" else 1)


def verify_280_pi_adaptive_collision_proxy(
    *,
    official_gripper_root: Path,
    output_dir: Path,
) -> CollisionProxyReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = official_gripper_root.expanduser()
    scene_path = output_dir / "mycobot_280_pi_adaptive_collision_proxy_scene.xml"
    build_mycobot_nexus_scene_model(
        model_path=Path(""),
        scene_path=scene_path,
        official_gripper_root=root,
        model_profile=MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
    )

    urdf_path = root / OFFICIAL_ADAPTIVE_GRIPPER_URDF_RELATIVE_PATH
    expected = _expected_proxy_specs(root)
    actual = _actual_proxy_specs(scene_path)
    comparisons = []
    for pad_name, expected_spec in expected.items():
        actual_spec = actual.get(pad_name, {})
        errors = _compare_proxy(expected_spec, actual_spec)
        comparisons.append(
            CollisionProxyComparison(
                pad_name=pad_name,
                status="passed" if not errors else "failed",
                errors=errors,
                expected=expected_spec,
                actual=actual_spec,
            )
        )

    failed = [item for item in comparisons if item.status != "passed"]
    report = CollisionProxyReport(
        status="passed" if not failed else "failed",
        model_profile=MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
        source_urdf=str(urdf_path),
        generated_scene=str(scene_path),
        compared_proxy_count=len(comparisons),
        passed_proxy_count=len(comparisons) - len(failed),
        failed_proxy_count=len(failed),
        comparisons=comparisons,
        artifacts={},
    )
    return replace(report, artifacts=_write_artifacts(report, output_dir))


def _expected_proxy_specs(official_gripper_root: Path) -> dict[str, dict[str, Any]]:
    specs = {}
    mesh_root = official_gripper_root / OFFICIAL_ADAPTIVE_GRIPPER_MESH_RELATIVE_PATH
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for pad_name, pad_spec in PAD_SPECS.items():
            link_name = str(pad_spec["link"])
            obj_path = tmp_path / f"{link_name}.obj"
            _convert_collada_mesh_to_obj(
                mesh_root / f"{link_name}.dae",
                obj_path,
                bake_visual_scene=False,
            )
            vertices = _obj_vertices(obj_path)
            mins, maxs = _bbox(vertices)
            specs[pad_name] = {
                "parent": pad_spec["parent"],
                "pos": pad_spec["pos"],
                "type": "box",
                "size": ADAPTIVE_280_FINGER_PAD_SIZE,
                "friction": ADAPTIVE_280_FINGER_PAD_FRICTION,
                "condim": ADAPTIVE_FINGER_PAD_CONDIM,
                "contype": 1,
                "conaffinity": 1,
                "selection": pad_spec["selection"],
                "mesh_bbox": {"mins": mins, "maxs": maxs},
                "mesh_alignment": _pad_mesh_alignment(pad_spec["pos"], mins, maxs),
            }
    return specs


def _pad_mesh_alignment(
    pad_pos: tuple[float, float, float],
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
) -> dict[str, Any]:
    mesh_center = tuple((mins[index] + maxs[index]) * 0.5 for index in range(3))
    signed_tip = maxs[0] if pad_pos[0] >= 0 else mins[0]
    pad_tip_delta = abs(abs(pad_pos[0]) - abs(signed_tip))
    lateral_delta = abs(pad_pos[1] - mesh_center[1])
    vertical_delta = abs(pad_pos[2] - mesh_center[2])
    errors = []
    if pad_tip_delta > PAD_TIP_EXTENSION_TOLERANCE:
        errors.append(
            "pad local X is not near the visible fingertip mesh tip: "
            f"pad_x={pad_pos[0]:.6g}, mesh_tip_x={signed_tip:.6g}, delta={pad_tip_delta:.6g}"
        )
    if lateral_delta > PAD_MESH_CENTER_TOLERANCE:
        errors.append(
            "pad local Y is too far from the visible fingertip mesh center: "
            f"pad_y={pad_pos[1]:.6g}, mesh_center_y={mesh_center[1]:.6g}, delta={lateral_delta:.6g}"
        )
    if vertical_delta > PAD_MESH_CENTER_TOLERANCE:
        errors.append(
            "pad local Z is too far from the visible fingertip mesh center: "
            f"pad_z={pad_pos[2]:.6g}, mesh_center_z={mesh_center[2]:.6g}, delta={vertical_delta:.6g}"
        )
    return {
        "passed": not errors,
        "errors": errors,
        "mesh_center": mesh_center,
        "signed_mesh_tip_x": signed_tip,
        "pad_tip_delta": pad_tip_delta,
        "lateral_delta": lateral_delta,
        "vertical_delta": vertical_delta,
        "tip_tolerance": PAD_TIP_EXTENSION_TOLERANCE,
        "center_tolerance": PAD_MESH_CENTER_TOLERANCE,
    }


def _actual_proxy_specs(scene_path: Path) -> dict[str, dict[str, Any]]:
    root = ET.parse(scene_path).getroot()
    actual: dict[str, dict[str, Any]] = {}
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"missing worldbody in {scene_path}")
    for body in worldbody.findall(".//body"):
        parent = body.attrib.get("name", "")
        for geom in body.findall("geom"):
            name = geom.attrib.get("name", "")
            if name not in PAD_SPECS:
                continue
            actual[name] = {
                "parent": parent,
                "pos": _float_tuple(geom.attrib["pos"]),
                "type": geom.attrib.get("type"),
                "size": _float_tuple(geom.attrib["size"]),
                "friction": _float_tuple(geom.attrib["friction"]),
                "condim": int(geom.attrib["condim"]),
                "contype": int(geom.attrib["contype"]),
                "conaffinity": int(geom.attrib["conaffinity"]),
            }
    return actual


def _compare_proxy(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    if not actual:
        return ["missing generated contact proxy geom"]
    errors = []
    for field in ("parent", "type", "condim", "contype", "conaffinity"):
        if expected[field] != actual.get(field):
            errors.append(f"{field}: expected {expected[field]!r}, got {actual.get(field)!r}")
    for field in ("pos", "size", "friction"):
        if not _tuple_close(expected[field], actual.get(field)):
            errors.append(f"{field}: expected {expected[field]}, got {actual.get(field)}")
    alignment = expected.get("mesh_alignment", {})
    if alignment and not alignment.get("passed", False):
        errors.extend(alignment.get("errors", []))
    return errors


def _write_artifacts(report: CollisionProxyReport, output_dir: Path) -> dict[str, str]:
    json_path = output_dir / "collision_proxy_report.json"
    md_path = output_dir / "collision_proxy_report.md"
    svg_path = output_dir / "collision_proxy_gate.svg"
    json_path.write_text(
        json.dumps(_report_to_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    svg_path.write_text(_render_svg(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "svg": str(svg_path)}


def _render_markdown(report: CollisionProxyReport) -> str:
    lines = [
        "# myCobot 280 Pi Adaptive Collision Proxy Gate",
        "",
        f"- status: `{report.status}`",
        f"- model profile: `{report.model_profile}`",
        f"- compared proxies: `{report.compared_proxy_count}`",
        f"- passed: `{report.passed_proxy_count}`",
        f"- failed: `{report.failed_proxy_count}`",
        "",
        "| pad | parent | status | pos | size | friction | errors |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in report.comparisons:
        errors = "<br>".join(item.errors) if item.errors else ""
        lines.append(
            f"| `{item.pad_name}` | `{item.expected['parent']}` | `{item.status}` | "
            f"`{_format_tuple(item.expected['pos'])}` | "
            f"`{_format_tuple(item.expected['size'])}` | "
            f"`{_format_tuple(item.expected['friction'])}` | {errors} |"
        )
    return "\n".join(lines) + "\n"


def _render_svg(report: CollisionProxyReport) -> str:
    width = 1320
    height = 680
    status_color = "#2e7d32" if report.status == "passed" else "#b3261e"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f8f5"/>',
        _svg_text(36, 52, "myCobot 280 Pi adaptive collision proxy gate", 30, "700"),
        f'<rect x="36" y="76" width="140" height="34" rx="6" fill="{status_color}"/>',
        _svg_text(58, 99, report.status.upper(), 17, "400", "#ffffff"),
        _svg_text(
            204,
            99,
            f"{report.passed_proxy_count}/{report.compared_proxy_count} proxies passed",
            17,
            "400",
            "#4b4f52",
        ),
    ]
    y = 150
    for item in report.comparisons:
        color = "#2e7d32" if item.status == "passed" else "#b3261e"
        parts.append(f'<rect x="48" y="{y}" width="1224" height="206" rx="8" fill="#ffffff"/>')
        parts.append(f'<circle cx="82" cy="{y + 38}" r="10" fill="{color}"/>')
        parts.append(_svg_text(104, y + 45, item.pad_name, 20, "700"))
        parts.append(_svg_text(104, y + 78, f"parent: {item.expected['parent']}", 15))
        parts.append(_svg_text(104, y + 108, f"pos: {_format_tuple(item.expected['pos'])}", 15))
        parts.append(_svg_text(104, y + 138, f"size: {_format_tuple(item.expected['size'])}", 15))
        parts.append(
            _svg_text(
                104,
                y + 168,
                f"friction/condim: {_format_tuple(item.expected['friction'])} / "
                f"{item.expected['condim']}",
                15,
            )
        )
        y += 236
    parts.append("</svg>")
    return "\n".join(parts)


def _obj_vertices(obj_path: Path) -> list[tuple[float, float, float]]:
    vertices: list[tuple[float, float, float]] = []
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()[:4]
            vertices.append((float(x), float(y), float(z)))
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {obj_path}")
    return vertices


def _bbox(
    vertices: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mins = tuple(min(vertex[index] for vertex in vertices) for index in range(3))
    maxs = tuple(max(vertex[index] for vertex in vertices) for index in range(3))
    return mins, maxs


def _float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(value) for value in raw.split())


def _tuple_close(left: Any, right: Any) -> bool:
    if left is None or right is None or len(left) != len(right):
        return False
    return all(abs(float(a) - float(b)) <= FLOAT_TOLERANCE for a, b in zip(left, right))


def _format_tuple(values: Any) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def _svg_text(
    x: int,
    y: int,
    text: str,
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


def _report_to_jsonable(report: CollisionProxyReport) -> dict[str, Any]:
    return {
        **asdict(report),
        "comparisons": [asdict(item) for item in report.comparisons],
    }


if __name__ == "__main__":
    main()
