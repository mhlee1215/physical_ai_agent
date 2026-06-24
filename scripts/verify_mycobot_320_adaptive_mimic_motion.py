#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_ai_agent.sim.mycobot_nexus_env import (
    OFFICIAL_320_ADAPTIVE_URDF_RELATIVE_PATH,
    OFFICIAL_320_GRIPPER_JOINT_NAMES,
    OFFICIAL_320_GRIPPER_LINK_NAMES,
    OFFICIAL_320_LINK_NAMES,
    _convert_collada_mesh_to_obj,
)


FLOAT_TOLERANCE = 1e-6
LEFT_TIP_LINK = "gripper_left1"
RIGHT_TIP_LINK = "gripper_right1"
DRIVER_JOINT = "gripper_controller"


@dataclass(frozen=True)
class MimicSample:
    label: str
    gripper_controller: float
    joint_values: dict[str, float]
    jaw_gap_xy: float
    left_tip_center: tuple[float, float, float]
    right_tip_center: tuple[float, float, float]


@dataclass(frozen=True)
class MimicMotionReport:
    status: str
    source_urdf: str
    driver_joint: str
    sample_count: int
    controller_increase_opens: bool
    closed_jaw_gap_xy: float
    open_jaw_gap_xy: float
    jaw_gap_delta: float
    samples: list[MimicSample]
    artifacts: dict[str, str]


@dataclass(frozen=True)
class LinkVisual:
    link_name: str
    origin: tuple[float, float, float]
    visual_center: tuple[float, float, float]
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify myCobot 320 adaptive gripper mimic motion and jaw-gap "
            "direction from the upstream ROS2 URDF."
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
        default=Path("_workspace/mycobot_320_adaptive_mimic_motion_verify"),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of gripper_controller samples across the upstream range.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify_adaptive_mimic_motion(
        official_gripper_root=args.official_gripper_root,
        output_dir=args.output_dir,
        sample_count=args.samples,
    )
    print(json.dumps(_report_to_jsonable(report), indent=2, sort_keys=True))
    raise SystemExit(0 if report.status == "passed" else 1)


def verify_adaptive_mimic_motion(
    *,
    official_gripper_root: Path,
    output_dir: Path,
    sample_count: int = 5,
) -> MimicMotionReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = official_gripper_root / OFFICIAL_320_ADAPTIVE_URDF_RELATIVE_PATH
    root = ET.parse(urdf_path).getroot()
    joints = _urdf_joints(root)
    driver = joints[DRIVER_JOINT]
    lower = float(driver["lower"])
    upper = float(driver["upper"])
    sample_count = max(5, int(sample_count))
    driver_values = [
        lower + (upper - lower) * index / (sample_count - 1)
        for index in range(sample_count)
    ]
    labels = _sample_labels(sample_count)
    link_visual_specs = _urdf_link_visual_specs(root)

    samples: list[MimicSample] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mesh_vertices = _mesh_vertices_by_link(
            official_gripper_root,
            link_visual_specs,
            tmp_path,
        )
        for label, controller_value in zip(labels, driver_values, strict=True):
            joint_values = _expanded_joint_values(joints, controller_value)
            visuals = _link_visuals_at_pose(joints, link_visual_specs, mesh_vertices, joint_values)
            left = visuals[LEFT_TIP_LINK].visual_center
            right = visuals[RIGHT_TIP_LINK].visual_center
            samples.append(
                MimicSample(
                    label=label,
                    gripper_controller=controller_value,
                    joint_values={
                        name: joint_values[name]
                        for name in OFFICIAL_320_GRIPPER_JOINT_NAMES
                    },
                    jaw_gap_xy=_xy_distance(left, right),
                    left_tip_center=left,
                    right_tip_center=right,
                )
            )

    monotonic_opening = all(
        later.jaw_gap_xy >= earlier.jaw_gap_xy - FLOAT_TOLERANCE
        for earlier, later in zip(samples, samples[1:], strict=False)
    )
    close_reduces_gap = samples[0].jaw_gap_xy < samples[-1].jaw_gap_xy
    report = MimicMotionReport(
        status=(
            "passed"
            if monotonic_opening and close_reduces_gap
            else "failed"
        ),
        source_urdf=str(urdf_path),
        driver_joint=DRIVER_JOINT,
        sample_count=len(samples),
        controller_increase_opens=monotonic_opening,
        closed_jaw_gap_xy=samples[0].jaw_gap_xy,
        open_jaw_gap_xy=samples[-1].jaw_gap_xy,
        jaw_gap_delta=samples[-1].jaw_gap_xy - samples[0].jaw_gap_xy,
        samples=samples,
        artifacts={},
    )
    return replace(report, artifacts=_write_artifacts(report, output_dir))


def _sample_labels(count: int) -> list[str]:
    labels = []
    for index in range(count):
        if index == 0:
            labels.append("closed")
        elif index == count // 2:
            labels.append("middle")
        elif index == count - 1:
            labels.append("open")
        else:
            labels.append(f"sample_{index}")
    return labels


def _urdf_joints(root: ET.Element) -> dict[str, dict[str, Any]]:
    joints: dict[str, dict[str, Any]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        joints[joint.attrib["name"]] = {
            "name": joint.attrib["name"],
            "type": joint.attrib.get("type", "fixed"),
            "parent": parent.attrib["link"],
            "child": child.attrib["link"],
            "xyz": _origin_tuple(origin, "xyz"),
            "rpy": _origin_tuple(origin, "rpy"),
            "axis": _float_tuple(axis.attrib.get("xyz", "0 0 1") if axis is not None else "0 0 1"),
            "lower": float(limit.attrib.get("lower", "0")) if limit is not None else 0.0,
            "upper": float(limit.attrib.get("upper", "0")) if limit is not None else 0.0,
            "mimic_joint": mimic.attrib.get("joint") if mimic is not None else None,
            "mimic_multiplier": (
                float(mimic.attrib.get("multiplier", "1")) if mimic is not None else 1.0
            ),
            "mimic_offset": float(mimic.attrib.get("offset", "0")) if mimic is not None else 0.0,
        }
    return joints


def _expanded_joint_values(
    joints: dict[str, dict[str, Any]],
    controller_value: float,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in OFFICIAL_320_GRIPPER_JOINT_NAMES:
        joint = joints[name]
        if name == DRIVER_JOINT:
            raw_value = controller_value
        elif joint["mimic_joint"] == DRIVER_JOINT:
            raw_value = (
                controller_value * float(joint["mimic_multiplier"])
                + float(joint["mimic_offset"])
            )
        else:
            raw_value = 0.0
        values[name] = max(float(joint["lower"]), min(float(joint["upper"]), raw_value))
    return values


def _urdf_link_visual_specs(root: ET.Element) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        visual = link.find("visual")
        mesh = visual.find("geometry/mesh") if visual is not None else None
        if mesh is None:
            continue
        origin = visual.find("origin") if visual is not None else None
        specs[link_name] = {
            "filename": mesh.attrib["filename"],
            "xyz": _origin_tuple(origin, "xyz"),
            "rpy": _origin_tuple(origin, "rpy"),
        }
    missing = [name for name in OFFICIAL_320_LINK_NAMES if name not in specs]
    if missing:
        raise ValueError("missing URDF visual specs for: " + ", ".join(missing))
    return specs


def _mesh_vertices_by_link(
    official_gripper_root: Path,
    link_visual_specs: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> dict[str, list[tuple[float, float, float]]]:
    mesh_vertices: dict[str, list[tuple[float, float, float]]] = {}
    for link_name in OFFICIAL_320_LINK_NAMES:
        spec = link_visual_specs[link_name]
        obj_path = tmp_path / f"{link_name}.obj"
        dae_path = _resolve_package_mesh(official_gripper_root, spec["filename"])
        _convert_collada_mesh_to_obj(dae_path, obj_path, bake_visual_scene=False)
        mesh_vertices[link_name] = _obj_vertices(obj_path)
    return mesh_vertices


def _link_visuals_at_pose(
    joints: dict[str, dict[str, Any]],
    link_visual_specs: dict[str, dict[str, Any]],
    mesh_vertices: dict[str, list[tuple[float, float, float]]],
    joint_values: dict[str, float],
) -> dict[str, LinkVisual]:
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    for joint in joints.values():
        children_by_parent.setdefault(str(joint["parent"]), []).append(joint)
    link_poses = {"base": _identity_matrix4()}
    _walk_link_poses("base", children_by_parent, joint_values, link_poses)
    visuals: dict[str, LinkVisual] = {}
    for link_name in OFFICIAL_320_LINK_NAMES:
        spec = link_visual_specs[link_name]
        transform = _matrix_multiply(
            link_poses[link_name],
            _xyz_rpy_matrix(spec["xyz"], spec["rpy"]),
        )
        world_vertices = [
            _transform_point(transform, vertex)
            for vertex in mesh_vertices[link_name]
        ]
        visuals[link_name] = LinkVisual(
            link_name=link_name,
            origin=_matrix_translation(link_poses[link_name]),
            visual_center=_center(world_vertices),
            bbox=_bbox(world_vertices),
        )
    return visuals


def _walk_link_poses(
    parent: str,
    children_by_parent: dict[str, list[dict[str, Any]]],
    joint_values: dict[str, float],
    link_poses: dict[str, list[list[float]]],
) -> None:
    for joint in children_by_parent.get(parent, []):
        child = str(joint["child"])
        joint_origin = _xyz_rpy_matrix(joint["xyz"], joint["rpy"])
        joint_motion = (
            _axis_angle_matrix(joint["axis"], joint_values.get(str(joint["name"]), 0.0))
            if joint["type"] != "fixed"
            else _identity_matrix4()
        )
        link_poses[child] = _matrix_multiply(
            link_poses[parent],
            _matrix_multiply(joint_origin, joint_motion),
        )
        _walk_link_poses(child, children_by_parent, joint_values, link_poses)


def _write_artifacts(report: MimicMotionReport, output_dir: Path) -> dict[str, str]:
    json_path = output_dir / "mimic_motion_report.json"
    md_path = output_dir / "mimic_motion_report.md"
    svg_path = output_dir / "mimic_motion_gate.svg"
    json_path.write_text(
        json.dumps(_report_to_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    svg_path.write_text(_render_svg(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "svg": str(svg_path)}


def _render_markdown(report: MimicMotionReport) -> str:
    lines = [
        "# myCobot 320 Adaptive Mimic Motion Gate",
        "",
        f"- status: `{report.status}`",
        f"- sample count: `{report.sample_count}`",
        f"- controller increase opens: `{report.controller_increase_opens}`",
        f"- closed jaw gap xy: `{report.closed_jaw_gap_xy:.12g}`",
        f"- open jaw gap xy: `{report.open_jaw_gap_xy:.12g}`",
        f"- jaw gap delta: `{report.jaw_gap_delta:.12g}`",
        "",
        "| label | gripper_controller | jaw gap xy |",
        "|---|---:|---:|",
    ]
    for sample in report.samples:
        lines.append(
            f"| `{sample.label}` | {sample.gripper_controller:.12g} | "
            f"{sample.jaw_gap_xy:.12g} |"
        )
    return "\n".join(lines) + "\n"


def _render_svg(report: MimicMotionReport) -> str:
    width = 1320
    height = 1540
    status_color = "#2e7d32" if report.status == "passed" else "#b3261e"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f8f5"/>',
        _svg_text(36, 52, "myCobot 320 adaptive mimic motion gate", size=30, weight="700"),
        f'<rect x="36" y="76" width="140" height="34" rx="6" fill="{status_color}"/>',
        _svg_text(58, 99, report.status.upper(), size=17, fill="#ffffff"),
        _svg_text(
            204,
            99,
            (
                f"{report.sample_count} samples; jaw gap "
                f"{report.closed_jaw_gap_xy:.3f} -> {report.open_jaw_gap_xy:.3f} m"
            ),
            size=17,
            fill="#4b4f52",
        ),
    ]
    x = 46
    y = 150
    panel_width = 1228
    for sample in _key_samples(report.samples):
        parts.extend(_render_sample_panel(x, y, panel_width, 260, sample))
        y += 290
    parts.extend(_render_table(46, 1040, report))
    parts.append("</svg>")
    return "\n".join(parts)


def _key_samples(samples: list[MimicSample]) -> list[MimicSample]:
    return [samples[0], samples[len(samples) // 2], samples[-1]]


def _render_sample_panel(
    x: int,
    y: int,
    width: int,
    height: int,
    sample: MimicSample,
) -> list[str]:
    points = [sample.left_tip_center, sample.right_tip_center]
    min_x = min(point[0] for point in points) - 0.05
    max_x = max(point[0] for point in points) + 0.05
    min_y = min(point[1] for point in points) - 0.05
    max_y = max(point[1] for point in points) + 0.05
    mapper = _make_xy_mapper((min_x, max_x, min_y, max_y), x + 32, y + 72, width - 64, height - 118)
    lx, ly = mapper(sample.left_tip_center)
    rx, ry = mapper(sample.right_tip_center)
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="#ffffff"/>',
        f'<rect x="{x}" y="{y}" width="{width}" height="44" rx="8" fill="#eef1ec"/>',
        _svg_text(x + 18, y + 29, sample.label, size=18, weight="700"),
        _svg_text(
            x + 18,
            y + 66,
            f"controller {sample.gripper_controller:.3f}; gap {sample.jaw_gap_xy:.3f} m",
            size=13,
            fill="#4b4f52",
        ),
        f'<line x1="{lx:.2f}" y1="{ly:.2f}" x2="{rx:.2f}" y2="{ry:.2f}" '
        'stroke="#202124" stroke-width="3"/>',
        f'<circle cx="{lx:.2f}" cy="{ly:.2f}" r="11" fill="#b75f38"/>',
        f'<circle cx="{rx:.2f}" cy="{ry:.2f}" r="11" fill="#3f73ad"/>',
        _svg_text(int(lx + 14), int(ly - 10), LEFT_TIP_LINK, size=12),
        _svg_text(int(rx + 14), int(ry - 10), RIGHT_TIP_LINK, size=12),
    ]
    return parts


def _render_table(x: int, y: int, report: MimicMotionReport) -> list[str]:
    parts = [
        _svg_text(x, y, "sampled mimic expansion", size=18, weight="700"),
        f'<rect x="{x}" y="{y + 18}" width="1428" height="210" rx="8" fill="#ffffff"/>',
    ]
    row_y = y + 54
    for sample in report.samples:
        parts.append(
            _svg_text(
                x + 18,
                row_y,
                (
                    f"{sample.label}: controller={sample.gripper_controller:.3f}, "
                    f"jaw_gap_xy={sample.jaw_gap_xy:.4f}, "
                    f"left2={sample.joint_values['gripper_base_to_gripper_left2']:.3f}, "
                    f"left1={sample.joint_values['gripper_left3_to_gripper_left1']:.3f}, "
                    f"right3={sample.joint_values['gripper_base_to_gripper_right3']:.3f}, "
                    f"right2={sample.joint_values['gripper_base_to_gripper_right2']:.3f}, "
                    f"right1={sample.joint_values['gripper_right3_to_gripper_right1']:.3f}"
                ),
                size=13,
                fill="#202124",
            )
        )
        row_y += 34
    return parts


def _make_xy_mapper(
    bounds: tuple[float, float, float, float],
    x: int,
    y: int,
    width: int,
    height: int,
) -> Any:
    min_x, max_x, min_y, max_y = bounds
    scale = min(width / (max_x - min_x), height / (max_y - min_y))
    content_w = (max_x - min_x) * scale
    content_h = (max_y - min_y) * scale
    offset_x = x + (width - content_w) * 0.5
    offset_y = y + (height - content_h) * 0.5

    def map_point(point: tuple[float, float, float]) -> tuple[float, float]:
        px = offset_x + (point[0] - min_x) * scale
        py = offset_y + content_h - (point[1] - min_y) * scale
        return px, py

    return map_point


def _resolve_package_mesh(official_gripper_root: Path, filename: str) -> Path:
    if not filename.startswith("package://mycobot_description/"):
        raise ValueError(f"unsupported mesh filename: {filename}")
    relative = filename.removeprefix("package://mycobot_description/")
    return official_gripper_root / "mycobot_description" / relative


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


def _center(vertices: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return tuple(sum(vertex[index] for vertex in vertices) / len(vertices) for index in range(3))


def _xy_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _xyz_rpy_matrix(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> list[list[float]]:
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rotation = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], xyz[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], xyz[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _axis_angle_matrix(
    axis: tuple[float, float, float],
    angle: float,
) -> list[list[float]]:
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 0:
        return _identity_matrix4()
    x, y, z = x / norm, y / norm, z / norm
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _identity_matrix4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matrix_multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def _transform_point(
    matrix: list[list[float]],
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def _matrix_translation(matrix: list[list[float]]) -> tuple[float, float, float]:
    return (matrix[0][3], matrix[1][3], matrix[2][3])


def _float_tuple(raw: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in raw.split())
    if len(values) != 3:
        raise ValueError(f"expected xyz/rpy triplet, got: {raw}")
    return values


def _origin_tuple(origin: ET.Element | None, field: str) -> tuple[float, float, float]:
    return _float_tuple(origin.attrib.get(field, "0 0 0") if origin is not None else "0 0 0")


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


def _report_to_jsonable(report: MimicMotionReport) -> dict[str, Any]:
    return {
        **asdict(report),
        "samples": [asdict(item) for item in report.samples],
    }


if __name__ == "__main__":
    main()
