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
    MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
    OFFICIAL_320_ADAPTIVE_URDF_RELATIVE_PATH,
    OFFICIAL_320_LINK_NAMES,
    build_mycobot_nexus_scene_model,
    _convert_collada_mesh_to_obj,
)


FLOAT_TOLERANCE = 1e-6
ARM_LINKS = set(OFFICIAL_320_LINK_NAMES[:7])


@dataclass(frozen=True)
class VisualPoseComparison:
    link_name: str
    status: str
    link_origin_delta: float
    visual_center_delta: float
    reference_origin: tuple[float, float, float]
    generated_origin: tuple[float, float, float]
    reference_visual_center: tuple[float, float, float]
    generated_visual_center: tuple[float, float, float]


@dataclass(frozen=True)
class VisualPoseReport:
    status: str
    source_urdf: str
    generated_scene: str
    compared_link_count: int
    passed_link_count: int
    failed_link_count: int
    max_link_origin_delta: float
    max_visual_center_delta: float
    comparisons: list[VisualPoseComparison]
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
            "Render and compare the upstream ROS2 myCobot 320 adaptive gripper "
            "zero-pose visual model against the generated MuJoCo body tree."
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
        default=Path("_workspace/mycobot_320_adaptive_visual_pose_verify"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify_adaptive_visual_pose(
        official_gripper_root=args.official_gripper_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(_report_to_jsonable(report), indent=2, sort_keys=True))
    raise SystemExit(0 if report.status == "passed" else 1)


def verify_adaptive_visual_pose(
    *,
    official_gripper_root: Path,
    output_dir: Path,
) -> VisualPoseReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "mycobot_nexus_scene.xml"
    build_mycobot_nexus_scene_model(
        model_path=Path(""),
        scene_path=scene_path,
        official_gripper_root=official_gripper_root,
        model_profile=MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
    )

    urdf_path = official_gripper_root / OFFICIAL_320_ADAPTIVE_URDF_RELATIVE_PATH
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        reference = _reference_urdf_visuals(urdf_path, official_gripper_root, tmp_path)
        generated = _generated_mujoco_visuals(scene_path)

    comparisons: list[VisualPoseComparison] = []
    for link_name in OFFICIAL_320_LINK_NAMES:
        expected = reference[link_name]
        actual = generated[link_name]
        link_origin_delta = _distance(expected.origin, actual.origin)
        visual_center_delta = _distance(expected.visual_center, actual.visual_center)
        comparisons.append(
            VisualPoseComparison(
                link_name=link_name,
                status=(
                    "passed"
                    if link_origin_delta <= FLOAT_TOLERANCE
                    and visual_center_delta <= FLOAT_TOLERANCE
                    else "failed"
                ),
                link_origin_delta=link_origin_delta,
                visual_center_delta=visual_center_delta,
                reference_origin=expected.origin,
                generated_origin=actual.origin,
                reference_visual_center=expected.visual_center,
                generated_visual_center=actual.visual_center,
            )
        )

    failed = [item for item in comparisons if item.status != "passed"]
    report = VisualPoseReport(
        status="passed" if not failed else "failed",
        source_urdf=str(urdf_path),
        generated_scene=str(scene_path),
        compared_link_count=len(comparisons),
        passed_link_count=len(comparisons) - len(failed),
        failed_link_count=len(failed),
        max_link_origin_delta=max(
            (item.link_origin_delta for item in comparisons),
            default=0.0,
        ),
        max_visual_center_delta=max(
            (item.visual_center_delta for item in comparisons),
            default=0.0,
        ),
        comparisons=comparisons,
        artifacts={},
    )
    return replace(report, artifacts=_write_artifacts(report, output_dir, reference, generated))


def _reference_urdf_visuals(
    urdf_path: Path,
    official_gripper_root: Path,
    tmp_path: Path,
) -> dict[str, LinkVisual]:
    root = ET.parse(urdf_path).getroot()
    link_visual_specs = _urdf_link_visual_specs(root)
    link_poses = _urdf_link_poses(root)
    visuals: dict[str, LinkVisual] = {}
    for link_name in OFFICIAL_320_LINK_NAMES:
        spec = link_visual_specs[link_name]
        obj_path = tmp_path / f"reference_{link_name}.obj"
        dae_path = _resolve_package_mesh(official_gripper_root, spec["filename"])
        _convert_collada_mesh_to_obj(dae_path, obj_path, bake_visual_scene=False)
        vertices = _obj_vertices(obj_path)
        transform = _matrix_multiply(
            link_poses[link_name],
            _xyz_rpy_matrix(spec["xyz"], spec["rpy"]),
        )
        world_vertices = [_transform_point(transform, vertex) for vertex in vertices]
        visuals[link_name] = LinkVisual(
            link_name=link_name,
            origin=_matrix_translation(link_poses[link_name]),
            visual_center=_center(world_vertices),
            bbox=_bbox(world_vertices),
        )
    return visuals


def _generated_mujoco_visuals(scene_path: Path) -> dict[str, LinkVisual]:
    root = ET.parse(scene_path).getroot()
    mesh_files = {
        mesh.attrib["name"]: Path(mesh.attrib["file"])
        for mesh in root.findall("asset/mesh")
        if "name" in mesh.attrib and "file" in mesh.attrib
    }
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"missing worldbody in {scene_path}")
    body_poses: dict[str, list[list[float]]] = {}
    body_nodes: dict[str, ET.Element] = {}
    for body in worldbody.findall("body"):
        _collect_body_poses(body, _identity_matrix4(), body_poses, body_nodes)

    visuals: dict[str, LinkVisual] = {}
    for link_name in OFFICIAL_320_LINK_NAMES:
        body = body_nodes[link_name]
        geom = body.find("geom[@type='mesh']")
        if geom is None:
            raise ValueError(f"missing generated visual mesh geom for {link_name}")
        mesh_path = mesh_files[geom.attrib["mesh"]]
        vertices = _obj_vertices(mesh_path)
        transform = _matrix_multiply(
            body_poses[link_name],
            _xyz_rpy_matrix(
                _float_tuple(geom.attrib.get("pos", "0 0 0")),
                _float_tuple(geom.attrib.get("euler", "0 0 0")),
            ),
        )
        world_vertices = [_transform_point(transform, vertex) for vertex in vertices]
        visuals[link_name] = LinkVisual(
            link_name=link_name,
            origin=_matrix_translation(body_poses[link_name]),
            visual_center=_center(world_vertices),
            bbox=_bbox(world_vertices),
        )
    return visuals


def _collect_body_poses(
    body: ET.Element,
    parent_matrix: list[list[float]],
    body_poses: dict[str, list[list[float]]],
    body_nodes: dict[str, ET.Element],
) -> None:
    name = body.attrib.get("name", "")
    matrix = _matrix_multiply(
        parent_matrix,
        _xyz_rpy_matrix(
            _float_tuple(body.attrib.get("pos", "0 0 0")),
            _float_tuple(body.attrib.get("euler", "0 0 0")),
        ),
    )
    body_poses[name] = matrix
    body_nodes[name] = body
    for child in body.findall("body"):
        _collect_body_poses(child, matrix, body_poses, body_nodes)


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


def _urdf_link_poses(root: ET.Element) -> dict[str, list[list[float]]]:
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        children_by_parent.setdefault(parent.attrib["link"], []).append(
            {
                "child": child.attrib["link"],
                "xyz": _origin_tuple(origin, "xyz"),
                "rpy": _origin_tuple(origin, "rpy"),
            }
        )
    poses = {"base": _identity_matrix4()}
    _walk_urdf_link_poses("base", children_by_parent, poses)
    missing = [name for name in OFFICIAL_320_LINK_NAMES if name not in poses]
    if missing:
        raise ValueError("missing URDF link poses for: " + ", ".join(missing))
    return poses


def _walk_urdf_link_poses(
    parent: str,
    children_by_parent: dict[str, list[dict[str, Any]]],
    poses: dict[str, list[list[float]]],
) -> None:
    for joint in children_by_parent.get(parent, []):
        child = str(joint["child"])
        poses[child] = _matrix_multiply(
            poses[parent],
            _xyz_rpy_matrix(joint["xyz"], joint["rpy"]),
        )
        _walk_urdf_link_poses(child, children_by_parent, poses)


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


def _write_artifacts(
    report: VisualPoseReport,
    output_dir: Path,
    reference: dict[str, LinkVisual],
    generated: dict[str, LinkVisual],
) -> dict[str, str]:
    json_path = output_dir / "visual_pose_report.json"
    md_path = output_dir / "visual_pose_report.md"
    svg_path = output_dir / "visual_pose_gate.svg"
    json_path.write_text(
        json.dumps(_report_to_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    svg_path.write_text(_render_svg(report, reference, generated), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "svg": str(svg_path)}


def _render_markdown(report: VisualPoseReport) -> str:
    lines = [
        "# myCobot 320 Adaptive Visual Pose Gate",
        "",
        f"- status: `{report.status}`",
        f"- compared links: `{report.compared_link_count}`",
        f"- max link origin delta: `{report.max_link_origin_delta:.12g}`",
        f"- max visual center delta: `{report.max_visual_center_delta:.12g}`",
        "",
        "| link | status | link origin delta | visual center delta |",
        "|---|---|---:|---:|",
    ]
    for item in report.comparisons:
        lines.append(
            f"| `{item.link_name}` | `{item.status}` | "
            f"{item.link_origin_delta:.12g} | {item.visual_center_delta:.12g} |"
        )
    return "\n".join(lines) + "\n"


def _render_svg(
    report: VisualPoseReport,
    reference: dict[str, LinkVisual],
    generated: dict[str, LinkVisual],
) -> str:
    width = 1680
    height = 1800
    status_color = "#2e7d32" if report.status == "passed" else "#b3261e"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f8f5"/>',
        _svg_text(36, 54, "myCobot 320 adaptive visual pose gate", size=30, weight="700"),
        f'<rect x="36" y="76" width="140" height="34" rx="6" fill="{status_color}"/>',
        _svg_text(58, 99, report.status.upper(), size=17, fill="#ffffff"),
        _svg_text(
            204,
            99,
            (
                f"{report.passed_link_count}/{report.compared_link_count} links passed; "
                f"max origin/visual delta "
                f"{report.max_link_origin_delta:.3g}/{report.max_visual_center_delta:.3g}"
            ),
            size=17,
            fill="#4b4f52",
        ),
    ]
    views = [
        ("front x-z", (0, 2), False),
        ("side y-z", (1, 2), False),
        ("top x-y", (0, 1), False),
        ("wrist close x-y", (0, 1), True),
    ]
    y = 145
    for label, axes, gripper_only in views:
        parts.append(_svg_text(38, y - 16, label, size=19, weight="700"))
        parts.extend(
            _render_panel(
                38,
                y,
                780,
                360,
                "official URDF reference",
                reference,
                axes,
                gripper_only,
            )
        )
        parts.extend(
            _render_panel(
                862,
                y,
                780,
                360,
                "generated MuJoCo XML",
                generated,
                axes,
                gripper_only,
            )
        )
        y += 410
    parts.append("</svg>")
    return "\n".join(parts)


def _render_panel(
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    visuals: dict[str, LinkVisual],
    axes: tuple[int, int],
    gripper_only: bool,
) -> list[str]:
    names = [
        name
        for name in OFFICIAL_320_LINK_NAMES
        if not gripper_only or name.startswith("gripper")
    ]
    points = []
    for name in names:
        mins, maxs = visuals[name].bbox
        points.extend(_bbox_corners(mins, maxs))
        points.append(visuals[name].origin)
    bounds = _projected_bounds(points, axes)
    mapper = _make_mapper(bounds, axes, x + 34, y + 58, width - 68, height - 86)
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="#ffffff"/>',
        f'<rect x="{x}" y="{y}" width="{width}" height="38" rx="8" fill="#eef1ec"/>',
        _svg_text(x + 18, y + 25, title, size=15, weight="700", fill="#202124"),
    ]
    for parent, child in _tree_edges():
        if parent not in names or child not in names:
            continue
        x1, y1 = mapper(visuals[parent].origin)
        x2, y2 = mapper(visuals[child].origin)
        parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            'stroke="#6f777a" stroke-width="2"/>'
        )
    for name in names:
        visual = visuals[name]
        mins, maxs = visual.bbox
        corners = _bbox_corners(mins, maxs)
        xs, ys = zip(*(mapper(point) for point in corners), strict=True)
        rect_x = min(xs)
        rect_y = min(ys)
        rect_w = max(2.0, max(xs) - rect_x)
        rect_h = max(2.0, max(ys) - rect_y)
        color = "#73808a" if name in ARM_LINKS else "#b75f38" if "left" in name else "#3f73ad"
        if name == "gripper_base":
            color = "#6d6a58"
        parts.append(
            f'<rect x="{rect_x:.2f}" y="{rect_y:.2f}" width="{rect_w:.2f}" '
            f'height="{rect_h:.2f}" fill="{color}" opacity="0.24" stroke="{color}" '
            'stroke-width="1.2"/>'
        )
        ox, oy = mapper(visual.origin)
        parts.append(f'<circle cx="{ox:.2f}" cy="{oy:.2f}" r="4" fill="{color}"/>')
        if gripper_only or name in {"base", "link3", "link6", "gripper_base"}:
            parts.append(_svg_text(int(ox + 6), int(oy - 6), name, size=11, fill="#202124"))
    return parts


def _tree_edges() -> list[tuple[str, str]]:
    return [
        ("base", "link1"),
        ("link1", "link2"),
        ("link2", "link3"),
        ("link3", "link4"),
        ("link4", "link5"),
        ("link5", "link6"),
        ("link6", "gripper_base"),
        ("gripper_base", "gripper_left3"),
        ("gripper_base", "gripper_left2"),
        ("gripper_left3", "gripper_left1"),
        ("gripper_base", "gripper_right3"),
        ("gripper_base", "gripper_right2"),
        ("gripper_right3", "gripper_right1"),
    ]


def _projected_bounds(
    points: list[tuple[float, float, float]],
    axes: tuple[int, int],
) -> tuple[float, float, float, float]:
    xs = [point[axes[0]] for point in points]
    ys = [point[axes[1]] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(min_x, max_x):
        min_x -= 0.05
        max_x += 0.05
    if math.isclose(min_y, max_y):
        min_y -= 0.05
        max_y += 0.05
    return min_x, max_x, min_y, max_y


def _make_mapper(
    bounds: tuple[float, float, float, float],
    axes: tuple[int, int],
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
        px = offset_x + (point[axes[0]] - min_x) * scale
        py = offset_y + content_h - (point[axes[1]] - min_y) * scale
        return px, py

    return map_point


def _bbox_corners(
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    return [
        (x, y, z)
        for x in (mins[0], maxs[0])
        for y in (mins[1], maxs[1])
        for z in (mins[2], maxs[2])
    ]


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


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


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


def _report_to_jsonable(report: VisualPoseReport) -> dict[str, Any]:
    return {
        **asdict(report),
        "comparisons": [asdict(item) for item in report.comparisons],
    }


if __name__ == "__main__":
    main()
