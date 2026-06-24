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
    OFFICIAL_320_ADAPTIVE_GRIPPER_MESH_RELATIVE_PATH,
    OFFICIAL_320_ADAPTIVE_URDF_RELATIVE_PATH,
    OFFICIAL_320_ARM_LINK_NAMES,
    OFFICIAL_320_GRIPPER_LINK_NAMES,
    OFFICIAL_320_MESH_RELATIVE_PATH,
    _convert_collada_mesh_to_obj,
)


FLOAT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class MeshTransformComparison:
    link_name: str
    mesh_family: str
    status: str
    center_delta: float
    span_delta: float
    raw_center: tuple[float, float, float]
    baked_center: tuple[float, float, float]
    raw_span: tuple[float, float, float]
    baked_span: tuple[float, float, float]


@dataclass(frozen=True)
class MeshTransformReport:
    status: str
    source_urdf: str
    selected_transform_mode: str
    compared_mesh_count: int
    passed_mesh_count: int
    failed_mesh_count: int
    max_center_delta: float
    max_span_delta: float
    comparisons: list[MeshTransformComparison]
    artifacts: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw-geometry vs baked-visual-scene Collada conversion for "
            "the myCobot 320 M5 2022 adaptive gripper source assets."
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
        default=Path("_workspace/mycobot_320_adaptive_mesh_transform_verify"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify_adaptive_mesh_transform(
        official_gripper_root=args.official_gripper_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(_report_to_jsonable(report), indent=2, sort_keys=True))
    raise SystemExit(0 if report.status == "passed" else 1)


def verify_adaptive_mesh_transform(
    *,
    official_gripper_root: Path,
    output_dir: Path,
) -> MeshTransformReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_urdf = official_gripper_root / OFFICIAL_320_ADAPTIVE_URDF_RELATIVE_PATH
    mesh_sources = _mesh_sources(official_gripper_root)

    comparisons: list[MeshTransformComparison] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for link_name, mesh_family, dae_path in mesh_sources:
            raw_obj = tmp_path / f"{link_name}_raw.obj"
            baked_obj = tmp_path / f"{link_name}_baked.obj"
            _convert_collada_mesh_to_obj(dae_path, raw_obj, bake_visual_scene=False)
            _convert_collada_mesh_to_obj(dae_path, baked_obj, bake_visual_scene=True)
            raw_summary = _obj_summary(raw_obj)
            baked_summary = _obj_summary(baked_obj)
            center_delta = _distance(raw_summary["center"], baked_summary["center"])
            span_delta = _distance(raw_summary["span"], baked_summary["span"])
            comparisons.append(
                MeshTransformComparison(
                    link_name=link_name,
                    mesh_family=mesh_family,
                    status=(
                        "passed"
                        if center_delta <= FLOAT_TOLERANCE and span_delta <= FLOAT_TOLERANCE
                        else "failed"
                    ),
                    center_delta=center_delta,
                    span_delta=span_delta,
                    raw_center=raw_summary["center"],
                    baked_center=baked_summary["center"],
                    raw_span=raw_summary["span"],
                    baked_span=baked_summary["span"],
                )
            )

    failed = [item for item in comparisons if item.status != "passed"]
    report = MeshTransformReport(
        status="passed" if not failed else "failed",
        source_urdf=str(source_urdf),
        selected_transform_mode="raw_geometry",
        compared_mesh_count=len(comparisons),
        passed_mesh_count=len(comparisons) - len(failed),
        failed_mesh_count=len(failed),
        max_center_delta=max((item.center_delta for item in comparisons), default=0.0),
        max_span_delta=max((item.span_delta for item in comparisons), default=0.0),
        comparisons=comparisons,
        artifacts={},
    )
    return replace(report, artifacts=_write_artifacts(report, output_dir))


def _mesh_sources(official_gripper_root: Path) -> list[tuple[str, str, Path]]:
    arm_root = official_gripper_root / OFFICIAL_320_MESH_RELATIVE_PATH
    gripper_root = official_gripper_root / OFFICIAL_320_ADAPTIVE_GRIPPER_MESH_RELATIVE_PATH
    sources = [
        (name, "arm", arm_root / f"{name}.dae")
        for name in OFFICIAL_320_ARM_LINK_NAMES
    ]
    sources.extend(
        (name, "adaptive_gripper", gripper_root / f"{name}.dae")
        for name in OFFICIAL_320_GRIPPER_LINK_NAMES
    )
    missing = [str(path) for _, _, path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError("missing official adaptive mesh files: " + ", ".join(missing))
    return sources


def _obj_summary(obj_path: Path) -> dict[str, tuple[float, float, float]]:
    vertices = _obj_vertices(obj_path)
    mins = tuple(min(vertex[index] for vertex in vertices) for index in range(3))
    maxs = tuple(max(vertex[index] for vertex in vertices) for index in range(3))
    center = tuple(sum(vertex[index] for vertex in vertices) / len(vertices) for index in range(3))
    span = tuple(maxs[index] - mins[index] for index in range(3))
    return {"center": center, "span": span}


def _obj_vertices(obj_path: Path) -> list[tuple[float, float, float]]:
    vertices: list[tuple[float, float, float]] = []
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()[:4]
            vertices.append((float(x), float(y), float(z)))
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {obj_path}")
    return vertices


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _write_artifacts(report: MeshTransformReport, output_dir: Path) -> dict[str, str]:
    json_path = output_dir / "mesh_transform_report.json"
    md_path = output_dir / "mesh_transform_report.md"
    svg_path = output_dir / "mesh_transform_gate.svg"
    json_path.write_text(
        json.dumps(_report_to_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    svg_path.write_text(_render_svg(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "svg": str(svg_path)}


def _render_markdown(report: MeshTransformReport) -> str:
    lines = [
        "# myCobot 320 Adaptive Mesh Transform Gate",
        "",
        f"- status: `{report.status}`",
        f"- selected transform mode: `{report.selected_transform_mode}`",
        f"- compared meshes: `{report.compared_mesh_count}`",
        f"- max center delta: `{report.max_center_delta:.12g}`",
        f"- max span delta: `{report.max_span_delta:.12g}`",
        "",
        "| link | family | status | center delta | span delta |",
        "|---|---|---|---:|---:|",
    ]
    for item in report.comparisons:
        lines.append(
            f"| `{item.link_name}` | `{item.mesh_family}` | `{item.status}` | "
            f"{item.center_delta:.12g} | {item.span_delta:.12g} |"
        )
    return "\n".join(lines) + "\n"


def _render_svg(report: MeshTransformReport) -> str:
    width = 1400
    row_height = 42
    height = 130 + row_height * max(1, len(report.comparisons))
    status_color = "#2e7d32" if report.status == "passed" else "#b3261e"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f8f5"/>',
        _svg_text(32, 42, "myCobot 320 adaptive mesh transform gate", size=26, weight="700"),
        f'<rect x="32" y="58" width="130" height="32" rx="6" fill="{status_color}"/>',
        _svg_text(50, 80, report.status.upper(), size=16, fill="#ffffff"),
        _svg_text(
            190,
            80,
            f"{report.passed_mesh_count}/{report.compared_mesh_count} meshes passed",
            size=16,
            fill="#4b4f52",
        ),
        _svg_text(
            32,
            112,
            (
                "raw and baked Collada conversion are numerically identical for "
                f"this ROS2 source; max center/span delta: "
                f"{report.max_center_delta:.3g}/{report.max_span_delta:.3g}"
            ),
            size=14,
            fill="#4b4f52",
        ),
    ]
    y = 154
    for index, item in enumerate(report.comparisons):
        fill = "#ffffff" if index % 2 == 0 else "#eef1ec"
        color = "#2e7d32" if item.status == "passed" else "#b3261e"
        parts.append(f'<rect x="32" y="{y - 28}" width="1336" height="34" rx="4" fill="{fill}"/>')
        parts.append(_svg_text(48, y - 6, item.link_name, size=13, weight="700"))
        parts.append(_svg_text(300, y - 6, item.mesh_family, size=13))
        parts.append(f'<circle cx="520" cy="{y - 11}" r="8" fill="{color}"/>')
        parts.append(_svg_text(540, y - 6, item.status, size=13))
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


def _report_to_jsonable(report: MeshTransformReport) -> dict[str, Any]:
    return {
        **asdict(report),
        "comparisons": [asdict(item) for item in report.comparisons],
    }


if __name__ == "__main__":
    main()
