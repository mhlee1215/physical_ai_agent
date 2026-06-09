#!/usr/bin/env python3
"""Validate oracle point projection and overlay rendering.

This script is intentionally lightweight: it only needs numpy and Pillow.
It validates the overlay utility without requiring ManiSkill, Vulkan, or
SmolVLA model loading.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from physical_ai_agent.perception.affordance_overlay import (
    build_center_overlay_from_image_path,
    build_oracle_affordance_overlay,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _save_plain_image(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def _marker_pixel_count(path: Path) -> int:
    arr = np.asarray(Image.open(path).convert("RGB"))
    green = (arr[:, :, 1] > 180) & (arr[:, :, 0] < 80) & (arr[:, :, 2] < 120)
    return int(green.sum())


def _project_xy_for_synthetic(rgb: np.ndarray, obj_xyz: tuple[float, float, float]) -> tuple[int, int]:
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    fx = fy = 120.0
    cx = width / 2.0
    cy = height / 2.0
    x, y, z = obj_xyz
    return int(round((fx * x / z) + cx)), int(round((fy * y / z) + cy))


def _make_projection_scene(
    width: int,
    height: int,
    obj_xyz: tuple[float, float, float],
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
) -> np.ndarray:
    x_grad = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_grad = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    blend = np.clip((x_grad + y_grad) / 2.0, 0.0, 1.0)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for channel in range(3):
        rgb[:, :, channel] = (
            color_a[channel] * (1.0 - blend) + color_b[channel] * blend
        ).astype(np.uint8)

    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    px, py = _project_xy_for_synthetic(rgb, obj_xyz)
    radius = max(10, min(width, height) // 14)
    draw.ellipse(
        (px - radius, py - radius, px + radius, py + radius),
        fill=(220, 90, 45),
        outline=(80, 30, 20),
        width=3,
    )
    draw.ellipse(
        (px - radius // 3, py - radius // 3, px + radius // 3, py + radius // 3),
        fill=(255, 220, 60),
    )
    return np.asarray(image, dtype=np.uint8)


def _synthetic_obs(
    rgb: np.ndarray,
    obj_xyz: tuple[float, float, float] | None,
    pose_style: str = "flat",
    camera_param_location: str = "sensor_param",
) -> dict[str, Any]:
    intrinsic_cv = np.array(
        [
            [120.0, 0.0, rgb.shape[1] / 2.0],
            [0.0, 120.0, rgb.shape[0] / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    extrinsic_cv = np.eye(4, dtype=np.float32)
    obs: dict[str, Any] = {
        "sensor_data": {
            "base_camera": {
                "rgb": rgb,
            }
        },
        "sensor_param": {
            "base_camera": {
                "intrinsic_cv": intrinsic_cv,
                "extrinsic_cv": extrinsic_cv,
            }
        },
    }
    if camera_param_location == "sensor_data":
        obs["sensor_data"]["base_camera"]["intrinsic_cv"] = intrinsic_cv
        obs["sensor_data"]["base_camera"]["extrinsic_cv"] = extrinsic_cv
        obs.pop("sensor_param")
    if obj_xyz is not None:
        if pose_style == "dict_p":
            obs["obj_pose"] = {"p": np.array(obj_xyz, dtype=np.float32)}
        elif pose_style == "dict_position":
            obs["obj_pose"] = {"position": np.array(obj_xyz, dtype=np.float32)}
        else:
            obs["obj_pose"] = np.array(obj_xyz, dtype=np.float32)
    return obs


def validate_projection(output_dir: Path) -> list[dict[str, Any]]:
    projected_specs = [
        ("projected_center", 220, 160, (0.00, 0.00, 1.00), (32, 45, 58), (70, 105, 120)),
        ("projected_right_up", 220, 160, (0.22, -0.12, 1.00), (28, 36, 60), (90, 90, 110)),
        ("projected_left_down", 220, 160, (-0.24, 0.18, 1.00), (52, 38, 28), (105, 82, 65)),
        ("projected_far_center", 260, 180, (0.00, 0.00, 1.45), (20, 45, 42), (70, 100, 92)),
        ("projected_near_right", 260, 180, (0.18, 0.08, 0.82), (45, 32, 60), (105, 72, 120)),
        ("projected_wide_left", 320, 128, (-0.38, -0.05, 1.25), (35, 45, 22), (88, 105, 50)),
        ("projected_wide_right", 320, 128, (0.38, 0.06, 1.25), (22, 36, 48), (70, 88, 125)),
        ("projected_tall_upper", 144, 240, (0.06, -0.46, 1.25), (42, 30, 45), (88, 70, 100)),
        ("projected_tall_lower", 144, 240, (-0.05, 0.46, 1.25), (30, 45, 42), (82, 100, 90)),
        ("projected_small_image", 96, 96, (0.08, -0.08, 1.00), (44, 36, 28), (110, 88, 62)),
        ("projected_large_image", 384, 256, (-0.30, 0.16, 1.20), (24, 38, 58), (80, 108, 138)),
        ("projected_corner_safe", 260, 180, (0.55, -0.32, 1.20), (48, 32, 25), (120, 70, 55)),
    ]
    cases = [
        (
            name,
            _make_projection_scene(width, height, xyz, color_a, color_b),
            xyz,
        )
        for name, width, height, xyz, color_a, color_b in projected_specs
    ]
    cases.append(("fallback_no_pose", np.full((160, 220, 3), 96, dtype=np.uint8), None))
    results: list[dict[str, Any]] = []
    for name, rgb, xyz in cases:
        out_path = output_dir / "projection" / f"{name}.png"
        _, overlay = build_oracle_affordance_overlay(
            _synthetic_obs(rgb, xyz),
            output_path=out_path,
            label=name,
        )
        marker_pixels = _marker_pixel_count(out_path)
        passed = out_path.exists() and marker_pixels > 0
        if name.startswith("projected"):
            passed = passed and overlay.mode == "projected_object_pose"
        else:
            passed = passed and "fallback" in str(overlay.mode)
        results.append(
            {
                "section": "projection",
                "case": name,
                "passed": bool(passed),
                "output": str(out_path),
                "marker_pixels": marker_pixels,
                "metadata": _jsonable(overlay.metadata()),
            }
        )
    return results


def validate_rendering(output_dir: Path) -> list[dict[str, Any]]:
    base_dir = output_dir / "rendering" / "inputs"
    cases = {
        "dark_square": np.zeros((192, 192, 3), dtype=np.uint8),
        "bright_square": np.full((192, 192, 3), 235, dtype=np.uint8),
        "small": np.full((64, 64, 3), 120, dtype=np.uint8),
        "wide_gradient": np.dstack(
            [
                np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (120, 1)),
                np.full((120, 320), 80, dtype=np.uint8),
                np.tile(np.linspace(255, 0, 320, dtype=np.uint8), (120, 1)),
            ]
        ),
    }
    results: list[dict[str, Any]] = []
    for name, rgb in cases.items():
        input_path = base_dir / f"{name}.png"
        output_path = output_dir / "rendering" / f"{name}_overlay.png"
        _save_plain_image(input_path, rgb)
        overlay = build_center_overlay_from_image_path(
            input_path,
            output_path,
            label=name,
        )
        marker_pixels = _marker_pixel_count(output_path)
        results.append(
            {
                "section": "rendering",
                "case": name,
                "passed": bool(output_path.exists() and marker_pixels > 0),
                "input": str(input_path),
                "output": str(output_path),
                "marker_pixels": marker_pixels,
                "metadata": _jsonable(overlay.metadata()),
            }
        )
    return results


def validate_projection_trajectory(output_dir: Path, frames: int = 20) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    trajectory_dir = output_dir / "projection_trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    saved_frames = []
    for idx in range(frames):
        t = idx / max(1, frames - 1)
        x = -0.42 + 0.84 * t
        y = 0.22 * np.sin(t * np.pi * 2.0)
        z = 1.05 + 0.18 * np.cos(t * np.pi)
        xyz = (float(x), float(y), float(z))
        rgb = _make_projection_scene(
            320,
            200,
            xyz,
            (28, 42, 48),
            (92, 76, 54),
        )
        out_path = trajectory_dir / f"trajectory_{idx:03d}.png"
        expected_xy = _project_xy_for_synthetic(rgb, xyz)
        _, overlay = build_oracle_affordance_overlay(
            _synthetic_obs(rgb, xyz),
            output_path=out_path,
            label=f"traj {idx:02d}",
        )
        actual_xy = tuple(int(v) for v in overlay.point_xy)
        error_px = float(np.linalg.norm(np.asarray(actual_xy) - np.asarray(expected_xy)))
        marker_pixels = _marker_pixel_count(out_path)
        passed = (
            out_path.exists()
            and marker_pixels > 0
            and overlay.mode == "projected_object_pose"
            and error_px <= 1.0
        )
        saved_frames.append(out_path)
        results.append(
            {
                "section": "projection_trajectory",
                "case": f"trajectory_{idx:03d}",
                "passed": bool(passed),
                "output": str(out_path),
                "marker_pixels": marker_pixels,
                "metadata": {
                    "expected_xy": list(expected_xy),
                    "actual_xy": list(actual_xy),
                    "error_px": error_px,
                    "object_xyz": list(xyz),
                    **_jsonable(overlay.metadata()),
                },
            }
        )
    _write_gif(saved_frames, output_dir / "projection_trajectory.gif")
    return results


def validate_dict_pose_trajectory(output_dir: Path, frames: int = 20) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    trajectory_dir = output_dir / "dict_pose_trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    saved_frames = []
    for idx in range(frames):
        t = idx / max(1, frames - 1)
        x = 0.34 * np.sin(t * np.pi * 2.0)
        y = -0.28 + 0.56 * t
        z = 1.18 + 0.12 * np.sin(t * np.pi)
        xyz = (float(x), float(y), float(z))
        rgb = _make_projection_scene(
            300,
            220,
            xyz,
            (36, 35, 50),
            (92, 82, 62),
        )
        out_path = trajectory_dir / f"dict_pose_{idx:03d}.png"
        expected_xy = _project_xy_for_synthetic(rgb, xyz)
        pose_style = "dict_p" if idx % 2 == 0 else "dict_position"
        _, overlay = build_oracle_affordance_overlay(
            _synthetic_obs(rgb, xyz, pose_style=pose_style),
            output_path=out_path,
            label=f"dict pose {idx:02d}",
        )
        actual_xy = tuple(int(v) for v in overlay.point_xy)
        error_px = float(np.linalg.norm(np.asarray(actual_xy) - np.asarray(expected_xy)))
        marker_pixels = _marker_pixel_count(out_path)
        passed = (
            out_path.exists()
            and marker_pixels > 0
            and overlay.mode == "projected_object_pose"
            and error_px <= 1.0
        )
        saved_frames.append(out_path)
        results.append(
            {
                "section": "dict_pose_trajectory",
                "case": f"dict_pose_{idx:03d}",
                "passed": bool(passed),
                "output": str(out_path),
                "marker_pixels": marker_pixels,
                "metadata": {
                    "pose_style": pose_style,
                    "expected_xy": list(expected_xy),
                    "actual_xy": list(actual_xy),
                    "error_px": error_px,
                    "object_xyz": list(xyz),
                    **_jsonable(overlay.metadata()),
                },
            }
        )
    _write_gif(saved_frames, output_dir / "dict_pose_trajectory.gif")
    return results


def validate_sensor_data_camera_trajectory(output_dir: Path, frames: int = 20) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    trajectory_dir = output_dir / "sensor_data_camera_trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    saved_frames = []
    for idx in range(frames):
        t = idx / max(1, frames - 1)
        x = -0.36 + 0.72 * t
        y = 0.24 * np.cos(t * np.pi * 2.0)
        z = 1.10 + 0.16 * np.sin(t * np.pi)
        xyz = (float(x), float(y), float(z))
        rgb = _make_projection_scene(
            300,
            180,
            xyz,
            (30, 44, 58),
            (92, 74, 48),
        )
        out_path = trajectory_dir / f"sensor_data_camera_{idx:03d}.png"
        expected_xy = _project_xy_for_synthetic(rgb, xyz)
        _, overlay = build_oracle_affordance_overlay(
            _synthetic_obs(
                rgb,
                xyz,
                pose_style="dict_p",
                camera_param_location="sensor_data",
            ),
            output_path=out_path,
            label=f"sensor cam {idx:02d}",
        )
        actual_xy = tuple(int(v) for v in overlay.point_xy)
        error_px = float(np.linalg.norm(np.asarray(actual_xy) - np.asarray(expected_xy)))
        marker_pixels = _marker_pixel_count(out_path)
        passed = (
            out_path.exists()
            and marker_pixels > 0
            and overlay.mode == "projected_object_pose"
            and error_px <= 1.0
        )
        saved_frames.append(out_path)
        results.append(
            {
                "section": "sensor_data_camera_trajectory",
                "case": f"sensor_data_camera_{idx:03d}",
                "passed": bool(passed),
                "output": str(out_path),
                "marker_pixels": marker_pixels,
                "metadata": {
                    "camera_param_location": "sensor_data",
                    "pose_style": "dict_p",
                    "expected_xy": list(expected_xy),
                    "actual_xy": list(actual_xy),
                    "error_px": error_px,
                    "object_xyz": list(xyz),
                    **_jsonable(overlay.metadata()),
                },
            }
        )
    _write_gif(saved_frames, output_dir / "sensor_data_camera_trajectory.gif")
    return results


def _multi_camera_obs(
    primary_rgb: np.ndarray,
    secondary_rgb: np.ndarray,
    obj_xyz: tuple[float, float, float],
    include_preferred: bool,
) -> dict[str, Any]:
    primary_intrinsic = np.array(
        [
            [120.0, 0.0, primary_rgb.shape[1] / 2.0],
            [0.0, 120.0, primary_rgb.shape[0] / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    secondary_intrinsic = np.array(
        [
            [96.0, 0.0, secondary_rgb.shape[1] / 2.0],
            [0.0, 96.0, secondary_rgb.shape[0] / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    sensor_data: dict[str, Any] = {
        "wrist_camera": {"rgb": secondary_rgb},
    }
    sensor_param: dict[str, Any] = {
        "wrist_camera": {
            "intrinsic_cv": secondary_intrinsic,
            "extrinsic_cv": np.eye(4, dtype=np.float32),
        }
    }
    if include_preferred:
        sensor_data["base_camera"] = {"rgb": primary_rgb}
        sensor_param["base_camera"] = {
            "intrinsic_cv": primary_intrinsic,
            "extrinsic_cv": np.eye(4, dtype=np.float32),
        }
    else:
        sensor_data["aux_camera"] = {"rgb": primary_rgb}
        sensor_param["aux_camera"] = {
            "intrinsic_cv": primary_intrinsic,
            "extrinsic_cv": np.eye(4, dtype=np.float32),
        }
    return {
        "sensor_data": sensor_data,
        "sensor_param": sensor_param,
        "obj_pose": {"p": np.array(obj_xyz, dtype=np.float32)},
    }


def validate_multi_camera_trajectory(output_dir: Path, frames: int = 20) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    trajectory_dir = output_dir / "multi_camera_trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    saved_frames = []
    for idx in range(frames):
        t = idx / max(1, frames - 1)
        x = -0.28 + 0.56 * t
        y = 0.18 * np.sin(t * np.pi * 2.0)
        z = 1.12 + 0.10 * np.cos(t * np.pi * 2.0)
        xyz = (float(x), float(y), float(z))
        include_preferred = idx % 2 == 0
        primary_rgb = _make_projection_scene(
            300,
            190,
            xyz,
            (32, 42, 54),
            (88, 76, 58),
        )
        secondary_rgb = _make_projection_scene(
            180,
            140,
            xyz,
            (48, 34, 38),
            (100, 70, 72),
        )
        obs = _multi_camera_obs(primary_rgb, secondary_rgb, xyz, include_preferred=include_preferred)
        expected_camera = "base_camera" if include_preferred else "aux_camera"
        expected_rgb = primary_rgb
        expected_xy = _project_xy_for_synthetic(expected_rgb, xyz)
        out_path = trajectory_dir / f"multi_camera_{idx:03d}.png"
        _, overlay = build_oracle_affordance_overlay(
            obs,
            output_path=out_path,
            preferred_camera="base_camera",
            label=f"multi cam {idx:02d}",
        )
        actual_xy = tuple(int(v) for v in overlay.point_xy)
        error_px = float(np.linalg.norm(np.asarray(actual_xy) - np.asarray(expected_xy)))
        marker_pixels = _marker_pixel_count(out_path)
        passed = (
            out_path.exists()
            and marker_pixels > 0
            and overlay.mode == "projected_object_pose"
            and overlay.camera_name == expected_camera
            and error_px <= 1.0
        )
        saved_frames.append(out_path)
        results.append(
            {
                "section": "multi_camera_trajectory",
                "case": f"multi_camera_{idx:03d}",
                "passed": bool(passed),
                "output": str(out_path),
                "marker_pixels": marker_pixels,
                "metadata": {
                    "include_preferred": include_preferred,
                    "expected_camera": expected_camera,
                    "actual_camera": overlay.camera_name,
                    "expected_xy": list(expected_xy),
                    "actual_xy": list(actual_xy),
                    "error_px": error_px,
                    "object_xyz": list(xyz),
                    **_jsonable(overlay.metadata()),
                },
            }
        )
    _write_gif(saved_frames, output_dir / "multi_camera_trajectory.gif")
    return results


def validate_projection_edge_cases(output_dir: Path, frames: int = 20) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    edge_dir = output_dir / "projection_edge_cases"
    edge_dir.mkdir(parents=True, exist_ok=True)
    saved_frames = []
    mutations = [
        "behind_camera",
        "far_out_left",
        "far_out_right",
        "far_out_top",
        "far_out_bottom",
        "missing_intrinsic",
        "missing_extrinsic",
        "bad_intrinsic_shape",
        "bad_extrinsic_shape",
        "invalid_pose",
    ]
    for idx in range(frames):
        mutation = mutations[idx % len(mutations)]
        rgb = _make_projection_scene(
            260,
            180,
            (0.0, 0.0, 1.0),
            (42, 42, 42),
            (92, 82, 62),
        )
        xyz = (0.0, 0.0, 1.0)
        if mutation == "behind_camera":
            xyz = (0.0, 0.0, -1.0)
        elif mutation == "far_out_left":
            xyz = (-4.0, 0.0, 1.0)
        elif mutation == "far_out_right":
            xyz = (4.0, 0.0, 1.0)
        elif mutation == "far_out_top":
            xyz = (0.0, -4.0, 1.0)
        elif mutation == "far_out_bottom":
            xyz = (0.0, 4.0, 1.0)
        obs = _synthetic_obs(rgb, xyz, pose_style="dict_p")
        if mutation == "missing_intrinsic":
            obs["sensor_param"]["base_camera"].pop("intrinsic_cv")
        elif mutation == "missing_extrinsic":
            obs["sensor_param"]["base_camera"].pop("extrinsic_cv")
        elif mutation == "bad_intrinsic_shape":
            obs["sensor_param"]["base_camera"]["intrinsic_cv"] = np.eye(2, dtype=np.float32)
        elif mutation == "bad_extrinsic_shape":
            obs["sensor_param"]["base_camera"]["extrinsic_cv"] = np.eye(3, dtype=np.float32)
        elif mutation == "invalid_pose":
            obs["obj_pose"] = {"p": "not-a-pose"}
        out_path = edge_dir / f"edge_{idx:03d}_{mutation}.png"
        _, overlay = build_oracle_affordance_overlay(
            obs,
            output_path=out_path,
            label=f"edge {mutation}",
        )
        marker_pixels = _marker_pixel_count(out_path)
        expected_xy = [rgb.shape[1] // 2, rgb.shape[0] // 2]
        passed = (
            out_path.exists()
            and marker_pixels > 0
            and overlay.mode == "image_center_fallback"
            and overlay.point_xy == expected_xy
        )
        saved_frames.append(out_path)
        results.append(
            {
                "section": "projection_edge_cases",
                "case": f"edge_{idx:03d}_{mutation}",
                "passed": bool(passed),
                "output": str(out_path),
                "marker_pixels": marker_pixels,
                "metadata": {
                    "mutation": mutation,
                    "expected_mode": "image_center_fallback",
                    "expected_xy": expected_xy,
                    **_jsonable(overlay.metadata()),
                },
            }
        )
    _write_gif(saved_frames, output_dir / "projection_edge_cases.gif")
    return results


def validate_sim_frames(
    output_dir: Path,
    sim_frame_root: Path | None,
    max_frames: int,
    skip_missing: bool = False,
) -> list[dict[str, Any]]:
    if sim_frame_root is None or not sim_frame_root.exists():
        if skip_missing:
            return []
        return [
            {
                "section": "sim_frame_overlay",
                "case": "sim_frame_root_missing",
                "passed": False,
                "output": None,
                "metadata": {"sim_frame_root": str(sim_frame_root) if sim_frame_root else None},
            }
        ]
    pngs = sorted(sim_frame_root.rglob("*.png"))[:max_frames]
    if not pngs:
        return [
            {
                "section": "sim_frame_overlay",
                "case": "no_png_frames",
                "passed": False,
                "output": None,
                "metadata": {"sim_frame_root": str(sim_frame_root)},
            }
        ]
    results: list[dict[str, Any]] = []
    for idx, input_path in enumerate(pngs):
        output_path = output_dir / "sim_frames" / f"{idx:03d}_{input_path.stem}_overlay.png"
        overlay = build_center_overlay_from_image_path(
            input_path,
            output_path,
            label="oracle point",
        )
        marker_pixels = _marker_pixel_count(output_path)
        results.append(
            {
                "section": "sim_frame_overlay",
                "case": input_path.name,
                "passed": bool(output_path.exists() and marker_pixels > 0),
                "input": str(input_path),
                "output": str(output_path),
                "marker_pixels": marker_pixels,
                "metadata": _jsonable(overlay.metadata()),
            }
        )
    return results


def write_reports(output_dir: Path, results: list[dict[str, Any]]) -> None:
    report_json = output_dir / "validation_report.json"
    report_md = output_dir / "validation_report.md"
    report_html = output_dir / "validation_report.html"
    passed = sum(1 for row in results if row.get("passed"))
    total = len(results)
    report_json.write_text(json.dumps({"passed": passed, "total": total, "results": results}, indent=2), encoding="utf-8")

    lines = [
        "# Oracle Point Overlay Validation Report",
        "",
        f"- Passed: {passed}/{total}",
        f"- Output directory: `{output_dir}`",
        "",
        "## Cases",
        "",
    ]
    for row in results:
        status = "PASS" if row.get("passed") else "FAIL"
        lines.extend(
            [
                f"### {status} - {row.get('section')} / {row.get('case')}",
                "",
                f"- Output: `{row.get('output')}`",
                f"- Marker pixels: `{row.get('marker_pixels', 'n/a')}`",
                f"- Metadata: `{json.dumps(row.get('metadata', {}), sort_keys=True)}`",
                "",
            ]
        )
    report_md.write_text("\n".join(lines), encoding="utf-8")
    _write_contact_sheets(output_dir, results)
    _write_html_report(report_html, output_dir, results, passed, total)


def _write_gif(image_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths if path.exists()]
    if not images:
        return
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=180,
        loop=0,
    )


def _write_contact_sheets(output_dir: Path, results: list[dict[str, Any]]) -> None:
    for section in sorted({str(row.get("section")) for row in results}):
        image_paths = [
            Path(str(row.get("output")))
            for row in results
            if row.get("section") == section and row.get("output")
        ]
        image_paths = [path for path in image_paths if path.exists()]
        if not image_paths:
            continue
        thumbs = []
        for path in image_paths[:24]:
            image = Image.open(path).convert("RGB")
            image.thumbnail((180, 140))
            canvas = Image.new("RGB", (200, 170), (245, 245, 240))
            canvas.paste(image, ((200 - image.width) // 2, 8))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 148), path.stem[:28], fill=(20, 20, 20))
            thumbs.append(canvas)
        cols = 4
        rows = int(np.ceil(len(thumbs) / cols))
        sheet = Image.new("RGB", (cols * 200, rows * 170), (232, 232, 225))
        for idx, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((idx % cols) * 200, (idx // cols) * 170))
        sheet.save(output_dir / f"contact_sheet_{section}.png")


def _relative_image_path(report_path: Path, image_path: Path) -> str:
    try:
        return image_path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return image_path.as_posix()


def _write_html_report(
    report_path: Path,
    output_dir: Path,
    results: list[dict[str, Any]],
    passed: int,
    total: int,
) -> None:
    sections = sorted({str(row.get("section")) for row in results})
    section_blocks = []
    for section in sections:
        rows = [row for row in results if row.get("section") == section]
        image_cards = []
        for row in rows:
            output = row.get("output")
            if not output:
                continue
            image_path = Path(str(output))
            if not image_path.exists():
                continue
            metadata = row.get("metadata", {})
            image_cards.append(
                f"""
                <figure class="card {'pass' if row.get('passed') else 'fail'}">
                  <img src="{html.escape(_relative_image_path(report_path, image_path))}" alt="{html.escape(str(row.get('case')))}">
                  <figcaption>
                    <strong>{html.escape(str(row.get('case')))}</strong>
                    <span>{'PASS' if row.get('passed') else 'FAIL'} / marker pixels: {html.escape(str(row.get('marker_pixels', 'n/a')))}</span>
                    <code>{html.escape(json.dumps(metadata, sort_keys=True))}</code>
                  </figcaption>
                </figure>
                """
            )
        sheet_path = output_dir / f"contact_sheet_{section}.png"
        sheet_html = ""
        if sheet_path.exists():
            sheet_html = f"""
            <h3>Contact sheet</h3>
            <img class="sheet" src="{html.escape(_relative_image_path(report_path, sheet_path))}" alt="{html.escape(section)} contact sheet">
            """
        section_blocks.append(
            f"""
            <section>
              <h2>{html.escape(section)}</h2>
              <p>{sum(1 for row in rows if row.get('passed'))}/{len(rows)} pass</p>
              {sheet_html}
              <div class="grid">{''.join(image_cards)}</div>
            </section>
            """
        )
    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Oracle Point Overlay Validation</title>
  <style>
    :root {{
      --ink: #171711;
      --paper: #f5f1e8;
      --panel: #fffaf0;
      --line: #d7cab4;
      --green: #00d968;
      --bad: #c7422f;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 18% 8%, #fff4cf, transparent 28%),
        linear-gradient(135deg, #ede3cf, #f8f4eb 44%, #e6eee8);
      color: var(--ink);
      font-family: Charter, Georgia, serif;
    }}
    header {{
      padding: 42px 48px 24px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(32px, 5vw, 64px);
      letter-spacing: -0.045em;
    }}
    .summary {{
      display: inline-flex;
      gap: 18px;
      padding: 12px 16px;
      background: rgba(255, 250, 240, 0.78);
      border: 1px solid var(--line);
      border-radius: 999px;
      font-family: Avenir Next, Helvetica, sans-serif;
    }}
    main {{
      padding: 28px 48px 60px;
    }}
    section {{
      margin: 0 0 42px;
      padding: 24px;
      background: rgba(255, 250, 240, 0.78);
      border: 1px solid var(--line);
      border-radius: 26px;
      box-shadow: 0 18px 48px rgba(60, 42, 20, 0.08);
    }}
    h2 {{
      margin: 0;
      font-size: 30px;
      letter-spacing: -0.03em;
    }}
    .sheet {{
      width: min(100%, 900px);
      border-radius: 18px;
      border: 1px solid var(--line);
      background: white;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 18px;
      margin-top: 20px;
    }}
    .card {{
      margin: 0;
      padding: 12px;
      background: white;
      border: 2px solid var(--green);
      border-radius: 18px;
    }}
    .card.fail {{
      border-color: var(--bad);
    }}
    .card img {{
      width: 100%;
      border-radius: 12px;
      image-rendering: auto;
      background: #2b2b2b;
    }}
    figcaption {{
      display: grid;
      gap: 6px;
      margin-top: 10px;
      font-family: Avenir Next, Helvetica, sans-serif;
      font-size: 13px;
    }}
    code {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      padding: 8px;
      border-radius: 10px;
      background: #f2eee5;
      font-size: 11px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Oracle Point Overlay Validation</h1>
    <div class="summary">
      <span><strong>{passed}/{total}</strong> pass</span>
      <span>projection + rendering + sim-frame overlay</span>
    </div>
  </header>
  <main>
    {''.join(section_blocks)}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sim-frame-root")
    parser.add_argument("--max-sim-frames", type=int, default=20)
    parser.add_argument("--skip-missing-sim-frames", action="store_true")
    parser.add_argument("--trajectory-frames", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_frame_root = Path(args.sim_frame_root) if args.sim_frame_root else None

    results = []
    results.extend(validate_projection(output_dir))
    results.extend(validate_projection_trajectory(output_dir, frames=args.trajectory_frames))
    results.extend(validate_dict_pose_trajectory(output_dir, frames=args.trajectory_frames))
    results.extend(validate_sensor_data_camera_trajectory(output_dir, frames=args.trajectory_frames))
    results.extend(validate_multi_camera_trajectory(output_dir, frames=args.trajectory_frames))
    results.extend(validate_projection_edge_cases(output_dir, frames=args.trajectory_frames))
    results.extend(validate_rendering(output_dir))
    results.extend(
        validate_sim_frames(
            output_dir,
            sim_frame_root,
            args.max_sim_frames,
            skip_missing=args.skip_missing_sim_frames,
        )
    )
    write_reports(output_dir, results)

    failed = [row for row in results if not row.get("passed")]
    print(json.dumps({"passed": len(results) - len(failed), "total": len(results), "failed": failed[:5]}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
