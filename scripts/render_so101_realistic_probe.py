#!/usr/bin/env python3
"""Render a quick SO101 visual-realism comparison from the current MuJoCo scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from physical_ai_agent.sim.so101_camera_input import _make_camera, postprocess_camera_frame
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, sample_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Render default vs enhanced SO101 MuJoCo visuals.")
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_realistic_render_probe"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = render_comparison(
        env_id=args.env_id,
        output_dir=args.output_dir,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def render_comparison(
    *,
    env_id: str,
    output_dir: Path,
    seed: int,
    warmup_steps: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    import gymnasium as gym
    import mujoco
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    env = gym.make(env_id, render_mode=None)
    renderers: dict[str, Any] = {}
    try:
        env.reset(seed=seed)
        for step in range(warmup_steps):
            env.step(sample_action(env.action_space, step / max(1, warmup_steps - 1)))

        renderers = {
            name: mujoco.Renderer(env.unwrapped.model, height=height, width=width)
            for name in ("scene", "egocentric_cam", "wrist_cam", "top_down")
        }
        default_frames = _render_views(env, renderers)
        _apply_enhanced_visuals(env.unwrapped.model)
        enhanced_frames = _render_views(env, renderers)
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()

    written = _write_outputs(default_frames, enhanced_frames, output_dir)
    report = {
        "env_id": env_id,
        "seed": seed,
        "warmup_steps": warmup_steps,
        "width": width,
        "height": height,
        "renderer": "mujoco.Renderer",
        "changes": [
            "warmer lower-saturation SO101 body material",
            "less mirror-like black servo material",
            "warmer floor color",
            "stronger key light plus ambient fill",
            "opaque fingertip pads for clearer contact geometry",
            "small contrast/sharpness tone pass after MuJoCo render",
        ],
        **written,
    }
    (output_dir / "render_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _render_views(env: Any, renderers: dict[str, Any]) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for name, renderer in renderers.items():
        if name == "scene":
            renderer.update_scene(env.unwrapped.data, camera=_studio_camera())
            pixels = renderer.render()
        else:
            renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, name))
            pixels = postprocess_camera_frame(name, renderer.render())
        frames[name] = np.asarray(pixels)
    return frames


def _apply_enhanced_visuals(model: Any) -> None:
    _set_geom_rgba(model, "floor", [0.58, 0.56, 0.51, 1.0])
    _set_geom_rgba(model, "static_finger_pad", [0.03, 0.08, 0.18, 1.0])
    _set_geom_rgba(model, "moving_finger_pad", [0.55, 0.06, 0.04, 1.0])

    for mat_id in range(model.nmat):
        name = model.mat(mat_id).name
        if "so101" in name or "holder" in name or "arm" in name or "jaw" in name or "wrist" in name:
            model.mat_rgba[mat_id] = [0.95, 0.72, 0.18, 1.0]
            model.mat_specular[mat_id] = 0.18
            model.mat_shininess[mat_id] = 0.28
            model.mat_roughness[mat_id] = 0.62
        if "sts3215" in name:
            model.mat_rgba[mat_id] = [0.025, 0.028, 0.030, 1.0]
            model.mat_specular[mat_id] = 0.10
            model.mat_shininess[mat_id] = 0.18
            model.mat_roughness[mat_id] = 0.74

    for light_id in range(model.nlight):
        model.light_ambient[light_id] = [0.055, 0.052, 0.048]
        model.light_diffuse[light_id] = [0.58, 0.53, 0.46]
        model.light_specular[light_id] = [0.16, 0.14, 0.12]
    if model.nlight >= 1:
        model.light_pos[0] = [1.2, -0.8, 2.6]
        model.light_dir[0] = [-0.38, 0.28, -0.88]
    if model.nlight >= 2:
        model.light_pos[1] = [-0.7, 0.9, 2.4]
        model.light_dir[1] = [0.28, -0.36, -0.89]

    model.vis.quality.shadowsize = max(int(model.vis.quality.shadowsize), 4096)
    model.vis.quality.offsamples = max(int(model.vis.quality.offsamples), 4)
    model.vis.map.fogstart = 4.0
    model.vis.map.fogend = 8.0


def _studio_camera() -> Any:
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.14, 0.02, 0.06]
    camera.distance = 0.55
    camera.azimuth = 225
    camera.elevation = -38
    return camera


def _set_geom_rgba(model: Any, name: str, rgba: list[float]) -> None:
    for geom_id in range(model.ngeom):
        if model.geom(geom_id).name == name:
            model.geom_rgba[geom_id] = rgba
            return


def _write_outputs(
    default_frames: dict[str, np.ndarray],
    enhanced_frames: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for label, frames in (("default", default_frames), ("enhanced", enhanced_frames)):
        for view, pixels in frames.items():
            image = _finish_image(pixels) if label == "enhanced" else Image.fromarray(pixels)
            path = output_dir / f"{label}_{view}.png"
            image.save(path)
            paths[f"{label}_{view}_path"] = str(path)

    sheet = _comparison_sheet(default_frames, enhanced_frames)
    sheet_path = output_dir / "so101_realistic_comparison.png"
    sheet.save(sheet_path)
    paths["comparison_path"] = str(sheet_path)
    return paths


def _finish_image(pixels: np.ndarray) -> Image.Image:
    image = Image.fromarray(pixels).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.clip((arr - 0.5) * 1.06 + 0.5, 0.0, 1.0)
    arr = np.clip(arr ** 1.02, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _comparison_sheet(default_frames: dict[str, np.ndarray], enhanced_frames: dict[str, np.ndarray]) -> Image.Image:
    views = ("scene", "egocentric_cam", "wrist_cam", "top_down")
    cell_w, cell_h = 360, 270
    label_h = 34
    sheet = Image.new("RGB", (cell_w * len(views), (cell_h + label_h) * 2), (238, 238, 232))
    draw = ImageDraw.Draw(sheet)
    for col, view in enumerate(views):
        for row, (label, frames) in enumerate((("default", default_frames), ("enhanced", enhanced_frames))):
            image = _finish_image(frames[view]) if label == "enhanced" else Image.fromarray(frames[view]).convert("RGB")
            image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = col * cell_w + (cell_w - image.width) // 2
            y = row * (cell_h + label_h) + label_h + (cell_h - image.height) // 2
            sheet.paste(image, (x, y))
            draw.text((col * cell_w + 12, row * (cell_h + label_h) + 9), f"{label} / {view}", fill=(30, 30, 30))
    return sheet


if __name__ == "__main__":
    main()
