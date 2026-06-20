#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.policies.so101_visual_reach_delta import (
    load_so101_visual_reach_delta_checkpoint,
)
from physical_ai_agent.sim.so101_camera_input import _make_camera
from physical_ai_agent.sim.so101_live_viewer import _cartesian_error_controller_action
from physical_ai_agent.sim.so101_visual_rl import SO101VisualRLConfig, _json_safe_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a high-resolution multi-view rollout for a visual-reach checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env-id", default="MuJoCoReach-v1")
    parser.add_argument("--policy-camera", default="")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--policy-width", type=int, default=64)
    parser.add_argument("--policy-height", type=int, default=64)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/so101_visual_rl/reach_delta_artifacts/highres_rollout"),
    )
    args = parser.parse_args()

    report = render_rollout(
        checkpoint=args.checkpoint,
        env_id=args.env_id,
        policy_camera=args.policy_camera,
        steps=args.steps,
        seed=args.seed,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        policy_width=args.policy_width,
        policy_height=args.policy_height,
        fps=args.fps,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def render_rollout(
    *,
    checkpoint: Path,
    env_id: str,
    policy_camera: str,
    steps: int,
    seed: int,
    panel_width: int,
    panel_height: int,
    policy_width: int,
    policy_height: int,
    fps: int,
    output_dir: Path,
) -> dict[str, Any]:
    import gymnasium as gym
    import imageio.v2 as imageio
    import mujoco
    import so101_nexus_mujoco  # noqa: F401
    from PIL import Image, ImageDraw

    model, metadata = load_so101_visual_reach_delta_checkpoint(checkpoint)
    model_config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    policy_camera = policy_camera or model_config.get("camera_name", "top_down")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    env = gym.make(env_id, render_mode=None)
    obs, info = env.reset(seed=seed)
    renderers = {
        "scene_3d": mujoco.Renderer(env.unwrapped.model, height=panel_height, width=panel_width),
        "wrist_cam": mujoco.Renderer(env.unwrapped.model, height=panel_height, width=panel_width),
        "egocentric_cam": mujoco.Renderer(env.unwrapped.model, height=panel_height, width=panel_width),
        "top_down": mujoco.Renderer(env.unwrapped.model, height=panel_height, width=panel_width),
        "policy": mujoco.Renderer(env.unwrapped.model, height=policy_height, width=policy_width),
    }
    video_frames = []
    records = []
    try:
        for step in range(steps):
            policy_pixels = _render_camera(env, renderers["policy"], policy_camera)
            observation = {
                "image": policy_pixels.transpose(2, 0, 1),
                "state": np.asarray(obs, dtype=np.float32).reshape(-1),
            }
            pred_error = _predict_error(model, observation)
            action = _cartesian_error_controller_action(env, pred_error)
            distance_before = float(np.linalg.norm(_reach_error(env)))
            frames = {
                "scene_3d": _render_scene(env, renderers["scene_3d"]),
                "wrist_cam": _render_camera(env, renderers["wrist_cam"], "wrist_cam"),
                "egocentric_cam": _render_camera(env, renderers["egocentric_cam"], "egocentric_cam"),
                "top_down": _render_camera(env, renderers["top_down"], "top_down"),
            }
            composed = _compose_multiview_frame(
                frames=frames,
                step=step,
                distance=distance_before,
                pred_error=pred_error,
                action=action,
                info=_json_safe_info(info),
                panel_width=panel_width,
                panel_height=panel_height,
            )
            frame_path = frames_dir / f"frame_{step:04d}.jpg"
            Image.fromarray(composed).save(frame_path, quality=90)
            video_frames.append(composed)

            obs, reward, terminated, truncated, info = env.step(action)
            records.append(
                {
                    "step": step,
                    "distance_before": distance_before,
                    "reward": float(reward),
                    "success": bool(info.get("success", False)),
                    "predicted_error_norm": float(np.linalg.norm(pred_error)),
                    "frame_path": str(frame_path),
                }
            )
            if terminated or truncated:
                break
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()

    gif_path = output_dir / "rollout_multiview.gif"
    mp4_path = output_dir / "rollout_multiview.mp4"
    imageio.mimsave(gif_path, video_frames, duration=1.0 / max(1, fps))
    mp4_error = ""
    try:
        imageio.mimsave(mp4_path, video_frames, fps=fps, macro_block_size=8)
    except Exception as exc:  # pragma: no cover - depends on local ffmpeg plugin.
        mp4_error = str(exc)
        mp4_path = Path("")

    manifest = {
        "operation": "render_so101_visual_reach_rollout",
        "env_id": env_id,
        "checkpoint": str(checkpoint),
        "policy_camera": policy_camera,
        "steps": len(records),
        "seed": seed,
        "panel_shape": [panel_height, panel_width],
        "gif_path": str(gif_path),
        "mp4_path": str(mp4_path) if mp4_path else "",
        "mp4_error": mp4_error,
        "initial_distance": records[0]["distance_before"] if records else None,
        "final_distance": records[-1]["distance_before"] if records else None,
        "success": any(record["success"] for record in records),
        "records": records,
    }
    manifest_path = output_dir / "rollout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def _render_scene(env: Any, renderer: Any) -> Any:
    renderer.update_scene(env.unwrapped.data)
    return renderer.render()


def _render_camera(env: Any, renderer: Any, camera_name: str) -> Any:
    renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, camera_name))
    return renderer.render()


def _predict_error(model: Any, observation: dict[str, Any]) -> Any:
    import torch

    with torch.no_grad():
        return model(observation).detach().cpu().numpy()[0].astype(float)


def _reach_error(env: Any) -> Any:
    model = env.unwrapped.model
    data = env.unwrapped.data
    target = data.site_xpos[model.site("reach_target").id]
    gripper = data.site_xpos[model.site("gripperframe").id]
    return np.asarray(target - gripper, dtype=float)


def _compose_multiview_frame(
    *,
    frames: dict[str, Any],
    step: int,
    distance: float,
    pred_error: Any,
    action: list[float],
    info: dict[str, Any],
    panel_width: int,
    panel_height: int,
) -> Any:
    from PIL import Image, ImageDraw

    labels = ["scene_3d", "wrist_cam", "egocentric_cam", "top_down"]
    telemetry_h = 104
    canvas = Image.new("RGB", (panel_width * 2, panel_height * 2 + telemetry_h), (245, 245, 240))
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(labels):
        image = Image.fromarray(frames[label]).convert("RGB")
        x = (index % 2) * panel_width
        y = (index // 2) * panel_height
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + panel_width - 1, y + 24), fill=(245, 245, 240))
        draw.text((x + 10, y + 7), label, fill=(20, 20, 20))

    y0 = panel_height * 2 + 12
    pred_norm = float(np.linalg.norm(pred_error))
    success = bool(info.get("success", False))
    draw.text(
        (16, y0),
        (
            f"step {step:03d}  distance {distance:.4f}m  "
            f"pred |delta| {pred_norm:.4f}m  success {success}"
        ),
        fill=(25, 25, 25),
    )
    draw.text(
        (16, y0 + 28),
        "pred delta xyz "
        + ", ".join(f"{float(value):+.3f}" for value in pred_error)
        + "    action "
        + ", ".join(f"{float(value):+.2f}" for value in action[:6]),
        fill=(25, 25, 25),
    )
    if "tcp_to_target_dist" in info:
        draw.text((16, y0 + 56), f"env tcp_to_target_dist {info['tcp_to_target_dist']}", fill=(25, 25, 25))
    return np.asarray(canvas)


if __name__ == "__main__":
    main()
