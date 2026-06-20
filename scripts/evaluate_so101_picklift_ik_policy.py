#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.sim.so101_live_viewer import _cartesian_error_controller_action
from physical_ai_agent.sim.so101_visual_rl import SO101VisualRLConfig, _json_safe_info
from train_so101_visual_picklift_bc import _compose_picklift_frame, _plot_bars, _solve_pregrasp_qpos
from train_so101_visual_picklift_delta import _object_set_description, make_diverse_picklift_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the SO101 PickLift IK controller-prior policy on diverse cubes."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_visual_rl/picklift_ik_policy"))
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=19400)
    parser.add_argument("--render-seed", type=int, default=19400)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    report = evaluate_ik_policy(
        output_dir=args.output_dir,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        render_seed=args.render_seed,
        fps=args.fps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def evaluate_ik_policy(
    *,
    output_dir: Path,
    episodes: int,
    steps: int,
    seed: int,
    render_seed: int,
    fps: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    videos_dir = output_dir / "videos"
    plots_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)
    config = SO101VisualRLConfig(
        env_id="MuJoCoPickLift-v1",
        camera_name="top_down",
        width=64,
        height=64,
        include_state=True,
        channel_first=True,
    )
    episode_reports = []
    for episode in range(episodes):
        result = run_ik_episode(config=config, seed=seed + episode, steps=steps)
        episode_reports.append({"episode": episode, "seed": seed + episode, **result})
    success_plot = plots_dir / "ik_policy_success_curve.png"
    _plot_bars(episode_reports, success_plot)
    rollout = render_ik_rollout(
        config=config,
        seed=render_seed,
        steps=steps,
        fps=fps,
        output_dir=videos_dir,
    )
    success_steps = [
        item["steps_to_success"] for item in episode_reports if item["steps_to_success"] is not None
    ]
    manifest = {
        "operation": "evaluate_so101_picklift_ik_policy",
        "config": {
            "env_id": config.env_id,
            "camera_name": config.camera_name,
            "width": config.width,
            "height": config.height,
        },
        "object_set": _object_set_description(),
        "episodes": episode_reports,
        "success_rate": float(np.mean([item["success"] for item in episode_reports])),
        "mean_final_lift_height": float(np.mean([item["final_lift_height"] for item in episode_reports])),
        "mean_steps_to_success": float(np.mean(success_steps)) if success_steps else None,
        "artifacts": {
            "success_plot": str(success_plot),
            "rollout_gif": rollout["gif_path"],
            "rollout_mp4": rollout["mp4_path"],
            "rollout_manifest": rollout["manifest_path"],
        },
    }
    manifest_path = output_dir / "ik_policy_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def run_ik_episode(*, config: SO101VisualRLConfig, seed: int, steps: int) -> dict[str, Any]:
    env = make_diverse_picklift_env(config)
    try:
        _obs, info = env.reset(seed=seed)
        q_open = _solve_pregrasp_qpos(env)
        q_open[-1] = 0.25
        q_close = q_open.copy()
        q_close[-1] = float(env.action_space.low[-1])
        success_step = None
        for step in range(steps):
            action = ik_policy_action(env, step=step, q_open=q_open, q_close=q_close)
            _obs, _reward, terminated, truncated, info = env.step(action)
            if bool(info.get("success", False)) and success_step is None:
                success_step = step + 1
            if terminated or truncated:
                break
        return {
            "success": bool(info.get("success", False)),
            "steps_to_success": success_step,
            "final_is_grasped": float(info.get("is_grasped", 0.0)),
            "final_lift_height": float(info.get("lift_height", 0.0)),
            "final_tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        }
    finally:
        env.close()


def ik_policy_action(env: Any, *, step: int, q_open: np.ndarray, q_close: np.ndarray) -> np.ndarray:
    if step < 58:
        return q_open.copy()
    if step < 118:
        return q_close.copy()
    action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=float)
    action[-1] = q_close[-1]
    return np.clip(action, env.action_space.low, env.action_space.high)


def render_ik_rollout(
    *,
    config: SO101VisualRLConfig,
    seed: int,
    steps: int,
    fps: int,
    output_dir: Path,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import mujoco

    from physical_ai_agent.sim.so101_camera_input import _make_camera

    env = make_diverse_picklift_env(config)
    panel_width = 480
    panel_height = 360
    renderers = {
        "scene_3d": mujoco.Renderer(env.unwrapped.model, height=panel_height, width=panel_width),
        "wrist_cam": mujoco.Renderer(env.unwrapped.model, height=panel_height, width=panel_width),
        "egocentric_cam": mujoco.Renderer(env.unwrapped.model, height=panel_height, width=panel_width),
        "top_down": mujoco.Renderer(env.unwrapped.model, height=panel_height, width=panel_width),
    }
    frames: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    try:
        _obs, info = env.reset(seed=seed)
        q_open = _solve_pregrasp_qpos(env)
        q_open[-1] = 0.25
        q_close = q_open.copy()
        q_close[-1] = float(env.action_space.low[-1])
        for step in range(steps):
            action = ik_policy_action(env, step=step, q_open=q_open, q_close=q_close)
            views = {}
            for name, renderer in renderers.items():
                if name == "scene_3d":
                    renderer.update_scene(env.unwrapped.data)
                else:
                    renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, name))
                views[name] = renderer.render()
            frames.append(
                _compose_picklift_frame(
                    views=views,
                    step=step,
                    action=action,
                    info=_json_safe_info(info),
                    panel_width=panel_width,
                    panel_height=panel_height,
                )
            )
            _obs, reward, terminated, truncated, info = env.step(action)
            records.append(
                {
                    "step": step,
                    "reward": float(reward),
                    "success": bool(info.get("success", False)),
                    "is_grasped": float(info.get("is_grasped", 0.0)),
                    "lift_height": float(info.get("lift_height", 0.0)),
                    "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
                }
            )
            if terminated or truncated:
                break
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()
    gif_path = output_dir / "picklift_ik_policy_multiview.gif"
    mp4_path = output_dir / "picklift_ik_policy_multiview.mp4"
    imageio.mimsave(gif_path, frames, duration=1.0 / max(1, fps))
    mp4_error = ""
    try:
        imageio.mimsave(mp4_path, frames, fps=fps, macro_block_size=8)
    except Exception as exc:  # pragma: no cover
        mp4_error = str(exc)
        mp4_path = Path("")
    manifest = {
        "operation": "render_so101_picklift_ik_policy_rollout",
        "seed": seed,
        "steps": len(records),
        "success": any(record["success"] for record in records),
        "final_is_grasped": records[-1]["is_grasped"] if records else 0.0,
        "final_lift_height": records[-1]["lift_height"] if records else 0.0,
        "gif_path": str(gif_path),
        "mp4_path": str(mp4_path) if mp4_path else "",
        "mp4_error": mp4_error,
        "records": records,
    }
    manifest_path = output_dir / "picklift_ik_policy_rollout_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()
