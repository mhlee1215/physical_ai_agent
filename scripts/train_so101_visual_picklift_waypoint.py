#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.policies.so101_visual_actor_critic import (
    load_so101_visual_actor_critic_checkpoint,
    make_so101_visual_actor_critic,
    save_so101_visual_actor_critic_checkpoint,
)
from physical_ai_agent.sim.so101_live_viewer import _cartesian_error_controller_action
from physical_ai_agent.sim.so101_visual_rl import (
    SO101VisualRLConfig,
    _json_safe_info,
    make_so101_visual_rl_env,
)
from train_so101_visual_picklift_bc import (
    _compose_picklift_frame,
    _mean,
    _plot_bars,
    _plot_curve,
    _resolve_device,
    _solve_pregrasp_qpos,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a visual waypoint policy for SO101 PickLift with a grasp/lift controller prior."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_visual_rl/picklift_waypoint"))
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=8100)
    parser.add_argument("--camera-name", default="top_down")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--render-seed", type=int, default=8100)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    report = train_waypoint(
        output_dir=args.output_dir,
        episodes=args.episodes,
        epochs=args.epochs,
        eval_episodes=args.eval_episodes,
        steps=args.steps,
        seed=args.seed,
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        render_seed=args.render_seed,
        fps=args.fps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def train_waypoint(
    *,
    output_dir: Path,
    episodes: int,
    epochs: int,
    eval_episodes: int,
    steps: int,
    seed: int,
    camera_name: str,
    width: int,
    height: int,
    batch_size: int,
    lr: float,
    device: str,
    render_seed: int,
    fps: int,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    videos_dir = output_dir / "videos"
    plots_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)

    resolved_device = _resolve_device(device)
    config = SO101VisualRLConfig(
        env_id="MuJoCoPickLift-v1",
        camera_name=camera_name,
        width=width,
        height=height,
        include_state=True,
        channel_first=True,
    )
    env = make_so101_visual_rl_env(config)
    try:
        images, states, targets, teacher_reports = _collect_waypoints(
            env=env,
            episodes=episodes,
            seed=seed,
        )
        model = make_so101_visual_actor_critic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            hidden_dim=128,
        ).to(resolved_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        image_tensor = torch.as_tensor(images, dtype=torch.uint8)
        state_tensor = torch.as_tensor(states, dtype=torch.float32)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32)
        losses: list[float] = []
        generator = torch.Generator().manual_seed(seed)
        for _epoch in range(epochs):
            order = torch.randperm(len(target_tensor), generator=generator)
            epoch_losses = []
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                batch = {
                    "image": image_tensor[idx].to(resolved_device),
                    "state": state_tensor[idx].to(resolved_device),
                }
                target = target_tensor[idx].to(resolved_device)
                pred = model(batch).action_mean
                loss = torch.nn.functional.mse_loss(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(epoch_losses)))
        checkpoint_path = output_dir / "so101_visual_picklift_waypoint.pt"
        save_so101_visual_actor_critic_checkpoint(
            path=checkpoint_path,
            model=model.cpu(),
            observation_space=env.observation_space,
            action_space=env.action_space,
            metadata={
                "operation": "train_so101_visual_picklift_waypoint",
                "config": asdict(config),
                "episodes": episodes,
                "epochs": epochs,
                "steps": steps,
                "device": resolved_device,
                "controller_prior": "predict_pregrasp_waypoint_then_close_and_lift",
            },
        )
    finally:
        env.close()

    eval_report = _evaluate_waypoint_policy(
        checkpoint_path=checkpoint_path,
        config=config,
        episodes=eval_episodes,
        steps=steps,
        seed=seed + 10_000,
    )
    rollout_report = _render_waypoint_rollout(
        checkpoint_path=checkpoint_path,
        config=config,
        steps=steps,
        seed=render_seed,
        fps=fps,
        output_dir=videos_dir,
    )
    loss_plot = plots_dir / "waypoint_loss_curve.png"
    success_plot = plots_dir / "waypoint_eval_success_curve.png"
    _plot_curve(losses, loss_plot, title="PickLift visual waypoint loss", ylabel="MSE")
    _plot_bars(eval_report["episodes"], success_plot)

    manifest = {
        "operation": "train_so101_visual_picklift_waypoint",
        "checkpoint_path": str(checkpoint_path),
        "config": asdict(config),
        "device": resolved_device,
        "dataset_samples": int(len(targets)),
        "teacher": teacher_reports,
        "training": {"losses": losses, "final_loss": losses[-1] if losses else None},
        "evaluation": eval_report,
        "artifacts": {
            "loss_plot": str(loss_plot),
            "success_plot": str(success_plot),
            "rollout_gif": rollout_report["gif_path"],
            "rollout_mp4": rollout_report["mp4_path"],
            "rollout_manifest": rollout_report["manifest_path"],
        },
    }
    manifest_path = output_dir / "training_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _collect_waypoints(*, env: Any, episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    errors: list[float] = []
    for episode in range(episodes):
        obs, _info = env.reset(seed=seed + episode)
        target = _solve_pregrasp_qpos(env).astype(np.float32)
        images.append(np.asarray(obs["image"], dtype=np.uint8).copy())
        states.append(np.asarray(obs["state"], dtype=np.float32).copy())
        targets.append(target)
        # Add a few on-policy approach states so the regressor remains stable once the arm moves.
        for _ in range(3):
            obs, _reward, terminated, truncated, _info = env.step(target)
            if terminated or truncated:
                break
            images.append(np.asarray(obs["image"], dtype=np.uint8).copy())
            states.append(np.asarray(obs["state"], dtype=np.float32).copy())
            targets.append(target)
        errors.append(float(np.linalg.norm(target)))
    return (
        np.stack(images, axis=0),
        np.stack(states, axis=0),
        np.stack(targets, axis=0),
        {"episodes": episodes, "mean_target_norm": _mean(errors)},
    )


def _predict_waypoint(checkpoint_path: Path, obs: dict[str, Any]) -> np.ndarray:
    import torch

    model, _metadata = load_so101_visual_actor_critic_checkpoint(checkpoint_path)
    with torch.no_grad():
        return model.act(obs, deterministic=True)["action"].cpu().numpy()[0].astype(float)


def _run_waypoint_episode(
    *,
    env: Any,
    checkpoint_path: Path,
    steps: int,
    seed: int | None = None,
    render_callback: Any | None = None,
) -> dict[str, Any]:
    import torch

    model, _metadata = load_so101_visual_actor_critic_checkpoint(checkpoint_path)
    obs, info = env.reset(seed=seed)
    with torch.no_grad():
        q_open = model.act(obs, deterministic=True)["action"].cpu().numpy()[0].astype(float)
    q_open[-1] = 0.25
    q_open = np.clip(q_open, env.action_space.low, env.action_space.high)
    q_close = q_open.copy()
    q_close[-1] = float(env.action_space.low[-1])
    success_step = None
    records = []
    for step in range(steps):
        if step < 58:
            action = q_open.copy()
        elif step < 118:
            action = q_close.copy()
        else:
            action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=float)
            action[-1] = q_close[-1]
            action = np.clip(action, env.action_space.low, env.action_space.high)
        if render_callback is not None:
            render_callback(step, action, info)
        obs, reward, terminated, truncated, info = env.step(action)
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
        "predicted_q_open": q_open.tolist(),
        "records": records,
    }


def _evaluate_waypoint_policy(
    *,
    checkpoint_path: Path,
    config: SO101VisualRLConfig,
    episodes: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    env = make_so101_visual_rl_env(config)
    reports = []
    try:
        for episode in range(episodes):
            result = _run_waypoint_episode(
                env=env,
                checkpoint_path=checkpoint_path,
                steps=steps,
                seed=seed + episode,
            )
            reports.append(
                {
                    "episode": episode,
                    "seed": seed + episode,
                    "success": result["success"],
                    "steps_to_success": result["steps_to_success"],
                    "final_is_grasped": result["final_is_grasped"],
                    "final_lift_height": result["final_lift_height"],
                    "final_tcp_to_obj_dist": result["final_tcp_to_obj_dist"],
                }
            )
    finally:
        env.close()
    return {
        "episodes": reports,
        "success_rate": _mean([item["success"] for item in reports]),
        "mean_final_lift_height": _mean([item["final_lift_height"] for item in reports]),
        "mean_steps_to_success": _mean(
            [item["steps_to_success"] for item in reports if item["steps_to_success"] is not None]
        ),
    }


def _render_waypoint_rollout(
    *,
    checkpoint_path: Path,
    config: SO101VisualRLConfig,
    steps: int,
    seed: int,
    fps: int,
    output_dir: Path,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import mujoco

    from physical_ai_agent.sim.so101_camera_input import _make_camera

    env = make_so101_visual_rl_env(config)
    panel_width = 480
    panel_height = 360
    renderers = {
        "scene_3d": mujoco.Renderer(env.unwrapped.model, height=panel_height, width=panel_width),
        "wrist_cam": mujoco.Renderer(env.unwrapped.model, height=panel_height, width=panel_width),
        "egocentric_cam": mujoco.Renderer(env.unwrapped.model, height=panel_height, width=panel_width),
        "top_down": mujoco.Renderer(env.unwrapped.model, height=panel_height, width=panel_width),
    }
    frames: list[np.ndarray] = []
    def capture(step: int, action: np.ndarray, info: dict[str, Any]) -> None:
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

    try:
        result = _run_waypoint_episode(
            env=env,
            checkpoint_path=checkpoint_path,
            steps=steps,
            seed=seed,
            render_callback=capture,
        )
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()
    gif_path = output_dir / "picklift_waypoint_multiview.gif"
    mp4_path = output_dir / "picklift_waypoint_multiview.mp4"
    imageio.mimsave(gif_path, frames, duration=1.0 / max(1, fps))
    mp4_error = ""
    try:
        imageio.mimsave(mp4_path, frames, fps=fps, macro_block_size=8)
    except Exception as exc:  # pragma: no cover
        mp4_error = str(exc)
        mp4_path = Path("")
    manifest = {
        "operation": "render_so101_visual_picklift_waypoint_rollout",
        "checkpoint_path": str(checkpoint_path),
        "seed": seed,
        "steps": len(result["records"]),
        "success": result["success"],
        "final_is_grasped": result["final_is_grasped"],
        "final_lift_height": result["final_lift_height"],
        "gif_path": str(gif_path),
        "mp4_path": str(mp4_path) if mp4_path else "",
        "mp4_error": mp4_error,
        "records": result["records"],
    }
    manifest_path = output_dir / "picklift_waypoint_rollout_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()
