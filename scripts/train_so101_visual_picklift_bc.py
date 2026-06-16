#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.policies.so101_visual_actor_critic import (
    make_so101_visual_actor_critic,
    save_so101_visual_actor_critic_checkpoint,
)
from physical_ai_agent.sim.so101_live_viewer import _cartesian_error_controller_action
from physical_ai_agent.sim.so101_visual_rl import (
    SO101VisualRLConfig,
    _json_safe_info,
    make_so101_visual_rl_env,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a lightweight visual BC policy for SO101 MuJoCo PickLift."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_visual_rl/picklift_bc"))
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--camera-name", default="top_down")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--render-seed", type=int, default=7000)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    report = train_picklift_bc(
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


def train_picklift_bc(
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
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image, ImageDraw

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
        images, states, actions, teacher_reports = _collect_teacher_dataset(
            env=env,
            episodes=episodes,
            steps=steps,
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
        action_tensor = torch.as_tensor(actions, dtype=torch.float32)
        losses: list[float] = []
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(1, epochs + 1):
            order = torch.randperm(len(action_tensor), generator=generator)
            epoch_losses = []
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                batch = {
                    "image": image_tensor[idx].to(resolved_device),
                    "state": state_tensor[idx].to(resolved_device),
                }
                target = action_tensor[idx].to(resolved_device)
                pred = model(batch).action_mean
                loss = torch.nn.functional.mse_loss(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(epoch_losses)))

        checkpoint_path = output_dir / "so101_visual_picklift_bc.pt"
        save_so101_visual_actor_critic_checkpoint(
            path=checkpoint_path,
            model=model.cpu(),
            observation_space=env.observation_space,
            action_space=env.action_space,
            metadata={
                "operation": "train_so101_visual_picklift_bc",
                "config": asdict(config),
                "episodes": episodes,
                "epochs": epochs,
                "steps": steps,
                "device": resolved_device,
                "teacher_success_rate": _mean([r["success"] for r in teacher_reports]),
            },
        )
    finally:
        env.close()

    eval_report = _evaluate_policy(
        checkpoint_path=checkpoint_path,
        config=config,
        episodes=eval_episodes,
        steps=steps,
        seed=seed + 10_000,
    )
    rollout_report = _render_policy_rollout(
        checkpoint_path=checkpoint_path,
        config=config,
        steps=steps,
        seed=render_seed,
        fps=fps,
        output_dir=videos_dir,
    )
    loss_plot = plots_dir / "bc_loss_curve.png"
    _plot_curve(losses, loss_plot, title="PickLift visual BC loss", ylabel="MSE")
    success_plot = plots_dir / "eval_success_curve.png"
    _plot_bars(eval_report["episodes"], success_plot)

    manifest = {
        "operation": "train_so101_visual_picklift_bc",
        "checkpoint_path": str(checkpoint_path),
        "config": asdict(config),
        "device": resolved_device,
        "dataset_samples": int(len(actions)),
        "teacher": {
            "episodes": teacher_reports,
            "success_rate": _mean([r["success"] for r in teacher_reports]),
            "mean_final_lift_height": _mean([r["final_lift_height"] for r in teacher_reports]),
            "mean_steps_to_success": _mean(
                [r["steps_to_success"] for r in teacher_reports if r["steps_to_success"] is not None]
            ),
        },
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


def _collect_teacher_dataset(
    *,
    env: Any,
    episodes: int,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        q_open = _solve_pregrasp_qpos(env)
        q_close = q_open.copy()
        q_close[-1] = float(env.action_space.low[-1])
        success_step = None
        last_info = info
        for step in range(steps):
            action = _teacher_action(env=env, step=step, q_open=q_open, q_close=q_close)
            images.append(np.asarray(obs["image"], dtype=np.uint8).copy())
            states.append(np.asarray(obs["state"], dtype=np.float32).copy())
            actions.append(action.astype(np.float32).copy())
            obs, _reward, terminated, truncated, last_info = env.step(action)
            if bool(last_info.get("success", False)) and success_step is None:
                success_step = step + 1
            if terminated or truncated:
                break
        reports.append(
            {
                "episode": episode,
                "seed": seed + episode,
                "success": bool(last_info.get("success", False)),
                "steps_to_success": success_step,
                "final_is_grasped": float(last_info.get("is_grasped", 0.0)),
                "final_lift_height": float(last_info.get("lift_height", 0.0)),
            }
        )
    return (
        np.stack(images, axis=0),
        np.stack(states, axis=0),
        np.stack(actions, axis=0),
        reports,
    )


def _solve_pregrasp_qpos(env: Any) -> np.ndarray:
    import mujoco
    from scipy.optimize import least_squares

    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data
    joint_addrs = [model.jnt_qposadr[jid] for jid in unwrapped._joint_ids]
    low = np.asarray(env.action_space.low, dtype=float)
    high = np.asarray(env.action_space.high, dtype=float)
    static_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
    moving_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
    obj_pos = unwrapped._get_target_pose()[:3].copy()
    q_seed = np.asarray([data.qpos[addr] for addr in joint_addrs], dtype=float)
    axis = np.asarray([1.0, -1.0, 0.0], dtype=float) / math.sqrt(2.0)
    open_value = 0.25
    gap = 0.034
    desired_static = obj_pos - axis * gap * 0.5 + np.asarray([0.0, 0.0, 0.004])
    desired_moving = obj_pos + axis * gap * 0.5 + np.asarray([0.0, 0.0, 0.004])

    def set_qpos(qpos: np.ndarray) -> None:
        for addr, value in zip(joint_addrs, qpos):
            data.qpos[addr] = value
        data.ctrl[unwrapped._actuator_ids] = np.clip(qpos, low, high)
        mujoco.mj_forward(model, data)

    def residual(arm_qpos: np.ndarray) -> np.ndarray:
        qpos = np.concatenate([arm_qpos, np.asarray([open_value])])
        set_qpos(qpos)
        static_pos = data.geom_xpos[static_pad]
        moving_pos = data.geom_xpos[moving_pad]
        return np.concatenate(
            [
                (static_pos - desired_static) * 20.0,
                (moving_pos - desired_moving) * 20.0,
                np.maximum(0.0, 0.004 - static_pos[2:3]) * 20.0,
                np.maximum(0.0, 0.004 - moving_pos[2:3]) * 20.0,
                (arm_qpos - q_seed[:5]) * 0.05,
            ]
        )

    starts = [
        q_seed[:5],
        np.asarray([-0.5, 0.4, 0.1, 0.5, -1.3]),
        np.asarray([0.0, 0.55, -0.25, 0.85, 1.2]),
        np.asarray([0.6, 0.2, 0.2, 0.6, -1.0]),
        np.asarray([-0.8, 0.2, 0.2, 0.6, -1.0]),
    ]
    best = None
    for start in starts:
        result = least_squares(
            residual,
            np.clip(start, low[:5], high[:5]),
            bounds=(low[:5], high[:5]),
            max_nfev=180,
        )
        cost = float(np.linalg.norm(residual(result.x)))
        candidate = np.concatenate([result.x, np.asarray([open_value])])
        if best is None or cost < best[0]:
            best = (cost, candidate)
    assert best is not None
    return np.clip(best[1], low, high)


def _teacher_action(*, env: Any, step: int, q_open: np.ndarray, q_close: np.ndarray) -> np.ndarray:
    if step < 58:
        return q_open.copy()
    if step < 118:
        return q_close.copy()
    action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=float)
    action[-1] = q_close[-1]
    return np.clip(action, env.action_space.low, env.action_space.high)


def _evaluate_policy(
    *,
    checkpoint_path: Path,
    config: SO101VisualRLConfig,
    episodes: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    from physical_ai_agent.policies.so101_visual_actor_critic import (
        load_so101_visual_actor_critic_checkpoint,
    )

    import torch

    model, _metadata = load_so101_visual_actor_critic_checkpoint(checkpoint_path)
    env = make_so101_visual_rl_env(config)
    reports = []
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            success_step = None
            for step in range(steps):
                with torch.no_grad():
                    action = model.act(obs, deterministic=True)["action"].cpu().numpy()[0]
                obs, _reward, terminated, truncated, info = env.step(action)
                if bool(info.get("success", False)) and success_step is None:
                    success_step = step + 1
                if terminated or truncated:
                    break
            reports.append(
                {
                    "episode": episode,
                    "seed": seed + episode,
                    "success": bool(info.get("success", False)),
                    "steps_to_success": success_step,
                    "final_is_grasped": float(info.get("is_grasped", 0.0)),
                    "final_lift_height": float(info.get("lift_height", 0.0)),
                    "final_tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
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


def _render_policy_rollout(
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
    import torch
    from PIL import Image, ImageDraw

    from physical_ai_agent.policies.so101_visual_actor_critic import (
        load_so101_visual_actor_critic_checkpoint,
    )
    from physical_ai_agent.sim.so101_camera_input import _make_camera

    model, _metadata = load_so101_visual_actor_critic_checkpoint(checkpoint_path)
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
    records: list[dict[str, Any]] = []
    obs, info = env.reset(seed=seed)
    try:
        for step in range(steps):
            with torch.no_grad():
                action = model.act(obs, deterministic=True)["action"].cpu().numpy()[0]
            views = {}
            for name, renderer in renderers.items():
                if name == "scene_3d":
                    renderer.update_scene(env.unwrapped.data)
                else:
                    renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, name))
                views[name] = renderer.render()
            frame = _compose_picklift_frame(
                views=views,
                step=step,
                action=action,
                info=_json_safe_info(info),
                panel_width=panel_width,
                panel_height=panel_height,
            )
            frames.append(frame)
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
            if terminated or truncated:
                break
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()

    gif_path = output_dir / "picklift_policy_multiview.gif"
    mp4_path = output_dir / "picklift_policy_multiview.mp4"
    imageio.mimsave(gif_path, frames, duration=1.0 / max(1, fps))
    mp4_error = ""
    try:
        imageio.mimsave(mp4_path, frames, fps=fps, macro_block_size=8)
    except Exception as exc:  # pragma: no cover
        mp4_error = str(exc)
        mp4_path = Path("")
    manifest = {
        "operation": "render_so101_visual_picklift_bc_rollout",
        "checkpoint_path": str(checkpoint_path),
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
    manifest_path = output_dir / "picklift_policy_rollout_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _compose_picklift_frame(
    *,
    views: dict[str, np.ndarray],
    step: int,
    action: np.ndarray,
    info: dict[str, Any],
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    labels = ["scene_3d", "wrist_cam", "egocentric_cam", "top_down"]
    telemetry_h = 112
    canvas = Image.new("RGB", (panel_width * 2, panel_height * 2 + telemetry_h), (245, 245, 240))
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(labels):
        x = (index % 2) * panel_width
        y = (index // 2) * panel_height
        canvas.paste(Image.fromarray(views[label]).convert("RGB"), (x, y))
        draw.rectangle((x, y, x + panel_width - 1, y + 24), fill=(245, 245, 240))
        draw.text((x + 10, y + 7), label, fill=(20, 20, 20))
    y0 = panel_height * 2 + 12
    draw.text(
        (16, y0),
        (
            f"step {step:03d}  success {bool(info.get('success', False))}  "
            f"grasped {info.get('is_grasped', 0.0)}  lift {float(info.get('lift_height', 0.0)):.4f}m"
        ),
        fill=(25, 25, 25),
    )
    draw.text(
        (16, y0 + 30),
        f"tcp_to_obj {float(info.get('tcp_to_obj_dist', 0.0)):.4f}m    action "
        + ", ".join(f"{float(value):+.2f}" for value in action[:6]),
        fill=(25, 25, 25),
    )
    return np.asarray(canvas)


def _plot_curve(values: list[float], path: Path, *, title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(values) + 1), values, marker="o", linewidth=1.5)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_bars(episodes: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.arange(len(episodes))
    success = [1.0 if item["success"] else 0.0 for item in episodes]
    lift = [float(item["final_lift_height"]) for item in episodes]
    plt.figure(figsize=(8, 4))
    plt.bar(xs - 0.2, success, width=0.4, label="success")
    plt.bar(xs + 0.2, lift, width=0.4, label="final lift height (m)")
    plt.xlabel("eval episode")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _mean(values: list[Any]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


if __name__ == "__main__":
    main()
