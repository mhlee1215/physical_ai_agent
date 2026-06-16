#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.sim.so101_camera_input import _make_camera, postprocess_camera_frame
from physical_ai_agent.sim.so101_live_viewer import _cartesian_error_controller_action
from physical_ai_agent.sim.so101_visual_rl import SO101VisualRLConfig, _json_safe_info
from train_so101_visual_picklift_bc import _plot_bars, _plot_curve, _resolve_device, _solve_pregrasp_qpos
from train_so101_visual_picklift_delta import _object_set_description, make_diverse_picklift_env
from evaluate_so101_picklift_image_policy import detect_colored_object


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate SO101 PickLift from wrist_cam + egocentric_cam only."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_visual_rl/wrist_ego_picklift"))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=21400)
    parser.add_argument("--render-seed", type=int, default=21400)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    report = train_and_evaluate(
        output_dir=args.output_dir,
        samples=args.samples,
        epochs=args.epochs,
        eval_episodes=args.eval_episodes,
        steps=args.steps,
        seed=args.seed,
        render_seed=args.render_seed,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        fps=args.fps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def train_and_evaluate(
    *,
    output_dir: Path,
    samples: int,
    epochs: int,
    eval_episodes: int,
    steps: int,
    seed: int,
    render_seed: int,
    width: int,
    height: int,
    batch_size: int,
    lr: float,
    device: str,
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
        camera_name="wrist_cam",
        width=width,
        height=height,
        include_state=False,
        channel_first=True,
    )
    dataset = collect_teacher_dataset(config=config, samples=samples, seed=seed)
    model = WristEgoPickLiftPolicy(image_shape=(6, height, width), hidden_dim=192).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    image_tensor = torch.as_tensor(dataset["images"], dtype=torch.uint8)
    target_tensor = torch.as_tensor(
        normalize_targets(dataset["targets"], dataset["action_low"], dataset["action_high"]),
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    for _epoch in range(epochs):
        order = torch.randperm(len(target_tensor), generator=generator)
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            images = image_tensor[idx].to(resolved_device)
            targets = target_tensor[idx].to(resolved_device)
            pred = model(images)
            loss = torch.nn.functional.mse_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))

    checkpoint_path = output_dir / "wrist_ego_picklift_policy.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model.cpu(),
        metadata={
            "operation": "train_so101_wrist_ego_picklift_policy",
            "runtime_inputs": ["wrist_cam", "egocentric_cam"],
            "runtime_excludes": ["top_down", "camera_calibration", "object_pose", "mujoco_jacobian"],
            "object_set": _object_set_description(),
            "samples_requested": samples,
            "samples_used": int(len(dataset["targets"])),
            "teacher_success_rate": dataset["teacher_success_rate"],
            "width": width,
            "height": height,
            "epochs": epochs,
            "device": resolved_device,
        },
    )
    eval_report = evaluate_policy(
        checkpoint_path=checkpoint_path,
        config=config,
        episodes=eval_episodes,
        steps=steps,
        seed=seed + 10_000,
    )
    rollout = render_policy_rollout(
        checkpoint_path=checkpoint_path,
        config=config,
        seed=render_seed,
        steps=steps,
        fps=fps,
        output_dir=videos_dir,
    )
    loss_plot = plots_dir / "wrist_ego_loss_curve.png"
    success_plot = plots_dir / "wrist_ego_success_curve.png"
    _plot_curve(losses, loss_plot, title="Wrist+ego PickLift target loss", ylabel="MSE")
    _plot_bars(eval_report["episodes"], success_plot)
    manifest = {
        "operation": "train_so101_wrist_ego_picklift_policy",
        "runtime_inputs": ["wrist_cam", "egocentric_cam"],
        "runtime_excludes": ["top_down", "camera_calibration", "object_pose", "mujoco_jacobian"],
        "object_set": _object_set_description(),
        "checkpoint_path": str(checkpoint_path),
        "dataset": {
            "samples_requested": samples,
            "samples_used": int(len(dataset["targets"])),
            "teacher_success_rate": dataset["teacher_success_rate"],
        },
        "training": {"losses": losses, "final_loss": losses[-1] if losses else None},
        "evaluation": eval_report,
        "artifacts": {
            "loss_plot": str(loss_plot),
            "success_plot": str(success_plot),
            "rollout_gif": rollout["gif_path"],
            "rollout_mp4": rollout["mp4_path"],
            "rollout_manifest": rollout["manifest_path"],
        },
    }
    manifest_path = output_dir / "wrist_ego_policy_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


class WristEgoPickLiftPolicy:
    def __new__(cls, *, image_shape: tuple[int, int, int], hidden_dim: int = 192) -> Any:
        import torch
        from torch import nn

        class _Policy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                channels, _height, _width = image_shape
                self.encoder = nn.Sequential(
                    nn.Conv2d(channels, 24, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                self.head = nn.Sequential(
                    nn.Linear(96, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
            nn.Linear(hidden_dim, 12),
            nn.Tanh(),
                )

            def forward(self, image: Any) -> Any:
                image = image.float() / 255.0
                return self.head(self.encoder(image))

        return _Policy()


def collect_teacher_dataset(*, config: SO101VisualRLConfig, samples: int, seed: int) -> dict[str, Any]:
    import mujoco

    env = make_diverse_picklift_env(config)
    renderers = {
        "wrist_cam": mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width),
        "egocentric_cam": mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width),
    }
    images = []
    targets = []
    teacher_successes = []
    visible_successes = []
    dropped_no_visible = 0
    try:
        for index in range(samples):
            env.reset(seed=seed + index)
            visible, _search_steps = sweep_until_visible(env, renderers, max_sweeps=48)
            if not visible:
                dropped_no_visible += 1
                visible_successes.append(False)
                continue
            visible_successes.append(True)
            try:
                q_open, q_lift, teacher_success = make_teacher_targets(env)
            except Exception:
                teacher_successes.append(False)
                continue
            if not teacher_success:
                teacher_successes.append(False)
                continue
            image = render_wrist_ego(env, renderers)
            images.append(image)
            targets.append(np.concatenate([q_open, q_lift]).astype(np.float32))
            teacher_successes.append(True)
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()
    if not targets:
        raise RuntimeError("No successful teacher samples collected")
    return {
        "images": np.stack(images, axis=0),
        "targets": np.stack(targets, axis=0),
        "action_low": np.tile(np.asarray(env.action_space.low, dtype=np.float32), 2),
        "action_high": np.tile(np.asarray(env.action_space.high, dtype=np.float32), 2),
        "teacher_success_rate": float(np.mean(teacher_successes)) if teacher_successes else 0.0,
        "visible_after_sweep_rate": float(np.mean(visible_successes)) if visible_successes else 0.0,
        "dropped_no_visible_object": int(dropped_no_visible),
    }


def make_teacher_targets(env: Any) -> tuple[np.ndarray, np.ndarray, bool]:
    q_open = _solve_pregrasp_qpos(env)
    q_open[-1] = 0.25
    q_close = q_open.copy()
    q_close[-1] = float(env.action_space.low[-1])
    info: dict[str, Any] = {}
    q_lift = q_close.copy()
    for step in range(180):
        if step < 58:
            action = q_open
        elif step < 118:
            action = q_close
        else:
            action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=float)
            action[-1] = q_close[-1]
            q_lift = np.clip(action, env.action_space.low, env.action_space.high)
        _obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return q_open.astype(float), q_lift.astype(float), bool(info.get("success", False))


def render_wrist_ego(env: Any, renderers: dict[str, Any]) -> np.ndarray:
    frames = []
    for camera_name in ("wrist_cam", "egocentric_cam"):
        renderer = renderers[camera_name]
        renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, camera_name))
        pixels = postprocess_camera_frame(camera_name, renderer.render())
        frames.append(pixels.transpose(2, 0, 1))
    return np.concatenate(frames, axis=0).astype(np.uint8)


def object_visible_in_wrist_ego(env: Any, renderers: dict[str, Any]) -> bool:
    for camera_name in ("wrist_cam", "egocentric_cam"):
        renderer = renderers[camera_name]
        renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, camera_name))
        pixels = postprocess_camera_frame(camera_name, renderer.render())
        if detect_colored_object(pixels) is not None:
            return True
    return False


def sweep_until_visible(env: Any, renderers: dict[str, Any], *, max_sweeps: int) -> tuple[bool, int]:
    if object_visible_in_wrist_ego(env, renderers):
        return True, 0
    for step in range(max_sweeps):
        action = sweep_action(env, step)
        env.step(action)
        if object_visible_in_wrist_ego(env, renderers):
            return True, step + 1
    return False, max_sweeps


def sweep_action(env: Any, step: int) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=float)
    high = np.asarray(env.action_space.high, dtype=float)
    base = np.asarray(env.unwrapped.data.ctrl[env.unwrapped._actuator_ids], dtype=float).copy()
    phase = (step % 48) / 47.0
    shoulder_pan = -0.75 + 1.50 * phase
    base[0] = shoulder_pan
    base[1] = -1.15
    base[2] = 1.15
    base[3] = 0.55
    base[4] = 0.0
    base[5] = 0.45
    return np.clip(base, low, high)


def evaluate_policy(
    *,
    checkpoint_path: Path,
    config: SO101VisualRLConfig,
    episodes: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    import mujoco
    import torch

    model, metadata = load_checkpoint(checkpoint_path)
    env = make_diverse_picklift_env(config)
    renderers = {
        "wrist_cam": mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width),
        "egocentric_cam": mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width),
    }
    reports = []
    dropped = 0
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            visible, search_steps = sweep_until_visible(env, renderers, max_sweeps=48)
            if not visible:
                dropped += 1
                reports.append(
                    {
                        "episode": episode,
                        "seed": seed + episode,
                        "success": False,
                        "dropped": True,
                        "drop_reason": "dropped_no_visible_object",
                        "search_steps": search_steps,
                        "steps_to_success": None,
                        "final_is_grasped": 0.0,
                        "final_lift_height": 0.0,
                        "final_tcp_to_obj_dist": float("nan"),
                    }
                )
                continue
            image = torch.as_tensor(render_wrist_ego(env, renderers)).unsqueeze(0)
            with torch.no_grad():
                normalized = model(image).cpu().numpy()[0].astype(float)
            target = denormalize_targets(
                normalized,
                np.tile(np.asarray(env.action_space.low, dtype=float), 2),
                np.tile(np.asarray(env.action_space.high, dtype=float), 2),
            )
            result = run_predicted_targets_episode(env=env, q_open=target[:6], q_lift=target[6:], steps=steps)
            result.pop("records", None)
            result["search_steps"] = search_steps
            result["dropped"] = False
            reports.append({"episode": episode, "seed": seed + episode, **result})
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()
    success_steps = [item["steps_to_success"] for item in reports if item["steps_to_success"] is not None]
    return {
        "episodes": reports,
        "success_rate": float(np.mean([item["success"] for item in reports])),
        "mean_final_lift_height": float(np.mean([item["final_lift_height"] for item in reports])),
        "mean_steps_to_success": float(np.mean(success_steps)) if success_steps else None,
        "dropped_no_visible_object": dropped,
        "attempted_episodes": episodes - dropped,
        "metadata": metadata,
    }


def run_predicted_targets_episode(
    *,
    env: Any,
    q_open: np.ndarray,
    q_lift: np.ndarray,
    steps: int,
) -> dict[str, Any]:
    q_open = np.clip(np.asarray(q_open, dtype=float), env.action_space.low, env.action_space.high)
    q_open[-1] = 0.25
    q_close = q_open.copy()
    q_close[-1] = float(env.action_space.low[-1])
    q_lift = np.clip(np.asarray(q_lift, dtype=float), env.action_space.low, env.action_space.high)
    q_lift[-1] = q_close[-1]
    info: dict[str, Any] = {}
    success_step = None
    records = []
    for step in range(steps):
        if step < 58:
            action = q_open
        elif step < 118:
            action = q_close
        else:
            action = q_lift
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
        "records": records,
    }


def render_policy_rollout(
    *,
    checkpoint_path: Path,
    config: SO101VisualRLConfig,
    seed: int,
    steps: int,
    fps: int,
    output_dir: Path,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import mujoco
    import torch
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    model, _metadata = load_checkpoint(checkpoint_path)
    env = make_diverse_picklift_env(config)
    renderers = {
        "wrist_input": mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width),
        "ego_input": mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width),
        "wrist_cam": mujoco.Renderer(env.unwrapped.model, height=360, width=480),
        "egocentric_cam": mujoco.Renderer(env.unwrapped.model, height=360, width=480),
        "scene_3d": mujoco.Renderer(env.unwrapped.model, height=360, width=480),
    }
    frames = []
    try:
        env.reset(seed=seed)
        search_frames = []
        visible = object_visible_in_wrist_ego(
            env,
            {"wrist_cam": renderers["wrist_input"], "egocentric_cam": renderers["ego_input"]},
        )
        for search_step in range(48):
            if visible:
                break
            action = sweep_action(env, search_step)
            views = {}
            for name in ("wrist_cam", "egocentric_cam", "scene_3d"):
                renderer = renderers[name]
                if name == "scene_3d":
                    renderer.update_scene(env.unwrapped.data)
                else:
                    renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, name))
                views[name] = postprocess_camera_frame(name, renderer.render())
            frames.append(compose_wrist_ego_frame(views=views, step=search_step, action=action, info=env.unwrapped._get_info(), mode="search"))
            env.step(action)
            visible = object_visible_in_wrist_ego(
                env,
                {"wrist_cam": renderers["wrist_input"], "egocentric_cam": renderers["ego_input"]},
            )
        if not visible:
            records = []
            q_open = np.asarray(env.unwrapped.data.ctrl[env.unwrapped._actuator_ids], dtype=float)
            q_close = q_open.copy()
            q_lift = q_open.copy()
        else:
            records = []
            image = torch.as_tensor(
                render_wrist_ego(env, {"wrist_cam": renderers["wrist_input"], "egocentric_cam": renderers["ego_input"]})
            ).unsqueeze(0)
            with torch.no_grad():
                normalized = model(image).cpu().numpy()[0].astype(float)
            target = denormalize_targets(
                normalized,
                np.tile(np.asarray(env.action_space.low, dtype=float), 2),
                np.tile(np.asarray(env.action_space.high, dtype=float), 2),
            )
            q_open = target[:6]
            q_lift = target[6:]
        q_open = np.clip(q_open, env.action_space.low, env.action_space.high)
        q_open[-1] = 0.25
        q_close = q_open.copy()
        q_close[-1] = float(env.action_space.low[-1])
        q_lift = np.clip(q_lift, env.action_space.low, env.action_space.high)
        q_lift[-1] = q_close[-1]
        info = env.unwrapped._get_info()
        for step in range(steps):
            if step < 58:
                action = q_open
            elif step < 118:
                action = q_close
            else:
                action = q_lift
            views = {}
            for name in ("wrist_cam", "egocentric_cam", "scene_3d"):
                renderer = renderers[name]
                if name == "scene_3d":
                    renderer.update_scene(env.unwrapped.data)
                else:
                    renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, name))
                views[name] = postprocess_camera_frame(name, renderer.render())
            frames.append(compose_wrist_ego_frame(views=views, step=step, action=action, info=info, mode="act"))
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
    gif_path = output_dir / "wrist_ego_picklift_policy.gif"
    mp4_path = output_dir / "wrist_ego_picklift_policy.mp4"
    imageio.mimsave(gif_path, frames, duration=1.0 / max(1, fps))
    mp4_error = ""
    try:
        imageio.mimsave(mp4_path, frames, fps=fps, macro_block_size=8)
    except Exception as exc:  # pragma: no cover
        mp4_error = str(exc)
        mp4_path = Path("")
    manifest = {
        "operation": "render_wrist_ego_picklift_policy_rollout",
        "seed": seed,
        "runtime_inputs": ["wrist_cam", "egocentric_cam"],
        "steps": len(records),
        "success": any(record["success"] for record in records),
        "final_is_grasped": records[-1]["is_grasped"] if records else 0.0,
        "final_lift_height": records[-1]["lift_height"] if records else 0.0,
        "gif_path": str(gif_path),
        "mp4_path": str(mp4_path) if mp4_path else "",
        "mp4_error": mp4_error,
        "records": records,
    }
    manifest_path = output_dir / "wrist_ego_picklift_policy_rollout_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def compose_wrist_ego_frame(*, views: dict[str, np.ndarray], step: int, action: np.ndarray, info: dict[str, Any], mode: str = "act") -> np.ndarray:
    from PIL import Image, ImageDraw

    panel_w = 480
    panel_h = 360
    telemetry_h = 112
    canvas = Image.new("RGB", (panel_w * 2, panel_h * 2 + telemetry_h), (245, 245, 240))
    draw = ImageDraw.Draw(canvas)
    placements = [
        ("wrist_cam", 0, 0),
        ("egocentric_cam", panel_w, 0),
        ("scene_3d", 0, panel_h),
    ]
    for label, x, y in placements:
        canvas.paste(Image.fromarray(views[label]).convert("RGB"), (x, y))
        draw.rectangle((x, y, x + panel_w - 1, y + 24), fill=(245, 245, 240))
        draw.text((x + 10, y + 7), label, fill=(20, 20, 20))
    draw.rectangle((panel_w, panel_h, panel_w * 2 - 1, panel_h * 2 - 1), fill=(232, 232, 226))
    draw.text((panel_w + 16, panel_h + 20), f"mode: {mode}", fill=(25, 25, 25))
    draw.text((panel_w + 16, panel_h + 48), "runtime inputs: wrist_cam + egocentric_cam only", fill=(25, 25, 25))
    draw.text((panel_w + 16, panel_h + 76), "no top_down, no camera calibration, no object pose", fill=(25, 25, 25))
    y0 = panel_h * 2 + 12
    draw.text(
        (16, y0),
        f"step {step:03d}  success {bool(info.get('success', False))}  grasped {info.get('is_grasped', 0.0)}  lift {float(info.get('lift_height', 0.0)):.4f}m",
        fill=(25, 25, 25),
    )
    draw.text(
        (16, y0 + 30),
        "action " + ", ".join(f"{float(value):+.2f}" for value in action[:6]),
        fill=(25, 25, 25),
    )
    return np.asarray(canvas)


def save_checkpoint(*, path: Path, model: Any, metadata: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def normalize_targets(targets: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    midpoint = (high + low) * 0.5
    half_range = np.maximum((high - low) * 0.5, 1e-6)
    return np.clip((targets - midpoint) / half_range, -1.0, 1.0).astype(np.float32)


def denormalize_targets(target: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    midpoint = (high + low) * 0.5
    half_range = (high - low) * 0.5
    return midpoint + np.clip(target, -1.0, 1.0) * half_range


def load_checkpoint(path: Path) -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {})
    model = WristEgoPickLiftPolicy(
        image_shape=(6, int(metadata.get("height", 96)), int(metadata.get("width", 96))),
        hidden_dim=192,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, metadata


if __name__ == "__main__":
    main()
