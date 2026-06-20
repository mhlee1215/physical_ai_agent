#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.policies.so101_visual_reach_delta import (
    make_so101_visual_reach_delta,
    save_so101_visual_reach_delta_checkpoint,
)
from physical_ai_agent.sim.so101_live_viewer import _cartesian_error_controller_action
from physical_ai_agent.sim.so101_visual_rl import SO101VisualRLConfig, _json_safe_info, make_so101_visual_rl_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a visual target-error estimator for controller-prior SO101 Reach."
    )
    parser.add_argument("--env-id", default="MuJoCoReach-v1")
    parser.add_argument("--camera-name", default="top_down")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--reset-interval", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--video-steps", type=int, default=100)
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_visual_rl/reach_delta"))
    args = parser.parse_args()

    report = train_visual_reach_delta(
        config=SO101VisualRLConfig(
            env_id=args.env_id,
            camera_name=args.camera_name,
            width=args.width,
            height=args.height,
            include_state=True,
            channel_first=True,
        ),
        output_dir=args.output_dir,
        samples=args.samples,
        reset_interval=args.reset_interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.lr,
        device=args.device,
        eval_episodes=args.eval_episodes,
        eval_steps=args.eval_steps,
        eval_every=args.eval_every,
        video_steps=args.video_steps,
        video_fps=args.video_fps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def train_visual_reach_delta(
    *,
    config: SO101VisualRLConfig,
    output_dir: Path,
    samples: int,
    reset_interval: int,
    epochs: int,
    batch_size: int,
    seed: int,
    lr: float,
    device: str,
    eval_episodes: int,
    eval_steps: int,
    eval_every: int,
    video_steps: int,
    video_fps: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = _resolve_torch_device(device)
    print(f"training torch device: {torch_device}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    plots_dir = output_dir / "plots"
    checkpoints_dir = output_dir / "checkpoints"
    videos_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    env = make_so101_visual_rl_env(config)
    model = make_so101_visual_reach_delta(observation_space=env.observation_space).to(torch_device)
    images = []
    states = []
    labels = []
    collect_records = []
    reset_count = 0
    obs, _info = env.reset(seed=seed)
    try:
        for index in range(samples):
            label = _reach_error(env)
            images.append(np.asarray(obs["image"], dtype=np.uint8))
            states.append(np.asarray(obs["state"], dtype=np.float32))
            labels.append(label.astype(np.float32))
            action = _cartesian_error_controller_action(env, label)
            obs, reward, terminated, truncated, info = env.step(action)
            if index == 0 or (index + 1) % max(1, samples // 10) == 0:
                record = {"sample": index + 1, "reward": float(reward), "info": _json_safe_info(info)}
                collect_records.append(record)
                print(
                    f"collect {index + 1}/{samples} reward={record['reward']:.4f} "
                    f"dist={record['info'].get('tcp_to_target_dist')}",
                    flush=True,
                )
            if terminated or truncated or (reset_interval > 0 and (index + 1) % reset_interval == 0):
                reset_count += 1
                obs, _info = env.reset(seed=seed + reset_count)

        image_tensor = torch.as_tensor(np.stack(images), dtype=torch.uint8, device=torch_device)
        state_tensor = torch.as_tensor(np.stack(states), dtype=torch.float32, device=torch_device)
        label_tensor = torch.as_tensor(np.stack(labels), dtype=torch.float32, device=torch_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        generator = torch.Generator().manual_seed(seed)
        history = []
        eval_history = []
        grad_norm = torch.tensor(0.0)
        for epoch in range(epochs):
            permutation = torch.randperm(samples, generator=generator)
            losses = []
            for start in range(0, samples, batch_size):
                index = permutation[start : start + batch_size].to(torch_device)
                pred = model({"image": image_tensor[index], "state": state_tensor[index]})
                loss = F.mse_loss(pred, label_tensor[index])
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append(
                {
                    "epoch": epoch,
                    "mse": float(np.mean(losses)),
                    "rmse_m": float(np.sqrt(np.mean(losses))),
                    "grad_norm": float(grad_norm.detach().cpu()),
                }
            )
            print(
                f"epoch {epoch + 1}/{epochs} mse={history[-1]['mse']:.6f} "
                f"rmse_m={history[-1]['rmse_m']:.4f}",
                flush=True,
            )
            should_eval = epoch == 0 or (eval_every > 0 and (epoch + 1) % eval_every == 0) or epoch == epochs - 1
            if should_eval:
                eval_at_epoch = _evaluate_delta_policy(
                    model=model,
                    config=config,
                    episodes=eval_episodes,
                    steps=eval_steps,
                    seed=seed + 30_000 + epoch,
                    video_path=videos_dir / f"epoch_{epoch + 1:03d}.gif",
                    video_steps=video_steps,
                    video_fps=video_fps,
                )
                eval_history.append(
                    {
                        "epoch": epoch,
                        "mean_final_distance": eval_at_epoch["mean_final_distance"],
                        "mean_distance_delta": eval_at_epoch["mean_distance_delta"],
                        "success_rate": eval_at_epoch["success_rate"],
                        "video_path": eval_at_epoch.get("video_path", ""),
                    }
                )
                save_so101_visual_reach_delta_checkpoint(
                    path=checkpoints_dir / f"epoch_{epoch + 1:03d}.pt",
                    model=model,
                    observation_space=env.observation_space,
                    metadata={
                        "operation": "train_so101_visual_reach_delta_checkpoint",
                        "config": asdict(config),
                        "epoch": epoch,
                        "eval": eval_at_epoch,
                    },
                )
                _write_training_plots(history=history, eval_history=eval_history, output_dir=plots_dir)
                print(
                    f"eval epoch {epoch + 1}: final_dist={eval_at_epoch['mean_final_distance']:.4f}m "
                    f"delta={eval_at_epoch['mean_distance_delta']:.4f}m "
                    f"success={eval_at_epoch['success_rate']:.2f}",
                    flush=True,
                )

        eval_report = _evaluate_delta_policy(
            model=model,
            config=config,
            episodes=eval_episodes,
            steps=eval_steps,
            seed=seed + 30_000,
            video_path=videos_dir / "final.gif",
            video_steps=video_steps,
            video_fps=video_fps,
        )
    finally:
        env.close()

    checkpoint_path = output_dir / "so101_visual_reach_delta.pt"
    manifest_path = output_dir / "training_manifest.json"
    metadata = {
        "operation": "train_so101_visual_reach_delta",
        "control": "visual_delta_to_jacobian_controller",
        "config": asdict(config),
        "seed": seed,
        "samples": samples,
        "reset_interval": reset_interval,
        "reset_count": reset_count,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(torch_device),
        "collect_records": collect_records,
        "history": history,
        "eval": eval_report,
        "eval_history": eval_history,
        "plots": {
            "loss_curve": str(plots_dir / "loss_curve.png"),
            "eval_curve": str(plots_dir / "eval_curve.png"),
        },
        "videos_dir": str(videos_dir),
    }
    save_so101_visual_reach_delta_checkpoint(
        path=checkpoint_path,
        model=model,
        observation_space=env.observation_space,
        metadata=metadata,
    )
    manifest = {**metadata, "checkpoint_path": str(checkpoint_path), "manifest_path": str(manifest_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _evaluate_delta_policy(
    *,
    model: Any,
    config: SO101VisualRLConfig,
    episodes: int,
    steps: int,
    seed: int,
    video_path: Path | None = None,
    video_steps: int = 100,
    video_fps: int = 12,
) -> dict[str, Any]:
    env = make_so101_visual_rl_env(config)
    records = []
    video_frames = []
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            initial = float(np.linalg.norm(_reach_error(env)))
            rewards = []
            predicted_error_norms = []
            last_info = _json_safe_info(info)
            for _step in range(steps):
                if episode == 0 and video_path is not None and len(video_frames) < video_steps:
                    video_frames.append(
                        _annotated_frame(
                            obs=obs,
                            step=len(video_frames),
                            distance=float(np.linalg.norm(_reach_error(env))),
                            predicted_error_norm=(
                                predicted_error_norms[-1] if predicted_error_norms else 0.0
                            ),
                        )
                    )
                pred_error = _predict_error(model, obs)
                predicted_error_norms.append(float(np.linalg.norm(pred_error)))
                action = _cartesian_error_controller_action(env, pred_error)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                last_info = _json_safe_info(info)
                if terminated or truncated:
                    break
            final = float(np.linalg.norm(_reach_error(env)))
            records.append(
                {
                    "episode": episode,
                    "steps": len(rewards),
                    "initial_distance": initial,
                    "final_distance": final,
                    "distance_delta": initial - final,
                    "mean_predicted_error_norm": float(np.mean(predicted_error_norms)),
                    "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                    "success": bool(last_info.get("success", False)),
                    "last_info": last_info,
                }
            )
    finally:
        env.close()
    video_result = ""
    if video_path is not None and video_frames:
        import imageio.v2 as imageio

        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, video_frames, duration=1.0 / max(1, video_fps))
        video_result = str(video_path)
    return {
        "episodes": records,
        "mean_initial_distance": float(np.mean([r["initial_distance"] for r in records])),
        "mean_final_distance": float(np.mean([r["final_distance"] for r in records])),
        "mean_distance_delta": float(np.mean([r["distance_delta"] for r in records])),
        "success_rate": float(np.mean([r["success"] for r in records])),
        "video_path": video_result,
    }


def _predict_error(model: Any, obs: dict[str, Any]) -> Any:
    import torch

    with torch.no_grad():
        pred = model(obs).detach().cpu().numpy()[0]
    return np.asarray(pred, dtype=float)


def _annotated_frame(
    *,
    obs: dict[str, Any],
    step: int,
    distance: float,
    predicted_error_norm: float,
) -> Any:
    from PIL import Image, ImageDraw

    image = np.asarray(obs["image"], dtype=np.uint8).transpose(1, 2, 0)
    frame = Image.fromarray(image).convert("RGB").resize((320, 320))
    canvas = Image.new("RGB", (320, 380), (245, 245, 240))
    canvas.paste(frame, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 320, 320, 380), fill=(245, 245, 240))
    draw.text((10, 330), f"step {step:03d} dist {distance:.3f}m", fill=(25, 25, 25))
    draw.text((10, 354), f"pred |delta| {predicted_error_norm:.3f}m", fill=(25, 25, 25))
    return np.asarray(canvas)


def _write_training_plots(
    *,
    history: list[dict[str, Any]],
    eval_history: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] + 1 for item in history]
    losses = [item["mse"] for item in history]
    rmses = [item["rmse_m"] for item in history]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs, losses, label="MSE", color="#1f77b4")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("MSE")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(epochs, rmses, label="RMSE m", color="#ff7f0e")
    ax2.set_ylabel("RMSE (m)")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=140)
    plt.close(fig)

    if eval_history:
        eval_epochs = [item["epoch"] + 1 for item in eval_history]
        final_dist = [item["mean_final_distance"] for item in eval_history]
        success = [item["success_rate"] for item in eval_history]
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(eval_epochs, final_dist, marker="o", label="final distance", color="#2ca02c")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("mean final distance (m)")
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(eval_epochs, success, marker="s", label="success rate", color="#d62728")
        ax2.set_ylabel("success rate")
        fig.tight_layout()
        fig.savefig(output_dir / "eval_curve.png", dpi=140)
        plt.close(fig)


def _reach_error(env: Any) -> Any:
    model = env.unwrapped.model
    data = env.unwrapped.data
    target = data.site_xpos[model.site("reach_target").id]
    gripper = data.site_xpos[model.site("gripperframe").id]
    return np.asarray(target - gripper, dtype=float)


def _resolve_torch_device(requested: str) -> Any:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is false")
    return torch.device(requested)


if __name__ == "__main__":
    main()
