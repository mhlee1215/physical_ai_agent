#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from train_so101_wrist_ego_picklift_policy import sweep_until_visible
from train_so101_wrist_ego_visual_servo import (
    WristEgoServoConfig,
    _current_qpos,
    _make_policy_renderers,
    make_high_contrast_picklift_env,
)


TASK = "Grasp the visible cube and lift it up."


@dataclass(frozen=True)
class LeroBotVisualBCConfig:
    width: int = 96
    height: int = 96
    hidden_dim: int = 256
    spatial_pool_size: int = 3
    image_channels: int = 6
    state_dim: int = 7
    action_dim: int = 6


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a deterministic wrist+egocentric visual BC policy on a SO101 LeRobotDataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_visual_rl/lerobot_visual_bc"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=98000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=160)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda"])
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--spatial-pool-size", type=int, default=3)
    parser.add_argument("--close-phase-weight", type=float, default=1.0)
    parser.add_argument("--lift-phase-weight", type=float, default=1.0)
    parser.add_argument("--close-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lift-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-rollout-gif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--max-action-delta", type=float, default=0.0)
    parser.add_argument("--episode-limit", type=int, default=0)
    parser.add_argument("--target-mode", choices=["absolute", "arm_delta"], default="absolute")
    args = parser.parse_args()

    report = train_lerobot_visual_bc(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        eval_steps=args.eval_steps,
        device=args.device,
        config=LeroBotVisualBCConfig(width=args.width, height=args.height, spatial_pool_size=args.spatial_pool_size),
        close_phase_weight=args.close_phase_weight,
        lift_phase_weight=args.lift_phase_weight,
        close_gate=args.close_gate,
        lift_gate=args.lift_gate,
        record_rollout_gif=args.record_rollout_gif,
        gif_fps=args.gif_fps,
        load_checkpoint=args.load_checkpoint,
        max_action_delta=args.max_action_delta,
        episode_limit=args.episode_limit,
        target_mode=args.target_mode,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def train_lerobot_visual_bc(
    *,
    dataset_root: Path,
    repo_id: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    eval_episodes: int,
    eval_steps: int,
    device: str,
    config: LeroBotVisualBCConfig,
    close_phase_weight: float,
    lift_phase_weight: float,
    close_gate: bool,
    lift_gate: bool,
    record_rollout_gif: bool,
    gif_fps: int,
    load_checkpoint: Path | None,
    max_action_delta: float,
    episode_limit: int,
    target_mode: str,
) -> dict[str, Any]:
    import torch

    _require_device(torch, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    dataset = _load_lerobot_dataset(dataset_root=dataset_root, repo_id=repo_id, episode_limit=episode_limit)
    model = LeroBotVisualBCPolicy(config).to(device)
    images = torch.as_tensor(dataset["images"], dtype=torch.uint8)
    states = torch.as_tensor(dataset["states"], dtype=torch.float32)
    actions = torch.as_tensor(dataset["actions"], dtype=torch.float32)
    targets = torch.as_tensor(_training_targets(dataset=dataset, target_mode=target_mode), dtype=torch.float32)
    weights = torch.as_tensor(
        _phase_weights(
            states=dataset["states"],
            close_progress=float(dataset["metadata"]["close_progress"]),
            lift_progress=float(dataset["metadata"]["lift_progress"]),
            close_phase_weight=close_phase_weight,
            lift_phase_weight=lift_phase_weight,
        ),
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    losses: list[float] = []
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_path = load_checkpoint
        print(f"[lerobot-visual-bc] loaded checkpoint {load_checkpoint}", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        for epoch in range(1, epochs + 1):
            order = torch.randperm(len(actions), generator=generator)
            epoch_losses = []
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                pred = model(images[idx].to(device), states[idx].to(device))
                target = targets[idx].to(device)
                per_action_loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=0.02, reduction="none").mean(dim=1)
                loss = (per_action_loss * weights[idx].to(device)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(epoch_losses)))
            if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                print(f"[lerobot-visual-bc] epoch {epoch}/{epochs} loss={losses[-1]:.6f}", flush=True)

        checkpoint_path = output_dir / "so101_lerobot_visual_bc.pt"
        torch.save(
            {
                "model_state_dict": model.cpu().state_dict(),
                "config": asdict(config),
                "metadata": {
                    "operation": "train_so101_lerobot_visual_bc",
                    "dataset_root": str(dataset_root),
                    "repo_id": repo_id,
                    "episode_limit": episode_limit,
                    "target_mode": target_mode,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "device": device,
                    "task": TASK,
                    "state_features": ["qpos[0:6]", "episode_progress"],
                    "close_phase_weight": close_phase_weight,
                    "lift_phase_weight": lift_phase_weight,
                    "close_gate": close_gate,
                    "lift_gate": lift_gate,
                },
            },
            checkpoint_path,
        )
    model = model.to(device).eval()
    frame_audit = _audit_exact_training_frames(
        model=model,
        dataset=dataset,
        device=device,
        target_mode=target_mode,
    )
    eval_report = _evaluate_closed_loop(
        model=model,
        config=config,
        device=device,
        seed=seed,
        episodes=eval_episodes,
        steps=eval_steps,
        close_progress=float(dataset["metadata"]["close_progress"]),
        lift_progress=float(dataset["metadata"]["lift_progress"]),
        lift_target=np.asarray(dataset["metadata"]["lift_target"], dtype=float),
        close_gate=close_gate,
        lift_gate=lift_gate,
        output_dir=output_dir,
        record_rollout_gif=record_rollout_gif,
        gif_fps=gif_fps,
        max_action_delta=max_action_delta,
        target_mode=target_mode,
    )
    report = {
        "operation": "train_so101_lerobot_visual_bc",
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "checkpoint_path": str(checkpoint_path),
        "config": asdict(config),
        "device": device,
        "dataset": {
            "samples": int(len(actions)),
            "episode_limit": episode_limit,
            "action_min": np.min(dataset["actions"], axis=0).astype(float).tolist(),
            "action_max": np.max(dataset["actions"], axis=0).astype(float).tolist(),
            "metadata": dataset["metadata"],
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "close_phase_weight": close_phase_weight,
            "lift_phase_weight": lift_phase_weight,
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
        },
        "eval_config": {
            "close_gate": close_gate,
            "lift_gate": lift_gate,
            "record_rollout_gif": record_rollout_gif,
            "gif_fps": gif_fps,
            "max_action_delta": max_action_delta,
            "target_mode": target_mode,
        },
        "exact_training_frame_audit": frame_audit,
        "closed_loop_eval": eval_report,
        "duration_s": round(perf_counter() - started, 4),
    }
    report_path = output_dir / "training_manifest.json"
    report["manifest_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


class LeroBotVisualBCPolicy:
    def __new__(cls, config: LeroBotVisualBCConfig):
        import torch.nn as nn

        feature_grid_h = max(1, config.height // 16)
        feature_grid_w = max(1, config.width // 16)
        if config.spatial_pool_size > 0:
            image_feature_dim = 160 * config.spatial_pool_size * config.spatial_pool_size
            pool_or_flatten = [nn.AdaptiveAvgPool2d((config.spatial_pool_size, config.spatial_pool_size)), nn.Flatten()]
        else:
            image_feature_dim = 160 * feature_grid_h * feature_grid_w
            pool_or_flatten = [nn.Flatten()]

        class _Policy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(config.image_channels, 32, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 160, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    *pool_or_flatten,
                )
                self.head = nn.Sequential(
                    nn.Linear(image_feature_dim + config.state_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.action_dim),
                )

            def forward(self, image: Any, state: Any) -> Any:
                features = self.encoder(image.float() / 255.0)
                return self.head(__import__("torch").cat([features, state.float()], dim=1))

        return _Policy()


def _load_lerobot_dataset(*, dataset_root: Path, repo_id: str, episode_limit: int = 0) -> dict[str, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    images = []
    states = []
    actions = []
    loaded_indices = []
    for index in range(len(dataset)):
        sample = dataset[index]
        episode = int(np.asarray(sample.get("episode_index", [0])).reshape(-1)[0])
        if episode_limit > 0 and episode >= episode_limit:
            continue
        wrist = _image_chw_uint8(sample["observation.images.camera1"])
        ego = _image_chw_uint8(sample["observation.images.camera2"])
        images.append(np.concatenate([wrist, ego], axis=0))
        states.append(np.concatenate([np.asarray(sample["observation.state"], dtype=np.float32), [0.0]]))
        actions.append(np.asarray(sample["action"], dtype=np.float32))
        loaded_indices.append(index)
    if not actions:
        raise ValueError(f"no samples loaded from {dataset_root} with episode_limit={episode_limit}")
    state_array = np.stack(states, axis=0).astype(np.float32)
    _fill_episode_progress(dataset=dataset, states=state_array, loaded_indices=loaded_indices)
    return {
        "images": np.stack(images, axis=0),
        "states": state_array,
        "actions": np.stack(actions, axis=0),
        "metadata": _dataset_metadata(np.stack(actions, axis=0), state_array),
    }


def _dataset_metadata(actions: np.ndarray, states: np.ndarray) -> dict[str, Any]:
    gripper = actions[:, -1]
    midpoint = float((np.max(gripper) + np.min(gripper)) * 0.5)
    close_indices = np.flatnonzero(gripper < midpoint)
    close_index = int(close_indices[0]) if len(close_indices) else len(gripper)
    close_arm = actions[min(close_index, len(actions) - 1), :5]
    arm_after_close = np.linalg.norm(actions[:, :5] - close_arm[None, :], axis=1)
    lift_candidates = np.flatnonzero((np.arange(len(actions)) > close_index) & (arm_after_close > 0.02))
    lift_index = int(lift_candidates[0]) if len(lift_candidates) else max(close_index, len(actions) - 1)
    return {
        "gripper_open_value": float(np.max(gripper)),
        "gripper_closed_value": float(np.min(gripper)),
        "gripper_close_midpoint": midpoint,
        "first_close_index": close_index,
        "close_progress": float(states[min(close_index, len(states) - 1), -1]),
        "first_lift_index": lift_index,
        "lift_progress": float(states[min(lift_index, len(states) - 1), -1]),
        "lift_target": actions[-1].astype(float).tolist(),
    }


def _training_targets(*, dataset: dict[str, np.ndarray], target_mode: str) -> np.ndarray:
    actions = np.asarray(dataset["actions"], dtype=np.float32).copy()
    if target_mode == "absolute":
        return actions
    if target_mode == "arm_delta":
        actions[:, :5] = actions[:, :5] - np.asarray(dataset["states"], dtype=np.float32)[:, :5]
        return actions
    raise ValueError(f"unsupported target_mode: {target_mode}")


def _phase_weights(
    *,
    states: np.ndarray,
    close_progress: float,
    lift_progress: float,
    close_phase_weight: float,
    lift_phase_weight: float,
) -> np.ndarray:
    weights = np.ones((len(states),), dtype=np.float32)
    progress = states[:, -1]
    weights[progress >= close_progress] = float(close_phase_weight)
    weights[progress >= lift_progress] = float(lift_phase_weight)
    return weights


def _fill_episode_progress(*, dataset: Any, states: np.ndarray, loaded_indices: list[int]) -> None:
    episode_to_indices: dict[int, list[int]] = {}
    for loaded_index, dataset_index in enumerate(loaded_indices):
        sample = dataset[dataset_index]
        episode = int(np.asarray(sample.get("episode_index", [0])).reshape(-1)[0])
        episode_to_indices.setdefault(episode, []).append(loaded_index)
    for indices in episode_to_indices.values():
        denom = max(1, len(indices) - 1)
        for offset, index in enumerate(indices):
            states[index, -1] = float(offset / denom)


def _image_chw_uint8(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 3:
        array = np.transpose(array, (2, 0, 1))
    if array.dtype != np.uint8:
        if float(np.nanmax(array)) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _render_model_input(env: Any, renderers: dict[str, Any]) -> np.ndarray:
    wrist = _render_camera(env, renderers["wrist_cam"], "wrist_cam")
    ego = _render_camera(env, renderers["egocentric_cam"], "egocentric_cam")
    return np.concatenate([wrist, ego], axis=0)


def _render_camera(env: Any, renderer: Any, camera_name: str) -> np.ndarray:
    from physical_ai_agent.sim.so101_camera_input import _make_camera, postprocess_camera_frame

    renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, camera_name))
    image = postprocess_camera_frame(camera_name, renderer.render())
    return np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1))


def _predict(model: Any, image_chw: np.ndarray, state: np.ndarray, device: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        image = torch.as_tensor(image_chw[None], dtype=torch.uint8, device=device)
        qpos = torch.as_tensor(state[None], dtype=torch.float32, device=device)
        return np.asarray(model(image, qpos).detach().cpu()[0], dtype=float)


def _audit_exact_training_frames(*, model: Any, dataset: dict[str, np.ndarray], device: str, target_mode: str) -> dict[str, Any]:
    indices = sorted({0, 1, 2, len(dataset["actions"]) // 4, len(dataset["actions"]) // 2, len(dataset["actions"]) - 1})
    rows = []
    maes = []
    for index in indices:
        raw_pred = _predict(model, dataset["images"][index], dataset["states"][index], device)
        pred = _decode_prediction(raw_pred, state=dataset["states"][index], target_mode=target_mode)
        target = dataset["actions"][index].astype(float)
        mae = float(np.mean(np.abs(pred - target)))
        maes.append(mae)
        rows.append(
            {
                "index": int(index),
                "mae": mae,
                "target": target.tolist(),
                "pred": pred.tolist(),
                "raw_pred": raw_pred.tolist(),
            }
        )
    return {"mean_mae": float(np.mean(maes)), "max_mae": float(np.max(maes)), "rows": rows}


def _evaluate_closed_loop(
    *,
    model: Any,
    config: LeroBotVisualBCConfig,
    device: str,
    seed: int,
    episodes: int,
    steps: int,
    close_progress: float,
    lift_progress: float,
    lift_target: np.ndarray,
    close_gate: bool,
    lift_gate: bool,
    output_dir: Path,
    record_rollout_gif: bool,
    gif_fps: int,
    max_action_delta: float,
    target_mode: str,
) -> dict[str, Any]:
    servo_config = WristEgoServoConfig(width=config.width, height=config.height)
    env = make_high_contrast_picklift_env()
    renderers = _make_policy_renderers(env, servo_config)
    rows = []
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            visible, search_steps = sweep_until_visible(env, renderers, max_sweeps=servo_config.max_sweeps)
            records = []
            gif_frames = []
            info = env.unwrapped._get_info()
            previous_action = np.clip(_current_qpos(env).astype(float), env.action_space.low, env.action_space.high)
            if visible:
                for step in range(steps):
                    image = _render_model_input(env, renderers)
                    progress = float(step / max(1, steps - 1))
                    state = np.concatenate([_current_qpos(env).astype(np.float32), [progress]]).astype(np.float32)
                    raw_action = _predict(model, image, state, device)
                    action = _decode_prediction(raw_action, state=state, target_mode=target_mode)
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    if close_gate and progress >= close_progress:
                        action[-1] = float(env.action_space.low[-1])
                    if lift_gate and progress >= lift_progress:
                        action[:5] = lift_target[:5]
                        action[-1] = float(env.action_space.low[-1])
                    action = _rate_limit_action(action, previous_action, max_delta=max_action_delta)
                    previous_action = action.copy()
                    _obs, _reward, terminated, truncated, info = env.step(action)
                    if record_rollout_gif and (step % 2 == 0 or bool(info.get("success", False))):
                        gif_frames.append(_compose_rollout_frame(image, step=step, info=info))
                    records.append(
                        {
                            "step": step,
                            "action": [float(v) for v in action],
                            "raw_action": [float(v) for v in raw_action],
                            "success": bool(info.get("success", False)),
                            "is_grasped": float(info.get("is_grasped", 0.0)),
                            "lift_height": float(info.get("lift_height", 0.0)),
                            "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
                        }
                    )
                    if bool(info.get("success", False)) or terminated or truncated:
                        break
            gif_path = None
            if record_rollout_gif and gif_frames:
                videos_dir = output_dir / "videos"
                videos_dir.mkdir(parents=True, exist_ok=True)
                gif_path = videos_dir / f"rollout_episode_{episode:03d}.gif"
                duration_ms = max(20, int(1000 / max(1, gif_fps)))
                gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=duration_ms, loop=0)
            rows.append(
                {
                    "episode": episode,
                    "seed": seed + episode,
                    "visible": bool(visible),
                    "search_steps": int(search_steps),
                    "steps": len(records),
                    "success": bool(info.get("success", False)),
                    "final_is_grasped": float(info.get("is_grasped", 0.0)),
                    "final_lift_height": float(info.get("lift_height", 0.0)),
                    "final_tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
                    "rollout_gif": str(gif_path) if gif_path is not None else None,
                    "records": records,
                }
            )
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()
    return {
        "episodes": rows,
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "grasp_rate": float(np.mean([row["final_is_grasped"] > 0.5 for row in rows])) if rows else 0.0,
    }


def _decode_prediction(raw_prediction: np.ndarray, *, state: np.ndarray, target_mode: str) -> np.ndarray:
    action = np.asarray(raw_prediction, dtype=float).copy()
    if target_mode == "absolute":
        return action
    if target_mode == "arm_delta":
        action[:5] = np.asarray(state, dtype=float)[:5] + action[:5]
        return action
    raise ValueError(f"unsupported target_mode: {target_mode}")


def _rate_limit_action(action: np.ndarray, previous_action: np.ndarray, *, max_delta: float) -> np.ndarray:
    if max_delta <= 0:
        return action
    limited = action.copy()
    limited[:5] = np.clip(action[:5], previous_action[:5] - max_delta, previous_action[:5] + max_delta)
    return limited


def _compose_rollout_frame(image_chw: np.ndarray, *, step: int, info: dict[str, Any]) -> Any:
    from PIL import Image, ImageDraw

    wrist = _chw_to_pil(image_chw[:3]).resize((320, 320))
    ego = _chw_to_pil(image_chw[3:6]).resize((320, 320))
    canvas = Image.new("RGB", (640, 380), (20, 22, 24))
    canvas.paste(wrist, (0, 0))
    canvas.paste(ego, (320, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 320, 640, 380), fill=(20, 22, 24))
    draw.text((10, 328), "wrist_cam", fill=(235, 235, 235))
    draw.text((330, 328), "egocentric_cam", fill=(235, 235, 235))
    status = (
        f"step={step} success={bool(info.get('success', False))} "
        f"grasp={float(info.get('is_grasped', 0.0)):.1f} "
        f"lift={float(info.get('lift_height', 0.0)):.3f} "
        f"dist={float(info.get('tcp_to_obj_dist', 0.0)):.3f}"
    )
    draw.text((10, 354), status, fill=(130, 230, 170) if bool(info.get("success", False)) else (235, 235, 235))
    return canvas


def _chw_to_pil(image_chw: np.ndarray) -> Any:
    from PIL import Image

    return Image.fromarray(np.transpose(np.asarray(image_chw, dtype=np.uint8), (1, 2, 0)), mode="RGB")


def _require_device(torch: Any, device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")


if __name__ == "__main__":
    main()
