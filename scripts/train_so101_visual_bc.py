#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.policies.so101_visual_actor_critic import (
    make_so101_visual_actor_critic,
    save_so101_visual_actor_critic_checkpoint,
)
from physical_ai_agent.sim.so101_live_viewer import _reach_controller_action
from physical_ai_agent.sim.so101_visual_rl import SO101VisualRLConfig, _json_safe_info, make_so101_visual_rl_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a working SO101 visual policy by cloning the reach controller."
    )
    parser.add_argument("--env-id", default="MuJoCoReach-v1")
    parser.add_argument("--camera-name", default="wrist_cam")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--reset-interval", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dagger-iterations", type=int, default=0)
    parser.add_argument("--dagger-samples", type=int, default=1000)
    parser.add_argument("--dagger-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Torch device for policy training. Rendering still runs through MuJoCo on CPU/macOS GL.",
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=80)
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_visual_rl/train_reach_bc"))
    args = parser.parse_args()

    report = train_so101_visual_bc(
        config=SO101VisualRLConfig(
            env_id=args.env_id,
            camera_name=args.camera_name,
            width=args.width,
            height=args.height,
            include_state=not args.no_state,
            channel_first=True,
        ),
        output_dir=args.output_dir,
        samples=args.samples,
        reset_interval=args.reset_interval,
        epochs=args.epochs,
        dagger_iterations=args.dagger_iterations,
        dagger_samples=args.dagger_samples,
        dagger_epochs=args.dagger_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.lr,
        device=args.device,
        eval_episodes=args.eval_episodes,
        eval_steps=args.eval_steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def train_so101_visual_bc(
    *,
    config: SO101VisualRLConfig,
    output_dir: Path,
    samples: int = 3000,
    reset_interval: int = 80,
    epochs: int = 8,
    dagger_iterations: int = 0,
    dagger_samples: int = 1000,
    dagger_epochs: int = 4,
    batch_size: int = 128,
    seed: int = 0,
    lr: float = 1e-3,
    device: str = "auto",
    eval_episodes: int = 5,
    eval_steps: int = 80,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = _resolve_torch_device(device)
    print(f"training torch device: {torch_device}", flush=True)

    env = make_so101_visual_rl_env(config)
    model = make_so101_visual_actor_critic(
        observation_space=env.observation_space,
        action_space=env.action_space,
    ).to(torch_device)
    images: list[Any] = []
    states: list[Any] = []
    actions: list[Any] = []
    collect_records: list[dict[str, Any]] = []

    reset_count = 0
    obs, _info = env.reset(seed=seed + reset_count)
    try:
        for index in range(samples):
            teacher_action = np.asarray(_reach_controller_action(env), dtype=np.float32)
            images.append(np.asarray(obs["image"], dtype=np.uint8))
            if config.include_state:
                states.append(np.asarray(obs["state"], dtype=np.float32))
            actions.append(teacher_action)
            obs, reward, terminated, truncated, info = env.step(teacher_action)
            if index == 0 or (index + 1) % max(1, samples // 10) == 0:
                record = {
                    "sample": index + 1,
                    "reward": float(reward),
                    "info": _json_safe_info(info),
                }
                collect_records.append(record)
                print(
                    f"collect {index + 1}/{samples} "
                    f"reward={record['reward']:.4f} info={record['info']}",
                    flush=True,
                )
            if terminated or truncated:
                reset_count += 1
                obs, _info = env.reset(seed=seed + reset_count)
            elif reset_interval > 0 and (index + 1) % reset_interval == 0:
                reset_count += 1
                obs, _info = env.reset(seed=seed + reset_count)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history: list[dict[str, Any]] = []
        history.extend(
            _train_supervised_epochs(
                model=model,
                optimizer=optimizer,
                images=images,
                states=states if config.include_state else None,
                actions=actions,
                batch_size=batch_size,
                epochs=epochs,
                seed=seed,
                phase="bc",
                start_epoch=0,
            )
        )
        for dagger_iteration in range(dagger_iterations):
            obs, _info = env.reset(seed=seed + 1000 + dagger_iteration)
            for index in range(dagger_samples):
                teacher_action = np.asarray(_reach_controller_action(env), dtype=np.float32)
                images.append(np.asarray(obs["image"], dtype=np.uint8))
                if config.include_state:
                    states.append(np.asarray(obs["state"], dtype=np.float32))
                actions.append(teacher_action)
                with torch.no_grad():
                    policy_action = model.act(obs, deterministic=True)["action"].cpu().numpy()[0]
                obs, _reward, terminated, truncated, _info = env.step(policy_action)
                if terminated or truncated or (
                    reset_interval > 0 and (index + 1) % reset_interval == 0
                ):
                    obs, _info = env.reset(seed=seed + 1000 + dagger_iteration * dagger_samples + index)
            history.extend(
                _train_supervised_epochs(
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    states=states if config.include_state else None,
                    actions=actions,
                    batch_size=batch_size,
                    epochs=dagger_epochs,
                    seed=seed + 10_000 + dagger_iteration,
                    phase=f"dagger_{dagger_iteration}",
                    start_epoch=len(history),
                )
            )

        eval_report = _evaluate_policy(
            model=model,
            config=config,
            episodes=eval_episodes,
            steps=eval_steps,
            seed=seed + 20_000,
        )
    finally:
        env.close()

    checkpoint_path = output_dir / "so101_visual_bc_policy.pt"
    manifest_path = output_dir / "training_manifest.json"
    metadata = {
        "operation": "train_so101_visual_bc",
        "teacher": "reach_jacobian_controller",
        "config": asdict(config),
        "seed": seed,
        "samples": samples,
        "reset_interval": reset_interval,
        "reset_count": reset_count,
        "epochs": epochs,
        "dagger_iterations": dagger_iterations,
        "dagger_samples": dagger_samples,
        "dagger_epochs": dagger_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(torch_device),
        "collect_records": collect_records,
        "history": history,
        "eval": eval_report,
    }
    save_so101_visual_actor_critic_checkpoint(
        path=checkpoint_path,
        model=model,
        observation_space=env.observation_space,
        action_space=env.action_space,
        metadata=metadata,
    )
    manifest = {
        **metadata,
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _evaluate_policy(
    *,
    model: Any,
    config: SO101VisualRLConfig,
    episodes: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    env = make_so101_visual_rl_env(config)
    episode_records = []
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            initial_dist = _reach_distance(env)
            rewards = []
            last_info = _json_safe_info(info)
            for _step in range(steps):
                action_packet = model.act(obs, deterministic=True)
                action = action_packet["action"].detach().cpu().numpy()[0]
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                last_info = _json_safe_info(info)
                if terminated or truncated:
                    break
            final_dist = _reach_distance(env)
            episode_records.append(
                {
                    "episode": episode,
                    "steps": len(rewards),
                    "initial_distance": initial_dist,
                    "final_distance": final_dist,
                    "distance_delta": initial_dist - final_dist,
                    "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                    "total_reward": float(sum(rewards)),
                    "success": bool(last_info.get("success", False)),
                    "last_info": last_info,
                }
            )
    finally:
        env.close()
    return {
        "episodes": episode_records,
        "mean_initial_distance": float(np.mean([r["initial_distance"] for r in episode_records])),
        "mean_final_distance": float(np.mean([r["final_distance"] for r in episode_records])),
        "mean_distance_delta": float(np.mean([r["distance_delta"] for r in episode_records])),
        "success_rate": float(np.mean([r["success"] for r in episode_records])),
    }


def _train_supervised_epochs(
    *,
    model: Any,
    optimizer: Any,
    images: list[Any],
    states: list[Any] | None,
    actions: list[Any],
    batch_size: int,
    epochs: int,
    seed: int,
    phase: str,
    start_epoch: int,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    sample_count = len(actions)
    device = next(model.parameters()).device
    image_tensor = torch.as_tensor(np.stack(images), dtype=torch.uint8, device=device)
    state_tensor = (
        torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        if states is not None
        else None
    )
    action_tensor = torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)
    history = []
    grad_norm = torch.tensor(0.0)
    for local_epoch in range(epochs):
        permutation = torch.randperm(sample_count, generator=generator)
        epoch_losses = []
        for start in range(0, sample_count, batch_size):
            index = permutation[start : start + batch_size]
            batch_obs: dict[str, Any] = {"image": image_tensor[index]}
            if state_tensor is not None:
                batch_obs["state"] = state_tensor[index]
            pred = model(batch_obs).action_mean
            loss = F.mse_loss(pred, action_tensor[index])
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "phase": phase,
                "epoch": start_epoch + local_epoch,
                "samples": sample_count,
                "mse": float(np.mean(epoch_losses)),
                "grad_norm": float(grad_norm.detach().cpu()),
            }
        )
        print(
            f"{phase} epoch {local_epoch + 1}/{epochs} "
            f"samples={sample_count} mse={history[-1]['mse']:.6f}",
            flush=True,
        )
    return history


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


def _reach_distance(env: Any) -> float:
    import numpy as np

    model = env.unwrapped.model
    data = env.unwrapped.data
    target = data.site_xpos[model.site("reach_target").id]
    gripper = data.site_xpos[model.site("gripperframe").id]
    return float(np.linalg.norm(target - gripper))


if __name__ == "__main__":
    main()
