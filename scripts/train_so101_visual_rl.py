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
from physical_ai_agent.sim.so101_visual_rl import SO101VisualRLConfig, _json_safe_info, make_so101_visual_rl_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight visual RL policy for SO101-Nexus.")
    parser.add_argument("--env-id", default="MuJoCoReach-v1")
    parser.add_argument("--camera-name", default="wrist_cam")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Torch device for policy training. MuJoCo rendering remains separate.",
    )
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--deterministic-eval-steps", type=int, default=80)
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_visual_rl/train"))
    args = parser.parse_args()

    report = train_so101_visual_rl(
        config=SO101VisualRLConfig(
            env_id=args.env_id,
            camera_name=args.camera_name,
            width=args.width,
            height=args.height,
            include_state=not args.no_state,
            channel_first=True,
        ),
        output_dir=args.output_dir,
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        lr=args.lr,
        device=args.device,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        deterministic_eval_steps=args.deterministic_eval_steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def train_so101_visual_rl(
    *,
    config: SO101VisualRLConfig,
    output_dir: Path,
    updates: int = 40,
    rollout_steps: int = 32,
    seed: int = 0,
    lr: float = 3e-4,
    device: str = "auto",
    gamma: float = 0.97,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    deterministic_eval_steps: int = 80,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, Any]] = []
    obs, info = env.reset(seed=seed)

    try:
        for update in range(updates):
            log_probs = []
            values = []
            entropies = []
            rewards: list[float] = []
            last_info: dict[str, Any] = {}
            terminated_count = 0
            for _step in range(rollout_steps):
                action_packet = model.act(obs, deterministic=False)
                action = action_packet["action"].detach().cpu().numpy()[0]
                obs, reward, terminated, truncated, info = env.step(action)
                log_probs.append(action_packet["log_prob"].squeeze(0))
                values.append(action_packet["value"].squeeze(0))
                entropies.append(action_packet["entropy"].squeeze(0))
                rewards.append(float(reward))
                last_info = _json_safe_info(info)
                if terminated or truncated:
                    terminated_count += 1
                    obs, info = env.reset()

            returns = _discounted_returns(rewards, gamma=gamma)
            return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=torch_device)
            value_tensor = torch.stack(values)
            log_prob_tensor = torch.stack(log_probs)
            entropy_tensor = torch.stack(entropies)
            advantage = return_tensor - value_tensor.detach()
            policy_loss = -(log_prob_tensor * advantage).mean()
            value_loss = F.mse_loss(value_tensor, return_tensor)
            entropy = entropy_tensor.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            history.append(
                {
                    "update": update,
                    "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                    "total_reward": float(sum(rewards)),
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "terminated_count": terminated_count,
                    "last_info": last_info,
                }
            )
            if update == 0 or (update + 1) % max(1, updates // 10) == 0:
                print(
                    f"update {update + 1}/{updates} "
                    f"mean_reward={history[-1]['mean_reward']:.4f} "
                    f"loss={history[-1]['loss']:.4f}",
                    flush=True,
                )

        eval_report = _evaluate_policy(
            model=model,
            config=config,
            steps=deterministic_eval_steps,
            seed=seed + 10_000,
        )
    finally:
        env.close()

    checkpoint_path = output_dir / "so101_visual_rl_policy.pt"
    manifest_path = output_dir / "training_manifest.json"
    metadata = {
        "operation": "train_so101_visual_rl",
        "config": asdict(config),
        "seed": seed,
        "updates": updates,
        "rollout_steps": rollout_steps,
        "gamma": gamma,
        "lr": lr,
        "device": str(torch_device),
        "history_tail": history[-10:],
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
        "history": history,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _evaluate_policy(*, model: Any, config: SO101VisualRLConfig, steps: int, seed: int) -> dict[str, Any]:
    env = make_so101_visual_rl_env(config)
    rewards = []
    last_info: dict[str, Any] = {}
    obs, _info = env.reset(seed=seed)
    try:
        for _step in range(steps):
            action_packet = model.act(obs, deterministic=True)
            action = action_packet["action"].detach().cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            last_info = _json_safe_info(info)
            if terminated or truncated:
                obs, _info = env.reset()
    finally:
        env.close()
    return {
        "steps": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "total_reward": float(sum(rewards)),
        "last_info": last_info,
    }


def _discounted_returns(rewards: list[float], *, gamma: float) -> list[float]:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


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
