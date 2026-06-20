#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.policies.so101_visual_actor_critic import make_so101_visual_actor_critic
from physical_ai_agent.sim.so101_visual_rl import (
    SO101VisualRLConfig,
    _image_hwc,
    _json_safe_info,
    make_so101_visual_rl_env,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tiny trainability smoke for a visual SO101 actor-critic."
    )
    parser.add_argument("--env-id", default="MuJoCoPickLift-v1")
    parser.add_argument("--camera-name", default="wrist_cam")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument("--channel-last", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/so101_visual_rl/policy_smoke"),
    )
    args = parser.parse_args()

    report = run_visual_policy_smoke(
        config=SO101VisualRLConfig(
            env_id=args.env_id,
            camera_name=args.camera_name,
            width=args.width,
            height=args.height,
            include_state=not args.no_state,
            channel_first=not args.channel_last,
        ),
        output_dir=args.output_dir,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        deterministic=args.deterministic,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def run_visual_policy_smoke(
    *,
    config: SO101VisualRLConfig,
    output_dir: Path,
    rollout_steps: int = 8,
    seed: int = 0,
    lr: float = 3e-4,
    gamma: float = 0.97,
    deterministic: bool = False,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    env = make_so101_visual_rl_env(config)
    model = make_so101_visual_actor_critic(
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    records: list[dict[str, Any]] = []
    log_probs = []
    values = []
    entropies = []
    rewards: list[float] = []
    obs, info = env.reset(seed=seed)
    try:
        for step in range(rollout_steps):
            image_hwc = _image_hwc(obs["image"], channel_first=config.channel_first)
            image_path = frames_dir / f"policy_obs_{step:03d}.png"
            Image.fromarray(image_hwc).save(image_path)

            action_packet = model.act(obs, deterministic=deterministic)
            action = action_packet["action"].detach().cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)

            log_probs.append(action_packet["log_prob"].squeeze(0))
            values.append(action_packet["value"].squeeze(0))
            entropies.append(action_packet["entropy"].squeeze(0))
            rewards.append(float(reward))
            records.append(
                {
                    "step": step,
                    "image_path": str(image_path),
                    "reward": float(reward),
                    "action": action.astype(float).round(6).tolist(),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": _json_safe_info(info),
                }
            )
            if terminated or truncated:
                break

        returns = _discounted_returns(rewards, gamma=gamma)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32)
        value_tensor = torch.stack(values)
        log_prob_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)
        advantage = return_tensor - value_tensor.detach()
        policy_loss = -(log_prob_tensor * advantage).mean()
        value_loss = F.mse_loss(value_tensor, return_tensor)
        entropy_bonus = entropy_tensor.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    finally:
        env.close()

    manifest = {
        "operation": "so101_visual_rl_policy_smoke",
        "config": asdict(config),
        "seed": seed,
        "rollout_steps": len(records),
        "image_shape": list(env.observation_space.spaces["image"].shape),
        "state_shape": (
            list(env.observation_space.spaces["state"].shape)
            if "state" in env.observation_space.spaces
            else []
        ),
        "action_shape": list(env.action_space.shape),
        "total_reward": float(sum(rewards)),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy_bonus.detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
        "last_info": _json_safe_info(info),
        "records": records,
    }
    manifest_path = output_dir / "visual_policy_smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def _discounted_returns(rewards: list[float], *, gamma: float) -> list[float]:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


if __name__ == "__main__":
    main()
