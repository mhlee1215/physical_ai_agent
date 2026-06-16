from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VisualActorCriticOutput:
    action_mean: Any
    value: Any


class SO101VisualActorCritic:
    """Small CNN actor-critic for SO101 visual observations.

    The policy consumes observations from
    `physical_ai_agent.sim.so101_visual_rl.SO101VisualObservationWrapper`:
    a uint8 camera image plus an optional low-dimensional state vector.
    """

    def __new__(
        cls,
        *,
        image_shape: tuple[int, ...],
        action_low: Any,
        action_high: Any,
        state_dim: int = 0,
        hidden_dim: int = 128,
    ) -> Any:
        import torch
        from torch import nn

        class _TorchSO101VisualActorCritic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if len(image_shape) != 3:
                    raise ValueError(f"Expected 3D image shape, got {image_shape}")
                self.channel_first = image_shape[0] == 3
                channels = image_shape[0] if self.channel_first else image_shape[-1]
                if channels != 3:
                    raise ValueError(f"Expected RGB image shape, got {image_shape}")
                self.state_dim = int(state_dim)
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                fused_dim = 64 + self.state_dim
                self.trunk = nn.Sequential(
                    nn.Linear(fused_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                )
                action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)
                action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)
                if action_low_tensor.shape != action_high_tensor.shape:
                    raise ValueError("Action low/high bounds must have matching shapes")
                action_dim = int(action_low_tensor.numel())
                self.actor_mean = nn.Linear(hidden_dim, action_dim)
                self.critic = nn.Linear(hidden_dim, 1)
                self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
                self.register_buffer("action_low", action_low_tensor.reshape(1, -1))
                self.register_buffer("action_high", action_high_tensor.reshape(1, -1))

            def forward(self, observation: dict[str, Any]) -> VisualActorCriticOutput:
                features = self._features(observation)
                latent = self.trunk(features)
                normalized_mean = torch.tanh(self.actor_mean(latent))
                action_mean = self._scale_action(normalized_mean)
                value = self.critic(latent).squeeze(-1)
                return VisualActorCriticOutput(action_mean=action_mean, value=value)

            def distribution(self, observation: dict[str, Any]) -> tuple[Any, Any]:
                features = self._features(observation)
                latent = self.trunk(features)
                normalized_mean = torch.tanh(self.actor_mean(latent))
                std = self.log_std.exp().expand_as(normalized_mean)
                return torch.distributions.Normal(normalized_mean, std), self.critic(latent).squeeze(-1)

            def act(self, observation: dict[str, Any], *, deterministic: bool = False) -> dict[str, Any]:
                distribution, value = self.distribution(observation)
                normalized_action = distribution.mean if deterministic else distribution.rsample()
                bounded_normalized_action = normalized_action if deterministic else torch.tanh(normalized_action)
                action = self._scale_action(bounded_normalized_action)
                log_prob = distribution.log_prob(normalized_action).sum(dim=-1)
                entropy = distribution.entropy().sum(dim=-1)
                return {
                    "action": action,
                    "normalized_action": normalized_action,
                    "log_prob": log_prob,
                    "entropy": entropy,
                    "value": value,
                }

            def _features(self, observation: dict[str, Any]) -> Any:
                image = torch.as_tensor(observation["image"], device=self.action_low.device)
                if image.ndim == 3:
                    image = image.unsqueeze(0)
                image = image.float() / 255.0
                if not self.channel_first:
                    image = image.permute(0, 3, 1, 2)
                encoded = self.encoder(image)
                if self.state_dim <= 0:
                    return encoded
                state = torch.as_tensor(observation["state"], device=self.action_low.device).float()
                if state.ndim == 1:
                    state = state.unsqueeze(0)
                return torch.cat([encoded, state], dim=-1)

            def _scale_action(self, normalized_action: Any) -> Any:
                midpoint = (self.action_high + self.action_low) * 0.5
                half_range = (self.action_high - self.action_low) * 0.5
                return midpoint + normalized_action * half_range

        return _TorchSO101VisualActorCritic()


def make_so101_visual_actor_critic(
    *,
    observation_space: Any,
    action_space: Any,
    hidden_dim: int = 128,
) -> Any:
    state_dim = 0
    if hasattr(observation_space, "spaces") and "state" in observation_space.spaces:
        state_dim = int(observation_space.spaces["state"].shape[0])
    image_shape = tuple(observation_space.spaces["image"].shape)
    return SO101VisualActorCritic(
        image_shape=image_shape,
        state_dim=state_dim,
        action_low=action_space.low,
        action_high=action_space.high,
        hidden_dim=hidden_dim,
    )


def save_so101_visual_actor_critic_checkpoint(
    *,
    path: Path,
    model: Any,
    observation_space: Any,
    action_space: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    import torch

    state_shape = []
    if hasattr(observation_space, "spaces") and "state" in observation_space.spaces:
        state_shape = list(observation_space.spaces["state"].shape)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "image_shape": list(observation_space.spaces["image"].shape),
            "state_dim": int(state_shape[0]) if state_shape else 0,
            "action_low": action_space.low.tolist(),
            "action_high": action_space.high.tolist(),
        },
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_so101_visual_actor_critic_checkpoint(path: Path, *, map_location: str = "cpu") -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location=map_location)
    config = payload["model_config"]
    model = SO101VisualActorCritic(
        image_shape=tuple(config["image_shape"]),
        state_dim=int(config.get("state_dim", 0)),
        action_low=config["action_low"],
        action_high=config["action_high"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload.get("metadata", {})
