from __future__ import annotations

from pathlib import Path
from typing import Any


class SO101VisualReachDelta:
    """CNN that predicts target-minus-gripper Cartesian error for SO101 Reach."""

    def __new__(
        cls,
        *,
        image_shape: tuple[int, ...],
        state_dim: int = 0,
        hidden_dim: int = 128,
    ) -> Any:
        import torch
        from torch import nn

        class _TorchSO101VisualReachDelta(nn.Module):
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
                self.head = nn.Sequential(
                    nn.Linear(64 + self.state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 3),
                )

            def forward(self, observation: dict[str, Any]) -> Any:
                image = torch.as_tensor(observation["image"], device=next(self.parameters()).device)
                if image.ndim == 3:
                    image = image.unsqueeze(0)
                image = image.float() / 255.0
                if not self.channel_first:
                    image = image.permute(0, 3, 1, 2)
                encoded = self.encoder(image)
                if self.state_dim > 0:
                    state = torch.as_tensor(observation["state"], device=encoded.device).float()
                    if state.ndim == 1:
                        state = state.unsqueeze(0)
                    encoded = torch.cat([encoded, state], dim=-1)
                return self.head(encoded)

        return _TorchSO101VisualReachDelta()


def make_so101_visual_reach_delta(*, observation_space: Any, hidden_dim: int = 128) -> Any:
    state_dim = 0
    if hasattr(observation_space, "spaces") and "state" in observation_space.spaces:
        state_dim = int(observation_space.spaces["state"].shape[0])
    return SO101VisualReachDelta(
        image_shape=tuple(observation_space.spaces["image"].shape),
        state_dim=state_dim,
        hidden_dim=hidden_dim,
    )


def save_so101_visual_reach_delta_checkpoint(
    *,
    path: Path,
    model: Any,
    observation_space: Any,
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
        },
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_so101_visual_reach_delta_checkpoint(
    path: Path, *, map_location: str = "cpu"
) -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location=map_location)
    config = payload["model_config"]
    model = SO101VisualReachDelta(
        image_shape=tuple(config["image_shape"]),
        state_dim=int(config.get("state_dim", 0)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload.get("metadata", {})
