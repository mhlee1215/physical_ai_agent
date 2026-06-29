from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


VISUAL_SERVO_LABEL_KEYS = (
    "visual_servo.camera1",
    "visual_servo.camera1_visible",
    "visual_servo.camera2",
    "visual_servo.camera2_visible",
    "visual_servo.stop_label",
)


@dataclass(frozen=True)
class SO101VisualServoHeadConfig:
    hidden_dim: int = 128
    image_size: int = 256


class SO101VisualServoHead(nn.Module):
    """Small auxiliary head for image-space visual servo targets."""

    def __init__(self, config: SO101VisualServoHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or SO101VisualServoHeadConfig()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(6, 32), nn.ReLU(inplace=True))
        self.lang_encoder = nn.Sequential(nn.Linear(2, 16), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(32 + 32 + 16, int(self.config.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.config.hidden_dim), 7),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        image1 = _latest_observation(batch["observation.images.camera1"]).float()
        image2 = _latest_observation(batch["observation.images.camera2"]).float()
        state = _latest_observation(batch["observation.state"]).float()
        lang = self.lang_encoder(_language_features(batch, device=state.device, dtype=state.dtype))
        features = torch.cat([self.image_encoder(_append_xy_channels(torch.cat([image1, image2], dim=1))), self.state_encoder(state), lang], dim=-1)
        output = self.head(features)
        return {
            "camera1": output[:, 0:3],
            "camera2": output[:, 3:6],
            "stop_logit": output[:, 6],
        }


def _latest_observation(value: torch.Tensor) -> torch.Tensor:
    """Select the current observation from LeRobot's optional n_obs_steps axis."""
    if value.ndim in (3, 5):
        return value[:, -1]
    return value


def _append_xy_channels(image: torch.Tensor) -> torch.Tensor:
    batch, _channels, height, width = image.shape
    y = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype).view(1, 1, height, 1)
    x = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype).view(1, 1, 1, width)
    xy = torch.cat([x.expand(batch, 1, height, width), y.expand(batch, 1, height, width)], dim=1)
    return torch.cat([image, xy], dim=1)


def visual_servo_loss(
    head: SO101VisualServoHead | None,
    batch: dict[str, Any],
    *,
    weight: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if head is None or float(weight) <= 0.0 or not all(key in batch for key in VISUAL_SERVO_LABEL_KEYS):
        return None, {}
    pred = head(batch)
    loss = torch.zeros((), device=pred["stop_logit"].device, dtype=pred["stop_logit"].dtype)
    metrics: dict[str, float] = {}
    for camera in ("camera1", "camera2"):
        target = batch[f"visual_servo.{camera}"].to(device=pred[camera].device, dtype=pred[camera].dtype)
        visible = batch[f"visual_servo.{camera}_visible"].to(device=pred[camera].device).bool()
        if bool(visible.any()):
            camera_loss = torch.nn.functional.mse_loss(pred[camera][visible], target[visible])
            loss = loss + camera_loss
            metrics[f"visual_servo_{camera}_loss"] = float(camera_loss.detach().cpu())
    stop_target = batch["visual_servo.stop_label"].to(device=pred["stop_logit"].device, dtype=pred["stop_logit"].dtype)
    stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred["stop_logit"], stop_target)
    loss = loss + stop_loss
    metrics["visual_servo_stop_loss"] = float(stop_loss.detach().cpu())
    metrics["visual_servo_stop_accuracy"] = float(((torch.sigmoid(pred["stop_logit"]) >= 0.5) == (stop_target >= 0.5)).float().mean().detach().cpu())
    weighted = float(weight) * loss
    metrics["visual_servo_loss"] = float(loss.detach().cpu())
    metrics["visual_servo_loss_weight"] = float(weight)
    return weighted, metrics


def save_visual_servo_head(path: Path, head: SO101VisualServoHead, *, metadata: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "config": asdict(head.config),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_visual_servo_head(path: Path, *, device: str | torch.device = "cpu") -> SO101VisualServoHead:
    payload = torch.load(path, map_location=device)
    head = SO101VisualServoHead(SO101VisualServoHeadConfig(**dict(payload.get("config") or {})))
    state_dict = dict(payload["state_dict"])
    conv_key = "image_encoder.0.weight"
    if conv_key in state_dict and state_dict[conv_key].shape[1] == 6:
        state_dict[conv_key] = torch.cat(
            [
                state_dict[conv_key],
                torch.zeros(
                    (*state_dict[conv_key].shape[:1], 2, *state_dict[conv_key].shape[2:]),
                    dtype=state_dict[conv_key].dtype,
                    device=state_dict[conv_key].device,
                ),
            ],
            dim=1,
        )
    head.load_state_dict(state_dict)
    head.to(device)
    head.eval()
    return head


def _language_features(batch: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tokens = batch.get("observation.language.tokens")
    mask = batch.get("observation.language.attention_mask")
    if tokens is None:
        tokens = batch.get("language_tokens")
    if mask is None:
        mask = batch.get("language_attention_mask")
    batch_size = int(batch["observation.state"].shape[0])
    if tokens is None:
        return torch.zeros((batch_size, 2), device=device, dtype=dtype)
    tokens = tokens.to(device=device, dtype=dtype)
    if mask is None:
        mask = torch.ones_like(tokens, dtype=dtype, device=device)
    else:
        mask = mask.to(device=device, dtype=dtype)
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean_token = (tokens * mask).sum(dim=-1, keepdim=True) / denom
    length = denom / max(1, int(tokens.shape[-1]))
    return torch.cat([mean_token / 100000.0, length], dim=-1)
