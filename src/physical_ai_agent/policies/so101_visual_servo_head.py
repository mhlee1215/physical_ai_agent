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
    context_dim: int = 576


class SO101VisualServoHead(nn.Module):
    """Auxiliary head attached to camera image patch tokens plus SmolVLA prefix context."""

    def __init__(self, config: SO101VisualServoHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or SO101VisualServoHeadConfig()
        self.token_score = nn.Sequential(
            nn.LayerNorm(int(self.config.context_dim)),
            nn.Linear(int(self.config.context_dim), 1),
        )
        self.context_score = nn.Sequential(
            nn.LayerNorm(int(self.config.context_dim)),
            nn.Linear(int(self.config.context_dim), 1),
        )
        self.camera1_head = self._camera_head()
        self.camera2_head = self._camera_head()
        self.camera1_visible_head = self._scalar_head()
        self.camera2_visible_head = self._scalar_head()
        self.stop_head = self._scalar_head()
        for module in (
            self.token_score[-1],
            self.context_score[-1],
            self.camera1_head[-1],
            self.camera2_head[-1],
            self.camera1_visible_head[-1],
            self.camera2_visible_head[-1],
            self.stop_head[-1],
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def _camera_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(int(self.config.context_dim)),
            nn.Linear(int(self.config.context_dim), int(self.config.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.config.hidden_dim), 3),
        )

    def _scalar_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(int(self.config.context_dim)),
            nn.Linear(int(self.config.context_dim), int(self.config.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.config.hidden_dim), 1),
        )

    def forward(self, features: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        if torch.is_tensor(features):
            # ponytail: compatibility for old tests/checkpoints; real training passes camera patch tokens.
            features = {"camera1": features[:, None, :], "camera2": features[:, None, :], "context": features[:, None, :]}
        context = self._pool_context_tokens(features.get("context", 0.5 * (features["camera1"] + features["camera2"])))
        camera1 = self._pool_camera_tokens(features["camera1"])
        camera2 = self._pool_camera_tokens(features["camera2"])
        camera1 = camera1 + context
        camera2 = camera2 + context
        stop_context = 0.5 * (camera1 + camera2) + context
        return {
            "camera1": torch.tanh(self.camera1_head(camera1)),
            "camera2": torch.tanh(self.camera2_head(camera2)),
            "stop_logit": self.stop_head(stop_context).squeeze(-1),
            "camera1_visible_logit": self.camera1_visible_head(camera1).squeeze(-1),
            "camera2_visible_logit": self.camera2_visible_head(camera2).squeeze(-1),
        }

    def _pool_camera_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 2:
            tokens = tokens[:, None, :]
        tokens = tokens.float()
        weights = torch.softmax(self.token_score(tokens).squeeze(-1), dim=-1)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)

    def _pool_context_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 2:
            tokens = tokens[:, None, :]
        tokens = tokens.float()
        weights = torch.softmax(self.context_score(tokens).squeeze(-1), dim=-1)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)


def extract_smolvla_camera_patch_features(policy: Any, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    if getattr(policy, "name", None) != "smolvla" or not hasattr(policy, "model"):
        raise TypeError("visual servo camera patch features require a SmolVLA policy")
    images, img_masks = policy.prepare_images(batch)
    if len(images) < 2:
        raise ValueError(f"visual servo requires camera1 and camera2 images, got {len(images)} image tensor(s)")
    features = {
        "camera1": _embed_camera_image_tokens(policy, images[0]),
        "camera2": _embed_camera_image_tokens(policy, images[1]),
    }
    context = _embed_smolvla_prefix_context(policy, batch, images, img_masks)
    if context is not None:
        features["context"] = context
    return features


def _embed_smolvla_prefix_context(
    policy: Any,
    batch: dict[str, Any],
    images: list[torch.Tensor],
    img_masks: list[torch.Tensor],
) -> torch.Tensor | None:
    try:
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        state = policy.prepare_state(batch)
        prefix_embs, prefix_pad_masks, _ = policy.model.embed_prefix(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=state,
        )
    except Exception:
        return None
    weights = prefix_pad_masks.to(device=prefix_embs.device, dtype=prefix_embs.dtype)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (prefix_embs * weights.unsqueeze(-1)).sum(dim=1) / denom


def _embed_camera_image_tokens(policy: Any, image: torch.Tensor) -> torch.Tensor:
    while image.ndim > 4:
        image = image[:, -1]
    tokens = policy.model.vlm_with_expert.embed_image(image)
    token_dim = tokens.shape[-1]
    return tokens * torch.tensor(token_dim**0.5, dtype=tokens.dtype, device=tokens.device)


def visual_servo_loss(
    head: SO101VisualServoHead | None,
    batch: dict[str, Any],
    *,
    weight: float,
    policy: Any | None = None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if head is None or float(weight) <= 0.0 or not all(key in batch for key in VISUAL_SERVO_LABEL_KEYS):
        return None, {}
    if policy is None:
        return None, {}
    pred = head(extract_smolvla_camera_patch_features(policy, batch))
    loss = torch.zeros((), device=pred["stop_logit"].device, dtype=pred["stop_logit"].dtype)
    metrics: dict[str, float] = {}
    visible_errors = []
    for camera in ("camera1", "camera2"):
        target = batch[f"visual_servo.{camera}"].to(device=pred[camera].device, dtype=pred[camera].dtype)
        visible = batch[f"visual_servo.{camera}_visible"].to(device=pred[camera].device).bool()
        visible_target = visible.to(device=pred[f"{camera}_visible_logit"].device, dtype=pred[f"{camera}_visible_logit"].dtype)
        visible_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred[f"{camera}_visible_logit"],
            visible_target,
        )
        loss = loss + visible_loss
        metrics[f"visual_servo_{camera}_visible_loss"] = float(visible_loss.detach().cpu())
        metrics[f"visual_servo_{camera}_visible_accuracy"] = float(
            ((torch.sigmoid(pred[f"{camera}_visible_logit"]) >= 0.5) == visible).float().mean().detach().cpu()
        )
        if bool(visible.any()):
            error = pred[camera][visible] - target[visible]
            component_weights = torch.tensor([1.0, 1.0, 0.25], device=error.device, dtype=error.dtype)
            camera_mse = (error.square() * component_weights).mean()
            camera_raw_mse = error.square().mean()
            loss = loss + camera_mse
            visible_errors.append(error)
            metrics[f"visual_servo_{camera}_mse"] = float(camera_mse.detach().cpu())
            metrics[f"visual_servo_{camera}_rmse"] = float(torch.sqrt(camera_raw_mse).detach().cpu())
            metrics[f"visual_servo_{camera}_dx_mae"] = float(error[:, 0].abs().mean().detach().cpu())
            metrics[f"visual_servo_{camera}_dy_mae"] = float(error[:, 1].abs().mean().detach().cpu())
            metrics[f"visual_servo_{camera}_angle_mae"] = float(error[:, 2].abs().mean().detach().cpu())
    stop_target = batch["visual_servo.stop_label"].to(device=pred["stop_logit"].device, dtype=pred["stop_logit"].dtype)
    stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred["stop_logit"], stop_target)
    loss = loss + stop_loss
    metrics["visual_servo_stop_loss"] = float(stop_loss.detach().cpu())
    metrics["visual_servo_stop_accuracy"] = float(((torch.sigmoid(pred["stop_logit"]) >= 0.5) == (stop_target >= 0.5)).float().mean().detach().cpu())
    weighted = float(weight) * loss
    if visible_errors:
        all_errors = torch.cat(visible_errors, dim=0)
        raw_mse = all_errors.square().mean()
        metrics["visual_servo_mse"] = float(raw_mse.detach().cpu())
        metrics["visual_servo_rmse"] = float(torch.sqrt(raw_mse).detach().cpu())
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
    config = dict(payload.get("config") or {})
    config.pop("image_size", None)
    head = SO101VisualServoHead(SO101VisualServoHeadConfig(**config))
    state_dict = dict(payload["state_dict"])
    head.load_state_dict(state_dict)
    head.to(device)
    head.eval()
    return head
