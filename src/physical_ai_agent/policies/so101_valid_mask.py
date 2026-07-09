from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class SO101ValidMaskConfig:
    state_dim: int = 6
    action_dim: int = 6
    chunk_size: int = 50
    hidden_dim: int = 128
    threshold: float = 0.5
    consecutive_invalid: int = 2


def valid_labels_from_action_is_pad(action_is_pad: Any) -> Any:
    """Convert LeRobot action_is_pad labels into 1=valid, 0=padding labels."""

    import torch

    if action_is_pad is None:
        raise ValueError("action_is_pad is required to train a valid-mask head")
    pad = torch.as_tensor(action_is_pad)
    return (~pad.bool()).float()


def first_invalid_step(valid_probs: Any, *, threshold: float = 0.5, consecutive: int = 2) -> int | None:
    """Return the first action index where predicted validity terminates."""

    import torch

    probs = torch.as_tensor(valid_probs).detach().flatten().float()
    if probs.numel() == 0:
        return None
    consecutive = max(1, int(consecutive))
    invalid_run = 0
    for index, value in enumerate(probs.tolist()):
        if float(value) < float(threshold):
            invalid_run += 1
            if invalid_run >= consecutive:
                return index - consecutive + 1
        else:
            invalid_run = 0
    return None


def execution_horizon_from_valid_probs(
    valid_probs: Any,
    *,
    max_horizon: int,
    threshold: float = 0.5,
    consecutive: int = 2,
) -> tuple[int, str]:
    """Map valid probabilities to the number of action steps to execute."""

    probs = torch.as_tensor(valid_probs).detach().flatten().float()
    horizon = max(1, int(max_horizon))
    if probs.numel() > 0:
        horizon = min(horizon, int(probs.numel()))
    probs_for_horizon = probs[:horizon]
    stop_index = first_invalid_step(probs_for_horizon, threshold=threshold, consecutive=consecutive)
    if stop_index is None:
        return horizon, "max_horizon"
    return max(1, min(horizon, int(stop_index))), "valid_mask_stop"


class SO101ValidMaskHead(torch.nn.Module):
    """Small termination head trained from action_is_pad labels.

    The head is intentionally separate from SmolVLA so the baseline policy
    checkpoint stays untouched. At rollout time it consumes the current motor
    state plus a predicted action chunk and estimates which action positions
    are still valid for the current subgoal.
    """

    def __init__(self, config: SO101ValidMaskConfig):
        super().__init__()
        import torch

        self.config = config
        input_dim = int(config.state_dim) + int(config.chunk_size) * int(config.action_dim)
        output_dim = int(config.chunk_size)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(config.hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(config.hidden_dim), int(config.hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(config.hidden_dim), output_dim),
        )

    def forward(self, state: Any, action_chunk: Any) -> Any:
        import torch

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=_module_device(self))
        chunk_tensor = torch.as_tensor(action_chunk, dtype=torch.float32, device=state_tensor.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if state_tensor.ndim > 2:
            state_tensor = state_tensor.flatten(start_dim=1)
        if chunk_tensor.ndim == 2:
            chunk_tensor = chunk_tensor.unsqueeze(0)
        chunk_tensor = _pad_or_trim_action_chunk(chunk_tensor, self.config.chunk_size, self.config.action_dim)
        features = torch.cat([state_tensor[:, : self.config.state_dim], chunk_tensor.flatten(start_dim=1)], dim=1)
        return self.net(features)

    def predict_valid_probs(self, state: Any, action_chunk: Any) -> Any:
        import torch

        with torch.inference_mode():
            return torch.sigmoid(self.forward(state, action_chunk))


def load_valid_mask_head(path: str | Path, *, device: str | None = None) -> SO101ValidMaskHead:
    import torch

    checkpoint = torch.load(Path(path), map_location=device or "cpu")
    config = SO101ValidMaskConfig(**checkpoint["config"])
    model = SO101ValidMaskHead(config)
    model.load_state_dict(checkpoint["state_dict"])
    if device:
        model.to(device)
    model.eval()
    return model


def save_valid_mask_head(path: str | Path, model: SO101ValidMaskHead, metadata: dict[str, Any] | None = None) -> None:
    import torch

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        output_path,
    )


def _pad_or_trim_action_chunk(chunk: Any, chunk_size: int, action_dim: int) -> Any:
    import torch

    tensor = torch.as_tensor(chunk)
    if tensor.ndim != 3:
        raise ValueError(f"Expected action chunk [B, T, A], got {tuple(tensor.shape)}")
    if tensor.shape[-1] < action_dim:
        pad = torch.zeros(*tensor.shape[:-1], action_dim - tensor.shape[-1], dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad], dim=-1)
    tensor = tensor[:, :, :action_dim]
    if tensor.shape[1] < chunk_size:
        pad = torch.zeros(
            tensor.shape[0],
            chunk_size - tensor.shape[1],
            action_dim,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat([tensor, pad], dim=1)
    return tensor[:, :chunk_size, :]


def _module_device(module: Any) -> Any:
    import torch

    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")
