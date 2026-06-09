#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


OFFICIAL_NUM_VIEWS = 3
OFFICIAL_IMAGE_SIZE = 256
OFFICIAL_LANGUAGE_MAX_LENGTH = 50
OFFICIAL_DENOISING_STEPS = 10


def _make_synthetic_image(size: int, offset: float) -> torch.Tensor:
    ys = torch.linspace(0, 1, size).view(1, size, 1).expand(1, size, size)
    xs = torch.linspace(0, 1, size).view(1, 1, size).expand(1, size, size)
    mix = torch.full((1, size, size), offset)
    return torch.cat([xs, ys, mix], dim=0).mul(255).clamp(0, 255).to(torch.uint8)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "mps":
        raise ValueError("This port is intentionally scoped to --device mps.")

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available in this process. Run outside the sandbox/with normal macOS "
            "Metal access and use torch==2.4.1 + torchvision==0.19.1."
        )

    warmup = torch.ones(1, device="mps")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig  # noqa: F401
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

    started = time.time()
    cfg = PreTrainedConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    original_tokenizer_max_length = cfg.tokenizer_max_length

    # MPS-only runtime choices.
    cfg.device = "mps"
    cfg.dtype = "float32"

    # Public 2toINF X-VLA processor parity:
    # - XVLAProcessor.num_views = 3
    # - XVLAProcessor.language_max_length = 50
    # - README inference example sends 256x256 RGB images and steps=10.
    cfg.num_denoising_steps = OFFICIAL_DENOISING_STEPS
    cfg.num_image_views = OFFICIAL_NUM_VIEWS
    cfg.resize_imgs_with_padding = (OFFICIAL_IMAGE_SIZE, OFFICIAL_IMAGE_SIZE)
    cfg.tokenizer_max_length = OFFICIAL_LANGUAGE_MAX_LENGTH

    load_started = time.time()
    policy = XVLAPolicy.from_pretrained(args.model, config=cfg, local_files_only=args.local_files_only)
    load_seconds = time.time() - load_started
    policy.eval()

    preprocess, _ = make_xvla_pre_post_processors(cfg, dataset_stats=None)

    frame: dict[str, Any] = {
        "observation.images.image": _make_synthetic_image(OFFICIAL_IMAGE_SIZE, 0.25),
        "observation.images.image2": _make_synthetic_image(OFFICIAL_IMAGE_SIZE, 0.50),
        "observation.images.image3": _make_synthetic_image(OFFICIAL_IMAGE_SIZE, 0.75),
        "observation.state": torch.zeros(8, dtype=torch.float32),
        "task": args.task,
    }

    batch = preprocess(frame)
    infer_started = time.time()
    with torch.inference_mode():
        action = policy.select_action(batch)
    infer_seconds = time.time() - infer_started

    action_cpu = action.detach().cpu()
    result = {
        "ok": True,
        "model": args.model,
        "task": args.task,
        "device_requested": args.device,
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "warmup_device": str(warmup.device),
        "policy_first_parameter_device": str(next(policy.parameters()).device),
        "policy_first_parameter_dtype": str(next(policy.parameters()).dtype),
        "official_processor_parity": {
            "num_views": OFFICIAL_NUM_VIEWS,
            "image_size": OFFICIAL_IMAGE_SIZE,
            "language_max_length": OFFICIAL_LANGUAGE_MAX_LENGTH,
            "num_denoising_steps": OFFICIAL_DENOISING_STEPS,
            "original_lerobot_tokenizer_max_length": int(original_tokenizer_max_length),
            "active_tokenizer_max_length": int(cfg.tokenizer_max_length),
        },
        "views": OFFICIAL_NUM_VIEWS,
        "image_size": OFFICIAL_IMAGE_SIZE,
        "language_max_length": OFFICIAL_LANGUAGE_MAX_LENGTH,
        "num_denoising_steps": OFFICIAL_DENOISING_STEPS,
        "max_len_seq": int(cfg.max_len_seq),
        "batch_devices": {
            key: str(value.device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        },
        "batch_shapes": {
            key: list(value.shape)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        },
        "action_shape": list(action_cpu.shape),
        "action_device": str(action.device),
        "action_dtype": str(action.dtype),
        "action_head": [float(x) for x in action_cpu.flatten()[: min(20, action_cpu.numel())]],
        "load_seconds": round(load_seconds, 4),
        "inference_seconds": round(infer_seconds, 4),
        "total_seconds": round(time.time() - started, 4),
        "mps_current_allocated_memory": int(torch.mps.current_allocated_memory()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lerobot/xvla-base e2e inference on macOS MPS.")
    parser.add_argument("--model", default="lerobot/xvla-base")
    parser.add_argument("--task", default="pick up the green figure")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run(args)
    output_text = json.dumps(result, indent=2, sort_keys=True)
    print(output_text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
