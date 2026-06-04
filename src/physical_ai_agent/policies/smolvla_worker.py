from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from physical_ai_agent.policies.smolvla_adapter import DEFAULT_SMOLVLA_MODEL_ID


@dataclass(frozen=True)
class SmolVLAWorkerConfig:
    model_id: str = DEFAULT_SMOLVLA_MODEL_ID
    local_files_only: bool = True
    action_steps: int = 15


def run_worker(config: SmolVLAWorkerConfig) -> None:
    from physical_ai_agent.policies.smolvla_real import (
        _build_batch_for_policy,
        _clip_action,
        _load_pretrained_policy,
        _tensor_to_float_list,
    )

    with contextlib.redirect_stdout(sys.stderr):
        policy = _load_pretrained_policy(
            model_id=config.model_id,
            local_files_only=config.local_files_only,
        )
    chunk_size = int(getattr(policy.config, "chunk_size", config.action_steps))
    if config.action_steps < 1 or config.action_steps > chunk_size:
        raise ValueError(f"action_steps must be in [1, {chunk_size}], got {config.action_steps}")

    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("type") == "shutdown":
            break
        started_at = perf_counter()
        observation = [float(value) for value in request["observation"]]
        action_dim = int(request["action_dim"])
        camera_pixels = _decode_camera_pixels(request["camera_pixels"])
        with contextlib.redirect_stdout(sys.stderr):
            batch, image_feature_mapping = _build_batch_for_policy(policy, observation, camera_pixels)
            action_chunk = policy.predict_action_chunk(batch)
        actions = [
            _clip_action(_tensor_to_float_list(action_chunk[:, index, :]), action_dim)
            for index in range(min(config.action_steps, int(action_chunk.shape[1])))
        ]
        response = {
            "type": "action_chunk",
            "actions": actions,
            "predicted_chunk_size": int(action_chunk.shape[1]),
            "executed_action_steps": len(actions),
            "image_feature_mapping": image_feature_mapping,
            "latency_s": round(perf_counter() - started_at, 4),
        }
        print(json.dumps(response, sort_keys=True), flush=True)


def _decode_camera_pixels(encoded: dict[str, str]) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    pixels = {}
    for camera_name, payload in encoded.items():
        raw = base64.b64decode(payload.encode("ascii"))
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        pixels[camera_name] = np.asarray(image)
    return pixels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Line-delimited JSON SmolVLA action chunk worker.")
    parser.add_argument("--model-id", default=DEFAULT_SMOLVLA_MODEL_ID)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--action-steps", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_worker(
        SmolVLAWorkerConfig(
            model_id=args.model_id,
            local_files_only=not args.allow_download,
            action_steps=args.action_steps,
        )
    )


if __name__ == "__main__":
    main()
