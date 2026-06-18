#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

from physical_ai_agent.policies.smolvla_real import _load_pretrained_policy, _policy_device_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure SO101 SmolVLA action chunk temporal jitter on a LeRobot dataset split."
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=8, help="0 means full dataset.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--torch-seed", type=int, default=1000)
    args = parser.parse_args()

    report = evaluate_action_chunk_jitter(
        policy_path=args.policy_path,
        dataset_root=args.dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        device=args.device,
        local_files_only=args.local_files_only,
        torch_seed=args.torch_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def evaluate_action_chunk_jitter(
    *,
    policy_path: str,
    dataset_root: Path,
    dataset_repo_id: str,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    device: str,
    local_files_only: bool,
    torch_seed: int,
) -> dict[str, Any]:
    started = perf_counter()
    torch.manual_seed(int(torch_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(torch_seed))

    policy = _load_pretrained_policy(
        model_id=policy_path,
        local_files_only=local_files_only,
        device=device,
    )
    selected_device = str(_policy_device_metadata(policy).get("device_selected") or device)
    if selected_device != "auto":
        if hasattr(policy, "config"):
            policy.config.device = selected_device
        if hasattr(policy, "to"):
            policy.to(selected_device)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    preprocessor, _postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": selected_device}},
    )

    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    pred_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    samples_seen = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            torch.manual_seed(int(torch_seed) + batch_index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(torch_seed) + batch_index)
            batch = preprocessor(batch)
            if hasattr(policy, "reset"):
                policy.reset()
            pred = policy.predict_action_chunk(batch)
            target = batch["action"]
            pred_chunks.append(_valid_action_chunk(pred, batch.get("action_is_pad")).detach().float().cpu())
            target_chunks.append(_valid_action_chunk(target, batch.get("action_is_pad")).detach().float().cpu())
            samples_seen += int(pred.shape[0])

    if not pred_chunks:
        raise RuntimeError("No batches were evaluated")

    pred_all = torch.cat(pred_chunks, dim=0)
    target_all = torch.cat(target_chunks, dim=0)
    pred_metrics = _chunk_metrics(pred_all)
    target_metrics = _chunk_metrics(target_all)
    report = {
        "operation": "evaluate_so101_action_chunk_jitter",
        "metric_space": "preprocessed_normalized_action_space",
        "policy_path": policy_path,
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "dataset_num_frames": int(dataset.num_frames),
        "dataset_num_episodes": int(dataset.num_episodes),
        "batch_size": int(batch_size),
        "max_batches": int(max_batches),
        "batches_evaluated": len(pred_chunks),
        "samples_seen": int(samples_seen),
        "predicted": pred_metrics,
        "teacher": target_metrics,
        "predicted_to_teacher": _ratios(pred_metrics, target_metrics),
        "torch_seed": int(torch_seed),
        "duration_s": round(perf_counter() - started, 4),
        "device": _policy_device_metadata(policy),
        "delta_timestamps": delta_timestamps,
        "notes": [
            "Metrics use normalized action tensors after the saved LeRobot preprocessor.",
            "delta_* measures first temporal differences action[t+1] - action[t].",
            "jerk_* measures second temporal differences.",
            "Values are for action-chunk smoothness comparison, not task success.",
        ],
        "output_path": str(output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _valid_action_chunk(chunk: torch.Tensor, action_is_pad: torch.Tensor | None) -> torch.Tensor:
    if action_is_pad is None:
        return chunk
    valid = (~action_is_pad).to(device=chunk.device)
    if valid.ndim == 2 and valid.shape[1] == chunk.shape[1]:
        return chunk * valid.unsqueeze(-1).to(chunk.dtype)
    return chunk


def _chunk_metrics(chunks: torch.Tensor) -> dict[str, Any]:
    if chunks.ndim != 3:
        raise ValueError(f"Expected chunks with shape [B, T, A], got {tuple(chunks.shape)}")
    delta = chunks[:, 1:, :] - chunks[:, :-1, :]
    jerk = delta[:, 1:, :] - delta[:, :-1, :] if delta.shape[1] > 1 else torch.zeros_like(delta)
    endpoint = chunks[:, -1, :] - chunks[:, 0, :]
    path = delta.norm(dim=-1).sum(dim=-1)
    endpoint_norm = endpoint.norm(dim=-1)
    ratio = path / endpoint_norm.clamp_min(1e-8)
    return {
        "chunk_shape": list(chunks.shape),
        "delta_abs_mean": float(delta.abs().mean()),
        "delta_abs_max": float(delta.abs().max()),
        "delta_rms": float(torch.sqrt((delta * delta).mean())),
        "delta_l2_step_mean": float(delta.norm(dim=-1).mean()),
        "delta_l2_step_max": float(delta.norm(dim=-1).max()),
        "delta_per_dim_rms": [float(value) for value in torch.sqrt((delta * delta).mean(dim=(0, 1)))],
        "jerk_abs_mean": float(jerk.abs().mean()),
        "jerk_abs_max": float(jerk.abs().max()),
        "jerk_rms": float(torch.sqrt((jerk * jerk).mean())),
        "path_length_mean": float(path.mean()),
        "path_length_max": float(path.max()),
        "endpoint_l2_mean": float(endpoint_norm.mean()),
        "path_to_endpoint_ratio_mean": float(ratio.mean()),
        "path_to_endpoint_ratio_median": float(ratio.median()),
    }


def _ratios(pred: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    keys = [
        "delta_abs_mean",
        "delta_rms",
        "delta_l2_step_mean",
        "jerk_abs_mean",
        "jerk_rms",
        "path_length_mean",
        "path_to_endpoint_ratio_mean",
    ]
    return {
        key: float(pred[key]) / max(float(target[key]), 1e-8)
        for key in keys
    }


if __name__ == "__main__":
    main()
