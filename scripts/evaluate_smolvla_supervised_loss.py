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

from physical_ai_agent.policies.smolvla_real import (
    _load_pretrained_policy,
    _policy_device_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute supervised SmolVLA policy.forward loss on a LeRobotDataset split."
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--torch-seed", type=int, default=1000)
    args = parser.parse_args()

    report = evaluate_supervised_loss(
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


def evaluate_supervised_loss(
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

    losses: list[float] = []
    samples_seen = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            torch.manual_seed(int(torch_seed) + batch_index)
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            batch_size_seen = _batch_size(batch)
            samples_seen += batch_size_seen

    if not losses:
        raise RuntimeError("No validation batches were evaluated")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "operation": "evaluate_smolvla_supervised_loss",
        "policy_path": policy_path,
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "dataset_num_frames": int(dataset.num_frames),
        "dataset_num_episodes": int(dataset.num_episodes),
        "batch_size": int(batch_size),
        "max_batches": int(max_batches),
        "batches_evaluated": len(losses),
        "samples_seen": int(samples_seen),
        "loss_mean": float(sum(losses) / len(losses)),
        "loss_min": float(min(losses)),
        "loss_max": float(max(losses)),
        "losses": losses,
        "torch_seed": int(torch_seed),
        "duration_s": round(perf_counter() - started, 4),
        "device": _policy_device_metadata(policy),
        "delta_timestamps": delta_timestamps,
        "output_path": str(output_path),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _batch_size(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
    return 0


if __name__ == "__main__":
    main()
