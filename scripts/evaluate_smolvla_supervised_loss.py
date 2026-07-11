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
from lerobot.utils.constants import ACTION

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
    postprocessed_action_frame_rmses: list[float] = []
    postprocessed_action_global_squared_errors: list[float] = []
    postprocessed_action_step0_rmses: list[float] = []
    postprocessed_action_max_frame_rmses: list[float] = []
    samples_seen = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            torch.manual_seed(int(torch_seed) + batch_index)
            teacher_action = _as_cpu_float_tensor(batch.get(ACTION))
            action_is_pad = _as_cpu_bool_tensor(batch.get("action_is_pad"))
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            rmse_metrics = _postprocessed_action_rmse_metrics(
                policy=policy,
                postprocessor=_postprocessor,
                batch=batch,
                teacher_action=teacher_action,
                action_is_pad=action_is_pad,
            )
            if rmse_metrics:
                postprocessed_action_frame_rmses.extend(rmse_metrics["frame_rmses"])
                postprocessed_action_global_squared_errors.extend(rmse_metrics["squared_errors"])
                if rmse_metrics.get("step0_rmse") is not None:
                    postprocessed_action_step0_rmses.append(float(rmse_metrics["step0_rmse"]))
                if rmse_metrics.get("max_frame_rmse") is not None:
                    postprocessed_action_max_frame_rmses.append(float(rmse_metrics["max_frame_rmse"]))
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
        "postprocessed_action_rmse_mean": _mean(postprocessed_action_frame_rmses),
        "postprocessed_action_rmse_max": _max_or_none(postprocessed_action_max_frame_rmses),
        "postprocessed_action_rmse_step0_mean": _mean(postprocessed_action_step0_rmses),
        "postprocessed_action_global_rmse": _sqrt_mean(postprocessed_action_global_squared_errors),
        "postprocessed_action_rmse_frame_count": len(postprocessed_action_frame_rmses),
        "postprocessed_action_rmse_note": (
            "RMSE compares policy.predict_action_chunk after the saved postprocessor against dataset teacher actions; "
            "postprocessed_action_rmse_mean averages per-frame RMSE over action dims and is the closest validation "
            "counterpart to closed-loop action_rmse_sweep."
        ),
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


def _postprocessed_action_rmse_metrics(
    *,
    policy: Any,
    postprocessor: Any,
    batch: dict[str, Any],
    teacher_action: torch.Tensor | None,
    action_is_pad: torch.Tensor | None,
) -> dict[str, Any]:
    if teacher_action is None or not hasattr(policy, "predict_action_chunk"):
        return {}
    was_training = bool(getattr(policy, "training", False))
    try:
        if hasattr(policy, "reset"):
            policy.reset()
        predicted_raw = policy.predict_action_chunk(batch)
        predicted = _postprocess_action_chunk(postprocessor, predicted_raw)
        predicted = _as_cpu_float_tensor(predicted)
        if predicted is None or predicted.ndim != 3 or teacher_action.ndim != 3:
            return {}
        horizon = min(int(predicted.shape[1]), int(teacher_action.shape[1]))
        dim = min(int(predicted.shape[2]), int(teacher_action.shape[2]))
        if horizon <= 0 or dim <= 0:
            return {}
        predicted = predicted[:, :horizon, :dim]
        teacher = teacher_action[:, :horizon, :dim]
        squared = (predicted - teacher).pow(2)
        valid_mask = _valid_action_mask(action_is_pad, batch_size=predicted.shape[0], horizon=horizon)
        frame_rmse = squared.mean(dim=-1).sqrt()
        if valid_mask is not None:
            valid_mask = valid_mask.to(dtype=torch.bool)
            frame_rmse = frame_rmse[valid_mask]
            squared = squared[valid_mask]
        if frame_rmse.numel() == 0 or squared.numel() == 0:
            return {}
        step0_rmse = squared.new_tensor([])
        if valid_mask is None:
            step0_rmse = ((predicted[:, 0, :] - teacher[:, 0, :]).pow(2).mean(dim=-1).sqrt())
        else:
            step0_valid = valid_mask[:, 0] if valid_mask.ndim == 2 and valid_mask.shape[1] > 0 else None
            if step0_valid is not None and bool(step0_valid.any()):
                step0_rmse = ((predicted[:, 0, :] - teacher[:, 0, :]).pow(2).mean(dim=-1).sqrt())[step0_valid]
        return {
            "frame_rmses": [float(value) for value in frame_rmse.detach().cpu().tolist()],
            "squared_errors": [float(value) for value in squared.reshape(-1).detach().cpu().tolist()],
            "step0_rmse": float(step0_rmse.mean().item()) if step0_rmse.numel() else None,
            "max_frame_rmse": float(frame_rmse.max().item()),
        }
    finally:
        if was_training:
            policy.train()


def _postprocess_action_chunk(postprocessor: Any, raw_action_chunk: Any) -> torch.Tensor:
    try:
        return torch.as_tensor(postprocessor(raw_action_chunk))
    except Exception:
        chunk = torch.as_tensor(raw_action_chunk)
        if chunk.ndim != 3:
            raise
        pieces = [torch.as_tensor(postprocessor(chunk[:, index, :])) for index in range(chunk.shape[1])]
        return torch.stack(pieces, dim=1)


def _valid_action_mask(action_is_pad: torch.Tensor | None, *, batch_size: int, horizon: int) -> torch.Tensor | None:
    if action_is_pad is None or action_is_pad.ndim != 2:
        return None
    valid = ~action_is_pad[:, :horizon]
    if valid.shape[0] != batch_size:
        return None
    return valid


def _as_cpu_float_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value).detach().cpu().float().clone()


def _as_cpu_bool_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value).detach().cpu().bool().clone()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _max_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(values))


def _sqrt_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(torch.tensor(values, dtype=torch.float64).mean().sqrt().item())


if __name__ == "__main__":
    main()
