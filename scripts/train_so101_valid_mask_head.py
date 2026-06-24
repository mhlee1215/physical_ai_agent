#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from physical_ai_agent.policies.smolvla_real import _load_pretrained_policy, _policy_device_metadata
from physical_ai_agent.policies.so101_valid_mask import (
    SO101ValidMaskConfig,
    SO101ValidMaskHead,
    save_valid_mask_head,
    valid_labels_from_action_is_pad,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight SO101 action valid-mask termination head.")
    parser.add_argument("--policy-path", required=True, help="SmolVLA checkpoint used for preprocessing/action chunk config.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--validation-dataset-root", type=Path)
    parser.add_argument("--validation-dataset-repo-id")
    parser.add_argument("--image-cache-dir", type=Path)
    parser.add_argument("--validation-image-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 means full train split.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 means full validation split.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    report = train_valid_mask_head(
        policy_path=args.policy_path,
        dataset_root=args.dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        validation_dataset_root=args.validation_dataset_root,
        validation_dataset_repo_id=args.validation_dataset_repo_id,
        image_cache_dir=args.image_cache_dir,
        validation_image_cache_dir=args.validation_image_cache_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        device=args.device,
        local_files_only=args.local_files_only,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def train_valid_mask_head(
    *,
    policy_path: str,
    dataset_root: Path,
    dataset_repo_id: str,
    validation_dataset_root: Path | None,
    validation_dataset_repo_id: str | None,
    image_cache_dir: Path | None,
    validation_image_cache_dir: Path | None,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    max_train_batches: int,
    max_val_batches: int,
    lr: float,
    hidden_dim: int,
    device: str,
    local_files_only: bool,
    seed: int,
) -> dict[str, Any]:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from physical_ai_agent.lerobot_sampling_augmentation import PredecodedImageCacheDataset

    started = perf_counter()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    policy = _load_pretrained_policy(model_id=policy_path, local_files_only=local_files_only, device=device)
    selected_device = str(_policy_device_metadata(policy).get("device_selected") or "cpu")
    if hasattr(policy, "config"):
        policy.config.device = selected_device
    preprocessor, _postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": selected_device}},
    )

    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    train_dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, delta_timestamps=delta_timestamps, video_backend="torchcodec")
    if image_cache_dir is not None:
        train_dataset = PredecodedImageCacheDataset(train_dataset, image_cache_dir)
    val_dataset = None
    if validation_dataset_root is not None and validation_dataset_repo_id:
        val_dataset = LeRobotDataset(
            validation_dataset_repo_id,
            root=validation_dataset_root,
            delta_timestamps=delta_timestamps,
            video_backend="torchcodec",
        )
        if validation_image_cache_dir is not None:
            val_dataset = PredecodedImageCacheDataset(val_dataset, validation_image_cache_dir)

    train_loader = _make_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = _make_dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False) if val_dataset else None

    chunk_size = int(getattr(policy.config, "chunk_size", 50))
    action_dim = int(_feature_dim(getattr(policy.config, "output_features", {}).get("action"), default=6))
    state_dim = int(_feature_dim(getattr(policy.config, "input_features", {}).get("observation.state"), default=6))
    model = SO101ValidMaskHead(
        SO101ValidMaskConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=hidden_dim,
        )
    ).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history = []
    best_val_loss = float("inf")
    best_path = output_dir / "valid_mask_head.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            preprocessor=preprocessor,
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_batches=max_train_batches,
            device=selected_device,
            train=True,
        )
        val_metrics = (
            _run_epoch(
                model=model,
                loader=val_loader,
                preprocessor=preprocessor,
                optimizer=None,
                loss_fn=loss_fn,
                max_batches=max_val_batches,
                device=selected_device,
                train=False,
            )
            if val_loader is not None
            else {}
        )
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        current_val = float(val_metrics.get("loss", train_metrics["loss"]))
        if current_val <= best_val_loss:
            best_val_loss = current_val
            save_valid_mask_head(
                best_path,
                model,
                metadata={
                    "policy_path": policy_path,
                    "dataset_root": str(dataset_root),
                    "dataset_repo_id": dataset_repo_id,
                    "validation_dataset_root": str(validation_dataset_root) if validation_dataset_root else None,
                    "validation_dataset_repo_id": validation_dataset_repo_id,
                    "epoch": epoch,
                    "delta_timestamps": delta_timestamps,
                    "metric_space": "saved_smolvla_preprocessor_action_space",
                },
            )

    report = {
        "operation": "train_so101_valid_mask_head",
        "policy_path": policy_path,
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "validation_dataset_root": str(validation_dataset_root) if validation_dataset_root else None,
        "validation_dataset_repo_id": validation_dataset_repo_id,
        "image_cache_dir": str(image_cache_dir) if image_cache_dir else None,
        "validation_image_cache_dir": str(validation_image_cache_dir) if validation_image_cache_dir else None,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "best_checkpoint": str(best_path),
        "best_val_loss": best_val_loss,
        "history": history,
        "duration_s": round(perf_counter() - started, 4),
        "device": _policy_device_metadata(policy),
    }
    (output_dir / "valid_mask_head_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _run_epoch(
    *,
    model: SO101ValidMaskHead,
    loader: Any,
    preprocessor: Any,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: torch.nn.Module,
    max_batches: int,
    device: str,
    train: bool,
) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    correct = 0
    total = 0
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = preprocessor(batch)
            state = batch["observation.state"].to(device)
            action = batch["action"].to(device)
            labels = valid_labels_from_action_is_pad(batch.get("action_is_pad")).to(device)
            logits = model(state, action)
            labels = labels[:, : logits.shape[1]]
            loss = loss_fn(logits, labels)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            predicted = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((predicted == labels).sum().detach().cpu())
            total += int(labels.numel())
    if not losses:
        raise RuntimeError("No valid-mask batches were processed")
    return {
        "loss": float(sum(losses) / len(losses)),
        "accuracy": float(correct / max(total, 1)),
        "batches": float(len(losses)),
    }


def _make_dataloader(dataset: Any, *, batch_size: int, num_workers: int, shuffle: bool) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        persistent_workers=bool(num_workers > 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _feature_dim(feature: Any, *, default: int) -> int:
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if not shape:
        return int(default)
    return int(shape[0])


if __name__ == "__main__":
    main()
