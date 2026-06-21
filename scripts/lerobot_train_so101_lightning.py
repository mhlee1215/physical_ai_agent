#!/usr/bin/env python3

import argparse
import dataclasses
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import parser as lerobot_parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
try:
    from lerobot.datasets.utils import cycle
except ImportError:
    from itertools import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_train import make_dataset
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.random_utils import set_seed
try:
    from lerobot.utils.train_utils import get_step_checkpoint_dir, load_training_state, save_checkpoint, update_last_checkpoint
except ModuleNotFoundError:
    from lerobot.common.train_utils import get_step_checkpoint_dir, load_training_state, save_checkpoint, update_last_checkpoint

from physical_ai_agent.lerobot_sampling_augmentation import (
    SamplingAugmentationConfig,
    SamplingAugmentedDataset,
    PredecodedImageCacheDataset,
    augment_batch_on_device,
    write_sampling_augmentation_report,
)
from physical_ai_agent.so101_lerobot_concat import (
    LeRobotConcatDataset,
    validate_compatible_lerobot_datasets,
    validate_lerobot_dataset_infos,
)


def main() -> None:
    wrapper_args, lerobot_args = _parse_wrapper_args()
    _apply_augmentation_env(wrapper_args)
    if wrapper_args.help:
        _print_help()
        return

    sys.argv = [sys.argv[0], *lerobot_args]
    _train_lightning(wrapper_args=wrapper_args)


def _parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train SO101 SmolVLA with PyTorch Lightning and TensorBoard.",
        add_help=False,
    )
    parser.add_argument("--so101-state-jitter-std", type=float, default=0.0)
    parser.add_argument("--so101-state-jitter-arm-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--so101-state-dropout-prob", type=float, default=0.0)
    parser.add_argument("--so101-state-dropout-keep-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--so101-image-camera-dropout-prob", type=float, default=0.0)
    parser.add_argument("--so101-image-patch-dropout-prob", type=float, default=0.0)
    parser.add_argument("--so101-image-patch-mask-ratio", type=float, default=0.0)
    parser.add_argument("--so101-image-affine-degrees", type=float, default=0.0)
    parser.add_argument("--so101-image-affine-translate", type=float, default=0.0)
    parser.add_argument("--so101-gpu-image-augmentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--so101-action-prefix-loss-steps", type=int, default=0)
    parser.add_argument("--so101-action-prefix-loss-weight", type=float, default=1.0)
    parser.add_argument("--so101-image-cache-dir", type=Path)
    parser.add_argument("--train-datasets-json")
    parser.add_argument("--validation-image-cache-dir", type=Path)
    parser.add_argument("--so101-augmentation-report", type=Path)
    parser.add_argument("--tensorboard-log-dir", type=Path)
    parser.add_argument("--lightning-precision", default="16-mixed")
    parser.add_argument("--lightning-accelerator", default="auto")
    parser.add_argument("--lightning-devices", default="auto")
    parser.add_argument("--lightning-strategy", default="auto")
    parser.add_argument("--lightning-log-every-n-steps", type=int)
    parser.add_argument("--lightning-fast-dev-run", action="store_true")
    parser.add_argument("--validation-dataset-root", type=Path)
    parser.add_argument("--validation-dataset-repo-id")
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--validation-num-workers", type=int, default=0)
    parser.add_argument(
        "--so101-resume-checkpoint-path",
        type=Path,
        help=(
            "SO101 wrapper resume checkpoint directory. This is kept separate "
            "from the LeRobot CLI because some installed LeRobot versions expose "
            "cfg.checkpoint_path in config but not as an argparse flag."
        ),
    )
    parser.add_argument(
        "--post-checkpoint-loop-command-json",
        help=(
            "JSON argv list to run inside the training loop immediately after "
            "a checkpoint is saved. Used for mandatory one-shot closed-loop tests."
        ),
    )
    parser.add_argument(
        "--validation-every-n-train-steps",
        type=int,
        help="Deprecated alias for --validation-interval-steps.",
    )
    parser.add_argument(
        "--validation-interval-steps",
        type=int,
        help="Run validation after every N train steps. 0 disables step-based validation.",
    )
    parser.add_argument(
        "--validation-interval-epochs",
        type=int,
        help="Run validation after every N train epochs. Ignored when step-based validation is enabled.",
    )
    parser.add_argument(
        "--log-input-images-every-n-steps",
        type=int,
        default=1,
        help="Log camera1/camera2 input images to TensorBoard every N train steps. 0 disables image logging.",
    )
    parser.add_argument(
        "--log-input-image-cameras",
        default="camera1,camera2",
        help="Comma-separated SmolVLA camera names to log from observation.images.<name>.",
    )
    parser.add_argument(
        "--log-input-metadata-every-n-steps",
        type=int,
        help="Log prompt text and motor state inputs to TensorBoard every N train steps. Defaults to image logging cadence.",
    )
    parser.add_argument("--help", action="store_true")
    return parser.parse_known_args()


def _apply_augmentation_env(args: argparse.Namespace) -> None:
    os.environ["SO101_STATE_JITTER_STD"] = str(args.so101_state_jitter_std)
    os.environ["SO101_STATE_JITTER_ARM_ONLY"] = "1" if args.so101_state_jitter_arm_only else "0"
    os.environ["SO101_STATE_DROPOUT_PROB"] = str(args.so101_state_dropout_prob)
    os.environ["SO101_STATE_DROPOUT_KEEP_GRIPPER"] = "1" if args.so101_state_dropout_keep_gripper else "0"
    os.environ["SO101_IMAGE_CAMERA_DROPOUT_PROB"] = str(args.so101_image_camera_dropout_prob)
    os.environ["SO101_IMAGE_PATCH_DROPOUT_PROB"] = str(args.so101_image_patch_dropout_prob)
    os.environ["SO101_IMAGE_PATCH_MASK_RATIO"] = str(args.so101_image_patch_mask_ratio)
    os.environ["SO101_IMAGE_AFFINE_DEGREES"] = str(args.so101_image_affine_degrees)
    os.environ["SO101_IMAGE_AFFINE_TRANSLATE"] = str(args.so101_image_affine_translate)
    os.environ["SO101_GPU_IMAGE_AUGMENTATION"] = "1" if args.so101_gpu_image_augmentation else "0"
    if args.so101_image_cache_dir is not None:
        os.environ["SO101_IMAGE_CACHE_DIR"] = str(args.so101_image_cache_dir)


def _print_help() -> None:
    print("SO101 Lightning wrapper options:")
    print("  --tensorboard-log-dir PATH")
    print("  --lightning-precision 16-mixed|bf16-mixed|32-true")
    print("  --lightning-accelerator auto|cpu|gpu|cuda|mps")
    print("  --lightning-devices auto|1|...")
    print("  --lightning-fast-dev-run")
    print("\nPass regular lerobot-train options after these wrapper options.")
    sys.argv = [sys.argv[0], "--help"]
    from lerobot.scripts.lerobot_train import train

    train()


@lerobot_parser.wrap()
def _train_lightning(cfg: TrainPipelineConfig, wrapper_args: argparse.Namespace) -> None:
    lightning = _import_lightning()
    LightningModule = lightning.LightningModule
    Trainer = lightning.Trainer
    TensorBoardLogger = _import_tensorboard_logger()

    cfg.validate()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if cfg.seed is not None:
        set_seed(cfg.seed)
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    augmentation = SamplingAugmentationConfig.from_env()
    dataset = _make_dataset(cfg, augmentation, wrapper_args)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    if cfg.peft is not None:
        policy = policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(cfg.peft))
    preprocessor, postprocessor = _make_processors(cfg, dataset, policy)
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    resume_step = 0
    if cfg.resume:
        checkpoint_path = _resolve_resume_checkpoint_path(cfg, wrapper_args)
        resume_step, optimizer, scheduler = load_training_state(checkpoint_path, optimizer, scheduler)
        logging.info("Resumed LeRobot training state from %s at step %s", checkpoint_path, resume_step)
        print(f"Resumed LeRobot training state from {checkpoint_path} at step {resume_step}", flush=True)

    dataloader = _make_dataloader(cfg, dataset)
    validation_dataloader = _make_validation_dataloader(cfg, wrapper_args)
    validation_step_interval = _validation_step_interval(wrapper_args)
    validation_epoch_interval = _validation_epoch_interval(wrapper_args, validation_step_interval)
    logging.info(
        "Validation schedule: enabled=%s step_interval=%s epoch_interval=%s",
        validation_dataloader is not None,
        validation_step_interval,
        validation_epoch_interval,
    )
    print(
        "Validation schedule: "
        f"enabled={validation_dataloader is not None} "
        f"step_interval={validation_step_interval} "
        f"epoch_interval={validation_epoch_interval}",
        flush=True,
    )
    module = _SO101LightningModule(
        LightningModule=LightningModule,
        cfg=cfg,
        policy=policy,
        preprocessor=preprocessor,
        optimizer=optimizer,
        scheduler=scheduler,
        augmentation=augmentation,
        action_prefix_loss_steps=int(wrapper_args.so101_action_prefix_loss_steps),
        action_prefix_loss_weight=float(wrapper_args.so101_action_prefix_loss_weight),
        validation_dataloader=validation_dataloader,
        validation_step_interval=validation_step_interval,
        validation_epoch_interval=validation_epoch_interval,
        input_image_cameras=_parse_csv(wrapper_args.log_input_image_cameras),
        log_input_images_every_n_steps=int(wrapper_args.log_input_images_every_n_steps),
        log_input_metadata_every_n_steps=_metadata_log_interval(wrapper_args),
        initial_step=resume_step,
    )
    tb_log_dir = (wrapper_args.tensorboard_log_dir or cfg.output_dir / "tensorboard").resolve()
    logger = TensorBoardLogger(save_dir=str(tb_log_dir), name="so101_smolvla", version="")
    callbacks = []
    checkpoint_callback = _make_lerobot_checkpoint_callback(
        lightning.Callback,
        cfg=cfg,
        policy_module=module,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        save_freq=int(cfg.save_freq),
        enabled=bool(cfg.save_checkpoint),
        initial_step=resume_step,
        post_checkpoint_loop_command=_post_checkpoint_loop_command(wrapper_args),
    )
    callbacks.append(checkpoint_callback)
    log_every_n_steps = wrapper_args.lightning_log_every_n_steps or max(1, int(cfg.log_freq))
    remaining_steps = max(0, int(cfg.steps) - int(resume_step))
    trainer = Trainer(
        accelerator=wrapper_args.lightning_accelerator,
        devices=_parse_devices(wrapper_args.lightning_devices),
        strategy=wrapper_args.lightning_strategy,
        precision=wrapper_args.lightning_precision,
        max_steps=remaining_steps,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=False,
        enable_progress_bar=True,
        fast_dev_run=bool(wrapper_args.lightning_fast_dev_run),
        val_check_interval=None,
        check_val_every_n_epoch=None,
    )
    if wrapper_args.so101_augmentation_report is not None:
        write_sampling_augmentation_report(wrapper_args.so101_augmentation_report, augmentation, sys.argv[1:])
    logging.info("TensorBoard logs: %s", logger.log_dir)
    if remaining_steps <= 0:
        logging.info("No remaining steps: cfg.steps=%s resume_step=%s", cfg.steps, resume_step)
        return
    trainer.fit(module, train_dataloaders=dataloader)
    checkpoint_callback.save_final(trainer)


def _import_lightning() -> Any:
    try:
        import lightning.pytorch as lightning
    except ModuleNotFoundError:
        try:
            import pytorch_lightning as lightning
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyTorch Lightning is required. Install with: pip install lightning tensorboard"
            ) from exc
    return lightning


def _import_tensorboard_logger() -> Any:
    try:
        from lightning.pytorch.loggers import TensorBoardLogger
    except ModuleNotFoundError:
        from pytorch_lightning.loggers import TensorBoardLogger
    return TensorBoardLogger


def _post_checkpoint_loop_command(args: argparse.Namespace) -> list[str] | None:
    if not args.post_checkpoint_loop_command_json:
        return None
    try:
        command = json.loads(args.post_checkpoint_loop_command_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--post-checkpoint-loop-command-json must be a JSON list: {exc}") from exc
    if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
        raise SystemExit("--post-checkpoint-loop-command-json must be a non-empty JSON list of strings")
    return list(command)


def _resolve_resume_checkpoint_path(cfg: TrainPipelineConfig, wrapper_args: argparse.Namespace) -> Path:
    if wrapper_args.so101_resume_checkpoint_path is not None:
        return Path(wrapper_args.so101_resume_checkpoint_path)
    if cfg.checkpoint_path is not None:
        return Path(cfg.checkpoint_path)
    checkpoint_root = Path(cfg.output_dir) / "checkpoints"
    last_path = checkpoint_root / "last"
    if last_path.exists():
        return last_path
    candidates = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ] if checkpoint_root.exists() else []
    if candidates:
        return max(candidates, key=lambda path: int(path.name))
    raise ValueError(
        "cfg.resume is true but no checkpoint path was provided and no checkpoint "
        f"was found under {checkpoint_root}"
    )


def _make_dataset(
    cfg: TrainPipelineConfig,
    augmentation: SamplingAugmentationConfig,
    wrapper_args: argparse.Namespace,
) -> Any:
    entries = _train_dataset_entries(wrapper_args)
    if entries:
        return _make_concat_train_dataset(cfg, augmentation, entries)
    dataset = make_dataset(cfg)
    cache_dir = os.environ.get("SO101_IMAGE_CACHE_DIR")
    if cache_dir:
        dataset = PredecodedImageCacheDataset(dataset, Path(cache_dir))
    if augmentation.enabled and not augmentation.gpu_image_augmentation:
        dataset = SamplingAugmentedDataset(dataset, augmentation)
    return dataset


def _train_dataset_entries(wrapper_args: argparse.Namespace) -> list[dict[str, Any]]:
    if not wrapper_args.train_datasets_json:
        return []
    try:
        entries = json.loads(wrapper_args.train_datasets_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--train-datasets-json must be valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise SystemExit("--train-datasets-json must be a JSON list")
    result = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"--train-datasets-json[{index}] must be an object")
        result.append(dict(entry))
    return result


def _make_concat_train_dataset(
    cfg: TrainPipelineConfig,
    augmentation: SamplingAugmentationConfig,
    entries: list[dict[str, Any]],
) -> Any:
    summary = validate_lerobot_dataset_infos(entries)
    print(
        "Virtual train_datasets: "
        f"datasets={summary['dataset_count']} "
        f"episodes={summary['total_episodes']} "
        f"frames={summary['total_frames']}",
        flush=True,
    )
    datasets = []
    names = []
    for index, entry in enumerate(entries):
        name = str(entry.get("name") or entry.get("repo_id") or f"train_dataset_{index}")
        repo_id = str(entry.get("repo_id") or cfg.dataset.repo_id)
        root = Path(str(entry["root"]))
        metadata = LeRobotDatasetMetadata(repo_id, root=root)
        delta_timestamps = resolve_delta_timestamps(cfg.policy, metadata)
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.dataset.video_backend,
        )
        cache_dir = entry.get("image_cache_dir")
        if cache_dir:
            dataset = PredecodedImageCacheDataset(dataset, Path(str(cache_dir)))
        if augmentation.enabled and not augmentation.gpu_image_augmentation:
            dataset = SamplingAugmentedDataset(dataset, augmentation)
        datasets.append(dataset)
        names.append(name)
    validate_compatible_lerobot_datasets(datasets)
    return LeRobotConcatDataset(datasets, names=names)


def _make_processors(cfg: TrainPipelineConfig, dataset: Any, policy: Any) -> tuple[Any, Any]:
    processor_pretrained_path = cfg.policy.pretrained_path
    processor_kwargs: dict[str, Any] = {}
    postprocessor_kwargs: dict[str, Any] = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta
    if processor_pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": cfg.policy.device},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
    return make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=processor_pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )


def _make_dataloader(cfg: TrainPipelineConfig, dataset: Any) -> torch.utils.data.DataLoader:
    if getattr(dataset, "requires_dataset_balanced_sampler", False):
        sampler = dataset.make_dataset_balanced_sampler(num_samples=len(dataset))
        shuffle = False
        logging.info(
            "Using dataset-balanced random sampler for %s train datasets: lengths=%s",
            len(getattr(dataset, "source_lengths", [])),
            getattr(dataset, "source_lengths", []),
        )
        print(
            "Train sampler: dataset_balanced_random "
            f"datasets={len(getattr(dataset, 'source_lengths', []))} "
            f"lengths={getattr(dataset, 'source_lengths', [])}",
            flush=True,
        )
    elif hasattr(cfg.policy, "drop_n_last_frames") and not getattr(dataset, "disable_episode_aware_sampler", False):
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=str(cfg.policy.device) == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )


def _make_validation_dataloader(
    cfg: TrainPipelineConfig,
    wrapper_args: argparse.Namespace,
) -> torch.utils.data.DataLoader | None:
    if wrapper_args.validation_dataset_root is None:
        return None
    repo_id = wrapper_args.validation_dataset_repo_id or cfg.dataset.repo_id
    metadata = LeRobotDatasetMetadata(repo_id, root=wrapper_args.validation_dataset_root)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, metadata)
    dataset = LeRobotDataset(
        repo_id,
        root=wrapper_args.validation_dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=cfg.dataset.video_backend,
    )
    if wrapper_args.validation_image_cache_dir is not None:
        dataset = PredecodedImageCacheDataset(dataset, wrapper_args.validation_image_cache_dir)
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=wrapper_args.validation_num_workers,
        batch_size=wrapper_args.validation_batch_size or cfg.batch_size,
        shuffle=False,
        pin_memory=str(cfg.policy.device) == "cuda",
        drop_last=False,
        prefetch_factor=2 if wrapper_args.validation_num_workers > 0 else None,
        persistent_workers=wrapper_args.validation_num_workers > 0,
    )


def _validation_step_interval(args: argparse.Namespace) -> int:
    value = args.validation_interval_steps
    if value is None:
        value = args.validation_every_n_train_steps
    return 0 if value is None else max(0, int(value))


def _validation_epoch_interval(args: argparse.Namespace, step_interval: int) -> int:
    if step_interval > 0:
        return 0
    value = args.validation_interval_epochs
    return 0 if value is None else max(0, int(value))


def _metadata_log_interval(args: argparse.Namespace) -> int:
    value = args.log_input_metadata_every_n_steps
    if value is None:
        value = args.log_input_images_every_n_steps
    return max(0, int(value))


def _action_chunk_jitter_metrics(policy: Any, batch: dict[str, Any]) -> dict[str, Any]:
    if ACTION not in batch or not hasattr(policy, "predict_action_chunk"):
        return {}
    was_training = getattr(policy, "training", False)
    try:
        if hasattr(policy, "reset"):
            policy.reset()
        predicted = policy.predict_action_chunk(batch)
        teacher = batch[ACTION]
        predicted = _valid_action_chunk(predicted, batch.get("action_is_pad"))
        teacher = _valid_action_chunk(teacher, batch.get("action_is_pad"))
        predicted_metrics = _chunk_smoothness_metrics(predicted.detach().float())
        teacher_metrics = _chunk_smoothness_metrics(teacher.detach().float())
        return {
            "predicted": predicted_metrics,
            "teacher": teacher_metrics,
            "ratio": _smoothness_ratios(predicted_metrics, teacher_metrics),
        }
    finally:
        if was_training:
            policy.train()


def _valid_action_chunk(chunk: torch.Tensor, action_is_pad: torch.Tensor | None) -> torch.Tensor:
    if action_is_pad is None:
        return chunk
    valid = (~action_is_pad).to(device=chunk.device)
    if valid.ndim == 2 and valid.shape[1] == chunk.shape[1]:
        return chunk * valid.unsqueeze(-1).to(chunk.dtype)
    return chunk


def _chunk_smoothness_metrics(chunks: torch.Tensor) -> dict[str, float]:
    if chunks.ndim != 3 or chunks.shape[1] < 2:
        return {}
    delta = chunks[:, 1:, :] - chunks[:, :-1, :]
    jerk = delta[:, 1:, :] - delta[:, :-1, :] if delta.shape[1] > 1 else torch.zeros_like(delta)
    endpoint = chunks[:, -1, :] - chunks[:, 0, :]
    path = delta.norm(dim=-1).sum(dim=-1)
    endpoint_norm = endpoint.norm(dim=-1)
    path_ratio = path / endpoint_norm.clamp_min(1e-8)
    return {
        "delta_abs_mean": float(delta.abs().mean().detach().cpu()),
        "delta_rms": float(torch.sqrt((delta * delta).mean()).detach().cpu()),
        "delta_l2_step_mean": float(delta.norm(dim=-1).mean().detach().cpu()),
        "jerk_abs_mean": float(jerk.abs().mean().detach().cpu()),
        "jerk_rms": float(torch.sqrt((jerk * jerk).mean()).detach().cpu()),
        "path_length_mean": float(path.mean().detach().cpu()),
        "path_to_endpoint_ratio_mean": float(path_ratio.mean().detach().cpu()),
    }


def _smoothness_ratios(predicted: dict[str, float], teacher: dict[str, float]) -> dict[str, float]:
    keys = (
        "delta_abs_mean",
        "delta_rms",
        "delta_l2_step_mean",
        "jerk_abs_mean",
        "jerk_rms",
        "path_length_mean",
        "path_to_endpoint_ratio_mean",
    )
    return {
        key: float(predicted[key]) / max(float(teacher[key]), 1e-8)
        for key in keys
        if key in predicted and key in teacher
    }


class _SO101LightningModule:
    def __new__(
        cls,
        *,
        LightningModule: type[Any],
        cfg: TrainPipelineConfig,
        policy: Any,
        preprocessor: Any,
        optimizer: Any,
        scheduler: Any,
        augmentation: SamplingAugmentationConfig,
        action_prefix_loss_steps: int,
        action_prefix_loss_weight: float,
        validation_dataloader: torch.utils.data.DataLoader | None,
        validation_step_interval: int,
        validation_epoch_interval: int,
        input_image_cameras: tuple[str, ...],
        log_input_images_every_n_steps: int,
        log_input_metadata_every_n_steps: int,
        initial_step: int = 0,
    ) -> Any:
        class SO101LightningModuleImpl(LightningModule):
            def __init__(self) -> None:
                super().__init__()
                self.cfg = cfg
                self.policy = policy
                self.preprocessor = preprocessor
                self._optimizer = optimizer
                self._scheduler = scheduler
                self.augmentation = augmentation
                self.action_prefix_loss_steps = max(0, int(action_prefix_loss_steps))
                self.action_prefix_loss_weight = max(0.0, float(action_prefix_loss_weight))
                self.validation_dataloader = validation_dataloader
                self.validation_step_interval = max(0, int(validation_step_interval))
                self.validation_epoch_interval = max(0, int(validation_epoch_interval))
                self.validation_iter: Any | None = None
                self.initial_step = max(0, int(initial_step))
                self.train_batches_seen = self.initial_step
                self.input_image_cameras = input_image_cameras
                self.log_input_images_every_n_steps = max(0, int(log_input_images_every_n_steps))
                self.log_input_metadata_every_n_steps = max(0, int(log_input_metadata_every_n_steps))
                self._last_step_started = 0.0

            def forward(self, batch: dict[str, Any]) -> Any:
                return self.policy.forward(batch)

            def training_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
                started = time.perf_counter()
                self.policy.train()
                raw_batch = batch
                batch = self.preprocessor(batch)
                dataloading_s = time.perf_counter() - self._last_step_started if self._last_step_started else 0.0
                if self.augmentation.enabled and self.augmentation.gpu_image_augmentation:
                    augment_batch_on_device(batch, self.augmentation)
                self._log_input_images(batch, split="train")
                self._log_input_metadata(raw_batch, batch, split="train")
                loss, output_dict = _forward_policy_with_optional_prefix_loss(
                    self.policy,
                    batch,
                    prefix_steps=self.action_prefix_loss_steps,
                    prefix_weight=self.action_prefix_loss_weight,
                )
                batch_size = _batch_size(batch)
                update_s = time.perf_counter() - started
                self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
                self.log("important/train_loss", loss, on_step=True, on_epoch=False, batch_size=batch_size)
                self.log("train/batch_size", float(batch_size), on_step=True, on_epoch=False, batch_size=batch_size)
                self.log("train/update_s", update_s, on_step=True, on_epoch=False, batch_size=batch_size)
                self.log("train/data_s", dataloading_s, on_step=True, on_epoch=False, batch_size=batch_size)
                self.log(
                    "train/update_s_per_sample",
                    _seconds_per_sample(update_s, batch_size),
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch_size,
                )
                self.log(
                    "train/data_s_per_sample",
                    _seconds_per_sample(dataloading_s, batch_size),
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch_size,
                )
                self.log(
                    "train/samples_per_s",
                    _samples_per_second(batch_size, update_s),
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch_size,
                )
                self._log_system_metrics(batch_size=batch_size)
                for key, value in (output_dict or {}).items():
                    if key == "loss":
                        continue
                    if torch.is_tensor(value) and value.numel() == 1:
                        value = value.detach()
                    if isinstance(value, (int, float)):
                        self.log(f"train/{key}", float(value), on_step=True, on_epoch=False, batch_size=batch_size)
                    elif torch.is_tensor(value) and value.numel() == 1:
                        self.log(f"train/{key}", value, on_step=True, on_epoch=False, batch_size=batch_size)
                self.train_batches_seen += 1
                self._run_step_validation_if_due(completed_step=self.train_batches_seen)
                self._last_step_started = time.perf_counter()
                return loss

            def _log_system_metrics(self, *, batch_size: int) -> None:
                step = self.train_batches_seen + 1
                if step != 1 and step % 50 != 0:
                    return
                for tag, value in _system_metrics_for_current_process().items():
                    self.log(tag, value, on_step=True, on_epoch=False, batch_size=batch_size)

            def on_train_epoch_end(self) -> None:
                if self.validation_dataloader is None or self.validation_epoch_interval <= 0:
                    return
                epoch = int(getattr(self.trainer, "current_epoch", 0)) + 1
                if epoch > 0 and epoch % self.validation_epoch_interval == 0:
                    self._run_scheduled_validation(log_step=self._absolute_step())

            def _run_step_validation_if_due(self, *, completed_step: int) -> None:
                if self.validation_dataloader is None or self.validation_step_interval <= 0:
                    return
                if completed_step > 0 and completed_step % self.validation_step_interval == 0:
                    self._run_scheduled_validation(log_step=completed_step)

            def _run_scheduled_validation(self, *, log_step: int) -> None:
                if self.validation_iter is None:
                    self.validation_iter = cycle(self.validation_dataloader)
                print(f"Running validation batch at step {log_step}", flush=True)
                self.run_validation_batch(next(self.validation_iter), log_step=log_step)

            def configure_optimizers(self) -> Any:
                if self._scheduler is None:
                    return self._optimizer
                return {
                    "optimizer": self._optimizer,
                    "lr_scheduler": {
                        "scheduler": self._scheduler,
                        "interval": "step",
                        "frequency": 1,
                    },
                }

            def run_validation_batch(self, batch: dict[str, Any], *, log_step: int | None = None) -> Any:
                was_training = self.policy.training
                self.policy.eval()
                with torch.no_grad():
                    raw_batch = batch
                    batch = self.preprocessor(batch)
                    self._log_input_images(batch, split="val", log_step=log_step)
                    self._log_input_metadata(raw_batch, batch, split="val", log_step=log_step)
                    loss, output_dict = _forward_policy_with_optional_prefix_loss(
                        self.policy,
                        batch,
                        prefix_steps=self.action_prefix_loss_steps,
                        prefix_weight=self.action_prefix_loss_weight,
                    )
                    jitter_metrics = _action_chunk_jitter_metrics(self.policy, batch)
                batch_size = _batch_size(batch)
                self._log_validation_scalar("val/loss", loss, batch_size=batch_size, log_step=log_step)
                self._log_validation_scalar("important/val_loss", loss, batch_size=batch_size, log_step=log_step)
                for key, value in (output_dict or {}).items():
                    if key == "loss":
                        continue
                    self._log_validation_scalar(f"val/{key}", value, batch_size=batch_size, log_step=log_step)
                self._log_action_jitter_metrics(jitter_metrics, batch_size=batch_size, log_step=log_step)
                if was_training:
                    self.policy.train()
                self._log_validation_scalar(
                    "val/batch_size",
                    float(batch_size),
                    batch_size=batch_size,
                    log_step=log_step,
                )
                return loss

            def _log_validation_scalar(
                self,
                tag: str,
                value: Any,
                *,
                batch_size: int,
                log_step: int | None,
            ) -> None:
                if torch.is_tensor(value) and value.numel() == 1:
                    value = value.detach()
                if torch.is_tensor(value) and value.numel() != 1:
                    return
                if isinstance(value, (int, float)):
                    scalar = float(value)
                elif torch.is_tensor(value):
                    scalar = float(value.cpu())
                else:
                    return
                experiment = getattr(getattr(self, "logger", None), "experiment", None)
                if experiment is not None and hasattr(experiment, "add_scalar"):
                    experiment.add_scalar(tag, scalar, global_step=int(log_step or self.trainer.global_step))
                    return
                self.log(tag, scalar, on_step=True, on_epoch=False, batch_size=batch_size)

            def _log_input_images(
                self,
                batch: dict[str, Any],
                *,
                split: str,
                log_step: int | None = None,
            ) -> None:
                if self.log_input_images_every_n_steps <= 0:
                    return
                step = int(log_step if log_step is not None else getattr(self.trainer, "global_step", 0))
                if log_step is None:
                    step += self.initial_step
                if step % self.log_input_images_every_n_steps != 0:
                    return
                experiment = getattr(getattr(self, "logger", None), "experiment", None)
                if experiment is None or not hasattr(experiment, "add_image"):
                    return
                for camera in self.input_image_cameras:
                    key = f"observation.images.{camera}"
                    if key not in batch:
                        continue
                    image = _tensorboard_image_grid(batch[key]) if split == "val" else _tensorboard_image(batch[key])
                    if image is None:
                        continue
                    experiment.add_image(f"{split}/input_{camera}", image, global_step=step)

            def _log_input_metadata(
                self,
                raw_batch: dict[str, Any],
                processed_batch: dict[str, Any],
                *,
                split: str,
                log_step: int | None = None,
            ) -> None:
                if self.log_input_metadata_every_n_steps <= 0:
                    return
                step = int(log_step if log_step is not None else getattr(self.trainer, "global_step", 0))
                if log_step is None:
                    step += self.initial_step
                if step % self.log_input_metadata_every_n_steps != 0:
                    return
                experiment = getattr(getattr(self, "logger", None), "experiment", None)
                if experiment is None:
                    return
                prompt = _first_text(raw_batch, "task", "subtask")
                if prompt and hasattr(experiment, "add_text"):
                    experiment.add_text(f"{split}/input_prompt", _markdown_code(prompt), global_step=step)
                camera_contract = _camera_contract_markdown()
                if hasattr(experiment, "add_text"):
                    experiment.add_text(f"{split}/input_camera_contract", camera_contract, global_step=step)
                state = processed_batch.get("observation.state")
                raw_state = raw_batch.get("observation.state")
                state_text = _state_markdown(
                    title="observation.state",
                    state=state,
                    raw_state=raw_state,
                )
                if state_text and hasattr(experiment, "add_text"):
                    experiment.add_text(f"{split}/input_motor_state", state_text, global_step=step)
                vector = _first_vector(state)
                if vector is not None and hasattr(experiment, "add_scalar"):
                    for index, value in enumerate(vector):
                        experiment.add_scalar(
                            f"{split}/input_motor_state/dim_{index:02d}",
                            float(value),
                            global_step=step,
                        )

            def _log_action_jitter_metrics(
                self,
                metrics: dict[str, Any],
                *,
                batch_size: int,
                log_step: int | None,
            ) -> None:
                for namespace in ("predicted", "teacher", "ratio"):
                    values = metrics.get(namespace)
                    if not isinstance(values, dict):
                        continue
                    for key, value in values.items():
                        if isinstance(value, (int, float)):
                            self._log_validation_scalar(
                                f"val/action_jitter/{namespace}/{key}",
                                float(value),
                                batch_size=batch_size,
                                log_step=log_step,
                            )

            def _absolute_step(self) -> int:
                return self.initial_step + int(getattr(self.trainer, "global_step", 0))

        return SO101LightningModuleImpl()


def _make_lerobot_checkpoint_callback(Callback: type[Any], **kwargs: Any) -> Any:
    class LeRobotCheckpointCallback(Callback):
        def __init__(
            self,
            *,
            cfg: TrainPipelineConfig,
            policy_module: Any,
            preprocessor: Any,
            postprocessor: Any,
            save_freq: int,
            enabled: bool,
            initial_step: int,
            post_checkpoint_loop_command: list[str] | None,
        ) -> None:
            super().__init__()
            self.cfg = cfg
            self.policy_module = policy_module
            self.preprocessor = preprocessor
            self.postprocessor = postprocessor
            self.save_freq = max(1, int(save_freq))
            self.enabled = enabled
            self.initial_step = max(0, int(initial_step))
            self.post_checkpoint_loop_command = list(post_checkpoint_loop_command or [])
            self.saved_steps: set[int] = set()

        def on_train_batch_end(
            self,
            trainer: Any,
            pl_module: Any,
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ) -> None:
            del outputs, batch, batch_idx
            step = self.initial_step + int(trainer.global_step)
            if not self.enabled or step <= 0:
                return
            if step % self.save_freq == 0 or step >= int(self.cfg.steps):
                self._save(trainer, pl_module, step)

        def save_final(self, trainer: Any) -> None:
            step = self.initial_step + int(trainer.global_step)
            if self.enabled and step > 0:
                self._save(trainer, self.policy_module, step)

        def _save(self, trainer: Any, pl_module: Any, step: int) -> None:
            if step in self.saved_steps:
                return
            checkpoint_dir = get_step_checkpoint_dir(self.cfg.output_dir, self.cfg.steps, step)
            optimizer = trainer.optimizers[0] if trainer.optimizers else None
            scheduler = _first_scheduler(trainer)
            with nullcontext():
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=self.cfg,
                    policy=pl_module.policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    preprocessor=self.preprocessor,
                    postprocessor=self.postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
            self.saved_steps.add(step)
            self._run_post_checkpoint_loop_test(checkpoint_dir=checkpoint_dir, step=step)

        def _run_post_checkpoint_loop_test(self, *, checkpoint_dir: Path, step: int) -> None:
            if not self.post_checkpoint_loop_command:
                return
            env = {
                **os.environ,
                "SO101_CHECKPOINT_DIR": str(checkpoint_dir),
                "SO101_CHECKPOINT_STEP": str(step),
            }
            print(f"Running closed-loop test after checkpoint step {step}", flush=True)
            subprocess.run(self.post_checkpoint_loop_command, check=True, env=env)

    return LeRobotCheckpointCallback(**kwargs)


def _forward_policy_with_optional_prefix_loss(
    policy: Any,
    batch: dict[str, Any],
    *,
    prefix_steps: int,
    prefix_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if prefix_steps <= 0 or prefix_weight == 1.0 or getattr(policy, "name", None) != "smolvla":
        return policy.forward(batch)
    if not all(hasattr(policy, name) for name in ("prepare_images", "prepare_state", "prepare_action")):
        return policy.forward(batch)
    return _forward_smolvla_with_prefix_loss(
        policy,
        batch,
        prefix_steps=prefix_steps,
        prefix_weight=prefix_weight,
    )


def _forward_smolvla_with_prefix_loss(
    policy: Any,
    batch: dict[str, Any],
    *,
    prefix_steps: int,
    prefix_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if getattr(policy.config, "adapt_to_pi_aloha", False):
        batch[OBS_STATE] = policy._pi_aloha_decode_state(batch[OBS_STATE])
        batch[ACTION] = policy._pi_aloha_encode_actions_inv(batch[ACTION])

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
    actions = policy.prepare_action(batch)
    actions_is_pad = batch.get("action_is_pad")
    losses = policy.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, None, None)
    original_action_dim = policy.config.action_feature.shape[0]
    losses = losses[:, :, :original_action_dim]
    loss_dict: dict[str, Any] = {
        "losses_after_forward": losses.clone().mean().item(),
    }

    if actions_is_pad is not None:
        in_episode_bound = ~actions_is_pad
        losses = losses * in_episode_bound.unsqueeze(-1)
        loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()
    losses = losses[:, :, : policy.config.max_action_dim]
    loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()
    unweighted_loss = losses.mean()
    weights = _action_prefix_weights(
        losses,
        prefix_steps=min(prefix_steps, int(losses.shape[1])),
        prefix_weight=prefix_weight,
        actions_is_pad=actions_is_pad,
    )
    weighted_loss = (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(losses.dtype).eps)
    loss_dict["loss"] = weighted_loss.item()
    loss_dict["loss_unweighted"] = unweighted_loss.detach().item()
    loss_dict["loss_prefix_weight"] = float(prefix_weight)
    loss_dict["loss_prefix_steps"] = int(min(prefix_steps, int(losses.shape[1])))
    return weighted_loss, loss_dict


def _action_prefix_weights(
    losses: torch.Tensor,
    *,
    prefix_steps: int,
    prefix_weight: float,
    actions_is_pad: torch.Tensor | None,
) -> torch.Tensor:
    weights = torch.ones((1, int(losses.shape[1]), 1), dtype=losses.dtype, device=losses.device)
    if prefix_steps > 0:
        weights[:, :prefix_steps, :] = float(prefix_weight)
    weights = weights.expand_as(losses)
    if actions_is_pad is not None:
        valid = (~actions_is_pad).to(dtype=losses.dtype, device=losses.device).unsqueeze(-1)
        weights = weights * valid
    return weights


def _first_scheduler(trainer: Any) -> Any | None:
    configs = getattr(trainer, "lr_scheduler_configs", None) or []
    if not configs:
        return None
    return configs[0].scheduler


def _batch_size(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
    return 1


def _seconds_per_sample(seconds: float, batch_size: int) -> float:
    return float(seconds) / max(1, int(batch_size))


def _samples_per_second(batch_size: int, seconds: float) -> float:
    seconds = max(float(seconds), 1e-12)
    return float(max(1, int(batch_size))) / seconds


def _system_metrics_for_current_process() -> dict[str, float]:
    mps_available = _mps_available()
    metrics: dict[str, float] = {
        "system/accelerator_available": 1.0 if mps_available else 0.0,
        "system/mps_available": 1.0 if mps_available else 0.0,
    }
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except OSError:
        load_1m = load_5m = load_15m = 0.0
    metrics.update(
        {
            "system/load_avg_1m": float(load_1m),
            "system/load_avg_5m": float(load_5m),
            "system/load_avg_15m": float(load_15m),
        }
    )
    metrics.update({f"system/train_process_{key}": value for key, value in _current_process_metrics().items()})
    metrics.update({f"system/host_memory_{key}": value for key, value in _host_memory_metrics().items()})
    return metrics


def _mps_available() -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _current_process_metrics() -> dict[str, float]:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "pcpu=,pmem=,rss=", "-p", str(os.getpid())],
            text=True,
        )
    except Exception:
        return {}
    parts = output.strip().split()
    if len(parts) < 3:
        return {}
    return {
        "cpu_percent": _safe_float(parts[0]),
        "mem_percent": _safe_float(parts[1]),
        "rss_mb": _safe_float(parts[2]) / 1024.0,
    }


def _host_memory_metrics() -> dict[str, float]:
    if Path("/proc/meminfo").exists():
        return _linux_host_memory_metrics()
    if shutil.which("sysctl") and shutil.which("vm_stat"):
        return _macos_host_memory_metrics()
    return {}


def _linux_host_memory_metrics() -> dict[str, float]:
    rows: dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows[parts[0].rstrip(":")] = _safe_float(parts[1]) / 1024.0
    total = rows.get("MemTotal", 0.0)
    available = rows.get("MemAvailable", 0.0)
    return _memory_metrics(total_mb=total, available_mb=available)


def _macos_host_memory_metrics() -> dict[str, float]:
    try:
        total_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        vm_output = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return {}
    page_size = 4096
    rows: dict[str, int] = {}
    for line in vm_output.splitlines():
        if "page size of" in line:
            parts = line.split()
            for index, part in enumerate(parts):
                if part == "of" and index + 1 < len(parts):
                    try:
                        page_size = int(parts[index + 1])
                    except ValueError:
                        pass
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        digits = value.strip().rstrip(".").replace(".", "")
        if digits.isdigit():
            rows[key.strip()] = int(digits)
    available_pages = rows.get("Pages free", 0) + rows.get("Pages inactive", 0) + rows.get("Pages speculative", 0)
    return _memory_metrics(
        total_mb=total_bytes / (1024.0 * 1024.0),
        available_mb=(available_pages * page_size) / (1024.0 * 1024.0),
    )


def _memory_metrics(*, total_mb: float, available_mb: float) -> dict[str, float]:
    if total_mb <= 0:
        return {}
    used_mb = max(0.0, total_mb - available_mb)
    return {
        "total_mb": total_mb,
        "used_mb": used_mb,
        "available_mb": available_mb,
        "used_percent": 100.0 * used_mb / total_mb,
        "available_percent": 100.0 * available_mb / total_mb,
    }


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _tensorboard_image(value: Any) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    image = value.detach()
    while image.ndim > 3:
        image = image[0]
    if image.ndim != 3:
        return None
    if image.shape[0] not in (1, 3, 4) and image.shape[-1] in (1, 3, 4):
        image = image.permute(2, 0, 1)
    if image.shape[0] == 4:
        image = image[:3]
    if image.shape[0] not in (1, 3):
        return None
    image = image.float().cpu()
    if image.numel() == 0:
        return None
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > 2.0:
        image = image / 255.0
    elif min_value < 0.0:
        image = (image + 1.0) / 2.0
    return image.clamp(0.0, 1.0)


def _tensorboard_image_grid(value: Any, *, max_images: int = 16) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    tensor = value.detach()
    if tensor.ndim == 3:
        return _tensorboard_image(tensor)
    while tensor.ndim > 4:
        tensor = tensor[0]
    if tensor.ndim != 4:
        return _tensorboard_image(tensor)
    if tensor.shape[1] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.shape[1] == 4:
        tensor = tensor[:, :3]
    if tensor.shape[1] not in (1, 3):
        return _tensorboard_image(tensor)
    images = tensor[: max(1, int(max_images))].float().cpu()
    if images.numel() == 0:
        return None
    if float(images.max()) > 2.0:
        images = images / 255.0
    elif float(images.min()) < 0.0:
        images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)
    rows = int(max(1, round(float(images.shape[0]) ** 0.5)))
    cols = int((images.shape[0] + rows - 1) // rows)
    c, h, w = int(images.shape[1]), int(images.shape[2]), int(images.shape[3])
    grid = torch.zeros((c, rows * h, cols * w), dtype=images.dtype)
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        grid[:, row * h : (row + 1) * h, col * w : (col + 1) * w] = image
    return grid


def _camera_contract_markdown() -> str:
    return "\n".join(
        [
            "### SO101 camera contract",
            "",
            "| model input | source view |",
            "| --- | --- |",
            "| observation.images.camera1 | egocentric_cam |",
            "| observation.images.camera2 | wrist_cam |",
            "| observation.images.camera3 | wrist_cam duplicate |",
        ]
    )


def _first_text(batch: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = batch.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        if torch.is_tensor(value):
            continue
        return str(value)
    return ""


def _markdown_code(text: str) -> str:
    escaped = text.replace("```", "'''")
    return f"```text\n{escaped}\n```"


def _state_markdown(*, title: str, state: Any, raw_state: Any | None = None) -> str:
    vector = _first_vector(state)
    if vector is None:
        return ""
    raw_vector = _first_vector(raw_state)
    lines = [f"### {title}", "", "| dim | model_input | raw |", "| ---: | ---: | ---: |"]
    for index, value in enumerate(vector):
        raw = "" if raw_vector is None or index >= len(raw_vector) else f"{raw_vector[index]:.6g}"
        lines.append(f"| {index} | {value:.6g} | {raw} |")
    return "\n".join(lines)


def _first_vector(value: Any) -> list[float] | None:
    if value is None or not torch.is_tensor(value):
        return None
    tensor = value.detach().float().cpu()
    while tensor.ndim > 1:
        tensor = tensor[0]
    if tensor.ndim != 1:
        return None
    return [float(item) for item in tensor.tolist()]


def _parse_devices(value: str) -> Any:
    if value == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
