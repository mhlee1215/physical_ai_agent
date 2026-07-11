#!/usr/bin/env python3

import argparse
import dataclasses
import json
import logging
import os
import re
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
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
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
from physical_ai_agent.policies.so101_valid_mask import (
    SO101ValidMaskConfig,
    SO101ValidMaskHead,
    load_valid_mask_head,
    save_valid_mask_head,
    valid_labels_from_action_is_pad,
)
from physical_ai_agent.policies.so101_visual_servo_head import (
    SO101VisualServoHead,
    SO101VisualServoHeadConfig,
    load_visual_servo_head,
    save_visual_servo_head,
    visual_servo_loss,
)
from physical_ai_agent.so101_lerobot_concat import (
    GridBinBalancedDataset,
    LeRobotConcatDataset,
    validate_compatible_lerobot_datasets,
    validate_lerobot_dataset_infos,
)
from physical_ai_agent.so101_resolution_contract import require_lerobot_dataset_256

VAL_LOSS_TAG = "val/loss"
IMPORTANT_VAL_LOSS_TAG = "important/val_loss"


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
    parser.add_argument("--so101-image-blur-prob", type=float, default=0.0)
    parser.add_argument("--so101-image-blur-kernel-size", type=int, default=5)
    parser.add_argument("--so101-image-motion-blur-prob", type=float, default=0.0)
    parser.add_argument("--so101-image-motion-blur-kernel-size", type=int, default=7)
    parser.add_argument("--so101-image-noise-std", type=float, default=0.0)
    parser.add_argument("--so101-image-color-jitter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--so101-image-color-jitter-strength", type=float, default=0.08)
    parser.add_argument("--so101-image-sharpness-jitter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--so101-image-affine-degrees", type=float, default=0.0)
    parser.add_argument("--so101-image-affine-translate", type=float, default=0.0)
    parser.add_argument("--so101-gpu-image-augmentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--so101-action-prefix-loss-steps", type=int, default=0)
    parser.add_argument("--so101-action-prefix-loss-weight", type=float, default=1.0)
    parser.add_argument("--so101-action-chunk-consistency-steps", type=int, default=0)
    parser.add_argument("--so101-action-chunk-consistency-weight", type=float, default=0.0)
    parser.add_argument("--so101-action-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--so101-action-gripper-transition-loss-weight", type=float, default=0.0)
    parser.add_argument("--so101-action-terminal-loss-steps", type=int, default=0)
    parser.add_argument("--so101-action-terminal-loss-weight", type=float, default=1.0)
    parser.add_argument("--so101-action-smoothness-loss-weight", type=float, default=0.0)
    parser.add_argument("--so101-action-smoothness-include-gripper", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--so101-valid-mask-loss-weight", type=float, default=0.05)
    parser.add_argument("--so101-valid-mask-hidden-dim", type=int, default=128)
    parser.add_argument("--so101-visual-servo-loss-weight", type=float, default=0.0)
    parser.add_argument("--so101-visual-servo-hidden-dim", type=int, default=128)
    parser.add_argument("--so101-image-cache-dir", type=Path)
    parser.add_argument("--train-grid-bin-sidecar", type=Path)
    parser.add_argument("--train-datasets-json")
    parser.add_argument("--train-dataset-source-spans-json")
    parser.add_argument("--validation-image-cache-dir", type=Path)
    parser.add_argument("--so101-augmentation-report", type=Path)
    parser.add_argument("--tensorboard-log-dir", type=Path)
    parser.add_argument(
        "--training-run-summary-path",
        type=Path,
        help="JSON launch summary written by start_so101_training.py and mirrored into TensorBoard text.",
    )
    parser.add_argument("--lightning-precision", default="16-mixed")
    parser.add_argument("--lightning-accelerator", default="auto")
    parser.add_argument("--lightning-devices", default="auto")
    parser.add_argument("--lightning-strategy", default="auto")
    parser.add_argument("--lightning-log-every-n-steps", type=int)
    parser.add_argument("--lightning-fast-dev-run", action="store_true")
    parser.add_argument("--validation-dataset-root", type=Path)
    parser.add_argument("--validation-dataset-repo-id")
    parser.add_argument("--validation-datasets-json")
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
        "--checkpoint-retention-policy",
        choices=["all", "best_val_and_closed_loop"],
        default="all",
        help=(
            "Checkpoint retention policy. best_val_and_closed_loop keeps only "
            "checkpoints/best_closed_loop, checkpoints/best_val_loss, and "
            "checkpoints/best_train_loss."
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
    wrapper_args, lerobot_args = parser.parse_known_args()
    wrapper_args.lerobot_args = list(lerobot_args)
    return wrapper_args, lerobot_args


def _apply_augmentation_env(args: argparse.Namespace) -> None:
    os.environ["SO101_STATE_JITTER_STD"] = str(args.so101_state_jitter_std)
    os.environ["SO101_STATE_JITTER_ARM_ONLY"] = "1" if args.so101_state_jitter_arm_only else "0"
    os.environ["SO101_STATE_DROPOUT_PROB"] = str(args.so101_state_dropout_prob)
    os.environ["SO101_STATE_DROPOUT_KEEP_GRIPPER"] = "1" if args.so101_state_dropout_keep_gripper else "0"
    os.environ["SO101_IMAGE_CAMERA_DROPOUT_PROB"] = str(args.so101_image_camera_dropout_prob)
    os.environ["SO101_IMAGE_PATCH_DROPOUT_PROB"] = str(args.so101_image_patch_dropout_prob)
    os.environ["SO101_IMAGE_PATCH_MASK_RATIO"] = str(args.so101_image_patch_mask_ratio)
    os.environ["SO101_IMAGE_BLUR_PROB"] = str(args.so101_image_blur_prob)
    os.environ["SO101_IMAGE_BLUR_KERNEL_SIZE"] = str(args.so101_image_blur_kernel_size)
    os.environ["SO101_IMAGE_MOTION_BLUR_PROB"] = str(args.so101_image_motion_blur_prob)
    os.environ["SO101_IMAGE_MOTION_BLUR_KERNEL_SIZE"] = str(args.so101_image_motion_blur_kernel_size)
    os.environ["SO101_IMAGE_NOISE_STD"] = str(args.so101_image_noise_std)
    os.environ["SO101_IMAGE_COLOR_JITTER"] = "1" if args.so101_image_color_jitter else "0"
    os.environ["SO101_IMAGE_COLOR_JITTER_STRENGTH"] = str(args.so101_image_color_jitter_strength)
    os.environ["SO101_IMAGE_SHARPNESS_JITTER"] = "1" if args.so101_image_sharpness_jitter else "0"
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
    _require_training_datasets_256(cfg, wrapper_args)
    dataset = _make_dataset(cfg, augmentation, wrapper_args)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    if cfg.peft is not None:
        policy = policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(cfg.peft))
    preprocessor, postprocessor = _make_processors(cfg, dataset, policy)
    valid_mask_head = _make_valid_mask_head(
        policy=policy,
        hidden_dim=int(wrapper_args.so101_valid_mask_hidden_dim),
    )
    _load_valid_mask_head_from_policy_path_if_available(
        valid_mask_head,
        _policy_path_from_config_or_args(cfg, wrapper_args),
    )
    visual_servo_head = None
    if float(wrapper_args.so101_visual_servo_loss_weight) > 0.0:
        visual_servo_head = _make_visual_servo_head(
            policy=policy,
            hidden_dim=int(wrapper_args.so101_visual_servo_hidden_dim),
            device=getattr(policy.config, "device", "cpu"),
        )
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    _add_valid_mask_head_to_optimizer(optimizer, valid_mask_head)
    _add_visual_servo_head_to_optimizer(optimizer, visual_servo_head)
    resume_step = 0
    if cfg.resume:
        checkpoint_path = _resolve_resume_checkpoint_path(cfg, wrapper_args)
        resume_step, optimizer, scheduler = load_training_state(checkpoint_path, optimizer, scheduler)
        _load_valid_mask_head_if_available(valid_mask_head, checkpoint_path)
        if visual_servo_head is not None:
            _load_visual_servo_head_if_available(visual_servo_head, checkpoint_path)
        logging.info("Resumed LeRobot training state from %s at step %s", checkpoint_path, resume_step)
        print(f"Resumed LeRobot training state from {checkpoint_path} at step {resume_step}", flush=True)

    dataloader = _make_dataloader(cfg, dataset)
    validation_dataloader = _make_validation_dataloader(cfg, wrapper_args)
    validation_dataset_loaders = _make_validation_dataset_dataloaders(cfg, wrapper_args)
    validation_step_interval = _validation_step_interval(wrapper_args)
    validation_epoch_interval = _validation_epoch_interval(wrapper_args, validation_step_interval)
    _require_aligned_checkpoint_validation_loop_cadence(
        cfg=cfg,
        wrapper_args=wrapper_args,
        validation_step_interval=validation_step_interval,
        validation_epoch_interval=validation_epoch_interval,
        dataloader=dataloader,
    )
    logging.info(
        "Validation schedule: enabled=%s step_interval=%s epoch_interval=%s",
        validation_dataloader is not None or bool(validation_dataset_loaders),
        validation_step_interval,
        validation_epoch_interval,
    )
    print(
        "Validation schedule: "
        f"enabled={validation_dataloader is not None or bool(validation_dataset_loaders)} "
        f"step_interval={validation_step_interval} "
        f"epoch_interval={validation_epoch_interval}",
        flush=True,
    )
    module = _SO101LightningModule(
        LightningModule=LightningModule,
        cfg=cfg,
        policy=policy,
        valid_mask_head=valid_mask_head,
        valid_mask_loss_weight=float(wrapper_args.so101_valid_mask_loss_weight),
        visual_servo_head=visual_servo_head,
        visual_servo_loss_weight=float(wrapper_args.so101_visual_servo_loss_weight),
        preprocessor=preprocessor,
        optimizer=optimizer,
        scheduler=scheduler,
        augmentation=augmentation,
        action_prefix_loss_steps=int(wrapper_args.so101_action_prefix_loss_steps),
        action_prefix_loss_weight=float(wrapper_args.so101_action_prefix_loss_weight),
        action_chunk_consistency_steps=int(wrapper_args.so101_action_chunk_consistency_steps),
        action_chunk_consistency_weight=float(wrapper_args.so101_action_chunk_consistency_weight),
        action_delta_loss_weight=float(wrapper_args.so101_action_delta_loss_weight),
        action_gripper_transition_loss_weight=float(wrapper_args.so101_action_gripper_transition_loss_weight),
        action_terminal_loss_steps=int(wrapper_args.so101_action_terminal_loss_steps),
        action_terminal_loss_weight=float(wrapper_args.so101_action_terminal_loss_weight),
        action_smoothness_loss_weight=float(wrapper_args.so101_action_smoothness_loss_weight),
        action_smoothness_include_gripper=bool(wrapper_args.so101_action_smoothness_include_gripper),
        validation_dataloader=validation_dataloader,
        validation_dataset_loaders=validation_dataset_loaders,
        validation_step_interval=validation_step_interval,
        validation_epoch_interval=validation_epoch_interval,
        input_image_cameras=_parse_csv(wrapper_args.log_input_image_cameras),
        log_input_images_every_n_steps=int(wrapper_args.log_input_images_every_n_steps),
        log_input_metadata_every_n_steps=_metadata_log_interval(wrapper_args),
        initial_step=resume_step,
        checkpoint_save_freq=int(cfg.save_freq),
        total_steps=int(cfg.steps),
    )
    tb_log_dir = (wrapper_args.tensorboard_log_dir or cfg.output_dir / "tensorboard").resolve()
    logger = TensorBoardLogger(save_dir=str(tb_log_dir), name="", version="")
    _log_training_run_texts(
        logger=logger,
        summary_path=wrapper_args.training_run_summary_path,
        cfg=cfg,
        augmentation=augmentation,
        wrapper_args=wrapper_args,
        lerobot_args=sys.argv[1:],
    )
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
        post_checkpoint_loop_commands=_post_checkpoint_loop_commands(wrapper_args),
        retention_policy=str(wrapper_args.checkpoint_retention_policy),
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


def _log_training_run_texts(
    *,
    logger: Any,
    summary_path: Path | None,
    cfg: TrainPipelineConfig,
    augmentation: SamplingAugmentationConfig,
    wrapper_args: argparse.Namespace,
    lerobot_args: list[str],
) -> None:
    experiment = getattr(logger, "experiment", None)
    if experiment is None or not hasattr(experiment, "add_text"):
        return
    summary = _load_training_run_summary(summary_path)
    sections = {
        "training/summary": _training_summary_markdown(summary, cfg=cfg, summary_path=summary_path),
        "training/datasets": _training_datasets_markdown(summary),
        "training/closed_loop": _training_closed_loop_markdown(summary),
        "training/augmentation": _training_augmentation_markdown(summary, augmentation=augmentation),
        "training/runtime": _training_runtime_markdown(summary, wrapper_args=wrapper_args, cfg=cfg),
        "training/command": _training_command_markdown(summary, lerobot_args=lerobot_args),
    }
    for tag, text in sections.items():
        experiment.add_text(tag, text, global_step=0)


def _load_training_run_summary(summary_path: Path | None) -> dict[str, Any]:
    if summary_path is None:
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"summary_path": str(summary_path), "summary_load_error": "file not found"}
    except json.JSONDecodeError as exc:
        return {"summary_path": str(summary_path), "summary_load_error": str(exc)}


def _training_summary_markdown(
    summary: dict[str, Any],
    *,
    cfg: TrainPipelineConfig,
    summary_path: Path | None,
) -> str:
    dataset_config = _summary_mapping(summary.get("dataset_config"))
    training = _summary_mapping(dataset_config.get("training"))
    closed_loop = _summary_mapping(dataset_config.get("closed_loop"))
    rows = {
        "run_dir": summary.get("run_dir", ""),
        "summary_path": str(summary_path) if summary_path is not None else "",
        "policy_type": getattr(cfg.policy, "type", ""),
        "policy_repo_id": training.get("policy_repo_id") or getattr(cfg.policy, "repo_id", ""),
        "output_dir": str(cfg.output_dir),
        "steps": getattr(cfg, "steps", ""),
        "save_freq": getattr(cfg, "save_freq", ""),
        "batch_size": training.get("batch_size", getattr(cfg, "batch_size", "")),
        "validation_interval_epochs": _arg_value(summary.get("train_cmd"), "--validation-interval-epochs"),
        "validation_interval_steps": _arg_value(summary.get("train_cmd"), "--validation-interval-steps"),
        "closed_loop_policy": _arg_value(summary.get("post_checkpoint_loop_cmd"), "--closed-loop-policy"),
        "closed_loop_runner": closed_loop.get("runner", ""),
        "tensorboard_url": summary.get("tensorboard_url", ""),
        "mobile_tensorboard_url": summary.get("mobile_tensorboard_url", ""),
    }
    return _markdown_table("SO101 Training Summary", rows)


def _training_datasets_markdown(summary: dict[str, Any]) -> str:
    dataset_config = _summary_mapping(summary.get("dataset_config"))
    lines = ["### SO101 Dataset Contract", ""]
    if dataset_config.get("name"):
        lines.append(f"- config: `{dataset_config.get('name')}`")
    if dataset_config.get("task"):
        lines.append(f"- task: `{dataset_config.get('task')}`")
    if dataset_config.get("scenario"):
        lines.append(f"- scenario: `{dataset_config.get('scenario')}`")
    lines.append("")
    camera_contract = _summary_mapping(dataset_config.get("camera_contract"))
    if camera_contract:
        lines.append("#### Cameras")
        lines.extend(_dict_table_lines(camera_contract))
        lines.append("")
    train_datasets = dataset_config.get("train_datasets")
    if isinstance(train_datasets, list):
        lines.append("#### Train Datasets")
        lines.extend(_dataset_table_lines(train_datasets))
        lines.append("")
    train_dataset = dataset_config.get("train_dataset")
    if isinstance(train_dataset, dict):
        lines.append("#### Train Dataset")
        lines.extend(_dataset_table_lines([train_dataset]))
        lines.append("")
    validation = _summary_mapping(dataset_config.get("validation_dataset"))
    validation_sources = validation.get("hf_resolved_sources") or validation.get("hf_merge_sources")
    if isinstance(validation_sources, list):
        lines.append("#### Validation Datasets")
        lines.extend(_dataset_table_lines(validation_sources))
    elif validation:
        lines.append("#### Validation Dataset")
        lines.extend(_dataset_table_lines([validation]))
    return "\n".join(lines).strip()


def _training_closed_loop_markdown(summary: dict[str, Any]) -> str:
    dataset_config = _summary_mapping(summary.get("dataset_config"))
    closed_loop = _summary_mapping(dataset_config.get("closed_loop"))
    lines = ["### Closed-Loop Evaluation", ""]
    lines.extend(
        _dict_table_lines(
            {
                "runner": closed_loop.get("runner", ""),
                "env_id": closed_loop.get("env_id", ""),
                "task_prompt": closed_loop.get("task_prompt", ""),
                "qwen_object": closed_loop.get("qwen_object", ""),
                "valid_mask_checkpoint": closed_loop.get("valid_mask_checkpoint", ""),
                "success_metric": closed_loop.get("success_metric", ""),
            }
        )
    )
    test_cases = closed_loop.get("test_cases")
    if isinstance(test_cases, list):
        lines.extend(["", "#### Test Cases", "", "| id | episodes | steps | seed | start_contract | task_prompt | plan_json |", "| --- | ---: | ---: | ---: | --- | --- | --- |"])
        for test_case in test_cases:
            if not isinstance(test_case, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    _escape_md(
                        test_case.get(key, "")
                    )
                    for key in ("id", "episodes", "steps", "seed", "start_contract", "task_prompt", "plan_json")
                )
                + " |"
            )
    commands = summary.get("post_checkpoint_loop_cmds")
    if isinstance(commands, list) and commands:
        lines.extend(["", "#### Expanded Loop Commands", ""])
        for index, command in enumerate(commands, start=1):
            lines.append(f"```text\n# loop command {index}\n{_shell_join(command)}\n```")
    return "\n".join(lines).strip()


def _training_augmentation_markdown(
    summary: dict[str, Any],
    *,
    augmentation: SamplingAugmentationConfig,
) -> str:
    dataset_config = _summary_mapping(summary.get("dataset_config"))
    configured = _summary_mapping(dataset_config.get("augmentation"))
    actual = {
        field.name: getattr(augmentation, field.name)
        for field in dataclasses.fields(augmentation)
        if hasattr(augmentation, field.name)
    }
    lines = ["### Train-Time Augmentation", "", "Validation and closed-loop inputs are unaugmented.", ""]
    if configured:
        lines.append("#### Configured")
        lines.extend(_dict_table_lines(configured))
        lines.append("")
    lines.append("#### Effective Environment")
    lines.extend(_dict_table_lines(actual))
    return "\n".join(lines).strip()


def _training_runtime_markdown(
    summary: dict[str, Any],
    *,
    wrapper_args: argparse.Namespace,
    cfg: TrainPipelineConfig,
) -> str:
    rows = {
        "runtime_platform": _summary_mapping(summary.get("runtime_contract")).get("runtime_platform", ""),
        "training_device": _summary_mapping(summary.get("runtime_contract")).get("training_device", ""),
        "lightning_accelerator": wrapper_args.lightning_accelerator,
        "lightning_devices": wrapper_args.lightning_devices,
        "lightning_precision": wrapper_args.lightning_precision,
        "resume": getattr(cfg, "resume", ""),
        "resume_checkpoint": wrapper_args.so101_resume_checkpoint_path or "",
        "local_training_standard": _summary_mapping(summary.get("local_training_standard")).get("name", ""),
        "train_log": _summary_mapping(summary.get("logs")).get("train", ""),
        "tensorboard_log": _summary_mapping(summary.get("logs")).get("tensorboard", ""),
    }
    return _markdown_table("Runtime", rows)


def _training_command_markdown(summary: dict[str, Any], *, lerobot_args: list[str]) -> str:
    train_cmd = summary.get("train_cmd")
    lines = ["### Training Command", ""]
    if isinstance(train_cmd, list):
        lines.append(f"```text\n{_shell_join(train_cmd)}\n```")
    else:
        lines.append(f"```text\n{_shell_join([sys.argv[0], *lerobot_args])}\n```")
    tensorboard_cmd = summary.get("tensorboard_cmd")
    if isinstance(tensorboard_cmd, list):
        lines.extend(["", "### TensorBoard Command", "", f"```text\n{_shell_join(tensorboard_cmd)}\n```"])
    cache_cmds = summary.get("cache_build_cmds")
    if isinstance(cache_cmds, list) and cache_cmds:
        lines.extend(["", "### Cache Build Commands", ""])
        for command in cache_cmds:
            if isinstance(command, list):
                lines.append(f"```text\n{_shell_join(command)}\n```")
    return "\n".join(lines).strip()


def _summary_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _markdown_table(title: str, rows: dict[str, Any]) -> str:
    lines = [f"### {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        lines.append(f"| `{_escape_md(key)}` | {_escape_md(value)} |")
    return "\n".join(lines)


def _dict_table_lines(rows: dict[str, Any]) -> list[str]:
    lines = ["| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        lines.append(f"| `{_escape_md(key)}` | {_escape_md(value)} |")
    return lines


def _dataset_table_lines(rows: list[Any]) -> list[str]:
    lines = ["| name | repo_id | root | episodes | frames | hf_path |", "| --- | --- | --- | ---: | ---: | --- |"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                _escape_md(row.get(key, ""))
                for key in ("name", "repo_id", "root", "expected_episodes", "expected_frames", "hf_path_in_repo")
            )
            + " |"
        )
    return lines


def _arg_value(command: Any, flag: str) -> str:
    if not isinstance(command, list):
        return ""
    prefix = f"{flag}="
    for index, part in enumerate(command):
        if not isinstance(part, str):
            continue
        if part == flag and index + 1 < len(command):
            return str(command[index + 1])
        if part.startswith(prefix):
            return part[len(prefix):]
    return ""


def _shell_join(command: list[Any]) -> str:
    return " ".join(shlex_quote(str(part)) for part in command)


def shlex_quote(value: str) -> str:
    if not value:
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _escape_md(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True)
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


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


def _post_checkpoint_loop_commands(args: argparse.Namespace) -> list[list[str]]:
    if not args.post_checkpoint_loop_command_json:
        return []
    try:
        payload = json.loads(args.post_checkpoint_loop_command_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--post-checkpoint-loop-command-json must be JSON: {exc}") from exc
    if isinstance(payload, list) and payload and all(isinstance(part, str) for part in payload):
        return [list(payload)]
    if isinstance(payload, list) and all(
        isinstance(command, list) and command and all(isinstance(part, str) for part in command)
        for command in payload
    ):
        return [list(command) for command in payload]
    raise SystemExit(
        "--post-checkpoint-loop-command-json must be either one argv JSON list "
        "or a JSON list of argv lists"
    )


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


def _read_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _latest_closed_loop_success_for_command(command: list[str], checkpoint: str) -> float | None:
    run_dir = _command_arg_value(command, "--run-dir")
    test_id = _command_arg_value(command, "--closed-loop-test-id") or "default"
    if not run_dir:
        return None
    rows = [
        row
        for row in _read_jsonl_file(Path(run_dir) / "metrics" / "closed_loop_metrics.jsonl")
        if str(row.get("checkpoint")) == str(checkpoint)
        and str(row.get("test_id", "default")) == str(test_id)
    ]
    if not rows:
        return None
    return _float_or_none(rows[-1].get("success_rate"))


def _command_arg_value(command: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for index, part in enumerate(command):
        if part == flag and index + 1 < len(command):
            return str(command[index + 1])
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


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
    dataset = _maybe_wrap_visual_servo_labels(dataset, getattr(dataset, "root", None))
    if wrapper_args.train_grid_bin_sidecar is not None:
        dataset = GridBinBalancedDataset(dataset, wrapper_args.train_grid_bin_sidecar)
    source_spans = _train_dataset_source_spans(wrapper_args)
    if source_spans:
        dataset = _SourceAnnotatedDataset(dataset, source_spans)
    return dataset


def _require_training_datasets_256(cfg: TrainPipelineConfig, wrapper_args: argparse.Namespace) -> None:
    entries = _train_dataset_entries(wrapper_args)
    if entries:
        for index, entry in enumerate(entries):
            if entry.get("root"):
                require_lerobot_dataset_256(
                    Path(str(entry["root"])),
                    context=f"lerobot_train_so101_lightning train_datasets[{index}]",
                )
    else:
        dataset_root = getattr(cfg.dataset, "root", None)
        if dataset_root is not None:
            require_lerobot_dataset_256(
                Path(str(dataset_root)),
                context="lerobot_train_so101_lightning train dataset",
            )
    validation_entries = _validation_dataset_entries(wrapper_args)
    for index, entry in enumerate(validation_entries):
        if entry.get("root"):
            require_lerobot_dataset_256(
                Path(str(entry["root"])),
                context=f"lerobot_train_so101_lightning validation_datasets[{index}]",
            )
    if wrapper_args.validation_dataset_root is not None:
        require_lerobot_dataset_256(
            wrapper_args.validation_dataset_root,
            context="lerobot_train_so101_lightning validation dataset",
        )


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


def _train_dataset_source_spans(wrapper_args: argparse.Namespace) -> list[dict[str, Any]]:
    if not wrapper_args.train_dataset_source_spans_json:
        return []
    try:
        entries = json.loads(wrapper_args.train_dataset_source_spans_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--train-dataset-source-spans-json must be valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise SystemExit("--train-dataset-source-spans-json must be a JSON list")
    result = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"--train-dataset-source-spans-json[{index}] must be an object")
        if not entry.get("name"):
            raise SystemExit(f"--train-dataset-source-spans-json[{index}] must include name")
        length = entry.get("length", entry.get("expected_frames"))
        if length is None:
            raise SystemExit(f"--train-dataset-source-spans-json[{index}] must include length or expected_frames")
        result.append({"name": str(entry["name"]), "length": int(length)})
    return result


class _SourceAnnotatedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Any, source_spans: list[dict[str, Any]]) -> None:
        self.dataset = dataset
        self.source_names = [str(span["name"]) for span in source_spans]
        self.source_lengths = [int(span["length"]) for span in source_spans]
        self.source_ends: list[int] = []
        total = 0
        for length in self.source_lengths:
            total += max(0, int(length))
            self.source_ends.append(total)
        if total != len(dataset):
            raise SystemExit(
                "--train-dataset-source-spans-json total length does not match train dataset length: "
                f"spans={total} dataset={len(dataset)}"
            )
        self.meta = getattr(dataset, "meta")
        self.root = getattr(dataset, "root", None)
        self.repo_id = getattr(dataset, "repo_id", None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        source_index = self.source_index(index)
        item.setdefault("dataset_index", source_index)
        item.setdefault("dataset_name", self.source_names[source_index])
        start = 0 if source_index == 0 else self.source_ends[source_index - 1]
        item.setdefault("dataset_local_index", int(index) - start)
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def source_index(self, index: int) -> int:
        for source_index, end in enumerate(self.source_ends):
            if index < end:
                return source_index
        raise IndexError(index)


class _VisualServoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Any, dataset_root: Path) -> None:
        self.dataset = dataset
        self.dataset_root = Path(dataset_root)
        sidecar = self.dataset_root / "meta" / "visual_servo_labels" / "camera1_camera2_green_cube.parquet"
        if not sidecar.exists():
            raise FileNotFoundError(sidecar)
        import pandas as pd

        table = pd.read_parquet(sidecar).set_index("index")
        self.labels = table
        self.meta = getattr(dataset, "meta")
        self.root = getattr(dataset, "root", dataset_root)
        self.repo_id = getattr(dataset, "repo_id", None)
        for name in (
            "disable_episode_aware_sampler",
            "requires_grid_bin_balanced_sampler",
            "requires_dataset_balanced_sampler",
        ):
            if hasattr(dataset, name):
                setattr(self, name, getattr(dataset, name))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        dataset_index = int(item.get("index", index))
        try:
            row = self.labels.loc[dataset_index]
        except Exception:
            return item
        item["visual_servo.camera1"] = torch.tensor(
            [row.get("camera1_dx_norm", 0.0), row.get("camera1_dy_norm", 0.0), row.get("camera1_edge_angle_error", 0.0)],
            dtype=torch.float32,
        )
        item["visual_servo.camera1_visible"] = torch.tensor(bool(row.get("camera1_visible", False)))
        item["visual_servo.camera2"] = torch.tensor(
            [row.get("camera2_dx_norm", 0.0), row.get("camera2_dy_norm", 0.0), row.get("camera2_edge_angle_error", 0.0)],
            dtype=torch.float32,
        )
        item["visual_servo.camera2_visible"] = torch.tensor(bool(row.get("camera2_visible", False)))
        item["visual_servo.stop_label"] = torch.tensor(float(bool(row.get("stop_label", False))), dtype=torch.float32)
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def _maybe_wrap_visual_servo_labels(dataset: Any, root: Any) -> Any:
    if root is None:
        return dataset
    try:
        return _VisualServoLabelDataset(dataset, Path(str(root)))
    except FileNotFoundError:
        return dataset


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
        dataset = _maybe_wrap_visual_servo_labels(dataset, root)
        if entry.get("grid_bin_sidecar"):
            dataset = GridBinBalancedDataset(dataset, Path(str(entry["grid_bin_sidecar"])))
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
    if getattr(dataset, "requires_grid_bin_balanced_sampler", False):
        sampler = dataset.make_grid_bin_balanced_sampler(
            num_samples=len(dataset),
            drop_n_last_frames=int(getattr(cfg.policy, "drop_n_last_frames", 0) or 0),
        )
        shuffle = False
        print("Train sampler: camera_grid_bin_balanced", flush=True)
    elif getattr(dataset, "requires_dataset_balanced_sampler", False):
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
    dataset = _maybe_wrap_visual_servo_labels(dataset, wrapper_args.validation_dataset_root)
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


def _make_validation_dataset_dataloaders(
    cfg: TrainPipelineConfig,
    wrapper_args: argparse.Namespace,
) -> dict[str, torch.utils.data.DataLoader]:
    entries = _validation_dataset_entries(wrapper_args)
    if not entries:
        return {}
    loaders: dict[str, torch.utils.data.DataLoader] = {}
    for index, entry in enumerate(entries):
        name = str(entry.get("name") or entry.get("repo_id") or f"validation_dataset_{index}")
        repo_id = str(entry.get("repo_id") or wrapper_args.validation_dataset_repo_id or cfg.dataset.repo_id)
        root = Path(str(entry["root"]))
        metadata = LeRobotDatasetMetadata(repo_id, root=root)
        delta_timestamps = resolve_delta_timestamps(cfg.policy, metadata)
        dataset: Any = LeRobotDataset(
            repo_id,
            root=root,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.dataset.video_backend,
        )
        cache_dir = entry.get("image_cache_dir")
        if cache_dir:
            dataset = PredecodedImageCacheDataset(dataset, Path(str(cache_dir)))
        dataset = _maybe_wrap_visual_servo_labels(dataset, root)
        loaders[name] = torch.utils.data.DataLoader(
            dataset,
            num_workers=wrapper_args.validation_num_workers,
            batch_size=wrapper_args.validation_batch_size or cfg.batch_size,
            shuffle=False,
            pin_memory=str(cfg.policy.device) == "cuda",
            drop_last=False,
            prefetch_factor=2 if wrapper_args.validation_num_workers > 0 else None,
            persistent_workers=wrapper_args.validation_num_workers > 0,
        )
    return loaders


def _validation_dataset_entries(wrapper_args: argparse.Namespace) -> list[dict[str, Any]]:
    if not wrapper_args.validation_datasets_json:
        return []
    try:
        entries = json.loads(wrapper_args.validation_datasets_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--validation-datasets-json must be valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise SystemExit("--validation-datasets-json must be a JSON list")
    result = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"--validation-datasets-json[{index}] must be an object")
        if "root" not in entry:
            raise SystemExit(f"--validation-datasets-json[{index}] must include root")
        result.append(dict(entry))
    return result


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


def _require_aligned_checkpoint_validation_loop_cadence(
    *,
    cfg: TrainPipelineConfig,
    wrapper_args: argparse.Namespace,
    validation_step_interval: int,
    validation_epoch_interval: int,
    dataloader: Any,
) -> None:
    if not bool(cfg.save_checkpoint):
        return
    save_freq = int(cfg.save_freq)
    if save_freq <= 0:
        raise SystemExit("checkpoint save cadence must be positive; set --save_freq > 0.")
    if int(cfg.steps) > 0 and int(cfg.steps) % save_freq != 0:
        raise SystemExit(
            f"training steps ({int(cfg.steps)}) must be divisible by checkpoint save cadence "
            f"({save_freq}) so final checkpoint, validation, and loop test stay aligned."
        )
    has_loop_tests = bool(_post_checkpoint_loop_commands(wrapper_args))
    if validation_step_interval > 0:
        if validation_step_interval != save_freq:
            raise SystemExit(
                f"validation cadence ({validation_step_interval} steps) must match checkpoint save cadence "
                f"({save_freq} steps) so validation, checkpoint, and loop test run on the same step."
            )
        return
    if validation_epoch_interval > 0:
        try:
            steps_per_epoch = int(len(dataloader))
        except TypeError as exc:
            raise SystemExit(
                "epoch-based validation requires a dataloader with a known length so it can align with checkpoint cadence."
            ) from exc
        expected_save_freq = steps_per_epoch * validation_epoch_interval
        if expected_save_freq != save_freq:
            raise SystemExit(
                f"epoch validation cadence ({validation_epoch_interval} epochs x {steps_per_epoch} steps) "
                f"must match checkpoint save cadence ({save_freq} steps)."
            )
        return
    if has_loop_tests:
        raise SystemExit("loop tests require validation cadence; set --validation-interval-steps or --validation-interval-epochs.")


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


def _scalar_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu())
    return None


def _make_valid_mask_head(*, policy: Any, hidden_dim: int) -> SO101ValidMaskHead:
    config = getattr(policy, "config", None)
    state_dim = int(getattr(getattr(config, "robot_state_feature", None), "shape", [6])[0])
    action_dim = int(getattr(getattr(config, "action_feature", None), "shape", [6])[0])
    chunk_size = int(getattr(config, "chunk_size", 50))
    device = getattr(config, "device", "cpu")
    head = SO101ValidMaskHead(
        SO101ValidMaskConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=int(hidden_dim),
        )
    )
    return head.to(device)


def _make_visual_servo_head(*, policy: Any, hidden_dim: int, device: Any = "cpu") -> SO101VisualServoHead:
    context_dim = int(policy.model.vlm_with_expert.config.text_config.hidden_size)
    return SO101VisualServoHead(
        SO101VisualServoHeadConfig(hidden_dim=int(hidden_dim), context_dim=context_dim)
    ).to(device)


def _load_valid_mask_head_if_available(head: SO101ValidMaskHead, checkpoint_path: Path) -> None:
    head_path = Path(checkpoint_path) / "valid_mask_head.pt"
    if not head_path.exists():
        return
    loaded = load_valid_mask_head(head_path, device=str(_module_device(head)))
    head.load_state_dict(loaded.state_dict())
    logging.info("Loaded valid-mask head from %s", head_path)


def _load_valid_mask_head_from_policy_path_if_available(head: SO101ValidMaskHead, policy_path: Any) -> None:
    if not policy_path:
        return
    path = Path(str(policy_path))
    candidates = [path / "valid_mask_head.pt"]
    if path.name == "pretrained_model":
        candidates.append(path.parent / "valid_mask_head.pt")
    for head_path in candidates:
        if not head_path.exists():
            continue
        loaded = load_valid_mask_head(head_path, device=str(_module_device(head)))
        head.load_state_dict(loaded.state_dict())
        logging.info("Loaded valid-mask head from policy checkpoint sidecar %s", head_path)
        print(f"Loaded valid-mask head from {head_path}", flush=True)
        return


def _policy_path_from_config_or_args(cfg: TrainPipelineConfig, wrapper_args: argparse.Namespace) -> Any:
    policy_path = getattr(cfg.policy, "path", None)
    if policy_path:
        return policy_path
    for arg in getattr(wrapper_args, "lerobot_args", []):
        if arg.startswith("--policy.path="):
            return arg.split("=", 1)[1]
    args = getattr(wrapper_args, "lerobot_args", [])
    for index, arg in enumerate(args[:-1]):
        if arg == "--policy.path":
            return args[index + 1]
    return None


def _load_visual_servo_head_if_available(head: SO101VisualServoHead, checkpoint_path: Path) -> None:
    head_path = Path(checkpoint_path) / "visual_servo_head.pt"
    if not head_path.exists():
        return
    loaded = load_visual_servo_head(head_path, device=str(_module_device(head)))
    head.load_state_dict(loaded.state_dict())


def _add_valid_mask_head_to_optimizer(optimizer: Any, head: SO101ValidMaskHead | None) -> None:
    if head is None:
        return
    param_groups = getattr(optimizer, "param_groups", [])
    if not param_groups:
        return
    existing_params = {id(param) for group in param_groups for param in group.get("params", [])}
    params = [param for param in head.parameters() if param.requires_grad and id(param) not in existing_params]
    if not params:
        return
    param_groups[0]["params"].extend(params)


def _add_visual_servo_head_to_optimizer(optimizer: Any, head: SO101VisualServoHead | None) -> None:
    _add_valid_mask_head_to_optimizer(optimizer, head)  # same optimizer wiring


def _module_device(module: Any) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class _SO101LightningModule:
    def __new__(
        cls,
        *,
        LightningModule: type[Any],
        cfg: TrainPipelineConfig,
        policy: Any,
        valid_mask_head: SO101ValidMaskHead | None,
        valid_mask_loss_weight: float,
        visual_servo_head: SO101VisualServoHead | None,
        visual_servo_loss_weight: float,
        preprocessor: Any,
        optimizer: Any,
        scheduler: Any,
        augmentation: SamplingAugmentationConfig,
        action_prefix_loss_steps: int,
        action_prefix_loss_weight: float,
        action_chunk_consistency_steps: int,
        action_chunk_consistency_weight: float,
        action_delta_loss_weight: float,
        action_gripper_transition_loss_weight: float,
        action_terminal_loss_steps: int,
        action_terminal_loss_weight: float,
        action_smoothness_loss_weight: float,
        action_smoothness_include_gripper: bool,
        validation_dataloader: torch.utils.data.DataLoader | None,
        validation_dataset_loaders: dict[str, torch.utils.data.DataLoader],
        validation_step_interval: int,
        validation_epoch_interval: int,
        input_image_cameras: tuple[str, ...],
        log_input_images_every_n_steps: int,
        log_input_metadata_every_n_steps: int,
        initial_step: int = 0,
        checkpoint_save_freq: int = 0,
        total_steps: int = 0,
    ) -> Any:
        class SO101LightningModuleImpl(LightningModule):
            def __init__(self) -> None:
                super().__init__()
                self.cfg = cfg
                self.policy = policy
                self.valid_mask_head = valid_mask_head
                self.valid_mask_loss_weight = max(0.0, float(valid_mask_loss_weight))
                self.visual_servo_head = visual_servo_head
                self.visual_servo_loss_weight = max(0.0, float(visual_servo_loss_weight))
                self.preprocessor = preprocessor
                self._optimizer = optimizer
                self._scheduler = scheduler
                self.augmentation = augmentation
                self.action_prefix_loss_steps = max(0, int(action_prefix_loss_steps))
                self.action_prefix_loss_weight = max(0.0, float(action_prefix_loss_weight))
                self.action_chunk_consistency_steps = max(0, int(action_chunk_consistency_steps))
                self.action_chunk_consistency_weight = max(0.0, float(action_chunk_consistency_weight))
                self.action_delta_loss_weight = max(0.0, float(action_delta_loss_weight))
                self.action_gripper_transition_loss_weight = max(0.0, float(action_gripper_transition_loss_weight))
                self.action_terminal_loss_steps = max(0, int(action_terminal_loss_steps))
                self.action_terminal_loss_weight = max(0.0, float(action_terminal_loss_weight))
                self.action_smoothness_loss_weight = max(0.0, float(action_smoothness_loss_weight))
                self.action_smoothness_include_gripper = bool(action_smoothness_include_gripper)
                self.validation_dataloader = validation_dataloader
                self.validation_dataset_loaders = validation_dataset_loaders
                self.validation_step_interval = max(0, int(validation_step_interval))
                self.validation_epoch_interval = max(0, int(validation_epoch_interval))
                self.validation_iter: Any | None = None
                self.validation_dataset_iters: dict[str, Any] = {}
                self.initial_step = max(0, int(initial_step))
                self.train_batches_seen = self.initial_step
                self.input_image_cameras = input_image_cameras
                self.log_input_images_every_n_steps = max(0, int(log_input_images_every_n_steps))
                self.log_input_metadata_every_n_steps = max(0, int(log_input_metadata_every_n_steps))
                self.checkpoint_save_freq = max(0, int(checkpoint_save_freq))
                self.total_steps = max(0, int(total_steps))
                self._last_step_started = 0.0
                self.last_train_loss: float | None = None
                self.last_validation_loss: float | None = None

            def forward(self, batch: dict[str, Any]) -> Any:
                return self.policy.forward(batch)

            def training_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
                started = time.perf_counter()
                self.policy.train()
                raw_batch = batch
                batch = self.preprocessor(batch)
                _copy_visual_servo_labels(raw_batch, batch)
                dataloading_s = time.perf_counter() - self._last_step_started if self._last_step_started else 0.0
                pre_augmented_images = None
                augmentation = _single_view_augmentation_for_visual_servo_labels(
                    self.augmentation,
                    enabled=self.visual_servo_head is not None and self.visual_servo_loss_weight > 0.0,
                )
                if self.augmentation.enabled and self.augmentation.gpu_image_augmentation:
                    pre_augmented_images = _clone_image_tensors_for_logging(batch, self.input_image_cameras)
                    augment_batch_on_device(batch, augmentation)
                if pre_augmented_images is not None:
                    self._log_input_images(pre_augmented_images, split="train", tag_prefix="input")
                    self._log_input_images(batch, split="train", tag_prefix="augmented_input")
                    self._log_augmentation_image_delta(pre_augmented_images, batch)
                else:
                    self._log_input_images(batch, split="train", tag_prefix="input")
                self._log_input_metadata(raw_batch, batch, split="train")
                visual_servo_aux_loss, visual_servo_metrics = visual_servo_loss(
                    self.visual_servo_head,
                    batch,
                    weight=self.visual_servo_loss_weight,
                    policy=self.policy,
                )
                loss, output_dict = _forward_policy_with_optional_prefix_loss(
                    self.policy,
                    batch,
                    prefix_steps=self.action_prefix_loss_steps,
                    prefix_weight=self.action_prefix_loss_weight,
                    consistency_steps=self.action_chunk_consistency_steps,
                    consistency_weight=self.action_chunk_consistency_weight,
                    delta_weight=self.action_delta_loss_weight,
                    gripper_transition_weight=self.action_gripper_transition_loss_weight,
                    terminal_steps=self.action_terminal_loss_steps,
                    terminal_weight=self.action_terminal_loss_weight,
                    smoothness_weight=self.action_smoothness_loss_weight,
                    smoothness_include_gripper=self.action_smoothness_include_gripper,
                    valid_mask_head=self.valid_mask_head,
                    valid_mask_loss_weight=self.valid_mask_loss_weight,
                )
                if visual_servo_aux_loss is not None:
                    loss = loss + visual_servo_aux_loss
                    output_dict = dict(output_dict)
                    output_dict.update(visual_servo_metrics)
                    output_dict["loss"] = _detach_scalar(loss)
                batch_size = _batch_size(batch)
                update_s = time.perf_counter() - started
                train_log_step = self.train_batches_seen + 1
                self.last_train_loss = _scalar_float(loss)
                self._log_train_scalar("train/loss", loss, batch_size=batch_size, log_step=train_log_step)
                self._log_train_scalar(
                    "important/train_loss",
                    loss,
                    batch_size=batch_size,
                    log_step=train_log_step,
                )
                self.log(
                    "train/loss_progbar",
                    loss,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=False,
                    batch_size=batch_size,
                )
                self._log_train_dataset_losses(raw_batch, batch, batch_size=batch_size)
                self._log_train_scalar("train/batch_size", float(batch_size), batch_size=batch_size, log_step=train_log_step)
                self._log_checkpoint_schedule_metrics(batch_size=batch_size, log_step=train_log_step)
                self._log_train_scalar("extra/train/update_s", update_s, batch_size=batch_size, log_step=train_log_step)
                self._log_train_scalar("extra/train/data_s", dataloading_s, batch_size=batch_size, log_step=train_log_step)
                self._log_train_scalar(
                    "extra/train/update_s_per_sample",
                    _seconds_per_sample(update_s, batch_size),
                    batch_size=batch_size,
                    log_step=train_log_step,
                )
                self._log_train_scalar(
                    "extra/train/data_s_per_sample",
                    _seconds_per_sample(dataloading_s, batch_size),
                    batch_size=batch_size,
                    log_step=train_log_step,
                )
                self._log_train_scalar(
                    "train/samples_per_s",
                    _samples_per_second(batch_size, update_s),
                    batch_size=batch_size,
                    log_step=train_log_step,
                )
                self._log_system_metrics(batch_size=batch_size)
                for key, value in (output_dict or {}).items():
                    if key == "loss":
                        continue
                    if torch.is_tensor(value) and value.numel() == 1:
                        value = value.detach()
                    if isinstance(value, (int, float)):
                        self._log_train_scalar(
                            _scalar_metric_tag("train", key),
                            float(value),
                            batch_size=batch_size,
                            log_step=train_log_step,
                        )
                    elif torch.is_tensor(value) and value.numel() == 1:
                        self._log_train_scalar(
                            _scalar_metric_tag("train", key),
                            value,
                            batch_size=batch_size,
                            log_step=train_log_step,
                        )
                self.train_batches_seen += 1
                self._run_step_validation_if_due(completed_step=self.train_batches_seen)
                self._last_step_started = time.perf_counter()
                return loss

            def _log_system_metrics(self, *, batch_size: int) -> None:
                step = self.train_batches_seen + 1
                if step != 1 and step % 50 != 0:
                    return
                for tag, value in _system_metrics_for_current_process().items():
                    self._log_train_scalar(tag, value, batch_size=batch_size, log_step=step)

            def _log_checkpoint_schedule_metrics(self, *, batch_size: int, log_step: int) -> None:
                if self.checkpoint_save_freq <= 0:
                    return
                remainder = int(log_step) % self.checkpoint_save_freq
                remaining = 0 if remainder == 0 else self.checkpoint_save_freq - remainder
                if self.total_steps > 0:
                    remaining = min(remaining, max(0, self.total_steps - int(log_step)))
                self._log_train_scalar(
                    "train/checkpoint_steps_remaining",
                    float(remaining),
                    batch_size=batch_size,
                    log_step=log_step,
                )
                self._log_train_scalar(
                    "important/checkpoint_steps_remaining",
                    float(remaining),
                    batch_size=batch_size,
                    log_step=log_step,
                )

            def on_train_epoch_end(self) -> None:
                if not self._has_validation_loaders() or self.validation_epoch_interval <= 0:
                    return
                epoch = int(getattr(self.trainer, "current_epoch", 0)) + 1
                if epoch > 0 and epoch % self.validation_epoch_interval == 0:
                    self._run_scheduled_validation(log_step=self._absolute_step())

            def _run_step_validation_if_due(self, *, completed_step: int) -> None:
                if not self._has_validation_loaders() or self.validation_step_interval <= 0:
                    return
                if completed_step > 0 and completed_step % self.validation_step_interval == 0:
                    self._run_scheduled_validation(log_step=completed_step)

            def _run_scheduled_validation(self, *, log_step: int) -> None:
                dataset_losses = []
                if self.validation_dataloader is not None:
                    if self.validation_iter is None:
                        self.validation_iter = cycle(self.validation_dataloader)
                    print(f"Running validation batch at step {log_step}", flush=True)
                    self.run_validation_batch(next(self.validation_iter), log_step=log_step)
                for name, dataloader in self.validation_dataset_loaders.items():
                    if name not in self.validation_dataset_iters:
                        self.validation_dataset_iters[name] = cycle(dataloader)
                    print(f"Running validation batch for {name} at step {log_step}", flush=True)
                    dataset_losses.append(
                        self.run_validation_batch(
                        next(self.validation_dataset_iters[name]),
                        log_step=log_step,
                        dataset_name=name,
                        )
                    )
                if self.validation_dataloader is None and dataset_losses:
                    mean_loss = torch.stack([
                        loss.detach().float() if torch.is_tensor(loss) else torch.as_tensor(float(loss))
                        for loss in dataset_losses
                    ]).mean()
                    self.last_validation_loss = _scalar_float(mean_loss)
                    self._log_validation_scalar(
                        VAL_LOSS_TAG,
                        mean_loss,
                        batch_size=1,
                        log_step=log_step,
                    )
                    self._log_validation_scalar(
                        IMPORTANT_VAL_LOSS_TAG,
                        mean_loss,
                        batch_size=1,
                        log_step=log_step,
                    )

            def _has_validation_loaders(self) -> bool:
                return self.validation_dataloader is not None or bool(self.validation_dataset_loaders)

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

            def run_validation_batch(
                self,
                batch: dict[str, Any],
                *,
                log_step: int | None = None,
                dataset_name: str | None = None,
            ) -> Any:
                was_training = self.policy.training
                self.policy.eval()
                with torch.no_grad():
                    raw_batch = batch
                    batch = self.preprocessor(batch)
                    _copy_visual_servo_labels(raw_batch, batch)
                    self._log_input_images(batch, split="val", log_step=log_step, tag_prefix="input")
                    self._log_input_metadata(raw_batch, batch, split="val", log_step=log_step)
                    loss, output_dict = _forward_policy_with_optional_prefix_loss(
                        self.policy,
                        batch,
                        prefix_steps=self.action_prefix_loss_steps,
                        prefix_weight=self.action_prefix_loss_weight,
                        consistency_steps=self.action_chunk_consistency_steps,
                        consistency_weight=self.action_chunk_consistency_weight,
                        delta_weight=self.action_delta_loss_weight,
                        gripper_transition_weight=self.action_gripper_transition_loss_weight,
                        terminal_steps=self.action_terminal_loss_steps,
                        terminal_weight=self.action_terminal_loss_weight,
                        smoothness_weight=0.0,
                        smoothness_include_gripper=self.action_smoothness_include_gripper,
                        valid_mask_head=self.valid_mask_head,
                        valid_mask_loss_weight=self.valid_mask_loss_weight,
                    )
                    visual_servo_aux_loss, visual_servo_metrics = visual_servo_loss(
                        self.visual_servo_head,
                        batch,
                        weight=self.visual_servo_loss_weight,
                        policy=self.policy,
                    )
                    if visual_servo_aux_loss is not None:
                        loss = loss + visual_servo_aux_loss
                        output_dict = dict(output_dict)
                        output_dict.update(visual_servo_metrics)
                        output_dict["loss"] = _detach_scalar(loss)
                    jitter_metrics = _action_chunk_jitter_metrics(self.policy, batch)
                batch_size = _batch_size(batch)
                prefix = "val" if dataset_name is None else f"val/datasets/{_safe_tag(dataset_name)}"
                loss_tag = VAL_LOSS_TAG if dataset_name is None else f"{prefix}/loss"
                self._log_validation_scalar(loss_tag, loss, batch_size=batch_size, log_step=log_step)
                if dataset_name is None:
                    self.last_validation_loss = _scalar_float(loss)
                    self._log_validation_scalar(IMPORTANT_VAL_LOSS_TAG, loss, batch_size=batch_size, log_step=log_step)
                for key, value in (output_dict or {}).items():
                    if key == "loss":
                        continue
                    self._log_validation_scalar(
                        _scalar_metric_tag(prefix, key),
                        value,
                        batch_size=batch_size,
                        log_step=log_step,
                    )
                if dataset_name is None:
                    self._log_action_jitter_metrics(jitter_metrics, batch_size=batch_size, log_step=log_step)
                if was_training:
                    self.policy.train()
                self._log_validation_scalar(
                    f"{prefix}/batch_size",
                    float(batch_size),
                    batch_size=batch_size,
                    log_step=log_step,
                )
                return loss

            def _log_train_dataset_losses(
                self,
                raw_batch: dict[str, Any],
                processed_batch: dict[str, Any],
                *,
                batch_size: int,
            ) -> None:
                names = _batch_dataset_names(raw_batch)
                if not names:
                    return
                unique_names = []
                for name in names:
                    if name not in unique_names:
                        unique_names.append(name)
                was_training = self.policy.training
                with torch.no_grad():
                    for name in unique_names:
                        indexes = [idx for idx, value in enumerate(names) if value == name]
                        if not indexes:
                            continue
                        subset = _subset_batch(processed_batch, indexes)
                        if not subset:
                            continue
                        loss, _output_dict = _forward_policy_with_optional_prefix_loss(
                            self.policy,
                            subset,
                            prefix_steps=self.action_prefix_loss_steps,
                            prefix_weight=self.action_prefix_loss_weight,
                            consistency_steps=self.action_chunk_consistency_steps,
                            consistency_weight=self.action_chunk_consistency_weight,
                            delta_weight=self.action_delta_loss_weight,
                            gripper_transition_weight=self.action_gripper_transition_loss_weight,
                            terminal_steps=self.action_terminal_loss_steps,
                            terminal_weight=self.action_terminal_loss_weight,
                            smoothness_weight=0.0,
                            smoothness_include_gripper=self.action_smoothness_include_gripper,
                            valid_mask_head=self.valid_mask_head,
                            valid_mask_loss_weight=self.valid_mask_loss_weight,
                        )
                        self._log_validation_scalar(
                            f"train/datasets/{_safe_tag(name)}/loss",
                            loss,
                            batch_size=len(indexes),
                            log_step=self.train_batches_seen + 1,
                        )
                if was_training:
                    self.policy.train()

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

            def _log_train_scalar(
                self,
                tag: str,
                value: Any,
                *,
                batch_size: int,
                log_step: int,
            ) -> None:
                self._log_validation_scalar(
                    tag,
                    value,
                    batch_size=batch_size,
                    log_step=log_step,
                )

            def _log_input_images(
                self,
                batch: dict[str, Any],
                *,
                split: str,
                log_step: int | None = None,
                tag_prefix: str = "input",
            ) -> None:
                if self.log_input_images_every_n_steps <= 0:
                    return
                step = int(log_step if log_step is not None else getattr(self.trainer, "global_step", 0))
                if log_step is None:
                    step += self.initial_step
                if split != "val" and step % self.log_input_images_every_n_steps != 0:
                    return
                experiment = getattr(getattr(self, "logger", None), "experiment", None)
                if experiment is None or not hasattr(experiment, "add_image"):
                    return
                for camera in self.input_image_cameras:
                    key = f"observation.images.{camera}"
                    if key not in batch:
                        continue
                    image = (
                        _tensorboard_image_grid_with_visual_servo_target(batch, camera)
                        if split == "val"
                        else _tensorboard_image_with_visual_servo_target(batch, camera)
                    )
                    if image is None:
                        continue
                    experiment.add_image(f"{split}/{tag_prefix}_{camera}", image, global_step=step)

            def _log_augmentation_image_delta(
                self,
                before: dict[str, Any],
                after: dict[str, Any],
            ) -> None:
                if self.log_input_images_every_n_steps <= 0:
                    return
                step = int(getattr(self.trainer, "global_step", 0)) + self.initial_step
                if step % self.log_input_images_every_n_steps != 0:
                    return
                for camera in self.input_image_cameras:
                    key = f"observation.images.{camera}"
                    raw = before.get(key)
                    augmented = after.get(key)
                    if not torch.is_tensor(raw) or not torch.is_tensor(augmented):
                        continue
                    if tuple(raw.shape) != tuple(augmented.shape):
                        continue
                    diff = (raw.detach().float() - augmented.detach().float()).abs()
                    self._log_train_scalar(
                        f"extra/train/augmentation_mae_{camera}",
                        diff.mean(),
                        batch_size=_batch_size(after),
                        log_step=step,
                    )
                    self._log_train_scalar(
                        f"extra/train/augmentation_changed_pct_{camera}",
                        (diff > 0.01).float().mean() * 100.0,
                        batch_size=_batch_size(after),
                        log_step=step,
                    )

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
                    experiment.add_text(f"extra/{split}/input_motor_state", state_text, global_step=step)
                vector = _first_vector(state)
                if vector is not None and hasattr(experiment, "add_scalar"):
                    for index, value in enumerate(vector):
                        experiment.add_scalar(
                            f"extra/{split}/input_motor_state/dim_{index:02d}",
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
            post_checkpoint_loop_commands: list[list[str]] | None,
            retention_policy: str,
        ) -> None:
            super().__init__()
            self.cfg = cfg
            self.policy_module = policy_module
            self.preprocessor = preprocessor
            self.postprocessor = postprocessor
            self.save_freq = max(1, int(save_freq))
            self.enabled = enabled
            self.initial_step = max(0, int(initial_step))
            self.post_checkpoint_loop_commands = [list(command) for command in (post_checkpoint_loop_commands or [])]
            self.retention_policy = retention_policy
            self.saved_steps: set[int] = set()
            self.retention_state_path = Path(self.cfg.output_dir) / "checkpoints" / "retention_state.json"
            self.retention_events_path = Path(self.cfg.output_dir) / "checkpoints" / "retention_events.jsonl"
            self.retention_state = _read_json_file(self.retention_state_path) or {}

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
                if getattr(pl_module, "valid_mask_head", None) is not None:
                    save_valid_mask_head(
                        checkpoint_dir / "valid_mask_head.pt",
                        pl_module.valid_mask_head,
                        metadata={
                            "step": step,
                            "label_source": "action_is_pad_as_termination_proxy",
                            "loss_tag": "valid_mask_loss",
                        },
                    )
                if getattr(pl_module, "visual_servo_head", None) is not None:
                    save_visual_servo_head(
                        checkpoint_dir / "visual_servo_head.pt",
                        pl_module.visual_servo_head,
                        metadata={
                            "step": step,
                            "label_source": "visual_servo_labels_sidecar",
                            "loss_tag": "visual_servo_loss",
                        },
                    )
                update_last_checkpoint(checkpoint_dir)
            self.saved_steps.add(step)
            closed_loop_success = self._run_post_checkpoint_loop_test(checkpoint_dir=checkpoint_dir, step=step)
            self._apply_retention_policy(
                checkpoint_dir=checkpoint_dir,
                step=step,
                train_loss=_scalar_float(getattr(pl_module, "last_train_loss", None)),
                validation_loss=_scalar_float(getattr(pl_module, "last_validation_loss", None)),
                closed_loop_success=closed_loop_success,
            )

        def _run_post_checkpoint_loop_test(self, *, checkpoint_dir: Path, step: int) -> float | None:
            if not self.post_checkpoint_loop_commands:
                return None
            env = {
                **os.environ,
                "SO101_CHECKPOINT_DIR": str(checkpoint_dir),
                "SO101_CHECKPOINT_STEP": str(step),
            }
            success_values: list[float] = []
            for index, command in enumerate(self.post_checkpoint_loop_commands, start=1):
                print(
                    f"Running closed-loop test {index}/{len(self.post_checkpoint_loop_commands)} "
                    f"after checkpoint step {step}",
                    flush=True,
                )
                subprocess.run(command, check=True, env=env)
                success = _latest_closed_loop_success_for_command(command, checkpoint_dir.name)
                if success is not None:
                    success_values.append(success)
            return max(success_values) if success_values else None

        def _apply_retention_policy(
            self,
            *,
            checkpoint_dir: Path,
            step: int,
            train_loss: float | None,
            validation_loss: float | None,
            closed_loop_success: float | None,
        ) -> None:
            if self.retention_policy != "best_val_and_closed_loop":
                return
            retained = False
            if train_loss is not None:
                best_train = _float_or_none(self.retention_state.get("best_train_loss"))
                if best_train is None or train_loss < best_train:
                    self._promote_checkpoint(
                        checkpoint_dir,
                        "best_train_loss",
                        {
                            "kind": "best_train_loss",
                            "step": step,
                            "checkpoint": checkpoint_dir.name,
                            "train_loss": train_loss,
                            "previous_best_train_loss": best_train,
                        },
                    )
                    self.retention_state["best_train_loss"] = train_loss
                    self.retention_state["best_train_loss_step"] = step
                    self.retention_state["best_train_loss_source_checkpoint"] = checkpoint_dir.name
                    retained = True
            if validation_loss is not None:
                best_val = _float_or_none(self.retention_state.get("best_val_loss"))
                if best_val is None or validation_loss < best_val:
                    self._promote_checkpoint(
                        checkpoint_dir,
                        "best_val_loss",
                        {
                            "kind": "best_val_loss",
                            "step": step,
                            "checkpoint": checkpoint_dir.name,
                            "val_loss": validation_loss,
                            "previous_best_val_loss": best_val,
                        },
                    )
                    self.retention_state["best_val_loss"] = validation_loss
                    self.retention_state["best_val_loss_step"] = step
                    self.retention_state["best_val_loss_source_checkpoint"] = checkpoint_dir.name
                    retained = True
            if closed_loop_success is not None:
                best_loop = _float_or_none(self.retention_state.get("best_closed_loop_success_rate"))
                if best_loop is None or closed_loop_success > best_loop:
                    self._promote_checkpoint(
                        checkpoint_dir,
                        "best_closed_loop",
                        {
                            "kind": "best_closed_loop",
                            "step": step,
                            "checkpoint": checkpoint_dir.name,
                            "success_rate": closed_loop_success,
                            "previous_best_success_rate": best_loop,
                        },
                    )
                    self.retention_state["best_closed_loop_success_rate"] = closed_loop_success
                    self.retention_state["best_closed_loop_step"] = step
                    self.retention_state["best_closed_loop_source_checkpoint"] = checkpoint_dir.name
                    retained = True
            self.retention_state["latest_checkpoint_step"] = step
            self.retention_state["latest_source_checkpoint"] = checkpoint_dir.name
            self.retention_state_path.parent.mkdir(parents=True, exist_ok=True)
            self.retention_state_path.write_text(
                json.dumps(self.retention_state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if checkpoint_dir.exists() and checkpoint_dir.name.isdigit():
                shutil.rmtree(checkpoint_dir)
                self._append_retention_event(
                    {
                        "kind": "pruned_periodic_checkpoint",
                        "step": step,
                        "checkpoint": checkpoint_dir.name,
                        "retained_as_best": retained,
                    }
                )
            self._refresh_last_retained_checkpoint()

        def _promote_checkpoint(self, checkpoint_dir: Path, name: str, event: dict[str, Any]) -> None:
            target = Path(self.cfg.output_dir) / "checkpoints" / name
            tmp = target.with_name(f".{target.name}.tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(checkpoint_dir, tmp)
            tmp.rename(target)
            self._append_retention_event(event)

        def _append_retention_event(self, event: dict[str, Any]) -> None:
            self.retention_events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.retention_events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

        def _refresh_last_retained_checkpoint(self) -> None:
            checkpoint_root = Path(self.cfg.output_dir) / "checkpoints"
            for name in ("best_closed_loop", "best_val_loss", "best_train_loss"):
                target = checkpoint_root / name
                if target.exists():
                    update_last_checkpoint(target)
                    return

    return LeRobotCheckpointCallback(**kwargs)


def _forward_policy_with_optional_prefix_loss(
    policy: Any,
    batch: dict[str, Any],
    *,
    prefix_steps: int,
    prefix_weight: float,
    consistency_steps: int = 0,
    consistency_weight: float = 0.0,
    delta_weight: float = 0.0,
    gripper_transition_weight: float = 0.0,
    terminal_steps: int = 0,
    terminal_weight: float = 1.0,
    smoothness_weight: float = 0.0,
    smoothness_include_gripper: bool = False,
    valid_mask_head: SO101ValidMaskHead | None = None,
    valid_mask_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    use_prefix = prefix_steps > 0 and prefix_weight != 1.0
    use_consistency = consistency_steps > 0 and consistency_weight > 0.0
    use_teacher_importance = (
        float(delta_weight) > 0.0
        or float(gripper_transition_weight) > 0.0
        or (int(terminal_steps) > 0 and float(terminal_weight) != 1.0)
    )
    use_smoothness = float(smoothness_weight) > 0.0
    if (
        not use_prefix
        and not use_consistency
        and not use_teacher_importance
        and not use_smoothness
    ) or getattr(policy, "name", None) != "smolvla":
        return _add_valid_mask_auxiliary_loss(
            policy.forward(batch),
            valid_mask_head=valid_mask_head,
            batch=batch,
            weight=valid_mask_loss_weight,
        )
    if not all(hasattr(policy, name) for name in ("prepare_images", "prepare_state", "prepare_action")):
        return _add_valid_mask_auxiliary_loss(
            policy.forward(batch),
            valid_mask_head=valid_mask_head,
            batch=batch,
            weight=valid_mask_loss_weight,
        )
    action_loss, loss_dict = _forward_smolvla_with_prefix_loss(
        policy,
        batch,
        prefix_steps=prefix_steps,
        prefix_weight=prefix_weight,
        consistency_steps=consistency_steps,
        consistency_weight=consistency_weight,
        delta_weight=delta_weight,
        gripper_transition_weight=gripper_transition_weight,
        terminal_steps=terminal_steps,
        terminal_weight=terminal_weight,
        smoothness_weight=smoothness_weight,
        smoothness_include_gripper=smoothness_include_gripper,
    )
    return _add_valid_mask_auxiliary_loss(
        (action_loss, loss_dict),
        valid_mask_head=valid_mask_head,
        batch=batch,
        weight=valid_mask_loss_weight,
    )


def _add_valid_mask_auxiliary_loss(
    policy_output: Any,
    *,
    valid_mask_head: SO101ValidMaskHead | None,
    batch: dict[str, Any],
    weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    action_loss, loss_dict = _normalize_policy_loss_output(policy_output)
    if valid_mask_head is None or float(weight) <= 0.0 or batch.get("action_is_pad") is None:
        return action_loss, loss_dict
    state = batch[OBS_STATE]
    action = batch[ACTION]
    labels = valid_labels_from_action_is_pad(batch.get("action_is_pad")).to(device=state.device)
    logits = valid_mask_head(state, action)
    labels = labels[:, : logits.shape[1]]
    valid_mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    valid_probs = torch.sigmoid(logits.detach())
    valid_pred = (valid_probs >= 0.5).to(dtype=labels.dtype)
    valid_mask_accuracy = (valid_pred == labels).float().mean()
    total_loss = action_loss + float(weight) * valid_mask_loss
    loss_dict = dict(loss_dict)
    loss_dict["action_loss"] = _detach_scalar(action_loss)
    loss_dict["valid_mask_loss"] = _detach_scalar(valid_mask_loss)
    loss_dict["valid_mask_loss_weight"] = float(weight)
    loss_dict["valid_mask_accuracy"] = _detach_scalar(valid_mask_accuracy)
    loss_dict["loss"] = _detach_scalar(total_loss)
    return total_loss, loss_dict


def _normalize_policy_loss_output(policy_output: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(policy_output, tuple) and len(policy_output) == 2:
        loss, loss_dict = policy_output
        return loss, dict(loss_dict or {})
    if isinstance(policy_output, dict) and "loss" in policy_output:
        loss = policy_output["loss"]
        return loss, dict(policy_output)
    if torch.is_tensor(policy_output):
        return policy_output, {"loss": _detach_scalar(policy_output)}
    raise TypeError(f"Unsupported policy loss output type: {type(policy_output)!r}")


def _detach_scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def _single_view_augmentation_for_visual_servo_labels(
    config: SamplingAugmentationConfig,
    *,
    enabled: bool,
) -> SamplingAugmentationConfig:
    if not enabled:
        return config
    # Visual-servo dx/dy labels are transformed together with affine jitter.
    # Still disable transforms that may hide the labeled target.
    return dataclasses.replace(
        config,
        image_camera_dropout_prob=0.0,
        image_patch_dropout_prob=0.0,
        image_patch_mask_ratio=0.0,
    )


def _scalar_metric_tag(prefix: str, key: str) -> str:
    base = f"{prefix}/{key}"
    if key.startswith("losses_after_"):
        return f"extra/{base}"
    if key.startswith("visual_servo_") and key not in {"visual_servo_loss", "visual_servo_mse", "visual_servo_rmse"}:
        return f"extra/{base}"
    return base


def _smolvla_training_losses_and_action_hat(
    model: Any,
    images: torch.Tensor,
    img_masks: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SmolVLA's training forward while exposing a differentiable action estimate."""

    noise = model.sample_noise(actions.shape, actions.device)
    time_tensor = model.sample_time(actions.shape[0], actions.device)
    time_expanded = time_tensor[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state=state,
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time_tensor)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        fill_kv_cache=False,
    )
    suffix_out = suffix_out[:, -model.config.chunk_size :].to(dtype=torch.float32)
    v_t = model.action_out_proj(suffix_out)
    losses = torch.nn.functional.mse_loss(u_t, v_t, reduction="none")
    action_hat = noise - v_t
    return losses, action_hat


def _smoothness_action_dim(actions: torch.Tensor, include_gripper: bool) -> int:
    action_dim = int(actions.shape[-1])
    if include_gripper or action_dim <= 1:
        return action_dim
    return action_dim - 1


def _predicted_action_jerk_smoothness_loss(
    actions: torch.Tensor,
    *,
    actions_is_pad: torch.Tensor | None,
    include_gripper: bool,
) -> torch.Tensor | None:
    if actions.ndim != 3 or int(actions.shape[1]) < 3:
        return None
    smooth_dim = _smoothness_action_dim(actions, include_gripper)
    if smooth_dim <= 0:
        return None
    selected = actions[:, :, :smooth_dim]
    jerk = selected[:, 2:, :] - 2.0 * selected[:, 1:-1, :] + selected[:, :-2, :]
    if actions_is_pad is None:
        return jerk.square().mean()
    valid = ~actions_is_pad
    if int(valid.shape[1]) < 3:
        return None
    valid_triplet = (valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]).to(dtype=jerk.dtype, device=jerk.device)
    valid_triplet = valid_triplet.unsqueeze(-1)
    denom = valid_triplet.expand_as(jerk).sum()
    if denom <= 0:
        return None
    return (jerk.square() * valid_triplet).sum() / denom.clamp_min(torch.finfo(jerk.dtype).eps)


def _forward_smolvla_with_prefix_loss(
    policy: Any,
    batch: dict[str, Any],
    *,
    prefix_steps: int,
    prefix_weight: float,
    consistency_steps: int = 0,
    consistency_weight: float = 0.0,
    delta_weight: float = 0.0,
    gripper_transition_weight: float = 0.0,
    terminal_steps: int = 0,
    terminal_weight: float = 1.0,
    smoothness_weight: float = 0.0,
    smoothness_include_gripper: bool = False,
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
    losses, action_hat = _smolvla_training_losses_and_action_hat(
        policy.model,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
    )
    original_action_dim = policy.config.action_feature.shape[0]
    losses = losses[:, :, :original_action_dim]
    action_hat = action_hat[:, :, :original_action_dim]
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
    teacher_importance_weights, teacher_importance_metrics = _teacher_action_importance_weights(
        actions[:, :, : policy.config.max_action_dim],
        losses=losses,
        actions_is_pad=actions_is_pad,
        delta_weight=delta_weight,
        gripper_transition_weight=gripper_transition_weight,
        terminal_steps=terminal_steps,
        terminal_weight=terminal_weight,
    )
    weights = weights * teacher_importance_weights
    weighted_loss = (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(losses.dtype).eps)
    consistency_loss = _teacher_front_chunk_consistency_loss(
        losses,
        steps=consistency_steps,
        actions_is_pad=actions_is_pad,
    )
    total_loss = weighted_loss
    if consistency_loss is not None and float(consistency_weight) > 0.0:
        total_loss = total_loss + float(consistency_weight) * consistency_loss
        loss_dict["action_chunk_consistency_loss"] = consistency_loss.detach().item()
        loss_dict["action_chunk_consistency_weight"] = float(consistency_weight)
        loss_dict["action_chunk_consistency_steps"] = int(min(max(0, consistency_steps), max(0, int(losses.shape[1]) - 1)))
    smoothness_loss = _predicted_action_jerk_smoothness_loss(
        action_hat[:, :, : policy.config.max_action_dim],
        actions_is_pad=actions_is_pad,
        include_gripper=smoothness_include_gripper,
    )
    if smoothness_loss is not None and float(smoothness_weight) > 0.0:
        total_loss = total_loss + float(smoothness_weight) * smoothness_loss
        loss_dict["action_smoothness_loss"] = smoothness_loss.detach().item()
        loss_dict["action_smoothness_loss_weight"] = float(smoothness_weight)
        loss_dict["action_smoothness_include_gripper"] = bool(smoothness_include_gripper)
        loss_dict["action_smoothness_dims"] = int(_smoothness_action_dim(action_hat, smoothness_include_gripper))
    loss_dict["loss"] = total_loss.item()
    loss_dict["loss_unweighted"] = unweighted_loss.detach().item()
    loss_dict["loss_prefix_weight"] = float(prefix_weight)
    loss_dict["loss_prefix_steps"] = int(min(prefix_steps, int(losses.shape[1])))
    loss_dict.update(teacher_importance_metrics)
    return total_loss, loss_dict


def _teacher_action_importance_weights(
    actions: torch.Tensor,
    *,
    losses: torch.Tensor,
    actions_is_pad: torch.Tensor | None,
    delta_weight: float,
    gripper_transition_weight: float,
    terminal_steps: int,
    terminal_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Data-derived action loss weights.

    This intentionally avoids primitive-specific step numbers.  Important
    regions are inferred from the teacher sequence itself: action changes,
    gripper transitions, and the final valid horizon before padding.
    """

    weights = torch.ones_like(losses)
    metrics: dict[str, Any] = {
        "action_delta_loss_weight": float(delta_weight),
        "action_gripper_transition_loss_weight": float(gripper_transition_weight),
        "action_terminal_loss_steps": int(max(0, terminal_steps)),
        "action_terminal_loss_weight": float(terminal_weight),
    }
    if actions.ndim != 3 or actions.shape[:2] != losses.shape[:2]:
        metrics["action_importance_weight_mean"] = float(weights.detach().mean().cpu())
        return weights, metrics

    action_dim = min(int(actions.shape[-1]), int(losses.shape[-1]))
    teacher = actions[:, :, :action_dim].detach().to(dtype=losses.dtype, device=losses.device)
    deltas = torch.zeros_like(teacher)
    if teacher.shape[1] > 1:
        deltas[:, 1:, :] = (teacher[:, 1:, :] - teacher[:, :-1, :]).abs()

    if float(delta_weight) > 0.0:
        delta_mag = deltas.mean(dim=-1, keepdim=True)
        delta_norm = _normalize_sequence_importance(delta_mag, actions_is_pad)
        weights[:, :, :action_dim] = weights[:, :, :action_dim] * (1.0 + float(delta_weight) * delta_norm)
        metrics["action_delta_importance_mean"] = float(delta_norm.detach().mean().cpu())

    gripper_index = 5
    if float(gripper_transition_weight) > 0.0 and action_dim > gripper_index:
        gripper_delta = deltas[:, :, gripper_index : gripper_index + 1]
        gripper_norm = _normalize_sequence_importance(gripper_delta, actions_is_pad)
        weights[:, :, gripper_index : gripper_index + 1] = weights[:, :, gripper_index : gripper_index + 1] * (
            1.0 + float(gripper_transition_weight) * gripper_norm
        )
        metrics["action_gripper_transition_importance_mean"] = float(gripper_norm.detach().mean().cpu())

    if int(terminal_steps) > 0 and float(terminal_weight) != 1.0:
        terminal_mask = _terminal_valid_mask(
            actions_is_pad,
            batch_size=int(losses.shape[0]),
            horizon=int(losses.shape[1]),
            steps=int(terminal_steps),
            device=losses.device,
            dtype=losses.dtype,
        )
        weights = weights * (1.0 + (float(terminal_weight) - 1.0) * terminal_mask.unsqueeze(-1))
        metrics["action_terminal_importance_fraction"] = float(terminal_mask.detach().mean().cpu())

    if actions_is_pad is not None:
        valid = (~actions_is_pad[:, : losses.shape[1]]).to(dtype=losses.dtype, device=losses.device).unsqueeze(-1)
        weights = weights * valid
    metrics["action_importance_weight_mean"] = float(weights.detach().mean().cpu())
    metrics["action_importance_weight_max"] = float(weights.detach().max().cpu())
    return weights, metrics


def _normalize_sequence_importance(values: torch.Tensor, actions_is_pad: torch.Tensor | None) -> torch.Tensor:
    values = values.clamp_min(0.0)
    if actions_is_pad is not None:
        valid = (~actions_is_pad[:, : values.shape[1]]).to(dtype=values.dtype, device=values.device).unsqueeze(-1)
        values = values * valid
    denom = values.amax(dim=1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
    return values / denom


def _terminal_valid_mask(
    actions_is_pad: torch.Tensor | None,
    *,
    batch_size: int,
    horizon: int,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    steps = max(0, int(steps))
    if steps <= 0:
        return torch.zeros((batch_size, horizon), dtype=dtype, device=device)
    if actions_is_pad is None:
        index = torch.arange(horizon, device=device).unsqueeze(0).expand(batch_size, horizon)
        return (index >= max(0, horizon - steps)).to(dtype=dtype)
    valid = (~actions_is_pad[:, :horizon]).to(device=device)
    valid_count = valid.to(dtype=torch.long).sum(dim=1).clamp_min(0)
    index = torch.arange(horizon, device=device).unsqueeze(0).expand(batch_size, horizon)
    start = (valid_count - steps).clamp_min(0).unsqueeze(1)
    end = valid_count.unsqueeze(1)
    return (valid & (index >= start) & (index < end)).to(dtype=dtype)


def _teacher_front_chunk_consistency_loss(
    losses: torch.Tensor,
    *,
    steps: int,
    actions_is_pad: torch.Tensor | None,
) -> torch.Tensor | None:
    """Extra teacher-target loss on early future chunk steps.

    This is intentionally simple: it does not compare model predictions against
    model predictions. It reuses the differentiable SmolVLA teacher loss for
    chunk positions 1..N, which are the first positions affected when receding
    horizon inference re-queries the policy.
    """

    max_steps = min(max(0, int(steps)), max(0, int(losses.shape[1]) - 1))
    if max_steps <= 0:
        return None
    selected = losses[:, 1 : max_steps + 1, :]
    if actions_is_pad is None:
        return selected.mean()
    valid = (~actions_is_pad[:, 1 : max_steps + 1]).to(dtype=losses.dtype, device=losses.device).unsqueeze(-1)
    denom = valid.expand_as(selected).sum().clamp_min(torch.finfo(losses.dtype).eps)
    return (selected * valid).sum() / denom


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


def _copy_visual_servo_labels(source: dict[str, Any], target: dict[str, Any]) -> None:
    device = target.get(OBS_STATE).device if torch.is_tensor(target.get(OBS_STATE)) else None
    for key in (
        "visual_servo.camera1",
        "visual_servo.camera1_visible",
        "visual_servo.camera2",
        "visual_servo.camera2_visible",
        "visual_servo.stop_label",
    ):
        value = source.get(key)
        if value is None:
            continue
        if torch.is_tensor(value) and device is not None:
            value = value.to(device=device)
        target[key] = value


def _batch_dataset_names(batch: dict[str, Any]) -> list[str]:
    value = batch.get("dataset_name")
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if torch.is_tensor(value):
        return [str(item) for item in value.detach().cpu().tolist()]
    return []


def _subset_batch(batch: dict[str, Any], indexes: list[int]) -> dict[str, Any]:
    if not indexes:
        return {}
    index_tensor_cache: dict[str, torch.Tensor] = {}

    def subset_value(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.ndim == 0 or int(value.shape[0]) < max(indexes) + 1:
                return value
            cache_key = f"{value.device}:{len(indexes)}:{','.join(str(i) for i in indexes)}"
            if cache_key not in index_tensor_cache:
                index_tensor_cache[cache_key] = torch.as_tensor(indexes, dtype=torch.long, device=value.device)
            return value.index_select(0, index_tensor_cache[cache_key])
        if isinstance(value, list) and len(value) >= max(indexes) + 1:
            return [value[index] for index in indexes]
        if isinstance(value, tuple) and len(value) >= max(indexes) + 1:
            return tuple(value[index] for index in indexes)
        return value

    return {key: subset_value(value) for key, value in batch.items()}


def _clone_image_tensors_for_logging(batch: dict[str, Any], cameras: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for camera in cameras:
        key = f"observation.images.{camera}"
        value = batch.get(key)
        if torch.is_tensor(value):
            result[key] = value.detach().clone()
        for label_key in (f"visual_servo.{camera}", f"visual_servo.{camera}_visible"):
            label_value = batch.get(label_key)
            if torch.is_tensor(label_value):
                result[label_key] = label_value.detach().clone()
    return result


def _safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "dataset"


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


def _tensorboard_image_with_visual_servo_target(batch: dict[str, Any], camera: str) -> torch.Tensor | None:
    image = _tensorboard_image(batch.get(f"observation.images.{camera}"))
    if image is None:
        return None
    target = _first_visual_servo_target(batch, camera)
    if target is None:
        return image
    return _draw_visual_servo_target_on_image(image, target)


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


def _tensorboard_image_grid_with_visual_servo_target(
    batch: dict[str, Any],
    camera: str,
    *,
    max_images: int = 16,
) -> torch.Tensor | None:
    value = batch.get(f"observation.images.{camera}")
    if not torch.is_tensor(value):
        return None
    tensor = value.detach()
    if tensor.ndim == 3:
        return _tensorboard_image_with_visual_servo_target(batch, camera)
    while tensor.ndim > 4:
        tensor = tensor[0]
    if tensor.ndim != 4:
        return _tensorboard_image_with_visual_servo_target(batch, camera)
    if tensor.shape[1] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.shape[1] == 4:
        tensor = tensor[:, :3]
    if tensor.shape[1] not in (1, 3):
        return _tensorboard_image_with_visual_servo_target(batch, camera)
    images = tensor[: max(1, int(max_images))].float().cpu()
    if images.numel() == 0:
        return None
    if float(images.max()) > 2.0:
        images = images / 255.0
    elif float(images.min()) < 0.0:
        images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)
    targets = _visual_servo_targets(batch, camera, limit=int(images.shape[0]))
    if targets:
        images = images.clone()
        for index, target in targets.items():
            images[index] = _draw_visual_servo_target_on_image(images[index], target)
    rows = int(max(1, round(float(images.shape[0]) ** 0.5)))
    cols = int((images.shape[0] + rows - 1) // rows)
    c, h, w = int(images.shape[1]), int(images.shape[2]), int(images.shape[3])
    grid = torch.zeros((c, rows * h, cols * w), dtype=images.dtype)
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        grid[:, row * h : (row + 1) * h, col * w : (col + 1) * w] = image
    return grid


def _first_visual_servo_target(batch: dict[str, Any], camera: str) -> tuple[float, float] | None:
    targets = _visual_servo_targets(batch, camera, limit=1)
    return targets.get(0)


def _visual_servo_targets(batch: dict[str, Any], camera: str, *, limit: int) -> dict[int, tuple[float, float]]:
    target_value = batch.get(f"visual_servo.{camera}")
    visible_value = batch.get(f"visual_servo.{camera}_visible")
    if not torch.is_tensor(target_value):
        return {}
    target = target_value.detach().float().cpu()
    visible = visible_value.detach().bool().cpu() if torch.is_tensor(visible_value) else torch.ones(target.shape[:1], dtype=torch.bool)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if visible.ndim == 0:
        visible = visible.unsqueeze(0)
    result: dict[int, tuple[float, float]] = {}
    for index in range(min(int(limit), int(target.shape[0]), int(visible.shape[0]))):
        if not bool(visible[index]) or target.shape[-1] < 2:
            continue
        dx = float(target[index, 0].clamp(-1.0, 1.0))
        dy = float(target[index, 1].clamp(-1.0, 1.0))
        result[index] = (dx, dy)
    return result


def _draw_visual_servo_target_on_image(image: torch.Tensor, target: tuple[float, float]) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        return image
    result = image.clone()
    _, height, width = result.shape
    if height <= 0 or width <= 0:
        return result
    dx, dy = target
    x = int(round((width - 1) * (0.5 + 0.5 * dx)))
    y = int(round((height - 1) * (0.5 + 0.5 * dy)))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    color = torch.tensor([1.0, 0.9, 0.0], dtype=result.dtype)
    if result.shape[0] == 1:
        color = color[:1]
    radius = max(2, min(height, width) // 24)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    result[:, y, x0:x1] = color[:, None]
    result[:, y0:y1, x] = color[:, None]
    result[:, y0, x0:x1] = color[:, None]
    result[:, y1 - 1, x0:x1] = color[:, None]
    result[:, y0:y1, x0] = color[:, None]
    result[:, y0:y1, x1 - 1] = color[:, None]
    return result.clamp(0.0, 1.0)


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
