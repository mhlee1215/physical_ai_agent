#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from physical_ai_agent.lerobot_sampling_augmentation import (
    SamplingAugmentationConfig,
    patch_lerobot_train_gpu_augmentation,
    patch_lerobot_train_make_dataset,
    write_sampling_augmentation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LeRobot training with SO101 sample-time state augmentation.",
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
    parser.add_argument("--so101-image-cache-dir", type=Path)
    parser.add_argument("--so101-augmentation-report", type=Path)
    parser.add_argument("--help", action="store_true")
    args, remaining = parser.parse_known_args()

    if args.help:
        print("SO101 wrapper options:")
        parser.print_help()
        print("\nPass regular lerobot-train options after these wrapper options.")
        sys.argv = [sys.argv[0], "--help"]
        from lerobot.scripts.lerobot_train import train

        train()
        return

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
    config = SamplingAugmentationConfig.from_env()
    patch_lerobot_train_make_dataset(config)
    patch_lerobot_train_gpu_augmentation(config)

    report_before_train = args.so101_augmentation_report
    output_dir = _output_dir_from_args(remaining)
    if output_dir is not None and report_before_train is not None:
        try:
            report_before_train.relative_to(output_dir)
            report_before_train = None
        except ValueError:
            pass
    if report_before_train is not None:
        write_sampling_augmentation_report(report_before_train, config, remaining)

    sys.argv = [sys.argv[0], *remaining]
    from lerobot.scripts.lerobot_train import train

    train()
    if args.so101_augmentation_report is not None and report_before_train is None:
        write_sampling_augmentation_report(args.so101_augmentation_report, config, remaining)


def _output_dir_from_args(args: list[str]) -> Path | None:
    for index, arg in enumerate(args):
        if arg.startswith("--output_dir="):
            return Path(arg.split("=", 1)[1]).resolve()
        if arg == "--output_dir" and index + 1 < len(args):
            return Path(args[index + 1]).resolve()
    return None


if __name__ == "__main__":
    main()
