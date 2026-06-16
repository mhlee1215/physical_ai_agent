#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.sim.so101_camera_input import _make_camera, postprocess_camera_frame
from physical_ai_agent.sim.so101_live_viewer import _cartesian_error_controller_action
from physical_ai_agent.sim.so101_visual_rl import _json_safe_info
from train_so101_visual_picklift_bc import _plot_bars, _plot_curve, _resolve_device
from train_so101_visual_picklift_delta import _object_set_description
from train_so101_wrist_ego_picklift_policy import (
    compose_wrist_ego_frame,
    sweep_until_visible,
)
from evaluate_so101_picklift_image_policy import detect_colored_object


@dataclass(frozen=True)
class WristEgoServoConfig:
    width: int = 96
    height: int = 96
    hidden_dim: int = 256
    approach_steps: int = 120
    close_steps: int = 45
    close_visual_servo_interval: int = 2
    close_visual_servo_gain: float = 0.08
    close_visual_servo_gain_schedule: tuple[float, ...] = ()
    lift_steps: int = 70
    lift_gain: float = 0.25
    retry_attempts: int = 4
    retry_backoff_steps: int = 18
    retry_reapproach_steps: int = 55
    retry_close_steps: int = 32
    retry_lift_steps: int = 45
    correction_steps: int = 24
    approach_replan_interval: int = 0
    retry_approach_replan_interval: int = 0
    approach_replan_blend: float = 0.55
    approach_near_samples_per_candidate: int = 8
    approach_near_noise_scale: float = 0.045
    contact_anchor_samples_per_candidate: int = 4
    approach_convergence_tol: float = 0.018
    approach_min_steps: int = 16
    approach_patience: int = 5
    correction_convergence_tol: float = 0.010
    correction_min_steps: int = 6
    correction_patience: int = 4
    close_readiness_weight: float = 0.20
    close_readiness_threshold: float = 0.55
    close_readiness_threshold_schedule: tuple[float, ...] = ()
    close_readiness_patience: int = 3
    close_readiness_min_steps: int = 8
    close_readiness_positive_error: float = 0.090
    skip_close_if_not_ready: bool = True
    close_readiness_label_mode: str = "qpos"
    close_readiness_probe_open_steps: int = 18
    close_readiness_probe_steps: int = 10
    close_break_on_grasp_steps: int = 10
    lift_requires_grasp: bool = True
    servo_target_mode: str = "absolute"
    auxiliary_visual_weight: float = 0.05
    spatial_pool_size: int = 1
    servo_delta_scale: float = 0.45
    lift_delta_scale: float = 0.35
    correction_delta_scale: float = 0.08
    correction_noise_scale: float = 0.12
    max_sweeps: int = 48
    policy_camera_names: tuple[str, str] = ("wrist_cam", "egocentric_cam")
    teacher_camera_names: tuple[str, str, str, str] = ("wrist_cam", "egocentric_cam", "top_down", "scene_3d")
    grasp_mode_names: tuple[str, str, str] = ("front", "diagonal", "overhead")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a wrist+egocentric closed-loop visual-servo policy for SO101 PickLift. "
            "Runtime uses wrist_cam, egocentric_cam, joint positions, and phase only."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_visual_rl/wrist_ego_visual_servo"))
    parser.add_argument("--teacher-seeds", type=int, default=112)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=44000)
    parser.add_argument("--eval-seed", type=int, default=45000)
    parser.add_argument("--render-seed", type=int, default=45014)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--spatial-pool-size", type=int, default=1)
    parser.add_argument("--retry-attempts", type=int, default=4)
    parser.add_argument("--correction-steps", type=int, default=24)
    parser.add_argument("--close-visual-servo-interval", type=int, default=2)
    parser.add_argument("--close-visual-servo-gain", type=float, default=0.08)
    parser.add_argument("--close-visual-servo-gain-schedule", default="")
    parser.add_argument("--close-readiness-threshold", type=float, default=0.55)
    parser.add_argument("--close-readiness-threshold-schedule", default="")
    parser.add_argument("--correction-delta-scale", type=float, default=0.08)
    parser.add_argument("--approach-replan-interval", type=int, default=0)
    parser.add_argument("--retry-approach-replan-interval", type=int, default=0)
    parser.add_argument("--contact-anchor-samples-per-candidate", type=int, default=4)
    parser.add_argument("--lift-gain", type=float, default=0.25)
    parser.add_argument("--servo-target-mode", choices=["absolute", "delta"], default="absolute")
    parser.add_argument("--close-readiness-label-mode", choices=["qpos", "probe"], default="qpos")
    parser.add_argument("--skip-close-if-not-ready", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--use-selector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--correction-checkpoint-path", type=Path)
    parser.add_argument("--selector-checkpoint-path", type=Path)
    args = parser.parse_args()

    config = WristEgoServoConfig(
        width=args.width,
        height=args.height,
        spatial_pool_size=args.spatial_pool_size,
        retry_attempts=args.retry_attempts,
        correction_steps=args.correction_steps,
        close_visual_servo_interval=args.close_visual_servo_interval,
        close_visual_servo_gain=args.close_visual_servo_gain,
        close_visual_servo_gain_schedule=_parse_float_schedule(args.close_visual_servo_gain_schedule),
        close_readiness_threshold=args.close_readiness_threshold,
        close_readiness_threshold_schedule=_parse_float_schedule(args.close_readiness_threshold_schedule),
        correction_delta_scale=args.correction_delta_scale,
        contact_anchor_samples_per_candidate=args.contact_anchor_samples_per_candidate,
        lift_gain=args.lift_gain,
        skip_close_if_not_ready=args.skip_close_if_not_ready,
        approach_replan_interval=args.approach_replan_interval,
        retry_approach_replan_interval=args.retry_approach_replan_interval,
        servo_target_mode=args.servo_target_mode,
        close_readiness_label_mode=args.close_readiness_label_mode,
    )
    if args.eval_only:
        report = evaluate_and_render_existing(
            output_dir=args.output_dir,
            checkpoint_path=_required_path(args.checkpoint_path, "--checkpoint-path"),
            correction_checkpoint_path=_required_path(args.correction_checkpoint_path, "--correction-checkpoint-path"),
            selector_checkpoint_path=args.selector_checkpoint_path if args.use_selector else None,
            config=config,
            eval_episodes=args.eval_episodes,
            eval_seed=args.eval_seed,
            render_seed=args.render_seed,
            device=args.device,
            fps=args.fps,
            use_selector=args.use_selector,
        )
    else:
        report = train_and_evaluate(
            output_dir=args.output_dir,
            teacher_seeds=args.teacher_seeds,
            epochs=args.epochs,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            eval_seed=args.eval_seed,
            render_seed=args.render_seed,
            config=config,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            fps=args.fps,
            use_selector=args.use_selector,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


def train_and_evaluate(
    *,
    output_dir: Path,
    teacher_seeds: int,
    epochs: int,
    eval_episodes: int,
    seed: int,
    eval_seed: int,
    render_seed: int,
    config: WristEgoServoConfig,
    batch_size: int,
    lr: float,
    device: str,
    fps: int,
    use_selector: bool,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    videos_dir = output_dir / "videos"
    plots_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)

    resolved_device = _resolve_device(device)
    runtime_strategy = _runtime_strategy(config)
    print(
        "[visual-servo] start "
        f"teacher_seeds={teacher_seeds} epochs={epochs} eval_episodes={eval_episodes} "
        f"device={resolved_device} output_dir={output_dir}",
        flush=True,
    )
    dataset = collect_dense_servo_dataset(config=config, teacher_seeds=teacher_seeds, seed=seed)
    print(
        "[visual-servo] dataset ready "
        f"samples={dataset['summary']['samples']} "
        f"teacher_successes={dataset['summary']['teacher_successes']}/"
        f"{dataset['summary']['teacher_seeds_requested']} "
        f"visible_after_sweep={dataset['summary']['visible_after_sweep']}",
        flush=True,
    )
    model = WristEgoVisualServoPolicy(config=config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    image_tensor = torch.as_tensor(dataset["images"], dtype=torch.uint8)
    qpos_tensor = torch.as_tensor(dataset["qpos"], dtype=torch.float32)
    phase_tensor = torch.as_tensor(dataset["phase"], dtype=torch.float32)
    action_tensor = torch.as_tensor(dataset["targets"], dtype=torch.float32)
    auxiliary_tensor = torch.as_tensor(dataset["auxiliary"], dtype=torch.float32)
    correction_image_tensor = torch.as_tensor(dataset["correction_images"], dtype=torch.uint8)
    correction_qpos_tensor = torch.as_tensor(dataset["correction_qpos"], dtype=torch.float32)
    correction_phase_tensor = torch.as_tensor(dataset["correction_phase"], dtype=torch.float32)
    correction_target_tensor = torch.as_tensor(dataset["correction_targets"], dtype=torch.float32)
    correction_readiness_tensor = torch.as_tensor(dataset["correction_readiness"], dtype=torch.float32)
    selector_image_tensor = torch.as_tensor(dataset["selector_images"], dtype=torch.uint8)
    selector_qpos_tensor = torch.as_tensor(dataset["selector_qpos"], dtype=torch.float32)
    selector_target_tensor = torch.as_tensor(dataset["selector_targets"], dtype=torch.long)
    readiness_positive_rate = float(dataset["summary"].get("correction_readiness_positive_rate", 0.0))
    readiness_pos_weight_value = 1.0
    if 0.0 < readiness_positive_rate < 0.5:
        readiness_pos_weight_value = min(8.0, (1.0 - readiness_positive_rate) / readiness_positive_rate)
    readiness_pos_weight = torch.as_tensor([readiness_pos_weight_value], dtype=torch.float32).to(resolved_device)
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    losses: list[float] = []
    action_losses: list[float] = []
    auxiliary_losses: list[float] = []
    correction_losses: list[float] = []
    correction_delta_losses: list[float] = []
    close_readiness_losses: list[float] = []
    selector_losses: list[float] = []
    selector_accuracies: list[float] = []
    progress_every = max(1, epochs // 10)
    for _epoch in range(epochs):
        order = torch.randperm(len(action_tensor), generator=generator)
        epoch_losses = []
        epoch_action_losses = []
        epoch_auxiliary_losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            pred, aux_pred = model(
                image_tensor[idx].to(resolved_device),
                qpos_tensor[idx].to(resolved_device),
                phase_tensor[idx].to(resolved_device),
                return_aux=True,
            )
            target = action_tensor[idx].to(resolved_device)
            aux_target = auxiliary_tensor[idx].to(resolved_device)
            action_loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=0.05)
            aux_loss = _auxiliary_visual_loss(torch, aux_pred, aux_target)
            loss = action_loss + float(config.auxiliary_visual_weight) * aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_action_losses.append(float(action_loss.detach().cpu()))
            epoch_auxiliary_losses.append(float(aux_loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
        action_losses.append(float(np.mean(epoch_action_losses)))
        auxiliary_losses.append(float(np.mean(epoch_auxiliary_losses)))
        correction_order = torch.randperm(len(correction_target_tensor), generator=generator)
        if _epoch == 0:
            correction_model = WristEgoVisualCorrectionPolicy(config=config).to(resolved_device)
            correction_optimizer = torch.optim.AdamW(correction_model.parameters(), lr=lr, weight_decay=1e-4)
        correction_epoch_losses = []
        correction_delta_epoch_losses = []
        close_readiness_epoch_losses = []
        for start in range(0, len(correction_order), batch_size):
            idx = correction_order[start : start + batch_size]
            pred, readiness_logit = correction_model(
                correction_image_tensor[idx].to(resolved_device),
                correction_qpos_tensor[idx].to(resolved_device),
                correction_phase_tensor[idx].to(resolved_device),
                return_readiness=True,
            )
            target = correction_target_tensor[idx].to(resolved_device)
            readiness_target = correction_readiness_tensor[idx].to(resolved_device)
            correction_delta_loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=0.05)
            close_readiness_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                readiness_logit,
                readiness_target,
                pos_weight=readiness_pos_weight,
            )
            correction_loss = (
                correction_delta_loss
                + float(config.close_readiness_weight) * close_readiness_loss
            )
            correction_optimizer.zero_grad()
            correction_loss.backward()
            correction_optimizer.step()
            correction_epoch_losses.append(float(correction_loss.detach().cpu()))
            correction_delta_epoch_losses.append(float(correction_delta_loss.detach().cpu()))
            close_readiness_epoch_losses.append(float(close_readiness_loss.detach().cpu()))
        correction_losses.append(float(np.mean(correction_epoch_losses)))
        correction_delta_losses.append(float(np.mean(correction_delta_epoch_losses)))
        close_readiness_losses.append(float(np.mean(close_readiness_epoch_losses)))
        selector_order = torch.randperm(len(selector_target_tensor), generator=generator)
        if _epoch == 0:
            selector_model = WristEgoGraspModeSelector(config=config).to(resolved_device)
            selector_optimizer = torch.optim.AdamW(selector_model.parameters(), lr=lr, weight_decay=1e-4)
        selector_epoch_losses = []
        selector_correct = 0
        selector_seen = 0
        for start in range(0, len(selector_order), batch_size):
            idx = selector_order[start : start + batch_size]
            logits = selector_model(
                selector_image_tensor[idx].to(resolved_device),
                selector_qpos_tensor[idx].to(resolved_device),
            )
            target = selector_target_tensor[idx].to(resolved_device)
            selector_loss = torch.nn.functional.cross_entropy(logits, target)
            selector_optimizer.zero_grad()
            selector_loss.backward()
            selector_optimizer.step()
            selector_epoch_losses.append(float(selector_loss.detach().cpu()))
            selector_correct += int((logits.argmax(dim=1) == target).sum().detach().cpu())
            selector_seen += int(target.numel())
        selector_losses.append(float(np.mean(selector_epoch_losses)))
        selector_accuracies.append(float(selector_correct / max(1, selector_seen)))
        if (_epoch + 1) == 1 or (_epoch + 1) % progress_every == 0 or (_epoch + 1) == epochs:
            print(
                f"[visual-servo] epoch {_epoch + 1}/{epochs} "
                f"loss={losses[-1]:.6f} correction_loss={correction_losses[-1]:.6f} "
                f"readiness_loss={close_readiness_losses[-1]:.6f} "
                f"selector_loss={selector_losses[-1]:.6f} "
                f"selector_acc={selector_accuracies[-1]:.3f}",
                flush=True,
            )

    checkpoint_path = output_dir / "wrist_ego_visual_servo.pt"
    save_checkpoint(
        checkpoint_path,
        model.cpu(),
        {
            "operation": "train_so101_wrist_ego_visual_servo",
            "runtime_inputs": ["wrist_cam", "egocentric_cam", "joint_positions", "phase", "grasp_mode"],
            "runtime_excludes": ["top_down", "camera_calibration", "object_pose", "mujoco_jacobian"],
            "runtime_strategy": runtime_strategy,
            "training_teacher_uses": ["all simulated cameras", "simulation object pose", "IK", "lift controller"],
            "object_set": _high_contrast_object_description(),
            "config": asdict(config),
            "dataset": dataset["summary"],
            "epochs": epochs,
            "device": resolved_device,
        },
    )
    correction_checkpoint_path = output_dir / "wrist_ego_visual_correction.pt"
    save_checkpoint(
        correction_checkpoint_path,
        correction_model.cpu(),
        {
            "operation": "train_so101_wrist_ego_visual_correction",
            "runtime_inputs": ["wrist_cam", "egocentric_cam", "joint_positions", "grasp_mode"],
            "runtime_excludes": ["top_down", "camera_calibration", "object_pose", "mujoco_jacobian"],
            "training_teacher_uses": ["simulation object pose", "IK-selected close pregrasp target"],
            "object_set": _high_contrast_object_description(),
            "config": asdict(config),
            "dataset": dataset["summary"],
            "epochs": epochs,
            "device": resolved_device,
            "delta_scale": config.correction_delta_scale,
        },
    )
    selector_checkpoint_path = output_dir / "wrist_ego_grasp_mode_selector.pt"
    save_checkpoint(
        selector_checkpoint_path,
        selector_model.cpu(),
        {
            "operation": "train_so101_wrist_ego_grasp_mode_selector",
            "runtime_inputs": ["wrist_cam", "egocentric_cam", "joint_positions"],
            "runtime_excludes": ["top_down", "camera_calibration", "object_pose", "mujoco_jacobian"],
            "training_teacher_uses": ["all simulated cameras", "simulation object pose", "IK candidate scoring"],
            "object_set": _high_contrast_object_description(),
            "config": asdict(config),
            "dataset": dataset["summary"],
            "epochs": epochs,
            "device": resolved_device,
        },
    )

    eval_report = evaluate_policy(
        checkpoint_path=checkpoint_path,
        correction_checkpoint_path=correction_checkpoint_path,
        selector_checkpoint_path=selector_checkpoint_path if use_selector else None,
        config=config,
        episodes=eval_episodes,
        seed=eval_seed,
        device=resolved_device,
    )
    rollout = render_policy_rollout(
        checkpoint_path=checkpoint_path,
        correction_checkpoint_path=correction_checkpoint_path,
        selector_checkpoint_path=selector_checkpoint_path if use_selector else None,
        config=config,
        seed=render_seed,
        output_dir=videos_dir,
        fps=fps,
        device=resolved_device,
    )
    loss_plot = plots_dir / "wrist_ego_visual_servo_loss.png"
    success_plot = plots_dir / "wrist_ego_visual_servo_success.png"
    _plot_curve(losses, loss_plot, title="Wrist+ego visual-servo loss", ylabel="SmoothL1")
    _plot_bars(eval_report["episodes"], success_plot)
    manifest = {
        "operation": "train_so101_wrist_ego_visual_servo",
        "checkpoint_path": str(checkpoint_path),
        "correction_checkpoint_path": str(correction_checkpoint_path),
        "selector_checkpoint_path": str(selector_checkpoint_path),
        "runtime_inputs": ["wrist_cam", "egocentric_cam", "joint_positions", "phase", "grasp_mode"],
        "runtime_excludes": ["top_down", "camera_calibration", "object_pose", "mujoco_jacobian"],
        "runtime_strategy": runtime_strategy,
        "selector_used_at_runtime": use_selector,
        "training_teacher_uses": ["all simulated cameras", "simulation object pose", "IK", "lift controller"],
        "object_set": _high_contrast_object_description(),
        "dataset": dataset["summary"],
        "training": {
            "losses": losses,
            "action_losses": action_losses,
            "auxiliary_losses": auxiliary_losses,
            "final_loss": losses[-1] if losses else None,
            "correction_losses": correction_losses,
            "correction_delta_losses": correction_delta_losses,
            "close_readiness_losses": close_readiness_losses,
            "selector_losses": selector_losses,
            "selector_accuracies": selector_accuracies,
            "final_correction_loss": correction_losses[-1] if correction_losses else None,
            "final_close_readiness_loss": close_readiness_losses[-1] if close_readiness_losses else None,
            "final_selector_loss": selector_losses[-1] if selector_losses else None,
            "final_selector_accuracy": selector_accuracies[-1] if selector_accuracies else None,
            "close_readiness_pos_weight": readiness_pos_weight_value,
        },
        "evaluation": eval_report,
        "artifacts": {
            "loss_plot": str(loss_plot),
            "success_plot": str(success_plot),
            "rollout_gif": rollout["gif_path"],
            "rollout_mp4": rollout["mp4_path"],
            "rollout_manifest": rollout["manifest_path"],
        },
    }
    manifest_path = output_dir / "wrist_ego_visual_servo_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "[visual-servo] done "
        f"success_rate={eval_report['success_rate']:.3f} "
        f"grasp_rate={eval_report['grasp_rate']:.3f} manifest={manifest_path}",
        flush=True,
    )
    return manifest


def evaluate_and_render_existing(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    correction_checkpoint_path: Path,
    selector_checkpoint_path: Path | None,
    config: WristEgoServoConfig,
    eval_episodes: int,
    eval_seed: int,
    render_seed: int,
    device: str,
    fps: int,
    use_selector: bool,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    eval_report = evaluate_policy(
        checkpoint_path=checkpoint_path,
        correction_checkpoint_path=correction_checkpoint_path,
        selector_checkpoint_path=selector_checkpoint_path if use_selector else None,
        config=config,
        episodes=eval_episodes,
        seed=eval_seed,
        device=resolved_device,
    )
    rollout = render_policy_rollout(
        checkpoint_path=checkpoint_path,
        correction_checkpoint_path=correction_checkpoint_path,
        selector_checkpoint_path=selector_checkpoint_path if use_selector else None,
        config=config,
        seed=render_seed,
        output_dir=videos_dir,
        fps=fps,
        device=resolved_device,
    )
    manifest = {
        "operation": "eval_so101_wrist_ego_visual_servo",
        "checkpoint_path": str(checkpoint_path),
        "correction_checkpoint_path": str(correction_checkpoint_path),
        "selector_checkpoint_path": str(selector_checkpoint_path) if selector_checkpoint_path else None,
        "selector_used_at_runtime": use_selector,
        "runtime_inputs": ["wrist_cam", "egocentric_cam", "joint_positions", "phase", "grasp_mode"],
        "runtime_excludes": ["top_down", "camera_calibration", "object_pose", "mujoco_jacobian"],
        "runtime_strategy": _runtime_strategy(config),
        "device": resolved_device,
        "evaluation": eval_report,
        "artifacts": {
            "rollout_gif": rollout["gif_path"],
            "rollout_mp4": rollout["mp4_path"],
            "rollout_manifest": rollout["manifest_path"],
        },
    }
    manifest_path = output_dir / "wrist_ego_visual_servo_eval_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _required_path(value: Path | None, flag: str) -> Path:
    if value is None:
        raise ValueError(f"{flag} is required with --eval-only")
    return value


def _parse_float_schedule(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


class WristEgoVisualServoPolicy:
    def __new__(cls, *, config: WristEgoServoConfig) -> Any:
        import torch
        from torch import nn

        class _Policy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 160, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((config.spatial_pool_size, config.spatial_pool_size)),
                    nn.Flatten(),
                )
                feature_dim = 160 * config.spatial_pool_size * config.spatial_pool_size
                self.head = nn.Sequential(
                    nn.Linear(feature_dim + 6 + 2 + len(config.grasp_mode_names), config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, 6),
                    nn.Tanh(),
                )
                self.aux_head = nn.Sequential(
                    nn.Linear(feature_dim, 96),
                    nn.ReLU(),
                    nn.Linear(96, 6),
                    nn.Tanh(),
                )

            def forward(self, image: Any, qpos: Any, phase: Any, return_aux: bool = False) -> Any:
                features = self.encoder(image.float() / 255.0)
                action = self.head(torch.cat([features, qpos, phase], dim=1))
                if return_aux:
                    return action, self.aux_head(features)
                return action

        return _Policy()


class WristEgoVisualCorrectionPolicy:
    def __new__(cls, *, config: WristEgoServoConfig) -> Any:
        import torch
        from torch import nn

        class _Policy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 160, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((config.spatial_pool_size, config.spatial_pool_size)),
                    nn.Flatten(),
                )
                feature_dim = 160 * config.spatial_pool_size * config.spatial_pool_size
                self.head = nn.Sequential(
                    nn.Linear(feature_dim + 6 + 2 + len(config.grasp_mode_names), config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, 6),
                    nn.Tanh(),
                )
                self.readiness_head = nn.Sequential(
                    nn.Linear(feature_dim + 6 + 2 + len(config.grasp_mode_names), 96),
                    nn.ReLU(),
                    nn.Linear(96, 1),
                )

            def forward(self, image: Any, qpos: Any, condition: Any, return_readiness: bool = False) -> Any:
                features = self.encoder(image.float() / 255.0)
                conditioned = torch.cat([features, qpos, condition], dim=1)
                delta = self.head(conditioned)
                if return_readiness:
                    return delta, self.readiness_head(conditioned)
                return delta

        return _Policy()


class WristEgoGraspModeSelector:
    def __new__(cls, *, config: WristEgoServoConfig) -> Any:
        import torch
        from torch import nn

        class _Selector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 160, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                self.head = nn.Sequential(
                    nn.Linear(160 + 6, 128),
                    nn.ReLU(),
                    nn.Linear(128, len(config.grasp_mode_names)),
                )

            def forward(self, image: Any, qpos: Any) -> Any:
                features = self.encoder(image.float() / 255.0)
                return self.head(torch.cat([features, qpos], dim=1))

        return _Selector()


def collect_dense_servo_dataset(
    *,
    config: WristEgoServoConfig,
    teacher_seeds: int,
    seed: int,
) -> dict[str, Any]:
    import mujoco

    env = make_high_contrast_picklift_env()
    policy_renderers = _make_policy_renderers(env, config)
    teacher_renderers = _make_teacher_renderers(env, config)
    rng = np.random.default_rng(seed)
    images = []
    qpos_rows = []
    phase_rows = []
    targets = []
    auxiliary_rows = []
    correction_images = []
    correction_qpos_rows = []
    correction_phase_rows = []
    correction_targets = []
    correction_readiness_rows = []
    selector_images = []
    selector_qpos_rows = []
    selector_targets = []
    visible_count = 0
    teacher_count = 0
    teacher_visible_count = 0
    teacher_candidate_successes = 0
    teacher_candidate_attempts = 0
    teacher_grasp_modes: dict[str, int] = {}
    try:
        for index in range(teacher_seeds):
            episode_seed = seed + index
            env.reset(seed=episode_seed)
            teacher_visible = object_visible_to_teacher(env, teacher_renderers, config=config)
            visible, _search_steps = sweep_until_visible(env, policy_renderers, max_sweeps=config.max_sweeps)
            if not visible:
                continue
            teacher_visible = teacher_visible or object_visible_to_teacher(env, teacher_renderers, config=config)
            visible_count += 1
            q_start = _current_qpos(env)
            try:
                teacher_candidates = make_teacher_targets(env)
            except Exception:
                continue
            teacher_candidate_attempts += len(_grasp_candidate_specs(env))
            teacher_candidate_successes += len(teacher_candidates)
            if not teacher_candidates:
                continue
            teacher_count += 1
            teacher_visible_count += int(teacher_visible)
            best_candidate = max(teacher_candidates, key=lambda item: float(item["meta"].get("score", -1e9)))
            selector_images.append(render_wrist_ego(env, policy_renderers))
            selector_qpos_rows.append(_normalize_qpos(_current_qpos(env), env).astype(np.float32))
            selector_targets.append(int(config.grasp_mode_names.index(best_candidate["meta"]["mode"])))
            for candidate in teacher_candidates:
                q_open = candidate["q_open"]
                q_lift = candidate["q_lift"]
                grasp_mode = candidate["meta"]["mode"]
                teacher_grasp_modes[grasp_mode] = teacher_grasp_modes.get(grasp_mode, 0) + 1
                for alpha in np.linspace(0.0, 0.95, 9):
                    base = (1.0 - alpha) * q_start + alpha * q_open
                    for _repeat in range(2):
                        noise = np.concatenate([rng.normal(0.0, 0.12 * (1.0 - alpha) + 0.025, 5), [0.0]])
                        qpos = base + noise
                        qpos[-1] = _open_gripper_qpos(env)
                        _set_qpos(env, qpos)
                        _append_sample(
                            env=env,
                            renderers=policy_renderers,
                            images=images,
                            qpos_rows=qpos_rows,
                            phase_rows=phase_rows,
                            targets=targets,
                            auxiliary_rows=auxiliary_rows,
                            phase=_condition_vector(config, (1.0, 0.0), grasp_mode),
                            target=q_open,
                            target_mode=config.servo_target_mode,
                            delta_scale=config.servo_delta_scale,
                        )
                for _repeat in range(config.approach_near_samples_per_candidate):
                    qpos = q_open + np.concatenate(
                        [rng.normal(0.0, config.approach_near_noise_scale, 5), [0.0]]
                    )
                    qpos[-1] = _open_gripper_qpos(env)
                    _set_qpos(env, qpos)
                    _append_sample(
                        env=env,
                        renderers=policy_renderers,
                        images=images,
                        qpos_rows=qpos_rows,
                        phase_rows=phase_rows,
                        targets=targets,
                        auxiliary_rows=auxiliary_rows,
                        phase=_condition_vector(config, (1.0, 0.0), grasp_mode),
                        target=q_open,
                        target_mode=config.servo_target_mode,
                        delta_scale=config.servo_delta_scale,
                    )
                for anchor_index in range(config.contact_anchor_samples_per_candidate):
                    if anchor_index == 0:
                        qpos = q_open.copy()
                    else:
                        qpos = q_open + np.concatenate([rng.normal(0.0, 0.006, 5), [0.0]])
                    qpos[-1] = _open_gripper_qpos(env)
                    _set_qpos(env, qpos)
                    _append_sample(
                        env=env,
                        renderers=policy_renderers,
                        images=images,
                        qpos_rows=qpos_rows,
                        phase_rows=phase_rows,
                        targets=targets,
                        auxiliary_rows=auxiliary_rows,
                        phase=_condition_vector(config, (1.0, 0.0), grasp_mode),
                        target=q_open,
                        target_mode=config.servo_target_mode,
                        delta_scale=config.servo_delta_scale,
                    )
                    _append_correction_sample(
                        env=env,
                        renderers=policy_renderers,
                        images=correction_images,
                        qpos_rows=correction_qpos_rows,
                        condition_rows=correction_phase_rows,
                        targets=correction_targets,
                        readiness_rows=correction_readiness_rows,
                        condition=_condition_vector(config, (0.0, 0.0), grasp_mode),
                        target_qpos=q_open,
                        delta_scale=config.correction_delta_scale,
                        ready_error=config.close_readiness_positive_error,
                        label_mode=config.close_readiness_label_mode,
                        probe_open_steps=config.close_readiness_probe_open_steps,
                        probe_steps=config.close_readiness_probe_steps,
                    )
                for _repeat in range(8):
                    qpos = q_open + np.concatenate(
                        [rng.normal(0.0, config.correction_noise_scale, 5), [0.0]]
                    )
                    qpos[-1] = _open_gripper_qpos(env)
                    _set_qpos(env, qpos)
                    _append_correction_sample(
                        env=env,
                        renderers=policy_renderers,
                        images=correction_images,
                        qpos_rows=correction_qpos_rows,
                        condition_rows=correction_phase_rows,
                        targets=correction_targets,
                        readiness_rows=correction_readiness_rows,
                        condition=_condition_vector(config, (0.0, 0.0), grasp_mode),
                        target_qpos=q_open,
                        delta_scale=config.correction_delta_scale,
                        ready_error=config.close_readiness_positive_error,
                        label_mode=config.close_readiness_label_mode,
                        probe_open_steps=config.close_readiness_probe_open_steps,
                        probe_steps=config.close_readiness_probe_steps,
                    )
                for _repeat in range(6):
                    qpos = q_open + np.concatenate([rng.normal(0.0, 0.018, 5), [0.0]])
                    qpos[-1] = _open_gripper_qpos(env)
                    _set_qpos(env, qpos)
                    _append_correction_sample(
                        env=env,
                        renderers=policy_renderers,
                        images=correction_images,
                        qpos_rows=correction_qpos_rows,
                        condition_rows=correction_phase_rows,
                        targets=correction_targets,
                        readiness_rows=correction_readiness_rows,
                        condition=_condition_vector(config, (0.0, 0.0), grasp_mode),
                        target_qpos=q_open,
                        delta_scale=config.correction_delta_scale,
                        ready_error=config.close_readiness_positive_error,
                        label_mode=config.close_readiness_label_mode,
                        probe_open_steps=config.close_readiness_probe_open_steps,
                        probe_steps=config.close_readiness_probe_steps,
                    )
                for _repeat in range(6):
                    qpos = q_open + np.concatenate([rng.normal(0.0, 0.035, 5), [0.0]])
                    qpos[-1] = float(env.action_space.low[-1])
                    _set_qpos(env, qpos)
                    _append_sample(
                        env=env,
                        renderers=policy_renderers,
                        images=images,
                        qpos_rows=qpos_rows,
                        phase_rows=phase_rows,
                        targets=targets,
                        auxiliary_rows=auxiliary_rows,
                        phase=_condition_vector(config, (0.0, 1.0), grasp_mode),
                        target=q_lift,
                        target_mode=config.servo_target_mode,
                        delta_scale=config.lift_delta_scale,
                    )
            if (index + 1) == 1 or (index + 1) % max(1, teacher_seeds // 10) == 0 or (index + 1) == teacher_seeds:
                print(
                    "[visual-servo] teacher collection "
                    f"{index + 1}/{teacher_seeds} visible={visible_count} "
                    f"teacher_successes={teacher_count} samples={len(images)}",
                    flush=True,
                )
    finally:
        for renderer in [*policy_renderers.values(), *teacher_renderers.values()]:
            renderer.close()
        env.close()

    if not images:
        raise RuntimeError("No dense visual-servo samples collected")
    print(
        "[visual-servo] teacher collection complete "
        f"visible={visible_count}/{teacher_seeds} teacher_successes={teacher_count} "
        f"samples={len(images)}",
        flush=True,
    )
    return {
        "images": np.stack(images, axis=0),
        "qpos": np.stack(qpos_rows, axis=0),
        "phase": np.stack(phase_rows, axis=0),
        "targets": np.stack(targets, axis=0),
        "auxiliary": np.stack(auxiliary_rows, axis=0),
        "correction_images": np.stack(correction_images, axis=0),
        "correction_qpos": np.stack(correction_qpos_rows, axis=0),
        "correction_phase": np.stack(correction_phase_rows, axis=0),
        "correction_targets": np.stack(correction_targets, axis=0),
        "correction_readiness": np.asarray(correction_readiness_rows, dtype=np.float32).reshape(-1, 1),
        "selector_images": np.stack(selector_images, axis=0),
        "selector_qpos": np.stack(selector_qpos_rows, axis=0),
        "selector_targets": np.asarray(selector_targets, dtype=np.int64),
        "summary": {
            "teacher_seeds_requested": teacher_seeds,
            "visible_after_sweep": visible_count,
            "teacher_visible_in_any_camera": teacher_visible_count,
            "teacher_successes": teacher_count,
            "teacher_candidate_successes": teacher_candidate_successes,
            "teacher_candidate_attempts": teacher_candidate_attempts,
            "teacher_grasp_modes": teacher_grasp_modes,
            "samples": len(images),
            "correction_samples": len(correction_images),
            "selector_samples": len(selector_images),
            "selector_best_modes": {
                name: int(np.sum(np.asarray(selector_targets) == mode_index))
                for mode_index, name in enumerate(config.grasp_mode_names)
            },
            "correction_readiness_positive_rate": float(np.mean(correction_readiness_rows))
            if correction_readiness_rows
            else 0.0,
            "auxiliary_visible_wrist_rate": float(np.mean([row[0] > 0.0 for row in auxiliary_rows]))
            if auxiliary_rows
            else 0.0,
            "auxiliary_visible_ego_rate": float(np.mean([row[3] > 0.0 for row in auxiliary_rows]))
            if auxiliary_rows
            else 0.0,
            "samples_per_teacher": float(len(images) / teacher_count) if teacher_count else 0.0,
        },
    }


def _append_sample(
    *,
    env: Any,
    renderers: dict[str, Any],
    images: list[np.ndarray],
    qpos_rows: list[np.ndarray],
    phase_rows: list[np.ndarray],
    targets: list[np.ndarray],
    auxiliary_rows: list[np.ndarray],
    phase: tuple[float, float],
    target: np.ndarray,
    target_mode: str,
    delta_scale: float,
) -> None:
    current = _current_qpos(env)
    if target_mode == "delta":
        delta = np.asarray(target, dtype=float) - current
        scale = _servo_delta_scale_vector(env, delta_scale)
        normalized_target = np.clip(delta / scale, -1.0, 1.0)
        normalized_target[-1] = 0.0
    else:
        normalized_target = _normalize_qpos(target, env)
    image = render_wrist_ego(env, renderers)
    images.append(image)
    qpos_rows.append(_normalize_qpos(current, env).astype(np.float32))
    phase_rows.append(np.asarray(phase, dtype=np.float32))
    targets.append(normalized_target.astype(np.float32))
    auxiliary_rows.append(_visual_auxiliary_label(image).astype(np.float32))


def _append_correction_sample(
    *,
    env: Any,
    renderers: dict[str, Any],
    images: list[np.ndarray],
    qpos_rows: list[np.ndarray],
    condition_rows: list[np.ndarray],
    targets: list[np.ndarray],
    readiness_rows: list[float],
    condition: tuple[float, ...],
    target_qpos: np.ndarray,
    delta_scale: float,
    ready_error: float,
    label_mode: str,
    probe_open_steps: int,
    probe_steps: int,
) -> None:
    current = _current_qpos(env)
    delta = np.asarray(target_qpos, dtype=float) - current
    scale = _correction_delta_scale_vector(env, delta_scale)
    normalized_delta = np.clip(delta / scale, -1.0, 1.0)
    normalized_delta[-1] = 0.0
    image = render_wrist_ego(env, renderers)
    qpos_ready = float(np.linalg.norm(delta[:5]) <= float(ready_error))
    close_probe_ready = False
    if label_mode == "probe":
        close_probe_ready = _close_probe_is_grasped(
            env,
            open_steps=probe_open_steps,
            close_steps=probe_steps,
        )
    images.append(image)
    qpos_rows.append(_normalize_qpos(current, env).astype(np.float32))
    condition_rows.append(np.asarray(condition, dtype=np.float32))
    targets.append(normalized_delta.astype(np.float32))
    readiness_rows.append(float(close_probe_ready if label_mode == "probe" else qpos_ready))


def evaluate_policy(
    *,
    checkpoint_path: Path,
    correction_checkpoint_path: Path | None,
    selector_checkpoint_path: Path | None = None,
    config: WristEgoServoConfig,
    episodes: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    import mujoco
    import torch

    model, metadata = load_checkpoint(checkpoint_path)
    correction_model = load_correction_checkpoint(correction_checkpoint_path)[0] if correction_checkpoint_path else None
    selector_model = load_selector_checkpoint(selector_checkpoint_path)[0] if selector_checkpoint_path else None
    model = model.to(device).eval()
    if correction_model is not None:
        correction_model = correction_model.to(device).eval()
    if selector_model is not None:
        selector_model = selector_model.to(device).eval()
    env = make_high_contrast_picklift_env()
    renderers = _make_policy_renderers(env, config)
    rows = []
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            visible, search_steps = sweep_until_visible(env, renderers, max_sweeps=config.max_sweeps)
            if not visible:
                rows.append(_drop_row(episode=episode, seed=seed + episode, search_steps=search_steps))
                print(
                    f"[visual-servo] eval {episode + 1}/{episodes} seed={seed + episode} "
                    f"dropped search_steps={search_steps}",
                    flush=True,
                )
                continue
            result = run_policy_episode(
                env=env,
                renderers=renderers,
                model=model,
                correction_model=correction_model,
                selector_model=selector_model,
                config=config,
                device=device,
            )
            rows.append({"episode": episode, "seed": seed + episode, "search_steps": search_steps, **result})
            print(
                f"[visual-servo] eval {episode + 1}/{episodes} seed={seed + episode} "
                f"success={result['success']} grasp={result['final_is_grasped']:.1f} "
                f"lift={result['final_lift_height']:.4f} dist={result['final_tcp_to_obj_dist']:.4f}",
                flush=True,
            )
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()
    success_steps = [item["steps_to_success"] for item in rows if item["steps_to_success"] is not None]
    return {
        "episodes": rows,
        "success_rate": float(np.mean([item["success"] for item in rows])),
        "grasp_rate": float(np.mean([item["final_is_grasped"] > 0.5 for item in rows])),
        "mean_final_lift_height": float(np.mean([item["final_lift_height"] for item in rows])),
        "mean_steps_to_success": float(np.mean(success_steps)) if success_steps else None,
        "checkpoint_metadata": metadata,
        "metadata": {
            "config": asdict(config),
            "device": device,
            "runtime_inputs": ["wrist_cam", "egocentric_cam", "joint_positions", "phase", "grasp_mode"],
            "runtime_excludes": ["top_down", "camera_calibration", "object_pose", "mujoco_jacobian"],
            "runtime_strategy": _runtime_strategy(config),
        },
    }


def run_policy_episode(
    *,
    env: Any,
    renderers: dict[str, Any],
    model: Any,
    correction_model: Any | None,
    config: WristEgoServoConfig,
    device: str,
    selector_model: Any | None = None,
    frame_callback: Any | None = None,
) -> dict[str, Any]:
    import torch

    records = []
    info = env.unwrapped._get_info()
    success_step = None
    global_step = 0
    total_attempts = max(1, int(config.retry_attempts))
    grasp_mode_order = _predict_grasp_mode_order(
        env=env,
        renderers=renderers,
        model=selector_model,
        config=config,
        device=device,
    )
    for attempt in range(total_attempts):
        close_gain = _scheduled_value(
            config.close_visual_servo_gain_schedule,
            attempt,
            config.close_visual_servo_gain,
        )
        close_threshold = _scheduled_value(
            config.close_readiness_threshold_schedule,
            attempt,
            config.close_readiness_threshold,
        )
        grasp_mode = grasp_mode_order[attempt % len(grasp_mode_order)]
        approach_steps = config.approach_steps if attempt == 0 else config.retry_reapproach_steps
        approach_replan_interval = (
            config.approach_replan_interval
            if attempt == 0
            else config.retry_approach_replan_interval
        )
        close_steps = config.close_steps if attempt == 0 else config.retry_close_steps
        lift_steps = config.lift_steps if attempt == total_attempts - 1 else config.retry_lift_steps
        offset = _retry_grasp_offset(env, attempt)
        approach_target = _predict_servo_target(
            env=env,
            renderers=renderers,
            model=model,
            phase=_condition_vector(config, (1.0, 0.0), grasp_mode),
            device=device,
            target_mode=config.servo_target_mode,
            delta_scale=config.servo_delta_scale,
        )
        approach_target = np.clip(approach_target + offset, env.action_space.low, env.action_space.high)
        approach_target[-1] = _open_gripper_qpos(env)
        approach_settled = 0
        for _approach_index in range(approach_steps):
            if (
                approach_replan_interval > 0
                and _approach_index > 0
                and _approach_index % approach_replan_interval == 0
            ):
                refreshed_target = _predict_servo_target(
                    env=env,
                    renderers=renderers,
                    model=model,
                    phase=_condition_vector(config, (1.0, 0.0), grasp_mode),
                    device=device,
                    target_mode=config.servo_target_mode,
                    delta_scale=config.servo_delta_scale,
                )
                refreshed_target = np.clip(
                    refreshed_target + offset,
                    env.action_space.low,
                    env.action_space.high,
                )
                refreshed_target[-1] = _open_gripper_qpos(env)
                blend = float(np.clip(config.approach_replan_blend, 0.0, 1.0))
                approach_target = np.clip(
                    (1.0 - blend) * approach_target + blend * refreshed_target,
                    env.action_space.low,
                    env.action_space.high,
                )
                approach_target[-1] = _open_gripper_qpos(env)
                approach_settled = 0
            info = _step_towards(env, approach_target, gain=0.42, gripper=_open_gripper_qpos(env))
            _record(records, step=global_step, phase="approach", info=info, attempt=attempt)
            if frame_callback is not None:
                frame_callback(global_step, f"{grasp_mode}_attempt{attempt + 1}_approach", approach_target, info)
            global_step += 1
            target_error = _arm_target_error(env, approach_target)
            if target_error < config.approach_convergence_tol:
                approach_settled += 1
            else:
                approach_settled = 0
            if (
                _approach_index + 1 >= config.approach_min_steps
                and approach_settled >= config.approach_patience
            ):
                break
        q_close = np.clip(_current_qpos(env) + offset * 0.35, env.action_space.low, env.action_space.high)
        best_close_readiness = 0.0
        if correction_model is not None and config.correction_steps > 0:
            correction_settled = 0
            close_ready_settled = 0
            for _correction_index in range(config.correction_steps):
                delta, close_readiness = _predict_correction_delta(
                    env=env,
                    renderers=renderers,
                    model=correction_model,
                    condition=_condition_vector(config, (0.0, 0.0), grasp_mode),
                    device=device,
                    delta_scale=config.correction_delta_scale,
                    return_readiness=True,
                )
                best_close_readiness = max(best_close_readiness, float(close_readiness))
                target = np.clip(_current_qpos(env) + delta, env.action_space.low, env.action_space.high)
                target[-1] = _open_gripper_qpos(env)
                info = _step_towards(env, target, gain=0.55, gripper=_open_gripper_qpos(env))
                _record(
                    records,
                    step=global_step,
                    phase="visual_correct",
                    info=info,
                    attempt=attempt,
                    close_readiness=close_readiness,
                )
                if frame_callback is not None:
                    frame_callback(global_step, f"{grasp_mode}_attempt{attempt + 1}_visual_correct", target, info)
                global_step += 1
                delta_norm = float(np.linalg.norm(delta[:5]))
                if delta_norm < config.correction_convergence_tol:
                    correction_settled += 1
                else:
                    correction_settled = 0
                if close_readiness >= close_threshold:
                    close_ready_settled += 1
                else:
                    close_ready_settled = 0
                if (
                    _correction_index + 1 >= config.correction_min_steps
                    and correction_settled >= config.correction_patience
                ):
                    break
                if (
                    _correction_index + 1 >= config.close_readiness_min_steps
                    and close_ready_settled >= config.close_readiness_patience
                ):
                    break
        q_close = np.clip(_current_qpos(env) + offset * 0.35, env.action_space.low, env.action_space.high)
        q_close[-1] = float(env.action_space.low[-1])
        if (
            config.skip_close_if_not_ready
            and correction_model is not None
            and best_close_readiness < close_threshold
            and attempt < total_attempts - 1
        ):
            _record(
                records,
                step=global_step,
                phase="skip_close_not_ready",
                info=info,
                attempt=attempt,
                close_readiness=best_close_readiness,
            )
            global_step += 1
            open_qpos = _current_qpos(env)
            open_qpos[-1] = _open_gripper_qpos(env)
            retreat = open_qpos.copy()
            retreat[1] = np.clip(retreat[1] - 0.18, env.action_space.low[1], env.action_space.high[1])
            retreat[2] = np.clip(retreat[2] + 0.14, env.action_space.low[2], env.action_space.high[2])
            for _backoff_index in range(config.retry_backoff_steps):
                info = _step_towards(env, retreat, gain=0.35, gripper=_open_gripper_qpos(env))
                _record(records, step=global_step, phase="backoff", info=info, attempt=attempt)
                if frame_callback is not None:
                    frame_callback(global_step, f"{grasp_mode}_attempt{attempt + 1}_backoff_not_ready", retreat, info)
                global_step += 1
            continue
        grasped_steps = 0
        for _close_index in range(close_steps):
            close_action = q_close
            close_readiness = None
            if (
                correction_model is not None
                and config.close_visual_servo_interval > 0
                and _close_index % config.close_visual_servo_interval == 0
            ):
                delta, close_readiness = _predict_correction_delta(
                    env=env,
                    renderers=renderers,
                    model=correction_model,
                    condition=_condition_vector(config, (0.0, 0.0), grasp_mode),
                    device=device,
                    delta_scale=config.correction_delta_scale,
                    return_readiness=True,
                )
                close_action = np.clip(
                    _current_qpos(env) + float(close_gain) * delta,
                    env.action_space.low,
                    env.action_space.high,
                )
                close_action[-1] = q_close[-1]
            _obs, _reward, _terminated, _truncated, info = env.step(close_action)
            _record(
                records,
                step=global_step,
                phase="close_visual_servo" if close_readiness is not None else "close",
                info=info,
                attempt=attempt,
                close_readiness=close_readiness,
            )
            if frame_callback is not None:
                frame_callback(global_step, f"{grasp_mode}_attempt{attempt + 1}_close", close_action, info)
            global_step += 1
            if bool(info.get("is_grasped", False)):
                grasped_steps += 1
            else:
                grasped_steps = 0
            if grasped_steps >= config.close_break_on_grasp_steps:
                break
        if config.lift_requires_grasp and not bool(info.get("is_grasped", False)):
            if attempt < total_attempts - 1:
                open_qpos = _current_qpos(env)
                open_qpos[-1] = _open_gripper_qpos(env)
                retreat = open_qpos.copy()
                retreat[1] = np.clip(retreat[1] - 0.18, env.action_space.low[1], env.action_space.high[1])
                retreat[2] = np.clip(retreat[2] + 0.14, env.action_space.low[2], env.action_space.high[2])
                for _backoff_index in range(config.retry_backoff_steps):
                    info = _step_towards(env, retreat, gain=0.35, gripper=_open_gripper_qpos(env))
                    _record(records, step=global_step, phase="backoff", info=info, attempt=attempt)
                    if frame_callback is not None:
                        frame_callback(global_step, f"{grasp_mode}_attempt{attempt + 1}_backoff", retreat, info)
                    global_step += 1
                continue
            break
        for _lift_index in range(lift_steps):
            action = _predict_servo_target(
                env=env,
                renderers=renderers,
                model=model,
                phase=_condition_vector(config, (0.0, 1.0), grasp_mode),
                device=device,
                target_mode=config.servo_target_mode,
                delta_scale=config.lift_delta_scale,
            )
            action = np.clip(action + offset * 0.20, env.action_space.low, env.action_space.high)
            action[-1] = q_close[-1]
            info = _step_towards(env, action, gain=config.lift_gain, gripper=q_close[-1])
            _record(records, step=global_step, phase="lift", info=info, attempt=attempt)
            if bool(info.get("success", False)) and success_step is None:
                success_step = global_step + 1
            if frame_callback is not None:
                frame_callback(global_step, f"{grasp_mode}_attempt{attempt + 1}_lift", action, info)
            global_step += 1
            if bool(info.get("success", False)):
                break
        if bool(info.get("success", False)):
            break
        if attempt < total_attempts - 1:
            open_qpos = _current_qpos(env)
            open_qpos[-1] = _open_gripper_qpos(env)
            retreat = open_qpos.copy()
            retreat[1] = np.clip(retreat[1] - 0.18, env.action_space.low[1], env.action_space.high[1])
            retreat[2] = np.clip(retreat[2] + 0.14, env.action_space.low[2], env.action_space.high[2])
            for _backoff_index in range(config.retry_backoff_steps):
                info = _step_towards(env, retreat, gain=0.35, gripper=_open_gripper_qpos(env))
                _record(records, step=global_step, phase="backoff", info=info, attempt=attempt)
                if frame_callback is not None:
                    frame_callback(global_step, f"{grasp_mode}_attempt{attempt + 1}_backoff", retreat, info)
                global_step += 1
    return {
        "success": bool(info.get("success", False)),
        "steps_to_success": success_step,
        "final_is_grasped": float(info.get("is_grasped", 0.0)),
        "final_lift_height": float(info.get("lift_height", 0.0)),
        "final_tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        "records": records,
    }


def render_policy_rollout(
    *,
    checkpoint_path: Path,
    config: WristEgoServoConfig,
    correction_checkpoint_path: Path | None = None,
    selector_checkpoint_path: Path | None = None,
    seed: int,
    output_dir: Path,
    fps: int,
    device: str,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import mujoco

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[visual-servo] render rollout seed={seed} output_dir={output_dir}", flush=True)
    model, _metadata = load_checkpoint(checkpoint_path)
    correction_model = load_correction_checkpoint(correction_checkpoint_path)[0] if correction_checkpoint_path else None
    selector_model = load_selector_checkpoint(selector_checkpoint_path)[0] if selector_checkpoint_path else None
    model = model.to(device).eval()
    if correction_model is not None:
        correction_model = correction_model.to(device).eval()
    if selector_model is not None:
        selector_model = selector_model.to(device).eval()
    env = make_high_contrast_picklift_env()
    renderers = {
        "wrist_cam": mujoco.Renderer(env.unwrapped.model, height=360, width=480),
        "egocentric_cam": mujoco.Renderer(env.unwrapped.model, height=360, width=480),
        "scene_3d": mujoco.Renderer(env.unwrapped.model, height=360, width=480),
        "wrist_input": mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width),
        "ego_input": mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width),
    }
    frames = []
    try:
        env.reset(seed=seed)
        visible, search_steps = sweep_until_visible(
            env,
            {"wrist_cam": renderers["wrist_input"], "egocentric_cam": renderers["ego_input"]},
            max_sweeps=config.max_sweeps,
        )
        if not visible:
            result = {
                "success": False,
                "steps_to_success": None,
                "final_is_grasped": 0.0,
                "final_lift_height": 0.0,
                "final_tcp_to_obj_dist": float("nan"),
                "records": [],
                "search_steps": search_steps,
                "dropped": True,
            }
        else:
            def frame_callback(step: int, phase: str, action: np.ndarray, info: dict[str, Any]) -> None:
                views = _render_debug_views(env, renderers)
                frames.append(
                    compose_wrist_ego_frame(
                        views=views,
                        step=step,
                        action=action,
                        info=info,
                        mode=f"visual_servo_{phase}",
                    )
                )

            result = run_policy_episode(
                env=env,
                renderers={"wrist_cam": renderers["wrist_input"], "egocentric_cam": renderers["ego_input"]},
                model=model,
                correction_model=correction_model,
                selector_model=selector_model,
                config=config,
                device=device,
                frame_callback=frame_callback,
            )
            result["search_steps"] = search_steps
            result["dropped"] = False
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()
    if not frames:
        frames = [np.zeros((832, 960, 3), dtype=np.uint8)]
    gif_path = output_dir / "wrist_ego_visual_servo_rollout.gif"
    mp4_path = output_dir / "wrist_ego_visual_servo_rollout.mp4"
    imageio.mimsave(gif_path, frames, duration=1.0 / max(1, fps))
    mp4_error = ""
    try:
        imageio.mimsave(mp4_path, frames, fps=fps, macro_block_size=8)
    except Exception as exc:  # pragma: no cover
        mp4_error = str(exc)
        mp4_path = Path("")
    manifest = {
        "operation": "render_wrist_ego_visual_servo_rollout",
        "seed": seed,
        "runtime_inputs": ["wrist_cam", "egocentric_cam", "joint_positions", "phase", "grasp_mode"],
        "runtime_excludes": ["top_down", "camera_calibration", "object_pose", "mujoco_jacobian"],
        "runtime_strategy": _runtime_strategy(config),
        "device": device,
        **{key: value for key, value in result.items() if key != "records"},
        "records": result.get("records", []),
        "gif_path": str(gif_path),
        "mp4_path": str(mp4_path) if mp4_path else "",
        "mp4_error": mp4_error,
    }
    manifest_path = output_dir / "wrist_ego_visual_servo_rollout_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[visual-servo] render complete success={manifest['success']} "
        f"mp4={manifest['mp4_path']} gif={manifest['gif_path']}",
        flush=True,
    )
    return manifest


def _render_debug_views(env: Any, renderers: dict[str, Any]) -> dict[str, np.ndarray]:
    views = {}
    for name in ("wrist_cam", "egocentric_cam", "scene_3d"):
        renderer = renderers[name]
        if name == "scene_3d":
            renderer.update_scene(env.unwrapped.data)
        else:
            renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, name))
        views[name] = postprocess_camera_frame(name, renderer.render())
    return views


def render_wrist_ego(env: Any, renderers: dict[str, Any]) -> np.ndarray:
    frames = []
    for camera_name in ("wrist_cam", "egocentric_cam"):
        renderer = renderers[camera_name]
        renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, camera_name))
        pixels = postprocess_camera_frame(camera_name, renderer.render())
        frames.append(pixels.transpose(2, 0, 1))
    return np.concatenate(frames, axis=0).astype(np.uint8)


def _visual_auxiliary_label(image_chw: np.ndarray) -> np.ndarray:
    channels = np.asarray(image_chw, dtype=np.uint8)
    labels = []
    for start in (0, 3):
        rgb = channels[start : start + 3].transpose(1, 2, 0)
        detection = detect_colored_object(rgb)
        if detection is None:
            labels.extend([-1.0, 0.0, 0.0])
            continue
        height, width = rgb.shape[:2]
        u, v = detection["centroid"]
        labels.extend(
            [
                1.0,
                float(2.0 * (u / max(1.0, width - 1.0)) - 1.0),
                float(2.0 * (v / max(1.0, height - 1.0)) - 1.0),
            ]
        )
    return np.asarray(labels, dtype=np.float32)


def _auxiliary_visual_loss(torch: Any, pred: Any, target: Any) -> Any:
    visible_loss = torch.nn.functional.smooth_l1_loss(pred[:, [0, 3]], target[:, [0, 3]], beta=0.05)
    coord_loss = pred.new_tensor(0.0)
    count = 0
    for offset in (0, 3):
        mask = target[:, offset] > 0.0
        if bool(mask.any()):
            coord_loss = coord_loss + torch.nn.functional.smooth_l1_loss(
                pred[mask, offset + 1 : offset + 3],
                target[mask, offset + 1 : offset + 3],
                beta=0.05,
            )
            count += 1
    if count:
        coord_loss = coord_loss / float(count)
    return visible_loss + coord_loss


def make_high_contrast_picklift_env() -> Any:
    from so101_nexus_core.config import PickConfig
    from so101_nexus_core.objects import CubeObject
    from so101_nexus_mujoco.pick_env import PickLiftEnv

    objects = [
        CubeObject(half_size=half_size, mass=0.01, color=color)
        for half_size in (0.0125, 0.015, 0.0175)
        for color in ("red", "blue", "green")
    ]
    return PickLiftEnv(config=PickConfig(objects=objects), render_mode=None)


def _high_contrast_object_description() -> list[dict[str, Any]]:
    return [
        item
        for item in _object_set_description()
        if item["color"] in {"red", "blue", "green"} and item["half_size"] in {0.0125, 0.015, 0.0175}
    ]


def _make_policy_renderers(env: Any, config: WristEgoServoConfig) -> dict[str, Any]:
    import mujoco

    return {
        "wrist_cam": mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width),
        "egocentric_cam": mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width),
        "top_down": mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width),
    }


def _make_teacher_renderers(env: Any, config: WristEgoServoConfig) -> dict[str, Any]:
    import mujoco

    return {
        name: mujoco.Renderer(env.unwrapped.model, height=config.height, width=config.width)
        for name in config.teacher_camera_names
    }


def object_visible_to_teacher(env: Any, renderers: dict[str, Any], *, config: WristEgoServoConfig) -> bool:
    for camera_name in config.teacher_camera_names:
        renderer = renderers[camera_name]
        if camera_name == "scene_3d":
            renderer.update_scene(env.unwrapped.data)
        else:
            renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, camera_name))
        pixels = postprocess_camera_frame(camera_name, renderer.render())
        if detect_colored_object(pixels) is not None:
            return True
    return False


def make_teacher_targets(env: Any) -> list[dict[str, Any]]:
    snapshot = _snapshot_sim_state(env)
    candidates = _grasp_candidate_specs(env)
    successes_by_mode: dict[str, tuple[float, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for spec in candidates:
        _restore_sim_state(env, snapshot)
        try:
            q_open = _solve_pregrasp_qpos_variant(env, spec)
            q_close = q_open.copy()
            q_close[-1] = float(env.action_space.low[-1])
            q_lift = q_close.copy()
            info: dict[str, Any] = {}
            success_step = None
            for step in range(180):
                if step < 58:
                    action = q_open
                elif step < 118:
                    action = q_close
                else:
                    action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=float)
                    action[-1] = q_close[-1]
                    q_lift = np.clip(action, env.action_space.low, env.action_space.high)
                _obs, _reward, terminated, truncated, info = env.step(action)
                if bool(info.get("success", False)) and success_step is None:
                    success_step = step + 1
                if terminated or truncated:
                    break
            if bool(info.get("success", False)):
                score = (
                    float(info.get("lift_height", 0.0)) * 8.0
                    - float(info.get("tcp_to_obj_dist", 1.0))
                    - 0.001 * float(success_step or 180)
                    - 0.0005 * float(spec["candidate_index"])
                )
                meta = {
                    "mode": str(spec["grasp_mode"]),
                    "candidate_mode": str(spec["mode"]),
                    "axis": [float(value) for value in spec["axis"]],
                    "gap": float(spec["gap"]),
                    "z_offset": float(spec["z_offset"]),
                    "success_step": success_step,
                    "score": score,
                    "candidate_index": int(spec["candidate_index"]),
                    "candidate_attempts": len(candidates),
                }
                grasp_mode = str(spec["grasp_mode"])
                current = successes_by_mode.get(grasp_mode)
                candidate = (score, q_open.astype(float), q_lift.astype(float), meta)
                if current is None or score > current[0]:
                    successes_by_mode[grasp_mode] = candidate
        except Exception:
            continue
    _restore_sim_state(env, snapshot)
    rows = []
    for grasp_mode in _grasp_mode_order():
        success = successes_by_mode.get(grasp_mode)
        if success is None:
            continue
        _score, q_open, q_lift, meta = success
        meta["mode_successes"] = int(grasp_mode in successes_by_mode)
        rows.append({"q_open": q_open, "q_lift": q_lift, "meta": meta})
    return rows


def _grasp_candidate_specs(env: Any) -> list[dict[str, Any]]:
    axes = [
        ("diagonal", "diag_front", np.asarray([1.0, -1.0, 0.0], dtype=float)),
        ("front", "front_back", np.asarray([1.0, 0.0, 0.0], dtype=float)),
        ("front", "left_right", np.asarray([0.0, 1.0, 0.0], dtype=float)),
        ("diagonal", "diag_back", np.asarray([1.0, 1.0, 0.0], dtype=float)),
    ]
    specs = []
    index = 0
    for grasp_mode, name, axis in axes:
        axis = axis / max(1e-6, float(np.linalg.norm(axis)))
        for gap, z_offset, height_label in (
            (0.055, 0.006, "side"),
            (0.075, 0.012, "high"),
            (0.095, 0.020, "overhead_biased"),
        ):
            mode = "overhead" if height_label == "overhead_biased" else grasp_mode
            specs.append(
                {
                    "candidate_index": index,
                    "mode": f"{name}_{height_label}",
                    "grasp_mode": mode,
                    "axis": axis,
                    "gap": gap,
                    "z_offset": z_offset,
                    "open_value": _open_gripper_qpos(env),
                }
            )
            index += 1
    return specs


def _solve_pregrasp_qpos_variant(env: Any, spec: dict[str, Any]) -> np.ndarray:
    import mujoco
    from scipy.optimize import least_squares

    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data
    joint_addrs = [model.jnt_qposadr[jid] for jid in unwrapped._joint_ids]
    low = np.asarray(env.action_space.low, dtype=float)
    high = np.asarray(env.action_space.high, dtype=float)
    static_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
    moving_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
    obj_pos = unwrapped._get_target_pose()[:3].copy()
    q_seed = np.asarray([data.qpos[addr] for addr in joint_addrs], dtype=float)
    axis = np.asarray(spec["axis"], dtype=float)
    axis = axis / max(1e-6, float(np.linalg.norm(axis)))
    gap = float(spec["gap"])
    z_offset = float(spec["z_offset"])
    open_value = float(spec["open_value"])
    desired_static = obj_pos - axis * gap * 0.5 + np.asarray([0.0, 0.0, z_offset])
    desired_moving = obj_pos + axis * gap * 0.5 + np.asarray([0.0, 0.0, z_offset])

    def set_qpos(qpos: np.ndarray) -> None:
        for addr, value in zip(joint_addrs, qpos):
            data.qpos[addr] = value
        data.ctrl[unwrapped._actuator_ids] = np.clip(qpos, low, high)
        mujoco.mj_forward(model, data)

    def residual(arm_qpos: np.ndarray) -> np.ndarray:
        qpos = np.concatenate([arm_qpos, np.asarray([open_value])])
        set_qpos(qpos)
        static_pos = data.geom_xpos[static_pad]
        moving_pos = data.geom_xpos[moving_pad]
        center = 0.5 * (static_pos + moving_pos)
        desired_center = obj_pos + np.asarray([0.0, 0.0, z_offset])
        return np.concatenate(
            [
                (static_pos - desired_static) * 16.0,
                (moving_pos - desired_moving) * 16.0,
                (center - desired_center) * 10.0,
                np.maximum(0.0, z_offset - static_pos[2:3]) * 18.0,
                np.maximum(0.0, z_offset - moving_pos[2:3]) * 18.0,
                (arm_qpos - q_seed[:5]) * 0.035,
            ]
        )

    starts = [
        q_seed[:5],
        np.asarray([-0.5, 0.4, 0.1, 0.5, -1.3]),
        np.asarray([0.0, 0.55, -0.25, 0.85, 1.2]),
        np.asarray([0.6, 0.2, 0.2, 0.6, -1.0]),
        np.asarray([-0.8, 0.2, 0.2, 0.6, -1.0]),
        np.asarray([0.0, -0.15, 0.85, -0.75, 0.0]),
    ]
    best = None
    for start in starts:
        result = least_squares(
            residual,
            np.clip(start, low[:5], high[:5]),
            bounds=(low[:5], high[:5]),
            max_nfev=160,
        )
        cost = float(np.linalg.norm(residual(result.x)))
        candidate = np.concatenate([result.x, np.asarray([open_value])])
        if best is None or cost < best[0]:
            best = (cost, candidate)
    assert best is not None
    return np.clip(best[1], low, high)


def _snapshot_sim_state(env: Any) -> dict[str, np.ndarray]:
    data = env.unwrapped.data
    return {
        "qpos": np.asarray(data.qpos, dtype=float).copy(),
        "qvel": np.asarray(data.qvel, dtype=float).copy(),
        "ctrl": np.asarray(data.ctrl, dtype=float).copy(),
    }


def _restore_sim_state(env: Any, snapshot: dict[str, np.ndarray]) -> None:
    import mujoco

    data = env.unwrapped.data
    data.qpos[:] = snapshot["qpos"]
    data.qvel[:] = snapshot["qvel"]
    data.ctrl[:] = snapshot["ctrl"]
    mujoco.mj_forward(env.unwrapped.model, data)


def _close_probe_is_grasped(env: Any, *, open_steps: int, close_steps: int) -> bool:
    snapshot = _snapshot_sim_state(env)
    try:
        action = _current_qpos(env)
        action[-1] = _open_gripper_qpos(env)
        for _ in range(max(0, int(open_steps))):
            _obs, _reward, _terminated, _truncated, _info = env.step(action)
        action[-1] = float(env.action_space.low[-1])
        grasped_steps = 0
        for _ in range(max(1, int(close_steps))):
            _obs, _reward, _terminated, _truncated, info = env.step(action)
            if bool(info.get("is_grasped", False)):
                grasped_steps += 1
            else:
                grasped_steps = 0
        return grasped_steps > 0
    finally:
        _restore_sim_state(env, snapshot)


def _current_qpos(env: Any) -> np.ndarray:
    return np.asarray(env.unwrapped.data.ctrl[env.unwrapped._actuator_ids], dtype=float).copy()


def _set_qpos(env: Any, qpos: np.ndarray) -> None:
    import mujoco

    qpos = np.clip(np.asarray(qpos, dtype=float), env.action_space.low, env.action_space.high)
    for joint_id, value in zip(env.unwrapped._joint_ids, qpos):
        addr = env.unwrapped.model.jnt_qposadr[joint_id]
        env.unwrapped.data.qpos[addr] = value
    env.unwrapped.data.ctrl[env.unwrapped._actuator_ids] = qpos
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)


def _normalize_qpos(qpos: np.ndarray, env: Any) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=float)
    high = np.asarray(env.action_space.high, dtype=float)
    return 2.0 * (np.asarray(qpos, dtype=float) - low) / (high - low) - 1.0


def _denormalize_qpos(qpos: np.ndarray, env: Any) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=float)
    high = np.asarray(env.action_space.high, dtype=float)
    return low + (np.asarray(qpos, dtype=float) + 1.0) * 0.5 * (high - low)


def _predict_servo_target(
    *,
    env: Any,
    renderers: dict[str, Any],
    model: Any,
    phase: tuple[float, float],
    device: str,
    target_mode: str,
    delta_scale: float,
) -> np.ndarray:
    import torch

    image = torch.as_tensor(render_wrist_ego(env, renderers)).unsqueeze(0).to(device)
    qpos = torch.as_tensor(_normalize_qpos(_current_qpos(env), env)).unsqueeze(0).float().to(device)
    phase_tensor = torch.as_tensor([phase], dtype=torch.float32).to(device)
    with torch.no_grad():
        normalized = model(image, qpos, phase_tensor).cpu().numpy()[0]
    if target_mode != "delta":
        return _denormalize_qpos(normalized, env)
    delta = np.asarray(normalized, dtype=float) * _servo_delta_scale_vector(env, delta_scale)
    delta[-1] = 0.0
    return np.clip(_current_qpos(env) + delta, env.action_space.low, env.action_space.high)


def _predict_correction_delta(
    *,
    env: Any,
    renderers: dict[str, Any],
    model: Any,
    condition: tuple[float, ...],
    device: str,
    delta_scale: float,
    return_readiness: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    import torch

    image = torch.as_tensor(render_wrist_ego(env, renderers)).unsqueeze(0).to(device)
    qpos = torch.as_tensor(_normalize_qpos(_current_qpos(env), env)).unsqueeze(0).float().to(device)
    condition_tensor = torch.as_tensor([condition], dtype=torch.float32).to(device)
    with torch.no_grad():
        if return_readiness:
            normalized_tensor, readiness_logit = model(image, qpos, condition_tensor, return_readiness=True)
            normalized = normalized_tensor.cpu().numpy()[0]
            readiness = float(torch.sigmoid(readiness_logit).cpu().numpy()[0, 0])
        else:
            normalized = model(image, qpos, condition_tensor).cpu().numpy()[0]
            readiness = 0.0
    delta = np.asarray(normalized, dtype=float) * _correction_delta_scale_vector(env, delta_scale)
    delta[-1] = 0.0
    if return_readiness:
        return delta, readiness
    return delta


def _predict_grasp_mode_order(
    *,
    env: Any,
    renderers: dict[str, Any],
    model: Any | None,
    config: WristEgoServoConfig,
    device: str,
) -> tuple[str, ...]:
    if model is None:
        return tuple(config.grasp_mode_names)
    import torch

    image = torch.as_tensor(render_wrist_ego(env, renderers)).unsqueeze(0).to(device)
    qpos = torch.as_tensor(_normalize_qpos(_current_qpos(env), env)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(image, qpos).cpu().numpy()[0]
    order = [int(index) for index in np.argsort(-np.asarray(logits, dtype=float))]
    names = tuple(config.grasp_mode_names[index] for index in order)
    if len(names) == len(config.grasp_mode_names):
        return names
    return tuple(config.grasp_mode_names)


def _correction_delta_scale_vector(env: Any, delta_scale: float) -> np.ndarray:
    scale = np.full_like(np.asarray(env.action_space.low, dtype=float), float(delta_scale))
    scale[-1] = 1.0
    return scale


def _servo_delta_scale_vector(env: Any, delta_scale: float) -> np.ndarray:
    scale = np.full_like(np.asarray(env.action_space.low, dtype=float), float(delta_scale))
    scale[-1] = 1.0
    return scale


def _step_towards(env: Any, target: np.ndarray, *, gain: float, gripper: float) -> dict[str, Any]:
    qpos = _current_qpos(env)
    action = np.clip(qpos + gain * (np.asarray(target, dtype=float) - qpos), env.action_space.low, env.action_space.high)
    action[-1] = gripper
    _obs, _reward, _terminated, _truncated, info = env.step(action)
    return info


def _arm_target_error(env: Any, target: np.ndarray) -> float:
    qpos = _current_qpos(env)
    return float(np.linalg.norm(np.asarray(target, dtype=float)[:5] - qpos[:5]))


def _open_gripper_qpos(env: Any) -> float:
    return float(env.action_space.high[-1])


def _grasp_mode_order() -> tuple[str, str, str]:
    return ("front", "diagonal", "overhead")


def _condition_vector(
    config: WristEgoServoConfig,
    phase: tuple[float, float],
    grasp_mode: str,
) -> tuple[float, ...]:
    modes = tuple(config.grasp_mode_names)
    one_hot = [0.0] * len(modes)
    if grasp_mode in modes:
        one_hot[modes.index(grasp_mode)] = 1.0
    return tuple(float(value) for value in phase) + tuple(one_hot)


def _scheduled_value(schedule: tuple[float, ...], attempt: int, default: float) -> float:
    if not schedule:
        return float(default)
    return float(schedule[min(attempt, len(schedule) - 1)])


def _runtime_strategy(config: WristEgoServoConfig) -> str:
    first_approach = (
        f"closed-loop first approach every {config.approach_replan_interval} steps"
        if config.approach_replan_interval > 0
        else "single-shot first approach"
    )
    retry_approach = (
        f"retry approach refresh every {config.retry_approach_replan_interval} steps"
        if config.retry_approach_replan_interval > 0
        else "single-shot retry approach"
    )
    close = (
        f"close visual-servo every {config.close_visual_servo_interval} steps"
        if config.close_visual_servo_interval > 0
        else "fixed close action"
    )
    approach = f"{first_approach}, {retry_approach}, {close}"
    if config.servo_target_mode == "delta":
        return f"{approach}, image-conditioned coarse delta, convergence early-stop, close-readiness gated visual correction"
    return f"{approach}, absolute coarse pregrasp target, convergence early-stop, close-readiness gated visual correction"


def _retry_grasp_offset(env: Any, attempt: int) -> np.ndarray:
    return np.zeros_like(np.asarray(env.action_space.low, dtype=float))


def _record(
    records: list[dict[str, Any]],
    *,
    step: int,
    phase: str,
    info: dict[str, Any],
    attempt: int | None = None,
    close_readiness: float | None = None,
) -> None:
    row = {
        "step": step,
        "phase": phase,
        "attempt": attempt,
        "success": bool(info.get("success", False)),
        "is_grasped": float(info.get("is_grasped", 0.0)),
        "lift_height": float(info.get("lift_height", 0.0)),
        "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
    }
    if close_readiness is not None:
        row["close_readiness"] = float(close_readiness)
    records.append(row)


def _drop_row(*, episode: int, seed: int, search_steps: int) -> dict[str, Any]:
    return {
        "episode": episode,
        "seed": seed,
        "search_steps": search_steps,
        "success": False,
        "steps_to_success": None,
        "final_is_grasped": 0.0,
        "final_lift_height": 0.0,
        "final_tcp_to_obj_dist": float("nan"),
        "dropped": True,
    }


def save_checkpoint(path: Path, model: Any, metadata: dict[str, Any]) -> None:
    import torch

    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: Path) -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu")
    metadata = dict(payload["metadata"])
    config = WristEgoServoConfig(**metadata["config"])
    model = WristEgoVisualServoPolicy(config=config)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, metadata


def load_correction_checkpoint(path: Path | None) -> tuple[Any, dict[str, Any]]:
    import torch

    if path is None:
        raise ValueError("correction checkpoint path is required")
    payload = torch.load(path, map_location="cpu")
    metadata = dict(payload["metadata"])
    config = WristEgoServoConfig(**metadata["config"])
    model = WristEgoVisualCorrectionPolicy(config=config)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, metadata


def load_selector_checkpoint(path: Path | None) -> tuple[Any, dict[str, Any]]:
    import torch

    if path is None:
        raise ValueError("selector checkpoint path is required")
    payload = torch.load(path, map_location="cpu")
    metadata = dict(payload["metadata"])
    config = WristEgoServoConfig(**metadata["config"])
    model = WristEgoGraspModeSelector(config=config)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, metadata


if __name__ == "__main__":
    main()
