from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import ast
import argparse
from pathlib import Path
from unittest import TestCase, mock

import numpy as np

from physical_ai_agent.so101_smolvla_pipeline import (
    SO101AugmentationContract,
    SO101DatasetManifest,
    SO101TrainingSchedule,
    SmolVLASO101Contract,
    detect_overfit_stop,
    should_run_closed_loop,
    validate_smolvla_train_config,
)
from physical_ai_agent.so101_resolution_contract import (
    require_dataset_config_256,
    require_lerobot_dataset_256,
)
from physical_ai_agent.sim.so101_camera_input import EGOCENTRIC_CAMERA1_POSE


def _ensure_scripts_on_path() -> None:
    scripts_path = str(Path("scripts").resolve())
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)


def _write_lerobot_resolution_info(root: Path, image_shape: list[int]) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    features = {
        key: {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channels"]}
        for key in (
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        )
    }
    (meta / "info.json").write_text(json.dumps({"features": features}), encoding="utf-8")


class _PickleSafeFakeLeRobotDataset:
    def __init__(self, name: str, length: int) -> None:
        from types import SimpleNamespace

        self.name = name
        self.repo_id = name
        self.root = f"/tmp/{name}"
        self.meta = SimpleNamespace(
            fps=30,
            features={"action": {"shape": [1]}},
            stats={
                "action": {
                    "min": [0.0],
                    "max": [float(length)],
                    "mean": [float(length) / 2.0],
                    "std": [1.0],
                    "count": [length],
                },
            },
        )
        self.items = list(range(length))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, int | str]:
        return {"source": self.name, "value": self.items[index]}


class SO101SmolVLAPipelineTest(TestCase):
    def test_so101_contract_uses_egocentric_camera1_and_wrist_camera2(self) -> None:
        contract = SmolVLASO101Contract()

        self.assertEqual(contract.runtime_camera_mapping["observation.images.camera1"], "egocentric_cam")
        self.assertEqual(contract.runtime_camera_mapping["observation.images.camera2"], "wrist_cam")
        self.assertEqual(contract.runtime_camera_mapping["observation.images.camera3"], "wrist_cam duplicate")
        self.assertNotIn("top_down", contract.runtime_camera_mapping.values())

    def test_default_augmentation_contract_matches_training_configs(self) -> None:
        contract = SO101AugmentationContract()

        self.assertEqual(contract.validate(), [])
        self.assertEqual(contract.state_jitter_std, 0.003)
        self.assertEqual(contract.state_dropout_prob, 0.02)
        self.assertEqual(contract.image_camera_dropout_prob, 0.0)
        self.assertEqual(contract.image_patch_dropout_prob, 0.0)
        self.assertEqual(contract.image_patch_mask_ratio, 0.15)
        self.assertTrue(contract.image_color_jitter)
        self.assertTrue(contract.image_sharpness_jitter)
        self.assertEqual(contract.image_affine_degrees, 5.0)
        self.assertEqual(contract.image_affine_translate, 0.05)
        self.assertTrue(contract.run_after_batch_to_device)
        self.assertIn("cuda", contract.prefer_device_backends)
        self.assertIn("mps", contract.prefer_device_backends)

    def test_train_config_matches_smolvla_so101_contract(self) -> None:
        contract = SmolVLASO101Contract()
        config = {
            "policy": {
                "input_features": {
                    "observation.state": {"shape": [6]},
                    "observation.images.camera1": {"shape": [3, 256, 256]},
                    "observation.images.camera2": {"shape": [3, 256, 256]},
                    "observation.images.camera3": {"shape": [3, 256, 256]},
                },
                "output_features": {"action": {"shape": [6]}},
                "resize_imgs_with_padding": [512, 512],
                "chunk_size": 50,
                "n_action_steps": 50,
                "num_steps": 10,
                "tokenizer_max_length": 48,
                "pretrained_path": "lerobot/smolvla_base",
            }
        }

        self.assertEqual(validate_smolvla_train_config(config, contract), [])

        config["policy"]["input_features"]["observation.images.camera1"]["shape"] = [3, 96, 96]
        self.assertIn(
            "observation.images.camera1.shape [3, 96, 96] != [3, 256, 256]",
            validate_smolvla_train_config(config, contract),
        )

    def test_lerobot_dataset_resolution_contract_rejects_96px_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_lerobot_resolution_info(root, [96, 96, 3])

            with self.assertRaisesRegex(ValueError, "expected \\[256, 256, 3\\]"):
                require_lerobot_dataset_256(root, context="unit")

    def test_dataset_config_resolution_contract_checks_train_val_and_loop_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train"
            val = root / "val"
            loop = root / "loop"
            _write_lerobot_resolution_info(train, [256, 256, 3])
            _write_lerobot_resolution_info(val, [256, 256, 3])
            _write_lerobot_resolution_info(loop, [96, 96, 3])
            config = {
                "train_dataset": {"root": str(train)},
                "validation_dataset": {"root": str(val)},
                "closed_loop": {"test_cases": [{"start_dataset": {"root": str(loop)}}]},
            }

            with self.assertRaisesRegex(ValueError, "closed_loop.test_cases\\[0\\].start_dataset"):
                require_dataset_config_256(config, repo_root=Path.cwd(), context="unit")

    def test_dataset_manifest_requires_doubled_recovery_data_without_sticky_grasp(self) -> None:
        manifest = SO101DatasetManifest(
            dataset_id="physical-ai-agent/so101-pickplace-contact-train100-256-recovery",
            split="train",
            episodes=100,
            frames=23000,
            source_episode_count=50,
            target_expansion_factor=2.0,
            min_frames_per_episode=220,
            max_frames_per_episode=246,
            includes_recovery_or_off_nominal_states=True,
            sticky_grasp_allowed=False,
        )

        self.assertEqual(manifest.validate(), [])

        broken = SO101DatasetManifest(
            dataset_id="broken",
            split="train",
            episodes=80,
            frames=8000,
            source_episode_count=50,
            target_expansion_factor=2.0,
            min_frames_per_episode=220,
            max_frames_per_episode=246,
            includes_recovery_or_off_nominal_states=False,
            sticky_grasp_allowed=True,
        )
        errors = broken.validate()
        self.assertTrue(any("expected at least 100" in error for error in errors))
        self.assertTrue(any("expected at least 17600 from 220 min frames/episode" in error for error in errors))
        self.assertTrue(any("sticky_grasp_allowed" in error for error in errors))
        self.assertTrue(any("recovery/off-nominal" in error for error in errors))

        too_long = SO101DatasetManifest(
            dataset_id="too-long",
            split="train",
            episodes=100,
            frames=25000,
            source_episode_count=50,
            target_expansion_factor=2.0,
            min_frames_per_episode=220,
            max_frames_per_episode=246,
            includes_recovery_or_off_nominal_states=True,
            sticky_grasp_allowed=False,
        )
        self.assertTrue(
            any("expected at most 24600 from 246 max frames/episode" in error for error in too_long.validate())
        )

    def test_closed_loop_best_only_ignores_non_best_checkpoints(self) -> None:
        schedule = SO101TrainingSchedule(closed_loop_policy="best_only")
        rows = [
            {"checkpoint": "001490", "loss": 0.08},
            {"checkpoint": "002682", "loss": 0.14},
        ]

        self.assertTrue(
            should_run_closed_loop(
                schedule=schedule,
                checkpoint="001490",
                validation_rows=rows,
                closed_loop_rows=[],
            )
        )
        self.assertFalse(
            should_run_closed_loop(
                schedule=schedule,
                checkpoint="002682",
                validation_rows=rows,
                closed_loop_rows=[],
            )
        )

    def test_overfit_detector_stops_after_patience(self) -> None:
        rows = [
            {"checkpoint": "000894", "loss": 0.10},
            {"checkpoint": "001490", "loss": 0.08},
            {"checkpoint": "001788", "loss": 0.09},
            {"checkpoint": "002086", "loss": 0.11},
            {"checkpoint": "002682", "loss": 0.14},
        ]

        decision = detect_overfit_stop(rows, patience_checkpoints=3)

        self.assertTrue(decision["should_stop"])
        self.assertEqual(decision["best"]["checkpoint"], "001490")
        self.assertEqual(decision["reason"], "validation_loss_worse_than_best")

    def test_dataset_manifest_cli_validates_committed_targets(self) -> None:
        manifests = [
            Path("configs/so101/smolvla_pickplace_contact_train100_manifest.json"),
            Path("configs/so101/smolvla_pickplace_contact_val24_manifest.json"),
        ]

        for manifest in manifests:
            with self.subTest(manifest=manifest):
                completed = subprocess.run(
                    [sys.executable, "scripts/so101_dataset_manifest.py", "validate", str(manifest)],
                    check=False,
                    text=True,
                    capture_output=True,
                    env={**os.environ, "PYTHONPATH": "src"},
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)
                payload = json.loads(completed.stdout)
                self.assertEqual(payload["validation_errors"], [])

    def test_dataset_manifest_cli_rejects_under_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset_id": "broken",
                        "split": "train",
                        "episodes": 50,
                        "frames": 5000,
                        "source_episode_count": 50,
                        "target_expansion_factor": 2.0,
                        "min_frames_per_episode": 220,
                        "max_frames_per_episode": 240,
                        "includes_recovery_or_off_nominal_states": True,
                        "sticky_grasp_allowed": False,
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, "scripts/so101_dataset_manifest.py", "validate", str(manifest)],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("expected at least 100", completed.stderr)

    def test_teacher_exporter_exposes_recovery_data_flags(self) -> None:
        source = Path("scripts/export_so101_pickplace_teacher_rollouts_lerobot.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        constants = {
            node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        self.assertIn("--recovery-steps", constants)
        self.assertIn("--recovery-joint-std", constants)
        self.assertIn("--reject-preclose-contact", constants)
        self.assertIn("recovery", constants)
        self.assertIn("recovery_or_off_nominal_states", constants)
        self.assertIn("reject_preclose_contact", constants)

    def test_training_launcher_does_not_build_literal_empty_cache_paths(self) -> None:
        _ensure_scripts_on_path()
        import start_so101_training

        config = {
            "train_dataset": {"repo_id": "repo/train", "root": "_workspace/train"},
            "validation_dataset": {"repo_id": "repo/validation", "root": "_workspace/validation"},
            "predecoded_image_cache": {
                "default_root": "_workspace/so101_image_cache",
                "train": {},
                "validation": {},
            },
        }

        commands = start_so101_training._cache_build_commands(
            Path(sys.executable),
            Path.cwd(),
            config,
        )

        self.assertEqual(commands, [])

    def test_training_launcher_builds_explicit_split_image_caches(self) -> None:
        _ensure_scripts_on_path()
        import start_so101_training

        config = {
            "train_dataset": {"repo_id": "repo/train", "root": "_workspace/train"},
            "validation_dataset": {"repo_id": "repo/validation", "root": "_workspace/validation"},
            "predecoded_image_cache": {
                "default_root": "_workspace/so101_image_cache",
                "train": "train_cache",
                "validation": "validation_cache",
            },
        }

        commands = start_so101_training._cache_build_commands(
            Path(sys.executable),
            Path.cwd(),
            config,
        )

        self.assertEqual(len(commands), 2)
        self.assertIn("_workspace/so101_image_cache/train_cache", commands[0])
        self.assertIn("_workspace/so101_image_cache/validation_cache", commands[1])

    def test_pickplace_exporter_uses_egocentric_wrist_student_camera_contract(self) -> None:
        source = Path("scripts/export_so101_pickplace_teacher_rollouts_lerobot.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        constants = {
            node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        frame_function = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_make_pickplace_frame"
        )
        frame_constants = {
            node.value
            for node in ast.walk(frame_function)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        self.assertIn("wrist_cam", frame_constants)
        self.assertIn("egocentric_cam", frame_constants)
        self.assertIn("observation.images.camera1", frame_constants)
        self.assertIn("observation.images.camera2", frame_constants)
        self.assertIn("observation.images.camera3", frame_constants)
        self.assertIn("observation.images.camera1", constants)
        self.assertIn("observation.images.wrist_cam", constants)
        self.assertIn("observation.images.egocentric_cam", constants)
        self.assertIn("wrist_cam duplicate", constants)
        self.assertNotIn("egocentric_cam duplicate", constants)
        self.assertIn('"observation.images.camera1": ego', source)
        self.assertIn('"observation.images.camera2": wrist', source)
        self.assertIn('frame["observation.images.camera3"] = wrist.copy()', source)

    def test_lightning_train_entrypoint_uses_tensorboard_and_lerobot_checkpointing(self) -> None:
        source = Path("scripts/lerobot_train_so101_lightning.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

        self.assertIn("--tensorboard-log-dir", constants)
        self.assertIn("--validation-dataset-root", constants)
        self.assertIn("--validation-interval-steps", constants)
        self.assertIn("--validation-interval-epochs", constants)
        self.assertIn("--validation-every-n-train-steps", constants)
        self.assertIn("--post-checkpoint-loop-command-json", constants)
        self.assertIn("--log-input-images-every-n-steps", constants)
        self.assertIn("--log-input-metadata-every-n-steps", constants)
        self.assertIn("--so101-action-prefix-loss-steps", constants)
        self.assertIn("--so101-action-prefix-loss-weight", constants)
        self.assertIn("--so101-action-chunk-consistency-steps", constants)
        self.assertIn("--so101-action-chunk-consistency-weight", constants)
        self.assertIn("--so101-action-delta-loss-weight", constants)
        self.assertIn("--so101-action-gripper-transition-loss-weight", constants)
        self.assertIn("--so101-action-terminal-loss-steps", constants)
        self.assertIn("--so101-action-terminal-loss-weight", constants)
        self.assertIn("--so101-action-smoothness-loss-weight", constants)
        self.assertIn("--so101-action-smoothness-include-gripper", constants)
        self.assertIn("--so101-valid-mask-loss-weight", constants)
        self.assertIn("--so101-valid-mask-hidden-dim", constants)
        self.assertIn("loss_unweighted", constants)
        self.assertIn("loss_prefix_weight", constants)
        self.assertIn("loss_prefix_steps", constants)
        self.assertIn("action_chunk_consistency_loss", constants)
        self.assertIn("action_chunk_consistency_weight", constants)
        self.assertIn("action_chunk_consistency_steps", constants)
        self.assertIn("action_delta_loss_weight", constants)
        self.assertIn("action_gripper_transition_loss_weight", constants)
        self.assertIn("action_terminal_loss_steps", constants)
        self.assertIn("action_terminal_loss_weight", constants)
        self.assertIn("action_smoothness_loss", constants)
        self.assertIn("action_smoothness_loss_weight", constants)
        self.assertIn("action_smoothness_include_gripper", constants)
        self.assertIn("action_importance_weight_mean", constants)
        self.assertIn("valid_mask_loss", constants)
        self.assertIn("valid_mask_accuracy", constants)
        self.assertIn("action_is_pad_as_termination_proxy", constants)
        self.assertIn("camera1,camera2", constants)
        self.assertIn("input", constants)
        self.assertIn("augmented_input", constants)
        self.assertIn("/input_prompt", constants)
        self.assertIn("/input_motor_state", constants)
        self.assertIn("/input_camera_contract", constants)
        self.assertIn("--training-run-summary-path", constants)
        self.assertIn("training/summary", constants)
        self.assertIn("training/datasets", constants)
        self.assertIn("training/closed_loop", constants)
        self.assertIn("training/augmentation", constants)
        self.assertIn("training/runtime", constants)
        self.assertIn("training/command", constants)
        self.assertIn("observation.state", constants)
        self.assertIn("val/loss", constants)
        self.assertIn("important/train_loss", constants)
        self.assertIn("important/val_loss", constants)
        self.assertIn("val/action_jitter/", constants)
        self.assertIn("path_to_endpoint_ratio_mean", constants)
        self.assertIn("TensorBoardLogger", names)
        self.assertIn("Trainer", names)
        self.assertIn("save_checkpoint", names)
        self.assertIn("save_valid_mask_head", names)
        self.assertIn("augment_batch_on_device", names)
        self.assertIn("_action_chunk_jitter_metrics", names)
        self.assertIn("_tensorboard_image_with_visual_servo_target", names)
        self.assertIn("_tensorboard_image_grid_with_visual_servo_target", names)
        self.assertIn("_draw_visual_servo_target_on_image", names)
        self.assertIn("visual_servo.camera1_visible", constants)
        self.assertIn("visual_servo.camera2_visible", constants)

    def test_lightning_validation_dataloader_returns_dataloader(self) -> None:
        source = Path("scripts/lerobot_train_so101_lightning.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_make_validation_dataloader"
        )
        returns = [node.value for node in ast.walk(function) if isinstance(node, ast.Return)]

        self.assertTrue(
            any(
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "DataLoader"
                for value in returns
            )
        )

    def test_visual_servo_training_uses_label_safe_single_view_augmentation(self) -> None:
        _ensure_scripts_on_path()
        import lerobot_train_so101_lightning as train
        from physical_ai_agent.lerobot_sampling_augmentation import SamplingAugmentationConfig

        config = SamplingAugmentationConfig(
            image_affine_degrees=5.0,
            image_affine_translate=0.05,
            image_patch_mask_ratio=0.15,
            image_color_jitter=True,
            image_sharpness_jitter=True,
            gpu_image_augmentation=True,
            enabled=True,
        )

        safe = train._single_view_augmentation_for_visual_servo_labels(config, enabled=True)

        self.assertEqual(safe.image_affine_degrees, 5.0)
        self.assertEqual(safe.image_affine_translate, 0.05)
        self.assertEqual(safe.image_patch_mask_ratio, 0.0)
        self.assertTrue(safe.image_color_jitter)
        self.assertTrue(safe.image_sharpness_jitter)

    def test_training_scalar_tags_keep_visual_servo_summary_in_main_view(self) -> None:
        _ensure_scripts_on_path()
        import lerobot_train_so101_lightning as train

        self.assertEqual(train._scalar_metric_tag("train", "visual_servo_loss"), "train/visual_servo_loss")
        self.assertEqual(train._scalar_metric_tag("train", "visual_servo_mse"), "train/visual_servo_mse")
        self.assertEqual(train._scalar_metric_tag("train", "visual_servo_rmse"), "train/visual_servo_rmse")
        self.assertEqual(
            train._scalar_metric_tag("train", "visual_servo_camera1_dx_mae"),
            "extra/train/visual_servo_camera1_dx_mae",
        )
        self.assertEqual(
            train._scalar_metric_tag("train", "visual_servo_camera1_rmse"),
            "extra/train/visual_servo_camera1_rmse",
        )
        self.assertEqual(
            train._scalar_metric_tag("train", "losses_after_forward"),
            "extra/train/losses_after_forward",
        )

    def test_training_dashboard_is_dataset_and_closed_loop_only(self) -> None:
        source = Path("scripts/serve_so101_training_dashboard.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        html = "\n".join(constants)

        self.assertIn('data-tab="datasetPanel"', html)
        self.assertIn('data-tab="closedLoopPanel"', html)
        self.assertNotIn('data-tab="trainingPanel">Training', html)

    def test_training_monitor_writes_important_closed_loop_scalar(self) -> None:
        source = Path("scripts/monitor_so101_training_dashboard.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        self.assertIn("success_rate", constants)
        self.assertIn("important/closed_loop_success_rate", constants)

    def test_training_monitor_writes_closed_loop_camera_rollout_videos(self) -> None:
        source = Path("scripts/monitor_so101_training_dashboard.py").read_text(encoding="utf-8")

        self.assertIn("writer.add_video", source)
        self.assertIn("rollout_{camera_name}", source)
        self.assertIn("_closed_loop_frame_label", source)
        self.assertIn('"camera1"', source)
        self.assertIn('"camera2"', source)
        self.assertIn("observation.images.camera1", source)
        self.assertIn("observation.images.camera2", source)

    def test_training_monitor_exports_review_gifs_with_tensorboard_style(self) -> None:
        source = Path("scripts/monitor_so101_training_dashboard.py").read_text(encoding="utf-8")

        self.assertIn("def _export_closed_loop_tensorboard_style_gifs", source)
        self.assertIn("_closed_loop_policy_camera_side_by_side_videos(report)", source)
        self.assertIn("_write_tensorboard_video_gif", source)
        self.assertIn("tensorboard_closed_loop_camera1_camera2_side_by_side", source)

    def test_training_monitor_overlays_episode_and_frame_on_closed_loop_rollout(self) -> None:
        _ensure_scripts_on_path()
        import monitor_so101_training_dashboard as monitor
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "step_0007_egocentric_cam.png"
            Image.new("RGB", (96, 64), color=(255, 255, 255)).save(image_path)
            trace_path = root / "trace.jsonl"
            trace_path.write_text(
                json.dumps(
                    {
                        "episode": 2,
                        "global_step": 7,
                        "prompt": "Close the gripper on the green cube edge and lift.",
                        "image_feature_mapping": {"observation.images.camera1": "egocentric_cam"},
                        "media": {"policy_input_images": {"egocentric_cam": str(image_path)}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = {"episodes": [{"episode": 2, "trace_path": str(trace_path)}]}

            frames = monitor._closed_loop_policy_camera_frames(
                report,
                "observation.images.camera1",
                max_frames=4,
            )
            self.assertEqual(
                frames,
                [(image_path, "ep 002 | frame 007\nprompt: Close the gripper on the green cube edge...")],
            )
            self.assertEqual(
                monitor._short_closed_loop_prompt("  close   the   gripper  "),
                "close the gripper",
            )

            video = monitor._frames_to_tensorboard_video(frames)
            self.assertIsNotNone(video)
            self.assertEqual(tuple(video.shape), (1, 1, 3, 64, 96))
            self.assertLess(int(video[0, 0, :, 0, 0].max()), 20)

    def test_training_monitor_overlays_rollout_prediction_target(self) -> None:
        _ensure_scripts_on_path()
        import monitor_so101_training_dashboard as monitor
        import numpy as np

        image = np.zeros((80, 100, 3), dtype=np.uint8)
        rendered = monitor._overlay_closed_loop_frame_label(
            image,
            "ep 000 | frame 000",
            target={"dx_norm": 0.5, "dy_norm": -0.5, "label": "pred", "color": (0, 220, 255)},
        )

        self.assertGreater(int(rendered[:, :, 1].max()), 200)
        self.assertGreater(int(rendered[:, :, 2].max()), 200)

    def test_training_monitor_rollout_prediction_target_shows_visible_probability(self) -> None:
        _ensure_scripts_on_path()
        import monitor_so101_training_dashboard as monitor

        low = monitor._target_overlay_from_mapping(
            {"dx_norm": 0.0, "dy_norm": 0.0, "visible": True, "visible_prob": 0.2},
            camera="camera1",
            label="pred",
            color=(0, 220, 255),
        )
        high = monitor._target_overlay_from_mapping(
            {"dx_norm": 0.0, "dy_norm": 0.0, "visible": True, "visible_prob": 0.9},
            camera="camera1",
            label="pred",
            color=(0, 220, 255),
        )

        self.assertEqual(low["label"], "pred vis=0.20")
        self.assertEqual(high["label"], "pred vis=0.90")
        self.assertLess(low["color"][1], high["color"][1])
        self.assertLess(low["color"][2], high["color"][2])

    def test_training_monitor_marks_active_servo_camera_and_phase(self) -> None:
        _ensure_scripts_on_path()
        import monitor_so101_training_dashboard as monitor
        import numpy as np

        row = {
            "primitive_id": "align_fixed_jaw_cube_edge",
            "policy_output": {
                "visual_servo_prediction": {
                    "servo_selection": {"servo_camera": "camera2"},
                }
            },
        }

        camera1 = monitor._closed_loop_frame_metadata(row, "observation.images.camera1")
        camera2 = monitor._closed_loop_frame_metadata(row, "observation.images.camera2")

        self.assertFalse(camera1["active"])
        self.assertTrue(camera2["active"])
        self.assertEqual(camera2["phase"], "align_fixed_jaw_cube_edge")

        rendered = monitor._overlay_closed_loop_frame_label(
            np.zeros((80, 100, 3), dtype=np.uint8),
            "ep 000 | frame 000",
            metadata=camera2,
        )
        self.assertGreater(int(rendered[:, :, 2].max()), 200)
        self.assertIn("phase: align_fixed_jaw_cube_edge", monitor._phase_label(camera2))

    def test_training_monitor_marks_only_recorded_inference_frames(self) -> None:
        _ensure_scripts_on_path()
        import monitor_so101_training_dashboard as monitor

        self.assertTrue(
            monitor._is_closed_loop_inference_frame(
                {"policy_output": {"visual_servo_hold": {"inference_frame": True}}},
                1,
                15,
            )
        )
        self.assertFalse(
            monitor._is_closed_loop_inference_frame(
                {"policy_output": {"visual_servo_hold": {"inference_frame": False}}},
                0,
                15,
            )
        )

    def test_training_monitor_reads_reference_gt_target_from_visual_servo_sidecar(self) -> None:
        _ensure_scripts_on_path()
        import monitor_so101_training_dashboard as monitor
        import pandas as pd

        row = pd.Series({"episode_index": 0, "frame_index": 2})
        sidecar = pd.DataFrame(
            [
                {
                    "episode_index": 0,
                    "frame_index": 2,
                    "camera1_visible": True,
                    "camera1_dx_norm": -0.25,
                    "camera1_dy_norm": 0.5,
                }
            ]
        ).set_index(["episode_index", "frame_index"])

        target = monitor._reference_target_overlay(sidecar, row, "camera1")

        self.assertEqual(target["label"], "gt")
        self.assertEqual(target["dx_norm"], -0.25)
        self.assertEqual(target["dy_norm"], 0.5)

    def test_training_monitor_uses_macos_safe_mujoco_gl(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        import monitor_so101_training_dashboard as monitor

        with mock.patch.dict(os.environ, {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}, clear=False):
            with mock.patch("platform.system", return_value="Darwin"):
                self.assertEqual(monitor._mujoco_render_env("auto"), ("glfw", None))

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("platform.system", return_value="Linux"):
                self.assertEqual(monitor._mujoco_render_env("auto"), ("egl", "egl"))

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("platform.system", return_value="Darwin"):
                self.assertEqual(monitor._mujoco_render_env("egl"), ("egl", "egl"))
                self.assertEqual(monitor._mujoco_render_env("glfw"), ("glfw", None))

    def test_training_monitor_qwen_chain_runner_reads_qwen_report(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        import monitor_so101_training_dashboard as monitor

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            args = argparse.Namespace(
                python=sys.executable,
                repo_root=Path.cwd(),
                closed_loop_task_prompt="pick and lift the green cube",
                qwen_object="green cube",
                qwen_env_object_color="green",
                qwen_model="qwen3-vl-8b-instruct-mlx",
                qwen_plan_json=None,
                qwen_response_json=Path("configs/agent/qwen3_so101_tool_planner_mock_response.json"),
                qwen_base_url=None,
                qwen_api_key=None,
                closed_loop_episodes=1,
                closed_loop_env_id="MuJoCoPickLift-v1",
                closed_loop_seed=98100,
                closed_loop_start_contract="full_chain_reset",
                policy_device="cpu",
                closed_loop_steps=2,
                policy_n_action_steps=15,
                policy_num_steps=10,
                closed_loop_valid_mask_checkpoint=Path("/tmp/valid_mask_head.pt"),
                closed_loop_valid_mask_threshold=0.5,
                closed_loop_valid_mask_consecutive=2,
                record_loop_artifacts=True,
                render_loop_media=True,
                loop_artifact_width=128,
                loop_artifact_height=128,
                loop_artifact_fps=12,
                loop_artifact_every_n_steps=1,
                local_files_only=True,
                mujoco_gl="glfw",
            )
            captured_cmd = []

            def fake_run(cmd, cwd, env, text, capture_output, check):
                del cwd, env, text, capture_output, check
                captured_cmd[:] = cmd
                output_dir = Path(cmd[cmd.index("--output-dir") + 1])
                output_dir.mkdir(parents=True)
                (output_dir / "qwen_closed_loop_eval_report.json").write_text(
                    json.dumps(
                        {
                            "operation": "so101_qwen_closed_loop_eval",
                            "status": "passed",
                            "success_rate": 1.0,
                            "episodes": [{"final_success": True}],
                            "plan": {"task": "pick and lift the green cube"},
                            "report_path": str(output_dir / "qwen_closed_loop_eval_report.json"),
                        }
                    ),
                    encoding="utf-8",
                )
                return argparse.Namespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(monitor.subprocess, "run", side_effect=fake_run):
                report = monitor._run_qwen_chain_closed_loop_eval(
                    args,
                    run_dir,
                    "000224",
                    Path("/tmp/policy"),
                )

        self.assertEqual(report["operation"], "so101_qwen_closed_loop_eval")
        self.assertEqual(report["success_rate"], 1.0)
        self.assertEqual(report["eval_skill_mode"], "qwen_edge_chain")
        self.assertIn("--policy-n-action-steps", captured_cmd)
        self.assertEqual(captured_cmd[captured_cmd.index("--policy-n-action-steps") + 1], "15")
        self.assertIn("--policy-num-steps", captured_cmd)
        self.assertEqual(captured_cmd[captured_cmd.index("--policy-num-steps") + 1], "10")
        self.assertIn("--valid-mask-checkpoint", captured_cmd)
        self.assertEqual(captured_cmd[captured_cmd.index("--valid-mask-checkpoint") + 1], "/tmp/valid_mask_head.pt")
        self.assertIn("--env-id", captured_cmd)
        self.assertEqual(captured_cmd[captured_cmd.index("--env-id") + 1], "MuJoCoPickLift-v1")
        self.assertIn("--env-object-color", captured_cmd)
        self.assertEqual(captured_cmd[captured_cmd.index("--env-object-color") + 1], "green")
        self.assertIn("--start-contract", captured_cmd)
        self.assertEqual(captured_cmd[captured_cmd.index("--start-contract") + 1], "full_chain_reset")
        self.assertIn("--record-loop-artifacts", captured_cmd)
        self.assertIn("--render-loop-media", captured_cmd)
        self.assertEqual(captured_cmd[captured_cmd.index("--artifact-width") + 1], "128")

    def test_virtual_merge_concat_dataset_len_is_sum(self) -> None:
        from types import SimpleNamespace

        from physical_ai_agent.so101_lerobot_concat import LeRobotConcatDataset

        class FakeDataset:
            def __init__(self, name: str, length: int) -> None:
                self.name = name
                self.repo_id = name
                self.root = f"/tmp/{name}"
                self.meta = SimpleNamespace(
                    name=name,
                    stats={
                        "action": {
                            "min": [0.0],
                            "max": [float(length)],
                            "mean": [float(length) / 2.0],
                            "std": [1.0],
                            "count": [length],
                        },
                    },
                )
                self.items = list(range(length))

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, index: int) -> dict[str, int | str]:
                return {"source": self.name, "value": self.items[index]}

        dataset = LeRobotConcatDataset(
            [FakeDataset("a", 2), FakeDataset("b", 3)],
            names=["a", "b"],
        )

        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset[0]["source"], "a")
        self.assertEqual(dataset[4]["source"], "b")
        self.assertEqual(dataset.source_for_index(4), {"dataset_index": 1, "dataset_name": "b", "local_index": 2})
        self.assertTrue(dataset.disable_episode_aware_sampler)
        self.assertTrue(dataset.requires_dataset_balanced_sampler)

    def test_virtual_merge_concat_dataset_is_pickle_safe_for_workers(self) -> None:
        import pickle

        from physical_ai_agent.so101_lerobot_concat import LeRobotConcatDataset

        dataset = LeRobotConcatDataset(
            [_PickleSafeFakeLeRobotDataset("a", 2), _PickleSafeFakeLeRobotDataset("b", 3)],
            names=["a", "b"],
        )

        restored = pickle.loads(pickle.dumps(dataset))

        self.assertEqual(len(restored), 5)
        self.assertEqual(restored[4]["source"], "b")
        self.assertEqual(restored.meta.fps, 30)
        self.assertIn("action", restored.meta.stats)

    def test_virtual_merge_balanced_sampler_draws_each_dataset_evenly(self) -> None:
        import torch
        from types import SimpleNamespace

        from physical_ai_agent.so101_lerobot_concat import LeRobotConcatDataset

        class FakeDataset:
            def __init__(self, name: str, length: int) -> None:
                self.name = name
                self.repo_id = name
                self.root = f"/tmp/{name}"
                self.meta = SimpleNamespace(
                    name=name,
                    stats={
                        "action": {
                            "min": [0.0],
                            "max": [float(length)],
                            "mean": [float(length) / 2.0],
                            "std": [1.0],
                            "count": [length],
                        },
                    },
                )
                self.items = list(range(length))

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, index: int) -> dict[str, int | str]:
                return {"source": self.name, "value": self.items[index]}

        dataset = LeRobotConcatDataset(
            [FakeDataset("small", 2), FakeDataset("large", 8)],
            names=["small", "large"],
        )

        weights = dataset.balanced_sample_weights()
        self.assertAlmostEqual(float(weights[:2].sum()), 0.5)
        self.assertAlmostEqual(float(weights[2:].sum()), 0.5)

        generator = torch.Generator().manual_seed(7)
        sampler = dataset.make_dataset_balanced_sampler(num_samples=1000, generator=generator)
        counts = {"small": 0, "large": 0}
        for index in sampler:
            counts[dataset.source_for_index(int(index))["dataset_name"]] += 1

        self.assertGreater(counts["small"], 450)
        self.assertLess(counts["small"], 550)
        self.assertGreater(counts["large"], 450)
        self.assertLess(counts["large"], 550)

    def test_virtual_merge_aggregates_child_stats_instead_of_first_dataset_only(self) -> None:
        from types import SimpleNamespace

        from physical_ai_agent.so101_lerobot_concat import LeRobotConcatDataset

        class FakeDataset:
            def __init__(
                self,
                name: str,
                *,
                mean: float,
                std: float,
                minimum: float,
                maximum: float,
                count: int,
            ) -> None:
                self.name = name
                self.repo_id = name
                self.root = f"/tmp/{name}"
                self.items = [None] * count
                self.meta = SimpleNamespace(
                    stats={
                        "action": {
                            "min": [minimum],
                            "max": [maximum],
                            "mean": [mean],
                            "std": [std],
                            "count": [count],
                        },
                    }
                )

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, index: int) -> dict[str, int]:
                return {"index": index}

        dataset = LeRobotConcatDataset(
            [
                FakeDataset(
                    "constant_gripper_move",
                    mean=-0.17453,
                    std=0.0,
                    minimum=-0.17453,
                    maximum=-0.17453,
                    count=10,
                ),
                FakeDataset(
                    "variable_gripper_grip",
                    mean=1.0,
                    std=0.5,
                    minimum=-0.1,
                    maximum=1.7,
                    count=10,
                ),
            ]
        )

        action_stats = dataset.meta.stats["action"]
        self.assertLess(float(action_stats["min"][0]), 0.0)
        self.assertGreater(float(action_stats["max"][0]), 1.0)
        self.assertGreater(float(action_stats["std"][0]), 0.0)

    def test_virtual_merge_dataloader_uses_dataset_balanced_sampler(self) -> None:
        import torch
        from types import SimpleNamespace

        from scripts.lerobot_train_so101_lightning import _make_dataloader

        class FakeConcatDataset(torch.utils.data.Dataset):
            requires_dataset_balanced_sampler = True
            source_lengths = [2, 8]

            def __init__(self) -> None:
                self.sampler_calls = 0

            def __len__(self) -> int:
                return 10

            def __getitem__(self, index: int) -> dict[str, int]:
                return {"index": index}

            def make_dataset_balanced_sampler(self, *, num_samples: int, generator=None):
                self.sampler_calls += 1
                return torch.utils.data.WeightedRandomSampler(
                    weights=torch.ones(len(self), dtype=torch.double),
                    num_samples=num_samples,
                    replacement=True,
                    generator=generator,
                )

        dataset = FakeConcatDataset()
        cfg = SimpleNamespace(
            num_workers=0,
            batch_size=2,
            dataset=SimpleNamespace(streaming=False),
            policy=SimpleNamespace(device="cpu"),
        )

        dataloader = _make_dataloader(cfg, dataset)

        self.assertEqual(dataset.sampler_calls, 1)
        self.assertIsInstance(dataloader.sampler, torch.utils.data.WeightedRandomSampler)

    def test_virtual_merge_rejects_schema_mismatch(self) -> None:
        from physical_ai_agent.so101_lerobot_concat import validate_lerobot_dataset_infos

        with tempfile.TemporaryDirectory() as tmpdir:
            root_a = Path(tmpdir) / "a"
            root_b = Path(tmpdir) / "b"
            _write_lerobot_info(root_a, camera1_shape=[256, 256, 3])
            _write_lerobot_info(root_b, camera1_shape=[128, 128, 3])

            with self.assertRaisesRegex(ValueError, "shape must be"):
                validate_lerobot_dataset_infos(
                    [
                        {"name": "a", "root": str(root_a), "repo_id": "a", "expected_episodes": 1, "expected_frames": 2},
                        {"name": "b", "root": str(root_b), "repo_id": "b", "expected_episodes": 1, "expected_frames": 2},
                    ]
                )

    def test_single_training_launcher_is_canonical_and_lock_guarded(self) -> None:
        source = Path("scripts/start_so101_training.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        self.assertIn("lerobot_train_so101_lightning.py", constants)
        self.assertIn("serve_so101_training_dashboard.py", constants)
        self.assertIn("log_gpu_metrics_tensorboard.py", constants)
        self.assertIn(
            "Refusing to start: an SO101 training run is already active. "
            "Use `status`, `stop`, or `start --replace`.",
            constants,
        )
        self.assertIn("--tensorboard-log-dir", constants)
        self.assertIn("--dataset-config", constants)
        self.assertIn("--validation-interval-steps", constants)
        self.assertIn("--validation-interval-epochs", constants)
        self.assertIn("run_so101_training_loop_test.py", constants)

    def test_closed_loop_eval_exposes_optional_subgoal_valid_mask_chain(self) -> None:
        source = Path("scripts/evaluate_so101_picklift_smolvla_policy.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        self.assertIn("--subgoal-chain-mode", constants)
        self.assertIn("--subgoal-sequence", constants)
        self.assertIn("--valid-mask-checkpoint", constants)
        self.assertIn("valid-mask", constants)
        self.assertIn("move_over_cube", constants)
        self.assertIn("subgoal_chain", constants)

    def test_training_launcher_passes_subgoal_chain_flags_to_monitor(self) -> None:
        source = Path("scripts/start_so101_training.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        self.assertIn("--closed-loop-subgoal-chain-mode", constants)
        self.assertIn("--closed-loop-subgoal-sequence", constants)
        self.assertIn("--closed-loop-valid-mask-checkpoint", constants)
        self.assertIn("--closed-loop-policy-n-action-steps", constants)
        self.assertIn("--closed-loop-policy-num-steps", constants)

    def test_training_monitor_passes_subgoal_chain_flags_to_closed_loop_eval(self) -> None:
        source = Path("scripts/monitor_so101_training_dashboard.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        self.assertIn("--closed-loop-subgoal-chain-mode", constants)
        self.assertIn("--subgoal-chain-mode", constants)
        self.assertIn("--closed-loop-valid-mask-checkpoint", constants)
        self.assertIn("--valid-mask-checkpoint", constants)

    def test_single_training_launcher_dry_run_builds_one_training_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(Path(tmpdir) / "run"),
                    "--validation-interval-steps",
                    "10",
                    "--",
                    "--dataset.repo_id=physical-ai-agent/test",
                    "--policy.type=smolvla",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["operation"], "start_so101_training")
            train_cmd = payload["train_cmd"]
            self.assertEqual(sum("lerobot_train_so101_lightning.py" in part for part in train_cmd), 1)
            self.assertIn("--dataset.repo_id=physical-ai-agent/test", train_cmd)
            self.assertIn("--validation-interval-steps=10", train_cmd)
            self.assertTrue(any(str(part).startswith("--output_dir=") for part in train_cmd))
            self.assertIn("tensorboard_cmd", payload)
            self.assertIn("mobile_tensorboard_url", payload)
            self.assertIn("external_tensorboard_url", payload)
            self.assertIn("external_tensorboard_note", payload)
            self.assertIsNone(payload["dashboard_cmd"])
            self.assertIsNone(payload["gpu_monitor_cmd"])
            self.assertIsNone(payload["progress_monitor_cmd"])

    def test_single_training_launcher_defaults_validation_to_closed_loop_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(Path(tmpdir) / "run"),
                    "--",
                    "--dataset.repo_id=physical-ai-agent/test",
                    "--policy.type=smolvla",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertIn("--validation-interval-epochs=1", payload["train_cmd"])

    def test_single_training_launcher_resumes_from_current_run_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            last_checkpoint = run_dir / "model" / "checkpoints" / "last"
            last_checkpoint.mkdir(parents=True)
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(run_dir),
                    "--",
                    "--dataset.repo_id=physical-ai-agent/test",
                    "--policy.type=smolvla",
                    "--resume=true",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertIn(f"--so101-resume-checkpoint-path={last_checkpoint.resolve()}", payload["train_cmd"])

    def test_training_closed_loop_episode_defaults_are_ten(self) -> None:
        start_source = Path("scripts/start_so101_training.py").read_text(encoding="utf-8")
        monitor_source = Path("scripts/monitor_so101_training_dashboard.py").read_text(encoding="utf-8")
        standard = Path("docs/so101_local_training_standard.md").read_text(encoding="utf-8")

        self.assertIn('parser.add_argument("--closed-loop-episodes", type=int, default=10)', start_source)
        self.assertIn('parser.add_argument("--closed-loop-episodes", type=int, default=10)', monitor_source)
        self.assertIn("closed-loop tests must always run exactly 10 episodes", standard)

    def test_dataset_config_launcher_defaults_closed_loop_to_ten_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(Path(tmpdir) / "run"),
                    "--use-local-dataset-roots",
                    "--dataset-config",
                    "configs/so101/training_datasets/qwen_edge_primitives.json",
                    "--runtime-platform",
                    "macos",
                    "--training-device",
                    "mps",
                    "--",
                    "--config_path=_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/qwen_edge_primitives/model/checkpoints/003136/pretrained_model/train_config.json",
                    "--steps=224",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            loop_cmd = payload["post_checkpoint_loop_cmd"]
            self.assertIsNone(payload["progress_monitor_cmd"])
            self.assertIn("--closed-loop-episodes", loop_cmd)
            self.assertEqual(loop_cmd[loop_cmd.index("--closed-loop-episodes") + 1], "10")
            self.assertIn("--iterations", loop_cmd)
            self.assertIn("1", loop_cmd)
            self.assertNotIn("--skip-validation", loop_cmd)
            self.assertTrue(any("run_so101_training_loop_test.py" in part for part in loop_cmd))
            self.assertFalse(any("monitor_so101_training_dashboard.py" in part for part in loop_cmd))

    def test_closed_loop_tensorboard_uses_separate_run(self) -> None:
        source = Path("scripts/monitor_so101_training_dashboard.py").read_text(encoding="utf-8")

        self.assertIn('run_dir / "tensorboard" / "so101_closed_loop"', source)
        self.assertNotIn('log_dir = run_dir / "tensorboard" / "so101_smolvla"', source)

    def test_qwen_edge_export_templates_are_training_prompts(self) -> None:
        _ensure_scripts_on_path()
        from scripts.export_so101_teacher_rollouts_lerobot import COLOR_SHAPE_SKILL_TASK_TEMPLATES

        self.assertEqual(
            COLOR_SHAPE_SKILL_TASK_TEMPLATES["move_over_cube_edge"],
            "Move the gripper above one visible {color} {shape} edge.",
        )
        self.assertEqual(
            COLOR_SHAPE_SKILL_TASK_TEMPLATES["align_fixed_jaw_cube_edge"],
            "Align the gripper jaws around one visible {color} {shape} edge.",
        )
        self.assertEqual(
            COLOR_SHAPE_SKILL_TASK_TEMPLATES["move_and_align_cube_edge"],
            "Move above one visible {color} {shape} edge and align the gripper jaws around it.",
        )
        self.assertEqual(
            COLOR_SHAPE_SKILL_TASK_TEMPLATES["grip_from_edge_cube"],
            "grip the {color} {shape} and lift",
        )
        self.assertEqual(
            COLOR_SHAPE_SKILL_TASK_TEMPLATES["grip_from_above_edge_cube"],
            "grip the {color} {shape} and lift",
        )
        self.assertEqual(
            COLOR_SHAPE_SKILL_TASK_TEMPLATES["grip_the_cube_v1"],
            "grip the {color} {shape} and lift",
        )
        for template in COLOR_SHAPE_SKILL_TASK_TEMPLATES.values():
            self.assertNotIn("static finger pad", template)

    def test_grip_the_cube_v1_requires_explicit_roll_alignment_phase(self) -> None:
        config_path = Path("configs/so101/training_datasets/grip_the_cube_v1.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        generation = config["dataset_generation"]

        self.assertEqual(
            generation["phases"],
            [
                "move_to_cube",
                "roll_align_with_cube_edge",
                "gripper_descend",
                "settle_aligned",
                "close",
                "lift",
                "terminal_hold",
            ],
        )
        alignment = generation["alignment_contract"]
        self.assertEqual(alignment["roll_alignment_phase"], "roll_align_with_cube_edge")
        self.assertEqual(alignment["descend_phase"], "gripper_descend")
        self.assertTrue(alignment["pre_close_required"])
        self.assertLessEqual(alignment["pre_close_edge_parallel_threshold_deg"], 8.0)
        self.assertLessEqual(alignment["pre_close_static_finger_xy_threshold"], 0.012)
        self.assertNotIn("pre_close_wrist_camera_centered", alignment)

    def test_qwen_edge_merge_preserves_dataset_prompt_without_rewrite(self) -> None:
        _ensure_scripts_on_path()
        from scripts.merge_so101_lerobot_shards import _frame_from_source

        sample = {
            "observation.images.camera1": np.zeros((3, 2, 2), dtype=np.float32),
            "observation.images.camera2": np.zeros((3, 2, 2), dtype=np.float32),
            "observation.images.camera3": np.zeros((3, 2, 2), dtype=np.float32),
            "observation.state": np.zeros(6, dtype=np.float32),
            "action": np.zeros(6, dtype=np.float32),
            "task": "Align the static finger pad with one visible red cube edge.",
        }

        frame = _frame_from_source(sample, include_camera3=True)

        self.assertEqual(frame["task"], sample["task"])

    def test_dataset_viewer_uses_training_config_group_order_for_default_dataset(self) -> None:
        source = Path("scripts/serve_so101_dataset_viewer.py").read_text(encoding="utf-8")

        self.assertIn('Path("configs/so101/training_datasets/qwen_edge_primitives.json")', source)
        self.assertIn("const orderedNames = []", source)
        self.assertIn("for (const group of payload.dataset_groups || [])", source)
        self.assertNotIn("Object.keys(datasets).map(name => `<option", source)

    def test_single_training_launcher_dataset_config_injects_dataset_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "dataset_config.json"
            config.write_text(
                json.dumps(
                    {
                        "train_dataset": {
                            "repo_id": "physical-ai-agent/train",
                            "root": "_workspace/train",
                        },
                        "validation_dataset": {
                            "repo_id": "physical-ai-agent/val",
                            "root": "_workspace/val",
                        },
                        "training": {
                            "num_workers": 4,
                            "policy_repo_id": "mhlee1215/test-policy",
                            "policy_push_to_hub": False,
                            "lightning_precision": "bf16-mixed",
                            "steps_per_epoch": 42,
                        },
                        "closed_loop": {
                            "eval_skill_mode": "pick_from_top_cube",
                            "task_prompt": "Pick the cube from the top and lift it cleanly.",
                            "record_rollout_gif": True,
                        },
                        "predecoded_image_cache": {
                            "root_env": "SO101_TEST_CACHE_ROOT",
                            "default_root": "_workspace/cache",
                            "train": "train",
                            "validation": "val",
                        },
                        "tensorboard": {
                            "log_input_images_every_n_steps": 10,
                            "log_input_metadata_every_n_steps": 10,
                        },
                        "action_chunk_consistency": {
                            "steps": 15,
                            "weight": 0.05,
                        },
                        "action_smoothness": {
                            "weight": 0.01,
                            "include_gripper": False,
                        },
                        "action_teacher_importance": {
                            "delta_weight": 0.75,
                            "gripper_transition_weight": 1.5,
                            "terminal_steps": 12,
                            "terminal_weight": 1.4,
                        },
                        "augmentation": {
                            "state_jitter_std": 0.01,
                            "state_jitter_arm_only": False,
                            "state_dropout_prob": 0.02,
                            "state_dropout_keep_gripper": True,
                            "image_camera_dropout_prob": 0.04,
                            "image_patch_dropout_prob": 0.05,
                            "image_patch_mask_ratio": 0.15,
                            "image_blur_prob": 0.1,
                            "image_blur_kernel_size": 5,
                            "image_noise_std": 0.02,
                            "image_color_jitter": True,
                            "image_sharpness_jitter": True,
                            "image_affine_degrees": 5.0,
                            "image_affine_translate": 0.05,
                            "gpu_image_augmentation": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(Path(tmpdir) / "run"),
                    "--dataset-config",
                    str(config),
                    "--runtime-platform",
                    "macos",
                    "--",
                    "--policy.type=smolvla",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src", "SO101_TEST_CACHE_ROOT": "/tmp/so101-cache"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            train_cmd = payload["train_cmd"]
            self.assertEqual(
                payload["local_training_standard"]["name"],
                "primitive training with qwen validation v1",
            )
            self.assertTrue(payload["local_training_standard"]["doc"].endswith("docs/so101_local_training_standard.md"))
            self.assertIn(
                "Local SO101/SmolVLA training launches outside the Codex sandbox.",
                payload["local_training_standard"]["summary"],
            )
            self.assertIn("--dataset.repo_id=physical-ai-agent/train", train_cmd)
            self.assertIn("--dataset.root=_workspace/train", train_cmd)
            self.assertIn("--validation-dataset-repo-id=physical-ai-agent/val", train_cmd)
            self.assertIn("--validation-dataset-root=_workspace/val", train_cmd)
            self.assertIn("--num_workers=0", train_cmd)
            self.assertIn("--policy.path=mhlee1215/test-policy", train_cmd)
            self.assertIn("--policy.push_to_hub=false", train_cmd)
            self.assertIn("--lightning-precision=bf16-mixed", train_cmd)
            self.assertIn("--validation-interval-epochs=1", train_cmd)
            self.assertIn("--save_freq=42", train_cmd)
            self.assertIn("--so101-image-cache-dir=/tmp/so101-cache/train", train_cmd)
            self.assertIn("--validation-image-cache-dir=/tmp/so101-cache/val", train_cmd)
            self.assertIn("tensorboard_cmd", payload)
            self.assertIsNone(payload["dashboard_cmd"])
            self.assertIsNone(payload["gpu_monitor_cmd"])
            self.assertIsNone(payload["progress_monitor_cmd"])
            self.assertIsNotNone(payload["post_checkpoint_loop_cmd"])
            self.assertIn("--post-checkpoint-loop-command-json", train_cmd)
            self.assertIn("training_run_summary_path", payload)
            self.assertIn("--training-run-summary-path", train_cmd)
            self.assertEqual(
                train_cmd[train_cmd.index("--training-run-summary-path") + 1],
                payload["training_run_summary_path"],
            )
            self.assertEqual(payload["runtime_contract"]["runtime_platform"], "macos")
            self.assertEqual(payload["runtime_contract"]["training_device"], "mps")
            self.assertEqual(payload["runtime_contract"]["closed_loop_mujoco_gl"], "glfw")
            self.assertIn("--policy.device=mps", train_cmd)
            self.assertIn("--lightning-accelerator=mps", train_cmd)
            self.assertEqual(
                payload["cache_build_cmds"][0][-1],
                "/tmp/so101-cache/train",
            )
            self.assertEqual(
                payload["cache_build_cmds"][1][-1],
                "/tmp/so101-cache/val",
            )
            self.assertIn("--log-input-images-every-n-steps=10", train_cmd)
            self.assertIn("--log-input-metadata-every-n-steps=10", train_cmd)
            self.assertIn("--so101-action-chunk-consistency-steps=15", train_cmd)
            self.assertIn("--so101-action-chunk-consistency-weight=0.05", train_cmd)
            self.assertIn("--so101-action-smoothness-loss-weight=0.01", train_cmd)
            self.assertIn("--no-so101-action-smoothness-include-gripper", train_cmd)
            self.assertIn("--so101-action-delta-loss-weight=0.75", train_cmd)
            self.assertIn("--so101-action-gripper-transition-loss-weight=1.5", train_cmd)
            self.assertIn("--so101-action-terminal-loss-steps=12", train_cmd)
            self.assertIn("--so101-action-terminal-loss-weight=1.4", train_cmd)
            self.assertIn("--so101-state-jitter-std=0.01", train_cmd)
            self.assertIn("--no-so101-state-jitter-arm-only", train_cmd)
            self.assertIn("--so101-state-dropout-prob=0.02", train_cmd)
            self.assertIn("--so101-state-dropout-keep-gripper", train_cmd)
            self.assertIn("--so101-image-camera-dropout-prob=0.04", train_cmd)
            self.assertIn("--so101-image-patch-dropout-prob=0.05", train_cmd)
            self.assertIn("--so101-image-patch-mask-ratio=0.15", train_cmd)
            self.assertIn("--so101-image-blur-prob=0.1", train_cmd)
            self.assertIn("--so101-image-blur-kernel-size=5", train_cmd)
            self.assertIn("--so101-image-noise-std=0.02", train_cmd)
            self.assertIn("--so101-image-color-jitter", train_cmd)
            self.assertIn("--so101-image-sharpness-jitter", train_cmd)
            self.assertIn("--so101-image-affine-degrees=5.0", train_cmd)
            self.assertIn("--so101-image-affine-translate=0.05", train_cmd)
            self.assertIn("--so101-gpu-image-augmentation", train_cmd)
            self.assertFalse(any("action-dropout" in arg for arg in train_cmd))
            self.assertEqual(payload["dataset_config"]["train_dataset"]["repo_id"], "physical-ai-agent/train")

    def test_single_training_launcher_uses_presets_instead_of_extra_wrapper_scripts(self) -> None:
        self.assertFalse(Path("scripts/start_so101_loopfix_training_local.sh").exists())
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--json",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--preset",
                    "qwen-edge-loopfix-local",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            train_cmd = payload["train_cmd"]

            self.assertEqual(payload["dataset_config"]["name"], "qwen_edge_primitives")
            self.assertTrue(str(payload["run_dir"]).endswith("qwen_edge_primitives_resume_009632_loopfix_30000"))
            self.assertEqual(payload["tensorboard_url"], "http://127.0.0.1:6015/")
            self.assertIn("--policy.device=mps", train_cmd)
            self.assertIn("--lightning-accelerator=mps", train_cmd)
            self.assertIn("--steps=30000", train_cmd)
            self.assertIn("--training-run-summary-path", train_cmd)
            self.assertEqual(
                [case["id"] for case in payload["dataset_config"]["closed_loop"]["test_cases"]],
                ["move_over_cube_edge", "align_fixed_jaw_cube_edge", "grip_from_edge_cube"],
            )

    def test_single_training_launcher_records_stable_training_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "dataset_config.json"
            train_root = Path(tmpdir) / "train"
            val_root = Path(tmpdir) / "val"
            train_root.mkdir()
            val_root.mkdir()
            config.write_text(
                json.dumps(
                    {
                        "name": "tiny_grip_debug",
                        "train_dataset": {
                            "name": "grip_from_edge_cube_train",
                            "repo_id": "physical-ai-agent/train",
                            "root": str(train_root),
                        },
                        "validation_dataset": {
                            "name": "grip_from_edge_cube_val",
                            "repo_id": "physical-ai-agent/val",
                            "root": str(val_root),
                        },
                        "training": {"steps_per_epoch": 10, "policy_push_to_hub": False},
                        "closed_loop": {
                            "runner": "qwen_chain",
                            "task_prompt": "Close the gripper on the green cube edge and lift.",
                            "valid_mask_checkpoint": "_workspace/so101_valid_mask_head/qwen_edge_primitives/valid_mask_head.pt",
                            "test_cases": [
                                {
                                    "id": "grip_from_edge_cube",
                                    "episodes": 10,
                                    "seed": 98300,
                                    "start_contract": "grip_from_edge_cube",
                                    "task_prompt": "Close the gripper on the green cube edge and lift.",
                                    "plan_json": "configs/agent/qwen3_so101_tool_plan_grip_from_edge_cube_green_cube.json",
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--json",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(Path(tmpdir) / "run"),
                    "--training-id",
                    "debug.grip.001",
                    "--dataset-config",
                    str(config),
                    "--use-local-dataset-roots",
                    "--runtime-platform",
                    "macos",
                    "--training-device",
                    "mps",
                    "--closed-loop-runner",
                    "qwen_chain",
                    "--closed-loop-every-epochs",
                    "1",
                    "--closed-loop-policy",
                    "best_or_periodic",
                    "--",
                    "--steps=20",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["training_id"], "debug.grip.001")
            self.assertEqual(payload["dataset_config"]["name"], "tiny_grip_debug")

    def test_training_manager_lists_runs_by_training_id(self) -> None:
        _ensure_scripts_on_path()
        import serve_so101_training_manager as manager

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            run_dir = repo_root / "_workspace" / "so101_training" / "runs" / "debug_run"
            run_dir.mkdir(parents=True)
            summary_path = run_dir / "training_run_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "training_id": "debug.grip.001",
                        "run_dir": str(run_dir),
                        "dataset_config": {"name": "tiny_grip_debug", "task": "grip_from_edge_cube"},
                        "tensorboard_url": "http://127.0.0.1:6015/",
                    }
                ),
                encoding="utf-8",
            )
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir()
            (metrics_dir / "training_metrics.jsonl").write_text('{"step": 1, "loss": 0.2}\n', encoding="utf-8")
            (metrics_dir / "validation_metrics.jsonl").write_text('{"step": 1, "loss": 0.3}\n', encoding="utf-8")
            (metrics_dir / "closed_loop_metrics.jsonl").write_text(
                '{"step": 1, "test_id": "grip_from_edge_cube", "success_rate": 0.4}\n',
                encoding="utf-8",
            )

            listing = manager._runs_payload(repo_root)
            detail = manager._run_detail(repo_root, "debug.grip.001")

        self.assertEqual(listing["runs"][0]["training_id"], "debug.grip.001")
        self.assertEqual(listing["runs"][0]["latest_train_loss"], 0.2)
        self.assertEqual(listing["runs"][0]["latest_val_loss"], 0.3)
        self.assertEqual(detail["metrics"]["closed_loop"][0]["test_id"], "grip_from_edge_cube")

    def test_single_training_launcher_uses_linux_runpod_runtime_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "dataset_config.json"
            config.write_text(
                json.dumps(
                    {
                        "train_dataset": {
                            "repo_id": "physical-ai-agent/train",
                            "root": "_workspace/train",
                        },
                        "validation_dataset": {
                            "repo_id": "physical-ai-agent/val",
                            "root": "_workspace/val",
                        },
                        "training": {
                            "steps_per_epoch": 42,
                            "policy_push_to_hub": False,
                        },
                        "closed_loop": {
                            "eval_skill_mode": "pick_from_top_cube",
                            "task_prompt": "Pick the cube from the top and lift it cleanly.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(Path(tmpdir) / "run"),
                    "--dataset-config",
                    str(config),
                    "--runtime-platform",
                    "linux",
                    "--",
                    "--policy.type=smolvla",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["runtime_contract"]["runtime_platform"], "linux")
            self.assertEqual(payload["runtime_contract"]["training_device"], "cuda")
            self.assertEqual(payload["runtime_contract"]["lightning_accelerator"], "cuda")
            self.assertEqual(payload["runtime_contract"]["closed_loop_mujoco_gl"], "egl")
            self.assertIn("--policy.device=cuda", payload["train_cmd"])
            self.assertIn("--lightning-accelerator=cuda", payload["train_cmd"])
            self.assertIsNone(payload["progress_monitor_cmd"])
            self.assertIsNotNone(payload["post_checkpoint_loop_cmd"])

    def test_single_training_launcher_enables_progress_monitor_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "dataset_config.json"
            config.write_text(
                json.dumps(
                    {
                        "train_dataset": {
                            "repo_id": "physical-ai-agent/train",
                            "root": "_workspace/train",
                        },
                        "validation_dataset": {
                            "repo_id": "physical-ai-agent/val",
                            "root": "_workspace/val",
                        },
                        "training": {
                            "steps_per_epoch": 42,
                            "policy_push_to_hub": False,
                        },
                        "closed_loop": {
                            "eval_skill_mode": "pick_from_top_cube",
                            "task_prompt": "Pick the cube from the top and lift it cleanly.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--with-progress-monitor",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(Path(tmpdir) / "run"),
                    "--dataset-config",
                    str(config),
                    "--runtime-platform",
                    "linux",
                    "--",
                    "--policy.type=smolvla",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertIn("--mujoco-gl", payload["progress_monitor_cmd"])
            self.assertIn("egl", payload["progress_monitor_cmd"])
            self.assertIn("--closed-loop-episodes", payload["progress_monitor_cmd"])
            self.assertIn("10", payload["progress_monitor_cmd"])

    def test_single_training_launcher_can_use_local_dataset_roots_for_debugging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "dataset_config.json"
            config.write_text(
                json.dumps(
                    {
                        "train_dataset": {
                            "repo_id": "physical-ai-agent/train",
                            "root": "_workspace/local_train",
                            "hf_repo_id": "mhlee1215/so101-nexus-sim-dataset",
                            "hf_repo_type": "dataset",
                            "hf_path_in_repo": "datasets/tiny/train",
                        },
                        "validation_dataset": {
                            "repo_id": "physical-ai-agent/val",
                            "root": "_workspace/local_val",
                            "hf_repo_id": "mhlee1215/so101-nexus-sim-dataset",
                            "hf_repo_type": "dataset",
                            "hf_path_in_repo": "datasets/tiny/validation",
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--use-local-dataset-roots",
                    "--allow-incomplete-monitoring",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(Path(tmpdir) / "run"),
                    "--dataset-config",
                    str(config),
                    "--",
                    "--policy.type=smolvla",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            train_cmd = payload["train_cmd"]
            self.assertIn("--dataset.root=_workspace/local_train", train_cmd)
            self.assertIn("--validation-dataset-root=_workspace/local_val", train_cmd)
            self.assertNotIn("hf_dataset_downloads", payload["dataset_config"])
            self.assertNotIn("hf_resolved_root", payload["dataset_config"]["train_dataset"])

    def test_single_training_launcher_fails_fast_without_validation_and_closed_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "dataset_config.json"
            config.write_text(
                json.dumps(
                    {
                        "train_dataset": {
                            "repo_id": "physical-ai-agent/train",
                            "root": "_workspace/train",
                        },
                        "training": {"steps_per_epoch": 42},
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(Path(tmpdir) / "run"),
                    "--dataset-config",
                    str(config),
                    "--",
                    "--policy.type=smolvla",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("SO101 monitored training contract failed", completed.stderr)
            self.assertIn("validation_dataset", completed.stderr)
            self.assertIn("validation-dataset-root", completed.stderr)

    def test_single_training_launcher_fails_fast_when_checkpoint_cadence_is_too_dense(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "dataset_config.json"
            config.write_text(
                json.dumps(
                    {
                        "train_dataset": {
                            "repo_id": "physical-ai-agent/train",
                            "root": "_workspace/train",
                        },
                        "validation_dataset": {
                            "repo_id": "physical-ai-agent/val",
                            "root": "_workspace/val",
                        },
                        "training": {"steps_per_epoch": 325},
                        "closed_loop": {
                            "eval_skill_mode": "pick_from_top_cube",
                            "task_prompt": "Pick the cube from the top and lift it cleanly.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(Path(tmpdir) / "run"),
                    "--dataset-config",
                    str(config),
                    "--closed-loop-every-epochs",
                    "1",
                    "--",
                    "--policy.type=smolvla",
                    "--steps=50000",
                    "--save_freq=325",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("would create about", completed.stderr)
            self.assertIn("--max-monitored-checkpoints", completed.stderr)

    def test_single_training_launcher_allows_dense_checkpoint_candidates_with_strict_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "dataset_config.json"
            config.write_text(
                json.dumps(
                    {
                        "train_dataset": {
                            "repo_id": "physical-ai-agent/train",
                            "root": "_workspace/train",
                        },
                        "validation_dataset": {
                            "repo_id": "physical-ai-agent/val",
                            "root": "_workspace/val",
                        },
                        "training": {
                            "checkpoint_retention_policy": "best_val_and_closed_loop",
                            "steps_per_epoch": 325,
                        },
                        "closed_loop": {
                            "eval_skill_mode": "pick_from_top_cube",
                            "task_prompt": "Pick the cube from the top and lift it cleanly.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(Path(tmpdir) / "run"),
                    "--dataset-config",
                    str(config),
                    "--closed-loop-every-epochs",
                    "1",
                    "--",
                    "--policy.type=smolvla",
                    "--steps=50000",
                    "--save_freq=325",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertIn("--checkpoint-retention-policy=best_val_and_closed_loop", payload["train_cmd"])

    def test_single_training_launcher_clears_tensorboard_old_data_by_default(self) -> None:
        from scripts.start_so101_training import _clear_tensorboard_old_data

        with tempfile.TemporaryDirectory() as tmpdir:
            tensorboard_dir = Path(tmpdir) / "tensorboard"
            active_run_dir = tensorboard_dir / "so101_smolvla"
            active_run_dir.mkdir(parents=True)
            old_event = active_run_dir / "events.out.tfevents.1.host.123.0"
            profile_marker = active_run_dir / "trace.profile-empty"
            keep_metadata = active_run_dir / "README.txt"
            old_event.write_text("old", encoding="utf-8")
            profile_marker.write_text("old", encoding="utf-8")
            keep_metadata.write_text("keep", encoding="utf-8")

            removed = _clear_tensorboard_old_data(tensorboard_dir)

            self.assertEqual(removed, 2)
            self.assertFalse(old_event.exists())
            self.assertFalse(profile_marker.exists())
            self.assertTrue(keep_metadata.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "dataset_config.json"
            config.write_text(
                json.dumps(
                    {
                        "train_dataset": {
                            "repo_id": "physical-ai-agent/train",
                            "root": "_workspace/train",
                        },
                        "validation_dataset": {
                            "repo_id": "physical-ai-agent/val",
                            "root": "_workspace/val",
                        },
                        "training": {"steps_per_epoch": 10},
                        "closed_loop": {
                            "eval_skill_mode": "pick_from_top_cube",
                            "task_prompt": "Pick the cube from the top and lift it cleanly.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/start_so101_training.py",
                    "start",
                    "--dry-run",
                    "--lock-file",
                    str(Path(tmpdir) / "active.json"),
                    "--run-dir",
                    str(Path(tmpdir) / "run"),
                    "--dataset-config",
                    str(config),
                    "--",
                    "--policy.type=smolvla",
                    "--steps=10",
                    "--save_freq=10",
                ],
                check=False,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": "src"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["clear_tensorboard_old_data"])

    def test_single_training_launcher_reports_tensorboard_access_set(self) -> None:
        from scripts.start_so101_training import _human_status, _tensorboard_tunnel_url_from_log

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "tensorboard_tunnel.log"
            log_path.write_text(
                "2026-07-08 INF +--------------------------------------------------------------------------------------------+\n"
                "2026-07-08 INF |  https://example-alpha.trycloudflare.com                                      |\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _tensorboard_tunnel_url_from_log(log_path),
                "https://example-alpha.trycloudflare.com",
            )

        rendered = _human_status(
            {
                "active": True,
                "run_dir": "/tmp/run",
                "train": {"alive": True, "pid": 1},
                "tensorboard": {"alive": True, "pid": 2},
                "tensorboard_tunnel": {"alive": True, "pid": 3},
                "dashboard": {"alive": None, "pid": None},
                "gpu_monitor": {"alive": None, "pid": None},
                "progress_monitor": {"alive": None, "pid": None},
                "tensorboard_url": "http://127.0.0.1:6006/",
                "mobile_tensorboard_url": "http://192.168.4.46:6006/",
                "external_tensorboard_url": "https://example-alpha.trycloudflare.com",
            }
        )

        self.assertIn("tensorboard_url: http://127.0.0.1:6006/", rendered)
        self.assertIn("mobile_tensorboard_url: http://192.168.4.46:6006/", rendered)
        self.assertIn("external_tensorboard_url: https://example-alpha.trycloudflare.com", rendered)

    def test_prune_so101_checkpoints_keeps_best_validation_and_latest_candidate(self) -> None:
        from scripts.prune_so101_checkpoints import prune_once

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            checkpoints = run_dir / "model" / "checkpoints"
            metrics = run_dir / "metrics"
            metrics.mkdir(parents=True)
            for name in ("000325", "000650", "000975"):
                checkpoint = checkpoints / name
                (checkpoint / "pretrained_model").mkdir(parents=True)
                (checkpoint / "training_state").mkdir()
                for path in (
                    checkpoint / "pretrained_model" / "model.safetensors",
                    checkpoint / "pretrained_model" / "train_config.json",
                    checkpoint / "training_state" / "training_step.json",
                    checkpoint / "training_state" / "optimizer_state.safetensors",
                    checkpoint / "training_state" / "scheduler_state.json",
                ):
                    path.write_text("x", encoding="utf-8")
            (metrics / "validation_metrics.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"checkpoint": "000325", "loss": 0.8}),
                        json.dumps({"checkpoint": "000650", "loss": 0.4}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            prune_once(
                run_dir,
                checkpoint_root=None,
                keep=set(),
                keep_latest_complete=1,
                keep_best_validation=True,
            )

            self.assertFalse((checkpoints / "000325").exists())
            self.assertTrue((checkpoints / "000650").exists())
            self.assertTrue((checkpoints / "000975").exists())
            self.assertIn("checkpoint_pruned", (metrics / "monitor_events.jsonl").read_text(encoding="utf-8"))

    def test_lightning_retention_policy_keeps_only_three_best_aliases(self) -> None:
        source = Path("scripts/lerobot_train_so101_lightning.py").read_text(encoding="utf-8")

        self.assertIn('"best_closed_loop", "best_val_loss", "best_train_loss"', source)
        self.assertIn('"best_train_loss"', source)
        self.assertIn('"retained_as_best": retained', source)
        self.assertIn("if checkpoint_dir.exists() and checkpoint_dir.name.isdigit():", source)
        self.assertNotIn("checkpoint_dir.exists() and checkpoint_dir.name.isdigit() and retained", source)

    def test_validation_input_images_are_logged_on_each_validation_call(self) -> None:
        source = Path("scripts/lerobot_train_so101_lightning.py").read_text(encoding="utf-8")

        self.assertIn('if split != "val" and step % self.log_input_images_every_n_steps != 0:', source)

    def test_policy_checkpoint_sidecar_valid_mask_head_is_loaded(self) -> None:
        source = Path("scripts/lerobot_train_so101_lightning.py").read_text(encoding="utf-8")

        self.assertIn("_load_valid_mask_head_from_policy_path_if_available", source)
        self.assertIn("wrapper_args.lerobot_args = list(lerobot_args)", source)
        self.assertIn("_policy_path_from_config_or_args", source)
        self.assertIn('if path.name == "pretrained_model":', source)
        self.assertIn('candidates.append(path.parent / "valid_mask_head.pt")', source)

    def test_so101_training_launcher_resolves_hf_dataset_subfolders_before_training(self) -> None:
        from scripts import start_so101_training

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            cache_root = Path(tmpdir) / "hf_datasets"
            repo_root.mkdir()
            config = {
                "train_dataset": {
                    "repo_id": "physical-ai-agent/train",
                    "root": "_workspace/local_train",
                    "hf_repo_id": "mhlee1215/so101-nexus-sim-dataset",
                    "hf_repo_type": "dataset",
                    "hf_path_in_repo": "datasets/tiny/train",
                },
                "validation_dataset": {
                    "repo_id": "physical-ai-agent/validation",
                    "root": "_workspace/local_validation",
                    "hf_repo_id": "mhlee1215/so101-nexus-sim-dataset",
                    "hf_repo_type": "dataset",
                    "hf_path_in_repo": "datasets/tiny/validation",
                },
                "predecoded_image_cache": {
                    "default_root": str(Path(tmpdir) / "image_cache"),
                    "train": "train",
                    "validation": "validation",
                },
            }

            with mock.patch.object(start_so101_training, "_snapshot_download") as snapshot_download:
                resolved = start_so101_training._resolve_hf_dataset_downloads(
                    config,
                    repo_root=repo_root,
                    cache_root=cache_root,
                    download=True,
                )

            local_repo_dir = cache_root / "mhlee1215__so101-nexus-sim-dataset"
            train_root = local_repo_dir / "datasets/tiny/train"
            validation_root = local_repo_dir / "datasets/tiny/validation"
            self.assertEqual(snapshot_download.call_count, 2)
            snapshot_download.assert_any_call(
                repo_id="mhlee1215/so101-nexus-sim-dataset",
                repo_type="dataset",
                allow_patterns=["datasets/tiny/train/**"],
                local_dir=local_repo_dir,
                local_files_only=False,
            )
            snapshot_download.assert_any_call(
                repo_id="mhlee1215/so101-nexus-sim-dataset",
                repo_type="dataset",
                allow_patterns=["datasets/tiny/validation/**"],
                local_dir=local_repo_dir,
                local_files_only=False,
            )
            self.assertEqual(resolved["train_dataset"]["root"], str(train_root))
            self.assertEqual(resolved["validation_dataset"]["root"], str(validation_root))
            self.assertEqual(config["train_dataset"]["root"], "_workspace/local_train")

            train_args = start_so101_training._with_dataset_config([], resolved)
            self.assertIn("--dataset.repo_id=physical-ai-agent/train", train_args)
            self.assertIn(f"--dataset.root={train_root}", train_args)
            self.assertIn("--validation-dataset-repo-id=physical-ai-agent/validation", train_args)
            self.assertIn(f"--validation-dataset-root={validation_root}", train_args)

            cache_cmds = start_so101_training._cache_build_commands(
                Path(sys.executable),
                repo_root,
                resolved,
            )
            self.assertIn(str(train_root), cache_cmds[0])
            self.assertIn(str(validation_root), cache_cmds[1])
            self.assertEqual(resolved["hf_dataset_downloads"][0]["path_in_repo"], "datasets/tiny/train")

    def test_virtual_merge_train_datasets_resolves_multiple_hf_sources(self) -> None:
        from scripts import start_so101_training

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            cache_root = Path(tmpdir) / "hf_datasets"
            repo_root.mkdir()
            config = {
                "train_datasets": [
                    {
                        "name": "pick_cube_train",
                        "repo_id": "physical-ai-agent/source-a",
                        "root": "_workspace/local_a",
                        "hf_repo_id": "mhlee1215/so101-nexus-sim-dataset",
                        "hf_repo_type": "dataset",
                        "hf_path_in_repo": "datasets/pick_cube/train",
                    },
                    {
                        "name": "pick_place_train",
                        "repo_id": "physical-ai-agent/source-b",
                        "root": "_workspace/local_b",
                        "hf_repo_id": "mhlee1215/so101-nexus-sim-dataset",
                        "hf_repo_type": "dataset",
                        "hf_path_in_repo": "datasets/pick_and_place_cube/train",
                    },
                ],
                "validation_dataset": {
                    "repo_id": "physical-ai-agent/validation",
                    "root": "_workspace/local_validation",
                    "hf_repo_id": "mhlee1215/so101-nexus-sim-dataset",
                    "hf_repo_type": "dataset",
                    "hf_path_in_repo": "datasets/pick_and_place_cube/validation",
                },
                "closed_loop": {
                    "eval_skill_mode": "pick_and_place_cube",
                    "task_prompt": "Pick up the small red cube and place it on the blue circle.",
                    "record_rollout_gif": True,
                },
                "predecoded_image_cache": {
                    "default_root": "_workspace/cache",
                    "train": {
                        "pick_cube_train": "pick_cube_cache",
                        "pick_place_train": "pick_place_cache",
                    },
                    "validation": "validation_cache",
                },
            }

            with mock.patch.object(start_so101_training, "_snapshot_download") as snapshot_download:
                resolved = start_so101_training._resolve_hf_dataset_downloads(
                    config,
                    repo_root=repo_root,
                    cache_root=cache_root,
                    download=True,
                )
            prepared = start_so101_training._prepare_merged_train_dataset(
                resolved,
                repo_root=repo_root,
                python=Path(sys.executable),
                merge=False,
            )

            local_repo_dir = cache_root / "mhlee1215__so101-nexus-sim-dataset"
            self.assertEqual(snapshot_download.call_count, 3)
            self.assertIs(prepared, resolved)
            self.assertEqual(resolved["train_datasets"][0]["root"], str(local_repo_dir / "datasets/pick_cube/train"))
            self.assertEqual(
                resolved["train_datasets"][1]["root"],
                str(local_repo_dir / "datasets/pick_and_place_cube/train"),
            )
            self.assertNotIn("merge_command", resolved["train_datasets"][0])
            self.assertNotIn("train_dataset", resolved)

            train_args = start_so101_training._with_dataset_config([], resolved)
            self.assertIn(f"--dataset.root={local_repo_dir / 'datasets/pick_cube/train'}", train_args)
            self.assertTrue(any(arg.startswith("--train-datasets-json=") for arg in train_args))
            progress_cmd = start_so101_training._progress_monitor_command(
                args=argparse.Namespace(
                    python=Path(sys.executable),
                    progress_monitor_interval_s=600,
                    closed_loop_every_epochs=10,
                    closed_loop_episodes=8,
                    closed_loop_steps=120,
                    closed_loop_policy="periodic",
                    closed_loop_eval_skill_mode=None,
                    closed_loop_task_prompt=None,
                    closed_loop_record_rollout_gif=False,
                    record_loop_artifacts=True,
                    loop_artifact_width=128,
                    loop_artifact_height=128,
                    loop_artifact_fps=12,
                    loop_artifact_every_n_steps=1,
                    closed_loop_runner="auto",
                    qwen_model="qwen3-vl-8b-instruct-mlx",
                    qwen_base_url=None,
                    qwen_api_key=None,
                    qwen_response_json=None,
                    qwen_plan_json=None,
                    qwen_object="green cube",
                    qwen_env_object_color=None,
                    closed_loop_env_id=None,
                    closed_loop_subgoal_chain_mode="off",
                    closed_loop_fixed_subgoal_chunks=1,
                    closed_loop_valid_mask_threshold=0.5,
                    closed_loop_valid_mask_consecutive=2,
                    closed_loop_valid_mask_checkpoint=Path("/tmp/valid_mask_head.pt"),
                    closed_loop_policy_n_action_steps=15,
                    closed_loop_policy_num_steps=10,
                ),
                repo_root=repo_root,
                run_dir=repo_root / "run",
                train_output_dir=repo_root / "run/model",
                dataset_config=resolved,
                training_args=[],
                train_pid_file=repo_root / "run/train.pid",
                runtime_contract={
                    "runtime_platform": "linux",
                    "training_device": "cuda",
                    "lightning_accelerator": "cuda",
                    "closed_loop_device": "cuda",
                    "closed_loop_mujoco_gl": "egl",
                },
            )
            self.assertIn("--closed-loop-eval-skill-mode", progress_cmd)
            self.assertIn("pick_and_place_cube", progress_cmd)
            self.assertIn("--closed-loop-task-prompt", progress_cmd)
            self.assertIn("--closed-loop-record-rollout-gif", progress_cmd)

    def test_training_launcher_resolves_hf_merge_sources_for_train_and_validation(self) -> None:
        from scripts import start_so101_training

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            cache_root = Path(tmpdir) / "hf_datasets"
            repo_root.mkdir()
            config = {
                "train_dataset": {
                    "repo_id": "physical-ai-agent/merged",
                    "root": "_workspace/merged/train",
                    "hf_merge_sources": [
                        {
                            "name": "pick_cube_train",
                            "repo_id": "physical-ai-agent/source-a",
                            "hf_repo_id": "mhlee1215/so101-nexus-sim-dataset",
                            "hf_repo_type": "dataset",
                            "hf_path_in_repo": "datasets/pick_cube/train",
                        },
                        {
                            "name": "pick_place_train",
                            "repo_id": "physical-ai-agent/source-b",
                            "hf_repo_id": "mhlee1215/so101-nexus-sim-dataset",
                            "hf_repo_type": "dataset",
                            "hf_path_in_repo": "datasets/pick_and_place_cube/train",
                        },
                    ],
                },
                "validation_dataset": {
                    "repo_id": "physical-ai-agent/merged-validation",
                    "root": "_workspace/merged/validation",
                    "hf_merge_sources": [
                        {
                            "name": "pick_cube_validation",
                            "repo_id": "physical-ai-agent/source-a-validation",
                            "hf_repo_id": "mhlee1215/so101-nexus-sim-dataset",
                            "hf_repo_type": "dataset",
                            "hf_path_in_repo": "datasets/pick_cube/validation",
                        },
                        {
                            "name": "pick_place_validation",
                            "repo_id": "physical-ai-agent/source-b-validation",
                            "hf_repo_id": "mhlee1215/so101-nexus-sim-dataset",
                            "hf_repo_type": "dataset",
                            "hf_path_in_repo": "datasets/pick_and_place_cube/validation",
                        },
                    ],
                },
                "closed_loop": {
                    "eval_skill_mode": "pick_and_place_cube",
                    "task_prompt": "Pick up the small red cube and place it on the blue circle.",
                    "record_rollout_gif": True,
                },
            }

            with mock.patch.object(start_so101_training, "_snapshot_download") as snapshot_download:
                resolved = start_so101_training._resolve_hf_dataset_downloads(
                    config,
                    repo_root=repo_root,
                    cache_root=cache_root,
                    download=True,
                )
            resolved = start_so101_training._prepare_merged_train_dataset(
                resolved,
                repo_root=repo_root,
                python=Path(sys.executable),
                merge=False,
            )

            local_repo_dir = cache_root / "mhlee1215__so101-nexus-sim-dataset"
            merged_root = repo_root / "_workspace/merged/train"
            validation_merged_root = repo_root / "_workspace/merged/validation"
            self.assertEqual(snapshot_download.call_count, 4)
            self.assertEqual(resolved["train_dataset"]["root"], str(merged_root))
            self.assertEqual(resolved["validation_dataset"]["root"], str(validation_merged_root))
            self.assertEqual(len(resolved["train_dataset"]["hf_resolved_sources"]), 2)
            self.assertEqual(len(resolved["validation_dataset"]["hf_resolved_sources"]), 2)
            self.assertIn(str(local_repo_dir / "datasets/pick_cube/train"), resolved["train_dataset"]["merged_from"])
            self.assertIn(str(local_repo_dir / "datasets/pick_and_place_cube/train"), resolved["train_dataset"]["merged_from"])
            self.assertIn(
                str(local_repo_dir / "datasets/pick_cube/validation"),
                resolved["validation_dataset"]["merged_from"],
            )
            self.assertIn(
                str(local_repo_dir / "datasets/pick_and_place_cube/validation"),
                resolved["validation_dataset"]["merged_from"],
            )
            self.assertIn("--output-root", resolved["train_dataset"]["merge_command"])
            self.assertIn("--output-root", resolved["validation_dataset"]["merge_command"])

            train_args = start_so101_training._with_dataset_config([], resolved)
            self.assertIn(f"--dataset.root={merged_root}", train_args)
            self.assertIn(f"--validation-dataset-root={validation_merged_root}", train_args)

            resolved["execution_policy"] = "qwen_edge_chain"
            progress_cmd = start_so101_training._progress_monitor_command(
                args=argparse.Namespace(
                    python=Path(sys.executable),
                    progress_monitor_interval_s=600,
                    closed_loop_every_epochs=1,
                    closed_loop_episodes=1,
                    closed_loop_steps=90,
                    closed_loop_policy="best_or_periodic",
                    closed_loop_eval_skill_mode=None,
                    closed_loop_task_prompt=None,
                    closed_loop_record_rollout_gif=False,
                    closed_loop_runner="auto",
                    qwen_model="qwen3-vl-8b-instruct-mlx",
                    qwen_base_url=None,
                    qwen_api_key=None,
                    qwen_response_json=None,
                    qwen_plan_json=None,
                    qwen_object="green cube",
                    qwen_env_object_color=None,
                    closed_loop_env_id=None,
                    record_loop_artifacts=True,
                    render_loop_media=True,
                    loop_artifact_width=128,
                    loop_artifact_height=128,
                    loop_artifact_fps=12,
                    loop_artifact_every_n_steps=1,
                    closed_loop_subgoal_chain_mode="off",
                    closed_loop_fixed_subgoal_chunks=1,
                    closed_loop_valid_mask_threshold=0.5,
                    closed_loop_valid_mask_consecutive=2,
                    closed_loop_valid_mask_checkpoint=Path("/tmp/valid_mask_head.pt"),
                    closed_loop_policy_n_action_steps=15,
                    closed_loop_policy_num_steps=10,
                ),
                repo_root=repo_root,
                run_dir=repo_root / "run",
                train_output_dir=repo_root / "run/model",
                dataset_config=resolved,
                training_args=[],
                train_pid_file=repo_root / "run/train.pid",
                runtime_contract={
                    "runtime_platform": "macos",
                    "training_device": "mps",
                    "lightning_accelerator": "mps",
                    "closed_loop_device": "mps",
                    "closed_loop_mujoco_gl": "glfw",
                },
            )
            self.assertIn("--closed-loop-runner", progress_cmd)
            self.assertIn("qwen_chain", progress_cmd)
            self.assertIn("--record-loop-artifacts", progress_cmd)
            self.assertIn("--render-loop-media", progress_cmd)
            self.assertIn("--loop-artifact-width", progress_cmd)
            self.assertIn("--qwen-response-json", progress_cmd)
            self.assertIn("configs/agent/qwen3_so101_tool_planner_mock_response.json", progress_cmd)
            self.assertIn("--closed-loop-valid-mask-checkpoint", progress_cmd)
            self.assertIn("/tmp/valid_mask_head.pt", progress_cmd)
            self.assertIn("--policy-n-action-steps", progress_cmd)
            self.assertIn("15", progress_cmd)
            self.assertIn("--policy-num-steps", progress_cmd)
            self.assertIn("10", progress_cmd)

            resolved["closed_loop"]["action_contract_mode"] = "visual_servo_delta_q"
            visual_servo_cmd = start_so101_training._progress_monitor_command(
                args=argparse.Namespace(
                    python=Path(sys.executable),
                    progress_monitor_interval_s=600,
                    closed_loop_every_epochs=1,
                    closed_loop_episodes=1,
                    closed_loop_steps=90,
                    closed_loop_policy="best_or_periodic",
                    closed_loop_eval_skill_mode=None,
                    closed_loop_task_prompt=None,
                    closed_loop_record_rollout_gif=False,
                    closed_loop_runner="auto",
                    qwen_model="qwen3-vl-8b-instruct-mlx",
                    qwen_base_url=None,
                    qwen_api_key=None,
                    qwen_response_json=None,
                    qwen_plan_json=None,
                    qwen_object="green cube",
                    qwen_env_object_color=None,
                    closed_loop_env_id=None,
                    record_loop_artifacts=True,
                    render_loop_media=True,
                    loop_artifact_width=128,
                    loop_artifact_height=128,
                    loop_artifact_fps=12,
                    loop_artifact_every_n_steps=1,
                    closed_loop_subgoal_chain_mode="off",
                    closed_loop_fixed_subgoal_chunks=1,
                    closed_loop_valid_mask_threshold=0.5,
                    closed_loop_valid_mask_consecutive=2,
                    closed_loop_valid_mask_checkpoint=Path("/tmp/valid_mask_head.pt"),
                    closed_loop_policy_n_action_steps=15,
                    closed_loop_policy_num_steps=10,
                ),
                repo_root=repo_root,
                run_dir=repo_root / "run",
                train_output_dir=repo_root / "run/model",
                dataset_config=resolved,
                training_args=[],
                train_pid_file=repo_root / "run/train.pid",
                runtime_contract={
                    "runtime_platform": "macos",
                    "training_device": "mps",
                    "lightning_accelerator": "mps",
                    "closed_loop_device": "mps",
                    "closed_loop_mujoco_gl": "glfw",
                },
            )
            self.assertIn("--closed-loop-action-contract-mode", visual_servo_cmd)
            self.assertIn("visual_servo_delta_q", visual_servo_cmd)
            self.assertNotIn("--closed-loop-valid-mask-checkpoint", visual_servo_cmd)

    def test_qwen_edge_dataset_config_requires_valid_mask_checkpoint(self) -> None:
        config = json.loads(Path("configs/so101/training_datasets/qwen_edge_primitives.json").read_text(encoding="utf-8"))
        closed_loop = config["closed_loop"]

        self.assertEqual(config["execution_policy"], "qwen_edge_chain")
        self.assertEqual(closed_loop["execution_policy"], "qwen_edge_chain")
        self.assertEqual(closed_loop["env_id"], "MuJoCoPickLift-v1")
        self.assertEqual(closed_loop["qwen_object"], "green cube")
        self.assertEqual(closed_loop["env_object_color"], "green")
        self.assertIn("valid_mask_checkpoint", closed_loop)
        self.assertTrue(str(closed_loop["valid_mask_checkpoint"]).endswith("valid_mask_head.pt"))

    def test_move_and_align_reachable_bins_loop_test_excludes_top_row_bins(self) -> None:
        import pandas as pd

        config = json.loads(
            Path(
                "configs/so101/training_datasets/move_and_align_cube_edge_v2_delta_q_reachable_bins_mixed_start.json"
            ).read_text(encoding="utf-8")
        )
        test_case = config["closed_loop"]["test_cases"][0]
        root = Path(test_case["start_dataset"]["root"])
        sidecar = root / "meta/camera_grid_bins/observation_images_camera1_4x4_frame0.parquet"

        self.assertEqual(test_case["id"], "move_and_align_cube_edge_loop_validation_reachable_bins_5_14")
        self.assertTrue((root / "so101_lerobot_export_report.json").exists())
        self.assertTrue(sidecar.exists())
        bins = sorted(pd.read_parquet(sidecar)["grid_bin"].astype(int).unique().tolist())
        self.assertEqual(bins, list(range(5, 15)))
        self.assertTrue(set(config["reachable_bin_filter"]["excluded_bins"]).isdisjoint(bins))

    def test_move_and_align_reachable_bins_loop_command_uses_configured_5_14_start_dataset(self) -> None:
        _ensure_scripts_on_path()
        import start_so101_training

        config = json.loads(
            Path(
                "configs/so101/training_datasets/move_and_align_cube_edge_v2_delta_q_reachable_bins_mixed_start.json"
            ).read_text(encoding="utf-8")
        )
        test_case = config["closed_loop"]["test_cases"][0]
        base = [
            sys.executable,
            "scripts/run_so101_training_loop_test.py",
            "--closed-loop-test-id",
            "stale",
            "--closed-loop-start-report-path",
            "_workspace/so101_lerobot/move_and_align_cube_edge_loop_validation10_ego_wrist_256_seed123500/so101_lerobot_export_report.json",
        ]

        commands = start_so101_training._post_checkpoint_loop_commands(
            progress_monitor_cmd=base,
            dataset_config=config,
        )

        self.assertEqual(len(commands), 1)
        command = commands[0]
        expected_report = str(Path(test_case["start_dataset"]["root"]) / "so101_lerobot_export_report.json")
        self.assertEqual(command[command.index("--closed-loop-test-id") + 1], test_case["id"])
        self.assertEqual(command[command.index("--closed-loop-start-report-path") + 1], expected_report)
        self.assertNotIn("seed123500", " ".join(command))

    def test_launcher_rejects_closed_loop_command_mismatched_to_test_case_start_dataset(self) -> None:
        _ensure_scripts_on_path()
        import start_so101_training

        config = json.loads(
            Path(
                "configs/so101/training_datasets/move_and_align_cube_edge_v2_delta_q_reachable_bins_mixed_start.json"
            ).read_text(encoding="utf-8")
        )
        errors: list[str] = []
        start_so101_training._validate_closed_loop_test_case_commands(
            dataset_config=config,
            post_checkpoint_loop_cmds=[
                [
                    sys.executable,
                    "scripts/run_so101_training_loop_test.py",
                    "--closed-loop-test-id",
                    "move_and_align_cube_edge_loop_validation_reachable_bins_5_14",
                    "--closed-loop-start-report-path",
                    "_workspace/so101_lerobot/move_and_align_cube_edge_loop_validation10_ego_wrist_256_seed123500/so101_lerobot_export_report.json",
                ]
            ],
            errors=errors,
        )

        self.assertTrue(errors)
        self.assertIn("start report mismatch", errors[0])

    def test_move_and_align_debug_loop_uses_supervised_train_split(self) -> None:
        _ensure_scripts_on_path()
        import start_so101_training

        config = json.loads(
            Path("configs/so101/training_datasets/move_and_align_cube_edge_only.json").read_text(encoding="utf-8")
        )
        train = config["train_dataset"]
        test_cases = config["closed_loop"]["test_cases"]

        self.assertEqual(len(test_cases), 1)
        test_case = test_cases[0]
        self.assertEqual(test_case["id"], "move_and_align_cube_edge_train_aligned_debug")
        self.assertEqual(test_case["episodes"], 10)
        self.assertEqual(test_case["start_contract"], "move_and_align_cube_edge")
        self.assertIn("Debug-only", test_case["description"])
        self.assertEqual(test_case["start_dataset"]["name"], train["name"])
        self.assertEqual(test_case["start_dataset"]["root"], train["root"])
        self.assertEqual(test_case["start_dataset"]["repo_id"], train["repo_id"])
        self.assertEqual(test_case["start_dataset"]["expected_episodes"], train["expected_episodes"])

        base = [
            sys.executable,
            "scripts/run_so101_training_loop_test.py",
            "--closed-loop-test-id",
            "default",
            "--closed-loop-seed",
            "98100",
            "--closed-loop-steps",
            "160",
        ]
        commands = start_so101_training._post_checkpoint_loop_commands(
            progress_monitor_cmd=base,
            dataset_config=config,
        )

        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command[command.index("--closed-loop-test-id") + 1], test_case["id"])
        self.assertEqual(command[command.index("--closed-loop-start-contract") + 1], "move_and_align_cube_edge")
        self.assertEqual(
            command[command.index("--closed-loop-start-report-path") + 1],
            str(Path(train["root"]) / "so101_lerobot_export_report.json"),
        )

    def test_move_and_align_v2_dataset_generation_augmentation_contract(self) -> None:
        exporter_source = Path("scripts/export_so101_teacher_rollouts_lerobot.py").read_text(encoding="utf-8")
        exporter_constants = {
            node.value
            for node in ast.walk(ast.parse(exporter_source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("--terminal-hold-steps", exporter_constants)
        self.assertIn("--move-and-align-near-target-correction-ratio", exporter_constants)
        self.assertIn("near_target_correction", exporter_constants)

        recipes = json.loads(Path("configs/so101/training_datasets/export_recipes.json").read_text(encoding="utf-8"))
        recipe = next(item for item in recipes["recipes"] if item["name"] == "move_and_align_cube_edge_train_v2")
        self.assertEqual(recipe["contract"], "skill_dataset_contract")
        self.assertEqual(recipe["dataset"], "move_and_align_cube_edge_v2")
        self.assertEqual(recipe["split"], "train")
        self.assertEqual(recipe["episodes"], 300)
        self.assertEqual(recipe["args"]["skill_mode"], "move_and_align_cube_edge")
        self.assertEqual(recipe["args"]["terminal_hold_steps"], 20)
        self.assertEqual(recipe["args"]["move_and_align_near_target_correction_ratio"], 0.5)
        self.assertEqual(recipe["args"]["target_object_color"], "green")

        contract = json.loads(Path("configs/so101/training_datasets/skill_dataset_contract.json").read_text(encoding="utf-8"))
        v2 = contract["datasets"]["move_and_align_cube_edge_v2"]
        self.assertEqual(v2["train"]["root"], recipe["root"])
        self.assertEqual(v2["train"]["repo_id"], recipe["repo_id"])
        self.assertEqual(v2["train"]["expected_episodes"], recipe["episodes"])
        self.assertTrue(v2["generation_augmentation"]["terminal_hold_included"])
        self.assertTrue(v2["generation_augmentation"]["near_target_correction_included"])

        config = json.loads(
            Path("configs/so101/training_datasets/move_and_align_cube_edge_v2_only.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["train_dataset"]["root"], recipe["root"])
        self.assertEqual(config["train_dataset"]["expected_episodes"], 300)
        self.assertEqual(config["dataset_generation_augmentation"]["terminal_hold_steps"], 20)
        self.assertEqual(config["dataset_generation_augmentation"]["near_target_correction_ratio"], 0.5)

    def test_qwen_edge_loop_validation_cases_match_primitive_dataset_names(self) -> None:
        config = json.loads(Path("configs/so101/training_datasets/qwen_edge_primitives.json").read_text(encoding="utf-8"))
        self.assertIn("test_cases", config["closed_loop"])
        self.assertNotIn("suites", config["closed_loop"])
        test_cases = config["closed_loop"]["test_cases"]

        self.assertEqual(
            [test_case["id"] for test_case in test_cases],
            ["move_over_cube_edge", "align_fixed_jaw_cube_edge", "grip_from_edge_cube"],
        )
        expected_calls = {
            "move_over_cube_edge": ["move"],
            "align_fixed_jaw_cube_edge": ["align"],
            "grip_from_edge_cube": ["pick_up"],
        }
        expected_closed_loop_roots = {
            "move_over_cube_edge": "_workspace/so101_lerobot/move_over_cube_edge_loop_validation10_ego_wrist_256_seed116500",
            "align_fixed_jaw_cube_edge": "_workspace/so101_lerobot/align_fixed_jaw_cube_edge_loop_validation10_ego_wrist_256_seed118500",
            "grip_from_edge_cube": "_workspace/so101_lerobot/grip_from_edge_cube_loop_validation10_ego_wrist_256_seed121000",
        }
        for test_case in test_cases:
            with self.subTest(test_case=test_case["id"]):
                self.assertEqual(test_case["episodes"], 10)
                self.assertIn("seed", test_case)
                self.assertEqual(test_case["start_contract"], test_case["id"])
                self.assertIn("loop_validation split", test_case["description"])
                self.assertEqual(test_case["qwen_object"], "green cube")
                self.assertEqual(test_case["env_object_color"], "green")
                self.assertIn("start_dataset", test_case)
                self.assertEqual(test_case["start_dataset"]["name"], f"{test_case['id']}_loop_validation")
                self.assertEqual(test_case["start_dataset"]["expected_episodes"], 10)
                self.assertEqual(test_case["start_dataset"]["root"], expected_closed_loop_roots[test_case["id"]])
                self.assertTrue(test_case["start_dataset"]["repo_id"].endswith("loop-validation10-ego-wrist-256"))
                plan_path = Path(test_case["plan_json"])
                self.assertTrue(plan_path.exists(), str(plan_path))
                plan = json.loads(plan_path.read_text(encoding="utf-8"))["plan"]
                self.assertEqual([call["fn"] for call in plan["calls"]], expected_calls[test_case["id"]])
                self.assertTrue(all(call["object"] == "green cube" for call in plan["calls"]))
                self.assertNotIn("precondition_plan_json", test_case)
                self.assertEqual(len(plan["calls"]), 1)
                self.assertEqual(plan["task"], test_case["task_prompt"])
                self.assertEqual(plan["calls"][0]["primitive_id"], test_case["id"])
                self.assertEqual(plan["calls"][0]["prompt"], test_case["task_prompt"])
        self.assertEqual([test_case["seed"] for test_case in test_cases], [98100, 98200, 98300])
        self.assertEqual(
            [test_case["start_contract"] for test_case in test_cases],
            ["move_over_cube_edge", "align_fixed_jaw_cube_edge", "grip_from_edge_cube"],
        )

    def test_qwen_edge_loop_validation_case_commands_use_case_seed_and_start_contract(self) -> None:
        _ensure_scripts_on_path()
        import start_so101_training

        config = json.loads(Path("configs/so101/training_datasets/qwen_edge_primitives.json").read_text(encoding="utf-8"))
        base = [
            sys.executable,
            "scripts/run_so101_training_loop_test.py",
            "--closed-loop-test-id",
            "default",
            "--closed-loop-seed",
            "98100",
            "--closed-loop-steps",
            "160",
        ]
        commands = start_so101_training._post_checkpoint_loop_commands(
            progress_monitor_cmd=base,
            dataset_config=config,
        )

        self.assertEqual(len(commands), 3)
        for command, test_case in zip(commands, config["closed_loop"]["test_cases"]):
            with self.subTest(test_case=test_case["id"]):
                self.assertEqual(command[command.index("--closed-loop-test-id") + 1], test_case["id"])
                self.assertEqual(command[command.index("--closed-loop-seed") + 1], str(test_case["seed"]))
                self.assertEqual(
                    command[command.index("--closed-loop-start-contract") + 1],
                    test_case["start_contract"],
                )
                self.assertIn("--closed-loop-start-report-path", command)
                self.assertEqual(
                    command[command.index("--closed-loop-start-report-path") + 1],
                    str(Path(test_case["start_dataset"]["root"]) / "so101_lerobot_export_report.json"),
                )
                self.assertEqual(command[command.index("--qwen-env-object-color") + 1], "green")
                self.assertNotIn("--closed-loop-precondition-plan-json", command)

    def test_qwen_edge_primitives_training_uses_virtual_merge_only(self) -> None:
        _ensure_scripts_on_path()
        import start_so101_training

        config = json.loads(Path("configs/so101/training_datasets/qwen_edge_primitives.json").read_text(encoding="utf-8"))

        self.assertIn("train_datasets", config)
        self.assertNotIn("train_dataset", config)
        prepared = start_so101_training._prepare_merged_train_dataset(
            config,
            repo_root=Path.cwd(),
            python=Path(sys.executable),
            merge=False,
        )
        train_args = start_so101_training._with_dataset_config([], prepared)

        self.assertTrue(any(arg.startswith("--train-datasets-json=") for arg in train_args))
        self.assertFalse(any("merge_so101_lerobot_shards.py" in str(value) for value in prepared.values()))
        self.assertFalse(any("--output-root" in str(value) for value in prepared.values()))
        self.assertNotIn("merge_command", json.dumps(prepared, sort_keys=True))

    def test_training_harness_logs_source_dataset_losses_and_loop_test_case_metrics(self) -> None:
        lightning_source = Path("scripts/lerobot_train_so101_lightning.py").read_text(encoding="utf-8")
        launcher_source = Path("scripts/start_so101_training.py").read_text(encoding="utf-8")
        monitor_source = Path("scripts/monitor_so101_training_dashboard.py").read_text(encoding="utf-8")

        self.assertIn("--train-dataset-source-spans-json", lightning_source)
        self.assertIn("train/datasets/", lightning_source)
        self.assertIn("val/datasets/", lightning_source)
        self.assertIn("--validation-datasets-json", lightning_source)
        self.assertIn("train-dataset-source-spans-json", launcher_source)
        self.assertIn("_post_checkpoint_loop_commands", launcher_source)
        self.assertIn("_loop_validation_cases", launcher_source)
        self.assertIn("--closed-loop-test-id", launcher_source)
        self.assertIn("--closed-loop-start-contract", launcher_source)
        self.assertIn("IMPORTANT_CLOSED_LOOP_SUCCESS_RATE_TAG", monitor_source)
        self.assertIn("loop_validation_id", monitor_source)

    def test_validation_loss_eval_reports_postprocessed_action_rmse(self) -> None:
        eval_source = Path("scripts/evaluate_smolvla_supervised_loss.py").read_text(encoding="utf-8")
        monitor_source = Path("scripts/monitor_so101_training_dashboard.py").read_text(encoding="utf-8")

        self.assertIn("postprocessed_action_rmse_mean", eval_source)
        self.assertIn("postprocessed_action_global_rmse", eval_source)
        self.assertIn("policy.predict_action_chunk(batch)", eval_source)
        self.assertIn("_postprocess_action_chunk", eval_source)
        self.assertIn("val/postprocessed_action_rmse", monitor_source)
        self.assertIn("IMPORTANT_VAL_POSTPROCESSED_ACTION_RMSE_TAG", monitor_source)

    def test_closed_loop_harness_attaches_action_rmse_sweep_plot_to_tensorboard(self) -> None:
        config = json.loads(
            Path("configs/so101/training_datasets/qwen_edge_grip_from_above_edge_cube_only.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(config["closed_loop"]["action_rmse_sweep"]["enabled"])
        self.assertEqual(config["closed_loop"]["action_rmse_sweep"]["n_action_steps"], [1, 3, 5, 10, 15, 30, 40, 50])

        launcher_source = Path("scripts/start_so101_training.py").read_text(encoding="utf-8")
        monitor_source = Path("scripts/monitor_so101_training_dashboard.py").read_text(encoding="utf-8")
        self.assertIn("--closed-loop-action-rmse-sweep-n-action-steps", launcher_source)
        self.assertIn("_attach_action_rmse_sweep", monitor_source)
        self.assertIn("closed_loop/{test_id}/action_rmse_sweep", monitor_source)
        self.assertIn("_plot_action_rmse_sweep", monitor_source)

    def test_delta_q_training_config_preserves_absolute_dataset_and_forwards_closed_loop_contract(self) -> None:
        config = json.loads(
            Path("configs/so101/training_datasets/move_and_align_cube_edge_v2_delta_q_only.json").read_text(
                encoding="utf-8"
            )
        )
        absolute_config = json.loads(
            Path("configs/so101/training_datasets/move_and_align_cube_edge_v2_only.json").read_text(encoding="utf-8")
        )

        self.assertEqual(config["action_mode"], "delta_q")
        self.assertEqual(config["closed_loop"]["action_contract_mode"], "processor_delta_q")
        self.assertNotEqual(config["train_dataset"]["root"], absolute_config["train_dataset"]["root"])
        self.assertNotEqual(config["validation_dataset"]["root"], absolute_config["validation_dataset"]["root"])
        self.assertIn("convert_so101_lerobot_actions_to_delta.py", config["delta_action_source"]["converter"])

        monitor_source = Path("scripts/monitor_so101_training_dashboard.py").read_text(encoding="utf-8")
        self.assertIn("--closed-loop-action-contract-mode", monitor_source)
        self.assertIn("--action-contract-mode", monitor_source)
        self.assertIn("processor_delta_q", monitor_source)

    def test_so101_training_configs_default_to_moderate_augmentation_without_action_dropout(self) -> None:
        for config_path in (
            Path("configs/so101/training_datasets/pick.json"),
            Path("configs/so101/training_datasets/pick_place.json"),
            Path("configs/so101/training_datasets/move_over_cube.json"),
            Path("configs/so101/training_datasets/pick_from_top_cube.json"),
            Path("configs/so101/training_datasets/all_hf_train_pick_place_closed_loop.json"),
            Path("configs/so101/training_datasets/qwen_edge_primitives.json"),
        ):
            with self.subTest(config=str(config_path)):
                config = json.loads(config_path.read_text(encoding="utf-8"))
                augmentation = config["augmentation"]
                self.assertEqual(augmentation["state_jitter_std"], 0.003)
                self.assertEqual(augmentation["state_dropout_prob"], 0.02)
                self.assertEqual(augmentation["image_patch_mask_ratio"], 0.15)
                self.assertTrue(augmentation["image_color_jitter"])
                self.assertTrue(augmentation["image_sharpness_jitter"])
                self.assertEqual(augmentation["image_affine_degrees"], 5.0)
                self.assertEqual(augmentation["image_affine_translate"], 0.05)
                self.assertTrue(augmentation["gpu_image_augmentation"])
                self.assertEqual(augmentation["image_camera_dropout_prob"], 0.0)
                self.assertEqual(augmentation["image_patch_dropout_prob"], 0.0)
                self.assertNotIn("action_dropout_prob", augmentation)

    def test_training_launcher_forces_zero_workers_for_local_macos(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        import start_so101_training

        config = {
            "train_dataset": {
                "repo_id": "physical-ai-agent/train",
                "root": "_workspace/train",
            },
            "training": {
                "num_workers": 4,
                "batch_size": 32,
            },
        }

        macos_args = start_so101_training._with_dataset_config([], config, runtime_platform="macos")
        linux_args = start_so101_training._with_dataset_config([], config, runtime_platform="linux")

        self.assertIn("--num_workers=0", macos_args)
        self.assertIn("--num_workers=4", linux_args)

    def test_so101_dataset_configs_use_approved_egocentric_camera1(self) -> None:
        for config_path in (
            Path("configs/so101/training_datasets/pick.json"),
            Path("configs/so101/training_datasets/pick_place.json"),
            Path("configs/so101/training_datasets/move_over_cube.json"),
            Path("configs/so101/training_datasets/pick_from_top_cube.json"),
            Path("configs/so101/training_datasets/all_hf_train_pick_place_closed_loop.json"),
        ):
            with self.subTest(config=str(config_path)):
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(config["camera_contract"]["observation.images.camera1"], "egocentric_cam")
                self.assertEqual(config["camera_contract"]["observation.images.camera2"], "wrist_cam")
                train_specs = config.get("train_datasets") or [config["train_dataset"]]
                for train_spec in train_specs:
                    self.assertNotIn("top-wrist", train_spec["repo_id"])
                    self.assertNotIn("top_wrist", train_spec["root"])

    def test_so101_egocentric_camera1_pose_is_single_source_of_truth(self) -> None:
        expected_pose = {
            "type": "free",
            "lookat": [0.245, 0.11, 0.035],
            "distance": 0.63,
            "azimuth": 270,
            "elevation": -82,
            "rotation_degrees": 90,
        }
        self.assertEqual(EGOCENTRIC_CAMERA1_POSE, expected_pose)

        for contract_path in (
            Path("configs/so101/training_datasets/dataset_contract.json"),
            Path("configs/so101/training_datasets/skill_dataset_contract.json"),
        ):
            with self.subTest(contract=str(contract_path)):
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                pose = contract["policy"]["camera_pose_contract"]["observation.images.camera1"]["pose"]
                self.assertEqual(pose, expected_pose)

        docs = "\n".join(
            Path(path).read_text(encoding="utf-8")
            for path in (
                "docs/so101_camera_contract.md",
                "docs/harness/physical-ai/team-spec.md",
                "Summary.md",
            )
        )
        self.assertIn('"lookat": [0.245, 0.11, 0.035]', docs)
        self.assertIn('"distance": 0.63', docs)
        self.assertIn('"elevation": -82', docs)

    def test_so101_export_recipes_cover_all_contract_datasets(self) -> None:
        recipes = json.loads(
            Path("configs/so101/training_datasets/export_recipes.json").read_text(encoding="utf-8")
        )
        recipe_keys = {
            (recipe["contract"], recipe["dataset"], recipe["split"])
            for recipe in recipes["recipes"]
        }
        expected_keys = set()
        for contract_name, contract_path in (
            ("dataset_contract", Path("configs/so101/training_datasets/dataset_contract.json")),
            ("skill_dataset_contract", Path("configs/so101/training_datasets/skill_dataset_contract.json")),
        ):
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            for dataset_name, dataset in contract["datasets"].items():
                for split_name, split in dataset.items():
                    if isinstance(split, dict) and split.get("repo_id") and split.get("root"):
                        expected_keys.add((contract_name, dataset_name, split_name))

        self.assertEqual(recipe_keys, expected_keys)
        self.assertEqual(recipes["camera_pose_source"], "physical_ai_agent.sim.so101_camera_input.EGOCENTRIC_CAMERA1_POSE")
        self.assertEqual(recipes["defaults"]["width"], 256)
        self.assertEqual(recipes["defaults"]["height"], 256)

        for recipe in recipes["recipes"]:
            with self.subTest(recipe=recipe["name"]):
                self.assertRegex(recipe["root"], r"(ego_wrist_256|true256)")
                self.assertRegex(recipe["repo_id"], r"(ego-wrist-256|true256)")
                self.assertIn(recipe["script"], {
                    "scripts/export_so101_teacher_rollouts_lerobot.py",
                    "scripts/export_so101_pickplace_teacher_rollouts_lerobot.py",
                })

    def test_so101_export_recipe_dry_run_builds_expected_commands(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/export_so101_training_datasets.py",
                "--dry-run",
                "--overwrite",
                "--only",
                "move_over_cube_train",
            ],
            check=False,
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONPATH": "src"},
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        command = payload["commands"][0]
        self.assertIn("scripts/export_so101_teacher_rollouts_lerobot.py", command)
        self.assertIn("--overwrite", command)
        self.assertIn("--width", command)
        self.assertIn("256", command)
        self.assertIn("--skill-mode", command)
        self.assertIn("move_over_cube", command)

    def test_so101_harness_documents_augmentation_and_smoothness_contract(self) -> None:
        docs = "\n".join(
            Path(path).read_text(encoding="utf-8")
            for path in (
                "Summary.md",
                "docs/harness/physical-ai/team-spec.md",
                "docs/so101_smolvla_training_pipeline.md",
                "configs/so101/training_datasets/README.md",
                ".agents/skills/physical-ai-orchestrator/SKILL.md",
            )
        )

        self.assertIn("state_jitter_std=0.003", docs)
        self.assertIn("state_dropout_prob=0.02", docs)
        self.assertIn("image_patch_mask_ratio=0.15", docs)
        self.assertIn("image_affine_degrees=5.0", docs)
        self.assertIn("image_affine_translate=0.05", docs)
        self.assertIn("gpu_image_augmentation=true", docs)
        self.assertIn("CUDA and MPS", docs)
        self.assertIn("Validation and closed-loop test", docs)
        self.assertIn("teacher-action dropout", docs)
        self.assertIn("temporal smoothness loss", docs)
        self.assertIn("temporal ensembling", docs)

    def test_so101_training_dataset_configs_match_checksum_manifest(self) -> None:
        checksum_path = Path("configs/so101/training_datasets/checksums.json")
        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
        self.assertEqual(checksums["algorithm"], "sha256")
        contracts = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in (
                "configs/so101/training_datasets/dataset_contract.json",
                "configs/so101/training_datasets/skill_dataset_contract.json",
            )
        ]

        expected_checksum_keys = set()
        for contract in contracts:
            for dataset_name, dataset_spec in contract["datasets"].items():
                for split_name, suffix in (
                    ("train", "train"),
                    ("validation", "val"),
                    ("loop_validation", "loop_validation"),
                ):
                    if split_name not in dataset_spec:
                        continue
                    checksum_key = f"{dataset_name}_{suffix}"
                    expected_checksum_keys.add(checksum_key)
                    dataset = checksums["datasets"][checksum_key]
                    split = dataset_spec[split_name]
                    self.assertEqual(split["repo_id"], dataset["repo_id"])
                    self.assertEqual(split["root"], dataset["root"])
                    self.assertGreater(dataset["episodes"], 0)
                    self.assertGreater(dataset["frames"], dataset["episodes"])
                    self.assertGreater(dataset["size_bytes"], 0)
                    self.assertRegex(dataset["directory_sha256"], r"^[0-9a-f]{64}$")
                    self.assertRegex(dataset["export_report_sha256"], r"^[0-9a-f]{64}$")
                    self.assertRegex(dataset["audit_sha256"], r"^[0-9a-f]{64}$")
                    self.assertEqual(
                        dataset["camera_contract"],
                        {
                            "observation.images.camera1": "egocentric_cam",
                            "observation.images.camera2": "wrist_cam",
                            "observation.images.camera3": "wrist_cam duplicate",
                        },
                    )
                    self.assertEqual(
                        dataset["camera_pose_contract"]["observation.images.camera1"],
                        EGOCENTRIC_CAMERA1_POSE,
                    )
                    self.assertEqual(dataset["sample_shapes"]["observation.images.camera1"], [3, 256, 256])
                    self.assertEqual(dataset["sample_shapes"]["observation.images.camera2"], [3, 256, 256])
        self.assertEqual(set(checksums["datasets"]), expected_checksum_keys)

    def test_so101_dataset_contract_opens_every_required_split(self) -> None:
        import pyarrow.parquet as pq

        contract_path = Path("configs/so101/training_datasets/dataset_contract.json")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        camera_contract = contract["policy"]["camera_contract"]
        image_shape_chw = contract["policy"]["image_shape_chw"]

        self.assertEqual(
            camera_contract,
            {
                "observation.images.camera1": "egocentric_cam",
                "observation.images.camera2": "wrist_cam",
                "observation.images.camera3": "wrist_cam duplicate",
            },
        )

        for dataset_name, dataset_spec in contract["datasets"].items():
            for split_name in ("train", "validation"):
                with self.subTest(dataset=dataset_name, split=split_name):
                    split = dataset_spec[split_name]
                    root = Path(split["root"])
                    report_path = root / "so101_lerobot_export_report.json"
                    audit_path = root / "so101_lerobot_audit.json"
                    self.assertTrue(report_path.exists(), f"missing dataset report: {report_path}")
                    self.assertTrue(audit_path.exists(), f"missing dataset audit: {audit_path}")

                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    audit = report["audit"]
                    self.assertEqual(report["repo_id"], split["repo_id"])
                    self.assertEqual(report["exported_episodes"], split["expected_episodes"])
                    self.assertEqual(audit["status"], "passed")
                    self.assertEqual(audit["num_episodes"], split["expected_episodes"])
                    self.assertGreater(audit["dataset_len"], split["expected_episodes"])
                    self.assertEqual(report["feature_mapping"], {**camera_contract, **{
                        "observation.state": "SO101 qpos/control state",
                        "action": "SO101 qpos target action",
                        "task": report["task"],
                    }})
                    self.assertEqual(
                        audit["sample_shapes"]["observation.images.camera1"],
                        image_shape_chw,
                    )
                    self.assertEqual(
                        audit["sample_shapes"]["observation.images.camera2"],
                        image_shape_chw,
                    )

                    exported_episodes = [episode for episode in report["episodes"] if episode.get("success")]
                    self.assertEqual(len(exported_episodes), split["expected_episodes"])

                    start_modes = {episode["start_mode"] for episode in exported_episodes}
                    self.assertEqual(start_modes, {dataset_spec["start_mode"]})
                    if dataset_spec["task"] == "pick":
                        self.assertIn("cube", report["task"].lower())
                        self.assertTrue(all(ep["final_info"]["is_grasped"] for ep in exported_episodes))
                        self.assertTrue(all(ep["final_info"]["lift_height"] > 0.035 for ep in exported_episodes))
                    else:
                        self.assertIn("place", report["task"].lower())
                        self.assertTrue(all(ep["object_shape"]["name"] == "cube_small" for ep in exported_episodes))
                        self.assertTrue(all(ep["final_info"]["is_obj_placed"] for ep in exported_episodes))

                    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
                    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
                    self.assertTrue(data_files, f"missing parquet data files under {root}")
                    self.assertTrue(episode_files, f"missing episode parquet files under {root}")
                    row = pq.read_table([str(data_files[0])]).slice(0, 1).to_pydict()
                    self.assertIn("observation.images.camera1", row)
                    self.assertIn("observation.images.camera2", row)
                    self.assertIn("observation.state", row)
                    self.assertIn("action", row)
                    self.assertIn("task_index", row)
                    self.assertTrue(row["observation.images.camera1"][0]["bytes"])
                    self.assertTrue(row["observation.images.camera2"][0]["bytes"])
                    self.assertEqual(len(row["observation.state"][0]), 6)
                    self.assertEqual(len(row["action"][0]), 6)


def _write_lerobot_info(root: Path, *, camera1_shape: list[int]) -> None:
    (root / "meta").mkdir(parents=True)
    features = {
        "observation.images.camera1": {"dtype": "image", "shape": camera1_shape, "names": ["height", "width", "channels"]},
        "observation.images.camera2": {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channels"]},
        "observation.images.camera3": {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": [6], "names": ["joint"]},
        "action": {"dtype": "float32", "shape": [6], "names": ["joint"]},
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "total_frames": 2,
                "fps": 12,
                "features": features,
            }
        ),
        encoding="utf-8",
    )
