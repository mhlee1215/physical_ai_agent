from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import ast
from pathlib import Path
from unittest import TestCase

from physical_ai_agent.so101_smolvla_pipeline import (
    SO101DatasetManifest,
    SO101TrainingSchedule,
    SmolVLASO101Contract,
    detect_overfit_stop,
    should_run_closed_loop,
    validate_smolvla_train_config,
)


class SO101SmolVLAPipelineTest(TestCase):
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

    def test_pickplace_exporter_uses_top_wrist_student_camera_contract(self) -> None:
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
        self.assertIn("top_down", frame_constants)
        self.assertIn("observation.images.camera1", frame_constants)
        self.assertIn("observation.images.camera2", frame_constants)
        self.assertIn("observation.images.camera3", frame_constants)
        self.assertIn("observation.images.camera1", constants)
        self.assertIn("observation.images.wrist_cam", constants)
        self.assertIn("observation.images.top", constants)
        self.assertIn("wrist_cam duplicate", constants)
        self.assertNotIn("egocentric_cam duplicate", constants)
        self.assertIn('"observation.images.camera1": top', source)
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
        self.assertIn("--log-input-images-every-n-steps", constants)
        self.assertIn("--log-input-metadata-every-n-steps", constants)
        self.assertIn("--so101-action-prefix-loss-steps", constants)
        self.assertIn("--so101-action-prefix-loss-weight", constants)
        self.assertIn("loss_unweighted", constants)
        self.assertIn("loss_prefix_weight", constants)
        self.assertIn("loss_prefix_steps", constants)
        self.assertIn("camera1,camera2", constants)
        self.assertIn("/input_", constants)
        self.assertIn("/input_prompt", constants)
        self.assertIn("/input_motor_state", constants)
        self.assertIn("/input_camera_contract", constants)
        self.assertIn("observation.state", constants)
        self.assertIn("val/loss", constants)
        self.assertIn("train_vs_val", constants)
        self.assertIn("Multiline", constants)
        self.assertIn("TensorBoardLogger", names)
        self.assertIn("Trainer", names)
        self.assertIn("save_checkpoint", names)
        self.assertIn("augment_batch_on_device", names)

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
        self.assertIn(
            "Refusing to start: an SO101 training run is already active. "
            "Use `status`, `stop`, or `start --replace`.",
            constants,
        )
        self.assertIn("--tensorboard-log-dir", constants)
        self.assertIn("--dataset-config", constants)
        self.assertIn("--validation-interval-steps", constants)
        self.assertIn("--validation-interval-epochs", constants)

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
            self.assertIn("dashboard_cmd", payload)

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
                        "tensorboard": {
                            "log_input_images_every_n_steps": 10,
                            "log_input_metadata_every_n_steps": 10,
                        },
                        "augmentation": {
                            "state_jitter_std": 0.01,
                            "state_jitter_arm_only": False,
                            "state_dropout_prob": 0.02,
                            "state_dropout_keep_gripper": True,
                            "action_dropout_prob": 0.03,
                            "image_camera_dropout_prob": 0.04,
                            "image_patch_dropout_prob": 0.05,
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
            self.assertIn("--dataset.repo_id=physical-ai-agent/train", train_cmd)
            self.assertIn("--dataset.root=_workspace/train", train_cmd)
            self.assertIn("--validation-dataset-repo-id=physical-ai-agent/val", train_cmd)
            self.assertIn("--validation-dataset-root=_workspace/val", train_cmd)
            self.assertIn("--log-input-images-every-n-steps=10", train_cmd)
            self.assertIn("--log-input-metadata-every-n-steps=10", train_cmd)
            self.assertIn("--so101-state-jitter-std=0.01", train_cmd)
            self.assertIn("--no-so101-state-jitter-arm-only", train_cmd)
            self.assertIn("--so101-state-dropout-prob=0.02", train_cmd)
            self.assertIn("--so101-state-dropout-keep-gripper", train_cmd)
            self.assertIn("--so101-action-dropout-prob=0.03", train_cmd)
            self.assertIn("--so101-image-camera-dropout-prob=0.04", train_cmd)
            self.assertIn("--so101-image-patch-dropout-prob=0.05", train_cmd)
            self.assertIn("--so101-gpu-image-augmentation", train_cmd)
            self.assertEqual(payload["dataset_config"]["train_dataset"]["repo_id"], "physical-ai-agent/train")

    def test_so101_training_dataset_configs_match_checksum_manifest(self) -> None:
        checksum_path = Path("configs/so101/training_datasets/checksums.json")
        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
        self.assertEqual(checksums["algorithm"], "sha256")

        expected = {
            "pick": ("pick_train", "pick_val"),
            "pick_place": ("pick_place_train", "pick_place_val"),
        }
        for config_name, (train_key, val_key) in expected.items():
            config = json.loads(
                Path(f"configs/so101/training_datasets/{config_name}.json").read_text(encoding="utf-8")
            )
            for split_name, checksum_key in (("train_dataset", train_key), ("validation_dataset", val_key)):
                dataset = checksums["datasets"][checksum_key]
                self.assertEqual(config[split_name]["repo_id"], dataset["repo_id"])
                self.assertEqual(config[split_name]["root"], dataset["root"])
                self.assertGreater(dataset["episodes"], 0)
                self.assertGreater(dataset["frames"], dataset["episodes"])
                self.assertGreater(dataset["size_bytes"], 0)
                self.assertRegex(dataset["directory_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(dataset["export_report_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(dataset["audit_sha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(
                    dataset["camera_contract"],
                    {
                        "observation.images.camera1": "top_down",
                        "observation.images.camera2": "wrist_cam",
                        "observation.images.camera3": "wrist_cam duplicate",
                    },
                )
                self.assertEqual(dataset["sample_shapes"]["observation.images.camera1"], [3, 256, 256])
                self.assertEqual(dataset["sample_shapes"]["observation.images.camera2"], [3, 256, 256])
