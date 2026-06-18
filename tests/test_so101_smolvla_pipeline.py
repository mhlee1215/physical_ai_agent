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
        self.assertIn("important/loss", constants)
        self.assertIn("important_train_loss", constants)
        self.assertIn("important_val_loss", constants)
        self.assertIn("train_vs_val", constants)
        self.assertIn("action_chunk_smoothness", constants)
        self.assertIn("teacher_vs_predicted_jitter", constants)
        self.assertIn("predicted_to_teacher_ratio", constants)
        self.assertIn("val/action_jitter/", constants)
        self.assertIn("path_to_endpoint_ratio_mean", constants)
        self.assertIn("Multiline", constants)
        self.assertIn("TensorBoardLogger", names)
        self.assertIn("Trainer", names)
        self.assertIn("save_checkpoint", names)
        self.assertIn("augment_batch_on_device", names)
        self.assertIn("_action_chunk_jitter_metrics", names)

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
            self.assertIn("gpu_monitor_cmd", payload)
            self.assertTrue(
                any("log_gpu_metrics_tensorboard.py" in part for part in payload["gpu_monitor_cmd"])
            )

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
                        "augmentation": {
                            "state_jitter_std": 0.01,
                            "state_jitter_arm_only": False,
                            "state_dropout_prob": 0.02,
                            "state_dropout_keep_gripper": True,
                            "image_camera_dropout_prob": 0.04,
                            "image_patch_dropout_prob": 0.05,
                            "image_patch_mask_ratio": 0.15,
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
                env={**os.environ, "PYTHONPATH": "src", "SO101_TEST_CACHE_ROOT": "/tmp/so101-cache"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            train_cmd = payload["train_cmd"]
            self.assertIn("--dataset.repo_id=physical-ai-agent/train", train_cmd)
            self.assertIn("--dataset.root=_workspace/train", train_cmd)
            self.assertIn("--validation-dataset-repo-id=physical-ai-agent/val", train_cmd)
            self.assertIn("--validation-dataset-root=_workspace/val", train_cmd)
            self.assertIn("--num_workers=4", train_cmd)
            self.assertIn("--so101-image-cache-dir=/tmp/so101-cache/train", train_cmd)
            self.assertIn("--validation-image-cache-dir=/tmp/so101-cache/val", train_cmd)
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
            self.assertIn("--so101-state-jitter-std=0.01", train_cmd)
            self.assertIn("--no-so101-state-jitter-arm-only", train_cmd)
            self.assertIn("--so101-state-dropout-prob=0.02", train_cmd)
            self.assertIn("--so101-state-dropout-keep-gripper", train_cmd)
            self.assertIn("--so101-image-camera-dropout-prob=0.04", train_cmd)
            self.assertIn("--so101-image-patch-dropout-prob=0.05", train_cmd)
            self.assertIn("--so101-image-patch-mask-ratio=0.15", train_cmd)
            self.assertIn("--so101-gpu-image-augmentation", train_cmd)
            self.assertFalse(any("action-dropout" in arg for arg in train_cmd))
            self.assertEqual(payload["dataset_config"]["train_dataset"]["repo_id"], "physical-ai-agent/train")

    def test_so101_training_configs_default_to_moderate_augmentation_without_action_dropout(self) -> None:
        for config_path in (
            Path("configs/so101/training_datasets/pick.json"),
            Path("configs/so101/training_datasets/pick_place.json"),
            Path("configs/so101/training_datasets/pick_from_top_cube.json"),
        ):
            with self.subTest(config=str(config_path)):
                config = json.loads(config_path.read_text(encoding="utf-8"))
                augmentation = config["augmentation"]
                self.assertEqual(augmentation["state_jitter_std"], 0.003)
                self.assertEqual(augmentation["state_dropout_prob"], 0.02)
                self.assertEqual(augmentation["image_patch_mask_ratio"], 0.15)
                self.assertTrue(augmentation["gpu_image_augmentation"])
                self.assertEqual(augmentation["image_camera_dropout_prob"], 0.0)
                self.assertEqual(augmentation["image_patch_dropout_prob"], 0.0)
                self.assertNotIn("action_dropout_prob", augmentation)

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
        self.assertIn("gpu_image_augmentation=true", docs)
        self.assertIn("Validation and closed-loop test", docs)
        self.assertIn("teacher-action dropout", docs)
        self.assertIn("temporal smoothness loss", docs)
        self.assertIn("temporal ensembling", docs)

    def test_so101_training_dataset_configs_match_checksum_manifest(self) -> None:
        checksum_path = Path("configs/so101/training_datasets/checksums.json")
        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
        self.assertEqual(checksums["algorithm"], "sha256")
        contract = json.loads(
            Path("configs/so101/training_datasets/dataset_contract.json").read_text(encoding="utf-8")
        )

        for dataset_name, dataset_spec in contract["datasets"].items():
            for split_name, suffix in (("train", "train"), ("validation", "val")):
                checksum_key = f"{dataset_name}_{suffix}"
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
                self.assertEqual(dataset["sample_shapes"]["observation.images.camera1"], [3, 256, 256])
                self.assertEqual(dataset["sample_shapes"]["observation.images.camera2"], [3, 256, 256])

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
