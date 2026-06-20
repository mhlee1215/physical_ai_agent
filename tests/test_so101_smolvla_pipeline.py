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

from physical_ai_agent.so101_smolvla_pipeline import (
    SO101DatasetManifest,
    SO101TrainingSchedule,
    SmolVLASO101Contract,
    detect_overfit_stop,
    should_run_closed_loop,
    validate_smolvla_train_config,
)
from physical_ai_agent.sim.so101_camera_input import EGOCENTRIC_CAMERA1_POSE


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
        self.assertIn("important/train_loss", constants)
        self.assertIn("important/val_loss", constants)
        self.assertIn("val/action_jitter/", constants)
        self.assertIn("path_to_endpoint_ratio_mean", constants)
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
                qwen_model="qwen3-vl-8b-instruct-mlx",
                qwen_plan_json=None,
                qwen_response_json=Path("configs/agent/qwen3_so101_tool_planner_mock_response.json"),
                qwen_base_url=None,
                qwen_api_key=None,
                closed_loop_episodes=1,
                closed_loop_seed=98100,
                policy_device="cpu",
                closed_loop_steps=2,
                policy_n_action_steps=15,
                policy_num_steps=10,
                record_loop_artifacts=True,
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
        self.assertIn("--record-loop-artifacts", captured_cmd)
        self.assertEqual(captured_cmd[captured_cmd.index("--artifact-width") + 1], "128")

    def test_virtual_merge_concat_dataset_len_is_sum(self) -> None:
        from physical_ai_agent.so101_lerobot_concat import LeRobotConcatDataset

        class FakeDataset:
            def __init__(self, name: str, length: int) -> None:
                self.name = name
                self.repo_id = name
                self.root = f"/tmp/{name}"
                self.meta = {"name": name}
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

    def test_virtual_merge_balanced_sampler_draws_each_dataset_evenly(self) -> None:
        import torch

        from physical_ai_agent.so101_lerobot_concat import LeRobotConcatDataset

        class FakeDataset:
            def __init__(self, name: str, length: int) -> None:
                self.name = name
                self.repo_id = name
                self.root = f"/tmp/{name}"
                self.meta = {"name": name}
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
            self.assertIn("dashboard_cmd", payload)
            self.assertIn("gpu_monitor_cmd", payload)
            self.assertTrue(
                any("log_gpu_metrics_tensorboard.py" in part for part in payload["gpu_monitor_cmd"])
            )
            self.assertIn("--backend", payload["gpu_monitor_cmd"])
            self.assertIn("auto", payload["gpu_monitor_cmd"])
            self.assertIn("--train-pid-file", payload["gpu_monitor_cmd"])

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

    def test_qwen_edge_merge_normalizes_legacy_static_finger_prompts(self) -> None:
        from scripts.merge_so101_lerobot_shards import _normalize_task_prompt

        self.assertEqual(
            _normalize_task_prompt("Move the static finger pad above one visible green cube edge."),
            "Move the gripper above one visible green cube edge.",
        )
        self.assertEqual(
            _normalize_task_prompt("Align the static finger pad with one visible red cube edge."),
            "Align the gripper jaws around one visible red cube edge.",
        )
        self.assertEqual(
            _normalize_task_prompt(
                "Keep the static finger pad at the blue cube edge, close the gripper, and lift."
            ),
            "Close the gripper on the blue cube edge and lift.",
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
            self.assertIn("--num_workers=4", train_cmd)
            self.assertIn("--policy.repo_id=mhlee1215/test-policy", train_cmd)
            self.assertIn("--policy.push_to_hub=false", train_cmd)
            self.assertIn("--lightning-precision=bf16-mixed", train_cmd)
            self.assertIn("--validation-interval-epochs=1", train_cmd)
            self.assertIn("--save_freq=42", train_cmd)
            self.assertIn("--so101-image-cache-dir=/tmp/so101-cache/train", train_cmd)
            self.assertIn("--validation-image-cache-dir=/tmp/so101-cache/val", train_cmd)
            self.assertIn("tensorboard_cmd", payload)
            self.assertIn("progress_monitor_cmd", payload)
            self.assertIn("--closed-loop-eval-skill-mode", payload["progress_monitor_cmd"])
            self.assertIn("pick_from_top_cube", payload["progress_monitor_cmd"])
            self.assertIn("--mujoco-gl", payload["progress_monitor_cmd"])
            self.assertIn("glfw", payload["progress_monitor_cmd"])
            self.assertIn("--closed-loop-task-prompt", payload["progress_monitor_cmd"])
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
            self.assertIn("--mujoco-gl", payload["progress_monitor_cmd"])
            self.assertIn("egl", payload["progress_monitor_cmd"])

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
            self.assertIn("closed-loop eval skill mode", completed.stderr)

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
            self.assertIn("--loop-artifact-width", progress_cmd)
            self.assertIn("--qwen-response-json", progress_cmd)
            self.assertIn("configs/agent/qwen3_so101_tool_planner_mock_response.json", progress_cmd)

    def test_so101_training_configs_default_to_moderate_augmentation_without_action_dropout(self) -> None:
        for config_path in (
            Path("configs/so101/training_datasets/pick.json"),
            Path("configs/so101/training_datasets/pick_place.json"),
            Path("configs/so101/training_datasets/move_over_cube.json"),
            Path("configs/so101/training_datasets/pick_from_top_cube.json"),
            Path("configs/so101/training_datasets/all_hf_train_pick_place_closed_loop.json"),
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
            for dataset_name in contract["datasets"]:
                expected_keys.add((contract_name, dataset_name, "train"))
                expected_keys.add((contract_name, dataset_name, "validation"))

        self.assertEqual(recipe_keys, expected_keys)
        self.assertEqual(recipes["camera_pose_source"], "physical_ai_agent.sim.so101_camera_input.EGOCENTRIC_CAMERA1_POSE")
        self.assertEqual(recipes["defaults"]["width"], 256)
        self.assertEqual(recipes["defaults"]["height"], 256)

        for recipe in recipes["recipes"]:
            with self.subTest(recipe=recipe["name"]):
                self.assertIn("ego_wrist_256", recipe["root"])
                self.assertIn("ego-wrist-256", recipe["repo_id"])
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
        self.assertIn("gpu_image_augmentation=true", docs)
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
                for split_name, suffix in (("train", "train"), ("validation", "val")):
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
