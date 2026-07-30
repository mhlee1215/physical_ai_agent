from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_mycobot280_smolvla_tiny import run_mycobot280_smolvla_tiny_finetune


class MyCobot280SmolVLATinyFineTuneTest(unittest.TestCase):
    def test_incomplete_native_dataset_writes_blocked_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "train"

            report = run_mycobot280_smolvla_tiny_finetune(
                dataset_root=tmp_path / "missing_native_dataset",
                dataset_repo_id="physical-ai-agent/mycobot-280-ground-pickup-tiny-smoke",
                policy_path="lerobot/smolvla_base",
                output_dir=output_dir,
                batch_size=1,
                steps=2,
                learning_rate=1e-6,
                num_workers=0,
                device="cpu",
                local_files_only=True,
            )

            self.assertEqual(report["status"], "blocked")
            self.assertIn("native LeRobotDataset root is incomplete", report["blocker"])
            self.assertTrue((output_dir / "tiny_finetune.json").exists())

    def test_complete_native_dataset_calls_fake_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = _write_minimal_native_dataset_root(tmp_path / "native_dataset")
            output_dir = tmp_path / "train"
            calls: list[dict[str, object]] = []

            def fake_trainer(**kwargs: object) -> dict[str, object]:
                calls.append(kwargs)
                out = Path(kwargs["output_dir"])
                checkpoint = out / "checkpoints" / "latest" / "pretrained_model"
                checkpoint.mkdir(parents=True)
                (checkpoint / "config.json").write_text("{}", encoding="utf-8")
                (out / "checkpoints" / "latest" / "optimizer_state.pt").write_bytes(b"optimizer")
                (out / "checkpoints" / "latest" / "training_state.json").write_text(
                    json.dumps({"status": "passed", "steps": kwargs["steps"]}),
                    encoding="utf-8",
                )
                (out / "train.log").write_text('{"event":"step","step":1}\n', encoding="utf-8")
                (out / "tensorboard").mkdir()
                (out / "tensorboard" / "events.out.tfevents.fake").write_text("", encoding="utf-8")
                return {
                    "operation": "fake_tiny_finetune",
                    "status": "passed",
                    "steps": kwargs["steps"],
                    "loss_initial": 0.5,
                    "loss_final": 0.4,
                    "checkpoint_model_dir": str(checkpoint),
                    "optimizer_state_path": str(out / "checkpoints" / "latest" / "optimizer_state.pt"),
                    "train_log": str(out / "train.log"),
                    "tensorboard_dir": str(out / "tensorboard"),
                }

            report = run_mycobot280_smolvla_tiny_finetune(
                dataset_root=dataset_root,
                dataset_repo_id="physical-ai-agent/mycobot-280-ground-pickup-tiny-smoke",
                policy_path="local/smolvla-test",
                output_dir=output_dir,
                batch_size=1,
                steps=2,
                learning_rate=1e-6,
                num_workers=0,
                device="cpu",
                local_files_only=True,
                require_runtime=True,
                trainer=fake_trainer,
            )

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["operation"], "train_mycobot280_smolvla_tiny")
            self.assertEqual(report["training_report"]["steps"], 2)
            self.assertEqual(calls[0]["dataset_root"], dataset_root.resolve())
            self.assertEqual(calls[0]["learning_rate"], 1e-6)
            self.assertIn("optimizer-step fine-tune smoke", report["claim_boundary"])
            self.assertTrue((output_dir / "tiny_finetune.json").exists())
            self.assertTrue((output_dir / "checkpoints" / "latest" / "pretrained_model").exists())
            self.assertTrue((output_dir / "checkpoints" / "latest" / "optimizer_state.pt").exists())
            self.assertTrue((output_dir / "train.log").exists())
            self.assertTrue((output_dir / "tensorboard" / "events.out.tfevents.fake").exists())


def _write_minimal_native_dataset_root(root: Path) -> Path:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "robot_type": "mycobot_280_pi_adaptive_gripper",
                "fps": 12,
                "features": {
                    "observation.images.camera1": {"dtype": "image", "shape": [2, 2, 3]},
                    "observation.state": {"dtype": "float32", "shape": [7]},
                    "action": {"dtype": "float32", "shape": [7]},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"PAR1fake")
    (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"PAR1fake")
    (root / "meta" / "tasks.parquet").write_bytes(b"PAR1fake")
    return root


if __name__ == "__main__":
    unittest.main()
