from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_mycobot_280_pi_smolvla_tiny_smoke import (
    audit_native_lerobot_dataset_root,
    run_mycobot_280_pi_smolvla_tiny_smoke,
)


class MyCobot280PiSmolVLATinySmokeTest(unittest.TestCase):
    def test_incomplete_native_dataset_writes_blocked_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "native_dataset"
            output_path = tmp_path / "smoke.json"

            report = run_mycobot_280_pi_smolvla_tiny_smoke(
                dataset_root=dataset_root,
                dataset_repo_id="physical-ai-agent/mycobot-280pi-adaptive-test",
                policy_path="lerobot/smolvla_base",
                output_path=output_path,
                batch_size=1,
                max_batches=1,
                device="cpu",
                local_files_only=True,
            )

            self.assertEqual(report["status"], "blocked")
            self.assertIn("native LeRobotDataset root is incomplete", report["blocker"])
            self.assertTrue(output_path.exists())
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["dataset_audit"]["status"], "blocked")

    def test_complete_native_dataset_calls_fake_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = _write_minimal_native_dataset_root(tmp_path / "native_dataset")
            output_path = tmp_path / "smoke.json"
            calls: list[dict[str, object]] = []

            def fake_evaluator(**kwargs: object) -> dict[str, object]:
                calls.append(kwargs)
                Path(kwargs["output_path"]).write_text(
                    json.dumps({"operation": "fake_loss", "batches_evaluated": 1}),
                    encoding="utf-8",
                )
                return {"operation": "fake_loss", "batches_evaluated": 1, "loss_mean": 0.123}

            report = run_mycobot_280_pi_smolvla_tiny_smoke(
                dataset_root=dataset_root,
                dataset_repo_id="physical-ai-agent/mycobot-280pi-adaptive-test",
                policy_path="local/smolvla-test",
                output_path=output_path,
                batch_size=1,
                max_batches=1,
                device="cpu",
                local_files_only=True,
                require_runtime=True,
                evaluator=fake_evaluator,
            )

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["loss_report"]["loss_mean"], 0.123)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["dataset_root"], dataset_root.resolve())
            self.assertEqual(calls[0]["max_batches"], 1)
            self.assertEqual(calls[0]["local_files_only"], True)
            self.assertEqual(report["dataset_audit"]["status"], "passed")
            self.assertTrue(output_path.exists())
            loss_path = output_path.with_name(output_path.stem + "_supervised_loss.json")
            self.assertTrue(loss_path.exists())

    def test_native_dataset_audit_requires_parquet_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = _write_minimal_native_dataset_root(tmp_path / "native_dataset")

            report = audit_native_lerobot_dataset_root(dataset_root)

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["robot_type"], "mycobot_280_pi_adaptive_gripper")
            self.assertTrue(report["data_files"])
            self.assertTrue(report["episode_files"])


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
                    "observation.images.camera2": {"dtype": "image", "shape": [2, 2, 3]},
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
    (root / "mycobot_280_pi_lerobot_convert_report.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    return root


if __name__ == "__main__":
    unittest.main()
