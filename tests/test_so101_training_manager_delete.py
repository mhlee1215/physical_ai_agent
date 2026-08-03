from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import serve_so101_training_manager as manager


class TrainingManagerDeleteTests(unittest.TestCase):
    def _create_run(self, repo_root: Path, training_id: str) -> Path:
        run_dir = repo_root / "_workspace" / "so101_training" / "runs" / training_id
        run_dir.mkdir(parents=True)
        (run_dir / "checkpoint.bin").write_bytes(b"checkpoint")
        (run_dir / "training_run_summary.json").write_text(
            json.dumps(
                {
                    "training_id": training_id,
                    "run_dir": str(run_dir),
                    "dataset_config": {"name": f"dataset_{training_id}"},
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    def _write_registry(self, repo_root: Path, rows: list[dict[str, str]]) -> Path:
        path = repo_root / "_workspace" / "so101_training" / "training_runs_index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"runs": rows}), encoding="utf-8")
        return path

    def test_bulk_delete_removes_selected_runs_and_registry_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            first = self._create_run(repo_root, "past.001")
            second = self._create_run(repo_root, "past.002")
            retained = self._create_run(repo_root, "past.003")
            registry = self._write_registry(
                repo_root,
                [
                    {"training_id": "past.001", "run_dir": str(first)},
                    {"training_id": "past.002", "run_dir": str(second)},
                    {"training_id": "past.003", "run_dir": str(retained)},
                ],
            )

            result = manager._delete_runs(
                repo_root,
                {
                    "training_ids": ["past.002", "past.001"],
                    "confirmation": "DELETE 2 TRAINING RUNS",
                },
            )

            self.assertEqual(result["deleted_training_ids"], ["past.001", "past.002"])
            self.assertGreater(result["size_bytes"], 0)
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
            self.assertTrue(retained.exists())
            registry_rows = json.loads(registry.read_text(encoding="utf-8"))["runs"]
            self.assertEqual([row["training_id"] for row in registry_rows], ["past.003"])

    def test_bulk_delete_rejects_active_run_before_deleting_anything(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            past = self._create_run(repo_root, "past.001")
            active = self._create_run(repo_root, "active.001")
            state_root = repo_root / "_workspace" / "so101_training"
            (state_root / "active_training.json").write_text(
                json.dumps(
                    {
                        "training_id": "active.001",
                        "run_dir": str(active),
                        "train_pid": os.getpid(),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PermissionError, "active training run"):
                manager._delete_runs(
                    repo_root,
                    {
                        "training_ids": ["past.001", "active.001"],
                        "confirmation": "DELETE 2 TRAINING RUNS",
                    },
                )

            self.assertTrue(past.exists())
            self.assertTrue(active.exists())

    def test_bulk_delete_requires_exact_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            run_dir = self._create_run(repo_root, "past.001")

            with self.assertRaisesRegex(ValueError, "must exactly match"):
                manager._delete_runs(
                    repo_root,
                    {
                        "training_ids": ["past.001"],
                        "confirmation": "DELETE TRAINING RUN",
                    },
                )

            self.assertTrue(run_dir.exists())

    def test_bulk_delete_rejects_run_directory_outside_training_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            outside = repo_root / "outside"
            outside.mkdir(parents=True)
            runs_root = repo_root / "_workspace" / "so101_training" / "runs" / "bad"
            runs_root.mkdir(parents=True)
            (runs_root / "training_run_summary.json").write_text(
                json.dumps({"training_id": "bad.001", "run_dir": str(outside)}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PermissionError, "outside"):
                manager._delete_runs(
                    repo_root,
                    {
                        "training_ids": ["bad.001"],
                        "confirmation": "DELETE 1 TRAINING RUNS",
                    },
                )

            self.assertTrue(outside.exists())


if __name__ == "__main__":
    unittest.main()
