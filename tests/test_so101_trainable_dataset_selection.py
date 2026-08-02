from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from physical_ai_agent.so101_trainable_dataset_selection import (
    LEGACY_TRAINABLE_DATASET_SELECTION_PATH,
    dataset_role_counts,
    load_dataset_role_selection,
    load_trainable_dataset_selection,
    loop_test_cases_from_selection,
    trainable_dataset_selection_path,
    training_dataset_entries_from_selection,
    update_dataset_role_selection,
    update_trainable_dataset_selection,
    validation_dataset_entries_from_selection,
)


class SO101TrainableDatasetSelectionTests(unittest.TestCase):
    def test_persists_deduplicated_portable_roots_and_resolves_training_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            first_root = repo_root / "_workspace/so101_lerobot/first"
            second_root = repo_root / "_workspace/so101_lerobot/second"
            first_root.mkdir(parents=True)
            second_root.mkdir(parents=True)
            sidecar = first_root / "meta/camera_grid_bins.json"
            sidecar.parent.mkdir(parents=True)
            sidecar.write_text("{}\n", encoding="utf-8")

            update_trainable_dataset_selection(
                repo_root,
                additions=[
                    {
                        "catalog_name": "first_train",
                        "root": str(first_root),
                        "repo_id": "example/first",
                        "expected_episodes": 10,
                        "expected_frames": 100,
                        "grid_bin_sidecar": str(sidecar),
                    },
                    {
                        "catalog_name": "second_train",
                        "root": str(second_root),
                        "repo_id": "example/second",
                    },
                ],
            )
            updated = update_trainable_dataset_selection(
                repo_root,
                additions=[
                    {
                        "catalog_name": "first_train_renamed",
                        "root": str(first_root),
                        "repo_id": "example/first",
                        "expected_episodes": 10,
                        "expected_frames": 100,
                        "grid_bin_sidecar": str(sidecar),
                    }
                ],
            )

            self.assertEqual(len(updated.datasets), 2)
            self.assertTrue(trainable_dataset_selection_path(repo_root).is_file())
            first = next(entry for entry in updated.datasets if entry.repo_id == "example/first")
            self.assertEqual(first.catalog_name, "first_train_renamed")
            self.assertEqual(first.root, "_workspace/so101_lerobot/first")
            self.assertEqual(
                first.grid_bin_sidecar,
                "_workspace/so101_lerobot/first/meta/camera_grid_bins.json",
            )

            training_entries = training_dataset_entries_from_selection(repo_root)
            self.assertEqual(len(training_entries), 2)
            first_training = next(row for row in training_entries if row["repo_id"] == "example/first")
            self.assertEqual(Path(first_training["root"]), first_root.resolve())
            self.assertEqual(Path(first_training["grid_bin_sidecar"]), sidecar.resolve())

    def test_remove_and_missing_root_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "_workspace/so101_lerobot/train"
            dataset_root.mkdir(parents=True)
            update_trainable_dataset_selection(
                repo_root,
                additions=[
                    {
                        "catalog_name": "train",
                        "root": str(dataset_root),
                        "repo_id": "example/train",
                    }
                ],
            )

            dataset_root.rmdir()
            with self.assertRaisesRegex(ValueError, "no longer exists"):
                training_dataset_entries_from_selection(repo_root)

            update_trainable_dataset_selection(repo_root, remove_roots=[dataset_root])
            self.assertEqual(load_trainable_dataset_selection(repo_root).datasets, [])
            with self.assertRaisesRegex(ValueError, "no datasets are marked"):
                training_dataset_entries_from_selection(repo_root)

    def test_persists_three_independent_training_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            train_root = repo_root / "_workspace/so101_lerobot/train"
            validation_root = repo_root / "_workspace/so101_lerobot/validation"
            train_root.mkdir(parents=True)
            validation_root.mkdir(parents=True)
            start_report = validation_root / "meta/closed_loop/start10.json"
            start_report.parent.mkdir(parents=True)
            start_report.write_text("{}\n", encoding="utf-8")

            selection = update_dataset_role_selection(
                repo_root,
                additions=[
                    {
                        "role": "training",
                        "catalog_name": "train",
                        "root": str(train_root),
                        "repo_id": "example/train",
                    },
                    {
                        "role": "validation",
                        "catalog_name": "validation",
                        "root": str(validation_root),
                        "repo_id": "example/validation",
                    },
                    {
                        "role": "loop_test",
                        "catalog_name": "validation_loop_test",
                        "root": str(validation_root),
                        "repo_id": "example/validation",
                        "loop_test_case": {
                            "id": "validation_loop_test",
                            "episodes": 10,
                            "start_report_path": str(start_report),
                            "start_dataset": {
                                "name": "validation",
                                "repo_id": "example/validation",
                                "root": str(validation_root),
                            },
                        },
                    },
                ],
            )

            self.assertEqual(selection.schema_version, 2)
            self.assertEqual(
                dataset_role_counts(repo_root),
                {"training": 1, "validation": 1, "loop_test": 1},
            )
            self.assertEqual(
                validation_dataset_entries_from_selection(repo_root)[0]["name"],
                "validation",
            )
            loop_case = loop_test_cases_from_selection(repo_root)[0]
            self.assertEqual(loop_case["id"], "validation_loop_test")
            self.assertEqual(Path(loop_case["start_report_path"]), start_report.resolve())
            self.assertEqual(
                Path(loop_case["start_dataset"]["root"]),
                validation_root.resolve(),
            )

    def test_reads_legacy_trainable_file_as_training_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "_workspace/so101_lerobot/train"
            dataset_root.mkdir(parents=True)
            legacy_path = repo_root / LEGACY_TRAINABLE_DATASET_SELECTION_PATH
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.write_text(
                """{
  "schema_version": 1,
  "updated_at": "2026-08-02T00:00:00+00:00",
  "datasets": [{
    "catalog_name": "train",
    "root": "_workspace/so101_lerobot/train",
    "repo_id": "example/train",
    "marked_at": "2026-08-02T00:00:00+00:00"
  }]
}\n""",
                encoding="utf-8",
            )

            selection = load_dataset_role_selection(repo_root)

            self.assertEqual(selection.schema_version, 2)
            self.assertEqual(selection.datasets[0].role, "training")


if __name__ == "__main__":
    unittest.main()
