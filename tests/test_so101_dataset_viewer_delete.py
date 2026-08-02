from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path("scripts").resolve()))

import serve_so101_dataset_viewer as dataset_viewer
from physical_ai_agent.so101_trainable_dataset_selection import (
    load_trainable_dataset_selection,
    update_trainable_dataset_selection,
)


class SO101DatasetViewerDeleteTests(unittest.TestCase):
    def test_deletes_registered_workspace_dataset_and_reports_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "_workspace/so101_lerobot/delete_me"
            dataset_root.mkdir(parents=True)
            (dataset_root / "payload.bin").write_bytes(b"dataset")
            roots = {
                "delete_me": Path("_workspace/so101_lerobot/delete_me"),
                "delete_me_alias": Path("_workspace/so101_lerobot/delete_me"),
            }

            with patch.dict(dataset_viewer.DATASETS, roots, clear=True):
                result = dataset_viewer._delete_dataset(
                    repo_root,
                    {"name": "delete_me", "confirm_name": "delete_me"},
                )

            self.assertFalse(dataset_root.exists())
            self.assertEqual(result["status"], "deleted")
            self.assertEqual(result["name"], "delete_me")
            self.assertEqual(result["affected_names"], ["delete_me", "delete_me_alias"])
            self.assertGreater(result["size_bytes"], 0)

    def test_requires_exact_name_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "_workspace/so101_lerobot/keep_me"
            dataset_root.mkdir(parents=True)

            with patch.dict(
                dataset_viewer.DATASETS,
                {"keep_me": Path("_workspace/so101_lerobot/keep_me")},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "confirmation does not match"):
                    dataset_viewer._delete_dataset(
                        repo_root,
                        {"name": "keep_me", "confirm_name": "wrong"},
                    )

            self.assertTrue(dataset_root.exists())

    def test_refuses_dataset_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "external_dataset"
            dataset_root.mkdir()

            with patch.dict(
                dataset_viewer.DATASETS,
                {"external": dataset_root},
                clear=True,
            ):
                with self.assertRaisesRegex(PermissionError, "outside"):
                    dataset_viewer._delete_dataset(
                        repo_root,
                        {"name": "external", "confirm_name": "external"},
                    )

            self.assertTrue(dataset_root.exists())

    def test_bulk_delete_preflights_all_roots_before_removing_anything(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            workspace_dataset = repo_root / "_workspace/so101_lerobot/keep"
            outside_dataset = repo_root / "outside"
            workspace_dataset.mkdir(parents=True)
            outside_dataset.mkdir()

            with patch.dict(
                dataset_viewer.DATASETS,
                {"keep": workspace_dataset, "outside": outside_dataset},
                clear=True,
            ):
                with self.assertRaisesRegex(PermissionError, "outside"):
                    dataset_viewer._delete_datasets(
                        repo_root,
                        {
                            "names": ["keep", "outside"],
                            "confirmation": "DELETE 2 DATASETS",
                        },
                    )

            self.assertTrue(workspace_dataset.exists())
            self.assertTrue(outside_dataset.exists())

    def test_bulk_delete_removes_multiple_roots_and_trainable_marks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            first = repo_root / "_workspace/so101_lerobot/first"
            second = repo_root / "_workspace/so101_lerobot/second"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "one.bin").write_bytes(b"one")
            (second / "two.bin").write_bytes(b"two")
            update_trainable_dataset_selection(
                repo_root,
                additions=[
                    {"catalog_name": "first", "root": str(first), "repo_id": "example/first"},
                    {"catalog_name": "second", "root": str(second), "repo_id": "example/second"},
                ],
            )

            with patch.dict(
                dataset_viewer.DATASETS,
                {"first": first, "first_alias": first, "second": second},
                clear=True,
            ):
                result = dataset_viewer._delete_datasets(
                    repo_root,
                    {
                        "names": ["first", "second"],
                        "confirmation": "DELETE 2 DATASETS",
                    },
                )

            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
            self.assertEqual(len(result["deleted_roots"]), 2)
            self.assertEqual(result["affected_names"], ["first", "first_alias", "second"])
            self.assertEqual(load_trainable_dataset_selection(repo_root).datasets, [])

    def test_bulk_delete_requires_counted_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "_workspace/so101_lerobot/keep"
            dataset_root.mkdir(parents=True)
            with patch.dict(dataset_viewer.DATASETS, {"keep": dataset_root}, clear=True):
                with self.assertRaisesRegex(ValueError, "DELETE 1 DATASETS"):
                    dataset_viewer._delete_datasets(
                        repo_root,
                        {"names": ["keep"], "confirmation": "keep"},
                    )
            self.assertTrue(dataset_root.exists())

    def test_trainable_selection_marks_only_completion_gated_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "_workspace/so101_lerobot/ready_train"
            dataset_root.mkdir(parents=True)
            candidate = {
                "name": "ready_train",
                "root": dataset_root,
                "category": "generated",
                "loader": "lerobot",
                "platform": "so101",
                "split_key": "train",
            }
            registry_entry = SimpleNamespace(
                split="train",
                training_ready=True,
                readiness_errors=(),
                repo_id="example/ready-train",
                episodes=12,
                frames=120,
                grid_sidecar=None,
            )

            with (
                patch.object(dataset_viewer, "_dataset_catalog_candidates", return_value=[candidate]),
                patch.object(
                    dataset_viewer,
                    "_training_registry_entries_by_root",
                    return_value={dataset_root.resolve(): [registry_entry]},
                ),
            ):
                marked = dataset_viewer._update_trainable_dataset_selection(
                    repo_root,
                    {"action": "mark", "names": ["ready_train"]},
                )
                removed = dataset_viewer._update_trainable_dataset_selection(
                    repo_root,
                    {"action": "remove", "names": ["ready_train"]},
                )

            self.assertEqual(marked["count"], 1)
            self.assertEqual(marked["datasets"][0]["repo_id"], "example/ready-train")
            self.assertEqual(removed["count"], 0)

    def test_role_selection_marks_validation_and_executable_loop_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            validation_root = repo_root / "_workspace/so101_lerobot/ready_validation"
            validation_root.mkdir(parents=True)
            start_report = validation_root / "meta/closed_loop/start10.json"
            start_report.parent.mkdir(parents=True)
            start_report.write_text("{}\n", encoding="utf-8")
            validation_candidate = {
                "name": "ready_validation",
                "repo_root": repo_root,
                "root": validation_root,
                "category": "generated",
                "loader": "lerobot",
                "platform": "so101",
                "split_key": "valid",
            }
            loop_candidate = {
                "name": "ready_loop_test",
                "repo_root": repo_root,
                "root": validation_root,
                "category": "closed_loop",
                "loader": "lerobot",
                "platform": "so101",
                "split_key": "closed_loop",
                "loop_test_case": {
                    "id": "ready_loop_test",
                    "episodes": 10,
                    "start_report_path": str(start_report),
                    "start_dataset": {
                        "name": "ready_validation",
                        "repo_id": "example/ready-validation",
                        "root": str(validation_root),
                    },
                },
            }
            registry_entry = SimpleNamespace(
                split="validation",
                training_ready=True,
                readiness_errors=(),
                repo_id="example/ready-validation",
                episodes=50,
                frames=500,
                grid_sidecar=None,
            )

            with (
                patch.object(
                    dataset_viewer,
                    "_dataset_catalog_candidates",
                    return_value=[validation_candidate, loop_candidate],
                ),
                patch.object(
                    dataset_viewer,
                    "_training_registry_entries_by_root",
                    return_value={validation_root.resolve(): [registry_entry]},
                ),
            ):
                validation = dataset_viewer._update_dataset_role_selection(
                    repo_root,
                    {
                        "action": "mark",
                        "role": "validation",
                        "names": ["ready_validation"],
                    },
                )
                loop = dataset_viewer._update_dataset_role_selection(
                    repo_root,
                    {
                        "action": "mark",
                        "role": "loop_test",
                        "names": ["ready_loop_test"],
                    },
                )
                removed = dataset_viewer._update_dataset_role_selection(
                    repo_root,
                    {
                        "action": "remove",
                        "role": "loop_test",
                        "names": ["ready_loop_test"],
                    },
                )

            self.assertEqual(validation["counts"]["validation"], 1)
            self.assertEqual(loop["counts"], {"training": 0, "validation": 1, "loop_test": 1})
            self.assertEqual(loop["datasets"][1]["loop_test_case"]["id"], "ready_loop_test")
            self.assertEqual(removed["counts"], {"training": 0, "validation": 1, "loop_test": 0})

    def test_delete_request_allows_private_hosts_without_proxy_headers(self) -> None:
        self.assertTrue(
            dataset_viewer._trusted_dataset_delete_request(
                "127.0.0.1",
                {"Origin": "http://127.0.0.1:8768"},
            )
        )
        self.assertTrue(
            dataset_viewer._trusted_dataset_delete_request(
                "192.168.4.20",
                {"Origin": "http://192.168.4.46:8768"},
            )
        )

    def test_delete_request_rejects_public_or_forwarded_clients(self) -> None:
        self.assertFalse(dataset_viewer._trusted_dataset_delete_request("203.0.113.4", {}))
        self.assertFalse(
            dataset_viewer._trusted_dataset_delete_request(
                "127.0.0.1",
                {"CF-Connecting-IP": "203.0.113.4"},
            )
        )
        self.assertFalse(
            dataset_viewer._trusted_dataset_delete_request(
                "127.0.0.1",
                {"Origin": "https://example.com"},
            )
        )


if __name__ == "__main__":
    unittest.main()
