from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import start_so101_training
from physical_ai_agent.so101_trainable_dataset_selection import (
    dataset_role_selection_path,
    trainable_dataset_selection_path,
    update_dataset_role_selection,
    update_trainable_dataset_selection,
)


class SO101MarkedTrainableLauncherTests(unittest.TestCase):
    def test_marked_set_replaces_only_training_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            first = repo_root / "_workspace/so101_lerobot/first"
            second = repo_root / "_workspace/so101_lerobot/second"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            update_trainable_dataset_selection(
                repo_root,
                additions=[
                    {
                        "catalog_name": "first_train",
                        "root": str(first),
                        "repo_id": "example/first",
                        "expected_episodes": 20,
                    },
                    {
                        "catalog_name": "second_train",
                        "root": str(second),
                        "repo_id": "example/second",
                        "expected_episodes": 30,
                    },
                ],
            )
            original = {
                "train_dataset": {"name": "old_train", "root": "/old/train"},
                "train_datasets": [{"name": "old_train", "root": "/old/train"}],
                "validation_dataset": {"name": "validation", "root": "/keep/validation"},
                "closed_loop": {"test_cases": ["keep-this-loop"]},
            }

            updated = start_so101_training._with_marked_trainable_set(
                original,
                repo_root=repo_root,
            )

            self.assertEqual(
                [dataset["name"] for dataset in updated["train_datasets"]],
                ["first_train", "second_train"],
            )
            self.assertEqual(updated["train_dataset"], updated["train_datasets"][0])
            self.assertEqual(updated["validation_dataset"], original["validation_dataset"])
            self.assertEqual(updated["closed_loop"], original["closed_loop"])
            self.assertEqual(updated["marked_trainable_set"]["count"], 2)
            self.assertEqual(
                Path(updated["marked_trainable_set"]["path"]),
                trainable_dataset_selection_path(repo_root),
            )
            self.assertEqual(original["train_dataset"]["name"], "old_train")

    def test_marked_set_requires_at_least_one_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(SystemExit, "no datasets are marked"):
                start_so101_training._with_marked_trainable_set(
                    {"train_dataset": {"name": "old"}},
                    repo_root=Path(tmpdir),
                )

    def test_hf_resolution_preserves_marked_train_roots_but_resolves_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            train_root = repo_root / "_workspace/so101_lerobot/train"
            train_root.mkdir(parents=True)
            config = {
                "train_datasets": [
                    {"name": "marked_train", "repo_id": "example/train", "root": str(train_root)}
                ],
                "train_dataset": {
                    "name": "marked_train",
                    "repo_id": "example/train",
                    "root": str(train_root),
                },
                "validation_datasets": [
                    {
                        "name": "validation",
                        "repo_id": "example/validation",
                        "hf_repo_id": "example/bundle",
                        "hf_path_in_repo": "datasets/validation",
                    }
                ],
            }

            resolved = start_so101_training._resolve_hf_dataset_downloads(
                config,
                repo_root=repo_root,
                cache_root=Path("_workspace/hf_cache"),
                download=False,
                preserve_local_train_roots=True,
            )

            self.assertEqual(Path(resolved["train_datasets"][0]["root"]), train_root.resolve())
            self.assertEqual(
                Path(resolved["validation_datasets"][0]["root"]),
                repo_root / "_workspace/hf_cache/example__bundle/datasets/validation",
            )
            self.assertEqual(len(resolved["hf_dataset_downloads"]), 1)

    def test_marked_role_set_replaces_training_validation_and_loop_test_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            train_root = repo_root / "_workspace/so101_lerobot/train"
            validation_root = repo_root / "_workspace/so101_lerobot/validation"
            train_root.mkdir(parents=True)
            validation_root.mkdir(parents=True)
            start_report = validation_root / "meta/closed_loop/start10.json"
            start_report.parent.mkdir(parents=True)
            start_report.write_text("{}\n", encoding="utf-8")
            update_dataset_role_selection(
                repo_root,
                additions=[
                    {
                        "role": "training",
                        "catalog_name": "marked_train",
                        "root": str(train_root),
                        "repo_id": "example/train",
                    },
                    {
                        "role": "validation",
                        "catalog_name": "marked_validation",
                        "root": str(validation_root),
                        "repo_id": "example/validation",
                    },
                    {
                        "role": "loop_test",
                        "catalog_name": "marked_loop_test",
                        "root": str(validation_root),
                        "repo_id": "example/validation",
                        "loop_test_case": {
                            "id": "marked_loop_test",
                            "episodes": 10,
                            "steps": 100,
                            "_contract_path": str(
                                validation_root / "meta/closed_loop/start10.contract.json"
                            ),
                            "_observation_renderer_contract": {
                                "mode": "blender_cycles_live",
                                "camera_rig_config": "configs/camera.json",
                            },
                            "start_report_path": str(start_report),
                            "start_dataset": {
                                "name": "marked_validation",
                                "repo_id": "example/validation",
                                "root": str(validation_root),
                            },
                        },
                    },
                ],
            )
            original = {
                "train_datasets": [{"name": "old_train"}],
                "validation_dataset": {"name": "old_validation"},
                "closed_loop": {
                    "runner": "picklift",
                    "test_cases": [{"id": "old_loop"}],
                },
            }

            updated = start_so101_training._with_marked_dataset_set(
                original,
                repo_root=repo_root,
            )

            self.assertEqual(updated["train_datasets"][0]["name"], "marked_train")
            self.assertEqual(
                updated["validation_datasets"][0]["name"],
                "marked_validation",
            )
            self.assertEqual(
                updated["closed_loop"]["test_cases"][0]["id"],
                "marked_loop_test",
            )
            self.assertNotIn(
                "_observation_renderer_contract",
                updated["closed_loop"]["test_cases"][0],
            )
            self.assertEqual(
                updated["closed_loop"]["observation_renderer"]["camera_rig_config"],
                "configs/camera.json",
            )
            self.assertEqual(updated["closed_loop"]["runner"], "picklift")
            self.assertEqual(
                Path(updated["marked_dataset_set"]["path"]),
                dataset_role_selection_path(repo_root),
            )
            self.assertEqual(
                updated["marked_dataset_set"]["counts"],
                {"training": 1, "validation": 1, "loop_test": 1},
            )
            self.assertEqual(
                updated["marked_dataset_set"]["loop_test_contract_paths"],
                [str(validation_root / "meta/closed_loop/start10.contract.json")],
            )
            self.assertEqual(original["closed_loop"]["test_cases"][0]["id"], "old_loop")

    def test_marked_role_set_requires_all_three_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            train_root = repo_root / "_workspace/so101_lerobot/train"
            train_root.mkdir(parents=True)
            update_trainable_dataset_selection(
                repo_root,
                additions=[
                    {
                        "catalog_name": "train",
                        "root": str(train_root),
                        "repo_id": "example/train",
                    }
                ],
            )

            with self.assertRaisesRegex(SystemExit, "validation set"):
                start_so101_training._with_marked_dataset_set(
                    {"train_datasets": []},
                    repo_root=repo_root,
                )

    def test_hf_resolution_preserves_marked_training_and_validation_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            train_root = repo_root / "_workspace/so101_lerobot/train"
            validation_root = repo_root / "_workspace/so101_lerobot/validation"
            train_root.mkdir(parents=True)
            validation_root.mkdir(parents=True)
            config = {
                "train_datasets": [
                    {"name": "train", "repo_id": "example/train", "root": str(train_root)}
                ],
                "validation_datasets": [
                    {
                        "name": "validation",
                        "repo_id": "example/validation",
                        "root": str(validation_root),
                    }
                ],
            }

            resolved = start_so101_training._resolve_hf_dataset_downloads(
                config,
                repo_root=repo_root,
                cache_root=Path("_workspace/hf_cache"),
                download=False,
                preserve_local_train_roots=True,
                preserve_local_validation_roots=True,
            )

            self.assertEqual(Path(resolved["train_datasets"][0]["root"]), train_root.resolve())
            self.assertEqual(
                Path(resolved["validation_datasets"][0]["root"]),
                validation_root.resolve(),
            )
            self.assertEqual(resolved.get("hf_dataset_downloads") or [], [])


if __name__ == "__main__":
    unittest.main()
