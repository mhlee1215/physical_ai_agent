from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from physical_ai_agent.so101_dataset_registry import (
    DatasetRegistryError,
    require_recipe_training_ready,
    scan_dataset_registry,
    validate_registered_recipe,
)


class SO101DatasetRegistryTests(unittest.TestCase):
    def test_complete_recipe_backed_splits_are_training_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            recipe_path = _write_recipe(repo_root, "cube_v3", include_validation=True)
            _write_ready_dataset(repo_root / "_workspace/so101_lerobot/cube_v3", episodes=4)
            _write_ready_dataset(
                repo_root / "_workspace/so101_lerobot/cube_v3_validation",
                episodes=2,
                closed_loop_output="meta/closed_loop/cube_v3_validation_start2.json",
            )

            registry = scan_dataset_registry(repo_root)
            selected = require_recipe_training_ready(repo_root, recipe_path)

        self.assertTrue(registry.valid)
        self.assertTrue(registry.training_ready)
        self.assertTrue(selected.training_ready)
        self.assertEqual(set(selected.training_manifests), {"cube_v3"})
        self.assertEqual([entry.catalog_name for entry in registry.entries], ["cube_v3", "cube_v3_validation"])
        self.assertTrue(all(entry.training_ready for entry in registry.entries))
        manifest = registry.training_manifests["cube_v3"]
        self.assertTrue(manifest["training_ready"])
        self.assertEqual(manifest["train_datasets"][0]["root"], "_workspace/so101_lerobot/cube_v3")
        self.assertEqual(
            manifest["validation_dataset"]["closed_loop_start"],
            "_workspace/so101_lerobot/cube_v3_validation/meta/closed_loop/cube_v3_validation_start2.json",
        )

    def test_registry_rejects_output_root_outside_canonical_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            recipe_path = _write_recipe(repo_root, "bad", output_root="_workspace/other/bad")

            registry = scan_dataset_registry(repo_root)

            self.assertFalse(registry.valid)
            self.assertIn("output_outside_dataset_root", {issue.code for issue in registry.issues})
            with self.assertRaises(DatasetRegistryError):
                validate_registered_recipe(repo_root, recipe_path)

    def test_registry_rejects_duplicate_output_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            _write_recipe(repo_root, "first", output_root="_workspace/so101_lerobot/shared")
            _write_recipe(repo_root, "second", output_root="_workspace/so101_lerobot/shared")

            registry = scan_dataset_registry(repo_root, inspect_artifacts=False)

        codes = {issue.code for issue in registry.issues}
        self.assertIn("duplicate_output_root", codes)
        self.assertIn("duplicate_catalog_name", codes)

    def test_registry_enforces_predictable_split_root_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            _write_recipe(
                repo_root,
                "cube_v4",
                output_root="_workspace/so101_lerobot/surprising_name",
            )

            registry = scan_dataset_registry(repo_root, inspect_artifacts=False)

        self.assertIn("output_name_mismatch", {issue.code for issue in registry.issues})

    def test_registry_cli_lists_current_repository_as_training_ready(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/so101_dataset_registry.py",
                "validate",
                "--require-training-ready",
                "--json",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["valid"])
        self.assertTrue(payload["training_ready"])
        self.assertGreaterEqual(payload["summary"]["splits"], 7)


def _write_recipe(
    repo_root: Path,
    dataset_id: str,
    *,
    output_root: str | None = None,
    include_validation: bool = False,
) -> Path:
    recipe_dir = repo_root / "configs/so101/dataset_generation"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    train_root = output_root or f"_workspace/so101_lerobot/{dataset_id}"
    splits: dict[str, object] = {
        "train": {
            "output_root": train_root,
            "repo_id": f"physical-ai-agent/{dataset_id}",
            "bins": [{"id": 5, "episodes": 4, "seed": 100}],
        }
    }
    if include_validation:
        splits["validation"] = {
            "output_root": f"_workspace/so101_lerobot/{dataset_id}_validation",
            "repo_id": f"physical-ai-agent/{dataset_id}-validation",
            "bins": [{"id": 5, "episodes": 2, "seed": 200}],
            "closed_loop": {
                "episodes": 2,
                "output": f"meta/closed_loop/{dataset_id}_validation_start2.json",
            },
        }
    path = recipe_dir / f"{dataset_id}.json"
    path.write_text(json.dumps({"schema_version": 1, "name": dataset_id, "splits": splits}), encoding="utf-8")
    return path


def _write_ready_dataset(
    root: Path,
    *,
    episodes: int,
    closed_loop_output: str | None = None,
) -> None:
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000/file-000.parquet").write_bytes(b"parquet-placeholder")
    (root / "meta/camera_grid_bins").mkdir(parents=True)
    (root / "meta/camera_grid_bins/camera1.parquet").write_bytes(b"sidecar")
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 12,
                "total_episodes": episodes,
                "total_frames": episodes * 10,
                "features": {
                    "observation.images.camera1": {"shape": [256, 256, 3]},
                    "observation.images.camera2": {"shape": [256, 256, 3]},
                    "observation.state": {"shape": [6]},
                    "action": {"shape": [6]},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "meta/stats.json").write_text("{}", encoding="utf-8")
    (root / "meta/tasks.parquet").write_bytes(b"tasks")
    (root / "so101_lerobot_export_report.json").write_text("{}", encoding="utf-8")
    (root / "so101_lerobot_merge_report.json").write_text("{}", encoding="utf-8")
    (root / "so101_lerobot_audit.json").write_text('{"status":"passed"}', encoding="utf-8")
    if closed_loop_output:
        path = root / closed_loop_output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
