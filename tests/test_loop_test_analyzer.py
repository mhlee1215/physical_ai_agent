from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path("scripts").resolve()))

import build_loop_test_analyzer_export as exporter
import serve_loop_test_analyzer as server
import serve_so101_dataset_viewer as dataset_viewer


VALID_1X1_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class LoopTestAnalyzerTest(unittest.TestCase):
    def test_dataset_contract_falls_back_to_preserved_hf_upload_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            contract_path = repo_root / dataset_viewer.DATASET_CONTRACT
            contract_path.parent.mkdir(parents=True)
            contract_path.write_text(
                json.dumps(
                    {
                        "datasets": {
                            "pick_cube": {
                                "train": {
                                    "root": "_workspace/so101_lerobot/deleted_original_train"
                                },
                                "validation": {
                                    "root": "_workspace/so101_lerobot/deleted_original_val"
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            preserved_train = (
                repo_root
                / "_workspace/hf_upload/so101-nexus-sim-dataset/datasets/pick_cube/train"
            )
            preserved_val = preserved_train.parent / "validation"
            preserved_train.mkdir(parents=True)
            preserved_val.mkdir(parents=True)

            roots = dataset_viewer._contract_dataset_roots(repo_root)

        self.assertEqual(roots["pick_cube_train"], preserved_train.resolve())
        self.assertEqual(roots["pick_cube_val"], preserved_val.resolve())

    def test_dataset_contract_prefers_configured_root_when_it_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            contract_path = repo_root / dataset_viewer.DATASET_CONTRACT
            contract_path.parent.mkdir(parents=True)
            configured = repo_root / "_workspace/so101_lerobot/pick_cube_train"
            configured.mkdir(parents=True)
            fallback = (
                repo_root
                / "_workspace/hf_upload/so101-nexus-sim-dataset/datasets/pick_cube/train"
            )
            fallback.mkdir(parents=True)
            contract_path.write_text(
                json.dumps(
                    {
                        "datasets": {
                            "pick_cube": {
                                "train": {
                                    "root": "_workspace/so101_lerobot/pick_cube_train"
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            roots = dataset_viewer._contract_dataset_roots(repo_root)

        self.assertEqual(roots["pick_cube_train"], configured.resolve())

    def test_dataset_generation_recipes_register_completed_split_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config_dir = repo_root / dataset_viewer.DATASET_GENERATION_CONFIG_DIR
            config_dir.mkdir(parents=True)
            train_root = repo_root / "_workspace/so101_lerobot/grip_the_cube_v2_5"
            val_root = repo_root / "_workspace/so101_lerobot/grip_the_cube_v2_5_validation"
            train_root.mkdir(parents=True)
            val_root.mkdir(parents=True)
            config_dir.joinpath("grip_the_cube_v2_5.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": "grip_the_cube_v2_5",
                        "splits": {
                            "train": {"output_root": str(train_root.relative_to(repo_root))},
                            "validation": {"output_root": str(val_root.relative_to(repo_root))},
                            "missing": {
                                "output_root": "_workspace/so101_lerobot/grip_the_cube_v2_5_missing"
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            roots = dataset_viewer._generation_recipe_dataset_roots(repo_root)

        self.assertEqual(
            roots,
            {
                "grip_the_cube_v2_5": train_root.resolve(),
                "grip_the_cube_v2_5_validation": val_root.resolve(),
            },
        )

    def test_dataset_catalog_uses_metadata_without_loading_full_frame_table(self) -> None:
        dataset = {
            "root": Path("/tmp/cube"),
            "info": {
                "total_episodes": 2,
                "total_frames": 20,
                "fps": 12,
                "features": {"observation.images.camera1": {"shape": [256, 256, 3]}},
            },
            "camera_keys": ["observation.images.camera1"],
            "episode_lengths": [10, 10],
            "size_bytes": 100,
            "data_bytes": 60,
            "image_bytes": 40,
        }
        with (
            patch.object(dataset_viewer, "_dataset_metadata", return_value=dataset),
            patch.object(dataset_viewer, "_dataset", side_effect=AssertionError("full dataset load")),
        ):
            item = dataset_viewer._dataset_catalog_item(
                Path("/tmp"), "cube", Path("/tmp/cube"), category="generated"
            )

        self.assertEqual(item["status"], "available")
        self.assertEqual(item["frames"], 20)

    def test_dataset_catalog_exposes_explicit_creation_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "cube"
            (root / "meta").mkdir(parents=True)
            (root / "meta" / "info.json").write_text(
                json.dumps({"created_at": "2026-07-28T12:34:56Z"}),
                encoding="utf-8",
            )
            dataset = {
                "root": root,
                "info": {
                    "total_episodes": 1,
                    "total_frames": 10,
                    "fps": 12,
                    "features": {
                        "observation.images.camera1": {"shape": [256, 256, 3]}
                    },
                },
                "camera_keys": ["observation.images.camera1"],
                "episode_lengths": [10],
                "size_bytes": 100,
                "data_bytes": 60,
                "image_bytes": 40,
            }
            with patch.object(dataset_viewer, "_dataset_metadata", return_value=dataset):
                item = dataset_viewer._dataset_catalog_item(
                    Path(tmpdir), "cube", root, category="generated"
                )

        self.assertEqual(item["created_at"], "2026-07-28T12:34:56Z")
        self.assertEqual(item["created_at_source"], "meta/info.json:created_at")
        self.assertEqual(item["summary"]["created_at"], "2026-07-28T12:34:56Z")
        self.assertGreater(item["created_at_epoch"], 0)

    def test_dataset_catalog_progress_reports_each_processed_dataset(self) -> None:
        repo_root = Path("/tmp/progress-repo")
        first_root = repo_root / "first"
        second_root = repo_root / "second"
        progress_events: list[tuple[int, int, str]] = []

        def catalog_item(_repo_root: Path, split: str, root: Path, *, category: str):
            summary = {"name": split, "root": str(root), "episodes": 1, "frames": 1}
            return {
                "name": split,
                "status": "available",
                "category": category,
                "summary": summary,
            }

        with (
            patch.object(dataset_viewer, "_official_dataset_roots", return_value={"first": first_root}),
            patch.object(dataset_viewer, "_skill_dataset_roots", return_value={}),
            patch.object(dataset_viewer, "_generation_recipe_dataset_roots", return_value={"second": second_root}),
            patch.object(dataset_viewer, "_discover_so101_photoreal_datasets", return_value={}),
            patch.object(dataset_viewer, "_discover_so101_photoreal_lerobot_datasets", return_value={}),
            patch.object(dataset_viewer, "_generation_closed_loop_views", return_value={}),
            patch.object(dataset_viewer, "_discover_temporary_datasets", return_value={}),
            patch.object(dataset_viewer, "_discover_mycobot_datasets", return_value={}),
            patch.object(dataset_viewer, "_dataset_catalog_item", side_effect=catalog_item),
            patch.object(dataset_viewer, "ARCHIVED_DATASET_SPLITS", []),
        ):
            payload = dataset_viewer._build_datasets_payload(
                repo_root,
                progress=lambda completed, total, message: progress_events.append(
                    (completed, total, message)
                ),
            )

        self.assertEqual(set(payload["datasets"]), {"first", "second"})
        self.assertEqual(progress_events[0], (0, 2, "Reading dataset metadata"))
        self.assertEqual(progress_events[-1], (2, 2, "Finalizing catalog"))
        self.assertEqual([event[0] for event in progress_events[1:3]], [1, 2])

    def test_dataset_catalog_progress_endpoint_exposes_current_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            cache_key = str(repo_root.resolve())
            with dataset_viewer.DATASETS_PROGRESS_LOCK:
                dataset_viewer.DATASETS_PROGRESS.pop(cache_key, None)
            idle = _handle_viewer_request(repo_root, "/api/datasets/progress")
            dataset_viewer._set_datasets_progress(
                repo_root,
                status="loading",
                percent=42,
                completed=21,
                total=50,
                message="Reading cube_train (21/50)",
            )
            loading = _handle_viewer_request(repo_root, "/api/datasets/progress")
            with dataset_viewer.DATASETS_PROGRESS_LOCK:
                dataset_viewer.DATASETS_PROGRESS.pop(cache_key, None)

        self.assertEqual(idle["status"], "idle")
        self.assertEqual(idle["percent"], 0)
        self.assertEqual(loading["status"], "loading")
        self.assertEqual(loading["percent"], 42)
        self.assertEqual(loading["completed"], 21)
        self.assertEqual(loading["total"], 50)

    def test_dataset_catalog_remote_page_loads_only_requested_ten_summaries(self) -> None:
        candidates = [
            {
                "name": f"cube_{index:02d}",
                "root": Path(f"/tmp/cube_{index:02d}"),
                "category": "generated",
                "loader": "lerobot",
                "platform": "so101",
                "platform_label": "SO101",
                "split_key": "train",
                "split_label": "Train",
                "render_key": "simulation",
                "render_label": "Standard sim",
                "created_at": f"2026-07-{index + 1:02d}T00:00:00Z",
                "created_at_epoch": float(index + 1),
            }
            for index in range(25)
        ]

        def load_candidate(_repo_root: Path, candidate: dict[str, object]) -> dict[str, object]:
            name = str(candidate["name"])
            return {
                "status": "available",
                "detail": "ready",
                "summary": {
                    "name": name,
                    "root": str(candidate["root"]),
                    "episodes": 2,
                    "frames": 20,
                    "episode_lengths": [10, 10],
                },
            }

        with (
            patch.object(dataset_viewer, "_dataset_catalog_candidates", return_value=candidates),
            patch.object(dataset_viewer, "_load_dataset_catalog_candidate", side_effect=load_candidate) as loader,
        ):
            payload = dataset_viewer._dataset_catalog_page_payload(
                Path("/tmp/catalog-repo"),
                {"page": ["2"], "size": ["10"], "sort": ["createdEpoch"], "dir": ["desc"]},
            )

        self.assertEqual(payload["total"], 25)
        self.assertEqual(payload["last_page"], 3)
        self.assertEqual(payload["page"], 2)
        self.assertEqual(len(payload["data"]), 10)
        self.assertEqual(loader.call_count, 10)
        self.assertEqual(payload["data"][0]["name"], "cube_14")
        self.assertEqual(payload["data"][-1]["name"], "cube_05")

    def test_dataset_catalog_marked_filter_runs_before_page_loading(self) -> None:
        candidates = [
            {
                "name": f"cube_{index}",
                "root": Path(f"/tmp/cube_{index}"),
                "category": "generated",
                "loader": "lerobot",
                "platform": "so101",
                "platform_label": "SO101",
                "split_key": "train",
                "split_label": "Train",
                "render_key": "simulation",
                "render_label": "Standard sim",
                "created_at": "2026-08-01T00:00:00Z",
                "created_at_epoch": float(index),
            }
            for index in range(6)
        ]

        def load_candidate(
            _repo_root: Path,
            candidate: dict[str, object],
        ) -> dict[str, object]:
            return {
                "status": "available",
                "summary": {
                    "name": candidate["name"],
                    "root": str(candidate["root"]),
                    "episodes": 1,
                    "frames": 1,
                    "episode_lengths": [1],
                },
            }

        marked = {
            "training": {"cube_1"},
            "validation": {"cube_3"},
            "loop_test": {"cube_5"},
        }
        with (
            patch.object(dataset_viewer, "_dataset_catalog_candidates", return_value=candidates),
            patch.object(
                dataset_viewer,
                "selected_catalog_names",
                side_effect=lambda _root, role: marked[role],
            ),
            patch.object(
                dataset_viewer,
                "dataset_role_counts",
                return_value={"training": 1, "validation": 1, "loop_test": 1},
            ),
            patch.object(
                dataset_viewer,
                "_load_dataset_catalog_candidate",
                side_effect=load_candidate,
            ) as loader,
        ):
            payload = dataset_viewer._dataset_catalog_page_payload(
                Path("/tmp/catalog-repo"),
                {"page": ["1"], "size": ["10"], "marked": ["1"]},
            )

        self.assertTrue(payload["marked_only"])
        self.assertEqual(payload["total"], 3)
        self.assertEqual(loader.call_count, 3)
        self.assertEqual(
            [row["name"] for row in payload["data"]],
            ["cube_5", "cube_3", "cube_1"],
        )
        self.assertTrue(all(row["markedRoles"] for row in payload["data"]))

    def test_dataset_catalog_named_item_supports_deep_links_outside_current_page(self) -> None:
        candidate = {
            "name": "grip_the_cube_v4_4_train",
            "root": Path("/tmp/grip_the_cube_v4_4_train"),
            "category": "generated",
            "loader": "lerobot",
            "platform": "so101",
            "platform_label": "SO101",
            "split_key": "train",
            "split_label": "Train",
            "render_key": "photoreal",
            "render_label": "Photoreal",
            "created_at": "2026-08-01T00:00:00Z",
            "created_at_epoch": 1.0,
        }
        loaded = {
            "status": "available",
            "summary": {
                "name": candidate["name"],
                "root": str(candidate["root"]),
                "episodes": 2,
                "frames": 20,
                "episode_lengths": [10, 10],
            },
        }
        marked = {
            "training": {candidate["name"]},
            "validation": set(),
            "loop_test": set(),
        }
        with (
            patch.object(dataset_viewer, "_dataset_catalog_candidates", return_value=[candidate]),
            patch.object(dataset_viewer, "_load_dataset_catalog_candidate", return_value=loaded) as loader,
            patch.object(dataset_viewer, "_training_registry_entries_by_root", return_value={}),
            patch.object(
                dataset_viewer,
                "selected_catalog_names",
                side_effect=lambda _root, role: marked[role],
            ),
        ):
            payload = dataset_viewer._dataset_catalog_named_item_payload(
                Path("/tmp/catalog-repo"),
                "grip_the_cube_v4_4_train",
            )

        self.assertEqual(loader.call_count, 1)
        self.assertEqual(payload["data"]["name"], "grip_the_cube_v4_4_train")
        self.assertEqual(payload["data"]["markedRoles"], ["training"])

        with patch.object(dataset_viewer, "_dataset_catalog_candidates", return_value=[candidate]):
            with self.assertRaises(FileNotFoundError):
                dataset_viewer._dataset_catalog_named_item_payload(
                    Path("/tmp/catalog-repo"),
                    "missing_dataset",
                )

    def test_dataset_catalog_row_exposes_persistent_trainable_mark(self) -> None:
        root = Path("/tmp/marked_train").resolve()
        candidate = {
            "name": "marked_train",
            "root": root,
            "category": "generated",
            "loader": "lerobot",
            "platform": "so101",
            "platform_label": "SO101",
            "split_key": "train",
            "split_label": "Train",
            "render_key": "simulation",
            "render_label": "Standard sim",
            "created_at": "2026-08-01T00:00:00Z",
            "created_at_epoch": 1.0,
        }
        row = dataset_viewer._dataset_catalog_remote_row(
            candidate,
            {
                "status": "available",
                "summary": {"root": str(root), "episodes": 4, "frames": 40},
            },
            registry_by_root={},
            marked_trainable_roots={root},
        )

        self.assertTrue(row["markedTrainable"])
        self.assertEqual(row["markedRoles"], ["training"])
        self.assertIn("validation", row["roleEligibility"])
        self.assertIn("loop_test", row["roleEligibility"])
        self.assertFalse(row["trainingEligible"])
        self.assertIn("not registered", row["trainingEligibilityReason"])

    def test_dataset_catalog_splits_family_and_version_without_rewriting_data(self) -> None:
        cases = {
            "grip_the_cube_v4_3": ("grip_the_cube", "grip the cube", "v4.3"),
            "grip_the_cube_v4_3_at_validation": (
                "grip_the_cube",
                "grip the cube",
                "v4.3_at",
            ),
            "grip_the_cube_v4_3_near": (
                "grip_the_cube",
                "grip the cube",
                "v4.3_near",
            ),
            "grip_the_cube_near_v1_train200": (
                "grip_the_cube",
                "grip the cube",
                "v1_near",
            ),
            "grip_the_cube_real_pose_canary_v1": (
                "grip_the_cube",
                "grip the cube",
                "v1_real_pose_canary",
            ),
            "grip_the_cube_v2_5_align_trajectory_photoreal_validation": (
                "grip_the_cube",
                "grip the cube",
                "v2.5_align_trajectory",
            ),
            "pick_cube_train": ("pick_cube", "pick cube", "Unversioned"),
        }

        for name, expected in cases.items():
            with self.subTest(name=name):
                identity = dataset_viewer._dataset_catalog_identity(name)
                self.assertEqual(
                    (
                        identity["family_key"],
                        identity["family_label"],
                        identity["version_label"],
                    ),
                    expected,
                )

    def test_dataset_catalog_remote_filters_before_loading_summaries(self) -> None:
        candidates = [
            {
                "name": f"cube_{index}",
                "root": Path(f"/tmp/cube_{index}"),
                "category": "generated",
                "loader": "lerobot",
                "platform": "so101",
                "platform_label": "SO101",
                "split_key": "valid" if index % 2 else "train",
                "split_label": "Validation" if index % 2 else "Train",
                "render_key": "photoreal" if index % 3 == 1 else "simulation",
                "render_label": "Photoreal" if index % 3 == 1 else "Standard sim",
                "created_at": "2026-07-01T00:00:00Z",
                "created_at_epoch": float(index),
            }
            for index in range(30)
        ]

        def load_candidate(_repo_root: Path, candidate: dict[str, object]) -> dict[str, object]:
            return {
                "status": "available",
                "summary": {
                    "name": candidate["name"],
                    "root": str(candidate["root"]),
                    "episodes": 1,
                    "frames": 1,
                    "episode_lengths": [1],
                },
            }

        with (
            patch.object(dataset_viewer, "_dataset_catalog_candidates", return_value=candidates),
            patch.object(dataset_viewer, "_load_dataset_catalog_candidate", side_effect=load_candidate) as loader,
        ):
            payload = dataset_viewer._dataset_catalog_page_payload(
                Path("/tmp/catalog-repo"),
                {"page": ["1"], "size": ["10"], "status": ["Validation"], "type": ["Photoreal"]},
            )

        self.assertEqual(payload["total"], 5)
        self.assertEqual(loader.call_count, 5)
        self.assertTrue(all(row["splitLabel"] == "Validation" for row in payload["data"]))
        self.assertTrue(all(row["renderLabel"] == "Photoreal" for row in payload["data"]))

    def test_dataset_catalog_filters_family_and_version_before_loading(self) -> None:
        candidates = [
            {
                "name": name,
                "root": Path(f"/tmp/{name}"),
                "category": "generated",
                "loader": "lerobot",
                "platform": "so101",
                "platform_label": "SO101",
                "split_key": "train",
                "split_label": "Train",
                "render_key": "simulation",
                "render_label": "Standard sim",
                "created_at": "2026-08-01T00:00:00Z",
                "created_at_epoch": 1.0,
            }
            for name in (
                "grip_the_cube_v4_3",
                "grip_the_cube_v4_3_at",
                "grip_the_cube_v4_3_near",
                "pick_cube_v1",
            )
        ]

        def load_candidate(_repo_root: Path, candidate: dict[str, object]) -> dict[str, object]:
            return {
                "status": "available",
                "summary": {
                    "name": candidate["name"],
                    "root": str(candidate["root"]),
                    "episodes": 1,
                },
            }

        with (
            patch.object(dataset_viewer, "_dataset_catalog_candidates", return_value=candidates),
            patch.object(dataset_viewer, "_load_dataset_catalog_candidate", side_effect=load_candidate) as loader,
        ):
            payload = dataset_viewer._dataset_catalog_page_payload(
                Path("/tmp/catalog-repo"),
                {"family": ["grip the cube"], "version": ["v4.3_at"]},
            )

        self.assertEqual(payload["total"], 1)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(payload["data"][0]["familyLabel"], "grip the cube")
        self.assertEqual(payload["data"][0]["versionLabel"], "v4.3_at")

    def test_recipe_catalog_name_replaces_legacy_alias_for_same_root(self) -> None:
        root = Path("/tmp/canonical")

        def catalog_item(_repo_root: Path, split: str, _root: Path, *, category: str):
            summary = {"name": split, "root": str(root), "episodes": 1, "frames": 1}
            return {"name": split, "status": "available", "category": category, "summary": summary}

        with (
            patch.object(dataset_viewer, "_official_dataset_roots", return_value={"legacy_alias": root}),
            patch.object(dataset_viewer, "_skill_dataset_roots", return_value={}),
            patch.object(dataset_viewer, "_generation_recipe_dataset_roots", return_value={"canonical": root}),
            patch.object(dataset_viewer, "_dataset_catalog_item", side_effect=catalog_item),
            patch.object(dataset_viewer, "_discover_temporary_datasets", return_value={}),
            patch.object(dataset_viewer, "_discover_mycobot_datasets", return_value={}),
            patch.object(dataset_viewer, "ARCHIVED_DATASET_SPLITS", []),
        ):
            payload = dataset_viewer._build_datasets_payload(Path("/tmp"))

        self.assertIn("canonical", payload["datasets"])
        self.assertNotIn("legacy_alias", payload["datasets"])
        self.assertEqual(payload["dataset_groups"][0]["items"][0]["name"], "canonical")

    def test_recipe_photoreal_derivative_uses_photoreal_catalog_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "photoreal_derivative"
            root.mkdir()
            (root / "photoreal_lerobot_manifest.json").write_text(
                json.dumps({"format": "so101_photoreal_lerobot_v1"}), encoding="utf-8"
            )

            def catalog_item(_repo_root: Path, split: str, _root: Path, *, category: str):
                summary = {"name": split, "root": str(root), "episodes": 1, "frames": 1}
                return {
                    "name": split,
                    "status": "available",
                    "category": category,
                    "summary": summary,
                }

            with (
                patch.object(dataset_viewer, "_official_dataset_roots", return_value={}),
                patch.object(dataset_viewer, "_skill_dataset_roots", return_value={}),
                patch.object(
                    dataset_viewer,
                    "_generation_recipe_dataset_roots",
                    return_value={"photoreal_derivative": root},
                ),
                patch.object(dataset_viewer, "_dataset_catalog_item", side_effect=catalog_item),
                patch.object(dataset_viewer, "_discover_temporary_datasets", return_value={}),
                patch.object(dataset_viewer, "_discover_so101_photoreal_datasets", return_value={}),
                patch.object(
                    dataset_viewer, "_discover_so101_photoreal_lerobot_datasets", return_value={}
                ),
                patch.object(dataset_viewer, "_discover_mycobot_datasets", return_value={}),
                patch.object(dataset_viewer, "ARCHIVED_DATASET_SPLITS", []),
            ):
                payload = dataset_viewer._build_datasets_payload(Path(tmpdir))

        groups = {group["id"]: group["items"] for group in payload["dataset_groups"]}
        self.assertEqual(groups["generated"], [])
        self.assertEqual(groups["photoreal"][0]["name"], "photoreal_derivative")
        self.assertEqual(groups["photoreal"][0]["category"], "photoreal")

    def test_standard_sim_phase_manifest_does_not_mark_dataset_photoreal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "standard_phase"
            root.mkdir()
            (root / "photoreal_lerobot_manifest.json").write_text(
                json.dumps(
                    {
                        "operation": "phase_subset",
                        "source_dataset_name": "grip_the_cube_v4_4",
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(dataset_viewer, "_official_dataset_roots", return_value={}),
                patch.object(dataset_viewer, "_skill_dataset_roots", return_value={}),
                patch.object(
                    dataset_viewer,
                    "_generation_recipe_dataset_roots",
                    return_value={"grip_the_cube_v4_4_move_phase_v1": root},
                ),
                patch.object(dataset_viewer, "_generation_closed_loop_views", return_value={}),
                patch.object(dataset_viewer, "_discover_temporary_datasets", return_value={}),
                patch.object(dataset_viewer, "_discover_so101_photoreal_datasets", return_value={}),
                patch.object(
                    dataset_viewer, "_discover_so101_photoreal_lerobot_datasets", return_value={}
                ),
                patch.object(dataset_viewer, "_discover_mycobot_datasets", return_value={}),
                patch.object(dataset_viewer, "_official_closed_loop_test_cases", return_value=[]),
                patch.object(dataset_viewer, "ARCHIVED_DATASET_SPLITS", []),
            ):
                candidates = dataset_viewer._dataset_catalog_candidates(Path(tmpdir))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["category"], "generated")
        self.assertEqual(candidates[0]["render_key"], "simulation")
        self.assertEqual(candidates[0]["render_label"], "Standard sim")

    def test_qwen_chain_loop_test_id_keeps_nact15_variant_distinct(self) -> None:
        self.assertEqual(
            exporter._loop_test_id_from_report_path(
                Path("closed_loop_evals/qwen_chain_seed98100_001792/qwen_closed_loop_eval_report.json"),
                "001792",
            ),
            "qwen_chain_001792",
        )
        self.assertEqual(
            exporter._loop_test_id_from_report_path(
                Path("closed_loop_evals/qwen_chain_seed98100_nact15_001792/qwen_closed_loop_eval_report.json"),
                "001792",
            ),
            "qwen_chain_nact15_001792",
        )

    def test_build_export_converts_qwen_chain_trace_to_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            output_dir = root / "export"
            report_dir = run_dir / "closed_loop_evals" / "qwen_chain_seed98100_000224"
            metrics_dir = run_dir / "metrics"
            report_dir.mkdir(parents=True)
            metrics_dir.mkdir(parents=True)
            trace_path = report_dir / "qwen_closed_loop_episode_000.jsonl"
            trace_path.write_text(
                "\n".join(
                    [
                        json.dumps(_record(0, "move", "move_over_cube_edge")),
                        json.dumps(_record(1, "move", "move_over_cube_edge")),
                        json.dumps(_record(2, "align", "align_fixed_jaw_cube_edge")),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (metrics_dir / "validation_metrics.jsonl").write_text(
                json.dumps({"checkpoint": "000224", "step": 224, "loss": 0.1}) + "\n",
                encoding="utf-8",
            )
            (report_dir / "qwen_closed_loop_eval_report.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "success_rate": 0.0,
                        "episodes_requested": 1,
                        "episodes_completed": 1,
                        "seed": 98100,
                        "policy_rollout_config": {
                            "chunk_size": 50,
                            "n_action_steps": 15,
                            "num_steps": 10,
                        },
                        "plan": {
                            "model": "qwen3-vl-8b-instruct-mlx",
                            "task": "pick and lift the green cube",
                            "thinking_mode": "non-thinking",
                            "calls": [{"fn": "move"}, {"fn": "align"}],
                        },
                        "episodes": [
                            {
                                "episode": 0,
                                "final_success": False,
                                "total_reward": 3.0,
                                "steps": 3,
                                "trace_path": str(trace_path),
                                "final_info": {"success": False},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = exporter.build_export(run_dir, output_dir, copy_source=True)
            payload = server._loop_tests_payload(output_dir)
            detail = server._loop_test_detail(output_dir, "qwen_chain_000224")

        self.assertEqual(manifest["summary"]["loop_tests"], 1)
        self.assertEqual(payload["loop_tests"][0]["validation_loss"], 0.1)
        self.assertEqual(detail["loop_test"]["status_meaning"], "evaluator_completed")
        self.assertIn("system", detail["loop_test"]["qwen_prompts"])
        timeline = detail["episodes"][0]["timeline"]
        self.assertEqual(timeline[0]["type"], "planner_call")
        self.assertIn("system_prompt", timeline[0]["policy_input"])
        self.assertIn("tool_call_start", [row["type"] for row in timeline])
        self.assertIn("tool_call_end", [row["type"] for row in timeline])
        self.assertEqual([row["iteration"] for row in timeline if row["type"] == "tool_call_start"], [1, 2])
        first_tool = next(row for row in timeline if row["type"] == "tool_call_start")
        self.assertEqual(first_tool["tool_parameters"]["primitive_id"], "move_over_cube_edge")
        first_step = next(row for row in timeline if row["type"] == "policy_step")
        self.assertEqual(first_step["policy_output"]["action_chunk"]["generated_count"], 50)
        self.assertEqual(first_step["policy_output"]["action_chunk"]["used_per_chunk"], 15)
        self.assertTrue(first_step["policy_output"]["action_chunk"]["confirmed_in_rollout"])
        self.assertEqual(detail["episodes"][0]["iterations"][0]["action_chunk_summary"]["chunk_count"], 1)

    def test_media_generation_command_uses_export_parent_as_run_dir(self) -> None:
        repo_root = Path("/repo")
        export_dir = Path("/run/qwen_edge_primitives/loop_test_analyzer_export")
        command = server._media_generation_command(repo_root, export_dir, python_executable="/python")

        self.assertEqual(command[0], "/python")
        self.assertIn(str(repo_root / "scripts" / "build_loop_test_analyzer_export.py"), command)
        self.assertIn("--copy-source", command)
        self.assertIn("--generate-media", command)
        self.assertEqual(command[command.index("--run-dir") + 1], str(export_dir.parent))
        self.assertEqual(command[command.index("--output-dir") + 1], str(export_dir))

    def test_media_generation_command_targets_selected_loop(self) -> None:
        repo_root = Path("/repo")
        export_dir = Path("/run/qwen_edge_primitives/loop_test_analyzer_export")
        command = server._media_generation_command(
            repo_root,
            export_dir,
            python_executable="/python",
            loop_test_id="qwen_chain_nact15_003136",
        )

        self.assertIn("--loop-test-id", command)
        self.assertEqual(command[command.index("--loop-test-id") + 1], "qwen_chain_nact15_003136")

    def test_filtered_export_preserves_existing_loop_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            output_dir = root / "export"
            _write_loop_report(run_dir, "qwen_chain_seed98100_000224", [_record(0, "move", "move_over_cube_edge")])
            _write_loop_report(run_dir, "qwen_chain_seed98100_000448", [_record(0, "align", "align_fixed_jaw_cube_edge")])

            exporter.build_export(run_dir, output_dir)
            manifest = exporter.build_export(run_dir, output_dir, loop_test_ids=["qwen_chain_000224"])

        self.assertEqual(manifest["summary"]["loop_tests"], 2)
        self.assertEqual(
            [row["loop_test_id"] for row in manifest["loop_tests"]],
            ["qwen_chain_000224", "qwen_chain_000448"],
        )

    def test_export_refuses_to_relabel_mismatched_camera_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            output_dir = root / "export"
            report_dir = run_dir / "closed_loop_evals" / "qwen_chain_seed98100_000224"
            report_dir.mkdir(parents=True)
            trace_path = report_dir / "qwen_closed_loop_episode_000.jsonl"
            bad_record = _record(0, "move", "move_over_cube_edge")
            bad_record["image_feature_mapping"] = {"observation.images.camera1": "wrist_cam"}
            trace_path.write_text(json.dumps(bad_record) + "\n", encoding="utf-8")
            (report_dir / "qwen_closed_loop_eval_report.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "success_rate": 0.0,
                        "episodes_requested": 1,
                        "episodes_completed": 1,
                        "seed": 98100,
                        "camera_contract": {
                            "observation.images.camera1": "egocentric_cam",
                            "observation.images.camera2": "wrist_cam",
                        },
                        "plan": {"model": "qwen3-vl-8b-instruct-mlx", "task": "pick and lift the green cube", "calls": []},
                        "episodes": [{"episode": 0, "steps": 1, "trace_path": str(trace_path)}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "camera mapping mismatch"):
                exporter.build_export(run_dir, output_dir)

    def test_replay_env_config_falls_back_to_green_cube_plan(self) -> None:
        report = {
            "plan": {
                "task": "pick and lift the green cube",
                "calls": [
                    {
                        "fn": "move",
                        "object": "green cube",
                        "prompt": "Move the gripper above one visible green cube edge.",
                    }
                ],
            }
        }

        env_config = exporter._env_config_for_replay(report, {})

        self.assertEqual(env_config["object_shape"], "cube")
        self.assertEqual(env_config["object_color"], "green")
        self.assertEqual(env_config["n_distractors"], 0)

    def test_replay_env_config_prefers_recorded_env_config(self) -> None:
        report = {"env_config": {"object_shape": "cube", "object_color": "green", "n_distractors": 0}}
        replay = {"env_config": {"object_shape": "cube", "object_color": "red", "n_distractors": 0}}

        env_config = exporter._env_config_for_replay(report, replay)

        self.assertEqual(env_config["object_color"], "green")

    def test_media_generation_repo_root_prefers_export_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "physical_ai_agent"
            export_dir = workspace / "_workspace" / "run" / "loop_test_analyzer_export"
            (workspace / "scripts").mkdir(parents=True)
            (workspace / "src").mkdir()
            (workspace / "scripts" / "build_loop_test_analyzer_export.py").write_text("", encoding="utf-8")

            self.assertEqual(server._media_generation_repo_root(Path("/server"), export_dir), workspace)

    def test_media_generation_python_prefers_export_workspace_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "physical_ai_agent"
            export_dir = workspace / "_workspace" / "run" / "loop_test_analyzer_export"
            python_path = workspace / ".venv" / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_text("", encoding="utf-8")

            self.assertEqual(
                server._media_generation_python(Path("/server"), export_dir, python_executable="/fallback"),
                str(python_path),
            )

    def test_dataset_viewer_exposes_closed_loop_prompt_and_start_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "physical_ai_agent"
            config_dir = repo_root / "configs" / "so101" / "training_datasets"
            config_dir.mkdir(parents=True)
            (config_dir / "qwen_edge_primitives.json").write_text(
                json.dumps(
                    {
                        "closed_loop": {
                            "test_cases": [
                                {
                                    "id": "move_over_cube_edge",
                                    "description": "Move primitive closed-loop test.",
                                    "episodes": 10,
                                    "steps": 120,
                                    "seed": 98100,
                                    "start_contract": "move_over_cube_edge",
                                    "task_prompt": "Move the gripper above one visible green cube edge.",
                                    "plan_json": "configs/agent/qwen3_so101_tool_plan_move_over_cube_edge_green_cube.json",
                                },
                                {
                                    "id": "align_fixed_jaw_cube_edge",
                                    "description": "Align primitive closed-loop test.",
                                    "episodes": 10,
                                    "steps": 120,
                                    "seed": 98200,
                                    "start_contract": "align_fixed_jaw_cube_edge",
                                    "task_prompt": "Align the gripper jaws around one visible green cube edge.",
                                    "plan_json": "configs/agent/qwen3_so101_tool_plan_align_fixed_jaw_cube_edge_green_cube.json",
                                },
                                {
                                    "id": "grip_from_edge_cube",
                                    "description": "Grip primitive closed-loop test.",
                                    "episodes": 10,
                                    "steps": 120,
                                    "seed": 98300,
                                    "start_contract": "grip_from_edge_cube",
                                    "task_prompt": "Close the gripper on the green cube edge and lift.",
                                    "plan_json": "configs/agent/qwen3_so101_tool_plan_grip_from_edge_cube_green_cube.json",
                                },
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            run_dir = repo_root / "_workspace" / "so101_training" / "runs" / "debug_run"
            loop_dir = run_dir / "closed_loop_evals" / "qwen_chain_grip_from_edge_cube_seed98300_000086"
            episode_dir = loop_dir
            media_dir = episode_dir / "media"
            policy_dir = media_dir / "policy_inputs"
            robot_dir = media_dir / "robot_frames"
            video_dir = media_dir / "videos"
            policy_dir.mkdir(parents=True)
            robot_dir.mkdir(parents=True)
            video_dir.mkdir(parents=True)
            tiny_png = b"\x89PNG\r\n\x1a\n"
            (policy_dir / "step_0000_egocentric_cam.png").write_bytes(tiny_png)
            (robot_dir / "step_0000_top_down.png").write_bytes(tiny_png)
            (video_dir / "iteration_01_grip_from_edge_cube.gif").write_bytes(b"GIF89a")
            timeline_path = episode_dir / "qwen_closed_loop_episode_000.jsonl"
            timeline_path.write_text(
                json.dumps(
                    {
                        "episode": 0,
                        "global_step": 0,
                        "primitive_step": 0,
                        "fn": "pick_up",
                        "primitive_id": "grip_from_edge_cube",
                        "prompt": "Close the gripper on the green cube edge and lift.",
                        "image_feature_mapping": {
                            "observation.images.camera1": "egocentric_cam",
                            "observation.images.camera2": "wrist_cam",
                        },
                        "action": [0.0],
                        "policy_rollout_config": {"n_action_steps": 15},
                        "info": {"tcp_to_obj_dist": 0.12, "is_grasped": 0.0},
                        "media": {
                            "policy_input_images": {
                                "egocentric_cam": str(policy_dir / "step_0000_egocentric_cam.png"),
                            },
                            "robot_frame": str(robot_dir / "step_0000_top_down.png"),
                            "iteration_video_gif": str(video_dir / "iteration_01_grip_from_edge_cube.gif"),
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (loop_dir / "qwen_closed_loop_eval_report.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "env_id": "MuJoCoPickLift-v1",
                        "seed": 98300,
                        "start_contract": "grip_from_edge_cube",
                        "success_rate": 0.0,
                        "episodes_completed": 1,
                        "camera_contract": {
                            "observation.images.camera1": "egocentric_cam",
                            "observation.images.camera2": "wrist_cam",
                        },
                        "plan": {"task": "Close the gripper on the green cube edge and lift.", "calls": []},
                        "qwen_prompts": {"user": "Task: Close the gripper on the green cube edge and lift."},
                        "episodes": [
                            {
                                "episode": 0,
                                "final_success": False,
                                "total_reward": 0.1,
                                "steps": 1,
                                "reset_info": {"tcp_to_obj_dist": 0.16},
                                "final_info": {"tcp_to_obj_dist": 0.12, "is_grasped": 0.0, "lift_height": 0.0},
                                "trace_path": str(timeline_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            listing = dataset_viewer._loop_tests_payload(repo_root)
            pick_item = listing["exports"][0]["loop_tests"][2]
            detail = dataset_viewer._loop_frame_payload(
                repo_root,
                listing["exports"][0]["id"],
                pick_item["loop_test_id"],
                0,
                0,
            )

        self.assertEqual(listing["source"], "official_closed_loop_test_cases")
        self.assertEqual(listing["exports"][0]["id"], "official_qwen_edge_test_cases")
        self.assertEqual(
            [item["test_case_id"] for item in listing["exports"][0]["loop_tests"]],
            ["move_over_cube_edge", "align_fixed_jaw_cube_edge", "grip_from_edge_cube"],
        )
        self.assertEqual(listing["exports"][0]["loop_tests"][0]["status"], "not_run")
        self.assertEqual(pick_item["checkpoint"], "000086")
        self.assertEqual(pick_item["configured_seed"], 98300)
        self.assertEqual(pick_item["start_contract"], "grip_from_edge_cube")
        self.assertEqual(pick_item["loop_test_id"], "qwen_chain_grip_from_edge_cube_seed98300_000086")
        self.assertEqual(pick_item["scenario"], "Close the gripper on the green cube edge and lift.")
        self.assertEqual(detail["plan"]["task"], "Close the gripper on the green cube edge and lift.")
        self.assertEqual(detail["step"]["policy_input_prompt"], "Close the gripper on the green cube edge and lift.")
        self.assertIn("egocentric_cam", detail["images"]["policy_inputs"])
        self.assertIn("data:image/png;base64", detail["images"]["policy_inputs"]["egocentric_cam"])
        self.assertIn("egocentric_cam", detail["start_images"]["policy_inputs"])
        self.assertIn("top_down", detail["start_images"]["robot_frames"])
        self.assertIn("top_down", detail["images"]["robot_frames"])
        self.assertEqual(detail["start_video"]["name"], "iteration_01_grip_from_edge_cube.gif")

    def test_dataset_viewer_closed_loop_preview_http_e2e(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "physical_ai_agent"
            _write_viewer_closed_loop_fixture(repo_root)
            html = _handle_viewer_request(repo_root, "/")
            listing = _handle_viewer_request(repo_root, "/api/loop-tests")
            preview = _handle_viewer_request(
                repo_root,
                "/api/loop-frame"
                "?export=official_qwen_edge_test_cases"
                "&loop=qwen_chain_grip_from_edge_cube_seed98300_000086"
                "&episode=0"
                "&step=0",
            )

        self.assertIn('id="loopPolicyCameras"', html)
        self.assertIn('id="loopStartCameras"', html)
        self.assertIn('href="/vendor/tabulator.min.css"', html)
        self.assertIn('src="/vendor/tabulator.min.js"', html)
        self.assertIn('id="datasetCatalogGrid"', html)
        self.assertIn('id="catalogLoading"', html)
        self.assertIn('id="catalogLoadingBar"', html)
        self.assertIn('id="catalogLoadingPercent"', html)
        self.assertIn('fetch("/api/datasets/progress"', html)
        self.assertIn("function renderCatalogLoading(progress = {})", html)
        self.assertIn("function finishCatalogLoading()", html)
        self.assertIn("generation === catalogLoadGeneration", html)
        self.assertIn('ajaxURL: "/api/datasets/catalog"', html)
        self.assertIn("ajaxRequestFunc: requestDatasetCatalogPage", html)
        self.assertIn('paginationMode: "remote"', html)
        self.assertIn('filterMode: "remote"', html)
        self.assertIn('sortMode: "remote"', html)
        self.assertIn("paginationSize: 10", html)
        self.assertNotIn('fetch("/api/datasets",', html)
        self.assertIn('id="catalogPreviousPage"', html)
        self.assertIn('id="catalogNextPage"', html)
        self.assertIn('id="catalogPageStatus"', html)
        self.assertIn('id="catalogBulkAction"', html)
        self.assertIn('id="catalogApplyBulkAction"', html)
        self.assertIn('id="catalogMarkedOnly"', html)
        self.assertIn('if (catalogMarkedOnly) query.set("marked", "1")', html)
        self.assertIn('catalogMarkedOnlyButton.addEventListener("click"', html)
        self.assertIn("function initialViewStateFromUrl()", html)
        self.assertIn("function syncCurrentViewUrl()", html)
        self.assertIn("window.history.replaceState", html)
        self.assertIn('fetch(`/api/datasets/catalog/item?name=${encodeURIComponent(name)}`', html)
        self.assertIn("paginationInitialPage: initialViewState.catalogPage", html)
        self.assertIn("initialHeaderFilter: initialCatalogHeaderFilters()", html)
        self.assertIn("restoreInitialViewState()", html)
        self.assertIn("fps: initialViewState.fps", html)
        self.assertIn("if (options.fps) fps.value = String(options.fps)", html)
        self.assertIn('className = "dataset-select-checkbox dataset-select-page-checkbox"', html)
        self.assertIn('className = "dataset-select-checkbox dataset-row-select-checkbox"', html)
        self.assertIn("Mark as training set", html)
        self.assertIn("Remove from training set", html)
        self.assertIn("Mark as validation set", html)
        self.assertIn("Remove from validation set", html)
        self.assertIn("Mark as loop test set", html)
        self.assertIn("Remove from loop test set", html)
        self.assertIn("Delete selected datasets", html)
        self.assertIn('fetch("/api/datasets/role-selection"', html)
        self.assertIn('fetch("/api/datasets/bulk-delete"', html)
        self.assertIn('class="trainable-set-badge"', html)
        self.assertIn('class="trainable-set-badge validation-role"', html)
        self.assertIn('class="trainable-set-badge loop-test-role"', html)
        self.assertIn("not eligible for a dataset role", html)
        self.assertNotIn("no selectable training role", html)
        self.assertIn("const datasetCatalogPageCache = new Map()", html)
        self.assertIn("function getCachedCatalogPage(key)", html)
        self.assertIn("function cacheCatalogPage(key, payload)", html)
        self.assertIn("function clearCatalogPageCache()", html)
        self.assertIn("restored from page cache", html)
        self.assertIn('class="dataset-catalog-shell">\n\t        <div id="catalogLoading"', html)
        self.assertIn('class="playback-actions"', html)
        self.assertIn('class="playback-icon-button"', html)
        self.assertIn("initialSort:", html)
        self.assertIn('title: "Status"', html)
        self.assertIn('title: "Type"', html)
        self.assertIn('title: "Dataset"', html)
        self.assertIn('field: "familyLabel"', html)
        self.assertIn('title: "Version"', html)
        self.assertIn('field: "versionLabel"', html)
        self.assertIn('query.set("family", filters.familyLabel)', html)
        self.assertIn('query.set("version", filters.versionLabel)', html)
        self.assertIn('class="dataset-id"', html)
        self.assertIn("function catalogSelectHeaderFilter(values)", html)
        self.assertIn('headerFilter: catalogSelectHeaderFilter(["Train", "Validation", "Closed loop"])', html)
        self.assertIn('title: "Created"', html)
        self.assertIn('class="dataset-delete-button"', html)
        self.assertIn('fetch("/api/datasets/delete"', html)
        self.assertIn("window.confirm(", html)
        self.assertIn("window.prompt(", html)
        self.assertIn("function isPrivateViewerHost()", html)
        self.assertIn('id="split" hidden', html)
        self.assertIn('id="episodeValue"', html)
        self.assertIn('id="frameValue"', html)
        self.assertIn('function splitForDataset(name)', html)
        self.assertIn('function renderTypeForDataset(name)', html)
        self.assertIn('Split: ${escapeHtml(splitLabel(datasetSplitByName[name]))}', html)
        self.assertIn('Render: ${escapeHtml(renderTypeLabel(datasetRenderTypeByName[name]))}', html)
        self.assertIn('function selectDataset(name, options = {})', html)
        self.assertNotIn('id="platformKind"', html)
        self.assertNotIn('id="viewKind"', html)
        self.assertNotIn('id="datasetTab"', html)
        self.assertNotIn('id="loopTab"', html)
        self.assertIn("Episode start images", html)
        self.assertIn("loopPlaybackTick", html)
        self.assertEqual(listing["source"], "official_closed_loop_test_cases")
        pick_item = listing["exports"][0]["loop_tests"][2]
        self.assertEqual(pick_item["test_case_id"], "grip_from_edge_cube")
        self.assertEqual(pick_item["configured_seed"], 98300)
        self.assertEqual(pick_item["start_contract"], "grip_from_edge_cube")
        self.assertEqual(pick_item["scenario"], "Close the gripper on the green cube edge and lift.")
        self.assertFalse(pick_item["loop_test_id"] == "")
        self.assertEqual(preview["loop_test"]["id"], "qwen_chain_grip_from_edge_cube_seed98300_000086")
        self.assertEqual(preview["step"]["policy_input_prompt"], "Close the gripper on the green cube edge and lift.")
        self.assertIn("egocentric_cam", preview["images"]["policy_inputs"])
        self.assertIn("egocentric_cam", preview["start_images"]["policy_inputs"])
        self.assertIn("wrist_cam", preview["images"]["policy_inputs"])
        self.assertIn("wrist_cam", preview["start_images"]["policy_inputs"])
        self.assertIn("top_down", preview["images"]["robot_frames"])
        self.assertIn("top_down", preview["start_images"]["robot_frames"])
        self.assertTrue(preview["images"]["policy_inputs"]["egocentric_cam"].startswith("data:image/png;base64,"))

    def test_dataset_viewer_serves_vendored_tabulator_assets(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        javascript = _handle_viewer_request(repo_root, "/vendor/tabulator.min.js")
        stylesheet = _handle_viewer_request(repo_root, "/vendor/tabulator.min.css")

        self.assertIn("Tabulator", javascript[:1000])
        self.assertIn(".tabulator", stylesheet[:1000])
        self.assertTrue((repo_root / "third_party/tabulator/LICENSE").is_file())

    def test_dataset_viewer_does_not_rewrite_report_plan_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "physical_ai_agent"
            _write_viewer_closed_loop_fixture(repo_root)
            report_path = (
                repo_root
                / "_workspace"
                / "so101_training"
                / "runs"
                / "debug_run"
                / "closed_loop_evals"
                / "qwen_chain_grip_from_edge_cube_seed98300_000086"
                / "qwen_closed_loop_eval_report.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["plan"]["task"] = "stale report prompt"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            preview = _handle_viewer_request(
                repo_root,
                "/api/loop-frame"
                "?export=official_qwen_edge_test_cases"
                "&loop=qwen_chain_grip_from_edge_cube_seed98300_000086"
                "&episode=0"
                "&step=0",
            )

        self.assertEqual(preview["loop_test"]["scenario"], "Close the gripper on the green cube edge and lift.")
        self.assertEqual(preview["plan"]["task"], "stale report prompt")
        self.assertTrue(preview["start_images"]["policy_inputs"]["egocentric_cam"].startswith("data:image/png;base64,"))
        self.assertEqual(preview["step"]["media_available"], True)

    def test_dataset_viewer_official_roots_include_qwen_edge_train_and_validation_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "physical_ai_agent"
            config_dir = repo_root / "configs" / "so101" / "training_datasets"
            config_dir.mkdir(parents=True)
            for rel in (
                "_workspace/hf_datasets/mhlee1215__so101-nexus-sim-dataset/datasets/move_over_cube_edge/train",
                "_workspace/hf_datasets/mhlee1215__so101-nexus-sim-dataset/datasets/align_fixed_jaw_cube_edge/train",
                "_workspace/hf_datasets/mhlee1215__so101-nexus-sim-dataset/datasets/grip_from_edge_cube/train",
                "_workspace/hf_datasets/mhlee1215__so101-nexus-sim-dataset/datasets/move_over_cube_edge/validation",
                "_workspace/hf_datasets/mhlee1215__so101-nexus-sim-dataset/datasets/align_fixed_jaw_cube_edge/validation",
                "_workspace/hf_datasets/mhlee1215__so101-nexus-sim-dataset/datasets/grip_from_edge_cube/validation",
            ):
                (repo_root / rel).mkdir(parents=True)
            (config_dir / "qwen_edge_primitives.json").write_text(
                json.dumps(
                    {
                        "train_datasets": [
                            {
                                "name": "move_over_cube_edge_train",
                                "root": "_workspace/hf_datasets/mhlee1215__so101-nexus-sim-dataset/datasets/move_over_cube_edge/train",
                            },
                            {
                                "name": "align_fixed_jaw_cube_edge_train",
                                "root": "_workspace/hf_datasets/mhlee1215__so101-nexus-sim-dataset/datasets/align_fixed_jaw_cube_edge/train",
                            },
                            {
                                "name": "grip_from_edge_cube_train",
                                "root": "_workspace/hf_datasets/mhlee1215__so101-nexus-sim-dataset/datasets/grip_from_edge_cube/train",
                            },
                        ],
                        "validation_dataset": {
                            "hf_merge_sources": [
                                {
                                    "name": "move_over_cube_edge_val",
                                    "hf_path_in_repo": "datasets/move_over_cube_edge/validation",
                                },
                                {
                                    "name": "align_fixed_jaw_cube_edge_val",
                                    "hf_path_in_repo": "datasets/align_fixed_jaw_cube_edge/validation",
                                },
                                {
                                    "name": "grip_from_edge_cube_val",
                                    "hf_path_in_repo": "datasets/grip_from_edge_cube/validation",
                                },
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            roots = dataset_viewer._official_dataset_roots(repo_root)

        self.assertEqual(
            set(roots),
            {
                "move_over_cube_edge_train",
                "align_fixed_jaw_cube_edge_train",
                "grip_from_edge_cube_train",
                "move_over_cube_edge_val",
                "align_fixed_jaw_cube_edge_val",
                "grip_from_edge_cube_val",
            },
        )

    def test_media_job_status_round_trips_in_export_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir) / "export"
            payload = {"status": "running", "progress": {"percent": 12.5}}

            server._save_media_job_status(export_dir, payload)

            self.assertEqual(server._load_media_job_status(export_dir), payload)

    def test_media_artifact_progress_counts_generated_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            export_dir = run_dir / "loop_test_analyzer_export"
            report_dir = run_dir / "closed_loop_evals" / "qwen_chain_seed98100_000224"
            other_report_dir = run_dir / "closed_loop_evals" / "qwen_chain_seed98100_000448"
            episode_dir = export_dir / "loop_tests" / "qwen_chain_000224" / "episodes" / "episode_000" / "media"
            report_dir.mkdir(parents=True)
            other_report_dir.mkdir(parents=True)
            trace_path = report_dir / "trace.jsonl"
            trace_path.write_text(json.dumps(_record(0, "move", "move_over_cube_edge")) + "\n", encoding="utf-8")
            (report_dir / "qwen_closed_loop_eval_report.json").write_text(
                json.dumps({"episodes": [{"trace_path": str(trace_path)}]}),
                encoding="utf-8",
            )
            other_trace_path = other_report_dir / "trace.jsonl"
            other_trace_path.write_text(
                json.dumps(_record(0, "align", "align_fixed_jaw_cube_edge"))
                + "\n"
                + json.dumps(_record(1, "align", "align_fixed_jaw_cube_edge"))
                + "\n",
                encoding="utf-8",
            )
            (other_report_dir / "qwen_closed_loop_eval_report.json").write_text(
                json.dumps({"episodes": [{"trace_path": str(other_trace_path)}]}),
                encoding="utf-8",
            )
            for folder, name in [
                ("policy_inputs", "step_0000_egocentric_cam.png"),
                ("policy_inputs", "step_0000_wrist_cam.png"),
                ("robot_frames", "step_0000_top_down.png"),
                ("videos", "iteration_01_move.gif"),
            ]:
                path = episode_dir / folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            progress = server._media_artifact_progress(export_dir, loop_test_id="qwen_chain_000224")

            self.assertEqual(progress["source_rollout_records"], 1)
            self.assertEqual(progress["expected_png_files"], 3)
            self.assertEqual(progress["png_files"], 3)
            self.assertEqual(progress["gif_files"], 1)
            self.assertEqual(progress["percent"], 100.0)
            self.assertEqual(progress["loop_test_id"], "qwen_chain_000224")
            self.assertEqual(server._media_artifact_progress(export_dir)["source_rollout_records"], 3)


def _record(step: int, fn: str, primitive_id: str) -> dict:
    return {
        "episode": 0,
        "global_step": step,
        "primitive_step": step,
        "fn": fn,
        "primitive_id": primitive_id,
        "prompt": f"{fn} prompt",
        "policy_path": "/policy",
        "observation": [float(step)],
        "action": [0.1, 0.2],
        "reward": 1.0,
        "info": {"tcp_to_target_dist": 0.1, "success": False},
        "terminated": False,
        "truncated": False,
        "image_feature_mapping": {
            "observation.images.camera1": "egocentric_cam",
            "observation.images.camera2": "wrist_cam",
            "observation.images.camera3": "wrist_cam",
        },
    }


def _write_loop_report(run_dir: Path, report_name: str, records: list[dict]) -> None:
    report_dir = run_dir / "closed_loop_evals" / report_name
    report_dir.mkdir(parents=True, exist_ok=True)
    trace_path = report_dir / "qwen_closed_loop_episode_000.jsonl"
    trace_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    (report_dir / "qwen_closed_loop_eval_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "success_rate": 0.0,
                "episodes_requested": 1,
                "episodes_completed": 1,
                "seed": 98100,
                "plan": {"model": "qwen3-vl-8b-instruct-mlx", "task": "pick and lift the green cube", "calls": []},
                "episodes": [
                    {
                        "episode": 0,
                        "final_success": False,
                        "total_reward": 1.0,
                        "steps": len(records),
                        "trace_path": str(trace_path),
                        "final_info": {"success": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_viewer_closed_loop_fixture(repo_root: Path) -> None:
    config_dir = repo_root / "configs" / "so101" / "training_datasets"
    config_dir.mkdir(parents=True)
    (config_dir / "qwen_edge_primitives.json").write_text(
        json.dumps(
            {
                "closed_loop": {
                    "test_cases": [
                        {
                            "id": "move_over_cube_edge",
                            "description": "Move primitive closed-loop test.",
                            "episodes": 10,
                            "steps": 120,
                            "seed": 98100,
                            "start_contract": "move_over_cube_edge",
                            "task_prompt": "Move the gripper above one visible green cube edge.",
                            "env_object_color": "green",
                            "plan_json": "configs/agent/qwen3_so101_tool_plan_move_over_cube_edge_green_cube.json",
                        },
                        {
                            "id": "align_fixed_jaw_cube_edge",
                            "description": "Align primitive closed-loop test.",
                            "episodes": 10,
                            "steps": 120,
                            "seed": 98200,
                            "start_contract": "align_fixed_jaw_cube_edge",
                            "task_prompt": "Align the gripper jaws around one visible green cube edge.",
                            "env_object_color": "green",
                            "plan_json": "configs/agent/qwen3_so101_tool_plan_align_fixed_jaw_cube_edge_green_cube.json",
                        },
                        {
                            "id": "grip_from_edge_cube",
                            "description": "Grip primitive closed-loop test.",
                            "episodes": 10,
                            "steps": 120,
                            "seed": 98300,
                            "start_contract": "grip_from_edge_cube",
                            "task_prompt": "Close the gripper on the green cube edge and lift.",
                            "env_object_color": "green",
                            "plan_json": "configs/agent/qwen3_so101_tool_plan_grip_from_edge_cube_green_cube.json",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    run_dir = repo_root / "_workspace" / "so101_training" / "runs" / "debug_run"
    loop_dir = run_dir / "closed_loop_evals" / "qwen_chain_grip_from_edge_cube_seed98300_000086"
    media_dir = loop_dir / "media"
    policy_dir = media_dir / "policy_inputs"
    robot_dir = media_dir / "robot_frames"
    policy_dir.mkdir(parents=True)
    robot_dir.mkdir(parents=True)
    (policy_dir / "step_0000_egocentric_cam.png").write_bytes(VALID_1X1_PNG)
    (policy_dir / "step_0000_wrist_cam.png").write_bytes(VALID_1X1_PNG)
    (robot_dir / "step_0000_top_down.png").write_bytes(VALID_1X1_PNG)
    trace_path = loop_dir / "qwen_closed_loop_episode_000.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "episode": 0,
                "global_step": 0,
                "primitive_step": 0,
                "fn": "pick_up",
                "primitive_id": "grip_from_edge_cube",
                "prompt": "Close the gripper on the green cube edge and lift.",
                "image_feature_mapping": {
                    "observation.images.camera1": "egocentric_cam",
                    "observation.images.camera2": "wrist_cam",
                },
                "policy_rollout_config": {"n_action_steps": 15},
                "action": [0.0],
                "info": {"tcp_to_obj_dist": 0.12, "is_grasped": 0.0},
                "media": {
                    "policy_input_images": {
                        "egocentric_cam": str(policy_dir / "step_0000_egocentric_cam.png"),
                        "wrist_cam": str(policy_dir / "step_0000_wrist_cam.png"),
                    },
                    "robot_frame": str(robot_dir / "step_0000_top_down.png"),
                    "render_mode": "inline",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (loop_dir / "qwen_closed_loop_eval_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "env_id": "MuJoCoPickLift-v1",
                "env_config": {"object_shape": "cube", "object_color": "green", "n_distractors": 0},
                "seed": 98300,
                "start_contract": "grip_from_edge_cube",
                "success_rate": 0.0,
                "episodes_completed": 1,
                "camera_contract": {
                    "observation.images.camera1": "egocentric_cam",
                    "observation.images.camera2": "wrist_cam",
                },
                "plan": {"task": "Close the gripper on the green cube edge and lift.", "calls": []},
                "episodes": [
                    {
                        "episode": 0,
                        "final_success": False,
                        "total_reward": 0.1,
                        "steps": 1,
                        "reset_info": {"tcp_to_obj_dist": 0.16},
                        "final_info": {"tcp_to_obj_dist": 0.12, "is_grasped": 0.0, "lift_height": 0.0},
                        "trace_path": str(trace_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _handle_viewer_request(repo_root: Path, path: str) -> Any:
    handler_cls = dataset_viewer.make_handler(repo_root)
    request = f"GET {path} HTTP/1.1\r\nHost: example.test\r\nConnection: close\r\n\r\n".encode("utf-8")
    fake_socket = _FakeSocket(request)
    handler_cls(fake_socket, ("127.0.0.1", 12345), _FakeServer())
    raw_response = fake_socket.output.getvalue()
    header_bytes, body = raw_response.split(b"\r\n\r\n", 1)
    headers = header_bytes.decode("iso-8859-1")
    if "application/json" in headers:
        return json.loads(body.decode("utf-8"))
    return body.decode("utf-8")


class _FakeSocket:
    def __init__(self, request: bytes) -> None:
        self.input = io.BytesIO(request)
        self.output = io.BytesIO()

    def makefile(self, mode: str, buffering: int | None = None):
        del buffering
        if "r" in mode:
            return self.input
        return self.output

    def sendall(self, payload: bytes) -> None:
        self.output.write(payload)


class _FakeServer:
    server_name = "example.test"
    server_port = 80


if __name__ == "__main__":
    unittest.main()
