from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from build_so101_dataset_distribution_report import (
    _episode_has_visible_policy_camera,
    _object_yaw_occupancy,
    _periodic_degrees,
    _radial_counts,
    build_distribution_report,
    require_distribution_report,
)


V4_RECIPE = Path("configs/so101/dataset_generation/grip_the_cube_v4.json")
V4_CATALOG = Path(
    "configs/so101/spawn_catalogs/grip_the_cube_v4_workspace_candidates.json"
)


class SO101DatasetDistributionReportTests(unittest.TestCase):
    def test_initial_visibility_uses_either_policy_camera(self) -> None:
        self.assertTrue(
            _episode_has_visible_policy_camera(
                {
                    "start_policy_camera_visibility": {
                        "camera1": {"visible": False},
                        "camera2": {"visible": True},
                    }
                }
            )
        )
        self.assertFalse(
            _episode_has_visible_policy_camera(
                {
                    "start_policy_camera_visibility": {
                        "camera1": {"visible": False},
                        "camera2": {"visible": False},
                    }
                }
            )
        )

    def test_object_yaw_occupancy_tracks_span_coverage_and_balance(self) -> None:
        occupancy = _object_yaw_occupancy(
            [5.0, 25.0, 50.0, 75.0],
            bins=4,
            bounds=(0.0, 90.0),
        )
        self.assertEqual(occupancy["occupied_bins"], 4)
        self.assertEqual(occupancy["coverage_ratio"], 1.0)
        self.assertEqual(occupancy["count_cv"], 0.0)
        self.assertEqual(occupancy["span_deg"], 70.0)

    def test_robot_relative_yaw_detects_spawn_coupled_cube_faces(self) -> None:
        angles = [-20.0, 10.0, 40.0, 70.0]
        coupled = [angle - 80.0 for angle in angles]
        relative = [
            _periodic_degrees(yaw - angle, period=90.0)
            for yaw, angle in zip(coupled, angles, strict=True)
        ]
        occupancy = _object_yaw_occupancy(
            relative,
            bins=4,
            bounds=(0.0, 90.0),
        )
        self.assertEqual(relative, [10.0, 10.0, 10.0, 10.0])
        self.assertEqual(occupancy["occupied_bins"], 1)
        self.assertEqual(occupancy["coverage_ratio"], 0.25)

    def test_radial_histogram_can_group_continuous_jitter_into_one_cm_bands(
        self,
    ) -> None:
        counts = _radial_counts(
            [0.2502, 0.2544, 0.2560, 0.2644, 0.2660, 0.2744],
            bin_width_m=0.01,
        )
        self.assertEqual(
            counts,
            {"0.2500": 2, "0.2600": 2, "0.2700": 2},
        )

    def test_report_writes_json_markdown_html_and_passes_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = json.loads(V4_CATALOG.read_text(encoding="utf-8"))
            candidates = catalog["candidates"][:4]
            recipe = json.loads(V4_RECIPE.read_text(encoding="utf-8"))
            recipe["splits"]["train"]["bins"] = [
                {
                    "id": 0,
                    "episodes": 4,
                    "seed": 1,
                    "lookup_start_index": 0,
                    "workspace_candidate_count": 10,
                    "shard": "test",
                }
            ]
            recipe["distribution_report"]["min_workspace_cell_coverage_ratio"] = 0.0
            recipe["distribution_report"]["max_radial_total_variation"] = 1.0
            recipe["distribution_report"][
                "max_all_policy_cameras_invisible_fraction"
            ] = 0.0
            recipe["distribution_report"][
                "require_distance_decay_nonincreasing"
            ] = False
            recipe_path = root / "recipe.json"
            recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
            (root / "so101_lerobot_export_report.json").write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "seed": 100 + index,
                                "success": True,
                                "frames": 100 + index,
                                "task": "grip the green cube and lift",
                                "workspace_spawn": candidate,
                                "camera1_grid_bin": (
                                    None
                                    if index == 0
                                    else candidate["camera1_grid_bin"]
                                ),
                                "start_policy_camera_visibility": {
                                    "camera1": {"visible": index % 2 == 0},
                                    "camera2": {"visible": index % 2 == 1},
                                },
                                "final_info": {"lift_height": 0.065},
                                "gripper_floor_clearance_gate": {
                                    "minimum_clearance_m": 0.011
                                },
                            }
                            for index, candidate in enumerate(candidates)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            sidecar_dir = root / "meta" / "camera_grid_bins"
            sidecar_dir.mkdir(parents=True)
            (sidecar_dir / "observation_images_camera1_4x4_frame0.json").write_text(
                json.dumps(
                    {
                        "grid_size": 4,
                        "invisible_episodes": 0,
                        "bin_counts_yx": [
                            [0, 0, 0, 0],
                            [0, 1, 1, 0],
                            [0, 1, 1, 0],
                            [0, 0, 0, 0],
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = build_distribution_report(
                dataset_root=root,
                recipe_path=recipe_path,
                split_name="train",
            )
            required = require_distribution_report(root)

            self.assertEqual(report["gate"]["status"], "passed")
            self.assertEqual(
                report["gate"]["checks"]["workspace_cell_coverage"]["declared"],
                len(report["workspace"]["expected_source_cell_counts"]),
            )
            self.assertEqual(
                report["summary"]["all_policy_cameras_invisible_episodes"],
                0,
            )
            self.assertEqual(required["summary"]["episodes"], 4)
            self.assertIn("SO101 Dataset Distribution", (
                root / "meta" / "distribution" / "distribution.md"
            ).read_text(encoding="utf-8"))
            self.assertIn("<svg", (
                root / "meta" / "distribution" / "distribution.html"
            ).read_text(encoding="utf-8"))
            self.assertIn(
                report["artifacts"]["markdown_sha256"],
                (root / "meta" / "distribution" / "distribution.html").read_text(
                    encoding="utf-8"
                ),
            )

            (root / "meta" / "distribution" / "distribution.md").write_text(
                "# stale report\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                require_distribution_report(root)

    def test_report_gate_rejects_duplicate_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = json.loads(V4_RECIPE.read_text(encoding="utf-8"))
            recipe["splits"]["train"]["bins"] = [
                {
                    "id": 0,
                    "episodes": 2,
                    "seed": 1,
                    "lookup_start_index": 0,
                    "workspace_candidate_count": 10,
                    "shard": "test",
                }
            ]
            recipe["distribution_report"]["min_workspace_cell_coverage_ratio"] = 0.0
            recipe["distribution_report"]["max_radial_total_variation"] = 1.0
            recipe["distribution_report"][
                "require_distance_decay_nonincreasing"
            ] = False
            recipe_path = root / "recipe.json"
            recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
            candidates = json.loads(V4_CATALOG.read_text(encoding="utf-8"))[
                "candidates"
            ][:2]
            rows = []
            for candidate in candidates:
                rows.append(
                    {
                        "seed": 7,
                        "success": True,
                        "frames": 10,
                        "task": "grip the green cube and lift",
                        "workspace_spawn": copy.deepcopy(candidate),
                    }
                )
            (root / "so101_lerobot_export_report.json").write_text(
                json.dumps({"episodes": rows}),
                encoding="utf-8",
            )

            report = build_distribution_report(
                dataset_root=root,
                recipe_path=recipe_path,
                split_name="train",
            )

            self.assertEqual(report["gate"]["status"], "failed")
            self.assertIn("unique_episode_seeds", report["gate"]["failed"])
            with self.assertRaises(ValueError):
                require_distribution_report(root)

    def test_report_enforces_two_dimensional_workspace_spread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = json.loads(V4_RECIPE.read_text(encoding="utf-8"))
            recipe["splits"]["train"]["bins"] = [
                {
                    "id": 0,
                    "episodes": 4,
                    "seed": 1,
                    "lookup_start_index": 0,
                    "workspace_candidate_count": 10,
                    "shard": "test",
                }
            ]
            recipe["distribution_report"].update(
                {
                    "min_workspace_cell_coverage_ratio": 0.0,
                    "max_radial_total_variation": 1.0,
                    "require_distance_decay_nonincreasing": False,
                    "min_radius_span_m": 0.09,
                    "min_angle_span_deg": 80.0,
                    "polar_radial_bins": 2,
                    "polar_angular_bins": 2,
                    "min_polar_cell_coverage_ratio": 1.0,
                    "max_polar_cell_count_cv": 0.0,
                    "min_nearest_neighbor_median_m": 0.05,
                    "object_yaw_histogram_bins": 4,
                    "min_object_yaw_span_deg": 60.0,
                    "min_object_yaw_bin_coverage_ratio": 1.0,
                    "max_object_yaw_bin_count_cv": 0.0,
                }
            )
            recipe_path = root / "recipe.json"
            recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
            candidates = copy.deepcopy(
                json.loads(V4_CATALOG.read_text(encoding="utf-8"))["candidates"][:4]
            )
            placements = [
                (0.20, 0.0, [0.20, 0.0]),
                (0.20, 90.0, [0.0, 0.20]),
                (0.30, 0.0, [0.30, 0.0]),
                (0.30, 90.0, [0.0, 0.30]),
            ]
            object_yaws = [5.0, 25.0, 50.0, 75.0]
            for index, (candidate, placement) in enumerate(
                zip(candidates, placements, strict=True)
            ):
                radius, angle, xy = placement
                candidate["radius_from_base_m"] = radius
                candidate["angle_from_base_deg"] = angle
                candidate["world_xy_m"] = xy
                candidate["source_cell_id"] = f"spread-{index}"
                candidate["object_yaw_deg"] = object_yaws[index]
            (root / "so101_lerobot_export_report.json").write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "seed": 100 + index,
                                "success": True,
                                "frames": 100,
                                "task": "grip the green cube and lift",
                                "workspace_spawn": candidate,
                            }
                            for index, candidate in enumerate(candidates)
                        ]
                    }
                ),
                encoding="utf-8",
            )

            passed = build_distribution_report(
                dataset_root=root,
                recipe_path=recipe_path,
                split_name="train",
            )
            self.assertEqual(passed["gate"]["status"], "passed")
            self.assertEqual(
                passed["workspace"]["polar_occupancy"]["occupied_cells"], 4
            )
            self.assertEqual(
                passed["workspace"]["object_yaw"]["occupied_bins"], 4
            )

            recipe["distribution_report"]["min_radius_span_m"] = 0.2
            recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
            failed = build_distribution_report(
                dataset_root=root,
                recipe_path=recipe_path,
                split_name="train",
            )
            self.assertIn("workspace_radius_span", failed["gate"]["failed"])


if __name__ == "__main__":
    unittest.main()
