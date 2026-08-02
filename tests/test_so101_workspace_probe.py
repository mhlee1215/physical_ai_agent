from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from physical_ai_agent.so101_workspace_probe import (
    WorkspaceProbeConfig,
    annotate_physical_outcomes,
    base_relative_to_world_xy,
    camera_grid_bin,
    grid_points,
    physical_outcome_metrics,
    successful_workspace_cells,
    summarize_workspace_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class SO101WorkspaceProbeTest(unittest.TestCase):
    def test_all_checked_in_probe_configs_are_valid_and_seed_disjoint(self) -> None:
        config_paths = sorted(
            (REPO_ROOT / "configs/so101/workspace_probes").glob("*.json")
        )
        configs = [
            WorkspaceProbeConfig.model_validate_json(path.read_bytes())
            for path in config_paths
        ]
        seed_ranges = sorted(
            (
                config.seed_base,
                config.seed_base + len(grid_points(config)) - 1,
                config.name,
            )
            for config in configs
        )
        for previous, current in zip(seed_ranges, seed_ranges[1:], strict=False):
            self.assertLess(
                previous[1],
                current[0],
                msg=f"probe seed ranges overlap: {previous} and {current}",
            )

    def test_checked_in_config_is_valid_and_has_broad_grid(self) -> None:
        path = (
            REPO_ROOT
            / "configs/so101/workspace_probes/grip_the_cube_v3_hardware_workspace_v1.json"
        )
        config = WorkspaceProbeConfig.model_validate_json(path.read_bytes())
        points = grid_points(config)
        self.assertEqual(len(points), 20 * 21)
        self.assertGreaterEqual(config.grid.x.max_m - config.grid.x.min_m, 0.7)
        self.assertGreaterEqual(config.grid.y.max_m - config.grid.y.min_m, 0.8)
        self.assertEqual(len({row["point_id"] for row in points}), len(points))

    def test_base_relative_transform_uses_base_rotation(self) -> None:
        world = base_relative_to_world_xy(
            (1.0, 2.0),
            np.asarray([[0.0, -1.0], [1.0, 0.0]]),
            (0.3, 0.1),
        )
        self.assertTrue(np.allclose(world, (0.9, 2.3)))

    def test_polar_grid_samples_base_radius_and_angle(self) -> None:
        config = WorkspaceProbeConfig.model_validate(
            {
                "schema_version": 1,
                "name": "polar",
                "description": "polar test",
                "camera_rig_config": "camera.json",
                "home_qpos": [0, 0, 0, 0, 0, 0],
                "object_yaw_degrees": [0],
                "seed_base": 1,
                "polar_grid": {
                    "radius": {"min_m": 0.2, "max_m": 0.3, "step_m": 0.1},
                    "angle": {
                        "min_deg": 0,
                        "max_deg": 90,
                        "step_deg": 90,
                    },
                },
            }
        )
        points = grid_points(config)
        self.assertEqual(len(points), 4)
        self.assertTrue(
            any(
                np.allclose((row["base_x_m"], row["base_y_m"]), (0.0, 0.3))
                for row in points
            )
        )

    def test_polar_grid_can_align_cube_face_normal_to_radial_angle(self) -> None:
        config = WorkspaceProbeConfig.model_validate(
            {
                "schema_version": 1,
                "name": "radial yaw",
                "description": "radial yaw test",
                "camera_rig_config": "camera.json",
                "home_qpos": [0, 0, 0, 0, 0, 0],
                "object_yaw_mode": "radial_face_normal",
                "radial_yaw_offsets_degrees": [-90],
                "seed_base": 1,
                "polar_grid": {
                    "radius": {"min_m": 0.2, "max_m": 0.3, "step_m": 0.1},
                    "angle": {
                        "min_deg": 20,
                        "max_deg": 30,
                        "step_deg": 10,
                    },
                },
            }
        )
        points = grid_points(config)
        point = next(row for row in points if row["angle_index"] == 1)
        self.assertEqual(point["yaw_deg"], -60.0)

    def test_summary_separates_physical_and_dataset_contract(self) -> None:
        records = [
            {
                "base_x_m": 0.1,
                "base_y_m": 0.0,
                "preflight_passed": True,
                "physical_success": True,
                "dataset_contract_success": False,
                "dataset_contract_failure_reason": "camera1_not_visible",
            },
            {
                "base_x_m": 0.2,
                "base_y_m": 0.1,
                "preflight_passed": True,
                "physical_success": True,
                "dataset_contract_success": True,
            },
            {
                "base_x_m": 0.4,
                "base_y_m": 0.2,
                "preflight_passed": False,
                "physical_success": False,
                "physical_failure_reason": "geometry_preflight_failed",
                "dataset_contract_success": False,
            },
        ]
        summary = summarize_workspace_records(records)
        self.assertEqual(summary["physical_successes"], 2)
        self.assertEqual(summary["dataset_contract_successes"], 1)
        self.assertEqual(
            summary["physical_failure_reasons"]["geometry_preflight_failed"], 1
        )

    def test_probe_mode_cannot_materialize_dataset_frames(self) -> None:
        script_path = REPO_ROOT / "scripts/export_so101_teacher_rollouts_lerobot.py"
        source = script_path.read_text(encoding="utf-8")
        self.assertIn("record_dataset_frames: bool = True", source)
        self.assertIn("if record_dataset_frames:", source)

        probe_source = (
            REPO_ROOT / "scripts/probe_so101_grasp_workspace.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "lift_success_height=(\n"
            "                    config.teacher.operational_lift_height_m\n"
            "                )",
            probe_source,
        )

        exporter_source = script_path.read_text(encoding="utf-8")
        self.assertIn(
            "_restore_sim_state(env, initial_snapshot)",
            exporter_source,
        )
        self.assertIn("contact_moving_target_error_m", exporter_source)

    def test_physical_outcome_ignores_post_lift_alignment_rejection(self) -> None:
        result = {
            "success": False,
            "reason": "teacher_replay_failed",
            "candidate_failures": [
                {
                    "reason": "wrist_roll_delta_gate_failed",
                    "final_info": {
                        "is_grasped": True,
                        "lift_height": 0.0654,
                    },
                }
            ],
        }
        metrics = physical_outcome_metrics(
            result,
            target_lift_height_m=0.065,
            operational_lift_height_m=0.06,
        )
        self.assertFalse(metrics["teacher_geometry_contract_success"])
        self.assertTrue(metrics["grasp_success"])
        self.assertTrue(metrics["target_lift_success"])

        record = annotate_physical_outcomes(
            {
                "physical_result": result,
                "physical_success": False,
                "physical_failure_reason": "teacher_replay_failed",
            },
            target_lift_height_m=0.065,
            operational_lift_height_m=0.06,
        )
        self.assertTrue(record["physical_success"])
        self.assertIsNone(record["physical_failure_reason"])

    def test_successful_cells_preserve_yaw_and_area_weights(self) -> None:
        records = [
            {
                "point_id": "outer",
                "base_x_m": 0.2,
                "base_y_m": 0.1,
                "world_x_m": 1.2,
                "world_y_m": 2.1,
                "radius_from_base_m": 0.26,
                "angle_from_base_deg": 20.0,
                "yaw_deg": -60.0,
                "initial_camera1_centroid": [0.7, 0.4],
                "point_cell_area_m2": 0.002,
                "dataset_contract_success": True,
            },
            {
                "point_id": "inner",
                "base_x_m": 0.2,
                "base_y_m": 0.0,
                "world_x_m": 1.2,
                "world_y_m": 2.0,
                "radius_from_base_m": 0.25,
                "angle_from_base_deg": 0.0,
                "yaw_deg": -80.0,
                "initial_camera1_centroid": [0.1, 0.2],
                "point_cell_area_m2": 0.001,
                "dataset_contract_success": True,
            },
            {
                "point_id": "failed",
                "dataset_contract_success": False,
            },
        ]
        cells = successful_workspace_cells(records)
        self.assertEqual([cell["point_id"] for cell in cells], ["inner", "outer"])
        self.assertEqual(cells[0]["object_yaw_deg"], -80.0)
        self.assertEqual(cells[0]["camera1_grid_bin"], 0)
        self.assertEqual(cells[1]["camera1_grid_bin"], 6)
        self.assertAlmostEqual(
            sum(cell["uniform_area_weight"] for cell in cells), 1.0
        )
        self.assertAlmostEqual(cells[1]["uniform_area_weight"], 2.0 / 3.0)

    def test_camera_grid_bin_clamps_border_centroids(self) -> None:
        self.assertEqual(camera_grid_bin([0.0, 0.0]), 0)
        self.assertEqual(camera_grid_bin([1.0, 1.0]), 15)


if __name__ == "__main__":
    unittest.main()
