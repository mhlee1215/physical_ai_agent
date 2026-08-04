from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from build_so101_workspace_catalog_distribution_report import (
    _periodic_yaw_counts,
    build_workspace_catalog_distribution_report,
)
from physical_ai_agent.so101_workspace_spawn_catalog import (
    build_continuous_area_workspace_spawn_catalog,
)


V4_2_CATALOG = Path(
    "configs/so101/spawn_catalogs/grip_the_cube_v4_2_workspace_candidates.json"
)


class SO101WorkspaceCatalogDistributionReportTests(unittest.TestCase):
    def test_relative_yaw_histogram_rejects_one_face_coupling(self) -> None:
        angles = [-20.0, 10.0, 40.0, 70.0]
        yaws = [angle - 80.0 for angle in angles]
        counts = _periodic_yaw_counts(
            [yaw - angle for yaw, angle in zip(yaws, angles, strict=True)],
            bins=4,
            period_deg=90.0,
        )
        self.assertEqual(sum(value > 0 for value in counts.values()), 1)
        self.assertEqual(sum(counts.values()), 4)

    def test_v4_2_catalog_rejects_robot_relative_same_face_coupling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_workspace_catalog_distribution_report(
                catalog_path=V4_2_CATALOG,
                output_dir=Path(tmp),
            )

            self.assertEqual(report["gate"]["status"], "failed")
            self.assertIn(
                "robot_relative_object_yaw_coverage",
                report["gate"]["failed"],
            )
            self.assertEqual(report["summary"]["primary_candidates"], 500)
            self.assertEqual(report["summary"]["cells"], 60)
            self.assertEqual(report["summary"]["boundary_hits"], 0)
            self.assertGreaterEqual(
                report["summary"]["nearest_neighbor_min_m"],
                0.0015,
            )
            self.assertIn("<svg", (Path(tmp) / "distribution.html").read_text())
            self.assertEqual(
                json.loads((Path(tmp) / "distribution.json").read_text())[
                    "artifacts"
                ]["markdown_sha256"],
                report["artifacts"]["markdown_sha256"],
            )

    def test_independent_cube_yaw_is_balanced_before_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "format": "so101_grasp_workspace_catalog_v1",
                        "camera_rig_config": "configs/so101/camera.json",
                        "home_qpos": [0, 0, 0, 0, 0, 0],
                        "object_color": "green",
                        "object_half_size_m": 0.015,
                        "base_contract": {"world_xyz_m": [0.1, 0.2, 0.0]},
                        "cells": [
                            {
                                "point_id": f"cell-{index}",
                                "radius_from_base_m": radius,
                                "angle_from_base_deg": angle,
                                "uniform_area_weight": 0.25,
                                "camera1_grid_bin": index,
                            }
                            for index, (radius, angle) in enumerate(
                                [
                                    (0.20, -20.0),
                                    (0.20, 20.0),
                                    (0.24, -20.0),
                                    (0.24, 20.0),
                                ]
                            )
                        ],
                    }
                ),
                encoding="utf-8",
            )
            catalog = build_continuous_area_workspace_spawn_catalog(
                source_path=source,
                catalog_id="yaw-audit",
                primary_target_count=160,
                backup_count=32,
                shard_count=2,
                radius_min_m=0.20,
                radius_max_m=0.24,
                angle_min_deg=-20.0,
                angle_max_deg=20.0,
                radial_strata=2,
                angular_strata=2,
                far_to_near_area_density_ratio=0.9,
                minimum_spacing_m=0.001,
                object_yaw_min_deg=0.0,
                object_yaw_max_deg=90.0,
                object_yaw_strata=4,
            )
            catalog_path = root / "catalog.json"
            catalog_path.write_text(
                json.dumps(catalog.model_dump(mode="json")),
                encoding="utf-8",
            )
            report = build_workspace_catalog_distribution_report(
                catalog_path=catalog_path,
                output_dir=root / "report",
            )

        self.assertEqual(
            report["gate"]["status"],
            "passed",
            msg=report["gate"],
        )
        self.assertEqual(report["summary"]["object_yaw_strata"], 4)
        self.assertLessEqual(report["summary"]["object_yaw_count_cv"], 0.10)
        self.assertEqual(
            report["summary"]["robot_relative_object_yaw_count_cv"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
