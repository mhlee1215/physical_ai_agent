from __future__ import annotations

import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from physical_ai_agent.so101_workspace_spawn_catalog import (
    WorkspaceCellQuotaScheduler,
    WorkspaceSpawnCatalog,
    WorkspaceSpawnCandidate,
    build_continuous_area_workspace_spawn_catalog,
    build_joint_feasible_workspace_spawn_catalog,
    build_workspace_spawn_catalog,
    load_workspace_spawn_catalog,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class SO101WorkspaceSpawnCatalogTests(unittest.TestCase):
    def test_builder_covers_cells_without_reusing_positions_and_decays_with_distance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "workspace.json"
            source.write_text(
                json.dumps(_workspace_source()),
                encoding="utf-8",
            )
            catalog = build_workspace_spawn_catalog(
                source_path=source,
                catalog_id="test",
                primary_target_count=12,
                backup_count=4,
                shard_count=2,
                distance_decay_rate_per_m=30.0,
                angular_jitter_max_deg=0.5,
            )

        primary = [row for row in catalog.candidates if row.stage == "primary"]
        self.assertEqual(len(primary), 12)
        self.assertEqual({row.source_cell_id for row in primary}, {"near", "far"})
        self.assertGreater(
            sum(row.source_cell_id == "near" for row in primary),
            sum(row.source_cell_id == "far" for row in primary),
        )
        self.assertEqual(
            len({tuple(row.world_xy_m) for row in catalog.candidates}),
            len(catalog.candidates),
        )
        self.assertEqual(sum(row.primary_target_count for row in catalog.shards), 12)

    def test_catalog_rejects_duplicate_world_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "workspace.json"
            source.write_text(json.dumps(_workspace_source()), encoding="utf-8")
            payload = build_workspace_spawn_catalog(
                source_path=source,
                catalog_id="test",
                primary_target_count=4,
                backup_count=2,
                shard_count=2,
                distance_decay_rate_per_m=20.0,
                angular_jitter_max_deg=0.5,
            ).model_dump(mode="json")
        payload["candidates"][1]["world_xy_m"] = payload["candidates"][0][
            "world_xy_m"
        ]
        with self.assertRaises(ValidationError):
            WorkspaceSpawnCatalog.model_validate(payload)

    def test_polar_stratified_sampling_spreads_radius_and_angle_deterministically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "workspace.json"
            source.write_text(json.dumps(_workspace_source()), encoding="utf-8")
            kwargs = {
                "source_path": source,
                "catalog_id": "polar",
                "primary_target_count": 20,
                "backup_count": 8,
                "shard_count": 2,
                "distance_decay_rate_per_m": 8.0,
                "angular_jitter_max_deg": 4.5,
                "radial_jitter_max_m": 0.0045,
            }
            first = build_workspace_spawn_catalog(**kwargs)
            second = build_workspace_spawn_catalog(**kwargs)

        self.assertEqual(first.sampling_strategy, "polar_stratified_v2")
        self.assertEqual(
            first.model_dump(mode="json"),
            second.model_dump(mode="json"),
        )
        self.assertGreater(
            len({round(row.radius_from_base_m, 6) for row in first.candidates}),
            2,
        )
        self.assertGreater(
            len({round(row.angle_from_base_deg, 6) for row in first.candidates}),
            2,
        )
        for source_cell_id in {"near", "far"}:
            primary = [
                row
                for row in first.candidates
                if row.stage == "primary"
                and row.source_cell_id == source_cell_id
            ]
            self.assertTrue(
                any(
                    row.radial_offset_m == 0.0
                    and row.angular_offset_deg == 0.0
                    for row in primary
                )
            )
        self.assertTrue(
            all(0.20 <= row.radius_from_base_m <= 0.25 for row in first.candidates)
        )
        self.assertTrue(
            all(0.0 <= row.angle_from_base_deg <= 20.0 for row in first.candidates)
        )

    def test_evidence_local_sampling_preserves_yaw_and_sequence_is_disjoint(
        self,
    ) -> None:
        source_payload = _workspace_source()
        source_payload["cells"][0]["object_yaw_deg"] = 12.0
        source_payload["cells"][1]["object_yaw_deg"] = 37.0
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "workspace.json"
            source.write_text(json.dumps(source_payload), encoding="utf-8")
            shared = {
                "source_path": source,
                "primary_target_count": 8,
                "backup_count": 4,
                "shard_count": 2,
                "distance_decay_rate_per_m": 1.0,
                "angular_jitter_max_deg": 1.0,
                "radial_jitter_max_m": 0.001,
                "preserve_evidence_object_yaw": True,
                "object_yaw_jitter_half_range_deg": 0.5,
            }
            first = build_workspace_spawn_catalog(
                catalog_id="first",
                candidate_sequence_offset=10_000,
                **shared,
            )
            second = build_workspace_spawn_catalog(
                catalog_id="second",
                candidate_sequence_offset=20_000,
                **shared,
            )

        self.assertEqual(first.sampling_strategy, "evidence_local_pose_jitter_v1")
        self.assertEqual(first.candidate_sequence_offset, 10_000)
        self.assertTrue(first.preserve_evidence_object_yaw)
        source_yaws = {"near": 12.0, "far": 37.0}
        for row in first.candidates:
            yaw_error = abs(
                ((row.object_yaw_deg - source_yaws[row.source_cell_id] + 180.0) % 360.0)
                - 180.0
            )
            self.assertLessEqual(yaw_error, 0.5 + 1e-9)
        self.assertFalse(
            {tuple(row.world_xy_m) for row in first.candidates}
            & {tuple(row.world_xy_m) for row in second.candidates}
        )

    def test_v4_1_catalog_is_a_broad_two_dimensional_append_only_source(
        self,
    ) -> None:
        catalog = load_workspace_spawn_catalog(
            REPO_ROOT
            / "configs/so101/spawn_catalogs/"
            "grip_the_cube_v4_1_workspace_candidates.json"
        )
        primary = [
            candidate
            for candidate in catalog.candidates
            if candidate.stage == "primary"
        ]
        radii = [candidate.radius_from_base_m for candidate in primary]
        angles = [candidate.angle_from_base_deg for candidate in primary]

        self.assertEqual(catalog.sampling_strategy, "polar_stratified_v2")
        self.assertEqual(len(primary), 500)
        self.assertEqual(len({row.source_cell_id for row in primary}), 44)
        self.assertGreaterEqual(max(radii) - min(radii), 0.029)
        self.assertGreaterEqual(max(angles) - min(angles), 98.0)
        self.assertEqual(
            len({tuple(row.world_xy_m) for row in catalog.candidates}),
            catalog.candidate_count,
        )
        source_radius_counts = Counter(
            round(row.radius_from_base_m - row.radial_offset_m, 2)
            for row in primary
        )
        ordered_counts = [
            count for _, count in sorted(source_radius_counts.items())
        ]
        self.assertTrue(
            all(
                near >= far
                for near, far in zip(
                    ordered_counts,
                    ordered_counts[1:],
                    strict=False,
                )
            )
        )

    def test_continuous_area_catalog_uses_open_cells_and_minimum_spacing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "workspace.json"
            source.write_text(
                json.dumps(_continuous_workspace_source()),
                encoding="utf-8",
            )
            catalog = build_continuous_area_workspace_spawn_catalog(
                source_path=source,
                catalog_id="continuous",
                primary_target_count=8,
                backup_count=8,
                shard_count=2,
                radius_min_m=0.20,
                radius_max_m=0.24,
                angle_min_deg=-20.0,
                angle_max_deg=20.0,
                radial_strata=2,
                angular_strata=2,
                far_to_near_area_density_ratio=0.8,
                minimum_spacing_m=0.002,
            )

        self.assertEqual(catalog.format, "so101_workspace_spawn_catalog_v2")
        self.assertTrue(catalog.enforce_cell_local_quota)
        self.assertEqual(len(catalog.cell_quotas), 4)
        self.assertTrue(
            all(
                0.20 < row.radius_from_base_m < 0.24
                and -20.0 < row.angle_from_base_deg < 20.0
                for row in catalog.candidates
            )
        )
        positions = [tuple(row.world_xy_m) for row in catalog.candidates]
        nearest = min(
            math.dist(left, right)
            for index, left in enumerate(positions)
            for right in positions[index + 1 :]
        )
        self.assertGreaterEqual(nearest, 0.002 - 1e-12)

    def test_continuous_catalog_stratifies_cube_yaw_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "workspace.json"
            source.write_text(
                json.dumps(_continuous_workspace_source()),
                encoding="utf-8",
            )
            kwargs = {
                "source_path": source,
                "catalog_id": "continuous-yaw",
                "primary_target_count": 16,
                "backup_count": 16,
                "shard_count": 2,
                "radius_min_m": 0.20,
                "radius_max_m": 0.24,
                "angle_min_deg": -20.0,
                "angle_max_deg": 20.0,
                "radial_strata": 2,
                "angular_strata": 2,
                "far_to_near_area_density_ratio": 0.8,
                "minimum_spacing_m": 0.001,
                "object_yaw_min_deg": 0.0,
                "object_yaw_max_deg": 90.0,
                "object_yaw_strata": 4,
                "object_yaw_periodicity_deg": 90.0,
            }
            first = build_continuous_area_workspace_spawn_catalog(**kwargs)
            second = build_continuous_area_workspace_spawn_catalog(**kwargs)

        self.assertEqual(
            first.sampling_strategy,
            "continuous_area_yaw_stratified_v4",
        )
        self.assertEqual(first.model_dump(mode="json"), second.model_dump(mode="json"))
        self.assertEqual(len(first.cell_quotas), 16)
        self.assertEqual(
            {row.yaw_index for row in first.cell_quotas},
            {0, 1, 2, 3},
        )
        primary = [row for row in first.candidates if row.stage == "primary"]
        relative_yaw_bins = [
            min(
                3,
                int(
                    ((row.object_yaw_deg - row.angle_from_base_deg) % 90.0)
                    // 22.5
                ),
            )
            for row in primary
        ]
        self.assertEqual(set(relative_yaw_bins), {0, 1, 2, 3})
        self.assertEqual(
            {index: relative_yaw_bins.count(index) for index in range(4)},
            {0: 4, 1: 4, 2: 4, 3: 4},
        )
        self.assertTrue(
            any(
                not math.isclose(
                    row.object_yaw_deg,
                    (row.angle_from_base_deg - 80.0) % 360.0,
                    abs_tol=1e-9,
                )
                for row in primary
            )
        )

    def test_cell_quota_scheduler_replaces_failure_only_inside_the_same_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "workspace.json"
            source.write_text(
                json.dumps(_continuous_workspace_source()),
                encoding="utf-8",
            )
            catalog = build_continuous_area_workspace_spawn_catalog(
                source_path=source,
                catalog_id="continuous",
                primary_target_count=8,
                backup_count=8,
                shard_count=2,
                radius_min_m=0.20,
                radius_max_m=0.24,
                angle_min_deg=-20.0,
                angle_max_deg=20.0,
                radial_strata=2,
                angular_strata=2,
                far_to_near_area_density_ratio=0.8,
                minimum_spacing_m=0.002,
            )
        shard = catalog.shards[0]
        rows = catalog.candidates[
            shard.start_index : shard.start_index + shard.candidate_count
        ]
        scheduler = WorkspaceCellQuotaScheduler(rows)
        rejected_cell = None
        while not scheduler.complete:
            candidate = scheduler.next_candidate()
            if rejected_cell is None:
                rejected_cell = candidate.source_cell_id
                continue
            scheduler.record_success(candidate)

        summary = scheduler.summary()
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["accepted_total"], shard.primary_target_count)
        self.assertEqual(summary["attempted_total"], shard.primary_target_count + 1)
        self.assertEqual(
            summary["cells"][rejected_cell]["accepted"],
            summary["cells"][rejected_cell]["target"],
        )

    def test_cell_quota_scheduler_skips_dense_backup_near_accepted_episode(self) -> None:
        def candidate(candidate_id: str, stage: str, x_m: float) -> WorkspaceSpawnCandidate:
            return WorkspaceSpawnCandidate(
                candidate_id=candidate_id,
                source_cell_id="cell",
                stage=stage,
                world_xy_m=[x_m, 0.0],
                base_xy_m=[x_m, 0.0],
                radius_from_base_m=max(x_m, 0.001),
                angle_from_base_deg=0.0,
                object_yaw_deg=1.0,
                sampling_weight=1.0,
            )

        scheduler = WorkspaceCellQuotaScheduler(
            [
                candidate("primary-0", "primary", 0.01),
                candidate("primary-1", "primary", 0.02),
                candidate("backup-0", "backup", 0.0101),
                candidate("backup-1", "backup", 0.03),
            ],
            accepted_minimum_spacing_m=0.005,
        )
        first = scheduler.next_candidate()
        scheduler.record_success(first)
        scheduler.next_candidate()  # Simulate a failed second primary.
        replacement = scheduler.next_candidate()
        self.assertEqual(replacement.candidate_id, "backup-1")
        scheduler.record_success(replacement)
        self.assertTrue(scheduler.complete)
        self.assertEqual(scheduler.summary()["spacing_skipped_total"], 1)

    def test_cell_quota_scheduler_respects_positions_accepted_by_other_shards(self) -> None:
        def candidate(candidate_id: str, stage: str, x_m: float) -> WorkspaceSpawnCandidate:
            return WorkspaceSpawnCandidate(
                candidate_id=candidate_id,
                source_cell_id="cell",
                stage=stage,
                world_xy_m=[x_m, 0.0],
                base_xy_m=[x_m, 0.0],
                radius_from_base_m=max(x_m, 0.001),
                angle_from_base_deg=0.0,
                object_yaw_deg=1.0,
                sampling_weight=1.0,
            )

        scheduler = WorkspaceCellQuotaScheduler(
            [
                candidate("primary-0", "primary", 0.01),
                candidate("backup-0", "backup", 0.03),
            ],
            accepted_minimum_spacing_m=0.005,
            forbidden_positions=[(0.0101, 0.0)],
        )

        replacement = scheduler.next_candidate()
        self.assertEqual(replacement.candidate_id, "backup-0")
        scheduler.record_success(replacement)
        summary = scheduler.summary()
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["spacing_skipped_total"], 1)
        self.assertEqual(summary["forbidden_position_count"], 1)

    def test_joint_feasible_catalog_preserves_sparse_support_and_axis_margins(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "workspace.json"
            source.write_text(
                json.dumps(_joint_feasible_workspace_source()),
                encoding="utf-8",
            )
            kwargs = {
                "source_path": source,
                "catalog_id": "joint-feasible",
                "radial_primary_counts": [8, 4],
                "yaw_primary_counts": [8, 2, 2],
                "radial_backup_counts": [8, 4],
                "yaw_backup_counts": [8, 2, 2],
                "shard_count": 2,
                "radius_min_m": 0.20,
                "radius_max_m": 0.24,
                "angle_min_deg": -20.0,
                "angle_max_deg": 20.0,
                "minimum_spacing_m": 0.0005,
                "evidence_radius_half_range_m": 0.003,
                "evidence_angle_half_range_deg": 1.5,
                "object_yaw_center_offset_deg": 2.0,
                "object_yaw_jitter_half_range_deg": 1.0,
                "max_robot_relative_yaw_count_cv": 1.0,
            }
            first = build_joint_feasible_workspace_spawn_catalog(**kwargs)
            second = build_joint_feasible_workspace_spawn_catalog(**kwargs)
            offset = build_joint_feasible_workspace_spawn_catalog(
                **kwargs,
                candidate_sequence_offset=20_000_000,
            )

        self.assertEqual(
            first.sampling_strategy,
            "continuous_joint_feasible_yaw_stratified_v5",
        )
        self.assertEqual(first.model_dump(mode="json"), second.model_dump(mode="json"))
        self.assertEqual(first.candidate_sequence_offset, 0)
        self.assertEqual(offset.candidate_sequence_offset, 20_000_000)
        self.assertNotEqual(
            [row.world_xy_m for row in first.candidates],
            [row.world_xy_m for row in offset.candidates],
        )
        primary = [row for row in first.candidates if row.stage == "primary"]
        radial_counts = Counter(
            min(range(2), key=lambda index: abs(row.radius_from_base_m - [0.20, 0.24][index]))
            for row in primary
        )
        yaw_counts = Counter(
            min(
                2,
                int(((row.object_yaw_deg - row.angle_from_base_deg) % 90.0) // 30.0),
            )
            for row in primary
        )
        angular_counts = Counter(
            0 if row.angle_from_base_deg < 0.0 else 1 for row in primary
        )
        self.assertEqual(radial_counts, {0: 8, 1: 4})
        self.assertEqual(yaw_counts, {0: 8, 1: 2, 2: 2})
        self.assertEqual(angular_counts, {0: 6, 1: 6})
        self.assertFalse(
            any(
                quota.radial_index == 0 and quota.yaw_index != 0
                for quota in first.cell_quotas
            )
        )
        self.assertTrue(
            all(
                0.0
                < (row.object_yaw_deg - row.angle_from_base_deg) % 90.0
                < 90.0
                for row in first.candidates
            )
        )
        self.assertEqual(
            first.continuous_distribution.evidence_radius_half_range_m,
            0.003,
        )
        self.assertEqual(
            first.continuous_distribution.evidence_angle_half_range_deg,
            1.5,
        )
        self.assertEqual(
            first.continuous_distribution.excluded_radius_relative_yaw_pairs,
            [],
        )
        self.assertTrue(
            all(
                min(abs(row.radius_from_base_m - center) for center in (0.20, 0.24))
                <= 0.003 + 1e-12
                for row in first.candidates
            )
        )
        self.assertTrue(
            all(
                min(abs(row.angle_from_base_deg - center) for center in (-20.0, 20.0))
                <= 1.5 + 1e-12
                for row in first.candidates
            )
        )


def _workspace_source() -> dict:
    return {
        "format": "so101_grasp_workspace_catalog_v1",
        "camera_rig_config": "configs/so101/camera.json",
        "home_qpos": [0, 0, 0, 0, 0, 0],
        "object_color": "green",
        "object_half_size_m": 0.015,
        "base_contract": {
            "world_xyz_m": [0.1, 0.2, 0.0],
        },
        "cells": [
            {
                "point_id": "near",
                "radius_from_base_m": 0.20,
                "angle_from_base_deg": 0.0,
                "uniform_area_weight": 0.5,
                "camera1_grid_bin": 5,
            },
            {
                "point_id": "far",
                "radius_from_base_m": 0.25,
                "angle_from_base_deg": 20.0,
                "uniform_area_weight": 0.5,
                "camera1_grid_bin": 6,
            },
        ],
    }


def _continuous_workspace_source() -> dict:
    payload = _workspace_source()
    payload["cells"] = [
        {
            "point_id": f"r{radius:.2f}_a{angle:+.0f}",
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
    ]
    return payload


def _joint_feasible_workspace_source() -> dict:
    payload = _workspace_source()
    rows = []
    for angle_index, angle in enumerate((-20.0, 20.0)):
        for radial_index, radius in enumerate((0.20, 0.24)):
            yaw_offsets = (0.0,) if radial_index == 0 else (0.0, 30.0, 60.0)
            for yaw_index, yaw_offset in enumerate(yaw_offsets):
                rows.append(
                    {
                        "point_id": f"r{radial_index}_a{angle_index}_y{yaw_index}",
                        "radius_from_base_m": radius,
                        "angle_from_base_deg": angle,
                        "object_yaw_deg": angle + yaw_offset,
                        "uniform_area_weight": 1.0,
                        "camera1_grid_bin": angle_index,
                    }
                )
    payload["cells"] = rows
    return payload


if __name__ == "__main__":
    unittest.main()
