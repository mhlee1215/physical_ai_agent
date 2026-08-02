from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class So101NearRangeTeacherTests(unittest.TestCase):
    def test_near_catalog_builder_keeps_only_initially_visible_candidates(self) -> None:
        from build_so101_near_range_spawn_catalog import (
            DEFAULT_BASE_ORIGIN_XY,
            DEFAULT_HOME_QPOS,
            build_near_range_spawn_catalog,
        )
        from physical_ai_agent.so101_workspace_spawn_catalog import (
            load_workspace_spawn_catalog,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            evidence_path = root / "evidence.json"
            report = build_near_range_spawn_catalog(
                catalog_output=catalog_path,
                evidence_output=evidence_path,
                catalog_id="near-visible-test",
                primary_count=4,
                backup_count=0,
                shard_count=2,
                radius_min_m=0.10,
                radius_max_m=0.18,
                radial_strata=2,
                angle_min_deg=-75.0,
                angle_max_deg=75.0,
                angle_strata=4,
                yaw_strata=4,
                sequence_offset=90_000,
                max_attempt_multiplier=50,
                minimum_spacing_m=0.001,
                minimum_area_pixels=20,
                min_floor_clearance_m=0.01,
                require_solver_prefilter=False,
                object_half_size_m=0.015,
                object_color="green",
                camera_rig_config=(
                    ROOT
                    / "configs"
                    / "so101"
                    / "camera_rigs"
                    / "official_32x32_uvc_photoreal_v10_fov_calibrated_direct_square.json"
                ),
                home_qpos=DEFAULT_HOME_QPOS,
                base_origin_xy=DEFAULT_BASE_ORIGIN_XY,
            )

            catalog = load_workspace_spawn_catalog(catalog_path)
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(report["accepted"], 4)
            self.assertEqual(catalog.candidate_count, 4)
            self.assertFalse(evidence["contract"]["solver_prefilter_enabled"])
            self.assertTrue(
                all(
                    placement["solver_candidate_count"] is None
                    for placement in evidence["placements"]
                )
            )
            self.assertTrue(
                all(
                    any(
                        camera["visible"]
                        for camera in placement["initial_visibility"].values()
                    )
                    for placement in evidence["placements"]
                )
            )
            self.assertTrue(
                all(
                    0.10 <= candidate.radius_from_base_m <= 0.18
                    for candidate in catalog.candidates
                )
            )

    def test_initial_visibility_gate_is_typed_and_forwarded_to_exporter(self) -> None:
        from generate_so101_dataset_recipe import _inspection_gate_args
        from physical_ai_agent.so101_dataset_generation_schema import (
            DatasetGenerationRecipe,
        )

        recipe_path = (
            ROOT
            / "configs"
            / "so101"
            / "dataset_generation"
            / "grip_the_cube_near_v1_canary.json"
        )
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
        payload["common"]["inspection_gates"].append(
            {
                "kind": "initial_target_visibility",
                "camera_keys": [
                    "observation.images.camera1",
                    "observation.images.camera2",
                ],
                "mode": "any",
                "min_area_pixels": 20,
            }
        )
        recipe = DatasetGenerationRecipe.model_validate(payload)
        args = _inspection_gate_args(
            [gate.model_dump(mode="json") for gate in recipe.common.inspection_gates]
        )

        self.assertIn("--require-initial-target-visible", args)
        self.assertEqual(
            args[args.index("--initial-target-visibility-cameras") + 1],
            "camera1,camera2",
        )
        self.assertEqual(
            args[args.index("--initial-target-min-area-pixels") + 1],
            "20",
        )

    def test_canary_recipe_uses_a_unique_near_range_spawn_catalog(self) -> None:
        from physical_ai_agent.so101_dataset_generation_schema import (
            load_dataset_generation_recipe,
        )
        from physical_ai_agent.so101_workspace_spawn_catalog import (
            load_workspace_spawn_catalog,
        )

        recipe_path = (
            ROOT
            / "configs"
            / "so101"
            / "dataset_generation"
            / "grip_the_cube_near_v1_canary.json"
        )
        recipe = load_dataset_generation_recipe(recipe_path)
        self.assertEqual(recipe.schema_version, 2)
        self.assertEqual(recipe.common.skill_mode, "grip_the_cube_near_v1")
        self.assertEqual(recipe.common.width, 256)
        self.assertEqual(recipe.common.height, 256)
        self.assertEqual(recipe.common.terminal_hold_steps, 12)
        self.assertEqual(recipe.source.mode, "from_spawn_catalog")
        self.assertEqual(len(recipe.source.catalogs), 1)

        catalog_path = ROOT / recipe.source.catalogs[0]
        catalog = load_workspace_spawn_catalog(catalog_path)
        self.assertEqual(catalog.candidate_count, 5)
        self.assertEqual(
            [round(candidate.radius_from_base_m, 2) for candidate in catalog.candidates],
            [0.10, 0.12, 0.14, 0.16, 0.18],
        )
        positions = {
            tuple(round(value, 12) for value in candidate.world_xy_m)
            for candidate in catalog.candidates
        }
        self.assertEqual(len(positions), catalog.candidate_count)

        source_path = ROOT / catalog.source_workspace_catalog
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        self.assertEqual(source_sha256, catalog.source_workspace_catalog_sha256)

    def test_train200_recipe_requires_visible_near_range_successes(self) -> None:
        from physical_ai_agent.so101_dataset_generation_schema import (
            load_dataset_generation_recipe,
        )
        from physical_ai_agent.so101_workspace_spawn_catalog import (
            load_workspace_spawn_catalog,
        )

        recipe_path = (
            ROOT
            / "configs"
            / "so101"
            / "dataset_generation"
            / "grip_the_cube_near_v1_train200.json"
        )
        recipe = load_dataset_generation_recipe(recipe_path)
        split = recipe.splits["train"]
        gates = {
            gate.kind: gate.model_dump(mode="json")
            for gate in recipe.common.inspection_gates
        }

        self.assertEqual(sum(row.episodes for row in split.bins), 200)
        self.assertEqual(recipe.common.spawn_min_radius, 0.10)
        self.assertEqual(recipe.common.spawn_max_radius, 0.18)
        self.assertEqual(recipe.common.width, 256)
        self.assertEqual(recipe.common.height, 256)
        self.assertIn("initial_target_visibility", gates)
        self.assertEqual(gates["initial_target_visibility"]["mode"], "any")
        self.assertEqual(
            gates["initial_target_visibility"]["camera_keys"],
            ["observation.images.camera1", "observation.images.camera2"],
        )
        self.assertEqual(
            recipe.distribution_report.max_all_policy_cameras_invisible_fraction,
            0.0,
        )

        catalog = load_workspace_spawn_catalog(ROOT / recipe.source.catalogs[0])
        self.assertEqual(catalog.primary_target_count, 200)
        self.assertGreaterEqual(catalog.candidate_count, 200)
        self.assertTrue(
            all(
                0.10 <= candidate.radius_from_base_m <= 0.18
                for candidate in catalog.candidates
            )
        )

    def test_near_skill_keeps_the_full_grip_observation_contract(self) -> None:
        from export_so101_teacher_rollouts_lerobot import (
            COLOR_SHAPE_SKILL_TASK_TEMPLATES,
            FIXED_JAW_SKILL_MODES,
            FULL_GRIP_SKILL_MODES,
        )

        self.assertIn("grip_the_cube_near_v1", FIXED_JAW_SKILL_MODES)
        self.assertIn("grip_the_cube_near_v1", FULL_GRIP_SKILL_MODES)
        self.assertEqual(
            COLOR_SHAPE_SKILL_TASK_TEMPLATES["grip_the_cube_near_v1"],
            "grip the {color} {shape} and lift",
        )

    def test_continuous_skill_keeps_the_full_grip_observation_contract(self) -> None:
        from export_so101_teacher_rollouts_lerobot import (
            COLOR_SHAPE_SKILL_TASK_TEMPLATES,
            FIXED_JAW_SKILL_MODES,
            FULL_GRIP_SKILL_MODES,
        )

        self.assertIn("grip_the_cube_continuous_v1", FIXED_JAW_SKILL_MODES)
        self.assertIn("grip_the_cube_continuous_v1", FULL_GRIP_SKILL_MODES)
        self.assertEqual(
            COLOR_SHAPE_SKILL_TASK_TEMPLATES["grip_the_cube_continuous_v1"],
            "grip the {color} {shape} and lift",
        )

    def test_continuous_teacher_routes_bridge_to_both_solvers(self) -> None:
        import export_so101_teacher_rollouts_lerobot as exporter

        near = {"q_open": np.zeros(6), "meta": {"score": -2.0}}
        mid = {"q_open": np.ones(6), "meta": {"score": -1.0}}
        with (
            patch.object(
                exporter,
                "_target_radius_from_shoulder_pan_axis",
                return_value=0.19,
            ),
            patch.object(
                exporter,
                "_make_near_range_fixed_jaw_teacher_targets",
                return_value=[near],
            ) as near_factory,
            patch.object(
                exporter,
                "_make_fast_fixed_jaw_teacher_targets",
                return_value=[mid],
            ) as mid_factory,
        ):
            rows = exporter._make_continuous_range_fixed_jaw_teacher_targets(
                object(), min_floor_clearance_m=0.01
            )

        near_factory.assert_called_once()
        mid_factory.assert_called_once()
        self.assertEqual(
            [row["meta"]["solver_profile"] for row in rows],
            ["near_contact", "mid_fixed_jaw"],
        )
        self.assertGreater(rows[0]["meta"]["score"], rows[1]["meta"]["score"])
        self.assertTrue(
            exporter._uses_near_contact_success_contract(
                "grip_the_cube_continuous_v1", rows[0]["meta"]
            )
        )
        self.assertFalse(
            exporter._uses_near_contact_success_contract(
                "grip_the_cube_continuous_v1", rows[1]["meta"]
            )
        )

    def test_continuous_teacher_avoids_extra_solver_outside_bridge(self) -> None:
        import export_so101_teacher_rollouts_lerobot as exporter

        for radius, expected in (
            (0.10, ("near_contact",)),
            (0.18, ("near_contact", "mid_fixed_jaw")),
            (0.22, ("mid_fixed_jaw", "near_contact")),
            (0.30, ("mid_fixed_jaw",)),
        ):
            with self.subTest(radius=radius):
                self.assertEqual(
                    exporter._continuous_teacher_solver_profiles(radius),
                    expected,
                )

    def test_continuous_auto_path_uses_verified_solver_trajectory(self) -> None:
        import export_so101_teacher_rollouts_lerobot as exporter

        self.assertEqual(
            exporter._resolve_full_grip_trajectory_variant(
                skill_mode="grip_the_cube_continuous_v1",
                requested_variant="auto",
                best_meta={"solver_profile": "near_contact"},
            ),
            "direct_align",
        )
        self.assertEqual(
            exporter._resolve_full_grip_trajectory_variant(
                skill_mode="grip_the_cube_continuous_v1",
                requested_variant="auto",
                best_meta={"solver_profile": "mid_fixed_jaw"},
            ),
            "standard",
        )

    def test_auto_path_is_rejected_for_legacy_full_grip_modes(self) -> None:
        import export_so101_teacher_rollouts_lerobot as exporter

        with self.assertRaisesRegex(
            ValueError, "requires grip_the_cube_continuous_v1"
        ):
            exporter._resolve_full_grip_trajectory_variant(
                skill_mode="grip_the_cube_v1",
                requested_variant="auto",
                best_meta={},
            )

    def test_workspace_probe_accepts_continuous_teacher_contract(self) -> None:
        from physical_ai_agent.so101_workspace_probe import TeacherContract

        contract = TeacherContract(
            skill_mode="grip_the_cube_continuous_v1",
            trajectory_variant="auto",
        )
        self.assertEqual(contract.skill_mode, "grip_the_cube_continuous_v1")
        self.assertEqual(contract.trajectory_variant, "auto")

    def test_contact_centric_near_teacher_grasps_at_ten_centimeters(self) -> None:
        from export_so101_teacher_rollouts_lerobot import (
            _make_near_range_fixed_jaw_teacher_targets,
            _set_target_object_xy,
            _set_target_object_yaw,
            _write_fixed_jaw_edge_episode,
        )
        from train_so101_wrist_ego_visual_servo import (
            WristEgoServoConfig,
            _make_policy_renderers,
            _set_qpos,
            make_high_contrast_picklift_env,
        )

        env = make_high_contrast_picklift_env(
            target_object_color="green",
            object_half_sizes=(0.015,),
            spawn_center=(0.15, 0.0),
            spawn_min_radius=0.1,
            spawn_max_radius=0.3,
            spawn_angle_half_range_deg=90.0,
        )
        renderers = None
        try:
            seed = 510_000_001
            home_qpos = np.asarray(
                [0.0, -math.pi / 2.0, math.pi / 2.0, 0.66, math.pi / 2.0, -0.17453],
                dtype=np.float32,
            )
            target_xy = [0.11547838125436784, 0.00000525602343]
            env.reset(seed=seed)
            _set_qpos(env, home_qpos)
            _set_target_object_xy(env, target_xy)
            _set_target_object_yaw(env, 47.0)
            renderers = _make_policy_renderers(
                env, WristEgoServoConfig(width=256, height=256)
            )
            candidates = _make_near_range_fixed_jaw_teacher_targets(
                env, min_floor_clearance_m=0.01
            )

            self.assertGreater(len(candidates), 0)
            self.assertTrue(
                all(
                    row["meta"]["solver_profile"] == "near_contact"
                    and row["meta"]["ik_close_sweep_floor_clearance_m"] >= 0.01
                    and row["meta"]["cube_face_normal_parallel_error_deg"] <= 3.0
                    for row in candidates
                )
            )

            successful = None
            for rank, candidate in enumerate(
                sorted(candidates, key=lambda row: row["meta"]["score"], reverse=True)
            ):
                env.reset(seed=seed)
                _set_qpos(env, home_qpos)
                _set_target_object_xy(env, target_xy)
                _set_target_object_yaw(env, 47.0)
                result = _write_fixed_jaw_edge_episode(
                    dataset=None,
                    env=env,
                    renderers=renderers,
                    q_open=candidate["q_open"],
                    seed=seed,
                    search_steps=0,
                    teacher_visible=True,
                    best_meta=candidate["meta"],
                    skill_mode="grip_the_cube_near_v1",
                    approach_steps=34,
                    settle_steps=10,
                    close_steps=42,
                    close_alignment_gate_mode="geometry_only",
                    close_alignment_limits=None,
                    trajectory_variant="direct_align",
                    grip_the_cube_start_profile="home",
                    lift_steps=90,
                    lift_target_height=0.06,
                    lift_success_height=0.05,
                    lift_controller_z_error=0.015,
                    episode_index=0,
                    random_start_joint_std=0.0,
                    move_target_z_offset=0.06,
                    terminal_hold_steps=12,
                    move_and_align_near_target_correction_ratio=0.0,
                    edge_contact_xy_success_threshold=0.012,
                    edge_contact_parallel_success_threshold_deg=3.0,
                    near_target_joint_std=0.0,
                    near_target_xy_std=0.0,
                    above_edge_start_joint_std=0.0,
                    above_edge_start_xy_std=0.0,
                    above_edge_start_z_std=0.0,
                    above_edge_start_min_actual_z=0.0,
                    above_edge_trajectory_variants="standard",
                    above_edge_start_gripper_profile="open",
                    above_edge_terminal_hold_jitter=0,
                    task="grip the green cube and lift",
                    include_camera3_duplicate=False,
                    reset_home_qpos=home_qpos,
                    exact_start_pose=True,
                    min_gripper_floor_clearance_m=0.01,
                    record_dataset_frames=False,
                )
                if result["success"]:
                    successful = (rank, result)
                    break

            self.assertIsNotNone(successful)
            _rank, result = successful
            self.assertTrue(result["final_info"]["is_grasped"])
            self.assertGreaterEqual(result["final_info"]["lift_height"], 0.05)
            self.assertGreaterEqual(
                result["gripper_floor_clearance_gate"]["minimum_clearance_m"],
                0.01,
            )
            contact_sample = result["near_contact_alignment_sample"]
            self.assertIsNotNone(contact_sample)
            self.assertLessEqual(contact_sample["parallel_error_deg"], 3.0)
            self.assertTrue(contact_sample["capture_geometry"]["cube_center_between_jaws"])
            self.assertEqual(result["phase_counts"]["terminal_hold"], 12)
        finally:
            if renderers is not None:
                for renderer in renderers.values():
                    renderer.close()
            env.close()


if __name__ == "__main__":
    unittest.main()
