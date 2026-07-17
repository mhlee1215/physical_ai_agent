from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


RECIPE_PATH = Path("configs/so101/dataset_generation/grip_the_cube_v2.json")
ADDITIONAL_RECIPE_PATH = Path("configs/so101/dataset_generation/grip_the_cube_v2_1.json")
V25_RECIPE_PATH = Path("configs/so101/dataset_generation/grip_the_cube_v2_5.json")
ALIGN_RECIPE_PATH = Path(
    "configs/so101/dataset_generation/grip_the_cube_v2_5_align_trajectory.json"
)
FULL_PHOTOREAL_RECIPE_PATHS = (
    Path("configs/so101/dataset_generation/grip_the_cube_v2_5_photoreal.json"),
    Path(
        "configs/so101/dataset_generation/"
        "grip_the_cube_v2_5_align_trajectory_photoreal.json"
    ),
)


class SO101DatasetGenerationRecipeTests(unittest.TestCase):
    def test_recipe_generation_is_append_only_by_default(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/generate_so101_dataset_recipe.py",
                "--recipe",
                str(RECIPE_PATH),
                "--dry-run",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        stages = json.loads(completed.stdout)["stages"]
        self.assertTrue(stages)
        self.assertTrue(all("--overwrite" not in stage["command"] for stage in stages))

    def test_append_only_preflight_rejects_existing_final_root(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from generate_so101_dataset_recipe import _require_append_only_output_roots

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "already_exists"
            root.mkdir()
            recipe = {"splits": {"train": {"output_root": str(root)}}}
            with self.assertRaisesRegex(FileExistsError, "append-only"):
                _require_append_only_output_roots(recipe, split="train", overwrite=False)

            _require_append_only_output_roots(recipe, split="train", overwrite=True)

    def test_grip_v2_dry_run_covers_full_pipeline(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/generate_so101_dataset_recipe.py",
                "--recipe",
                str(RECIPE_PATH),
                "--dry-run",
                "--overwrite",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        stages = {stage["name"]: stage["command"] for stage in payload["stages"]}

        self.assertEqual(
            [name for name in stages if name.startswith("export:train:")],
            ["export:train:bin5", "export:train:bin6", "export:train:bin9", "export:train:bin10"],
        )
        self.assertEqual(
            [name for name in stages if name.startswith("export:validation:")],
            [
                "export:validation:bin5",
                "export:validation:bin6",
                "export:validation:bin9",
                "export:validation:bin10",
            ],
        )
        train5 = stages["export:train:bin5"]
        validation5 = stages["export:validation:bin5"]
        self.assertEqual(_value_after(train5, "--seed"), "5000000")
        self.assertEqual(_value_after(validation5, "--seed"), "20000000")
        self.assertEqual(_value_after(validation5, "--grid-lookup-start-index"), "253")
        self.assertEqual(_value_after(train5, "--width"), "256")
        self.assertEqual(_value_after(train5, "--height"), "256")
        self.assertEqual(_value_after(train5, "--edge-contact-parallel-success-threshold-deg"), "3.0")
        self.assertIn("--deterministic-camera-bin-lookup", train5)
        self.assertIn("--overwrite", train5)
        self.assertIn("merge:train", stages)
        self.assertIn("sidecar:validation", stages)
        self.assertIn("closed-loop-starts:validation", stages)
        self.assertIn("audit:train-vs-validation", stages)
        self.assertEqual(
            _value_after(stages["audit:train-vs-validation"], "--expected-train-bins"),
            "5:75,6:75,9:75,10:75",
        )
        self.assertEqual(
            _value_after(stages["audit:train-vs-validation"], "--expected-validation-bins"),
            "5:13,6:13,9:12,10:12",
        )

    def test_recipe_rejects_overlapping_effective_seed_ranges(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from generate_so101_dataset_recipe import _validate_unique_seed_ranges

        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        broken = copy.deepcopy(recipe)
        broken["splits"]["validation"]["bins"][0].update(
            {"seed": 5_000_000, "lookup_start_index": 0}
        )
        with self.assertRaisesRegex(ValueError, "seed ranges overlap"):
            _validate_unique_seed_ranges(broken)

    def test_recipe_contract_is_train_300_validation_50(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(sum(row["episodes"] for row in recipe["splits"]["train"]["bins"]), 300)
        self.assertEqual(
            sum(row["episodes"] for row in recipe["splits"]["validation"]["bins"]), 50
        )
        self.assertEqual(recipe["splits"]["validation"]["closed_loop"]["episodes"], 10)
        self.assertEqual(recipe["common"]["terminal_hold_steps"], 12)
        self.assertEqual(recipe["audit"]["expected_prompt"], "grip the green cube and lift")

    def test_additional_recipe_is_balanced_train500_and_audits_both_v2_splits(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/generate_so101_dataset_recipe.py",
                "--recipe",
                str(ADDITIONAL_RECIPE_PATH),
                "--dry-run",
                "--overwrite",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        stages = {row["name"]: row["command"] for row in json.loads(completed.stdout)["stages"]}
        self.assertEqual(len([name for name in stages if name.startswith("export:train:")]), 8)
        self.assertEqual(
            sum(
                int(_value_after(command, "--episodes"))
                for name, command in stages.items()
                if name.startswith("export:train:")
            ),
            500,
        )
        bin10_names = [name for name in stages if name.startswith("export:train:bin10_part")]
        self.assertEqual(len(bin10_names), 5)
        self.assertEqual(
            [_value_after(stages[name], "--grid-lookup-start-index") for name in bin10_names],
            ["0", "234", "468", "702", "936"],
        )
        self.assertEqual(_value_after(stages[bin10_names[0]], "--grid-lookup-resolution"), "241")
        self.assertIn("audit:train-vs-grip_the_cube_v2_train", stages)
        self.assertIn("audit:train-vs-grip_the_cube_v2_validation", stages)

    def test_v25_recipes_require_real_lift_and_matching_terminal_hold(self) -> None:
        for path, profile in ((V25_RECIPE_PATH, "home"), (ALIGN_RECIPE_PATH, "correction")):
            recipe = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(recipe["common"]["grip_the_cube_start_profile"], profile)
            self.assertEqual(recipe["common"]["lift_target_height"], 0.065)
            self.assertEqual(recipe["common"]["lift_controller_z_error"], 0.015)
            self.assertEqual(recipe["common"]["move_target_z_offset"], 0.075)
            self.assertEqual(recipe["audit"]["expected_min_lift_height"], 0.06)
            self.assertGreaterEqual(recipe["audit"]["expected_min_lift_steps"], 20)
            self.assertEqual(recipe["audit"]["terminal_hold_action_tolerance"], 1e-5)
            closed_loop = recipe["splits"]["validation"]["closed_loop"]
            self.assertEqual(closed_loop["success_metric"], "env_success")
            self.assertEqual(closed_loop["lift_success_height"], 0.05)

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_so101_dataset_recipe.py",
                    "--recipe",
                    str(path),
                    "--dry-run",
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            stages = {row["name"]: row["command"] for row in json.loads(completed.stdout)["stages"]}
            export = next(command for name, command in stages.items() if name.startswith("export:train:"))
            self.assertEqual(_value_after(export, "--lift-target-height"), "0.065")
            self.assertEqual(_value_after(export, "--lift-controller-z-error"), "0.015")
            self.assertEqual(_value_after(export, "--move-target-z-offset"), "0.075")
            audit = stages["audit:train-vs-validation"]
            self.assertEqual(_value_after(audit, "--expected-min-lift-height"), "0.06")
            self.assertEqual(_value_after(audit, "--terminal-hold-action-tolerance"), "1e-05")
            loop = stages["closed-loop-starts:validation"]
            self.assertEqual(_value_after(loop, "--success-metric"), "env_success")
            self.assertEqual(_value_after(loop, "--lift-success-height"), "0.05")

    def test_full_photoreal_recipes_cover_all_train_and_validation_frames(self) -> None:
        expected_counts = ((300, 50), (220, 100))
        for path, (train_episodes, validation_episodes) in zip(
            FULL_PHOTOREAL_RECIPE_PATHS, expected_counts, strict=True
        ):
            recipe = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(recipe["common"]["capture_render_replay"])
            self.assertEqual(
                recipe["render_replay"]["capture_mode"],
                "verified_action_replay",
            )
            self.assertEqual(recipe["splits"]["train"]["kind"], "render_derivative")
            self.assertEqual(recipe["splits"]["validation"]["kind"], "render_derivative")
            self.assertEqual(recipe["splits"]["train"]["expected_episodes"], train_episodes)
            self.assertEqual(
                recipe["splits"]["validation"]["expected_episodes"],
                validation_episodes,
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_so101_dataset_recipe.py",
                    "--recipe",
                    str(path),
                    "--dry-run",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            stages = {
                row["name"]: row["command"] for row in json.loads(completed.stdout)["stages"]
            }
            self.assertIn("render-replay:train", stages)
            self.assertIn("render-replay:validation", stages)
            self.assertIn("closed-loop-starts:validation", stages)
            for split in ("train", "validation"):
                replay = stages[f"render-replay:{split}"]
                self.assertIn("--allow-verified-reconstruction", replay)
                self.assertEqual(
                    _value_after(replay, "--dataset-root"),
                    recipe["splits"][split]["source_dataset_root"],
                )
                self.assertEqual(
                    _value_after(stages[f"render:{split}"], "--render-replay-sidecar"),
                    recipe["splits"][split]["render_replay_sidecar"],
                )
            self.assertEqual(_value_after(stages["render:train"], "--frames"), "all")
            self.assertEqual(_value_after(stages["render:validation"], "--frames"), "all")
            self.assertIn("--skip-existing", stages["render:train"])
            self.assertIn("--skip-existing", stages["render:validation"])
            self.assertNotIn("--skip-existing", stages["render-determinism:train"])
            for split in ("train", "validation"):
                builder_cameras = _value_after(
                    stages[f"build-derivative:{split}"], "--camera-keys"
                ).split(",")
                self.assertEqual(
                    builder_cameras,
                    [
                        "observation.images.camera1",
                        "observation.images.camera2",
                        "observation.images.camera3",
                    ],
                )
            self.assertEqual(
                len(_value_after(stages["render:train"], "--episodes").split(",")),
                train_episodes,
            )
            self.assertEqual(
                len(_value_after(stages["render:validation"], "--episodes").split(",")),
                validation_episodes,
            )
            audit = stages["audit:train-vs-validation"]
            self.assertNotEqual(_value_after(audit, "--expected-train-bins"), "")
            self.assertNotEqual(_value_after(audit, "--expected-validation-bins"), "")

    def test_v25_closed_loop_excludes_both_training_reports(self) -> None:
        recipe = json.loads(V25_RECIPE_PATH.read_text(encoding="utf-8"))
        loop = recipe["splits"]["validation"]["closed_loop"]
        expected_reports = [
            "_workspace/so101_lerobot/grip_the_cube_v2_5/so101_lerobot_export_report.json",
            "_workspace/so101_lerobot/grip_the_cube_v2_5_align_trajectory/so101_lerobot_export_report.json",
        ]
        self.assertEqual(loop["exclude_source_reports"], expected_reports)
        self.assertEqual(
            loop["output"],
            "meta/closed_loop/grip_the_cube_v2_5_validation_clean_start10.json",
        )

        completed = subprocess.run(
            [
                sys.executable,
                "scripts/generate_so101_dataset_recipe.py",
                "--recipe",
                str(V25_RECIPE_PATH),
                "--dry-run",
                "--overwrite",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        stages = {row["name"]: row["command"] for row in json.loads(completed.stdout)["stages"]}
        command = stages["closed-loop-starts:validation"]
        actual_reports = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--exclude-source-report"
        ]
        self.assertEqual(actual_reports, expected_reports)

    def test_closed_loop_exclusion_preserves_original_validation_episode_index(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from build_so101_closed_loop_start_report import build_report

        episodes = [
            {
                "seed": seed,
                "forced_spawn_xy": [float(seed), 0.0],
                "grid_balance_bin": 5,
                "sim_snapshot": {},
            }
            for seed in (10, 11, 12)
        ]
        excluded = {
            "episodes": [
                {"seed": 10, "forced_spawn_xy": [99.0, 0.0]},
                {"seed": 99, "forced_spawn_xy": [11.0, 0.0]},
            ]
        }
        report = build_report(
            {"episodes": episodes},
            count=1,
            bins=[5],
            source_path=Path("validation.json"),
            excluded_sources=[(Path("train.json"), excluded)],
        )

        self.assertEqual(report["episodes"][0]["seed"], 12)
        self.assertEqual(report["episodes"][0]["source_validation_episode_index"], 2)
        self.assertEqual(report["exclusion_contract"]["excluded_validation_episodes"], 2)

    def test_v25_recipe_declares_every_source_dataset(self) -> None:
        for path in (V25_RECIPE_PATH, ALIGN_RECIPE_PATH):
            recipe = json.loads(path.read_text(encoding="utf-8"))
            declared = {Path(value) for value in recipe["source_datasets"]}
            referenced = {
                Path(report).parent
                for builder in recipe["lookup_builders"]
                for report in builder["source_reports"]
            }
            self.assertEqual(referenced, declared)

    def test_align_recipe_builds_split_specific_source_episode_lookups(self) -> None:
        recipe = json.loads(ALIGN_RECIPE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(sum(row["episodes"] for row in recipe["splits"]["train"]["bins"]), 220)
        self.assertEqual(
            sum(row["episodes"] for row in recipe["splits"]["validation"]["bins"]),
            100,
        )
        self.assertEqual(recipe["common"]["near_target_joint_std"], 0.015)
        self.assertEqual(recipe["common"]["near_target_xy_std"], 0.012)
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/generate_so101_dataset_recipe.py",
                "--recipe",
                str(ALIGN_RECIPE_PATH),
                "--dry-run",
                "--overwrite",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        stages = {row["name"]: row["command"] for row in json.loads(completed.stdout)["stages"]}
        self.assertIn("lookup:train_sources", stages)
        self.assertIn("lookup:validation_sources", stages)
        self.assertNotEqual(
            _value_after(stages["export:train:bin5"], "--grid-lookup-cache"),
            _value_after(stages["export:validation:bin5"], "--grid-lookup-cache"),
        )
        self.assertEqual(_value_after(stages["sidecar:train"], "--bin-source"), "report")

    def test_source_lookup_uses_only_successful_episode_contracts(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from build_so101_source_episode_spawn_lookup import build_source_lookup

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "success": True,
                                "seed": 101,
                                "forced_spawn_xy": [0.1, 0.2],
                                "grid_balance_bin": 5,
                            },
                            {
                                "success": False,
                                "seed": 102,
                                "forced_spawn_xy": [0.3, 0.4],
                                "grid_balance_bin": 5,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            payload = build_source_lookup(
                source_reports=[report_path],
                grid_size=4,
                resolution=161,
                x_range=(-0.1, 0.55),
                y_range=(-0.45, 0.45),
                bins=[5],
            )

        self.assertEqual(payload["candidate_kind"], "source_episode_manifest")
        self.assertEqual(payload["lookup"]["5"], [[0.1, 0.2, 101]])

    def test_source_lookup_can_reserve_disjoint_candidate_prefix(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from build_so101_source_episode_spawn_lookup import build_source_lookup

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "success": True,
                                "seed": seed,
                                "forced_spawn_xy": [0.1 + seed * 0.001, 0.2],
                                "grid_balance_bin": 5,
                            }
                            for seed in (101, 102, 103)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            payload = build_source_lookup(
                source_reports=[report_path],
                grid_size=4,
                resolution=161,
                x_range=(-0.1, 0.55),
                y_range=(-0.45, 0.45),
                bins=[5],
                candidate_start_index=2,
            )

        self.assertEqual(payload["candidate_start_index"], 2)
        self.assertEqual(payload["raw_candidate_counts"]["5"], 3)
        self.assertEqual(payload["candidate_counts"]["5"], 1)
        self.assertEqual(payload["lookup"]["5"][0][2], 103)

    def test_fixed_jaw_lift_ignores_success_termination_until_target(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from export_so101_teacher_rollouts_lerobot import (
            _fixed_jaw_lift_target_reached,
            _fixed_jaw_terminal_event_stops_episode,
        )

        self.assertFalse(
            _fixed_jaw_lift_target_reached(
                {"is_grasped": True, "lift_height": 0.051}, target_height=0.08
            )
        )
        self.assertTrue(
            _fixed_jaw_lift_target_reached(
                {"is_grasped": True, "lift_height": 0.081}, target_height=0.08
            )
        )
        self.assertFalse(
            _fixed_jaw_terminal_event_stops_episode(
                "lift", terminated=True, truncated=False
            )
        )
        self.assertFalse(
            _fixed_jaw_terminal_event_stops_episode(
                "terminal_hold", terminated=True, truncated=False
            )
        )
        self.assertTrue(
            _fixed_jaw_terminal_event_stops_episode(
                "lift", terminated=True, truncated=True
            )
        )

    def test_correction_visibility_tries_mirrored_offsets_before_zero(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from export_so101_teacher_rollouts_lerobot import _correction_visibility_scales

        scales = _correction_visibility_scales()
        self.assertEqual(scales[-1], 0.0)
        self.assertEqual(
            scales[:-1],
            (1.0, -1.0, 0.75, -0.75, 0.5, -0.5, 0.25, -0.25),
        )

    def test_correction_trajectory_converges_directly_to_grasp_prepose(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from export_so101_teacher_rollouts_lerobot import (
            _grip_the_cube_correction_phases,
        )

        q_start = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
        q_above = np.asarray([0.5, 0.6, 0.7], dtype=np.float32)
        q_edge = np.asarray([1.0, 1.1, 1.2], dtype=np.float32)
        q_close = np.asarray([1.0, 1.1, -1.2], dtype=np.float32)
        phases = _grip_the_cube_correction_phases(
            q_start=q_start,
            q_above=q_above,
            q_edge=q_edge,
            q_close=q_close,
            approach_steps=24,
            settle_steps=10,
            close_steps=42,
            lift_steps=58,
        )

        self.assertEqual(
            [phase for phase, _start, _target, _steps in phases],
            ["near_target_correct", "gripper_descend", "settle_aligned", "close", "lift"],
        )
        np.testing.assert_array_equal(phases[0][2], q_above)
        np.testing.assert_array_equal(phases[1][1], q_above)
        np.testing.assert_array_equal(phases[1][2], q_edge)
        np.testing.assert_array_equal(phases[3][2], q_close)
        self.assertIsNone(phases[4][2])

    def test_complete_export_shard_can_be_reused(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from generate_so101_dataset_recipe import _export_shard_is_complete

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "shard"
            root.mkdir()
            stage = {
                "command": [
                    sys.executable,
                    "export.py",
                    "--root",
                    str(root),
                    "--episodes",
                    "25",
                ]
            }
            report = root / "so101_lerobot_export_report.json"
            report.write_text(json.dumps({"exported_episodes": 25}), encoding="utf-8")
            self.assertTrue(_export_shard_is_complete(stage))
            report.write_text(json.dumps({"exported_episodes": 24}), encoding="utf-8")
            self.assertFalse(_export_shard_is_complete(stage))

    def test_split_audit_accepts_disjoint_hwc_lerobot_splits(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from audit_so101_dataset_splits import audit_splits

        with tempfile.TemporaryDirectory() as tmpdir:
            train = Path(tmpdir) / "train"
            validation = Path(tmpdir) / "validation"
            _write_fake_split(train, seed=100, spawn_xy=(0.1, 0.2), action_value=1.0)
            _write_fake_split(validation, seed=200, spawn_xy=(0.3, 0.4), action_value=2.0)
            report = audit_splits(
                train_root=train,
                validation_root=validation,
                expected_prompt="grip the green cube and lift",
                expected_resolution=(256, 256),
                expected_train_bins={5: 1},
                expected_validation_bins={5: 1},
                expected_terminal_hold_steps=12,
                max_pre_close_alignment_deg=3.0,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["overlap_counts"], {"seeds": 0, "spawn_xy": 0, "trajectory_hashes": 0})

    def test_terminal_hold_audit_rejects_action_different_from_final_lift(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from audit_so101_dataset_splits import _validate_terminal_hold_actions

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data" / "chunk-000").mkdir(parents=True)
            q_lift = np.arange(6, dtype=np.float32)
            actions = [q_lift.copy() for _ in range(12)]
            actions[-1] = actions[-1].copy()
            actions[-1][0] += 0.01
            pd.DataFrame(
                {"episode_index": [0] * 12, "action": actions}
            ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")

            with self.assertRaisesRegex(ValueError, "terminal hold action mismatch"):
                _validate_terminal_hold_actions(
                    root,
                    episodes=[{"q_lift": q_lift.tolist()}],
                    hold_steps=12,
                    tolerance=1e-5,
                )


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _write_fake_split(root: Path, *, seed: int, spawn_xy: tuple[float, float], action_value: float) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "so101_lerobot_export_report.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "seed": seed,
                        "forced_spawn_xy": list(spawn_xy),
                        "grid_balance_bin": 5,
                        "success": True,
                        "task_success": True,
                        "phase_counts": {"terminal_hold": 12},
                        "pre_close_cube_face_normal_parallel_error_deg": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.camera1": {
                        "shape": [256, 256, 3],
                        "names": ["height", "width", "channels"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"task_index": [0]}, index=pd.Index(["grip the green cube and lift"], name="task")
    ).to_parquet(root / "meta" / "tasks.parquet")
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "action": [np.full(6, action_value, dtype=np.float32)] * 2,
            "observation.state": [np.zeros(6, dtype=np.float32)] * 2,
            "task_index": [0, 0],
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")


if __name__ == "__main__":
    unittest.main()
