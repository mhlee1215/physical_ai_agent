from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_mycobot_280_ground_pickup_lerobot_dataset import export_plan
from scripts.evaluate_mycobot280_smolvla_policy import (
    _aggregate_episode_summaries,
    _clip_policy_action,
    _failed_gates,
    _failure_reason,
    _randomized_schedule_from_manifest,
    _resolve_render_camera_profile,
    _yaw_schedule,
    build_eval_report,
)
from scripts.plan_mycobot280_smolvla_training import build_dry_run_report
from scripts.validate_mycobot280_training_dataset import validate_config


JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint7_to_joint6",
    "gripper_controller",
]


class MyCobot280SmolVLAReadinessTest(unittest.TestCase):
    def test_repo_config_blocks_when_dataset_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = validate_config(
                config_path=Path("configs/mycobot280/training_datasets/ground_pickup_tiny_smoke.json"),
                dataset_root_override=Path(tmp) / "missing_dataset",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["config_summary"]["state_dim"], 7)
        self.assertEqual(report["config_summary"]["action_dim"], 7)
        self.assertIn("source dataset manifest is missing", report["warnings"][0])

    def test_validates_fixture_dataset_and_builds_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = _write_fixture_dataset(tmp_path / "dataset")
            config_path = _write_config(tmp_path / "config.json", dataset_root)

            validation = validate_config(config_path=config_path)
            dry_run = build_dry_run_report(config_path=config_path)

            self.assertEqual(validation["status"], "passed")
            self.assertEqual(validation["dataset_report"]["passed_episodes"], 2)
            self.assertEqual(dry_run["status"], "ready")
            self.assertEqual(dry_run["resolved_contract"]["state_dim"], 7)
            self.assertIn("tiny_smolvla_smoke_when_runtime_available", dry_run["commands"])
            self.assertIn("closed_loop_eval_stub", dry_run["commands"])

    def test_export_plan_reports_smolvla_features_without_lerobot_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = _write_fixture_dataset(tmp_path / "dataset")
            output_root = tmp_path / "lerobot_plan"

            report = export_plan(
                source_root=dataset_root,
                output_root=output_root,
                repo_id="physical-ai-agent/test-mycobot280",
                dry_run=True,
                overwrite=False,
            )

            self.assertEqual(report["status"], "passed")
            self.assertTrue((output_root / "mycobot280_ground_pickup_lerobot_plan.json").exists())
            self.assertEqual(report["features"]["observation.state"]["shape"], [7])
            self.assertEqual(report["features"]["action"]["names"], JOINT_NAMES)
            self.assertFalse(report["source_quality"]["teacher_attachment_enabled"])

    def test_split_manifest_validates_and_exports_selected_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = _write_split_fixture_dataset(tmp_path / "dataset")
            manifest_path = dataset_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["observation_camera"] = _close_camera_contract()
            manifest["image_mime_type"] = "image/png"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            config_path = _write_config(tmp_path / "config.json", dataset_root)
            config = json.loads(config_path.read_text())
            config["source_dataset"]["expected_generation_mode"] = (
                "deterministic_pose_diverse_teacher_aligned"
            )
            config["source_dataset"]["expected_splits"] = {"train": 1, "validation": 1}
            config["source_dataset"]["expected_all_frames_rendered"] = True
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

            validation = validate_config(config_path=config_path)
            output_root = tmp_path / "lerobot_train"
            report = export_plan(
                source_root=dataset_root,
                output_root=output_root,
                repo_id="physical-ai-agent/test-mycobot280-train",
                split="train",
                dry_run=False,
                overwrite=False,
            )
            episodes = [
                json.loads(line)
                for line in (output_root / "data" / "episodes.jsonl").read_text().splitlines()
            ]
            all_output_root = tmp_path / "lerobot_all"
            export_plan(
                source_root=dataset_root,
                output_root=all_output_root,
                repo_id="physical-ai-agent/test-mycobot280-all",
                split="all",
                dry_run=False,
                overwrite=False,
            )
            all_episodes = [
                json.loads(line)
                for line in (all_output_root / "data" / "episodes.jsonl").read_text().splitlines()
            ]
            all_frames = [
                json.loads(line)
                for line in (all_output_root / "data" / "frames.jsonl").read_text().splitlines()
            ]

            self.assertEqual(validation["status"], "passed")
            self.assertEqual(
                validation["dataset_report"]["split_counts"],
                {"train": 1, "validation": 1},
            )
            self.assertEqual(report["source_split"], "train")
            self.assertEqual(report["observation_camera"], _close_camera_contract())
            self.assertEqual(report["image_mime_type"], "image/png")
            self.assertEqual(report["episodes"], 1)
            self.assertEqual(report["exported_frames"], 2)
            self.assertEqual(episodes[0]["split"], "train")
            self.assertEqual([item["episode_index"] for item in all_episodes], [0, 1])
            self.assertEqual([item["source_episode_index"] for item in all_episodes], [0, 0])
            self.assertEqual({item["episode_index"] for item in all_frames}, {0, 1})
            for frame in all_frames:
                source_image = dataset_root / frame["metadata"]["source_image"]
                target_image = all_output_root / frame["observation.images.camera1"]
                self.assertEqual(frame["metadata"]["image_materialization"], "hardlink")
                self.assertEqual(
                    (source_image.stat().st_dev, source_image.stat().st_ino),
                    (target_image.stat().st_dev, target_image.stat().st_ino),
                )

    def test_eval_stub_blocks_without_policy_and_plans_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = _write_fixture_dataset(tmp_path / "dataset")
            config_path = _write_config(tmp_path / "config.json", dataset_root)
            output_dir = tmp_path / "eval"

            blocked = build_eval_report(
                policy_path=tmp_path / "missing_policy",
                config_path=config_path,
                output_dir=output_dir,
                episodes=None,
                dry_run=False,
                require_policy=False,
            )
            planned = build_eval_report(
                policy_path=tmp_path / "missing_policy",
                config_path=config_path,
                output_dir=output_dir,
                episodes=2,
                dry_run=True,
                require_policy=False,
            )

            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(planned["status"], "planned")
            self.assertEqual(planned["episodes"], 2)
            self.assertIn("max_pad_cube_penetration_m", planned["metrics"])

    def test_camera_profile_resolves_from_config_and_allows_explicit_override(self) -> None:
        config = {
            "closed_loop_stub": {
                "render_camera_profile": "ground_pickup_closeup",
            }
        }

        self.assertEqual(
            _resolve_render_camera_profile(config, None),
            "ground_pickup_closeup",
        )
        self.assertEqual(
            _resolve_render_camera_profile(config, "full_robot"),
            "full_robot",
        )
        with self.assertRaisesRegex(ValueError, "unsupported render camera profile"):
            _resolve_render_camera_profile(
                {"closed_loop_stub": {"render_camera_profile": "unknown"}},
                None,
            )

    def test_validator_rejects_source_and_evaluator_camera_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = _write_fixture_dataset(tmp_path / "dataset")
            config_path = _write_config(tmp_path / "config.json", dataset_root)
            config = json.loads(config_path.read_text())
            config["source_dataset"]["expected_observation_camera"] = (
                _close_camera_contract()
            )
            config["closed_loop_stub"]["render_camera_profile"] = "full_robot"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

            report = validate_config(config_path=config_path)

        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("render_camera_profile must match" in item for item in report["errors"])
        )

    def test_randomized_schedule_is_fresh_deterministic_and_manifest_bounded(self) -> None:
        manifest = {
            "dataset_id": "fixture-randomized-source",
            "randomization_enabled": True,
            "randomization": {
                "sampler": "numpy_pcg64_seeded_per_attempt",
                "yaw_delta_rad": {"min": -0.2, "max": 0.2},
                "cube_axis_offset_m": {
                    "center": 0.0015,
                    "jitter_min": -0.0,
                    "jitter_max": 0.0,
                },
                "cube_side_offset_m": {
                    "center": 0.005,
                    "jitter_min": -0.0,
                    "jitter_max": 0.0,
                },
                "cube_mass_kg": {"min": 0.028, "max": 0.036},
                "cube_friction": {"min": 3.4, "max": 4.0},
                "support_friction": {"fixed": 4.0},
                "pad_friction": {"fixed": 640.0},
            },
            "randomized_contact_calibration": {
                "lift_scale": 1.05,
                "pad_cube_solref": [0.01, 1.0],
            },
            "splits": {
                "train": {"episode_summaries": [{"seed": 2800}]},
                "validation": {"episode_summaries": [{"seed": 2802}]},
            },
            "rejected_attempts": [{"seed": 2801}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            schedule, contract = _randomized_schedule_from_manifest(
                manifest_path=manifest_path,
                episodes=3,
                seed_start=92000,
                torch_seed=20260731,
            )
            repeated, _ = _randomized_schedule_from_manifest(
                manifest_path=manifest_path,
                episodes=3,
                seed_start=92000,
                torch_seed=20260731,
            )
            with self.assertRaisesRegex(ValueError, "overlaps source seeds"):
                _randomized_schedule_from_manifest(
                    manifest_path=manifest_path,
                    episodes=2,
                    seed_start=2800,
                    torch_seed=20260731,
                )

        self.assertEqual(schedule, repeated)
        self.assertEqual([item["seed"] for item in schedule], [92000, 92001, 92002])
        self.assertEqual(contract["source_seed_overlap_count"], 0)
        self.assertEqual(
            contract["candidate_selection"],
            "direct_unfiltered_draws_without_teacher_rejection",
        )
        for item in schedule:
            candidate = item["candidate"]
            self.assertGreaterEqual(candidate["yaw_delta_rad"], -0.2)
            self.assertLessEqual(candidate["yaw_delta_rad"], 0.2)
            self.assertGreaterEqual(candidate["cube_mass_kg"], 0.028)
            self.assertLessEqual(candidate["cube_mass_kg"], 0.036)
            self.assertGreaterEqual(candidate["cube_friction"], 3.4)
            self.assertLessEqual(candidate["cube_friction"], 4.0)
            self.assertEqual(item["contact_solref"], [0.01, 1.0])

    def test_closed_loop_helpers_enforce_matched_schedule_and_action_bounds(self) -> None:
        schedule = _yaw_schedule(3, -0.2, 0.2)
        action, clipped = _clip_policy_action(
            [2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 3.0],
            arm_low=[-1.0] * 6,
            arm_high=[1.0] * 6,
        )
        aggregate = _aggregate_episode_summaries(
            [
                {
                    "success": True,
                    "failure_reason": "passed",
                    "final_cube_lift_m": 0.06,
                    "max_pad_cube_penetration_m": 0.002,
                },
                {
                    "success": False,
                    "failure_reason": "final_cube_lift_below_threshold",
                    "final_cube_lift_m": 0.01,
                    "max_pad_cube_penetration_m": 0.001,
                },
            ]
        )
        reason = _failure_reason(
            success=False,
            placement_guard={"passed": True},
            max_penetration=0.001,
            final_lift=0.01,
            final_contact_pads=0,
            lift_contact_steps=0,
            post_contact_steps=0,
            post_min_lift=0.0,
        )
        failed_gates = _failed_gates(
            placement_guard={"passed": True},
            max_penetration=0.0031,
            final_lift=0.01,
            final_contact_pads=2,
            lift_contact_steps=80,
            post_contact_steps=300,
            post_min_lift=0.01,
        )
        hidden_canonical_gate = _failed_gates(
            placement_guard={"passed": True},
            max_penetration=0.002,
            final_lift=0.06,
            final_contact_pads=2,
            lift_contact_steps=80,
            post_contact_steps=300,
            post_min_lift=0.05,
            initial_contact_pads=1,
        )

        self.assertEqual(schedule, [-0.2, 0.0, 0.2])
        self.assertEqual(action, [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(clipped, 3)
        self.assertEqual(aggregate["success_rate"], 0.5)
        self.assertEqual(
            aggregate["failure_reason_counts"],
            {"passed": 1, "final_cube_lift_below_threshold": 1},
        )
        self.assertEqual(reason, "final_cube_lift_below_threshold")
        self.assertEqual(
            failed_gates,
            [
                "max_pad_cube_penetration_exceeded",
                "final_cube_lift_below_threshold",
                "post_lift_height_below_threshold",
            ],
        )
        self.assertEqual(
            hidden_canonical_gate,
            ["initial_pad_cube_contact_present"],
        )


def _write_config(path: Path, dataset_root: Path) -> Path:
    config = json.loads(Path("configs/mycobot280/training_datasets/ground_pickup_tiny_smoke.json").read_text())
    config["source_dataset"]["root"] = str(dataset_root)
    config["source_dataset"]["expected_episodes"] = 2
    config["source_dataset"]["expected_min_frames"] = 4
    config["lerobot_conversion"]["output_root"] = str(path.parent / "lerobot")
    config["training_smoke"]["output_dir"] = str(path.parent / "train")
    config["training_smoke"]["tensorboard_dir"] = str(path.parent / "train" / "tensorboard")
    config["training_smoke"]["checkpoint_dir"] = str(path.parent / "train" / "checkpoints")
    config["closed_loop_stub"]["output_dir"] = str(path.parent / "eval")
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def _write_fixture_dataset(root: Path) -> Path:
    (root / "episodes").mkdir(parents=True)
    (root / "frames").mkdir(parents=True)
    for index in range(2):
        rows = []
        for frame_index in range(2):
            render_path = f"frames/episode_{index:04d}_frame_{frame_index:04d}.bmp"
            (root / render_path).write_bytes(b"fixture image placeholder")
            rows.append(
                {
                    "episode_index": index,
                    "frame_index": frame_index,
                    "task": "pick up the cube from the work mat with the myCobot 280 Pi adaptive gripper",
                    "observation": {"state": [0.0] * 7, "images": {"render": render_path}},
                    "action": [0.1] * 7,
                    "info": {
                        "joint_names": JOINT_NAMES,
                        "ground_pickup": {"cube_lift_m": 0.055, "pad_cube_contacted_pads": 2},
                    },
                }
            )
        episode_path = root / "episodes" / f"episode_{index:04d}.jsonl"
        episode_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    manifest = {
        "format": "mycobot_jsonl_v1",
        "dataset_id": "fixture",
        "robot": "myCobot 280 Pi + adaptive gripper",
        "model_profile": "mycobot_280_pi_adaptive_gripper",
        "task": "pick up the cube from the work mat with the myCobot 280 Pi adaptive gripper",
        "generation_mode": "deterministic_fixed_task",
        "randomization_enabled": False,
        "teacher_attachment_enabled": False,
        "object_teleport_during_pickup_lift": False,
        "cube_half_size_m": 0.015,
        "cube_mass_kg": 0.032,
        "success_criteria": {
            "final_cube_lift_m": 0.05,
            "final_gripper_cube_contact_pads": 2,
            "lift_best_sustained_two_pad_steps": 60,
            "post_lift_hold_best_sustained_two_pad_steps": 300,
            "post_lift_hold_min_cube_lift_m": 0.045,
            "max_pad_cube_penetration_m": 0.003,
        },
        "episodes": 2,
        "passed_episodes": 2,
        "frames": 4,
        "aggregate_metrics": {
            "passed_episodes": 2,
            "failed_episodes": 0,
            "min_final_cube_lift_m": 0.055,
            "min_lift_best_sustained_two_pad_steps": 60,
            "min_post_lift_hold_sustained_two_pad_steps": 300,
            "min_post_lift_hold_cube_lift_m": 0.046,
            "max_pad_cube_penetration_m": 0.0025,
            "max_lift_pad_cube_penetration_m": 0.002,
        },
        "joint_names": JOINT_NAMES,
        "action_names": JOINT_NAMES,
        "episode_summaries": [
            {
                "episode_index": 0,
                "path": "episodes/episode_0000.jsonl",
                "frames": 2,
                "rendered_frames": 2,
                "success": True,
            },
            {
                "episode_index": 1,
                "path": "episodes/episode_0001.jsonl",
                "frames": 2,
                "rendered_frames": 2,
                "success": True,
            },
        ],
        "failed_episodes": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _write_split_fixture_dataset(root: Path) -> Path:
    _write_fixture_dataset(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summaries = manifest.pop("episode_summaries")
    train_summary = {**summaries[0], "split": "train"}
    validation_summary = {**summaries[1], "episode_index": 0, "split": "validation"}
    manifest["generation_mode"] = "deterministic_pose_diverse_teacher_aligned"
    manifest["splits"] = {
        "train": {
            "requested_episodes": 1,
            "accepted_episodes": 1,
            "episode_summaries": [train_summary],
        },
        "validation": {
            "requested_episodes": 1,
            "accepted_episodes": 1,
            "episode_summaries": [validation_summary],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _close_camera_contract() -> dict[str, object]:
    return {
        "profile": "ground_pickup_closeup",
        "resolution_hw": [256, 256],
        "mode": "free_camera",
        "target": "initial_cube_xyz_plus_[0,0,0.035]_m",
        "distance_m": 0.24,
        "azimuth_deg": 215.0,
        "elevation_deg": -10.0,
    }


if __name__ == "__main__":
    unittest.main()
