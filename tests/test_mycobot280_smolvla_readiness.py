from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_mycobot_280_ground_pickup_lerobot_dataset import export_plan
from scripts.evaluate_mycobot280_smolvla_policy import build_eval_report
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


if __name__ == "__main__":
    unittest.main()
