from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scripts.export_mycobot_280_ground_pickup_teacher_dataset as teacher
from scripts.check_mycobot_280_ground_pickup_randomized_dataset import (
    validate_randomized_manifest,
)
from scripts.export_mycobot_280_ground_pickup_randomized_dataset import (
    build_parser,
    randomized_candidates,
    split_uniqueness_audit,
)


class MyCobot280GroundPickupRandomizedDatasetTest(unittest.TestCase):
    def test_cli_defaults_emit_training_compatible_all_frame_images(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual((args.width, args.height), (256, 256))
        self.assertEqual(args.render_every, 1)

    def test_candidates_are_seeded_reproducible_and_within_ranges(self) -> None:
        kwargs = {
            "seed": 2800,
            "max_attempts": 8,
            "yaw_min": -0.20,
            "yaw_max": 0.20,
            "axis_jitter_m": 0.0,
            "side_jitter_m": 0.0,
            "mass_min_kg": 0.028,
            "mass_max_kg": 0.036,
            "cube_friction_min": 3.4,
            "cube_friction_max": 4.0,
        }
        first = list(randomized_candidates(**kwargs))
        second = list(randomized_candidates(**kwargs))

        self.assertEqual(first, second)
        self.assertEqual(len({candidate.signature for candidate in first}), len(first))
        for index, candidate in enumerate(first):
            self.assertEqual(candidate.spawn_seed, 2800 + index)
            self.assertGreaterEqual(candidate.yaw_delta_rad, -0.20)
            self.assertLessEqual(candidate.yaw_delta_rad, 0.20)
            self.assertGreaterEqual(candidate.cube_mass_kg, 0.028)
            self.assertLessEqual(candidate.cube_mass_kg, 0.036)
            self.assertGreaterEqual(candidate.cube_friction, 3.4)
            self.assertLessEqual(candidate.cube_friction, 4.0)
            self.assertEqual(candidate.support_friction, 4.0)
            self.assertEqual(candidate.pad_friction, 640.0)

    def test_different_root_seed_changes_candidates(self) -> None:
        common = {
            "max_attempts": 2,
            "yaw_min": -0.20,
            "yaw_max": 0.20,
            "axis_jitter_m": 0.0,
            "side_jitter_m": 0.0,
            "mass_min_kg": 0.028,
            "mass_max_kg": 0.036,
            "cube_friction_min": 3.4,
            "cube_friction_max": 4.0,
        }
        first = list(randomized_candidates(seed=2800, **common))
        second = list(randomized_candidates(seed=3800, **common))

        self.assertNotEqual(first[0].signature, second[0].signature)

    def test_split_audit_detects_no_overlap(self) -> None:
        train = [_summary(index=0, seed=10, xy=(0.1, 0.2), trajectory_hash="train")]
        validation = [_summary(index=1, seed=20, xy=(0.2, 0.3), trajectory_hash="validation")]

        audit = split_uniqueness_audit(train, validation)

        self.assertEqual(audit["seed_overlap_count"], 0)
        self.assertEqual(audit["pose_overlap_count"], 0)
        self.assertEqual(audit["trajectory_hash_overlap_count"], 0)
        self.assertEqual(audit["factor_overlap_count"], 0)

    def test_split_audit_detects_factor_overlap_even_with_distinct_seeds(self) -> None:
        train = [_summary(index=0, seed=10, xy=(0.1, 0.2), trajectory_hash="train")]
        validation_summary = _summary(index=1, seed=20, xy=(0.2, 0.3), trajectory_hash="validation")
        validation_summary["candidate"] = dict(train[0]["candidate"])

        audit = split_uniqueness_audit(train, [validation_summary])

        self.assertEqual(audit["seed_overlap_count"], 0)
        self.assertEqual(audit["factor_overlap_count"], 1)

    def test_candidate_physics_refreshes_mujoco_constants_and_requires_scene_geoms(self) -> None:
        env = _FakeEnv()

        teacher._apply_candidate_physics(
            env,
            cube_mass=0.035,
            cube_friction=3.7,
            support_friction=4.0,
            pad_friction=640.0,
            refresh_model_constants=True,
            contact_solref=(0.01, 1.0),
        )

        self.assertEqual(env._mujoco.set_const_calls, 1)
        self.assertAlmostEqual(float(env.model.body_mass[0]), 0.035)
        self.assertTrue(np.all(env.model.body_inertia[0] > 0.0))
        self.assertAlmostEqual(float(env.model.geom_friction[0, 0]), 3.7)
        self.assertAlmostEqual(float(env.model.geom_friction[3, 0]), 640.0)
        self.assertTrue(np.allclose(env.model.geom_solref[[0, 3, 4]], [0.01, 1.0]))
        del env._mujoco.ids[("geom", teacher.ROBOT_RIGHT_PAD)]
        with self.assertRaisesRegex(RuntimeError, "fingertip pad"):
            teacher._apply_candidate_physics(
                env,
                cube_mass=0.035,
                cube_friction=3.7,
                support_friction=4.0,
                pad_friction=640.0,
                refresh_model_constants=True,
                contact_solref=(0.01, 1.0),
            )

    def test_manifest_validator_accepts_randomized_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = _summary(
                index=0,
                seed=2800,
                xy=(0.10, 0.20),
                trajectory_hash="train",
                path="splits/train/episodes/episode_0000.jsonl",
            )
            validation = _summary(
                index=1,
                seed=2801,
                xy=(0.11, 0.22),
                trajectory_hash="validation",
                path="splits/validation/episodes/episode_0000.jsonl",
            )
            manifest = _manifest(train, validation)
            _write_episode(root, train, split="train")
            _write_episode(root, validation, split="validation")

            report = validate_randomized_manifest(
                manifest,
                dataset_root=root,
                train_episodes=1,
                val_episodes=1,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["errors"], [])

    def test_manifest_validator_rejects_malformed_episode_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = _summary(
                index=0,
                seed=2800,
                xy=(0.10, 0.20),
                trajectory_hash="train",
                path="splits/train/episodes/episode_0000.jsonl",
            )
            validation = _summary(
                index=1,
                seed=2801,
                xy=(0.11, 0.22),
                trajectory_hash="validation",
                path="splits/validation/episodes/episode_0000.jsonl",
            )
            manifest = _manifest(train, validation)
            _write_episode(root, train, split="train")
            _write_episode(root, validation, split="validation")
            (root / str(train["path"])).write_text("{}\n", encoding="utf-8")

            report = validate_randomized_manifest(
                manifest,
                dataset_root=root,
                train_episodes=1,
                val_episodes=1,
            )

        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("missing fields" in error for error in report["errors"]))

    def test_manifest_validator_rejects_non_object_and_wrong_identity_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = _summary(
                index=0, seed=2800, xy=(0.10, 0.20), trajectory_hash="train",
                path="splits/train/episodes/episode_0000.jsonl",
            )
            validation = _summary(
                index=1, seed=2801, xy=(0.11, 0.22), trajectory_hash="validation",
                path="splits/validation/episodes/episode_0000.jsonl",
            )
            manifest = _manifest(train, validation)
            _write_episode(root, train, split="train")
            _write_episode(root, validation, split="validation")
            (root / str(train["path"])).write_text("null\n", encoding="utf-8")
            validation_row = json.loads((root / str(validation["path"])).read_text(encoding="utf-8"))
            validation_row["split"] = "train"
            (root / str(validation["path"])).write_text(json.dumps(validation_row) + "\n", encoding="utf-8")

            report = validate_randomized_manifest(
                manifest, dataset_root=root, train_episodes=1, val_episodes=1,
            )

        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("must be a JSON object" in error for error in report["errors"]))
        self.assertTrue(any("does not match summary" in error for error in report["errors"]))


def _summary(
    *,
    index: int,
    seed: int,
    xy: tuple[float, float],
    trajectory_hash: str,
    path: str = "episode.jsonl",
) -> dict[str, object]:
    return {
        "episode_index": index,
        "seed": seed,
        "initial_cube_xy": list(xy),
        "trajectory_hash": trajectory_hash,
        "path": path,
        "success": True,
        "frames": 1,
        "rendered_frames": 1,
        "candidate": {
            "cube_mass_kg": 0.030 + index * 0.002,
            "cube_friction": 3.5 + index * 0.5,
            "cube_axis_offset_m": 0.001 + index * 0.0002,
            "cube_side_offset_m": -0.002 + index * 0.0003,
            "yaw_delta_rad": -0.1 + index * 0.2,
            "support_friction": 4.0,
            "pad_friction": 640.0,
        },
    }


def _manifest(train: dict[str, object], validation: dict[str, object]) -> dict[str, object]:
    train["split"] = "train"
    train["split_episode_index"] = 0
    validation["split"] = "validation"
    validation["split_episode_index"] = 0
    return {
        "format": "mycobot_jsonl_v1",
        "schema_version": 2,
        "generation_mode": "seeded_randomized_teacher_aligned_rejection_sampled",
        "randomization_enabled": True,
        "teacher_attachment_enabled": False,
        "object_teleport_during_pickup_lift": False,
        "requested_episodes": 2,
        "accepted_episodes": 2,
        "failed_episodes": [],
        "acceptance_rate": 1.0,
        "randomized_contact_calibration": {"lift_scale": 1.05, "pad_cube_solref": [0.01, 1.0]},
        "render_every": 1,
        "joint_names": [f"joint_{index}" for index in range(7)],
        "action_names": [f"joint_{index}" for index in range(7)],
        "randomization": {
            "yaw_delta_rad": {"min": -0.2, "max": 0.2},
            "cube_mass_kg": {"min": 0.028, "max": 0.036},
            "cube_friction": {"min": 3.4, "max": 4.0},
            "cube_axis_offset_m": {"center": 0.0011, "jitter_min": -0.001, "jitter_max": 0.001},
            "cube_side_offset_m": {"center": -0.00185, "jitter_min": -0.001, "jitter_max": 0.001},
            "support_friction": {"fixed": 4.0},
            "pad_friction": {"fixed": 640.0},
        },
        "splits": {
            "train": {"accepted_episodes": 1, "episode_summaries": [train]},
            "validation": {"accepted_episodes": 1, "episode_summaries": [validation]},
        },
        "split_uniqueness_audit": split_uniqueness_audit([train], [validation]),
        "aggregate_metrics": {
            "min_final_cube_lift_m": 0.055,
            "min_post_lift_hold_sustained_two_pad_steps": 300,
            "min_post_lift_hold_cube_lift_m": 0.046,
            "max_pad_cube_penetration_m": 0.0028,
            "pose_coverage": {"unique_pose_count": 2, "unique_trajectory_hashes": 2},
            "factor_coverage": {
                "cube_mass_kg": {"span": 0.002},
                "cube_friction": {"span": 0.5},
                "cube_axis_offset_m": {"span": 0.0002},
                "cube_side_offset_m": {"span": 0.0003},
            },
        },
        "rejected_attempts": [],
    }


def _write_episode(root: Path, summary: dict[str, object], *, split: str) -> None:
    episode_path = root / str(summary["path"])
    image_path = f"splits/{split}/frames/episode_0000/frame_0000.bmp"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path = root / image_path
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(b"BM")
    row = {
        "episode_index": summary["episode_index"],
        "split": split,
        "split_episode_index": 0,
        "frame_index": 0,
        "timestamp": 0.0,
        "phase": "approach_down_to_cube_on_mat",
        "task": "pick up the cube",
        "observation": {"state": [0.0] * 10, "images": {"render": image_path}},
        "action": [0.0] * 7,
        "reward": 0.0,
        "done": False,
        "info": {"candidate": dict(summary["candidate"])},
    }
    episode_path.write_text(json.dumps(row) + "\n", encoding="utf-8")


class _FakeObj:
    mjOBJ_BODY = "body"
    mjOBJ_GEOM = "geom"


class _FakeMujoco:
    mjtObj = _FakeObj

    def __init__(self) -> None:
        self.set_const_calls = 0
        self.ids = {
            ("body", teacher.nexus.TASK_CUBE_BODY): 0,
            ("geom", teacher.nexus.TASK_CUBE_GEOM): 0,
            ("geom", "nexus_work_mat"): 1,
            ("geom", "nexus_floor"): 2,
            ("geom", teacher.ROBOT_LEFT_PAD): 3,
            ("geom", teacher.ROBOT_RIGHT_PAD): 4,
        }

    def mj_name2id(self, _model: object, object_type: str, name: str) -> int:
        return self.ids.get((object_type, name), -1)

    def mj_setConst(self, _model: object, _data: object) -> None:
        self.set_const_calls += 1


class _FakeModel:
    def __init__(self) -> None:
        self.body_mass = np.zeros(1, dtype=float)
        self.body_inertia = np.zeros((1, 3), dtype=float)
        self.geom_friction = np.zeros((5, 3), dtype=float)
        self.geom_solref = np.zeros((5, 2), dtype=float)


class _FakeEnv:
    def __init__(self) -> None:
        self._mujoco = _FakeMujoco()
        self.model = _FakeModel()
        self.data = object()


if __name__ == "__main__":
    unittest.main()
