from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import json

from physical_ai_agent.checkpoints.checkpoint_24 import (
    _build_maniskill_smolvla_batch,
    _extract_rgb_images,
    _write_smolvla_real_manifest,
    run_checkpoint,
)


class _FakeActionSpace:
    shape = (8,)


class _FakeEnv:
    action_space = _FakeActionSpace()


class _FakeFeature:
    shape = (3, 16, 16)


class _FakeStateFeature:
    shape = (6,)


class _FakePolicyConfig:
    robot_state_feature = _FakeStateFeature()
    image_features = {
        "observation.images.camera1": _FakeFeature(),
        "observation.images.camera2": _FakeFeature(),
    }
    device = "cpu"


class _FakePolicy:
    config = _FakePolicyConfig()


class Checkpoint24Test(TestCase):
    def test_maniskill_checkpoint_writes_plan_and_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            report = run_checkpoint(output_dir=Path(tmpdir), steps=1)

            self.assertEqual(report.status, "passed")
            self.assertTrue(Path(report.artifacts["checkpoint_report"]).exists())
            self.assertTrue(Path(report.artifacts["smolvla_maniskill_eval_plan"]).exists())
            self.assertTrue(Path(report.artifacts["robocasa_checkpoint_plan"]).exists())
            self.assertIn("rollout_status", report.metrics)

    def test_require_maniskill_fails_when_rollout_is_blocked(self) -> None:
        with TemporaryDirectory() as tmpdir:
            report = run_checkpoint(
                output_dir=Path(tmpdir),
                env_id="DefinitelyMissingManiSkillTask-v0",
                fallback_env_ids=(),
                steps=1,
                require_maniskill=True,
            )

            self.assertEqual(report.status, "failed")
            self.assertFalse(report.checks["cp24_require_real_maniskill_rollout"])
            self.assertTrue(Path(report.artifacts["maniskill_blocker"]).exists())

    def test_no_fallback_keeps_requested_env_authoritative(self) -> None:
        with TemporaryDirectory() as tmpdir:
            report = run_checkpoint(
                output_dir=Path(tmpdir),
                env_id="DefinitelyMissingManiSkillTask-v0",
                fallback_env_ids=(),
                steps=1,
                require_maniskill=True,
            )

            self.assertEqual(report.status, "failed")
            self.assertEqual(report.metrics["attempted_env_ids"], ["DefinitelyMissingManiSkillTask-v0"])
            self.assertEqual(report.metrics["executed_env_id"], "DefinitelyMissingManiSkillTask-v0")

    def test_maniskill_checkpoint_can_use_headless_fallback_env(self) -> None:
        with TemporaryDirectory() as tmpdir:
            report = run_checkpoint(
                output_dir=Path(tmpdir),
                env_id="DefinitelyMissingManiSkillTask-v0",
                fallback_env_ids=("Empty-v1",),
                episodes=2,
                steps=1,
                require_maniskill=True,
            )

            self.assertEqual(report.status, "passed")
            self.assertEqual(report.metrics["executed_env_id"], "Empty-v1")
            self.assertEqual(report.metrics["rollout_episodes"], 2)
            self.assertEqual(report.metrics["rollout_steps"], 2)
            self.assertIn("DefinitelyMissingManiSkillTask-v0", report.metrics["env_blockers"])

    def test_maniskill_checkpoint_records_policy_metrics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            report = run_checkpoint(
                output_dir=Path(tmpdir),
                env_id="Empty-v1",
                fallback_env_ids=(),
                episodes=1,
                steps=1,
                policies=("random", "zero"),
                require_maniskill=True,
            )
            metrics = json.loads(Path(report.artifacts["maniskill_metrics"]).read_text(encoding="utf-8"))

            self.assertEqual(report.status, "passed")
            self.assertEqual(metrics["policies"], ["random", "zero"])
            self.assertEqual(metrics["episodes_per_policy"], 1)
            self.assertEqual(metrics["episodes"], 2)
            self.assertIn("random", metrics["policy_metrics"])
            self.assertIn("zero", metrics["policy_metrics"])

    def test_real_image_episode_summaries_include_rollout_artifact_fields_when_blocked_or_run(self) -> None:
        with TemporaryDirectory() as tmpdir:
            report = run_checkpoint(
                output_dir=Path(tmpdir),
                env_id="DefinitelyMissingManiSkillTask-v0",
                fallback_env_ids=(),
                episodes=1,
                steps=1,
                policies=("zero",),
                require_maniskill=False,
                real_images=True,
            )

            self.assertEqual(report.metrics["real_images"], True)
            self.assertIn("maniskill_metrics", report.artifacts)

    def test_maniskill_checkpoint_records_smolvla_dry_bridge(self) -> None:
        with TemporaryDirectory() as tmpdir:
            report = run_checkpoint(
                output_dir=Path(tmpdir),
                env_id="Empty-v1",
                fallback_env_ids=(),
                episodes=1,
                steps=1,
                policies=("smolvla_dry",),
                require_maniskill=True,
            )
            metrics = json.loads(Path(report.artifacts["maniskill_metrics"]).read_text(encoding="utf-8"))
            bridge_path = Path(metrics["smolvla_dry_bridge_manifest"])
            bridge = json.loads(bridge_path.read_text(encoding="utf-8"))

            self.assertEqual(report.status, "passed")
            self.assertIn("smolvla_dry", metrics["policy_metrics"])
            self.assertTrue(bridge_path.exists())
            self.assertEqual(bridge["policy"], "smolvla_dry")
            self.assertIn("observation.state", bridge["feature_keys"])

    def test_smolvla_real_manifest_is_not_dry_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "smolvla_real_manifest.json"
            _write_smolvla_real_manifest(
                path=manifest_path,
                runtime={"smolvla_real": object()},
                env=_FakeEnv(),
                records=[
                    {
                        "policy": "smolvla_real",
                        "policy_metadata": {
                            "feature_keys": ["observation.state"],
                            "state_dim": 6,
                            "image_feature_mapping": {"observation.images.camera1": "zero_camera"},
                            "raw_action_dim": 6,
                        },
                    }
                ],
                model_id="lerobot/smolvla_base",
                local_files_only=False,
                input_image_path="smolvla_real_input.png",
                rollout_gif_path="smolvla_real_rollout.gif",
                rollout_frame_paths=["smolvla_real_frames/step_000.png"],
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["policy"], "smolvla_real")
            self.assertEqual(manifest["status"], "passed")
            self.assertTrue(manifest["loaded"])
            self.assertEqual(manifest["raw_action_dim"], 6)
            self.assertEqual(manifest["rollout_gif"], "smolvla_real_rollout.gif")
            self.assertEqual(manifest["rollout_frames"], ["smolvla_real_frames/step_000.png"])
            self.assertIn("select_action()", manifest["note"])

    def test_smolvla_real_batch_can_use_maniskill_rgb_observation(self) -> None:
        import numpy as np

        obs = {
            "agent": {"qpos": np.ones((1, 6), dtype=np.float32)},
            "sensor_data": {
                "base_camera": {
                    "rgb": np.full((1, 8, 10, 3), 127, dtype=np.uint8),
                }
            },
        }

        images = _extract_rgb_images(obs)
        batch, metadata = _build_maniskill_smolvla_batch(
            _FakePolicy(),
            obs,
            use_real_images=True,
        )

        self.assertEqual(sorted(images), ["base_camera"])
        self.assertTrue(metadata["real_images"])
        self.assertEqual(metadata["camera_sources"], ["base_camera"])
        self.assertEqual(metadata["image_feature_mapping"]["observation.images.camera1"], "base_camera")
        self.assertEqual(tuple(batch["observation.images.camera1"].shape), (1, 3, 16, 16))
        self.assertGreater(float(batch["observation.images.camera1"].sum()), 0.0)
