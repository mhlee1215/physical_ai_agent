from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.checkpoints.checkpoint_18 import run_checkpoint


class Checkpoint18Test(TestCase):
    def test_so101_egocentric_policy_inputs_record_roles(self) -> None:
        with TemporaryDirectory() as tmpdir:
            try:
                report = run_checkpoint(output_dir=Path(tmpdir), steps=2)
            except RuntimeError as exc:
                self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

            self.assertEqual(report.status, "passed")
            self.assertTrue(all(report.checks.values()))
            self.assertEqual(report.metrics["policy_input_names"], ["wrist_cam", "egocentric_cam"])
            self.assertEqual(report.metrics["debug_input_names"], ["top_down"])
            self.assertEqual(
                report.metrics["lerobot_policy_feature_keys"],
                ["observation.images.wrist_cam", "observation.images.egocentric_cam"],
            )
            for artifact_path in report.artifacts.values():
                path = Path(artifact_path)
                self.assertTrue(path.exists(), artifact_path)
                self.assertGreater(path.stat().st_size, 0, artifact_path)
