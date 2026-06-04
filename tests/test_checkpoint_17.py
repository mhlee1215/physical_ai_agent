from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.checkpoints.checkpoint_17 import run_checkpoint


class Checkpoint17Test(TestCase):
    def test_so101_multi_camera_input_capture_writes_two_visual_inputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            try:
                report = run_checkpoint(output_dir=Path(tmpdir), steps=2)
            except RuntimeError as exc:
                self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

            self.assertEqual(report.status, "passed")
            self.assertTrue(all(report.checks.values()))
            self.assertEqual(report.metrics["visual_input_names"], ["top_down", "wrist_cam"])
            self.assertIn("observation.images.wrist_cam", report.metrics["lerobot_feature_keys"])
            self.assertIn("observation.images.top_down", report.metrics["lerobot_feature_keys"])
            for artifact_path in report.artifacts.values():
                path = Path(artifact_path)
                self.assertTrue(path.exists(), artifact_path)
                self.assertGreater(path.stat().st_size, 0, artifact_path)
