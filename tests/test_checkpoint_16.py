from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.checkpoints.checkpoint_16 import run_checkpoint


class Checkpoint16Test(TestCase):
    def test_so101_camera_input_capture_writes_manifest_and_preview(self) -> None:
        with TemporaryDirectory() as tmpdir:
            try:
                report = run_checkpoint(output_dir=Path(tmpdir), steps=2)
            except RuntimeError as exc:
                self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

            self.assertEqual(report.status, "passed")
            self.assertTrue(all(report.checks.values()))
            self.assertEqual(report.metrics["current_camera_names"], ["wrist_cam"])
            for artifact_path in report.artifacts.values():
                path = Path(artifact_path)
                self.assertTrue(path.exists(), artifact_path)
                self.assertGreater(path.stat().st_size, 0, artifact_path)
