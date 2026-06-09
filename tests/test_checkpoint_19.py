from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.checkpoints.checkpoint_19 import run_checkpoint


class Checkpoint19Test(TestCase):
    def test_smolvla_real_camera_input_gate_reports_mapping_or_blocker(self) -> None:
        with TemporaryDirectory() as tmpdir:
            try:
                report = run_checkpoint(output_dir=Path(tmpdir), steps=1)
            except RuntimeError as exc:
                self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

            self.assertTrue(report.checks["cp19_real_smolvla_attempted"])
            self.assertTrue(report.checks["cp19_real_camera_inputs_enabled"])
            if report.metrics["smolvla_status"] == "passed":
                self.assertEqual(report.status, "passed")
                self.assertTrue(report.checks["cp19_zero_image_tensor_removed"])
                self.assertIn("wrist_cam", report.metrics["image_feature_mapping"].values())
                self.assertIn("egocentric_cam", report.metrics["image_feature_mapping"].values())
            else:
                self.assertEqual(report.status, "failed")
                self.assertTrue(Path(report.artifacts["smolvla_blocker"]).exists())
