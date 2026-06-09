from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.checkpoints.checkpoint_14_15 import run_checkpoint


class Checkpoint1415Test(TestCase):
    def test_render_and_real_smolvla_gates_write_reports_or_blockers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            try:
                report = run_checkpoint(output_dir=Path(tmpdir), steps=2)
            except RuntimeError as exc:
                self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

            self.assertEqual(report.status, "passed")
            self.assertTrue(report.checks["cp14_3d_render_attempted"])
            self.assertTrue(report.checks["cp14_3d_render_saved_or_blocker_documented"])
            self.assertTrue(report.checks["cp15_real_smolvla_inference_attempted"])
            self.assertTrue(report.checks["cp15_real_smolvla_action_or_blocker_documented"])
            self.assertTrue(Path(report.artifacts["checkpoint_report"]).exists())
            self.assertTrue(Path(report.artifacts["render_report"]).exists())
            self.assertTrue(Path(report.artifacts["smolvla_report"]).exists())
