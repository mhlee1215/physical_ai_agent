from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.checkpoints.checkpoint_02_04 import run_checkpoint


class Checkpoint0204Test(TestCase):
    def test_random_policy_eval_writes_required_artifacts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            report = run_checkpoint(
                output_dir=Path(tmpdir),
                episodes=1,
                episode_steps=4,
                seed=123,
            )

            self.assertEqual(report.status, "passed")
            self.assertTrue(all(report.checks.values()))
            self.assertEqual(report.metrics["episodes"], 1)
            for artifact_path in report.artifacts.values():
                path = Path(artifact_path)
                self.assertTrue(path.exists(), artifact_path)
                self.assertGreater(path.stat().st_size, 0, artifact_path)
