from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.checkpoints.checkpoint_07_13 import run_checkpoint
from physical_ai_agent.data.so101_demo_dataset import write_demo_dataset
from physical_ai_agent.sim.so101_nexus_env import SO101Step


class Checkpoint0713Test(TestCase):
    def test_so101_smolvla_dry_pipeline_writes_required_artifacts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            try:
                report = run_checkpoint(output_dir=Path(tmpdir), steps=4)
            except RuntimeError as exc:
                self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

            self.assertEqual(report.status, "passed")
            self.assertTrue(all(report.checks.values()))
            self.assertEqual(report.metrics["rollout_steps"], 4)
            self.assertEqual(report.metrics["action_dim"], 6)
            for artifact_path in report.artifacts.values():
                path = Path(artifact_path)
                self.assertTrue(path.exists(), artifact_path)
                self.assertGreater(path.stat().st_size, 0, artifact_path)

    def test_demo_dataset_rewrites_episode_file(self) -> None:
        step = SO101Step(
            step=0,
            observation=[0.1, 0.2],
            action=[0.0, 0.0],
            reward=1.0,
            terminated=False,
            truncated=False,
            info={},
        )
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_demo_dataset(output_dir, [step], "first")
            write_demo_dataset(output_dir, [step], "second")

            episodes = (output_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(episodes), 1)
            self.assertIn("second", episodes[0])
