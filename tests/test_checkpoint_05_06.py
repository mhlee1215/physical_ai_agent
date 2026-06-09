from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.checkpoints.checkpoint_05_06 import run_checkpoint
from physical_ai_agent.policies.smolvla_adapter import SmolVLAPolicyAdapter, probe_smolvla


class Checkpoint0506Test(TestCase):
    def test_policy_adapter_and_smolvla_probe_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            report = run_checkpoint(output_dir=Path(tmpdir))

            self.assertEqual(report.status, "passed")
            self.assertTrue(report.checks["cp05_action_chunk_created"])
            self.assertTrue(report.checks["cp05_action_chunk_executes_one_step"])
            self.assertTrue(report.checks["cp06_smolvla_adapter_created"])
            self.assertTrue(report.checks["cp06_smolvla_probe_ran"])
            self.assertTrue(report.checks["cp06_smolvla_policy_class_importable"])
            for artifact_path in report.artifacts.values():
                path = Path(artifact_path)
                self.assertTrue(path.exists(), artifact_path)
                self.assertGreater(path.stat().st_size, 0, artifact_path)

    def test_smolvla_probe_documents_missing_dependencies(self) -> None:
        probe = probe_smolvla()
        if not probe.ready:
            self.assertTrue(probe.blockers)
        self.assertIn("lerobot", probe.imports)

    def test_smolvla_adapter_fails_fast_when_dependencies_are_missing(self) -> None:
        adapter = SmolVLAPolicyAdapter()
        if not adapter.ready:
            with self.assertRaises(RuntimeError):
                adapter.action_chunk(object(), "test instruction")
