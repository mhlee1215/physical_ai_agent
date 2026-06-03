from pathlib import Path
from unittest import TestCase

from physical_ai_agent.checkpoints.checkpoint_01 import run_checkpoint


class Checkpoint01Test(TestCase):
    def test_scaffold_smoke_passes_without_strict_sim_deps(self) -> None:
        report = run_checkpoint(
            config_path=Path("configs/sim/libero.yaml"),
            strict_sim_deps=False,
            probe_libero_env=False,
        )

        self.assertEqual(report.checkpoint, "checkpoint_01_libero_smoke")
        self.assertEqual(report.status, "passed")
        self.assertTrue(any(result.name == "config:libero" for result in report.results))
