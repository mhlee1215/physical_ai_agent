from pathlib import Path
import sys
from unittest import TestCase

from physical_ai_agent.checkpoints.checkpoint_01 import run_checkpoint


class Checkpoint01Test(TestCase):
    def test_scaffold_smoke_passes_without_strict_sim_deps(self) -> None:
        if sys.version_info < (3, 11):
            self.skipTest("checkpoint 01 requires Python >= 3.11")

        report = run_checkpoint(
            config_path=Path("configs/sim/libero.yaml"),
            strict_local_sim=False,
            strict_sim_deps=False,
            probe_mujoco=False,
            probe_libero_env=False,
        )

        self.assertEqual(report.checkpoint, "checkpoint_01_libero_smoke")
        self.assertEqual(report.status, "passed")
        self.assertTrue(any(result.name == "config:libero" for result in report.results))
