from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.agent_core.planner import RuleBasedSO101Planner
from physical_ai_agent.agent_core.verifier import SO101SimulationStateVerifier
from physical_ai_agent.checkpoints.checkpoint_20 import run_checkpoint as run_checkpoint_20
from physical_ai_agent.checkpoints.checkpoint_21 import run_checkpoint as run_checkpoint_21
from physical_ai_agent.checkpoints.checkpoint_22 import run_checkpoint as run_checkpoint_22
from physical_ai_agent.checkpoints.checkpoint_23 import run_checkpoint as run_checkpoint_23


class Checkpoint20To23Test(TestCase):
    def test_rule_based_planner_creates_ordered_subgoals(self) -> None:
        plan = RuleBasedSO101Planner().plan(task="reach_target", env_id="MuJoCoReach-v1")

        self.assertEqual([subgoal.name for subgoal in plan.subgoals], [
            "stabilize_arm",
            "approach_target",
            "finish_reach",
        ])
        self.assertTrue(all(subgoal.retry_budget == 1 for subgoal in plan.subgoals))

    def test_simulation_state_verifier_uses_distance_threshold(self) -> None:
        subgoal = RuleBasedSO101Planner().plan("reach_target", "MuJoCoReach-v1").subgoals[1]
        decision = SO101SimulationStateVerifier().verify(
            subgoal,
            {"tcp_to_target_dist": "0.140", "success": "False"},
        )

        self.assertTrue(decision.passed)
        self.assertEqual(decision.metric_name, "tcp_to_target_dist")

    def test_checkpoint_20_writes_plan(self) -> None:
        with TemporaryDirectory() as tmpdir:
            report = run_checkpoint_20(output_dir=Path(tmpdir))

            self.assertEqual(report.status, "passed")
            self.assertTrue(Path(report.artifacts["plan"]).exists())

    def test_checkpoint_21_to_23_execute_or_skip_when_so101_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            try:
                cp21 = run_checkpoint_21(output_dir=root / "cp21")
                cp22 = run_checkpoint_22(output_dir=root / "cp22")
                cp23 = run_checkpoint_23(output_dir=root / "cp23")
            except RuntimeError as exc:
                self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

            self.assertEqual(cp21.status, "passed")
            self.assertEqual(cp22.status, "passed")
            self.assertEqual(cp23.status, "passed")
            self.assertTrue(Path(cp23.artifacts["comparison_markdown"]).exists())
            self.assertGreaterEqual(cp22.metrics["retry_events"], 1)
