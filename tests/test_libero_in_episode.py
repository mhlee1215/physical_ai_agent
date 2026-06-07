import json
import tempfile
from pathlib import Path
from unittest import TestCase

from physical_ai_agent.agent_core.libero_in_episode import (
    ScaleActionIntervention,
    StagnationVerifier,
    run_in_episode_rollout,
    write_result,
)
from scripts.run_libero_in_episode_instrumented_smoke import ConstantPolicy, ToyStagnationEnv


class LiberoInEpisodeRolloutTest(TestCase):
    def test_verifier_triggers_intervention_before_terminal_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            result = run_in_episode_rollout(
                env=ToyStagnationEnv(),
                policy=ConstantPolicy(action=0.6),
                verifier=StagnationVerifier(metric_name="progress", window=3, min_delta=1e-6),
                intervention=ScaleActionIntervention(scale=0.25, intervention_type="scale_next_action"),
                output_jsonl=trace_path,
                task_group="toy_libero_goal",
                task_id=0,
                seed=7,
                max_steps=8,
                max_interventions=1,
            )
            records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

        self.assertTrue(result.success)
        self.assertEqual(result.environment_resets, 1)
        self.assertEqual(result.intervention_count, 1)
        self.assertGreaterEqual(result.verifier_trigger_count, 1)
        self.assertGreater(result.action_step_count, 0)
        self.assertGreater(result.success_per_action_step, 0.0)
        intervention_steps = [record for record in records if record["intervention_type"] == "scale_next_action"]
        self.assertEqual(len(intervention_steps), 1)
        self.assertFalse(intervention_steps[0]["terminated"])

    def test_write_result_includes_cost_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_in_episode_rollout(
                env=ToyStagnationEnv(),
                policy=ConstantPolicy(action=0.6),
                verifier=StagnationVerifier(metric_name="progress", window=3, min_delta=1e-6),
                intervention=ScaleActionIntervention(scale=0.25),
                output_jsonl=tmp_path / "trace.jsonl",
                task_group="toy_libero_goal",
                task_id=0,
                seed=7,
                max_steps=8,
            )
            output_json = tmp_path / "metrics.json"
            output_md = tmp_path / "report.md"
            write_result(result, output_json, output_md)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            report = output_md.read_text(encoding="utf-8")

        self.assertIn("action_step_count", payload)
        self.assertIn("success_per_action_step", payload)
        self.assertIn("environment_resets", report)
