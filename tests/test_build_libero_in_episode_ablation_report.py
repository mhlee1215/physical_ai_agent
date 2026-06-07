import json
import tempfile
from pathlib import Path
from unittest import TestCase

from scripts.build_libero_in_episode_ablation_report import load_condition


class LiberoInEpisodeAblationReportTest(TestCase):
    def test_load_condition_reads_eval_info_and_rollout_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "noop"
            (root / "eval_logs").mkdir(parents=True)
            (root / "eval_logs" / "eval_info.json").write_text(
                json.dumps({"overall": {"pc_success": 100.0, "eval_s": 12.0}}),
                encoding="utf-8",
            )
            (root / "in_episode_trace.jsonl").write_text(
                json.dumps(
                    {
                        "event": "rollout_summary",
                        "success": True,
                        "action_step_count": 6,
                        "verifier_trigger_count": 1,
                        "intervention_count": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            row = load_condition(f"noop={root}")

        self.assertEqual(row.condition, "noop")
        self.assertTrue(row.success)
        self.assertEqual(row.action_step_count, 6)
        self.assertEqual(row.verifier_trigger_count, 1)
        self.assertEqual(row.intervention_count, 1)
        self.assertEqual(row.success_per_action_step, 1 / 6)
        self.assertEqual(row.success_per_eval_minute, 5.0)
