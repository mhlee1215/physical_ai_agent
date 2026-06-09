from unittest import TestCase

from physical_ai_agent.agent_core.libero_agentic_retry import (
    aggregate_retry_metrics,
    build_retry_plan,
    retry_task_ids_arg,
)


def _eval_info(successes_by_task: dict[int, list[bool]]) -> dict:
    per_task = []
    for task_id, successes in successes_by_task.items():
        per_task.append(
            {
                "task_group": "libero_10",
                "task_id": task_id,
                "metrics": {
                    "successes": successes,
                    "sum_rewards": [1.0 if value else 0.0 for value in successes],
                    "max_rewards": [1.0 if value else 0.0 for value in successes],
                },
            }
        )
    return {"per_task": per_task, "per_group": {}, "overall": {}}


class LiberoAgenticRetryTest(TestCase):
    def test_retry_plan_selects_failed_task_ids(self) -> None:
        plan = build_retry_plan(
            _eval_info({
                0: [True, True],
                6: [False, True],
                8: [False, False],
            }),
            task_group="libero_10",
        )

        self.assertEqual(plan.failed_task_ids, [6, 8])
        self.assertEqual(plan.failed_episodes, 3)
        self.assertEqual(plan.total_episodes, 6)
        self.assertEqual(retry_task_ids_arg(plan), "[6,8]")

    def test_aggregate_counts_success_once_and_recovery(self) -> None:
        baseline = _eval_info({
            0: [True, False],
            6: [False, False],
        })
        retry = _eval_info({
            0: [False, True],
            6: [True, False],
        })

        metrics, trace = aggregate_retry_metrics(baseline, retry, task_group="libero_10")

        self.assertEqual(metrics.baseline_success_rate, 25.0)
        self.assertEqual(metrics.retry_success_rate, 50.0)
        self.assertEqual(metrics.success_once_rate, 75.0)
        self.assertEqual(metrics.recovery_success_rate, 100.0 * 2 / 3)
        self.assertEqual(metrics.failed_episodes, 3)
        self.assertEqual(metrics.recovered_episodes, 2)
        self.assertEqual(metrics.baseline_attempts, 4)
        self.assertEqual(metrics.retry_attempts, 4)
        self.assertEqual(metrics.total_attempts, 8)
        self.assertEqual(metrics.environment_resets, 8)
        self.assertEqual(metrics.extra_environment_resets, 4)
        self.assertEqual(metrics.success_once_per_attempt, 3 / 8)
        self.assertEqual(metrics.recovered_per_retry_attempt, 2 / 4)
        self.assertFalse(metrics.action_step_count_available)
        self.assertEqual(len(trace), 4)
        self.assertTrue(any(record["retry_attempted"] for record in trace))
