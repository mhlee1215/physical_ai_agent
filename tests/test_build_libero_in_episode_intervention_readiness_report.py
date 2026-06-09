import json
import tempfile
from pathlib import Path
from unittest import TestCase

from scripts.build_libero_in_episode_intervention_readiness_report import (
    check_eval_infos,
    check_lerobot_eval_source,
    check_libero_env_source,
)


class LiberoInterventionReadinessReportTest(TestCase):
    def test_source_checks_find_rollout_hook_and_missing_default_action_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "lerobot_eval.py"
            source.write_text(
                "\n".join(
                    [
                        "def rollout(",
                        "    pass",
                        "all_actions.append(torch.from_numpy(action_numpy))",
                        'ret = {ACTION: torch.stack(all_actions, dim=1), "success": torch.stack(all_successes, dim=1), "done": torch.stack(all_dones, dim=1)}',
                        "observation, reward, terminated, truncated, info = env.step(action_numpy)",
                        '"per_episode": [{"episode_ix": 0}]',
                    ]
                ),
                encoding="utf-8",
            )

            statuses = {check.name: check.status for check in check_lerobot_eval_source(source)}

        self.assertEqual(statuses["lerobot_rollout_function"], "pass")
        self.assertEqual(statuses["rollout_records_actions"], "pass")
        self.assertEqual(statuses["rollout_records_success_done"], "pass")
        self.assertEqual(statuses["online_hook_location"], "pass")
        self.assertEqual(statuses["default_eval_info_action_steps"], "fail")

    def test_libero_env_check_warns_about_auto_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "libero.py"
            source.write_text(
                "\n".join(
                    [
                        "is_success = self._env.check_success()",
                        'info.update({"is_success": is_success})',
                        "if terminated:",
                        "    self.reset()",
                    ]
                ),
                encoding="utf-8",
            )

            statuses = {check.name: check.status for check in check_libero_env_source(source)}

        self.assertEqual(statuses["libero_step_exposes_success"], "pass")
        self.assertEqual(statuses["libero_step_auto_resets_on_terminal"], "warn")

    def test_eval_info_check_requires_action_step_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eval_info = Path(tmp) / "eval_info.json"
            eval_info.write_text(
                json.dumps(
                    {
                        "overall": {"eval_s": 12.5},
                        "per_task": [
                            {
                                "task_group": "libero_goal",
                                "task_id": 0,
                                "metrics": {"successes": [True, False]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            checks = check_eval_infos([eval_info])
            by_suffix = {check.name.split(":")[-1]: check.status for check in checks}

        self.assertEqual(by_suffix["eval_seconds"], "pass")
        self.assertEqual(by_suffix["successes"], "pass")
        self.assertEqual(by_suffix["action_step_counts"], "fail")
