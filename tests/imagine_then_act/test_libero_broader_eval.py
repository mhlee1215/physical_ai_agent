import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[2]


def load_broader_eval_module():
    spec = importlib.util.spec_from_file_location(
        "run_imagine_then_act_libero_broader_eval_for_test",
        ROOT / "scripts" / "run_imagine_then_act_libero_broader_eval.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LiberoBroaderEvalTest(TestCase):
    def test_parse_task_ids_supports_ranges_lists_and_deduplication(self) -> None:
        module = load_broader_eval_module()

        self.assertEqual(module.parse_task_ids("0-2,2,4"), (0, 1, 2, 4))
        with self.assertRaises(ValueError):
            module.parse_task_ids("3-1")

    def test_build_entrypoint_argv_uses_list_and_single_entrypoint_for_both_methods(self) -> None:
        module = load_broader_eval_module()
        with TemporaryDirectory() as tmpdir:
            config = module.build_config(
                Namespace(
                    suite="libero_goal",
                    task_ids="0-9",
                    seed=1201,
                    methods="policy_only,ita_baseline_fallback",
                    target="runpod",
                    output_dir=tmpdir,
                    python_bin=sys.executable,
                    policy_path="lerobot/smolvla_libero",
                    policy_num_steps=10,
                    policy_n_action_steps=15,
                    num_candidates=2,
                    chunk_steps=15,
                    dry_run=True,
                    monitor_interval=0.0,
                    early_stop_zero_at_half=False,
                )
            )
            policy_dir = Path(tmpdir) / "policy"
            ita_dir = Path(tmpdir) / "ita"
            policy_argv = module.build_entrypoint_argv(config, "policy_only", 0, policy_dir)
            ita_argv = module.build_entrypoint_argv(config, "ita_baseline_fallback", 0, ita_dir)

            self.assertIsInstance(policy_argv, list)
            self.assertIsInstance(ita_argv, list)
            self.assertIn("scripts/run_imagine_then_act.py", policy_argv)
            self.assertIn("scripts/run_imagine_then_act.py", ita_argv)
            self.assertIn("--eval-method", policy_argv)
            self.assertEqual(policy_argv[policy_argv.index("--eval-method") + 1], "policy_only")
            self.assertEqual(ita_argv[ita_argv.index("--eval-method") + 1], "ita_baseline_fallback")
            self.assertIn("--policy-num-steps", policy_argv)
            self.assertIn("10", policy_argv)
            self.assertIn("--policy-n-action-steps", policy_argv)
            self.assertIn("15", policy_argv)
            self.assertIn("--dry-run", policy_argv)
            self.assertNotIn(" ", policy_argv[policy_argv.index("--task-id") + 1])
            self.assertFalse(any(item == "&&" or item == ";" for item in policy_argv))

    def test_summary_aggregation_counts_success_and_records_early_stop_status(self) -> None:
        module = load_broader_eval_module()
        with TemporaryDirectory() as tmpdir:
            config = module.build_config(
                Namespace(
                    suite="libero_goal",
                    task_ids="0-3",
                    seed=1201,
                    methods="policy_only,ita_baseline_fallback",
                    target="runpod",
                    output_dir=tmpdir,
                    python_bin=sys.executable,
                    policy_path="lerobot/smolvla_libero",
                    policy_num_steps=10,
                    policy_n_action_steps=15,
                    num_candidates=2,
                    chunk_steps=15,
                    dry_run=False,
                    monitor_interval=0.0,
                    early_stop_zero_at_half=True,
                )
            )
            records = [
                {"event": "task_result", "method": "policy_only", "task_id": 0, "pc_success": 0.0, "status": "passed"},
                {"event": "task_result", "method": "policy_only", "task_id": 1, "pc_success": 100.0, "status": "passed"},
                {"event": "task_result", "method": "ita_baseline_fallback", "task_id": 0, "pc_success": 0.0, "status": "passed"},
            ]

            summary = module.summarize_results(records, config, stop_reason="ita zero success")

            self.assertEqual(summary["status"], "bug_suspect_zero_success_at_half")
            self.assertEqual(summary["by_method"]["policy_only"]["completed"], 2)
            self.assertEqual(summary["by_method"]["policy_only"]["success_count"], 1)
            self.assertEqual(summary["by_method"]["ita_baseline_fallback"]["completed"], 1)

    def test_result_record_reads_entrypoint_report_without_shell_path(self) -> None:
        module = load_broader_eval_module()
        with TemporaryDirectory() as tmpdir:
            config = module.build_config(
                Namespace(
                    suite="libero_goal",
                    task_ids="0",
                    seed=1201,
                    methods="policy_only",
                    target="runpod",
                    output_dir=tmpdir,
                    python_bin=sys.executable,
                    policy_path="lerobot/smolvla_libero",
                    policy_num_steps=10,
                    policy_n_action_steps=15,
                    num_candidates=2,
                    chunk_steps=15,
                    dry_run=False,
                    monitor_interval=0.0,
                    early_stop_zero_at_half=False,
                )
            )
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir()
            report_path = run_dir / "report.json"
            stdout_path = run_dir / "stdout.json"
            stderr_path = run_dir / "stderr.log"
            monitor_path = run_dir / "monitor.jsonl"
            report_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "eval_method": "policy_only",
                        "benchmark_result": {
                            "pc_success": 100.0,
                            "success": True,
                            "selected_candidate_applied": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            stdout_path.write_text(json.dumps({"report_path": str(report_path)}), encoding="utf-8")

            record = module.result_record(
                config=config,
                method="policy_only",
                task_id=0,
                argv=["python", "-B", "scripts/run_imagine_then_act.py"],
                run_dir=run_dir,
                process_result={
                    "exit_code": 0,
                    "elapsed_s": 1.0,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "monitor_path": str(monitor_path),
                },
            )

            self.assertFalse(record["used_shell"])
            self.assertEqual(record["entrypoint"], "scripts/run_imagine_then_act.py")
            self.assertEqual(record["pc_success"], 100.0)
            self.assertTrue(record["benchmark_success"])
            self.assertFalse(record["selected_candidate_applied"])
