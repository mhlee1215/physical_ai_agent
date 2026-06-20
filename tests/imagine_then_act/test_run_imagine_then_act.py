import json
import importlib.util
import io
import sys
import types
from argparse import Namespace
from contextlib import redirect_stdout
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from physical_ai_agent.imagine_then_act.utils import (
    build_run_config,
    build_execution_contract,
    build_real_backend_command,
    parse_candidate_seeds,
    prepare_run_artifacts,
    generate_candidate_chunks,
    imagine_candidates,
    judge_candidates,
    select_candidate,
    run_post_check,
    should_execute_real_backend,
    build_run_report,
    build_backend_command_tokens,
    evaluate_execution_readiness,
    load_benchmark_result,
    read_ita_application_summary,
    trace_event,
    write_run_outputs,
)


ROOT = Path(__file__).resolve().parents[2]


class ImagineThenActTest(TestCase):
    def test_parse_candidate_seeds_requires_exact_candidate_count(self) -> None:
        self.assertEqual(parse_candidate_seeds("4,5,6", 3, episode_seed=1200), (4, 5, 6))
        with self.assertRaises(ValueError):
            parse_candidate_seeds("4,5", 3, episode_seed=1200)

    def test_default_candidate_seeds_are_deterministic(self) -> None:
        self.assertEqual(parse_candidate_seeds(None, 3, episode_seed=1200), (1200, 1201, 1202))

    def test_local_dry_run_writes_trace_and_summary(self) -> None:
        with TemporaryDirectory() as tmpdir:
            args = Namespace(
                mode="local-dry-run",
                target="local",
                policy_path="lerobot/smolvla_libero",
                env_type=None,
                task_suite=None,
                task_id=6,
                num_candidates=3,
                candidate_seeds="10,11,12",
                imagination_backend="sim-rollout",
                judge_backend="heuristic",
                post_check_backend="heuristic",
                retry_budget=1,
                output_dir=tmpdir,
                dry_run=False,
                episode_seed=1200,
                chunk_steps=4,
                action_dim=3,
                policy_num_steps=10,
                policy_n_action_steps=15,
                instruction="move the object toward the target",
                selector_strategy="baseline_fallback",
            )
            config = build_run_config(args)
            self.assertEqual(config.policy_num_steps, 10)
            self.assertEqual(config.policy_n_action_steps, 15)
            artifacts = prepare_run_artifacts(config)
            contract = build_execution_contract(config)

            candidates = generate_candidate_chunks(config)
            imagined = imagine_candidates(config, candidates)
            judged = judge_candidates(config, candidates, imagined)
            selected = select_candidate(judged)
            post_check = run_post_check(config, selected, judged)
            execution_readiness, blockers, notes = evaluate_execution_readiness(config)
            trace_events = [
                trace_event("candidate_generation", {"count": len(candidates)}),
                trace_event("selection", {"candidate_id": selected.candidate_id}),
            ]
            report = build_run_report(
                config=config,
                artifacts=artifacts,
                contract=contract,
                selected_candidate=selected,
                post_check=post_check,
                trace_events=trace_events,
                blockers=blockers,
                notes=notes + [execution_readiness],
            )
            write_run_outputs(artifacts, trace_events, report)

            self.assertEqual(report.status, "passed")
            self.assertTrue(config.dry_run)
            self.assertTrue(Path(artifacts.trace_path).exists())
            self.assertTrue(Path(artifacts.summary_path).exists())
            self.assertEqual(len(candidates), 4)
            self.assertEqual(candidates[0].candidate_id, "candidate_00_policy_only")
            self.assertTrue(candidates[0].is_baseline)
            self.assertTrue(report.baseline_candidate_available)
            self.assertTrue(report.baseline_candidate_selected)
            self.assertEqual(report.policy_num_steps, 10)
            self.assertEqual(report.policy_n_action_steps, 15)
            self.assertEqual(report.selector_strategy, "baseline_fallback")
            self.assertTrue(report.selector_fallback_used)
            self.assertFalse(report.method_claim_ready)
            self.assertEqual(report.stage_backends["imagination"], "sim-rollout")
            self.assertFalse(report.benchmark_success_available)
            self.assertEqual(report.benchmark_result.source, "not_run")
            self.assertIsNotNone(report.post_check_rationale)
            trace_lines = Path(artifacts.trace_path).read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(trace_lines), 1)
            first_attempt = json.loads(trace_lines[0])
            self.assertIn("stage", first_attempt)

    def test_runpod_contract_records_remote_command_and_stop_note(self) -> None:
        with TemporaryDirectory() as tmpdir:
            args = Namespace(
                mode="runpod-libero",
                target="runpod",
                policy_path="lerobot/smolvla_libero",
                env_type=None,
                task_suite=None,
                task_id=6,
                num_candidates=4,
                candidate_seeds="1200,1201,1202,1203",
                imagination_backend="sim-rollout",
                judge_backend="vlm-placeholder",
                post_check_backend="oracle-state-placeholder",
                retry_budget=2,
                output_dir=tmpdir,
                dry_run=True,
                episode_seed=1200,
                chunk_steps=10,
                action_dim=7,
                policy_num_steps=10,
                policy_n_action_steps=15,
                instruction="move the target object toward the receptacle",
                selector_strategy="baseline_fallback",
            )

            config = build_run_config(args)
            contract = build_execution_contract(config)

            self.assertTrue(contract.requires_linux)
            self.assertIn("scripts/run_imagine_then_act.py", contract.benchmark_command or "")
            self.assertIn("--mode runpod-libero", contract.benchmark_command or "")
            self.assertIn("run_libero_in_episode_smolvla_instrumented.py", contract.backend_command or "")
            self.assertIn("--policy.path=lerobot/smolvla_libero", contract.backend_command or "")
            self.assertIn("/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results", contract.remote_output_dir or "")
            self.assertEqual(contract.environment_exports["MUJOCO_GL"], "egl")
            joined_notes = " ".join(contract.notes).lower()
            self.assertIn("stopped after fetching results", joined_notes)

    def test_real_backend_command_preserves_canonical_libero_baseline_settings(self) -> None:
        with TemporaryDirectory() as tmpdir:
            args = Namespace(
                mode="runpod-libero",
                target="runpod",
                policy_path="lerobot/smolvla_libero",
                env_type=None,
                task_suite=None,
                task_id=6,
                num_candidates=3,
                candidate_seeds="1200,1201,1202",
                imagination_backend="sim-rollout",
                judge_backend="heuristic",
                post_check_backend="oracle-state-placeholder",
                retry_budget=1,
                output_dir=tmpdir,
                dry_run=False,
                episode_seed=1200,
                chunk_steps=10,
                action_dim=7,
                instruction="move the target object toward the receptacle",
                selector_strategy="baseline_fallback",
            )

            config = build_run_config(args)
            artifacts = prepare_run_artifacts(config)
            command, env = build_real_backend_command(config, artifacts)
            command_text = " ".join(command)

            self.assertIn("run_libero_in_episode_smolvla_instrumented.py", command_text)
            self.assertIn("--trigger-mode semantic_no_progress", command_text)
            self.assertIn("--intervention-mode none", command_text)
            self.assertIn("--policy.path=lerobot/smolvla_libero", command)
            self.assertIn("--env.type=libero", command)
            self.assertIn("--env.task=libero_goal", command)
            self.assertIn("--env.task_ids=[6]", command)
            self.assertIn(
                '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}',
                command,
            )
            self.assertIn("--eval.n_episodes=1", command)
            self.assertIn("--eval.batch_size=1", command)
            self.assertIn("--eval.use_async_envs=false", command)
            self.assertIn("--env.max_parallel_tasks=1", command)
            self.assertIn("--policy.empty_cameras=0", command)
            self.assertIn("--seed=1200", command)
            self.assertNotIn("--policy.num_steps=10", command)
            self.assertNotIn("--policy.n_action_steps=15", command)
            self.assertIn("--ita-enable", command)
            self.assertIn("--ita-candidate-seeds", command)
            self.assertIn("1200,1201,1202", command)
            self.assertIn("--ita-num-candidates", command)
            self.assertIn("3", command)
            self.assertIn("--ita-commit-steps", command)
            self.assertIn("10", command)
            self.assertIn("--ita-selector-strategy", command)
            self.assertIn("baseline_fallback", command)
            tokens = build_backend_command_tokens(
                config=config,
                trace_path="trace.jsonl",
                eval_logs_dir="eval_logs",
                python_bin="python3",
            )
            self.assertNotIn("--policy.num_steps=10", tokens)
            self.assertNotIn("--policy.n_action_steps=15", tokens)
            self.assertEqual(env["MUJOCO_GL"], "egl")
            self.assertIn("HF_HOME", env)
            selected_command, _selected_env = build_real_backend_command(
                config,
                artifacts,
                selected_candidate_id="candidate_02",
            )
            self.assertIn("--ita-selected-candidate-id", selected_command)
            self.assertIn("candidate_02", selected_command)

            default_command, _default_env = build_real_backend_command(config, artifacts)
            self.assertNotIn("--ita-selected-candidate-id", default_command)
            self.assertNotIn("candidate_02", default_command)

    def test_real_backend_command_passes_optional_policy_horizon_overrides(self) -> None:
        with TemporaryDirectory() as tmpdir:
            args = Namespace(
                mode="runpod-libero",
                target="runpod",
                policy_path="lerobot/smolvla_libero",
                env_type=None,
                task_suite=None,
                task_id=6,
                num_candidates=3,
                candidate_seeds="1200,1201,1202",
                imagination_backend="sim-rollout",
                judge_backend="heuristic",
                post_check_backend="oracle-state-placeholder",
                retry_budget=1,
                output_dir=tmpdir,
                dry_run=False,
                episode_seed=1200,
                chunk_steps=10,
                action_dim=7,
                policy_num_steps=10,
                policy_n_action_steps=15,
                instruction="move the target object toward the receptacle",
                selector_strategy="baseline_fallback",
            )

            config = build_run_config(args)
            artifacts = prepare_run_artifacts(config)
            command, _env = build_real_backend_command(config, artifacts)
            tokens = build_backend_command_tokens(
                config=config,
                trace_path="trace.jsonl",
                eval_logs_dir="eval_logs",
                python_bin="python3",
            )

            self.assertEqual(config.policy_num_steps, 10)
            self.assertEqual(config.policy_n_action_steps, 15)
            self.assertIn("--policy.num_steps=10", command)
            self.assertIn("--policy.n_action_steps=15", command)
            self.assertIn("--policy.num_steps=10", tokens)
            self.assertIn("--policy.n_action_steps=15", tokens)
            self.assertEqual(command.count("--policy.num_steps=10"), 1)
            self.assertEqual(command.count("--policy.n_action_steps=15"), 1)
            self.assertEqual(tokens.count("--policy.num_steps=10"), 1)
            self.assertEqual(tokens.count("--policy.n_action_steps=15"), 1)

    def test_policy_only_backend_method_omits_ita_flags_and_allows_broader_task_ids(self) -> None:
        with TemporaryDirectory() as tmpdir:
            args = Namespace(
                mode="runpod-libero",
                target="runpod",
                eval_method="policy_only",
                policy_path="lerobot/smolvla_libero",
                env_type=None,
                task_suite=None,
                task_id=0,
                num_candidates=2,
                candidate_seeds="1201,1202",
                imagination_backend="sim-rollout",
                judge_backend="heuristic",
                post_check_backend="heuristic",
                retry_budget=1,
                output_dir=tmpdir,
                dry_run=False,
                episode_seed=1201,
                chunk_steps=15,
                action_dim=7,
                policy_num_steps=10,
                policy_n_action_steps=15,
                instruction="move the target object toward the receptacle",
                selector_strategy="baseline_fallback",
            )

            config = build_run_config(args)
            artifacts = prepare_run_artifacts(config)
            command, _env = build_real_backend_command(config, artifacts)

            self.assertEqual(config.eval_method, "policy_only")
            self.assertIn("--env.task_ids=[0]", command)
            self.assertIn("--policy.num_steps=10", command)
            self.assertIn("--policy.n_action_steps=15", command)
            self.assertNotIn("--ita-enable", command)
            self.assertNotIn("--ita-candidate-seeds", command)

    def test_non_dry_run_libero_modes_are_gated_to_backend_readiness(self) -> None:
        with TemporaryDirectory() as tmpdir:
            args = Namespace(
                mode="runpod-libero",
                target="runpod",
                policy_path="lerobot/smolvla_libero",
                env_type=None,
                task_suite=None,
                task_id=6,
                num_candidates=3,
                candidate_seeds="1200,1201,1202",
                imagination_backend="sim-rollout",
                judge_backend="heuristic",
                post_check_backend="oracle-state-placeholder",
                retry_budget=1,
                output_dir=tmpdir,
                dry_run=False,
                episode_seed=1200,
                chunk_steps=10,
                action_dim=7,
                instruction="move the target object toward the receptacle",
                selector_strategy="baseline_fallback",
            )

            config = build_run_config(args)
            with patch("physical_ai_agent.imagine_then_act.utils.is_runpod_execution_context", return_value=True):
                readiness, blockers, notes = evaluate_execution_readiness(config)

            self.assertFalse(config.dry_run)
            self.assertEqual(readiness, "benchmark_backend_ready")
            self.assertEqual(blockers, [])
            self.assertTrue(any("instrumented SmolVLA runner" in note for note in notes))
            self.assertTrue(should_execute_real_backend(config, blockers))

    def test_non_dry_run_runpod_requires_runpod_execution_context(self) -> None:
        with TemporaryDirectory() as tmpdir:
            args = Namespace(
                mode="runpod-libero",
                target="runpod",
                policy_path="lerobot/smolvla_libero",
                env_type=None,
                task_suite=None,
                task_id=6,
                num_candidates=3,
                candidate_seeds="1200,1201,1202",
                imagination_backend="sim-rollout",
                judge_backend="heuristic",
                post_check_backend="oracle-state-placeholder",
                retry_budget=1,
                output_dir=tmpdir,
                dry_run=False,
                episode_seed=1200,
                chunk_steps=10,
                action_dim=7,
                instruction="move the target object toward the receptacle",
                selector_strategy="baseline_fallback",
            )

            config = build_run_config(args)
            with patch("physical_ai_agent.imagine_then_act.utils.is_runpod_execution_context", return_value=False):
                readiness, blockers, notes = evaluate_execution_readiness(config)

            self.assertEqual(readiness, "blocked_runpod_runtime")
            self.assertTrue(blockers)
            self.assertFalse(should_execute_real_backend(config, blockers))
            self.assertTrue(any("does not remotely execute" in note for note in notes))

    def test_dry_run_runpod_libero_never_executes_real_backend(self) -> None:
        with TemporaryDirectory() as tmpdir:
            args = Namespace(
                mode="runpod-libero",
                target="runpod",
                policy_path="lerobot/smolvla_libero",
                env_type=None,
                task_suite=None,
                task_id=6,
                num_candidates=3,
                candidate_seeds="1200,1201,1202",
                imagination_backend="sim-rollout",
                judge_backend="heuristic",
                post_check_backend="oracle-state-placeholder",
                retry_budget=1,
                output_dir=tmpdir,
                dry_run=True,
                episode_seed=1200,
                chunk_steps=10,
                action_dim=7,
                instruction="move the target object toward the receptacle",
                selector_strategy="baseline_fallback",
            )

            config = build_run_config(args)
            readiness, blockers, _notes = evaluate_execution_readiness(config)

            self.assertEqual(readiness, "dry_run")
            self.assertEqual(blockers, [])
            self.assertFalse(should_execute_real_backend(config, blockers))

    def test_entrypoint_dry_run_never_calls_real_backend(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "run_imagine_then_act_for_test",
            ROOT / "scripts" / "run_imagine_then_act.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with TemporaryDirectory() as tmpdir:
            with patch.object(module, "execute_real_backend") as execute_backend:
                with redirect_stdout(io.StringIO()):
                    exit_code = module.main(
                        [
                            "--mode",
                            "runpod-libero",
                            "--target",
                            "runpod",
                            "--env-type",
                            "libero",
                            "--task-suite",
                            "libero_goal",
                            "--task-id",
                            "6",
                            "--num-candidates",
                            "3",
                            "--candidate-seeds",
                            "1200,1201,1202",
                            "--output-dir",
                            tmpdir,
                            "--dry-run",
                            "--json",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            execute_backend.assert_not_called()

    def test_entrypoint_smoke_runs_from_cli(self) -> None:
        with TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_imagine_then_act.py"),
                "--mode",
                "smoke",
                "--num-candidates",
                "2",
                "--candidate-seeds",
                "21,22",
                "--output-dir",
                tmpdir,
                "--json",
            ]
            completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)

            self.assertEqual(payload["status"], "passed")
            self.assertTrue(Path(payload["report_path"]).exists())
            self.assertTrue(Path(payload["summary_path"]).exists())
            report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["candidate_count"], 3)
            self.assertEqual(report["benchmark_success_available"], False)
            self.assertEqual(report["benchmark_result"]["source"], "not_run")
            self.assertIn("post_check_rationale", report)
            self.assertTrue(report["baseline_candidate_available"])
            self.assertTrue(report["baseline_candidate_selected"])
            self.assertEqual(report["selector_strategy"], "baseline_fallback")
            self.assertTrue(report["selector_fallback_used"])
            self.assertFalse(report["method_claim_ready"])
            joined_notes = " ".join(report["notes"]).lower()
            self.assertIn("environment", joined_notes)

    def test_entrypoint_runpod_dry_run_accepts_selector_strategy_and_reports_baseline_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_imagine_then_act.py"),
                "--mode",
                "runpod-libero",
                "--target",
                "runpod",
                "--env-type",
                "libero",
                "--task-suite",
                "libero_goal",
                "--task-id",
                "6",
                "--num-candidates",
                "2",
                "--candidate-seeds",
                "1200,1201",
                "--selector-strategy",
                "baseline_fallback",
                "--policy-num-steps",
                "10",
                "--policy-n-action-steps",
                "15",
                "--output-dir",
                tmpdir,
                "--dry-run",
                "--json",
            ]
            completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)
            report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
            trace_records = [
                json.loads(line)
                for line in Path(report["artifacts"]["trace"]).read_text(encoding="utf-8").splitlines()
            ]
            selection_record = next(record for record in trace_records if record["stage"] == "selection")

            self.assertEqual(payload["status"], "passed")
            self.assertEqual(payload["selected_candidate_id"], "candidate_00_policy_only")
            self.assertTrue(payload["baseline_candidate_available"])
            self.assertTrue(payload["baseline_candidate_selected"])
            self.assertEqual(payload["selector_strategy"], "baseline_fallback")
            self.assertTrue(payload["selector_fallback_used"])
            self.assertFalse(payload["method_claim_ready"])
            self.assertEqual(payload["policy_num_steps"], 10)
            self.assertEqual(payload["policy_n_action_steps"], 15)
            self.assertTrue(report["baseline_candidate_available"])
            self.assertTrue(report["baseline_candidate_selected"])
            self.assertEqual(report["policy_num_steps"], 10)
            self.assertEqual(report["policy_n_action_steps"], 15)
            self.assertEqual(report["selector_strategy"], "baseline_fallback")
            self.assertTrue(report["selector_fallback_used"])
            self.assertFalse(report["method_claim_ready"])
            self.assertTrue(selection_record["payload"]["baseline_candidate_available"])
            self.assertTrue(selection_record["payload"]["baseline_candidate_selected"])
            self.assertEqual(selection_record["payload"]["selector_strategy"], "baseline_fallback")
            self.assertTrue(selection_record["payload"]["selector_fallback_used"])
            self.assertFalse(selection_record["payload"]["method_claim_ready"])
            config_record = next(record for record in trace_records if record["stage"] == "config")
            self.assertEqual(config_record["payload"]["policy_num_steps"], 10)
            self.assertEqual(config_record["payload"]["policy_n_action_steps"], 15)
            self.assertIn("--policy.num_steps=10", report["backend_command"])
            self.assertIn("--policy.n_action_steps=15", report["backend_command"])

    def test_entrypoint_accepts_ita_selector_strategy_alias(self) -> None:
        with TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_imagine_then_act.py"),
                "--mode",
                "local-dry-run",
                "--target",
                "local",
                "--num-candidates",
                "1",
                "--candidate-seeds",
                "1200",
                "--ita-selector-strategy",
                "baseline_fallback",
                "--output-dir",
                tmpdir,
                "--json",
            ]
            completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)

            self.assertEqual(payload["status"], "passed")
            self.assertEqual(payload["selector_strategy"], "baseline_fallback")

    def test_trace_parser_sets_selected_candidate_applied_only_from_ita_action_source(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            trace_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event": "rollout_summary",
                                "selected_candidate_applied": True,
                                "ita_selected_candidate_id": "candidate_02",
                                "ita_selected_action_shape": [1, 7],
                                "ita_committed_action_steps": 1,
                                "ita_candidate_generation_source": "smolvla.predict_action_chunk",
                                "baseline_candidate_available": True,
                                "baseline_candidate_selected": True,
                                "selector_strategy": "baseline_fallback",
                                "selector_confidence": 1.0,
                                "selector_fallback_used": True,
                                "method_claim_ready": False,
                            }
                        ),
                        json.dumps(
                            {
                                "step": 0,
                                "ita": {
                                    "action_source": "ita_selected_candidate",
                                    "selected_candidate_applied": True,
                                    "selected_candidate_id": "candidate_02",
                                    "selected_action_shape": [1, 7],
                                    "committed_action_steps_count": 1,
                                    "baseline_candidate_available": True,
                                    "baseline_candidate_selected": True,
                                    "selector_strategy": "baseline_fallback",
                                    "selector_confidence": 1.0,
                                    "selector_fallback_used": True,
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = read_ita_application_summary(trace_path)

            self.assertTrue(summary["selected_candidate_applied"])
            self.assertEqual(summary["selected_candidate_id"], "candidate_02")
            self.assertEqual(summary["selected_action_shape"], [1, 7])
            self.assertEqual(summary["committed_action_steps"], 1)
            self.assertEqual(summary["candidate_generation_source"], "smolvla.predict_action_chunk")
            self.assertTrue(summary["baseline_candidate_available"])
            self.assertTrue(summary["baseline_candidate_selected"])
            self.assertEqual(summary["selector_strategy"], "baseline_fallback")
            self.assertEqual(summary["selector_confidence"], 1.0)
            self.assertTrue(summary["selector_fallback_used"])
            self.assertFalse(summary["method_claim_ready"])

    def test_baseline_trace_parser_keeps_selected_candidate_applied_false(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            trace_path.write_text(
                json.dumps({"event": "rollout_summary", "action_step_count": 1, "success": True}) + "\n"
                + json.dumps({"step": 0, "ita": {"action_source": "normal_policy"}}) + "\n",
                encoding="utf-8",
            )

            summary = read_ita_application_summary(trace_path)

            self.assertFalse(summary["selected_candidate_applied"])
            self.assertIsNone(summary["selected_candidate_id"])
            self.assertEqual(summary["committed_action_steps"], 0)
            self.assertFalse(summary["baseline_candidate_available"])
            self.assertFalse(summary["baseline_candidate_selected"])
            self.assertFalse(summary["method_claim_ready"])

    def test_benchmark_result_reads_selected_candidate_application_from_trace(self) -> None:
        with TemporaryDirectory() as tmpdir:
            args = Namespace(
                mode="runpod-libero",
                target="runpod",
                policy_path="lerobot/smolvla_libero",
                env_type=None,
                task_suite=None,
                task_id=6,
                num_candidates=3,
                candidate_seeds="1200,1201,1202",
                imagination_backend="sim-rollout",
                judge_backend="heuristic",
                post_check_backend="oracle-state-placeholder",
                retry_budget=1,
                output_dir=tmpdir,
                dry_run=False,
                episode_seed=1200,
                chunk_steps=10,
                action_dim=7,
                instruction="move the target object toward the receptacle",
                selector_strategy="baseline_fallback",
            )
            config = build_run_config(args)
            artifacts = prepare_run_artifacts(config)
            Path(artifacts.benchmark_trace_path).write_text(
                json.dumps(
                    {
                        "event": "rollout_summary",
                        "action_step_count": 1,
                        "success": True,
                        "selected_candidate_applied": True,
                        "ita_selected_candidate_id": "candidate_02",
                        "ita_selected_action_shape": [1, 7],
                        "ita_committed_action_steps": 1,
                        "ita_candidate_generation_source": "smolvla.predict_action_chunk",
                        "baseline_candidate_available": True,
                        "baseline_candidate_selected": False,
                        "selector_strategy": "debug_min_action_norm",
                        "selector_confidence": 0.0,
                        "selector_fallback_used": False,
                        "method_claim_ready": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            eval_info_path = Path(artifacts.benchmark_eval_info_path)
            eval_info_path.parent.mkdir(parents=True, exist_ok=True)
            eval_info_path.write_text(
                json.dumps({"overall": {"pc_success": 1.0, "eval_s": 12.5}}),
                encoding="utf-8",
            )

            result = load_benchmark_result(
                config=config,
                artifacts=artifacts,
                command_text="backend command",
                exit_code=0,
            )

            self.assertTrue(result.available)
            self.assertTrue(result.success)
            self.assertTrue(result.selected_candidate_applied)
            self.assertEqual(result.selected_candidate_id, "candidate_02")
            self.assertEqual(result.selected_action_shape, [1, 7])
            self.assertEqual(result.committed_action_steps, 1)
            self.assertEqual(result.candidate_generation_source, "smolvla.predict_action_chunk")
            self.assertTrue(result.baseline_candidate_available)
            self.assertFalse(result.baseline_candidate_selected)
            self.assertEqual(result.selector_strategy, "debug_min_action_norm")
            self.assertEqual(result.selector_confidence, 0.0)
            self.assertFalse(result.selector_fallback_used)
            self.assertFalse(result.method_claim_ready)

    def test_fake_policy_chunk_decision_returns_selected_candidate_action(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "run_libero_in_episode_smolvla_instrumented_for_test",
            ROOT / "scripts" / "run_libero_in_episode_smolvla_instrumented.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        policy = _FakeChunkPolicy()
        decision = module.build_ita_candidate_decision(
            policy=policy,
            observation={"state": _FakeTensor([[0.0]])},
            postprocessor=lambda action: action,
            env_postprocessor=lambda transition: transition,
            action_key="action",
            torch_module=_FakeTorch,
            numpy_module=_FakeNumpy,
            candidate_seeds=[5, 6],
            num_candidates=2,
            commit_steps=1,
            forced_candidate_id="candidate_02",
        )

        self.assertEqual(decision["candidate_generation_source"], "smolvla.predict_action_chunk")
        self.assertEqual(decision["selected_candidate_id"], "candidate_02")
        self.assertEqual(decision["selected_action_shape"], [1, 2])
        self.assertEqual(decision["selected_actions"][0].tolist(), [[3.0, 3.5]])

    def test_baseline_fallback_selects_policy_only_chunk_without_extra_resets(self) -> None:
        module = _load_instrumented_module()
        policy = _FixedChunkPolicy([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

        decision = module.build_ita_candidate_decision(
            policy=policy,
            observation={"state": _FakeTensor([[0.0]])},
            postprocessor=lambda action: action,
            env_postprocessor=lambda transition: transition,
            action_key="action",
            torch_module=_FakeTorch,
            numpy_module=_FakeNumpy,
            candidate_seeds=[1200, 1201],
            num_candidates=2,
            commit_steps=2,
            forced_candidate_id=None,
            selector_strategy="baseline_fallback",
        )

        self.assertEqual(policy.reset_count, 0)
        self.assertEqual(policy.predict_count, 1)
        self.assertEqual(decision["candidate_count"], 1)
        self.assertEqual(decision["selected_candidate_id"], "candidate_00_policy_only")
        self.assertTrue(decision["baseline_candidate_available"])
        self.assertTrue(decision["baseline_candidate_selected"])
        self.assertEqual(decision["selector_strategy"], "baseline_fallback")
        self.assertTrue(decision["selector_fallback_used"])
        self.assertEqual(decision["selected_actions"][0].tolist(), [[1.0, 2.0, 3.0]])
        self.assertEqual(decision["selected_actions"][1].tolist(), [[4.0, 5.0, 6.0]])

    def test_first_nonbaseline_selector_is_removed_from_cli(self) -> None:
        script = ROOT / "scripts" / "run_libero_in_episode_smolvla_instrumented.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--trace-path",
                "/tmp/trace.jsonl",
                "--ita-selector-strategy",
                "first_nonbaseline",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("invalid choice", completed.stderr)

    def test_progress_proxy_selector_requires_nonbaseline_to_clear_baseline_margin(self) -> None:
        module = _load_instrumented_module()
        candidates = [
            {"candidate_id": "candidate_00_policy_only", "progress_proxy_score": 0.5},
            {"candidate_id": "candidate_01", "progress_proxy_score": 0.5},
            {"candidate_id": "candidate_02", "progress_proxy_score": 0.49},
        ]
        selected = module.choose_ita_candidate(candidates, selector_strategy="progress_proxy_or_baseline")
        self.assertEqual(selected["candidate_id"], "candidate_00_policy_only")

        selected, metadata = module.choose_ita_candidate_with_metadata(
            candidates,
            selector_strategy="progress_proxy_or_baseline",
        )
        self.assertEqual(selected["candidate_id"], "candidate_00_policy_only")
        self.assertTrue(metadata["progress_proxy_available"])
        self.assertIn("did not clear", metadata["reason"])

        candidates[2]["progress_proxy_score"] = 0.53
        selected, metadata = module.choose_ita_candidate_with_metadata(
            candidates,
            selector_strategy="progress_proxy_or_baseline",
        )
        self.assertEqual(selected["candidate_id"], "candidate_00_policy_only")
        self.assertEqual(metadata["best_nonbaseline_progress_proxy_score"], 0.53)
        self.assertEqual(metadata["baseline_switch_margin"], 0.05)

        candidates[2]["progress_proxy_score"] = 0.75
        selected, metadata = module.choose_ita_candidate_with_metadata(
            candidates,
            selector_strategy="progress_proxy_or_baseline",
        )
        self.assertEqual(selected["candidate_id"], "candidate_02")
        self.assertEqual(metadata["baseline_progress_proxy_score"], 0.5)
        self.assertEqual(metadata["selected_progress_proxy_score"], 0.75)
        self.assertIn("cleared baseline switch margin", metadata["reason"])

    def test_progress_proxy_selector_falls_back_without_baseline_proxy(self) -> None:
        module = _load_instrumented_module()
        candidates = [
            {"candidate_id": "candidate_00_policy_only"},
            {"candidate_id": "candidate_01", "progress_proxy_score": 0.9},
        ]
        selected = module.choose_ita_candidate(candidates, selector_strategy="progress_proxy_or_baseline")
        self.assertEqual(selected["candidate_id"], "candidate_00_policy_only")

        selected, metadata = module.choose_ita_candidate_with_metadata(
            candidates,
            selector_strategy="progress_proxy_or_baseline",
        )
        self.assertEqual(selected["candidate_id"], "candidate_00_policy_only")
        self.assertTrue(metadata["progress_proxy_available"])
        self.assertIn("baseline proxy is missing", metadata["reason"])

    def test_adaptive_progress_proxy_selector_allows_high_score_medium_margin(self) -> None:
        module = _load_instrumented_module()
        candidates = [
            {"candidate_id": "candidate_00_policy_only", "progress_proxy_score": 0.67},
            {"candidate_id": "candidate_01", "progress_proxy_score": 0.704},
            {"candidate_id": "candidate_02", "progress_proxy_score": 0.69},
        ]

        selected, metadata = module.choose_ita_candidate_with_metadata(
            candidates,
            selector_strategy="progress_proxy_or_baseline",
        )
        self.assertEqual(selected["candidate_id"], "candidate_00_policy_only")
        self.assertIn("did not clear", metadata["reason"])

        selected, metadata = module.choose_ita_candidate_with_metadata(
            candidates,
            selector_strategy="adaptive_progress_proxy_or_baseline",
        )
        self.assertEqual(selected["candidate_id"], "candidate_01")
        self.assertEqual(metadata["adaptive_switch_margin"], 0.03)
        self.assertEqual(metadata["adaptive_absolute_score"], 0.70)
        self.assertIn("adaptive observation progress proxy", metadata["reason"])

    def test_adaptive_progress_proxy_selector_rejects_low_absolute_medium_margin(self) -> None:
        module = _load_instrumented_module()
        candidates = [
            {"candidate_id": "candidate_00_policy_only", "progress_proxy_score": 0.45},
            {"candidate_id": "candidate_01", "progress_proxy_score": 0.49},
        ]

        selected, metadata = module.choose_ita_candidate_with_metadata(
            candidates,
            selector_strategy="adaptive_progress_proxy_or_baseline",
        )
        self.assertEqual(selected["candidate_id"], "candidate_00_policy_only")
        self.assertIn("did not clear", metadata["reason"])

    def test_adaptive_progress_proxy_selector_keeps_baseline_for_left_right_pair_assignment(self) -> None:
        module = _load_instrumented_module()
        candidates = [
            {"candidate_id": "candidate_00_policy_only", "progress_proxy_score": 0.50, "is_baseline": True},
            {
                "candidate_id": "candidate_01",
                "progress_proxy_score": 0.95,
                "candidate_prompt": {
                    "subgoal_text": "Place the white mug on the left plate and the yellow mug on the right plate.",
                    "target_object": "white mug and yellow mug",
                    "target_region_or_point": "left plate and right plate",
                    "stop_condition": "both mugs are on their respective plates",
                },
            },
        ]

        selected, metadata = module.choose_ita_candidate_with_metadata(
            candidates,
            selector_strategy="adaptive_progress_proxy_or_baseline",
        )

        self.assertEqual(selected["candidate_id"], "candidate_00_policy_only")
        self.assertTrue(metadata["compound_objective_guardrail"])
        self.assertIn("compound left/right assignment", metadata["reason"])

    def test_adaptive_progress_proxy_selector_keeps_medium_margin_behavior_for_non_pair_assignment(self) -> None:
        module = _load_instrumented_module()
        candidates = [
            {"candidate_id": "candidate_00_policy_only", "progress_proxy_score": 0.67, "is_baseline": True},
            {
                "candidate_id": "candidate_01",
                "progress_proxy_score": 0.704,
                "candidate_prompt": {
                    "subgoal_text": "Place the white mug on the plate.",
                    "target_object": "white mug",
                    "target_region_or_point": "plate",
                    "stop_condition": "white mug is on the plate",
                },
            },
        ]

        selected, metadata = module.choose_ita_candidate_with_metadata(
            candidates,
            selector_strategy="adaptive_progress_proxy_or_baseline",
        )

        self.assertEqual(selected["candidate_id"], "candidate_01")
        self.assertNotIn("compound_objective_guardrail", metadata)
        self.assertIn("adaptive observation progress proxy", metadata["reason"])

    def test_unknown_selector_strategy_is_rejected_in_internal_helper(self) -> None:
        module = _load_instrumented_module()
        candidates = [{"candidate_id": "candidate_00_policy_only"}]

        with self.assertRaises(ValueError):
            module.choose_ita_candidate(candidates, selector_strategy="first_nonbaseline")

    def test_candidate_decision_attaches_observation_progress_proxy_when_semantic_state_exists(self) -> None:
        module = _load_instrumented_module()
        policy = _SequentialChunkPolicy(
            [
                [[[0.0, -1.0, 0.0]]],
                [[[1.0, 0.0, 0.0]]],
            ]
        )

        with TemporaryDirectory() as tmpdir:
            prompts_path = Path(tmpdir) / "candidate_prompts.json"
            prompts_path.write_text(
                json.dumps(
                    {
                        "candidate_prompts": [
                            {
                                "candidate_prompt": "Put the cream cheese in the bowl using a direct reach.",
                                "strategy_axis": "direct_reach",
                                "target_object": "cream_cheese_1",
                                "target_region_or_point": "akita_black_bowl_1",
                                "stop_condition": "cream cheese is in the bowl",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            decision = module.build_ita_candidate_decision(
                policy=policy,
                observation={"state": _FakeTensor([[0.0]]), "task": ["put the cream cheese in the bowl"]},
                postprocessor=lambda action: action,
                env_postprocessor=lambda transition: transition,
                action_key="action",
                torch_module=_FakeTorch,
                numpy_module=_FakeNumpy,
                candidate_seeds=[1201],
                num_candidates=1,
                commit_steps=1,
                candidate_prompts_json=prompts_path,
                selector_strategy="progress_proxy_or_baseline",
                semantic_state={
                    "eef_pos": [0.0, 0.0, 0.0],
                    "target_pos": [1.0, 0.0, 0.0],
                    "eef_to_target_dist": 1.0,
                    "target_object_key": "cream_cheese_1_pos",
                },
            )

        self.assertEqual(decision["selected_candidate_id"], "candidate_01")
        self.assertFalse(decision["baseline_candidate_selected"])
        self.assertTrue(decision["progress_proxy_available"])
        self.assertIn("cleared baseline switch margin", decision["selector_reason"])
        self.assertLess(decision["baseline_progress_proxy_score"], 0.5)
        self.assertEqual(decision["selected_progress_proxy_score"], 1.0)
        self.assertEqual(
            [row["candidate_id"] for row in decision["candidate_score_table"]],
            ["candidate_00_policy_only", "candidate_01"],
        )

    def test_chunk_progress_proxy_prefers_consistent_progress_over_one_step_lunge(self) -> None:
        module = _load_instrumented_module()
        semantic_state = {
            "eef_pos": [0.0, 0.0, 0.0],
            "target_pos": [1.0, 0.0, 0.0],
            "eef_to_target_dist": 1.0,
            "target_object_key": "target_pos",
        }

        lunge_then_reverse = module.compute_observation_progress_proxy(
            [
                [[1.0, 0.0, 0.0]],
                [[-1.0, 0.0, 0.0]],
                [[-1.0, 0.0, 0.0]],
            ],
            semantic_state,
        )
        steady_progress = module.compute_observation_progress_proxy(
            [
                [[0.2, 0.0, 0.0]],
                [[0.2, 0.0, 0.0]],
                [[0.2, 0.0, 0.0]],
            ],
            semantic_state,
        )

        self.assertIsNotNone(lunge_then_reverse)
        self.assertIsNotNone(steady_progress)
        self.assertGreater(steady_progress["score"], lunge_then_reverse["score"])
        self.assertGreater(lunge_then_reverse["reverse_penalty"], 0.0)

    def test_chunk_extraction_applies_postprocessors_for_batch1_chunk2_actiondim7(self) -> None:
        module = _load_instrumented_module()
        raw_chunk = [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
            ]
        ]
        policy = _FixedChunkPolicy(raw_chunk)

        decision = module.build_ita_candidate_decision(
            policy=policy,
            observation={"state": _FakeTensor([[0.0]])},
            postprocessor=lambda action: action.add_scalar(10.0),
            env_postprocessor=lambda transition: {"action": transition["action"].add_scalar(100.0)},
            action_key="action",
            torch_module=_FakeTorch,
            numpy_module=_FakeNumpy,
            candidate_seeds=[7],
            num_candidates=1,
            commit_steps=2,
            forced_candidate_id="candidate_01",
            selector_strategy="debug_min_action_norm",
        )

        self.assertEqual(policy.reset_count, 1)
        self.assertEqual(decision["selected_candidate_id"], "candidate_01")
        self.assertEqual(decision["selected_action_shape"], [1, 7])
        self.assertEqual(len(decision["selected_actions"]), 2)
        self.assertEqual(decision["selected_actions"][0].tolist(), [[111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0]])
        self.assertEqual(decision["selected_actions"][1].tolist(), [[121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0]])

    def test_fake_env_step_receives_selected_baseline_action(self) -> None:
        module = _load_instrumented_module()
        policy = _FixedChunkPolicy([[[0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -1.75], [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]]])
        env = _FakeEnv()

        decision = module.build_ita_candidate_decision(
            policy=policy,
            observation={"state": _FakeTensor([[0.0]])},
            postprocessor=lambda action: action,
            env_postprocessor=lambda transition: transition,
            action_key="action",
            torch_module=_FakeTorch,
            numpy_module=_FakeNumpy,
            candidate_seeds=[1, 2],
            num_candidates=2,
            commit_steps=2,
            selector_strategy="baseline_fallback",
        )
        env.step(decision["selected_actions"][0].to("cpu").numpy())
        env.step(decision["selected_actions"][1].to("cpu").numpy())

        self.assertEqual(decision["selected_candidate_id"], "candidate_00_policy_only")
        self.assertEqual(env.actions[0], [[0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -1.75]])
        self.assertEqual(env.actions[1], [[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]])

    def test_rollout_baseline_fallback_skips_post_queue_policy_reset_and_resumes_policy(self) -> None:
        module = _load_instrumented_module()
        fake_torch = _RolloutFakeTorch()
        fake_modules = _build_rollout_fake_modules(fake_torch)
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            policy = _RolloutFakePolicy(fake_torch.nn.Module)
            env = _RolloutFakeEnv()
            rollout = module.build_instrumented_rollout(
                trace_path=trace_path,
                intervention_step=99,
                trigger_mode="fixed_step",
                action_norm_threshold=999.0,
                intervention_mode="none",
                intervention_scale=1.0,
                action_clamp_norm=1.0,
                smooth_alpha=0.5,
                target_object_key="target",
                receptacle_object_key="receptacle",
                semantic_min_step=99,
                semantic_window=1,
                semantic_progress_threshold=0.0,
                semantic_distance_threshold=0.0,
                semantic_reach_gain=1.0,
                semantic_push_gain=1.0,
                semantic_contact_threshold=0.1,
                semantic_place_z_command=0.0,
                semantic_gripper_command=0.0,
                ita_enable=True,
                ita_candidate_seeds=[1200, 1201],
                ita_num_candidates=2,
                ita_commit_steps=2,
                ita_selected_candidate_id=None,
                ita_selector_strategy="baseline_fallback",
            )

            with patch.dict(sys.modules, fake_modules):
                rollout(
                    env=env,
                    policy=policy,
                    env_preprocessor=lambda observation: observation,
                    env_postprocessor=lambda transition: transition,
                    preprocessor=lambda observation: observation,
                    postprocessor=lambda action: action,
                    seeds=[1200],
                    return_observations=False,
                    render_callback=None,
                )

            self.assertEqual(policy.reset_count, 1)
            self.assertEqual(policy.predict_count, 1)
            self.assertEqual(policy.select_count, 1)
            self.assertEqual(env.actions[0], [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
            self.assertEqual(env.actions[1], [[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]])
            self.assertEqual(env.actions[2], [[91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0]])

            records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            summary = records[0]
            steps = records[1:]
            self.assertTrue(summary["selected_candidate_applied"])
            self.assertTrue(summary["baseline_candidate_available"])
            self.assertTrue(summary["baseline_candidate_selected"])
            self.assertEqual(summary["selector_strategy"], "baseline_fallback")
            self.assertFalse(summary["method_claim_ready"])
            self.assertEqual(steps[0]["ita"]["action_source"], "ita_selected_candidate")
            self.assertEqual(steps[1]["ita"]["action_source"], "ita_selected_candidate")
            self.assertEqual(steps[2]["ita"]["action_source"], "normal_policy")


def _load_instrumented_module():
    spec = importlib.util.spec_from_file_location(
        "run_libero_in_episode_smolvla_instrumented_for_test",
        ROOT / "scripts" / "run_libero_in_episode_smolvla_instrumented.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTensor:
    def __init__(self, data: object) -> None:
        self.data = data
        self.shape = self._shape(data)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, item: object) -> "_FakeTensor":
        if isinstance(item, tuple) and len(item) == 3:
            _batch, step, _dim = item
            return _FakeTensor([self.data[0][step]])
        return _FakeTensor(self.data[item])

    def detach(self) -> "_FakeTensor":
        return self

    def clone(self) -> "_FakeTensor":
        return _FakeTensor(json.loads(json.dumps(self.data)))

    def add_scalar(self, value: float) -> "_FakeTensor":
        return _FakeTensor(self._map_scalar(self.data, value))

    def to(self, _device: str) -> "_FakeTensor":
        return self

    def numpy(self) -> "_FakeTensor":
        return self

    def __array__(self, dtype: object = None) -> object:
        import numpy as np

        return np.array(self.data, dtype=dtype)

    def tolist(self) -> object:
        return self.data

    @staticmethod
    def _shape(value: object) -> tuple[int, ...]:
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                if value[0] and isinstance(value[0][0], list):
                    return (len(value), len(value[0]), len(value[0][0]))
                return (len(value), len(value[0]))
            return (len(value),)
        return ()

    @classmethod
    def _map_scalar(cls, value: object, offset: float) -> object:
        if isinstance(value, list):
            return [cls._map_scalar(item, offset) for item in value]
        return float(value) + offset


class _FakeChunkPolicy:
    name = "smolvla"

    def reset(self) -> None:
        return None

    def predict_action_chunk(self, _observation: object) -> _FakeTensor:
        import random

        seed_value = float(int(random.random() * 1000) % 10)
        return _FakeTensor([[[seed_value, seed_value + 0.5], [seed_value + 1.0, seed_value + 1.5]]])


class _FixedChunkPolicy:
    name = "smolvla"

    def __init__(self, chunk: list[list[list[float]]]) -> None:
        self.chunk = chunk
        self.reset_count = 0
        self.predict_count = 0
        self.observed_tasks: list[object] = []

    def reset(self) -> None:
        self.reset_count += 1

    def predict_action_chunk(self, _observation: object) -> _FakeTensor:
        self.predict_count += 1
        if isinstance(_observation, dict):
            self.observed_tasks.append(json.loads(json.dumps(_observation.get("task"))))
        return _FakeTensor(json.loads(json.dumps(self.chunk)))


class _SequentialChunkPolicy(_FixedChunkPolicy):
    def __init__(self, chunks: list[list[list[list[float]]]]) -> None:
        super().__init__(chunks[0])
        self.chunks = chunks

    def predict_action_chunk(self, _observation: object) -> _FakeTensor:
        self.predict_count += 1
        if isinstance(_observation, dict):
            self.observed_tasks.append(json.loads(json.dumps(_observation.get("task"))))
        index = min(self.predict_count - 1, len(self.chunks) - 1)
        return _FakeTensor(json.loads(json.dumps(self.chunks[index])))


class _FakeEnv:
    def __init__(self) -> None:
        self.actions: list[object] = []

    def step(self, action: object) -> tuple[None, list[float], list[bool], list[bool], dict[str, list[bool]]]:
        self.actions.append(action.tolist() if hasattr(action, "tolist") else action)
        return None, [0.0], [False], [False], {"is_success": [False]}


class _RolloutFakeTorch(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.nn = types.SimpleNamespace(Module=type("Module", (), {}))
        self.linalg = types.SimpleNamespace(vector_norm=self._vector_norm)

    @staticmethod
    def manual_seed(_seed: int) -> None:
        return None

    class inference_mode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> bool:
            return False

    @staticmethod
    def from_numpy(value: object) -> object:
        return value

    @staticmethod
    def tensor(value: object) -> _FakeTensor:
        return _FakeTensor(value)

    @staticmethod
    def stack(values: list[object], dim: int = 0) -> _FakeTensor:  # noqa: ARG004
        return _FakeTensor([value.tolist() if hasattr(value, "tolist") else value for value in values])

    @staticmethod
    def _vector_norm(value: object) -> "_FakeScalar":
        values = _flatten_fake_numeric(value.tolist() if hasattr(value, "tolist") else value)
        return _FakeScalar(sum(item * item for item in values) ** 0.5)


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _FakeMeanable:
    def __init__(self, value: float) -> None:
        self.value = value

    def numpy(self) -> "_FakeMeanable":
        return self

    def mean(self) -> _FakeScalar:
        return _FakeScalar(self.value)


class _FakeProgress:
    def __init__(self, _max_steps: int, **_kwargs: object) -> None:
        return None

    def set_postfix(self, _payload: dict[str, object]) -> None:
        return None

    def update(self) -> None:
        return None


class _RolloutFakePolicy:
    name = "smolvla"

    def __init__(self, module_base: type) -> None:
        self.__class__ = type("_RolloutFakePolicyInstance", (self.__class__, module_base), {})
        self.reset_count = 0
        self.predict_count = 0
        self.select_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict_action_chunk(self, _observation: object) -> _FakeTensor:
        self.predict_count += 1
        return _FakeTensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                ]
            ]
        )

    def select_action(self, _observation: object) -> _FakeTensor:
        self.select_count += 1
        return _FakeTensor([[91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0]])

    def use_original_modules(self) -> None:
        return None


class _RolloutFakeEnv:
    num_envs = 1

    def __init__(self) -> None:
        self.actions: list[object] = []
        self.step_count = 0

    def reset(self, seed: object = None) -> tuple[dict[str, _FakeTensor], dict[str, object]]:  # noqa: ARG002
        return {"state": _FakeTensor([[0.0]])}, {}

    def call(self, name: str) -> list[object]:
        if name == "_max_episode_steps":
            return [3]
        if name == "task_description":
            return ["fake task"]
        raise AttributeError(name)

    def step(self, action: object) -> tuple[dict[str, _FakeTensor], object, object, object, dict[str, object]]:
        import numpy as np

        self.actions.append(action.tolist() if hasattr(action, "tolist") else action)
        self.step_count += 1
        terminated = np.array([self.step_count >= 3])
        truncated = np.array([False])
        return {"state": _FakeTensor([[float(self.step_count)]])}, np.array([0.0]), terminated, truncated, {
            "is_success": np.array([False])
        }


def _build_rollout_fake_modules(fake_torch: _RolloutFakeTorch) -> dict[str, object]:
    fake_einops = types.ModuleType("einops")
    fake_einops.reduce = lambda *_args, **_kwargs: _FakeMeanable(0.0)

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.trange = lambda max_steps, **kwargs: _FakeProgress(max_steps, **kwargs)

    fake_lerobot_envs = types.ModuleType("lerobot.envs")
    fake_lerobot_envs.check_env_attributes_and_types = lambda _env: None
    fake_lerobot_envs.preprocess_observation = lambda observation: observation

    fake_constants = types.ModuleType("lerobot.utils.constants")
    fake_constants.ACTION = "action"
    fake_constants.OBS_STR = "observation"

    fake_utils = types.ModuleType("lerobot.utils.utils")
    fake_utils.inside_slurm = lambda: True

    return {
        "torch": fake_torch,
        "einops": fake_einops,
        "tqdm": fake_tqdm,
        "lerobot": types.ModuleType("lerobot"),
        "lerobot.envs": fake_lerobot_envs,
        "lerobot.utils": types.ModuleType("lerobot.utils"),
        "lerobot.utils.constants": fake_constants,
        "lerobot.utils.utils": fake_utils,
    }


def _flatten_fake_numeric(value: object) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            result.extend(_flatten_fake_numeric(item))
        return result
    return [float(value)]


class _FakeTorch:
    @staticmethod
    def manual_seed(_seed: int) -> None:
        return None

    class inference_mode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> bool:
            return False


class _FakeNumpy:
    class random:
        @staticmethod
        def seed(_seed: int) -> None:
            return None
