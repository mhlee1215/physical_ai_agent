import json
import importlib.util
import io
from argparse import Namespace
from contextlib import redirect_stdout
import subprocess
import sys
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
    evaluate_execution_readiness,
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
                instruction="move the object toward the target",
            )
            config = build_run_config(args)
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
            self.assertEqual(len(candidates), 3)
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
                instruction="move the target object toward the receptacle",
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
            self.assertEqual(env["MUJOCO_GL"], "egl")
            self.assertIn("HF_HOME", env)

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
            )

            config = build_run_config(args)
            with patch("physical_ai_agent.imagine_then_act.utils.is_runpod_execution_context", return_value=True):
                readiness, blockers, notes = evaluate_execution_readiness(config)

            self.assertFalse(config.dry_run)
            self.assertEqual(readiness, "benchmark_backend_ready")
            self.assertEqual(blockers, [])
            self.assertTrue(any("instrumented SmolVLA baseline runner" in note for note in notes))
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
            self.assertEqual(report["candidate_count"], 2)
            self.assertEqual(report["benchmark_success_available"], False)
            self.assertEqual(report["benchmark_result"]["source"], "not_run")
            self.assertIn("post_check_rationale", report)
            joined_notes = " ".join(report["notes"]).lower()
            self.assertIn("environment", joined_notes)
