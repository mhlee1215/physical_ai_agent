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
            self.assertIn("--ita-enable", command)
            self.assertIn("--ita-candidate-seeds", command)
            self.assertIn("1200,1201,1202", command)
            self.assertIn("--ita-num-candidates", command)
            self.assertIn("3", command)
            self.assertIn("--ita-commit-steps", command)
            self.assertIn("10", command)
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


class _FakeTensor:
    def __init__(self, data: object) -> None:
        self.data = data
        self.shape = self._shape(data)

    def __getitem__(self, item: object) -> "_FakeTensor":
        if isinstance(item, tuple) and len(item) == 3:
            _batch, step, _dim = item
            return _FakeTensor([self.data[0][step]])
        return _FakeTensor(self.data[item])

    def detach(self) -> "_FakeTensor":
        return self

    def clone(self) -> "_FakeTensor":
        return _FakeTensor(json.loads(json.dumps(self.data)))

    def to(self, _device: str) -> "_FakeTensor":
        return self

    def numpy(self) -> "_FakeTensor":
        return self

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


class _FakeChunkPolicy:
    name = "smolvla"

    def reset(self) -> None:
        return None

    def predict_action_chunk(self, _observation: object) -> _FakeTensor:
        import random

        seed_value = float(int(random.random() * 1000) % 10)
        return _FakeTensor([[[seed_value, seed_value + 0.5], [seed_value + 1.0, seed_value + 1.5]]])


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
