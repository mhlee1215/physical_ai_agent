import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from physical_ai_agent.imagine_then_act.risk_probes import (
    ActionChunkCandidate,
    BLOCKED,
    FAIL,
    PASS,
    WARN,
    RiskProbeConfig,
    apply_candidate_to_env,
    apply_torch_transformers_import_compatibility_patch,
    capture_sim_state,
    compute_clone_fidelity_metrics,
    compute_actual_oracle_or_proxy_metrics,
    compute_diversity_metrics,
    compute_oracle_upper_bound_metrics,
    find_sim_clone_handle,
    generate_mock_candidates,
    inspect_actual_env,
    restore_sim_state,
    run_direct_env_snapshot_replay,
    run_risk_probes,
    prepare_lerobot_policy_observation,
    sample_policy_action_candidates,
    simulate_mock_env,
)
from physical_ai_agent.imagine_then_act.direct_libero_imagination import (
    DirectLiberoProbeConfig,
    build_risk_probe_config,
)


ROOT = Path(__file__).resolve().parents[2]


class RiskProbeTest(TestCase):
    def make_config(self, output_dir: str) -> RiskProbeConfig:
        return RiskProbeConfig(
            preset="local-dry-run",
            backend="mock",
            suite="libero_goal",
            task_ids=(6,),
            seed=1201,
            num_candidates=5,
            chunk_steps=15,
            action_dim=7,
            output_dir=output_dir,
        )

    def test_diversity_metrics_detect_nontrivial_candidate_spread(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir)
            candidates = generate_mock_candidates(config)
            metrics = compute_diversity_metrics(config, candidates)

            self.assertEqual(metrics.verdict, PASS)
            self.assertGreater(metrics.min_pairwise_l2, 0.05)
            self.assertGreater(metrics.endpoint_spread_l2, 0.0)
            self.assertGreater(metrics.gripper_command_variance, 0.0)

    def test_diversity_metrics_fail_on_identical_candidates(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir)
            identical_chunk = [[0.01 for _dim in range(config.action_dim)] for _step in range(config.chunk_steps)]
            candidates = [
                ActionChunkCandidate(
                    candidate_id="candidate_00_policy_only",
                    source="test_identical",
                    action_chunk=identical_chunk,
                    privileged_success_proxy=0.1,
                    is_policy_only=True,
                ),
                ActionChunkCandidate(
                    candidate_id="candidate_01",
                    source="test_identical",
                    action_chunk=[row[:] for row in identical_chunk],
                    privileged_success_proxy=0.1,
                ),
            ]

            metrics = compute_diversity_metrics(config, candidates)

            self.assertEqual(metrics.verdict, FAIL)
            self.assertEqual(metrics.min_pairwise_l2, 0.0)
            self.assertIn("identical", metrics.rationale)

    def test_clone_fidelity_mock_path_matches_committed_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir)
            candidates = generate_mock_candidates(config)
            metrics = compute_clone_fidelity_metrics(config, candidates[-1])

            self.assertEqual(metrics.verdict, PASS)
            self.assertEqual(metrics.state_l2, 0.0)
            self.assertEqual(metrics.image_mse, 0.0)
            self.assertFalse(metrics.deterministic_replay_mismatch)

    def test_oracle_selector_selects_better_mock_candidate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir)
            candidates = generate_mock_candidates(config)
            outcomes = {candidate.candidate_id: simulate_mock_env(candidate.action_chunk) for candidate in candidates}
            metrics = compute_oracle_upper_bound_metrics(candidates, outcomes)

            self.assertEqual(metrics.verdict, PASS)
            self.assertTrue(metrics.oracle_beats_policy)
            self.assertTrue(metrics.oracle_beats_random)
            self.assertEqual(metrics.selected_candidate_id, candidates[-1].candidate_id)

    def test_html_report_contains_risk_evidence_and_image_links(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir)
            report = run_risk_probes(config)
            html_path = Path(report.artifacts["html_report"])
            html = html_path.read_text(encoding="utf-8")

            self.assertEqual(report.status, PASS)
            self.assertIn("Risk 1 Candidate Diversity", html)
            self.assertIn("Risk 2 Clone Fidelity", html)
            self.assertIn("Risk 5 Oracle Selector Upper-Bound", html)
            self.assertIn("candidate_action_heatmap.svg", html)
            self.assertIn("candidate_action_chunks.json", html)
            self.assertIn("clone_image_diff.svg", html)
            self.assertIn("oracle_scores.svg", html)
            for artifact_path in report.artifacts.values():
                self.assertTrue(Path(artifact_path).exists())
            chunk_payload = json.loads(Path(report.artifacts["candidate_action_chunks"]).read_text(encoding="utf-8"))
            self.assertEqual(chunk_payload["risk"], "risk_1_candidate_diversity")
            self.assertEqual(len(chunk_payload["candidates"]), config.num_candidates)

    def test_cli_dry_run_generates_artifact_bundle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "run_imagine_then_act_risk_probes.py"),
                "--preset",
                "local-dry-run",
                "--output-dir",
                tmpdir,
                "--json",
            ]
            completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)

            self.assertEqual(payload["status"], PASS)
            self.assertTrue(Path(payload["summary_path"]).exists())
            self.assertTrue(Path(payload["events_path"]).exists())
            self.assertTrue(Path(payload["html_report"]).exists())
            progress_path = Path(tmpdir) / "risk_probe_progress.jsonl"
            self.assertTrue(progress_path.exists())
            progress = progress_path.read_text(encoding="utf-8")
            self.assertIn('"phase": "start"', progress)
            self.assertIn('"phase": "summary_written"', progress)

    def test_cli_task_parser_supports_ranges(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "run_imagine_then_act_risk_probes_for_test",
            ROOT / "scripts" / "run_imagine_then_act_risk_probes.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        self.assertEqual(module.parse_task_ids("0-2,2,4", "local-dry-run"), (0, 1, 2, 4))
        parsed = module.build_parser().parse_args(["--preset", "runpod-libero-double-sim-smoke"])
        config = module.build_config(parsed)
        self.assertTrue(config.direct_libero_double_sim)
        self.assertEqual(config.actual_timeout_sec, 1800)
        parsed_timeout = module.build_parser().parse_args(
            ["--preset", "runpod-libero-double-sim-smoke", "--actual-timeout-sec", "12"]
        )
        self.assertEqual(module.build_config(parsed_timeout).actual_timeout_sec, 12)
        parsed_renderer = module.build_parser().parse_args(
            ["--preset", "runpod-libero-double-sim-smoke", "--renderer-backend", "osmesa"]
        )
        self.assertEqual(module.build_config(parsed_renderer).renderer_backend, "osmesa")

    def test_fake_actual_adapter_helpers_record_proxy_only_evidence(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir)
            candidates = generate_mock_candidates(config)
            selected = [candidates[0], candidates[1], candidates[-1]]
            env = _FakeActualEnv()
            introspection = inspect_actual_env(env)
            outcomes = {}
            for candidate in selected:
                evidence = apply_candidate_to_env(env, candidate, [config.seed], config.actual_max_steps, _FakeNumpy)
                outcomes[candidate.candidate_id] = {
                    "success_proxy": evidence["success_proxy"],
                    "state": evidence["state_vector"][:3],
                    "image": [[0]],
                }

            metrics = compute_actual_oracle_or_proxy_metrics(selected, outcomes, introspection)

            self.assertFalse(introspection["exact_state_clone_available"])
            self.assertFalse(introspection["privileged_state_available"])
            self.assertEqual(metrics.verdict, WARN)
            self.assertIn("proxy_only", metrics.rationale)

    def test_deep_internal_sim_traversal_finds_clone_restore_handle(self) -> None:
        env = _FakeVectorRobosuiteEnv()
        introspection = inspect_actual_env(env)
        handle = find_sim_clone_handle(env)

        self.assertTrue(introspection["exact_state_clone_available"])
        self.assertTrue(introspection["privileged_state_available"])
        self.assertIsNotNone(handle)
        self.assertIn("sim", handle["path"])

        snapshot = capture_sim_state(handle)
        env.step([[0.25, 0.5, 0.75]])
        self.assertNotEqual(env.sim.state, [0.0, 0.0, 0.0])
        forward_calls_before_restore = env.sim.forward_calls
        restored = restore_sim_state(handle, snapshot)

        self.assertTrue(snapshot["captured"])
        self.assertTrue(restored["restored"])
        self.assertEqual(env.sim.state, [0.0, 0.0, 0.0])
        self.assertEqual(env.sim.forward_calls, forward_calls_before_restore + 1)

    def test_sim_handle_prefers_nested_data_sim_over_broken_root_sim(self) -> None:
        env = _FakeBrokenRootGoodNestedEnv()
        handle = find_sim_clone_handle(env)
        snapshot = capture_sim_state(handle)
        env.env.sim.state = [0.4, 0.5, 0.6]
        restored = restore_sim_state(handle, snapshot)

        self.assertIsNotNone(handle)
        self.assertEqual(handle["path"], "root.env.sim")
        self.assertTrue(snapshot["captured"])
        self.assertTrue(restored["restored"])
        self.assertEqual(env.env.sim.state, [0.0, 0.0, 0.0])

    def test_qpos_qvel_only_sim_handle_can_restore_state(self) -> None:
        env = _FakeQposOnlyEnv()
        handle = find_sim_clone_handle(env)
        snapshot = capture_sim_state(handle)
        env.sim.data.qpos[:] = [0.3, 0.4]
        env.sim.data.qvel[:] = [0.5, 0.6]
        restored = restore_sim_state(handle, snapshot)

        self.assertEqual(handle["strategy"], "qpos_qvel")
        self.assertTrue(snapshot["captured"])
        self.assertTrue(restored["restored"])
        self.assertEqual(env.sim.data.qpos, [0.0, 0.0])
        self.assertEqual(env.sim.data.qvel, [0.0, 0.0])

    def test_clone_rollout_uses_restored_internal_sim_state(self) -> None:
        env = _FakeVectorRobosuiteEnv()
        candidate = ActionChunkCandidate(
            candidate_id="candidate_clone",
            source="fake_robosuite",
            action_chunk=[[0.2, 0.3, 0.4], [0.1, 0.1, 0.1]],
            privileged_success_proxy=0.8,
        )
        handle = find_sim_clone_handle(env)
        start_observation, start_info = env.reset(seed=[1201])
        snapshot = capture_sim_state(handle)

        committed = apply_candidate_to_env(
            env,
            candidate,
            [1201],
            2,
            _FakeNumpy,
            reset_before=False,
            initial_observation=start_observation,
            initial_info=start_info,
        )
        restored = restore_sim_state(handle, snapshot)
        clone = apply_candidate_to_env(
            env,
            candidate,
            [1201],
            2,
            _FakeNumpy,
            reset_before=False,
            initial_observation=start_observation,
            initial_info=start_info,
        )

        self.assertTrue(restored["restored"])
        self.assertEqual(committed["state_vector"], clone["state_vector"])
        self.assertEqual(committed["image_vector"], clone["image_vector"])
        self.assertIn("sim_data_pose", committed["state_source"])
        self.assertNotEqual(committed["privileged_success_proxy_source"], "unavailable")

    def test_privileged_oracle_metrics_are_not_proxy_only_when_internal_state_exists(self) -> None:
        candidates = [
            ActionChunkCandidate("candidate_00_policy_only", "policy", [[0.0]], 0.0, is_policy_only=True),
            ActionChunkCandidate("candidate_01_random", "random", [[0.0]], 0.0),
            ActionChunkCandidate("candidate_02_oracle", "oracle_fixture", [[0.0]], 0.0),
        ]
        outcomes = {
            "candidate_00_policy_only": {"success_proxy": 0.1},
            "candidate_01_random": {"success_proxy": 0.2},
            "candidate_02_oracle": {"success_proxy": 0.9},
        }
        introspection = inspect_actual_env(_FakeVectorRobosuiteEnv())

        metrics = compute_actual_oracle_or_proxy_metrics(candidates, outcomes, introspection)

        self.assertEqual(metrics.verdict, PASS)
        self.assertIn("oracle_available", metrics.rationale)
        self.assertTrue(metrics.oracle_beats_policy)
        self.assertTrue(metrics.oracle_beats_random)

    def test_direct_libero_double_sim_fake_env_replays_from_snapshot(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = RiskProbeConfig(
                preset="runpod-libero-double-sim-smoke",
                backend="libero-contract",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=3,
                chunk_steps=2,
                action_dim=3,
                output_dir=tmpdir,
                direct_libero_double_sim=True,
            )
            candidates = generate_mock_candidates(config)

            evidence = run_direct_env_snapshot_replay(
                env=_FakeVectorRobosuiteEnv(),
                init_state=[0.0, 0.0, 0.0],
                candidates=candidates,
                output_dir=Path(tmpdir),
                camera_name="agentview",
                max_steps=2,
                np_module=_FakeNumpy,
            )

            self.assertEqual(evidence["clone_fidelity"]["verdict"], PASS)
            self.assertTrue(evidence["clone_restore_evidence"]["snapshot_restored"])
            self.assertEqual(evidence["sync_scope"], "episode_start_init_state_only")
            self.assertEqual(evidence["mid_episode_sync"], "future_work")
            for artifact_path in evidence["image_artifacts"].values():
                self.assertTrue(Path(artifact_path).exists())
                self.assertEqual(Path(artifact_path).suffix, ".ppm")
                self.assertTrue(Path(artifact_path).read_text(encoding="ascii").startswith("P3\n"))

    def test_direct_replay_finds_sim_handle_after_reset_replaces_root_sim(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = RiskProbeConfig(
                preset="runpod-libero-double-sim-smoke",
                backend="libero-contract",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=3,
                chunk_steps=2,
                action_dim=3,
                output_dir=tmpdir,
                direct_libero_double_sim=True,
            )
            candidates = generate_mock_candidates(config)

            evidence = run_direct_env_snapshot_replay(
                env=_FakeResetReplacesSimEnv(),
                init_state=[0.0, 0.0, 0.0],
                candidates=candidates,
                output_dir=Path(tmpdir),
                camera_name="agentview",
                max_steps=2,
                np_module=_FakeNumpy,
            )

            self.assertEqual(evidence["clone_fidelity"]["verdict"], PASS)
            self.assertTrue(evidence["clone_restore_evidence"]["snapshot_restored"])
            self.assertEqual(evidence["clone_restore_evidence"]["snapshot_source"], "root.sim")
            self.assertEqual(evidence["clone_restore_evidence"]["selected_handle"]["type"].split(".")[-1], "_FakeMuJoCoSim")

    def test_direct_replay_records_forward_failure_without_failing_restore(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = RiskProbeConfig(
                preset="runpod-libero-double-sim-smoke",
                backend="libero-contract",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=3,
                chunk_steps=2,
                action_dim=3,
                output_dir=tmpdir,
                direct_libero_double_sim=True,
            )
            candidates = generate_mock_candidates(config)

            evidence = run_direct_env_snapshot_replay(
                env=_FakeForwardRaisesEnv(),
                init_state=[0.0, 0.0, 0.0],
                candidates=candidates,
                output_dir=Path(tmpdir),
                camera_name="agentview",
                max_steps=2,
                np_module=_FakeNumpy,
            )

            self.assertEqual(evidence["clone_fidelity"]["verdict"], PASS)
            self.assertTrue(evidence["clone_restore_evidence"]["snapshot_restored"])
            self.assertTrue(evidence["clone_restore_evidence"]["forward_called"])
            self.assertFalse(evidence["clone_restore_evidence"]["forward_succeeded"])
            self.assertIn("AttributeError", evidence["clone_restore_evidence"]["forward_error"])

    def test_libero_contract_writes_actionable_import_guard_blocker_without_dependencies(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = RiskProbeConfig(
                preset="runpod-libero-smoke",
                backend="libero-contract",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=3,
                chunk_steps=3,
                action_dim=7,
                output_dir=tmpdir,
            )
            report = run_risk_probes(config)

            self.assertIn(report.status, {"BLOCKED", "WARN", "PASS"})
            self.assertTrue(Path(report.artifacts["libero_adapter_evidence"]).exists())
            if report.status == "BLOCKED":
                self.assertTrue(any("LIBERO actual adapter" in blocker for blocker in report.blockers))

    def test_direct_libero_backend_import_guard_does_not_require_lerobot(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = RiskProbeConfig(
                preset="runpod-libero-double-sim-smoke",
                backend="direct-libero",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=3,
                chunk_steps=3,
                action_dim=7,
                output_dir=tmpdir,
                direct_libero_double_sim=True,
            )
            report = run_risk_probes(config)

            self.assertEqual(report.status, BLOCKED)
            self.assertTrue(Path(report.artifacts["direct_libero_double_sim_evidence"]).exists())
            self.assertTrue(any("direct LIBERO double-sim" in blocker for blocker in report.blockers))

    def test_direct_libero_probe_module_builds_risk_config(self) -> None:
        config = DirectLiberoProbeConfig(
            suite="libero_goal",
            task_id=6,
            output_dir="/tmp/direct_probe",
            backend="direct-libero",
        )
        risk_config = build_risk_probe_config(config)

        self.assertEqual(risk_config.backend, "direct-libero")
        self.assertEqual(risk_config.task_ids, (6,))
        self.assertTrue(risk_config.direct_libero_double_sim)

    def test_direct_libero_probe_cli_mock_dry_run(self) -> None:
        with TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "run_imagine_then_act_direct_libero_probe.py"),
                "--backend",
                "mock",
                "--output-dir",
                tmpdir,
                "--json",
            ]
            completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)

            self.assertEqual(payload["status"], PASS)
            self.assertTrue(Path(payload["summary_path"]).exists())
            self.assertTrue(Path(payload["html_report"]).exists())

    def test_diagnose_libero_sim_state_cli_import_guard(self) -> None:
        with TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "diagnose_libero_sim_state.py"),
                "--output-dir",
                tmpdir,
                "--json",
            ]
            completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)

            self.assertEqual(payload["status"], BLOCKED)
            self.assertTrue(Path(payload["artifact_path"]).exists())

    def test_actual_adapter_partial_outcomes_still_generate_full_artifact_bundle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = RiskProbeConfig(
                preset="runpod-libero-smoke",
                backend="libero-contract",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=4,
                chunk_steps=3,
                action_dim=7,
                output_dir=tmpdir,
            )

            def fake_adapter(config, candidates, output_dir):  # noqa: ANN001, ARG001
                evidence_path = Path(output_dir) / "libero_adapter_evidence.json"
                evidence_path.write_text("{}\n", encoding="utf-8")
                return {
                    "mode": "libero_actual_adapter",
                    "available": True,
                    "blockers": ["partial actual probe"],
                    "artifact_path": str(evidence_path),
                    "outcomes": {
                        candidates[0].candidate_id: {
                            "success_proxy": 0.0,
                            "state": [0.0, 0.0, 0.0],
                            "image": [[0]],
                        }
                    },
                }

            with patch(
                "physical_ai_agent.imagine_then_act.risk_probes.run_libero_actual_adapter",
                side_effect=fake_adapter,
            ):
                report = run_risk_probes(config)

            self.assertEqual(report.status, "BLOCKED")
            self.assertEqual(report.diversity.verdict, WARN)
            self.assertIn("synthetic candidate diversity", report.diversity.rationale)
            self.assertTrue(report.candidates[0]["evaluated_in_actual_adapter"])
            self.assertFalse(report.candidates[1]["evaluated_in_actual_adapter"])
            self.assertTrue(Path(report.artifacts["oracle_scores"]).exists())
            self.assertTrue(Path(report.artifacts["html_report"]).exists())
            oracle_svg = Path(report.artifacts["oracle_scores"]).read_text(encoding="utf-8")
            self.assertIn("candidate_00_policy_only", oracle_svg)
            self.assertNotIn("candidate_01", oracle_svg)

    def test_actual_adapter_egl_blocker_marks_actual_evidence_unavailable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = RiskProbeConfig(
                preset="runpod-libero-double-sim-smoke",
                backend="libero-contract",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=4,
                chunk_steps=3,
                action_dim=7,
                output_dir=tmpdir,
                renderer_backend="egl",
            )

            def fake_adapter(config, candidates, output_dir):  # noqa: ANN001, ARG001
                evidence_path = Path(output_dir) / "libero_adapter_evidence.json"
                evidence = {
                    "mode": "libero_actual_adapter",
                    "available": False,
                    "blocker_category": "libero_egl_l4_blocked",
                    "blockers": [
                        "LIBERO actual adapter failed during env rollout: ImportError: Cannot initialize a EGL device display."
                    ],
                    "risk1_actual_unavailable_reason": "ImportError: Cannot initialize a EGL device display.",
                    "risk5_actual_unavailable_reason": "ImportError: Cannot initialize a EGL device display.",
                    "artifact_path": str(evidence_path),
                }
                evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
                return evidence

            with patch(
                "physical_ai_agent.imagine_then_act.risk_probes.run_libero_actual_adapter",
                side_effect=fake_adapter,
            ):
                report = run_risk_probes(config)

            self.assertEqual(report.status, BLOCKED)
            self.assertEqual(report.diversity.verdict, WARN)
            self.assertEqual(report.diversity.provenance, "actual_unavailable")
            self.assertIn("libero_egl_l4_blocked", report.diversity.rationale)
            self.assertEqual(report.oracle_upper_bound.verdict, BLOCKED)
            self.assertIn("actual privileged oracle evidence unavailable", report.oracle_upper_bound.rationale)
            summary = json.loads(Path(report.artifacts["summary"]).read_text(encoding="utf-8"))
            self.assertIn("libero_egl_l4_blocked", summary["diversity"]["rationale"])

    def test_actual_adapter_synthetic_fallback_cannot_pass_risk1(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = RiskProbeConfig(
                preset="runpod-libero-double-sim-smoke",
                backend="libero-contract",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=4,
                chunk_steps=3,
                action_dim=7,
                output_dir=tmpdir,
            )

            def fake_adapter(config, candidates, output_dir):  # noqa: ANN001, ARG001
                evidence_path = Path(output_dir) / "libero_adapter_evidence.json"
                evidence = {
                    "mode": "libero_actual_adapter",
                    "available": True,
                    "blockers": [],
                    "candidate_generation": {
                        "source": "synthetic_fallback",
                        "candidate_count": 0,
                        "errors": ["candidate_00_policy_only policy candidate sampling failed: missing image features"],
                        "fallback_reason": "policy candidate sampling produced no usable candidates",
                    },
                    "action_candidates": [],
                    "fallback_candidates": [candidate.__dict__ for candidate in candidates],
                    "outcomes": {
                        candidates[0].candidate_id: {"success_proxy": 0.0, "state": [0.0], "image": [[0]]}
                    },
                    "artifact_path": str(evidence_path),
                }
                evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
                return evidence

            with patch(
                "physical_ai_agent.imagine_then_act.risk_probes.run_libero_actual_adapter",
                side_effect=fake_adapter,
            ):
                report = run_risk_probes(config)

            self.assertEqual(report.diversity.verdict, WARN)
            self.assertEqual(report.diversity.provenance, "policy_sampling_unavailable")
            self.assertIn("missing image features", report.diversity.rationale)
            self.assertNotEqual(report.risk_verdicts["risk_1_candidate_diversity"], PASS)

    def test_lerobot_policy_observation_preparation_adds_task_and_preprocessor_features(self) -> None:
        raw_observation = {"pixels": [1], "robot_state": [0.1, 0.2]}
        env = _FakeTaskEnv()

        def fake_preprocess(observation):  # noqa: ANN001
            return {"image": observation["pixels"], "state": observation["robot_state"]}

        def fake_env_preprocessor(observation):  # noqa: ANN001
            observation = dict(observation)
            observation["observation.images.camera1"] = observation.pop("image")
            return observation

        def fake_policy_preprocessor(observation):  # noqa: ANN001
            observation = dict(observation)
            observation["pixels"] = observation.pop("observation.images.camera1")
            observation["robot_state"] = observation.pop("state")
            return observation

        processed, metadata = prepare_lerobot_policy_observation(
            observation=raw_observation,
            env=env,
            env_preprocessor=fake_env_preprocessor,
            preprocessor=fake_policy_preprocessor,
            preprocess_observation_fn=fake_preprocess,
        )

        self.assertEqual(processed["task"], ["fake task"])
        self.assertIn("pixels", processed)
        self.assertIn("robot_state", processed)
        self.assertEqual(metadata["task_source"], "task_description")
        self.assertEqual(metadata["steps"], ["preprocess_observation", "env_preprocessor", "policy_preprocessor"])

    def test_policy_candidate_sampling_records_seeded_action_chunks(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = RiskProbeConfig(
                preset="runpod-libero-smoke",
                backend="libero-contract",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=4,
                chunk_steps=2,
                action_dim=3,
                output_dir=tmpdir,
            )

            candidates, metadata = sample_policy_action_candidates(
                policy=_FakeChunkPolicy(),
                observation={"state": [0.0]},
                postprocessor=lambda action: action,
                env_postprocessor=lambda transition: transition,
                action_key="action",
                config=config,
                torch_module=_FakeTorch,
                numpy_module=None,
            )
            metrics = compute_diversity_metrics(config, candidates)

            self.assertEqual(len(candidates), 4)
            self.assertEqual(candidates[0].candidate_id, "candidate_00_policy_only")
            self.assertTrue(candidates[0].is_policy_only)
            self.assertEqual(candidates[1].seed, 1201)
            self.assertEqual(metadata["source"], "policy")
            self.assertEqual(metadata["candidate_count"], 4)
            self.assertEqual(metrics.provenance, "policy_generated")
            self.assertGreater(metrics.min_pairwise_l2, 0.0)
            self.assertGreaterEqual(metrics.selected_vs_policy_l2, 0.0)

    def test_debug_noise_candidate_diversity_cannot_be_method_pass(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = RiskProbeConfig(
                preset="runpod-libero-smoke",
                backend="libero-contract",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=3,
                chunk_steps=2,
                action_dim=3,
                output_dir=tmpdir,
                debug_candidate_noise_scale=0.5,
            )

            candidates, _metadata = sample_policy_action_candidates(
                policy=_FakeConstantPolicy(),
                observation={"state": [0.0]},
                postprocessor=lambda action: action,
                env_postprocessor=lambda transition: transition,
                action_key="action",
                config=config,
                torch_module=_FakeTorch,
                numpy_module=None,
            )
            metrics = compute_diversity_metrics(config, candidates)

            self.assertEqual(metrics.verdict, WARN)
            self.assertIn("debug_noise", metrics.rationale)

    def test_identical_policy_candidates_warn_as_deterministic_sampling_limit(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = RiskProbeConfig(
                preset="runpod-libero-smoke",
                backend="libero-contract",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=3,
                chunk_steps=2,
                action_dim=3,
                output_dir=tmpdir,
            )

            candidates, _metadata = sample_policy_action_candidates(
                policy=_FakeConstantPolicy(),
                observation={"state": [0.0]},
                postprocessor=lambda action: action,
                env_postprocessor=lambda transition: transition,
                action_key="action",
                config=config,
                torch_module=_FakeTorch,
                numpy_module=None,
            )
            metrics = compute_diversity_metrics(config, candidates)

            self.assertEqual(metrics.verdict, WARN)
            self.assertIn("deterministic", metrics.rationale)

    def test_torch_transformers_import_compatibility_patch_adds_float8_alias(self) -> None:
        previous_torch = sys.modules.get("torch")
        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "2.5.1+cu124"
        fake_torch.float8_e5m2 = object()
        try:
            sys.modules["torch"] = fake_torch

            result = apply_torch_transformers_import_compatibility_patch()

            self.assertTrue(result["patched"])
            self.assertIs(fake_torch.float8_e8m0fnu, fake_torch.float8_e5m2)
            self.assertEqual(result["torch_version"], "2.5.1+cu124")
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch


class _FakeNumpy:
    float32 = float

    @staticmethod
    def asarray(value, dtype=None):  # noqa: ARG004
        return value


class _FakeTorch:
    @staticmethod
    def inference_mode():
        class _Context:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        return _Context()

    @staticmethod
    def manual_seed(seed):  # noqa: ANN001
        import random

        random.seed(seed)


class _FakeChunkPolicy:
    name = "fake_smolvla"

    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict_action_chunk(self, observation):  # noqa: ANN001, ARG002
        import random

        offset = self.reset_count * 0.01
        return [
            [
                [round(random.uniform(-0.2, 0.2) + offset, 4), 0.1 + offset, -0.1],
                [round(random.uniform(-0.2, 0.2) + offset, 4), 0.2 + offset, -0.2],
            ]
        ]


class _FakeConstantPolicy:
    name = "fake_smolvla"

    def reset(self) -> None:
        return None

    def predict_action_chunk(self, observation):  # noqa: ANN001, ARG002
        return [[[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]]


class _FakeTaskEnv:
    num_envs = 1

    def call(self, name):  # noqa: ANN001
        if name == "task_description":
            return ["fake task"]
        raise AttributeError(name)


class _FakeActualEnv:
    num_envs = 1

    def __init__(self) -> None:
        self.state = [0.0, 0.0, 0.0]

    def reset(self, seed=None):  # noqa: ARG002
        self.state = [0.0, 0.0, 0.0]
        return self._observation(), {"reset": True}

    def step(self, action):
        row = action[0]
        for index in range(3):
            self.state[index] += float(row[index])
        info = {"success": [self.state[0] > 1.0]}
        return self._observation(), [0.0], [False], [False], info

    def call(self, name):
        if name == "task_description":
            return ["fake libero task"]
        raise AttributeError(name)

    def _observation(self):
        base = int(max(0, min(255, self.state[0] * 100)))
        return {
            "state": [self.state[:]],
            "agentview_image": [[[base, base, base], [base, base, base]]],
        }


class _FakeVectorRobosuiteEnv:
    num_envs = 1

    def __init__(self) -> None:
        self.sim = _FakeMuJoCoSim()
        self.robosuite = _FakeRobosuiteEnv(self.sim)
        self.envs = [_FakeVectorLeaf(self.robosuite)]

    def reset(self, seed=None):  # noqa: ARG002
        self.sim.set_state([0.0, 0.0, 0.0])
        self.sim.forward()
        return self._observation(), {"reset": True}

    def set_init_state(self, init_state):  # noqa: ANN001
        self.sim.set_state(init_state)
        self.sim.forward()
        return self._observation()

    def step(self, action):
        row = action[0] if action and isinstance(action[0], list) else action
        self.sim.state = [self.sim.state[index] + float(row[index]) for index in range(3)]
        self.sim.forward()
        info = {"success": [self.robosuite.check_success()]}
        reward = [1.0 if info["success"][0] else 0.0]
        return self._observation(), reward, [False], [False], info

    def call(self, name):
        if name == "task_description":
            return ["fake deep libero task"]
        raise AttributeError(name)

    def _observation(self):
        base = int(max(0, min(255, sum(self.sim.state) * 80)))
        return {
            "state": [self.sim.state[:]],
            "agentview_image": [[[base, base, base], [base, base, base]]],
        }


class _FakeResetReplacesSimEnv(_FakeVectorRobosuiteEnv):
    def __init__(self) -> None:
        super().__init__()
        self.sim = _FakeBrokenRootSim()

    def reset(self, seed=None):  # noqa: ARG002
        self.sim = _FakeMuJoCoSim()
        self.robosuite = _FakeRobosuiteEnv(self.sim)
        self.envs = [_FakeVectorLeaf(self.robosuite)]
        return self._observation(), {"reset": True}

    def set_init_state(self, init_state):  # noqa: ANN001
        self.sim.set_state(init_state)
        self.robosuite = _FakeRobosuiteEnv(self.sim)
        self.envs = [_FakeVectorLeaf(self.robosuite)]
        return self._observation()


class _FakeForwardRaisesEnv(_FakeVectorRobosuiteEnv):
    def __init__(self) -> None:
        super().__init__()
        self.sim = _FakeForwardRaisesSim()
        self.robosuite = _FakeRobosuiteEnv(self.sim)
        self.envs = [_FakeVectorLeaf(self.robosuite)]

    def reset(self, seed=None):  # noqa: ARG002
        self.sim.set_state([0.0, 0.0, 0.0])
        return self._observation(), {"reset": True}

    def set_init_state(self, init_state):  # noqa: ANN001
        self.sim.set_state(init_state)
        return self._observation()

    def step(self, action):
        row = action[0] if action and isinstance(action[0], list) else action
        self.sim.state = [self.sim.state[index] + float(row[index]) for index in range(3)]
        self.sim.data = _FakeMuJoCoData(self.sim.state)
        info = {"success": [self.robosuite.check_success()]}
        reward = [1.0 if info["success"][0] else 0.0]
        return self._observation(), reward, [False], [False], info


class _FakeVectorLeaf:
    def __init__(self, robosuite_env) -> None:  # noqa: ANN001
        self._env = _FakeEnvWrapper(robosuite_env)


class _FakeEnvWrapper:
    def __init__(self, robosuite_env) -> None:  # noqa: ANN001
        self.unwrapped = _FakeUnwrapped(robosuite_env)


class _FakeUnwrapped:
    def __init__(self, robosuite_env) -> None:  # noqa: ANN001
        self.env = robosuite_env


class _FakeRobosuiteEnv:
    def __init__(self, sim) -> None:  # noqa: ANN001
        self.sim = sim
        self.target_pos = [0.3, 0.3, 0.3]
        self.object_pos = [0.0, 0.0, 0.0]

    def check_success(self) -> bool:
        return sum(self.sim.state) > 0.8

    def reward(self) -> float:
        return 1.0 if self.check_success() else 0.0


class _FakeMuJoCoSim:
    def __init__(self) -> None:
        self.state = [0.0, 0.0, 0.0]
        self.forward_calls = 0
        self.data = _FakeMuJoCoData(self.state)

    def get_state(self):
        return self.state[:]

    def set_state(self, state) -> None:  # noqa: ANN001
        self.state = [float(value) for value in state]
        self.data = _FakeMuJoCoData(self.state)

    def forward(self) -> None:
        self.forward_calls += 1
        self.data = _FakeMuJoCoData(self.state)


class _FakeForwardRaisesSim(_FakeMuJoCoSim):
    def forward(self) -> None:
        self.forward_calls += 1
        raise AttributeError("'MjSim' object has no attribute 'data'")


class _FakeMuJoCoData:
    def __init__(self, state: list[float]) -> None:
        self.body_xpos = [state[:], [value + 0.1 for value in state]]
        self.site_xpos = [[value + 0.2 for value in state]]
        self.qpos = state[:]
        self.qvel = [0.0 for _ in state]


class _FakeBrokenRootGoodNestedEnv:
    def __init__(self) -> None:
        self.sim = _FakeBrokenRootSim()
        self.env = _FakeNestedGoodEnv()


class _FakeBrokenRootSim:
    def get_state(self):
        return [9.0]

    def set_state(self, state) -> None:  # noqa: ANN001
        raise AttributeError("'MjSim' object has no attribute 'data'")

    def forward(self) -> None:
        raise AttributeError("'MjSim' object has no attribute 'data'")


class _FakeNestedGoodEnv:
    def __init__(self) -> None:
        self.sim = _FakeMuJoCoSim()


class _FakeQposOnlyEnv:
    def __init__(self) -> None:
        self.sim = _FakeQposOnlySim()


class _FakeQposOnlySim:
    def __init__(self) -> None:
        self.data = _FakeQposOnlyData()
        self.model = object()
        self.forward_calls = 0

    def forward(self) -> None:
        self.forward_calls += 1


class _FakeQposOnlyData:
    def __init__(self) -> None:
        self.qpos = [0.0, 0.0]
        self.qvel = [0.0, 0.0]
