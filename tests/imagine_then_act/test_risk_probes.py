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
    build_libero_risk_probe_rollout,
    build_risk1a_prompt_portfolio,
    build_risk1b_subgoal_portfolio,
    capture_sim_state,
    compute_risk1c_selector_evidence,
    compute_clone_fidelity_metrics,
    compute_actual_oracle_or_proxy_metrics,
    compute_diversity_metrics,
    compute_oracle_upper_bound_metrics,
    compute_task_relation_proxy,
    compute_task_relation_proxy_with_env_fallback,
    find_sim_clone_handle,
    generate_mock_candidates,
    inspect_actual_env,
    restore_sim_state,
    run_direct_env_snapshot_replay,
    run_risk_probes,
    prepare_lerobot_policy_observation,
    risk1b_subgoal_to_prompt,
    sample_policy_action_candidates,
    sample_policy_prompt_portfolio_candidates,
    sample_policy_vlm_subgoal_candidates,
    select_actual_probe_candidates,
    simulate_mock_env,
    validate_risk1b_subgoal_records,
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

    def test_smolvla_sampling_probe_cli_mock_writes_probe_json(self) -> None:
        with TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "probe_smolvla_sampling_api.py"),
                "--mode",
                "mock",
                "--output-dir",
                tmpdir,
                "--json",
            ]
            completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)
            probe_path = Path(payload["output_path"])
            probe = json.loads(probe_path.read_text(encoding="utf-8"))

            self.assertTrue(probe_path.exists())
            self.assertEqual(probe["probe"], "smolvla_sampling_api")
            self.assertEqual(probe["mock_sampling"]["candidate_generation"]["candidate_sampling_api"], "predict_action_chunk(noise=...)")
            self.assertIn(probe["sampling_api_available"], {"yes", "no", "unclear"})

    def test_cli_risk1a_prompt_portfolio_mock_writes_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "run_imagine_then_act_risk_probes.py"),
                "--preset",
                "local-dry-run",
                "--risk1a-prompt-portfolio",
                "--risk1a-ambiguity",
                "--output-dir",
                tmpdir,
                "--json",
            ]
            completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)
            artifact_path = Path(payload["risk1a_prompt_portfolio"])
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

            self.assertTrue(artifact_path.exists())
            self.assertEqual(artifact["risk"], "risk_1a_prompt_portfolio")
            self.assertEqual(artifact["provenance"], "mock_contract")
            self.assertEqual(len(artifact["prompts"]), 5)
            self.assertIn("native_noise_reference", artifact)
            self.assertNotEqual(artifact["provenance"], "policy_generated")

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
        parsed_risk1a = module.build_parser().parse_args(
            ["--preset", "runpod-libero-smoke", "--risk1a-prompt-portfolio", "--risk1a-ambiguity"]
        )
        risk1a_config = module.build_config(parsed_risk1a)
        self.assertTrue(risk1a_config.risk1a_prompt_portfolio)
        self.assertTrue(risk1a_config.risk1a_ambiguity)
        parsed_risk1b = module.build_parser().parse_args(
            [
                "--preset",
                "runpod-libero-smoke",
                "--risk1b-vlm-subgoals",
                "--risk1b-generator-backend",
                "json",
                "--risk1b-subgoals-json",
                "subgoals.json",
                "--risk1c-sim-selector",
                "--risk1c-selector-modes",
                "c0,c2",
            ]
        )
        risk1b_config = module.build_config(parsed_risk1b)
        self.assertTrue(risk1b_config.risk1b_vlm_subgoals)
        self.assertEqual(risk1b_config.risk1b_generator_backend, "json")
        self.assertEqual(risk1b_config.risk1b_subgoals_json, "subgoals.json")
        self.assertTrue(risk1b_config.risk1c_sim_selector)
        self.assertEqual(risk1b_config.risk1c_selector_modes, ("c0", "c2"))

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
        self.assertEqual(metrics.evidence_class, "privileged_oracle_available")
        self.assertTrue(metrics.privileged_oracle_available)
        self.assertTrue(metrics.upper_bound_testable)

    def test_proxy_only_oracle_metrics_cannot_pass(self) -> None:
        candidates = [
            ActionChunkCandidate("candidate_00_policy_only", "policy", [[0.0]], 0.0, is_policy_only=True),
            ActionChunkCandidate("candidate_01", "smolvla.predict_action_chunk", [[0.1]], 0.0),
        ]
        outcomes = {
            "candidate_00_policy_only": {"success_proxy": 0.1},
            "candidate_01": {"success_proxy": 0.9},
        }

        metrics = compute_actual_oracle_or_proxy_metrics(
            candidates,
            outcomes,
            {"privileged_state_available": False},
        )

        self.assertEqual(metrics.verdict, WARN)
        self.assertEqual(metrics.evidence_class, "proxy_only")
        self.assertFalse(metrics.privileged_oracle_available)
        self.assertFalse(metrics.upper_bound_testable)
        self.assertIn("proxy_only", metrics.rationale)

    def test_privileged_oracle_requires_policy_and_alternative_candidates(self) -> None:
        candidates = [
            ActionChunkCandidate("candidate_00_policy_only", "smolvla.predict_action_chunk", [[0.0]], 0.0, is_policy_only=True)
        ]
        outcomes = {"candidate_00_policy_only": {"success_proxy": 0.0}}

        metrics = compute_actual_oracle_or_proxy_metrics(
            candidates,
            outcomes,
            {"privileged_state_available": True},
        )

        self.assertEqual(metrics.verdict, WARN)
        self.assertEqual(metrics.evidence_class, "privileged_oracle_available")
        self.assertTrue(metrics.privileged_oracle_available)
        self.assertFalse(metrics.upper_bound_testable)
        self.assertIn("upper_bound_not_testable", metrics.rationale)

    def test_privileged_oracle_requires_score_spread_for_upper_bound_claim(self) -> None:
        candidates = [
            ActionChunkCandidate("candidate_00_policy_only", "smolvla.predict_action_chunk", [[0.0]], 0.0, is_policy_only=True),
            ActionChunkCandidate("candidate_01", "smolvla.predict_action_chunk", [[0.0]], 0.0),
        ]
        outcomes = {
            "candidate_00_policy_only": {"success_proxy": 0.0},
            "candidate_01": {"success_proxy": 0.0},
        }

        metrics = compute_actual_oracle_or_proxy_metrics(
            candidates,
            outcomes,
            {"privileged_state_available": True},
        )

        self.assertEqual(metrics.verdict, WARN)
        self.assertEqual(metrics.evidence_class, "privileged_oracle_available")
        self.assertFalse(metrics.upper_bound_testable)
        self.assertIn("no_score_spread", metrics.rationale)

    def test_actual_probe_selection_keeps_policy_and_nonrandom_actual_comparison(self) -> None:
        candidates = [
            ActionChunkCandidate("candidate_00_policy_only", "smolvla.predict_action_chunk", [[0.0]], 0.0, is_policy_only=True),
            ActionChunkCandidate("candidate_01", "smolvla.predict_action_chunk", [[0.0]], 0.0),
            ActionChunkCandidate("candidate_02", "smolvla.predict_action_chunk", [[0.0]], 0.0),
        ]

        selected = select_actual_probe_candidates(candidates)

        self.assertEqual([candidate.candidate_id for candidate in selected], ["candidate_00_policy_only", "candidate_01"])

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
            self.assertEqual(evidence["oracle_upper_bound"]["evidence_class"], "privileged_oracle_available")
            self.assertTrue(evidence["oracle_upper_bound"]["privileged_oracle_available"])
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

    def test_direct_privileged_oracle_boundary_overrides_wrapper_proxy_only_report(self) -> None:
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
                direct_libero_double_sim=True,
            )
            candidates = generate_mock_candidates(config)
            wrapper_oracle = compute_actual_oracle_or_proxy_metrics(
                candidates[:2],
                {
                    candidates[0].candidate_id: {"success_proxy": 0.0},
                    candidates[1].candidate_id: {"success_proxy": 0.0},
                },
                {"privileged_state_available": False},
            )
            direct_oracle = compute_actual_oracle_or_proxy_metrics(
                candidates[:2],
                {
                    candidates[0].candidate_id: {"success_proxy": 0.0},
                    candidates[1].candidate_id: {"success_proxy": 0.0},
                },
                {"privileged_state_available": True},
            )

            def fake_adapter(config, candidates, output_dir):  # noqa: ANN001, ARG001
                evidence_path = Path(output_dir) / "libero_adapter_evidence.json"
                evidence = {
                    "mode": "libero_actual_adapter",
                    "available": True,
                    "blockers": [],
                    "action_candidates": [candidate.__dict__ for candidate in candidates],
                    "outcomes": {
                        candidates[0].candidate_id: {"success_proxy": 0.0, "state": [0.0], "image": [[0]]},
                        candidates[1].candidate_id: {"success_proxy": 0.0, "state": [0.0], "image": [[0]]},
                    },
                    "oracle_upper_bound": wrapper_oracle.__dict__,
                    "direct_libero_double_sim": {"oracle_upper_bound": direct_oracle.__dict__},
                    "artifact_path": str(evidence_path),
                }
                evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
                return evidence

            with patch(
                "physical_ai_agent.imagine_then_act.risk_probes.run_libero_actual_adapter",
                side_effect=fake_adapter,
            ):
                report = run_risk_probes(config)

            self.assertEqual(report.oracle_upper_bound.verdict, WARN)
            self.assertEqual(report.oracle_upper_bound.evidence_class, "privileged_oracle_available")
            self.assertTrue(report.oracle_upper_bound.privileged_oracle_available)
            self.assertFalse(report.oracle_upper_bound.upper_bound_testable)
            self.assertIn("no_score_spread", report.oracle_upper_bound.rationale)

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

    def test_policy_candidate_sampling_uses_explicit_noise_when_policy_exposes_api(self) -> None:
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

            candidates, metadata = sample_policy_action_candidates(
                policy=_FakeNoiseAwareChunkPolicy(config.chunk_steps, config.action_dim),
                observation={"state": _FakeTensor([[0.0]])},
                postprocessor=lambda action: action,
                env_postprocessor=lambda transition: transition,
                action_key="action",
                config=config,
                torch_module=_FakeNoiseTorch,
                numpy_module=None,
            )

            self.assertEqual(metadata["candidate_sampling_api"], "predict_action_chunk(noise=...)")
            self.assertFalse(candidates[0].sampling_metadata["explicit_noise_requested"])
            self.assertTrue(candidates[1].sampling_metadata["explicit_noise_used"])
            self.assertEqual(candidates[1].sampling_metadata["explicit_noise_shape"], [1, 2, 3])
            self.assertNotEqual(candidates[1].action_chunk, candidates[2].action_chunk)

    def test_risk1a_prompt_portfolio_preserves_single_prompt_by_default(self) -> None:
        disabled = build_risk1a_prompt_portfolio(
            base_prompt="Move the object to the target.",
            num_prompts=5,
            enabled=False,
            ambiguity_requested=True,
        )
        obvious = build_risk1a_prompt_portfolio(
            base_prompt="Move the object to the target.",
            num_prompts=5,
            enabled=True,
            ambiguity_requested=False,
        )

        self.assertEqual(len(disabled), 1)
        self.assertEqual(len(obvious), 1)
        self.assertEqual(disabled[0]["strategy"], "baseline_original_prompt")

    def test_risk1a_prompt_portfolio_generates_subgoal_preserving_strategy_axes(self) -> None:
        prompts = build_risk1a_prompt_portfolio(
            base_prompt="Move the object to the target.",
            num_prompts=5,
            enabled=True,
            ambiguity_requested=True,
        )

        self.assertEqual(len(prompts), 5)
        self.assertEqual(prompts[0]["axis"], "baseline")
        self.assertIn("alignment_before_contact", {prompt["axis"] for prompt in prompts})
        self.assertTrue(all("Move the object to the target." in prompt["prompt"] for prompt in prompts))

    def test_risk1a_prompt_portfolio_sampling_records_prompt_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = RiskProbeConfig(
                preset="runpod-libero-smoke",
                backend="libero-contract",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=5,
                chunk_steps=2,
                action_dim=3,
                output_dir=tmpdir,
                risk1a_prompt_portfolio=True,
                risk1a_ambiguity=True,
            )

            candidates, metadata = sample_policy_prompt_portfolio_candidates(
                policy=_FakePromptAwarePolicy(config.chunk_steps, config.action_dim),
                raw_observation={"state": [0.0]},
                env=_FakeTaskEnv(),
                env_preprocessor=lambda observation: observation,
                preprocessor=lambda observation: observation,
                preprocess_observation_fn=lambda observation: dict(observation),
                postprocessor=lambda action: action,
                env_postprocessor=lambda transition: transition,
                action_key="action",
                config=config,
                torch_module=_FakeTorch,
                numpy_module=None,
            )
            metrics = compute_diversity_metrics(config, candidates)

            self.assertEqual(metadata["source"], "risk1a_prompt_portfolio")
            self.assertTrue(metadata["risk1a_prompt_portfolio"]["active"])
            self.assertEqual(len(candidates), 5)
            self.assertEqual(candidates[0].selection_role, "baseline_original_prompt")
            self.assertEqual(candidates[1].sampling_metadata["candidate_generation"], "risk1a_prompt_portfolio")
            self.assertIn("prompt_text", candidates[1].sampling_metadata)
            self.assertGreater(metrics.mean_normalized_pairwise_l2, 0.0)

    def test_risk1b_subgoal_schema_validation_rejects_missing_fields(self) -> None:
        valid, errors = validate_risk1b_subgoal_records(
            [{"subgoal_text": "align first", "confidence": 0.8}],
            limit=5,
        )

        self.assertEqual(valid, [])
        self.assertTrue(any("missing required fields" in error for error in errors))

    def test_risk1b_json_portfolio_validates_required_schema(self) -> None:
        with TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "vlm_subgoals.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                        "latency_ms": 3210,
                        "memory_mb": 11264,
                        "cost_usd": 0.0,
                        "subgoals": [
                            {
                                "subgoal_text": "Move the object to the target.",
                                "strategy_axis": "baseline",
                                "target_object": "object",
                                "target_region_or_point": "target region",
                                "stop_condition": "object reaches target",
                                "confidence": 1.0,
                            },
                            {
                                "subgoal_text": "Align the gripper before contact.",
                                "strategy_axis": "alignment",
                                "target_object": "object",
                                "target_region_or_point": "pre-contact region",
                                "stop_condition": "gripper is aligned",
                                "confidence": 0.7,
                                "rationale": "grounded alternative",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            subgoals, validation = build_risk1b_subgoal_portfolio(
                base_prompt="Move the object to the target.",
                num_subgoals=5,
                model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                generator_backend="json",
                subgoals_json=str(payload_path),
            )

            self.assertTrue(validation["valid"])
            self.assertEqual(validation["provenance"], "external_vlm_json")
            self.assertEqual(validation["latency_ms"], 3210)
            self.assertEqual(len(subgoals), 2)
            self.assertEqual(subgoals[1]["strategy_axis"], "alignment")

    def test_risk1b_vlm_subgoal_sampling_records_schema_and_policy_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "vlm_subgoals.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "subgoals": [
                            {
                                "subgoal_text": "Move the object to the target.",
                                "strategy_axis": "baseline",
                                "target_object": "object",
                                "target_region_or_point": "target",
                                "stop_condition": "done",
                                "confidence": 1.0,
                            },
                            {
                                "subgoal_text": "Approach from the open side before contact.",
                                "strategy_axis": "object_centric_direction",
                                "target_object": "object",
                                "target_region_or_point": "open side",
                                "stop_condition": "near contact",
                                "confidence": 0.8,
                            },
                            {
                                "subgoal_text": "Align gripper before closing.",
                                "strategy_axis": "gripper_alignment",
                                "target_object": "gripper and object",
                                "target_region_or_point": "pre-grasp",
                                "stop_condition": "aligned",
                                "confidence": 0.75,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = RiskProbeConfig(
                preset="runpod-libero-smoke",
                backend="libero-contract",
                suite="libero_goal",
                task_ids=(6,),
                seed=1201,
                num_candidates=5,
                chunk_steps=2,
                action_dim=3,
                output_dir=tmpdir,
                risk1b_vlm_subgoals=True,
                risk1b_generator_backend="json",
                risk1b_subgoals_json=str(payload_path),
            )

            candidates, metadata = sample_policy_vlm_subgoal_candidates(
                policy=_FakePromptAwarePolicy(config.chunk_steps, config.action_dim),
                raw_observation={"state": [0.0]},
                env=_FakeTaskEnv(),
                env_preprocessor=lambda observation: observation,
                preprocessor=lambda observation: observation,
                preprocess_observation_fn=lambda observation: dict(observation),
                postprocessor=lambda action: action,
                env_postprocessor=lambda transition: transition,
                action_key="action",
                config=config,
                torch_module=_FakeTorch,
                numpy_module=None,
            )

            self.assertEqual(metadata["source"], "risk1b_vlm_subgoals")
            self.assertEqual(metadata["risk1b_vlm_subgoals"]["provenance"], "external_vlm_json_policy_generated")
            self.assertEqual(len(candidates), 3)
            self.assertEqual(candidates[1].sampling_metadata["candidate_generation"], "risk1b_vlm_subgoals")
            self.assertEqual(candidates[1].sampling_metadata["strategy_axis"], "object_centric_direction")
            self.assertIn("Strategy instruction:", candidates[1].sampling_metadata["prompt_text"])

    def test_risk1b_subgoal_prompt_expands_strategy_axis_into_actionable_directive(self) -> None:
        prompt = risk1b_subgoal_to_prompt(
            "put the cream cheese in the bowl",
            {
                "subgoal_text": "Move the cream cheese into the bowl.",
                "strategy_axis": "object_centric_open_side",
                "target_object": "cream_cheese_1",
                "target_region_or_point": "akita_black_bowl_1",
                "stop_condition": "cream_cheese_1_in_bowl",
                "confidence": 0.8,
            },
        )

        self.assertIn("Strategy instruction:", prompt)
        self.assertIn("visible open side", prompt)
        self.assertIn("least obstructed side", prompt)
        self.assertIn("cream_cheese_1", prompt)
        self.assertIn("akita_black_bowl_1", prompt)

    def test_task_relation_proxy_scores_observation_object_target_distance(self) -> None:
        near = compute_task_relation_proxy(
            {
                "cream_cheese_1_pos": [0.1, 0.0, 0.0],
                "akita_black_bowl_1_pos": [0.2, 0.0, 0.0],
            },
            "put the cream cheese in the bowl",
        )
        far = compute_task_relation_proxy(
            {
                "cream_cheese_1_pos": [0.1, 0.0, 0.0],
                "akita_black_bowl_1_pos": [1.1, 0.0, 0.0],
            },
            "put the cream cheese in the bowl",
        )

        self.assertTrue(near["available"])
        self.assertEqual(near["source"], "observation_object_target_distance_proxy")
        self.assertGreater(near["score"], far["score"])

    def test_task_relation_proxy_uses_raw_env_observation_fallback(self) -> None:
        class RawObservationEnv:
            def _get_observations(self):
                return {
                    "cream_cheese_1_pos": [0.2, 0.0, 0.0],
                    "akita_black_bowl_1_pos": [0.4, 0.0, 0.0],
                }

        proxy = compute_task_relation_proxy_with_env_fallback(
            {"agentview_image": [[[0, 0, 0]]]},
            "put the cream cheese in the bowl",
            RawObservationEnv(),
        )

        self.assertTrue(proxy["available"])
        self.assertEqual(proxy["source"], "observation_object_target_distance_proxy")
        self.assertEqual(proxy["observation_source"], "root._get_observations()")
        self.assertEqual(proxy["primary_observation_proxy"]["reason"], "object_or_target_position_unavailable")

    def test_actual_rollout_closure_records_task_relation_proxy_without_name_error(self) -> None:
        previous_modules = {name: sys.modules.get(name) for name in ("lerobot", "lerobot.envs", "lerobot.utils", "lerobot.utils.constants", "torch")}
        lerobot_module = types.ModuleType("lerobot")
        envs_module = types.ModuleType("lerobot.envs")
        utils_module = types.ModuleType("lerobot.utils")
        constants_module = types.ModuleType("lerobot.utils.constants")
        torch_module = types.ModuleType("torch")
        envs_module.preprocess_observation = lambda observation: dict(observation)
        constants_module.ACTION = "action"
        torch_module.from_numpy = lambda value: value
        sys.modules["lerobot"] = lerobot_module
        sys.modules["lerobot.envs"] = envs_module
        sys.modules["lerobot.utils"] = utils_module
        sys.modules["lerobot.utils.constants"] = constants_module
        sys.modules["torch"] = torch_module
        try:
            with TemporaryDirectory() as tmpdir:
                payload_path = Path(tmpdir) / "vlm_subgoals.json"
                payload_path.write_text(
                    json.dumps(
                        {
                            "subgoals": [
                                {
                                    "subgoal_text": "Put the cream cheese in the bowl.",
                                    "strategy_axis": "baseline",
                                    "target_object": "cream_cheese_1",
                                    "target_region_or_point": "akita_black_bowl_1",
                                    "stop_condition": "cream_cheese_1_in_bowl",
                                    "confidence": 1.0,
                                },
                                {
                                    "subgoal_text": "Approach the cream cheese from the open side before placing it in the bowl.",
                                    "strategy_axis": "object_centric_open_side",
                                    "target_object": "cream_cheese_1",
                                    "target_region_or_point": "akita_black_bowl_1",
                                    "stop_condition": "cream_cheese_1_in_bowl",
                                    "confidence": 0.8,
                                },
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                config = RiskProbeConfig(
                    preset="runpod-libero-smoke",
                    backend="libero-contract",
                    suite="libero_goal",
                    task_ids=(6,),
                    seed=1201,
                    num_candidates=2,
                    chunk_steps=2,
                    action_dim=3,
                    output_dir=tmpdir,
                    risk1b_vlm_subgoals=True,
                    risk1b_generator_backend="json",
                    risk1b_subgoals_json=str(payload_path),
                )
                evidence_path = Path(tmpdir) / "libero_adapter_evidence.json"
                rollout = build_libero_risk_probe_rollout(
                    config=config,
                    evidence_path=evidence_path,
                    candidates=generate_mock_candidates(config),
                    seed=config.seed,
                    max_steps=2,
                )

                rollout(
                    _FakeTaskRelationVectorEnv(),
                    _FakePromptAwarePolicy(config.chunk_steps, config.action_dim),
                    env_preprocessor=lambda observation: observation,
                    env_postprocessor=lambda transition: transition,
                    preprocessor=lambda observation: observation,
                    postprocessor=lambda action: action,
                    seeds=[config.seed],
                )

                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                outcomes = evidence["outcomes"]
                self.assertTrue(evidence["available"])
                self.assertTrue(outcomes)
                self.assertTrue(
                    all(item["task_relation_proxy_source"] == "observation_object_target_distance_proxy" for item in outcomes.values())
                )
        finally:
            for name, module in previous_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_cli_risk1b_and_risk1c_mock_write_contract_artifacts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "run_imagine_then_act_risk_probes.py"),
                "--preset",
                "local-dry-run",
                "--risk1b-vlm-subgoals",
                "--risk1c-sim-selector",
                "--output-dir",
                tmpdir,
                "--json",
            ]
            completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)
            risk1b = json.loads(Path(payload["risk1b_vlm_subgoals"]).read_text(encoding="utf-8"))
            risk1c = json.loads(Path(payload["risk1c_sim_selector"]).read_text(encoding="utf-8"))

            self.assertEqual(risk1b["risk"], "risk_1b_external_vlm_subgoal_generator")
            self.assertEqual(risk1b["provenance"], "mock_contract")
            self.assertEqual(risk1b["verdict"], BLOCKED)
            self.assertIn("c0_privileged_oracle", risk1c["modes"])
            self.assertIn("c1_non_oracle_proxy", risk1c["modes"])
            self.assertIn("c2_action_only_debug", risk1c["modes"])
            self.assertEqual(risk1c["modes"]["c2_action_only_debug"]["verdict"], WARN)

    def test_risk1c_selector_evidence_separates_oracle_proxy_and_debug(self) -> None:
        config = RiskProbeConfig(
            preset="local-dry-run",
            backend="mock",
            suite="libero_goal",
            task_ids=(6,),
            seed=1201,
            num_candidates=4,
            chunk_steps=2,
            action_dim=3,
            output_dir="/tmp/risk1c-test",
            risk1c_sim_selector=True,
        )
        candidates = generate_mock_candidates(config)
        outcomes = {candidate.candidate_id: simulate_mock_env(candidate.action_chunk) for candidate in candidates}
        oracle = compute_oracle_upper_bound_metrics(candidates, outcomes)

        evidence = compute_risk1c_selector_evidence(
            config=config,
            candidates=candidates,
            outcomes=outcomes,
            oracle=oracle,
            actual_evidence={},
        )

        self.assertEqual(evidence["modes"]["c0_privileged_oracle"]["selector_class"], "C0")
        self.assertEqual(evidence["modes"]["c1_non_oracle_proxy"]["selector_class"], "C1")
        self.assertEqual(evidence["modes"]["c2_action_only_debug"]["selector_class"], "C2")
        self.assertIn("debug baseline only", evidence["modes"]["c2_action_only_debug"]["claim_boundary"])

    def test_risk1c_c1_prefers_observation_relation_proxy_when_available(self) -> None:
        config = RiskProbeConfig(
            preset="local-dry-run",
            backend="mock",
            suite="libero_goal",
            task_ids=(6,),
            seed=1201,
            num_candidates=2,
            chunk_steps=2,
            action_dim=3,
            output_dir="/tmp/risk1c-test",
            risk1c_sim_selector=True,
        )
        candidates = [
            ActionChunkCandidate("candidate_00_policy_only", "policy", [[0.0, 0.0, 0.0]], 0.0, is_policy_only=True),
            ActionChunkCandidate("candidate_01", "policy", [[0.1, 0.0, 0.0]], 0.0),
        ]
        outcomes = {
            "candidate_00_policy_only": {"success_proxy": 0.0, "task_relation_proxy": 0.3},
            "candidate_01": {"success_proxy": 0.0, "task_relation_proxy": 0.7},
        }
        oracle = compute_actual_oracle_or_proxy_metrics(
            candidates,
            {"candidate_00_policy_only": {"success_proxy": 0.0}, "candidate_01": {"success_proxy": 0.0}},
            {"privileged_state_available": False},
        )

        evidence = compute_risk1c_selector_evidence(
            config=config,
            candidates=candidates,
            outcomes=outcomes,
            oracle=oracle,
            actual_evidence={},
        )

        c1 = evidence["modes"]["c1_non_oracle_proxy"]
        self.assertEqual(c1["score_source"], "observation_object_target_distance_proxy")
        self.assertEqual(c1["score_spread"], 0.4)
        self.assertEqual(c1["selected_candidate_id"], "candidate_01")
        self.assertTrue(c1["non_baseline_selection"])
        self.assertEqual(c1["plausibility_failures"], [])

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


class _FakeNoiseTorch(_FakeTorch):
    float32 = "float32"

    @staticmethod
    def normal(**kwargs):  # noqa: ANN003
        import random

        size = kwargs["size"]
        return [
            [
                [round(random.uniform(-0.5, 0.5), 6) for _dim in range(size[2])]
                for _step in range(size[1])
            ]
            for _batch in range(size[0])
        ]


class _FakeTensor(list):
    @property
    def shape(self) -> tuple[int, ...]:
        if self and isinstance(self[0], list):
            return (len(self), len(self[0]))
        return (len(self),)

    @property
    def device(self) -> str:
        return "cpu"


class _FakeNoisePolicyConfig:
    def __init__(self, chunk_size: int, action_dim: int) -> None:
        self.chunk_size = chunk_size
        self.max_action_dim = action_dim


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


class _FakeNoiseAwareChunkPolicy:
    name = "fake_smolvla"

    def __init__(self, chunk_size: int, action_dim: int) -> None:
        self.config = _FakeNoisePolicyConfig(chunk_size, action_dim)

    def reset(self) -> None:
        return None

    def predict_action_chunk(self, observation, noise=None):  # noqa: ANN001, ARG002
        if noise is None:
            return [
                [
                    [0.0 for _dim in range(self.config.max_action_dim)]
                    for _step in range(self.config.chunk_size)
                ]
            ]
        return noise


class _FakePromptAwarePolicy:
    name = "fake_smolvla"

    def __init__(self, chunk_size: int, action_dim: int) -> None:
        self.config = _FakeNoisePolicyConfig(chunk_size, action_dim)

    def reset(self) -> None:
        return None

    def predict_action_chunk(self, observation, noise=None):  # noqa: ANN001, ARG002
        task = observation.get("task", [""])[0] if isinstance(observation, dict) else ""
        offset = (sum(ord(char) for char in task) % 17) / 100.0
        return [
            [
                [round(offset + 0.01 * step + 0.001 * dim, 6) for dim in range(self.config.max_action_dim)]
                for step in range(self.config.chunk_size)
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


class _FakeTaskRelationVectorEnv:
    num_envs = 1

    def __init__(self) -> None:
        self.cream_x = 0.0
        self.bowl_x = 1.0

    def reset(self, seed=None):  # noqa: ARG002
        self.cream_x = 0.0
        self.bowl_x = 1.0
        return self._observation(), {"reset": True}

    def step(self, action):
        row = action[0]
        self.cream_x = min(self.bowl_x, self.cream_x + abs(float(row[0])) + 0.05)
        return self._observation(), [0.0], [False], [False], {"is_success": [False]}

    def call(self, name):
        if name == "task_description":
            return ["put the cream cheese in the bowl"]
        raise AttributeError(name)

    def _observation(self):
        return {
            "cream_cheese_1_pos": [self.cream_x, 0.0, 0.0],
            "akita_black_bowl_1_pos": [self.bowl_x, 0.0, 0.0],
            "agentview_image": [[[0, 0, 0], [0, 0, 0]]],
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
