import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.imagine_then_act.risk_probes import (
    ActionChunkCandidate,
    FAIL,
    PASS,
    WARN,
    RiskProbeConfig,
    apply_candidate_to_env,
    apply_torch_transformers_import_compatibility_patch,
    compute_clone_fidelity_metrics,
    compute_actual_oracle_or_proxy_metrics,
    compute_diversity_metrics,
    compute_oracle_upper_bound_metrics,
    generate_mock_candidates,
    inspect_actual_env,
    run_risk_probes,
    simulate_mock_env,
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
            report = run_risk_probes(self.make_config(tmpdir))
            html_path = Path(report.artifacts["html_report"])
            html = html_path.read_text(encoding="utf-8")

            self.assertEqual(report.status, PASS)
            self.assertIn("Risk 1 Candidate Diversity", html)
            self.assertIn("Risk 2 Clone Fidelity", html)
            self.assertIn("Risk 5 Oracle Selector Upper-Bound", html)
            self.assertIn("candidate_action_heatmap.svg", html)
            self.assertIn("clone_image_diff.svg", html)
            self.assertIn("oracle_scores.svg", html)
            for artifact_path in report.artifacts.values():
                self.assertTrue(Path(artifact_path).exists())

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
