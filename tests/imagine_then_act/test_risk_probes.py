import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.imagine_then_act.risk_probes import (
    PASS,
    RiskProbeConfig,
    compute_clone_fidelity_metrics,
    compute_diversity_metrics,
    compute_oracle_upper_bound_metrics,
    generate_mock_candidates,
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
