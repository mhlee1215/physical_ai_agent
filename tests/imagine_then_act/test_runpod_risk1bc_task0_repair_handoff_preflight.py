from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "runpod_risk1bc_task0_repair_handoff_preflight.py"


def load_module():
    spec = importlib.util.spec_from_file_location("risk1bc_task0_handoff_preflight_for_test", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Risk1BCTask0RepairHandoffPreflightTests(TestCase):
    def test_phase_specs_use_workspace_paths_and_repair_lane_defaults(self) -> None:
        module = load_module()
        parser = module.build_parser()
        args = parser.parse_args([])
        phases = module.build_phase_specs(args)
        phase_by_name = {phase.name: phase for phase in phases}

        self.assertEqual(args.project_dir, "/workspace/physical-ai/physical_ai_agent")
        self.assertEqual(args.work_root, "/workspace/physical-ai")
        self.assertEqual(args.task_id, 0)
        self.assertEqual(args.seed, 1201)
        self.assertEqual(args.renderer_backend, "osmesa")
        self.assertEqual(args.qwen_readiness_mode, "model-load")
        self.assertEqual(phases[0].name, "pod_local_prereq")
        self.assertIn("/usr/bin/python3.12", phases[0].argv)

        for phase in phases:
            joined = " ".join(phase.argv)
            self.assertNotIn("/root/physical-ai", joined)

        self.assertEqual(
            phase_by_name["canonical_env_install"].argv,
            [
                "sh",
                "/workspace/physical-ai/physical_ai_agent/scripts/install/runpod_install.sh",
                "--component",
                "libero-smolvla",
            ],
        )
        self.assertEqual(
            phase_by_name["vlm_env_gate"].argv,
            [
                "sh",
                "/workspace/physical-ai/physical_ai_agent/scripts/install/runpod_check.sh",
                "--component",
                "risk1b-vlm",
            ],
        )
        context_argv = phase_by_name["shallow_osmesa_context_preflight"].argv
        self.assertIn("scripts/preflight_risk1b_context_capture.py", context_argv[2])
        self.assertIn("--renderer-backend", context_argv)
        self.assertIn("osmesa", context_argv)
        self.assertIn("--task-id", context_argv)
        self.assertIn("0", context_argv)
        self.assertIn("--seed", context_argv)
        self.assertIn("1201", context_argv)

    def test_dependency_check_mode_uses_generator_without_model_load_code(self) -> None:
        module = load_module()
        parser = module.build_parser()
        args = parser.parse_args(["--qwen-readiness-mode", "dependency-check"])
        phase_by_name = {phase.name: phase for phase in module.build_phase_specs(args)}
        qwen_argv = phase_by_name["qwen7b_readiness"].argv

        self.assertIn("scripts/generate_risk1b_vlm_subgoals.py", qwen_argv[2])
        self.assertIn("--dependency-check-only", qwen_argv)
        self.assertIn("--model-id", qwen_argv)
        self.assertIn("Qwen/Qwen2.5-VL-7B-Instruct", qwen_argv)

    def test_dry_run_writes_machine_readable_report(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            code = module.main(["--dry-run", "--output-dir", tmpdir, "--json"])
            self.assertEqual(code, 0)
            report_path = Path(tmpdir) / "risk1bc_task0_repair_handoff_preflight.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["operation"], "risk1bc_task0_repair_handoff_preflight")
        self.assertEqual(report["status"], "DRY_RUN")
        self.assertEqual(report["task_id"], 0)
        self.assertEqual(report["seed"], 1201)
        self.assertEqual(len(report["phase_results"]), 7)
        self.assertIn("does not run Qwen generation", report["claim_boundary"])

    def test_blocked_report_stops_after_first_failed_phase(self) -> None:
        module = load_module()
        parser = module.build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args(["--output-dir", tmpdir])
            phases = module.build_phase_specs(args)
            fake_results = [
                {"phase": "pod_local_prereq", "status": "PASS", "stdout_log": "a", "stderr_log": "b"},
                {
                    "phase": "canonical_env_gate",
                    "status": "BLOCKED",
                    "stdout_log": "c",
                    "stderr_log": "d",
                    "blocker_category": "RUNPOD_TASK0_REPAIR_CANONICAL_ENV_GATE_BLOCKED",
                },
            ]

            calls: list[str] = []

            def fake_run_phase(phase, output_dir, dry_run=False):  # noqa: ANN001
                calls.append(phase.name)
                return fake_results[len(calls) - 1]

            original = module.run_phase
            module.run_phase = fake_run_phase
            try:
                code, report = module.run_preflight(args)
            finally:
                module.run_phase = original

        self.assertEqual(code, 2)
        self.assertEqual(calls, [phases[0].name, phases[1].name])
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["blocked_phase"], "canonical_env_gate")
        self.assertEqual(report["blocker_category"], "RUNPOD_TASK0_REPAIR_CANONICAL_ENV_GATE_BLOCKED")
        self.assertIn("stop the pod", report["cleanup_instruction"])
