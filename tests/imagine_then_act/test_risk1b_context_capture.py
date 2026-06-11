import json
import importlib.util
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCRIPT = ROOT / "scripts" / "capture_risk1b_context.py"
GENERATOR_SCRIPT = ROOT / "scripts" / "generate_risk1b_vlm_subgoals.py"
PREFLIGHT_SCRIPT = ROOT / "scripts" / "preflight_risk1b_context_capture.py"


class Risk1BContextCaptureTest(TestCase):
    def test_mock_context_writes_expected_paths_and_nonactual_provenance(self) -> None:
        with TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(CAPTURE_SCRIPT),
                    "--backend",
                    "mock",
                    "--suite",
                    "libero_goal",
                    "--task-id",
                    "6",
                    "--seed",
                    "1201",
                    "--output-dir",
                    tmpdir,
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
            sheet = Path(result["contact_sheet"])
            context = Path(result["context_json"])
            payload = json.loads(context.read_text(encoding="utf-8"))

            self.assertEqual(sheet.name, "contact_sheet_task6_seed1201.png")
            self.assertEqual(context.name, "context_task6_seed1201.json")
            self.assertTrue(sheet.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(payload["suite"], "libero_goal")
            self.assertEqual(payload["task_id"], 6)
            self.assertFalse(payload["provenance"]["actual_context"])
            self.assertEqual(payload["observation_source"], "mock_contract")
            self.assertGreaterEqual(payload["camera_count"] if "camera_count" in payload else len(payload["camera_images"]), 2)

    def test_transformers_generator_rejects_mock_context(self) -> None:
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(CAPTURE_SCRIPT),
                    "--backend",
                    "mock",
                    "--output-dir",
                    tmpdir,
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            context = Path(tmpdir) / "context_task6_seed1201.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(GENERATOR_SCRIPT),
                    "--backend",
                    "transformers",
                    "--context-json",
                    str(context),
                    "--output-dir",
                    tmpdir,
                    "--json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("requires actual Risk1-B context JSON", completed.stderr)

    def test_artifact_context_repackages_actual_provenance(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source_image = Path(tmpdir) / "source.png"
            source_image.write_bytes(b"\x89PNG\r\n\x1a\nmock")
            source_context = Path(tmpdir) / "source_context.json"
            source_context.write_text(
                json.dumps(
                    {
                        "suite": "libero_goal",
                        "task_id": 6,
                        "seed": 1201,
                        "task_description": "actual context fixture",
                        "provenance": {"actual_context": True, "backend": "previous_actual_artifact"},
                    }
                ),
                encoding="utf-8",
            )
            out_dir = Path(tmpdir) / "risk1b_context"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(CAPTURE_SCRIPT),
                    "--backend",
                    "artifact",
                    "--artifact-image",
                    str(source_image),
                    "--artifact-context-json",
                    str(source_context),
                    "--output-dir",
                    str(out_dir),
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
            payload = json.loads(Path(result["context_json"]).read_text(encoding="utf-8"))

            self.assertTrue(payload["provenance"]["actual_context"])
            self.assertEqual(Path(result["contact_sheet"]).name, "contact_sheet_task6_seed1201.png")
            self.assertEqual(payload["observation_source"], "existing_actual_artifact")

    def test_context_capture_preflight_builds_exact_libero_command(self) -> None:
        module = load_preflight_module()
        args = module.build_parser().parse_args(
            [
                "--python-bin",
                "/workspace/physical-ai/envs/lerobot_py312/bin/python",
                "--suite",
                "libero_goal",
                "--task-id",
                "6",
                "--seed",
                "1201",
                "--policy-path",
                "lerobot/smolvla_libero",
                "--policy-num-steps",
                "10",
                "--policy-n-action-steps",
                "15",
                "--renderer-backend",
                "egl",
                "--output-dir",
                "_workspace/runpod_results/ita_risk_probes/risk1b_context_preflight",
            ]
        )

        argv = module.build_context_capture_argv(args)

        self.assertEqual(argv[0], "/workspace/physical-ai/envs/lerobot_py312/bin/python")
        self.assertIn("scripts/capture_risk1b_context.py", argv)
        self.assertEqual(argv[argv.index("--backend") + 1], "libero")
        self.assertEqual(argv[argv.index("--renderer-backend") + 1], "egl")
        self.assertEqual(argv[argv.index("--policy-num-steps") + 1], "10")
        self.assertEqual(argv[argv.index("--policy-n-action-steps") + 1], "15")
        self.assertTrue(any(item.endswith("risk1b_context") for item in argv))

    def test_context_capture_preflight_classifies_dri_permission(self) -> None:
        module = load_preflight_module()
        failure = module.classify_context_capture_failure(
            "libEGL warning: failed to open /dev/dri/renderD129: Permission denied\n"
            "context_capture_error: ImportError: Cannot initialize a EGL device display",
            "",
        )

        self.assertEqual(failure["category"], "CONTEXT_CAPTURE_LIBERO_BLOCKED_EGL_DEVICE_PERMISSION")
        self.assertIn("render/card devices", failure["hint"])

    def test_context_capture_preflight_dry_run_writes_report_without_libero(self) -> None:
        with TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(PREFLIGHT_SCRIPT),
                    "--python-bin",
                    sys.executable,
                    "--output-dir",
                    tmpdir,
                    "--dry-run",
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)

            self.assertEqual(result["status"], "DRY_RUN")
            self.assertEqual(result["operation"], "risk1b_context_capture_preflight")
            self.assertIn("scripts/capture_risk1b_context.py", result["command_argv"])
            self.assertTrue((Path(tmpdir) / "risk1b_context_capture_preflight.json").exists())


def load_preflight_module():
    spec = importlib.util.spec_from_file_location("risk1b_context_preflight_for_test", PREFLIGHT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load preflight script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
