import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCRIPT = ROOT / "scripts" / "capture_risk1b_context.py"
GENERATOR_SCRIPT = ROOT / "scripts" / "generate_risk1b_vlm_subgoals.py"


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
