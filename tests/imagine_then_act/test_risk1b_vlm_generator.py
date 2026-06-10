import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from physical_ai_agent.imagine_then_act.risk_probes import build_risk1b_subgoal_portfolio


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate_risk1b_vlm_subgoals.py"


class Risk1BVlmGeneratorTest(TestCase):
    def test_mock_backend_writes_valid_contract_json(self) -> None:
        with TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(SCRIPT),
                    "--backend",
                    "mock",
                    "--model-id",
                    "Qwen/Qwen2.5-VL-7B-Instruct",
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
            payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))

            self.assertEqual(result["status"], "PASS")
            self.assertTrue(Path(result["raw_output_path"]).exists())
            self.assertEqual(payload["generator_backend"], "mock")
            self.assertEqual(payload["provenance"], "mock_contract")
            self.assertTrue(payload["schema_validation"]["valid"])
            self.assertEqual(len(payload["subgoals"]), 5)
            self.assertIn("risk1b_subgoals_qwen2_5_vl_7b_instruct_libero_goal_task6_seed1201.json", result["output_path"])
            self.assertIn("cannot count as Risk1-B PASS", payload["boundary"])

    def test_fixture_backend_rejects_invalid_schema(self) -> None:
        with TemporaryDirectory() as tmpdir:
            fixture = Path(tmpdir) / "bad_output.txt"
            fixture.write_text(json.dumps({"subgoals": [{"subgoal_text": "missing required fields"}]}), encoding="utf-8")
            output_path = Path(tmpdir) / "invalid.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(SCRIPT),
                    "--backend",
                    "fixture",
                    "--fixture-output",
                    str(fixture),
                    "--output-path",
                    str(output_path),
                    "--json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("schema_validation_error", completed.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["schema_validation"]["valid"])
            self.assertEqual(payload["provenance"], "fixture_contract")

    def test_fixture_contract_ingestion_cannot_be_external_vlm_provenance(self) -> None:
        with TemporaryDirectory() as tmpdir:
            generated = Path(tmpdir) / "fixture_contract.json"
            generated.write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                        "generator_backend": "fixture",
                        "provenance": "external_vlm_json",
                        "subgoals": [
                            {
                                "subgoal_text": "baseline",
                                "strategy_axis": "baseline",
                                "target_object": "object",
                                "target_region_or_point": "target",
                                "stop_condition": "done",
                                "confidence": 1.0,
                            },
                            {
                                "subgoal_text": "align first",
                                "strategy_axis": "alignment",
                                "target_object": "object",
                                "target_region_or_point": "pre-contact",
                                "stop_condition": "aligned",
                                "confidence": 0.8,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            _subgoals, validation = build_risk1b_subgoal_portfolio(
                base_prompt="Move the object.",
                num_subgoals=5,
                model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                generator_backend="json",
                subgoals_json=str(generated),
            )

            self.assertTrue(validation["valid"])
            self.assertEqual(validation["provenance"], "fixture_contract")

    def test_parser_rejects_unknown_model(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with self.assertRaises(SystemExit):
            module.build_parser().parse_args(["--model-id", "unknown/model"])

    def test_select_transformers_model_class_falls_back_from_image_text_to_vision2seq(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        fake_transformers = SimpleNamespace(AutoModelForVision2Seq=object())

        cls, name = module.select_transformers_model_class(fake_transformers)

        self.assertIs(cls, fake_transformers.AutoModelForVision2Seq)
        self.assertEqual(name, "AutoModelForVision2Seq")

    def test_select_transformers_model_class_reports_specific_missing_loader(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with self.assertRaisesRegex(RuntimeError, "AutoModelForImageTextToText"):
            module.select_transformers_model_class(SimpleNamespace())

    def test_dependency_check_reports_specific_missing_imports_or_loader(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPT),
                "--backend",
                "transformers",
                "--dependency-check-only",
                "--json",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        if completed.returncode == 0:
            payload = json.loads(completed.stdout)
            self.assertIn("model_loader_class", payload)
            self.assertEqual(payload["status"], "PASS")
        else:
            self.assertIn("RUNPOD_VLM_ENV_OR_MODEL_LOAD_BLOCKED", completed.stderr)
