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
            self.assertEqual(
                payload["candidate_prompt_semantics"],
                "same_immediate_goal_strategy_variants_first_action_chunk",
            )
            self.assertTrue(payload["schema_validation"]["valid"])
            self.assertEqual(len(payload["subgoals"]), 5)
            self.assertEqual(payload["strategy_variants"], payload["subgoals"])
            self.assertEqual(payload["candidate_prompts"], payload["subgoals"])
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

    def test_fixture_backend_unwraps_nested_subgoal_records(self) -> None:
        with TemporaryDirectory() as tmpdir:
            context = Path(tmpdir) / "context.json"
            context.write_text(
                json.dumps(
                    {
                        "task_description": "put the cream cheese in the bowl",
                        "provenance": {"actual_context": True},
                    }
                ),
                encoding="utf-8",
            )
            fixture = Path(tmpdir) / "nested_output.txt"
            fixture.write_text(
                json.dumps(
                    {
                        "subgoals": [
                            {
                                "subgoal": {
                                    "subgoal_text": "put the cream cheese in the bowl",
                                    "strategy_axis": "baseline",
                                    "target_object": "cream cheese",
                                    "target_region_or_point": "inside the bowl",
                                    "stop_condition": "cream cheese is in the bowl",
                                    "confidence": 1.0,
                                }
                            },
                            {
                                "subgoal": {
                                    "subgoal_text": "align the cream cheese before placing it in the bowl",
                                    "strategy_axis": "pre_contact_alignment",
                                    "target_object": "cream cheese",
                                    "target_region_or_point": "inside the bowl",
                                    "stop_condition": "cream cheese is in the bowl",
                                    "confidence": 0.8,
                                }
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_path = Path(tmpdir) / "nested.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(SCRIPT),
                    "--backend",
                    "fixture",
                    "--fixture-output",
                    str(fixture),
                    "--context-json",
                    str(context),
                    "--output-path",
                    str(output_path),
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            result = json.loads(completed.stdout)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(payload["schema_validation"]["valid"])
            self.assertEqual(payload["subgoals"][0]["target_object"], "cream cheese")

    def test_fixture_backend_rejects_wrong_object_relation_for_cream_cheese_task(self) -> None:
        with TemporaryDirectory() as tmpdir:
            context = Path(tmpdir) / "context.json"
            context.write_text(
                json.dumps(
                    {
                        "task_description": "put the cream cheese in the bowl",
                        "provenance": {"actual_context": True},
                    }
                ),
                encoding="utf-8",
            )
            fixture = Path(tmpdir) / "wrong_object_output.txt"
            fixture.write_text(
                json.dumps(
                    [
                        {
                            "subgoal": {
                                "subgoal_text": "Pick up the black bowl.",
                                "strategy_axis": "object_centric_open_side",
                                "target_object": "akita_black_bowl_1",
                                "target_region_or_point": "akita_black_bowl_1_to_robot0_eef_pos",
                                "stop_condition": "gripper is closed and holding the bowl",
                                "confidence": 0.8,
                            }
                        },
                        {
                            "subgoal": {
                                "subgoal_text": "Pick up the black bowl.",
                                "strategy_axis": "pre_contact_alignment",
                                "target_object": "akita_black_bowl_1",
                                "target_region_or_point": "akita_black_bowl_1_to_robot0_eef_pos",
                                "stop_condition": "gripper is closed and holding the bowl",
                                "confidence": 0.8,
                            }
                        },
                    ]
                ),
                encoding="utf-8",
            )
            output_path = Path(tmpdir) / "wrong_object.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(SCRIPT),
                    "--backend",
                    "fixture",
                    "--fixture-output",
                    str(fixture),
                    "--context-json",
                    str(context),
                    "--output-path",
                    str(output_path),
                    "--json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 2)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["schema_validation"]["valid"])
            self.assertIn("expected cream cheese into bowl", "\n".join(payload["schema_validation"]["errors"]))

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

    def test_generation_prompt_requests_same_subgoal_strategy_portfolio(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        args = SimpleNamespace(
            num_subgoals=5,
            suite="libero_goal",
            task_id=6,
            seed=1201,
            task_description="Put the cream cheese in the black bowl.",
        )

        prompt = module.build_generation_prompt(
            args,
            {
                "provenance": {"actual_context": True},
                "observation_source": "direct_libero_offscreen_env_reset_observation",
            },
        )

        self.assertIn("SAME immediate next subgoal", prompt)
        self.assertIn("Do NOT decompose the task over time", prompt)
        self.assertIn("same target object, same target relation, and", prompt)
        self.assertIn("varying only the approach strategy", prompt)
        self.assertIn("Do not vary only verbs", prompt)
        self.assertIn("behaviorally distinct motion cue", prompt)
        self.assertIn("object_centric_open_side", prompt)
        self.assertIn("pre_contact_alignment", prompt)
        self.assertIn("collision_avoidant_approach", prompt)
        self.assertIn("high_clearance_over_rim", prompt)
        self.assertIn("vertical_drop_centering", prompt)
        self.assertIn("Candidate 0 strategy_axis must be exactly \"baseline\"", prompt)

    def test_generation_prompt_prefers_actual_context_task_goal(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        args = SimpleNamespace(
            num_subgoals=5,
            suite="libero_goal",
            task_id=6,
            seed=1201,
            task_description="Complete the LIBERO goal task from the current observation.",
        )

        prompt = module.build_generation_prompt(
            args,
            {
                "task_description": "put the cream cheese in the bowl",
                "provenance": {"actual_context": True},
                "observation_summary": {
                    "akita_black_bowl_1_pos": {"type": "ndarray"},
                    "cream_cheese_1_pos": {"type": "ndarray"},
                    "robot0_joint_pos": {"type": "ndarray"},
                },
            },
        )

        self.assertIn("task_goal=put the cream cheese in the bowl", prompt)
        self.assertIn("task_goal_source=context_json.task_description", prompt)
        self.assertIn("If task_goal says to put/place/move X in/on/to Y", prompt)
        self.assertIn("cream cheese as", prompt)
        self.assertIn('"object_state_keys": ["akita_black_bowl_1_pos", "cream_cheese_1_pos"]', prompt)
        self.assertNotIn("robot0_joint_pos", prompt)

    def test_output_payload_accepts_nested_subgoal_records(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        args = SimpleNamespace(
            backend="transformers",
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            suite="libero_goal",
            task_id=6,
            seed=1201,
            num_subgoals=2,
            task_description="put the cream cheese in the bowl",
            context_image="contact.png",
            context_json="context.json",
        )

        payload = module.build_output_payload(
            args=args,
            prompt="prompt",
            context_summary={"provenance": {"actual_context": True}},
            raw_output=json.dumps(
                [
                    {
                        "subgoal": {
                            "subgoal_text": "put the cream cheese in the bowl",
                            "strategy_axis": "baseline",
                            "target_object": "cream_cheese_1",
                            "target_region_or_point": "akita_black_bowl_1",
                            "stop_condition": "cream cheese is in the bowl",
                            "confidence": 1.0,
                        }
                    },
                    {
                        "subgoal": {
                            "subgoal_text": "put the cream cheese in the bowl using pre-contact alignment",
                            "strategy_axis": "pre_contact_alignment",
                            "target_object": "cream_cheese_1",
                            "target_region_or_point": "akita_black_bowl_1",
                            "stop_condition": "cream cheese is in the bowl",
                            "confidence": 0.8,
                        }
                    },
                ]
            ),
            parsed=[
                {
                    "subgoal": {
                        "subgoal_text": "put the cream cheese in the bowl",
                        "strategy_axis": "baseline",
                        "target_object": "cream_cheese_1",
                        "target_region_or_point": "akita_black_bowl_1",
                        "stop_condition": "cream cheese is in the bowl",
                        "confidence": 1.0,
                    }
                },
                {
                    "subgoal": {
                        "subgoal_text": "put the cream cheese in the bowl using pre-contact alignment",
                        "strategy_axis": "pre_contact_alignment",
                        "target_object": "cream_cheese_1",
                        "target_region_or_point": "akita_black_bowl_1",
                        "stop_condition": "cream cheese is in the bowl",
                        "confidence": 0.8,
                    }
                },
            ],
            latency_ms=1,
            memory_mb=None,
            raw_output_path=Path("raw.txt"),
        )

        self.assertTrue(payload["schema_validation"]["valid"])
        self.assertEqual(payload["subgoals"][0]["target_object"], "cream_cheese_1")
        self.assertEqual(payload["subgoals"][1]["strategy_axis"], "pre_contact_alignment")

    def test_extract_json_payload_repairs_malformed_wrapped_subgoal_objects(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        raw_output = """
assistant
```json
[
  {
    "subgoal": {
      "subgoal_text": "open the middle drawer of the cabinet",
      "strategy_axis": "baseline",
      "target_object": "drawer",
      "target_region_or_point": "middle drawer",
      "stop_condition": "drawer fully open",
      "confidence": 0.8
    }
  },
  {
    {
      "subgoal": {
        "subgoal_text": "open the middle drawer of the cabinet",
        "strategy_axis": "pre_contact_alignment",
        "target_object": "drawer",
        "target_region_or_point": "middle drawer",
        "stop_condition": "drawer fully open",
        "confidence": 0.7
      }
    },
    {
      "subgoal": {
        "subgoal_text": "open the middle drawer of the cabinet",
        "strategy_axis": "collision_avoidant_approach",
        "target_object": "drawer",
        "target_region_or_point": "middle drawer",
        "stop_condition": "drawer fully open",
        "confidence": 0.6
      }
    }
]
```
"""

        parsed = module.extract_json_payload(raw_output)

        self.assertEqual(parsed["_schema_repair"]["strategy"], "extract_wrapped_subgoal_objects")
        self.assertEqual(len(parsed["subgoals"]), 3)
        self.assertEqual(parsed["subgoals"][1]["subgoal"]["strategy_axis"], "pre_contact_alignment")

    def test_output_payload_rejects_inverted_task_object_relation(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        args = SimpleNamespace(
            backend="transformers",
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            suite="libero_goal",
            task_id=6,
            seed=1201,
            num_subgoals=2,
            task_description="put the cream cheese in the bowl",
            context_image="contact.png",
            context_json="context.json",
        )
        bad_records = [
            {
                "subgoal": {
                    "subgoal_text": "Pick up the black bowl.",
                    "strategy_axis": "object_centric_open_side",
                    "target_object": "akita_black_bowl_1",
                    "target_region_or_point": "akita_black_bowl_1_to_robot0_eef_pos",
                    "stop_condition": "gripper is closed and holding the bowl",
                    "confidence": 0.8,
                }
            },
            {
                "subgoal": {
                    "subgoal_text": "Pick up the black bowl using pre-contact alignment.",
                    "strategy_axis": "pre_contact_alignment",
                    "target_object": "akita_black_bowl_1",
                    "target_region_or_point": "akita_black_bowl_1_to_robot0_eef_pos",
                    "stop_condition": "gripper is closed and holding the bowl",
                    "confidence": 0.8,
                }
            },
        ]

        payload = module.build_output_payload(
            args=args,
            prompt="prompt",
            context_summary={"task_description": "put the cream cheese in the bowl"},
            raw_output=json.dumps(bad_records),
            parsed=bad_records,
            latency_ms=1,
            memory_mb=None,
            raw_output_path=Path("raw.txt"),
        )

        self.assertFalse(payload["schema_validation"]["valid"])
        self.assertIn("must preserve manipulated object: cream cheese", "\n".join(payload["schema_validation"]["errors"]))
        self.assertIn("targets destination as manipulated object", "\n".join(payload["schema_validation"]["errors"]))

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

    def test_select_processor_class_falls_back_when_autoprocessor_lazy_import_fails(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class _FakeTransformers:
            Qwen2_5_VLProcessor = object()

            def __getattr__(self, name):  # noqa: ANN001
                if name == "AutoProcessor":
                    raise ModuleNotFoundError(
                        "Could not import module 'AutoProcessor'. Are this object's requirements defined correctly?"
                    )
                raise AttributeError(name)

        cls, name, attempts = module.select_transformers_processor_class(
            _FakeTransformers(),
            "Qwen/Qwen2.5-VL-7B-Instruct",
        )

        self.assertIs(cls, _FakeTransformers.Qwen2_5_VLProcessor)
        self.assertEqual(name, "Qwen2_5_VLProcessor")
        self.assertFalse(attempts[0]["ok"])
        self.assertEqual(attempts[0]["class"], "AutoProcessor")

    def test_select_processor_class_reports_lazy_loader_attempts(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class _BrokenTransformers:
            def __getattr__(self, name):  # noqa: ANN001
                raise ModuleNotFoundError(f"Could not import module '{name}'.")

        with self.assertRaisesRegex(RuntimeError, "PROCESSOR_LOADER"):
            module.select_transformers_processor_class(_BrokenTransformers(), "Qwen/Qwen2.5-VL-7B-Instruct")

    def test_torch_transformers_compatibility_blocks_missing_float8_for_new_transformers(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        fake_torch = SimpleNamespace(__version__="2.5.1+cu124", float8_e5m2=object())
        fake_transformers = SimpleNamespace(__version__="5.10.2")

        diagnostics = module.diagnose_torch_transformers_compatibility(fake_torch, fake_transformers)

        self.assertEqual(diagnostics["torch_transformers_compatibility"], "blocked")
        self.assertFalse(diagnostics["torch_float8_e8m0fnu_available"])
        self.assertIn("RUNPOD_VLM_ENV_OR_MODEL_LOAD_BLOCKED_COMPATIBILITY", diagnostics["compatibility_blocker"])
        self.assertIn("transformers 5.10.2", diagnostics["compatibility_blocker"])

    def test_torch_transformers_compatibility_allows_pinned_transformers_without_new_float8(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        fake_torch = SimpleNamespace(__version__="2.5.1+cu124", float8_e5m2=object())
        fake_transformers = SimpleNamespace(__version__="4.49.0")

        diagnostics = module.diagnose_torch_transformers_compatibility(fake_torch, fake_transformers)

        self.assertEqual(diagnostics["torch_transformers_compatibility"], "pass")
        self.assertNotIn("compatibility_blocker", diagnostics)

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
