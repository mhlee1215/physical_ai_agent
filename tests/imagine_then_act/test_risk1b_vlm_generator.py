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

    def test_difficulty_eval_generation_command_enables_repair_and_fallback(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "risk1bc_difficulty_eval_for_test",
            ROOT / "scripts" / "run_imagine_then_act_risk1bc_difficulty_eval.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        sys.modules["risk1bc_difficulty_eval_for_test"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        manifest = {
            "policy_path": "lerobot/smolvla_libero",
            "policy_num_steps": 10,
            "policy_n_action_steps": 15,
        }
        args = SimpleNamespace(
            manifest="manifest.json",
            categories="libero_long_hard_all",
            output_dir="out",
            python_bin="python",
            vlm_python_bin="vlm-python",
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            renderer_backend="osmesa",
            context_backend="libero-shallow",
            risk_backend="libero-contract",
            num_candidates=5,
            chunk_steps=15,
            action_dim=7,
            policy_path=None,
            policy_num_steps=None,
            policy_n_action_steps=None,
            risk1b_repair_attempts=3,
            risk1b_fallback_on_validation_error="none",
            execute=False,
            json=False,
        )
        config = module.build_config(args, manifest)
        row = {
            "suite": "libero_10",
            "task_id": 0,
            "seed": 1201,
            "baseline_category": "libero_long_hard_all",
            "task_description": "put the alphabet soup in the basket",
        }

        argv = module.build_generation_argv(config, row, Path("out/row"))

        self.assertIn("--repair-attempts", argv)
        self.assertIn("3", argv)
        self.assertIn("--fallback-on-validation-error", argv)
        self.assertIn("none", argv)

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
                                    "subgoal_text": "put the cream cheese in the bowl using a pre-contact alignment cue",
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
        self.assertIn("completed-goal command", prompt)
        self.assertIn("Invalid forms include \"align X", prompt)
        self.assertIn("Strategy-axis words such as pre_contact_alignment", prompt)
        self.assertIn("object_centric_open_side", prompt)
        self.assertIn("pre_contact_alignment", prompt)
        self.assertIn("collision_avoidant_approach", prompt)
        self.assertIn("high_clearance_over_rim", prompt)
        self.assertIn("vertical_drop_centering", prompt)
        self.assertIn("LOCKED_TASK_FIELDS are immutable", prompt)
        self.assertIn('"original_task": "Put the cream cheese in the black bowl."', prompt)
        self.assertIn('"manipulated_object": "cream cheese"', prompt)
        self.assertIn('"target_region": "black bowl"', prompt)
        self.assertIn('"relation": "cream cheese -> black bowl"', prompt)
        self.assertIn("target_object must be exactly", prompt)
        self.assertIn("target_region_or_point must be exactly", prompt)
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
        self.assertIn('locked_task_fields={"manipulated_object": "cream cheese"', prompt)
        self.assertIn("If task_goal says to put/place/move X in/on/to Y", prompt)
        self.assertIn("cream cheese as", prompt)
        self.assertIn('"object_state_keys": ["akita_black_bowl_1_pos", "cream_cheese_1_pos"]', prompt)
        self.assertNotIn("robot0_joint_pos", prompt)

    def test_locked_task_fields_cover_failed_libero10_relation_cases(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        failed_cases = [
            (
                0,
                "put both the alphabet soup and the tomato sauce in the basket",
                "alphabet soup and tomato sauce",
                "basket",
                "alphabet soup -> basket; tomato sauce -> basket",
            ),
            (
                1,
                "put both the cream cheese box and the butter in the basket",
                "cream cheese box and butter",
                "basket",
                "cream cheese box -> basket; butter -> basket",
            ),
            (
                2,
                "turn on the stove and put the moka pot on it",
                "moka pot",
                "stove",
                "moka pot -> stove; turn on stove",
            ),
            (
                4,
                "put the white mug on the left plate and put the yellow and white mug on the right plate",
                "white mug and yellow and white mug",
                "left plate and right plate",
                "white mug -> left plate; yellow and white mug -> right plate",
            ),
            (
                6,
                "put the white mug on the plate and put the chocolate pudding to the right of the plate",
                "white mug and chocolate pudding",
                "plate and right of plate",
                "white mug -> plate; chocolate pudding -> right of plate",
            ),
            (
                5,
                "pick up the book and place it in the back compartment of the caddy",
                "book",
                "back compartment of caddy",
                "book -> back compartment of caddy",
            ),
            (
                7,
                "put both the alphabet soup and the cream cheese box in the basket",
                "alphabet soup and cream cheese box",
                "basket",
                "alphabet soup -> basket; cream cheese box -> basket",
            ),
            (
                9,
                "put the yellow and white mug in the microwave and close it",
                "yellow and white mug",
                "microwave",
                "yellow and white mug -> microwave; close microwave",
            ),
        ]
        for task_id, task, manipulated_object, target_region, relation in failed_cases:
            with self.subTest(task_id=task_id):
                args = SimpleNamespace(
                    backend="transformers",
                    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                    suite="libero_10",
                    task_id=task_id,
                    seed=1201,
                    num_subgoals=5,
                    task_description="Complete the LIBERO-10 task from the current observation.",
                    context_image="contact.png",
                    context_json="context.json",
                )
                context = {
                    "task_description": task,
                    "provenance": {"actual_context": True},
                }

                locked_fields = module.build_locked_task_fields(task)
                self.assertEqual(
                    locked_fields,
                    {
                        "original_task": task,
                        "manipulated_object": manipulated_object,
                        "target_region": target_region,
                        "relation": relation,
                    },
                )
                prompt = module.build_generation_prompt(args, context)
                self.assertIn(json.dumps(locked_fields, sort_keys=True), prompt)
                self.assertIn(json.dumps(locked_fields, indent=2, sort_keys=True).splitlines()[1].strip(), prompt)

                raw_output = module.deterministic_locked_field_output(args, context)
                parsed = module.extract_json_payload(raw_output)
                payload = module.build_output_payload(
                    args=args,
                    prompt=prompt,
                    context_summary=context,
                    raw_output=raw_output,
                    parsed=parsed,
                    latency_ms=1,
                    memory_mb=None,
                    raw_output_path=Path("fallback.raw.txt"),
                    fallback={"strategy": "deterministic_locked_fields"},
                )

                self.assertTrue(payload["schema_validation"]["valid"], payload["schema_validation"]["errors"])
                self.assertEqual(payload["locked_task_fields"], locked_fields)
                for record in payload["candidate_prompts"]:
                    self.assertEqual(record["target_object"], manipulated_object)
                    self.assertEqual(record["target_region_or_point"], target_region)

    def test_relation_validator_rejects_failed_libero10_target_drift_fixtures(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        failed_tasks = {
            0: "put both the alphabet soup and the tomato sauce in the basket",
            1: "put both the cream cheese box and the butter in the basket",
            2: "turn on the stove and put the moka pot on it",
            4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
            5: "pick up the book and place it in the back compartment of the caddy",
            6: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
            7: "put both the alphabet soup and the cream cheese box in the basket",
            9: "put the yellow and white mug in the microwave and close it",
        }
        for task_id, task in failed_tasks.items():
            with self.subTest(task_id=task_id):
                locked_fields = module.build_locked_task_fields(task)
                args = SimpleNamespace(
                    backend="fixture",
                    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                    suite="libero_10",
                    task_id=task_id,
                    seed=1201,
                    num_subgoals=2,
                    task_description=task,
                    context_image=None,
                    context_json=None,
                )
                drifted_records = [
                    {
                        "subgoal_text": f"move {locked_fields['manipulated_object']} to the table",
                        "strategy_axis": "baseline" if index == 0 else "pre_contact_alignment",
                        "target_object": locked_fields["manipulated_object"],
                        "target_region_or_point": "table",
                        "stop_condition": f"{locked_fields['manipulated_object']} is on the table",
                        "confidence": 0.8,
                    }
                    for index in range(2)
                ]

                payload = module.build_output_payload(
                    args=args,
                    prompt="prompt",
                    context_summary={"task_description": task},
                    raw_output=json.dumps({"subgoals": drifted_records}),
                    parsed={"subgoals": drifted_records},
                    latency_ms=1,
                    memory_mb=None,
                    raw_output_path=Path("drifted.raw.txt"),
                )

                self.assertFalse(payload["schema_validation"]["valid"])
                self.assertIn("must preserve target region/relation", "\n".join(payload["schema_validation"]["errors"]))

    def test_repair_prompt_restates_locked_task_fields_exactly(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        task = "put the white mug on the plate and put the chocolate pudding to the right of the plate"

        repair_prompt = module.build_repair_prompt(
            base_prompt="base",
            task_description=task,
            validation_errors=["candidate_prompt[1] must preserve target region/relation: right of plate"],
            previous_output='{"subgoals": []}',
        )

        self.assertIn("VALIDATION FAILED", repair_prompt)
        self.assertIn('"original_task": "' + task + '"', repair_prompt)
        self.assertIn('"manipulated_object": "white mug and chocolate pudding"', repair_prompt)
        self.assertIn('"target_region": "plate and right of plate"', repair_prompt)
        self.assertIn('"relation": "white mug -> plate; chocolate pudding -> right of plate"', repair_prompt)
        self.assertIn("Every candidate target_object must exactly equal", repair_prompt)
        self.assertIn("Repair intermediate-only candidates", repair_prompt)
        self.assertIn("replace 'align X with Y'", repair_prompt)
        self.assertIn("final relation complete", repair_prompt)

    def test_prompt_and_repair_forbid_multi_object_candidate_splitting(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        task = "put both the alphabet soup and the tomato sauce in the basket"
        args = SimpleNamespace(
            num_subgoals=5,
            suite="libero_10",
            task_id=0,
            seed=1201,
            task_description=task,
        )

        prompt = module.build_generation_prompt(args, {"task_description": task})
        repair_prompt = module.build_repair_prompt(
            base_prompt=prompt,
            task_description=task,
            validation_errors=["candidate_prompt[1] target_object must exactly match locked manipulated_object"],
            previous_output='{"target_object": "alphabet soup"}',
        )

        self.assertIn("Do not split one object per candidate", prompt)
        self.assertIn("The target\n  region is the destination, not the manipulated object.", prompt)
        self.assertIn("alphabet soup -> basket; tomato sauce -> basket", prompt)
        self.assertIn("every candidate must keep the entire object group", repair_prompt)
        self.assertIn("do not make one-object candidates", repair_prompt)
        self.assertIn("Do not tell the robot to move, align, center", repair_prompt)

    def test_prompt_includes_multi_relation_pair_lock(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        task = "put the white mug on the left plate and put the yellow and white mug on the right plate"
        args = SimpleNamespace(
            num_subgoals=5,
            suite="libero_10",
            task_id=4,
            seed=1201,
            task_description=task,
        )

        prompt = module.build_generation_prompt(args, {"task_description": task})

        self.assertIn("MULTI-RELATION LOCK", prompt)
        self.assertIn("white mug -> left plate; yellow and white mug -> right plate", prompt)
        self.assertIn("white mug to left plate AND yellow and white mug to right plate", prompt)
        self.assertIn("Do not split relation pairs across candidates", prompt)

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

    def test_output_payload_accepts_single_wrapper_list_with_subgoals(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        task = "put both the alphabet soup and the tomato sauce in the basket"
        args = SimpleNamespace(
            backend="transformers",
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            suite="libero_10",
            task_id=0,
            seed=1201,
            num_subgoals=2,
            task_description=task,
            context_image=None,
            context_json=None,
        )
        records = [
            {
                "subgoal_text": task,
                "strategy_axis": "baseline",
                "target_object": "alphabet soup and tomato sauce",
                "target_region_or_point": "basket",
                "stop_condition": "alphabet soup and tomato sauce are in the basket",
                "confidence": 0.9,
            },
            {
                "subgoal_text": f"{task}; approach the basket from the open side.",
                "strategy_axis": "object_centric_open_side",
                "target_object": "alphabet soup and tomato sauce",
                "target_region_or_point": "basket",
                "stop_condition": "alphabet soup and tomato sauce are in the basket",
                "confidence": 0.8,
            },
        ]

        payload = module.build_output_payload(
            args=args,
            prompt="prompt",
            context_summary={"task_description": task},
            raw_output=json.dumps([{"subgoals": records}]),
            parsed=[{"subgoals": records}],
            latency_ms=1,
            memory_mb=None,
            raw_output_path=Path("wrapped.raw.txt"),
        )

        self.assertTrue(payload["schema_validation"]["valid"], payload["schema_validation"]["errors"])
        self.assertEqual(len(payload["candidate_prompts"]), 2)

    def test_output_payload_rejects_intermediate_alignment_without_completion(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        task = "put both moka pots on the stove"
        args = SimpleNamespace(
            backend="transformers",
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            suite="libero_10",
            task_id=8,
            seed=1201,
            num_subgoals=2,
            task_description=task,
            context_image=None,
            context_json=None,
        )
        records = [
            {
                "subgoal_text": task,
                "strategy_axis": "baseline",
                "target_object": "moka pots",
                "target_region_or_point": "stove",
                "stop_condition": "both moka pots are on the stove",
                "confidence": 1.0,
            },
            {
                "subgoal_text": "align both moka pots with the stove burners",
                "strategy_axis": "pre_contact_alignment",
                "target_object": "moka pots",
                "target_region_or_point": "stove",
                "stop_condition": "both moka pots are aligned with the stove burners",
                "confidence": 0.7,
            },
        ]

        payload = module.build_output_payload(
            args=args,
            prompt="prompt",
            context_summary={"task_description": task},
            raw_output=json.dumps({"candidate_prompts": records}),
            parsed={"candidate_prompts": records},
            latency_ms=1,
            memory_mb=None,
            raw_output_path=Path("moka.raw.txt"),
        )

        self.assertFalse(payload["schema_validation"]["valid"])
        self.assertTrue(
            any("must describe completed object-target relation" in error for error in payload["schema_validation"]["errors"])
        )

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

    def test_output_payload_rejects_destination_motion_in_candidate_text(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        task = "put both the alphabet soup and the tomato sauce in the basket"
        args = SimpleNamespace(
            backend="transformers",
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            suite="libero_10",
            task_id=0,
            seed=1201,
            num_subgoals=2,
            task_description=task,
            context_image="contact.png",
            context_json="context.json",
        )
        records = [
            {
                "subgoal_text": task,
                "strategy_axis": "baseline",
                "target_object": "alphabet soup and tomato sauce",
                "target_region_or_point": "basket",
                "stop_condition": "alphabet soup and tomato sauce are in the basket",
                "confidence": 0.9,
            },
            {
                "subgoal_text": "Move the basket closer to the alphabet soup and tomato sauce before placing them in the basket.",
                "strategy_axis": "pre_contact_alignment",
                "target_object": "alphabet soup and tomato sauce",
                "target_region_or_point": "basket",
                "stop_condition": "alphabet soup and tomato sauce are in the basket",
                "confidence": 0.8,
            },
        ]

        payload = module.build_output_payload(
            args=args,
            prompt="prompt",
            context_summary={"task_description": task},
            raw_output=json.dumps({"subgoals": records}),
            parsed={"subgoals": records},
            latency_ms=1,
            memory_mb=None,
            raw_output_path=Path("bad_destination_motion.raw.txt"),
        )

        self.assertFalse(payload["schema_validation"]["valid"])
        self.assertIn("text manipulates destination/target region", "\n".join(payload["schema_validation"]["errors"]))

    def test_output_payload_rejects_missing_required_auxiliary_action(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        task = "turn on the stove and put the moka pot on it"
        args = SimpleNamespace(
            backend="transformers",
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            suite="libero_10",
            task_id=2,
            seed=1201,
            num_subgoals=2,
            task_description=task,
            context_image="contact.png",
            context_json="context.json",
        )
        records = [
            {
                "subgoal_text": "place the moka pot on the stove",
                "strategy_axis": "baseline",
                "target_object": "moka pot",
                "target_region_or_point": "stove",
                "stop_condition": "moka pot is on the stove",
                "confidence": 0.9,
            },
            {
                "subgoal_text": "center the moka pot on the stove burner",
                "strategy_axis": "pre_contact_alignment",
                "target_object": "moka pot",
                "target_region_or_point": "stove",
                "stop_condition": "moka pot is centered on the stove burner",
                "confidence": 0.8,
            },
        ]

        payload = module.build_output_payload(
            args=args,
            prompt="prompt",
            context_summary={"task_description": task},
            raw_output=json.dumps({"subgoals": records}),
            parsed={"subgoals": records},
            latency_ms=1,
            memory_mb=None,
            raw_output_path=Path("missing_required_action.raw.txt"),
        )

        self.assertFalse(payload["schema_validation"]["valid"])
        self.assertIn("must preserve required task action: turn on stove", "\n".join(payload["schema_validation"]["errors"]))

    def test_output_payload_rejects_multi_relation_text_omission_even_with_full_fields(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        task = "put the white mug on the left plate and put the yellow and white mug on the right plate"
        args = SimpleNamespace(
            backend="transformers",
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            suite="libero_10",
            task_id=4,
            seed=1201,
            num_subgoals=2,
            task_description=task,
            context_image="contact.png",
            context_json="context.json",
        )
        records = [
            {
                "subgoal_text": "Put the white mug on the left plate.",
                "strategy_axis": "baseline",
                "target_object": "white mug and yellow and white mug",
                "target_region_or_point": "left plate and right plate",
                "stop_condition": "The white mug is placed on the left plate.",
                "confidence": 0.9,
            },
            {
                "subgoal_text": "Align the white mug with the left plate before placing.",
                "strategy_axis": "pre_contact_alignment",
                "target_object": "white mug and yellow and white mug",
                "target_region_or_point": "left plate and right plate",
                "stop_condition": "The white mug is aligned with the left plate.",
                "confidence": 0.8,
            },
        ]

        payload = module.build_output_payload(
            args=args,
            prompt="prompt",
            context_summary={"task_description": task},
            raw_output=json.dumps({"subgoals": records}),
            parsed={"subgoals": records},
            latency_ms=1,
            memory_mb=None,
            raw_output_path=Path("missing_multi_relation_text.raw.txt"),
        )

        errors = "\n".join(payload["schema_validation"]["errors"])
        self.assertFalse(payload["schema_validation"]["valid"])
        self.assertIn("subgoal_text/stop_condition must preserve relation object: yellow and white mug", errors)
        self.assertIn("subgoal_text/stop_condition must preserve relation target: right plate", errors)

    def test_output_payload_rejects_task0_partial_multi_object_candidate(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        task = "put both the alphabet soup and the tomato sauce in the basket"
        args = SimpleNamespace(
            backend="transformers",
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            suite="libero_10",
            task_id=0,
            seed=1201,
            num_subgoals=2,
            task_description=task,
            context_image="contact.png",
            context_json="context.json",
        )
        records = [
            {
                "subgoal_text": "Put the alphabet soup in the basket.",
                "strategy_axis": "baseline",
                "target_object": "alphabet soup and tomato sauce",
                "target_region_or_point": "basket",
                "stop_condition": "The alphabet soup is in the basket.",
                "confidence": 0.9,
            },
            {
                "subgoal_text": "Place the tomato sauce in the basket.",
                "strategy_axis": "object_centric_open_side",
                "target_object": "alphabet soup and tomato sauce",
                "target_region_or_point": "basket",
                "stop_condition": "The tomato sauce is in the basket.",
                "confidence": 0.8,
            },
        ]

        payload = module.build_output_payload(
            args=args,
            prompt="prompt",
            context_summary={"task_description": task},
            raw_output=json.dumps({"subgoals": records}),
            parsed={"subgoals": records},
            latency_ms=1,
            memory_mb=None,
            raw_output_path=Path("partial_multi_object.raw.txt"),
        )

        errors = "\n".join(payload["schema_validation"]["errors"])
        self.assertFalse(payload["schema_validation"]["valid"])
        self.assertIn("subgoal_text/stop_condition must preserve relation object: tomato sauce", errors)
        self.assertIn("subgoal_text/stop_condition must preserve relation object: alphabet soup", errors)

    def test_prompt_and_repair_include_stove_action_lock(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        task = "turn on the stove and put the moka pot on it"
        args = SimpleNamespace(
            num_subgoals=5,
            suite="libero_10",
            task_id=2,
            seed=1201,
            task_description=task,
        )

        prompt = module.build_generation_prompt(args, {"task_description": task})
        repair_prompt = module.build_repair_prompt(
            base_prompt=prompt,
            task_description=task,
            validation_errors=["candidate_prompt[1] must preserve required task action: turn on stove"],
            previous_output='{"subgoal_text": "place the moka pot on the stove"}',
        )

        self.assertIn("STOVE ACTION LOCK", prompt)
        self.assertIn("turning on the stove and placing the moka pot on the stove", prompt)
        self.assertIn("Do not make candidate 0 only 'turn on the stove'", prompt)
        self.assertIn("STOVE ACTION LOCK", repair_prompt)
        self.assertIn("Every stop_condition must include both 'stove is turned on'", repair_prompt)

    def test_deterministic_locked_field_fallback_preserves_relation_and_provenance(self) -> None:
        spec = importlib.util.spec_from_file_location("risk1b_generator_for_test", SCRIPT)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        args = SimpleNamespace(
            backend="transformers",
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            suite="libero_10",
            task_id=2,
            seed=1201,
            num_subgoals=5,
            task_description="put the moka pot on the stove",
            context_image="contact.png",
            context_json="context.json",
        )
        context = {
            "task_description": "put the moka pot on the stove",
            "provenance": {"actual_context": True},
        }
        raw_output = module.deterministic_locked_field_output(args, context)
        parsed = module.extract_json_payload(raw_output)
        payload = module.build_output_payload(
            args=args,
            prompt="prompt",
            context_summary=context,
            raw_output=raw_output,
            parsed=parsed,
            latency_ms=1,
            memory_mb=None,
            raw_output_path=Path("fallback.raw.txt"),
            generation_attempts=[{"attempt": 0, "valid": False, "errors": ["bad relation"]}],
            fallback={"strategy": "deterministic_locked_fields"},
        )

        self.assertTrue(payload["schema_validation"]["valid"])
        self.assertEqual(payload["provenance"], "deterministic_locked_fields_fallback")
        self.assertEqual(payload["fallback"]["strategy"], "deterministic_locked_fields")
        for record in payload["candidate_prompts"]:
            combined = " ".join(
                str(record[field])
                for field in ("subgoal_text", "target_object", "target_region_or_point", "stop_condition")
            ).lower()
            self.assertIn("moka", combined)
            self.assertIn("stove", combined)

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
