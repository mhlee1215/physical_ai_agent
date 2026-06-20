import unittest

from physical_ai_agent.imagine_then_act.sequential_substeps import (
    SubstepVerifierSpec,
    normalize_substep_plan,
    should_retry_or_advance,
    task0_primitive_relaxed_substep_plan,
    task0_sequential_substep_plan,
    verify_substep_completion,
)


class SequentialSubstepsTest(unittest.TestCase):
    def test_task0_plan_has_two_required_object_in_target_substeps(self) -> None:
        substeps = normalize_substep_plan(task0_sequential_substep_plan())

        self.assertEqual([substep.substep_id for substep in substeps], [
            "task0_step01_alphabet_soup_to_basket",
            "task0_step02_tomato_sauce_to_basket",
        ])
        self.assertTrue(all(substep.required for substep in substeps))
        self.assertEqual(substeps[0].target_object_key, "alphabet_soup_1_pos")
        self.assertEqual(substeps[1].target_object_key, "tomato_sauce_1_pos")
        self.assertEqual(substeps[0].receptacle_object_key, "basket_1_pos")

    def test_object_in_target_pass_requires_threshold(self) -> None:
        spec = SubstepVerifierSpec(
            substep_id="s1",
            prompt="Put the object in the basket.",
            verifier_type="object_in_target",
            target_object_key="obj_1_pos",
            receptacle_object_key="basket_1_pos",
            distance_threshold=0.08,
        )

        decision = verify_substep_completion(spec, {"target_to_receptacle_dist": 0.079})

        self.assertEqual(decision.status, "pass")
        self.assertIn("le_0.080000", decision.reason)

    def test_progress_without_completion_is_unknown_not_pass(self) -> None:
        spec = SubstepVerifierSpec(
            substep_id="s1",
            prompt="Put the object in the basket.",
            verifier_type="object_in_target",
            target_object_key="obj_1_pos",
            receptacle_object_key="basket_1_pos",
            distance_threshold=0.08,
            progress_threshold=0.015,
        )

        decision = verify_substep_completion(
            spec,
            {"target_to_receptacle_dist": 0.11},
            previous_semantic_state={"target_to_receptacle_dist": 0.14},
        )

        self.assertEqual(decision.status, "unknown")
        self.assertEqual(decision.score, 0.5)
        self.assertIn("progress_without_completion", decision.reason)

    def test_relaxed_progress_can_pass_without_final_distance(self) -> None:
        spec = SubstepVerifierSpec(
            substep_id="s1",
            prompt="Put the object in the basket.",
            verifier_type="object_in_target",
            target_object_key="obj_1_pos",
            receptacle_object_key="basket_1_pos",
            distance_threshold=0.08,
            progress_threshold=0.015,
            pass_on_progress=True,
        )

        decision = verify_substep_completion(
            spec,
            {"target_to_receptacle_dist": 0.11},
            previous_semantic_state={"target_to_receptacle_dist": 0.14},
        )

        self.assertEqual(decision.status, "pass")
        self.assertEqual(decision.score, 0.6)
        self.assertIn("target_receptacle_progress", decision.reason)

    def test_relaxed_eef_progress_can_pass_without_object_motion(self) -> None:
        spec = SubstepVerifierSpec(
            substep_id="s1",
            prompt="Move to the object.",
            verifier_type="object_in_target",
            target_object_key="obj_1_pos",
            receptacle_object_key="basket_1_pos",
            distance_threshold=0.08,
            eef_distance_threshold=0.2,
            eef_progress_threshold=0.03,
            pass_on_progress=True,
        )

        decision = verify_substep_completion(
            spec,
            {"target_to_receptacle_dist": 0.4, "eef_to_target_dist": 0.25},
            previous_semantic_state={"target_to_receptacle_dist": 0.4, "eef_to_target_dist": 0.29},
        )

        self.assertEqual(decision.status, "pass")
        self.assertEqual(decision.score, 0.4)
        self.assertIn("eef_target_progress", decision.reason)

    def test_relaxed_task0_plan_marks_progress_policy(self) -> None:
        substeps = normalize_substep_plan(task0_sequential_substep_plan(relaxed_progress=True))

        self.assertTrue(all(substep.pass_on_progress for substep in substeps))
        self.assertTrue(all(substep.max_attempts == 3 for substep in substeps))
        self.assertTrue(all(substep.distance_threshold == 0.12 for substep in substeps))

    def test_primitive_relaxed_plan_has_reach_lift_place_substeps(self) -> None:
        substeps = normalize_substep_plan(task0_primitive_relaxed_substep_plan())

        self.assertEqual(len(substeps), 6)
        self.assertEqual(substeps[0].verifier_type, "eef_near_object")
        self.assertEqual(substeps[1].verifier_type, "object_lifted")
        self.assertEqual(substeps[2].verifier_type, "object_in_target")
        self.assertEqual(substeps[3].target_object_key, "tomato_sauce_1_pos")

    def test_eef_near_object_can_pass_on_progress(self) -> None:
        spec = SubstepVerifierSpec(
            substep_id="reach",
            prompt="Move to object.",
            verifier_type="eef_near_object",
            target_object_key="obj_1_pos",
            eef_distance_threshold=0.18,
            eef_progress_threshold=0.025,
            pass_on_progress=True,
        )

        decision = verify_substep_completion(
            spec,
            {"eef_to_target_dist": 0.25},
            previous_semantic_state={"eef_to_target_dist": 0.29},
        )

        self.assertEqual(decision.status, "pass")
        self.assertIn("eef_target_progress", decision.reason)

    def test_object_lifted_passes_on_z_progress(self) -> None:
        spec = SubstepVerifierSpec(
            substep_id="lift",
            prompt="Lift object.",
            verifier_type="object_lifted",
            target_object_key="obj_1_pos",
            progress_threshold=0.015,
            pass_on_progress=True,
        )

        decision = verify_substep_completion(
            spec,
            {"target_pos": [0.0, 0.0, 0.45]},
            previous_semantic_state={"target_pos": [0.0, 0.0, 0.43]},
        )

        self.assertEqual(decision.status, "pass")
        self.assertIn("target_lift_progress", decision.reason)

    def test_missing_state_is_unknown_not_pass(self) -> None:
        spec = SubstepVerifierSpec(
            substep_id="s1",
            prompt="Put the object in the basket.",
            verifier_type="object_in_target",
            target_object_key="obj_1_pos",
            receptacle_object_key="basket_1_pos",
        )

        decision = verify_substep_completion(spec, {})

        self.assertEqual(decision.status, "unknown")
        self.assertIsNone(decision.score)

    def test_required_max_attempts_never_counts_as_pass(self) -> None:
        spec = SubstepVerifierSpec(
            substep_id="s1",
            prompt="Put the object in the basket.",
            verifier_type="object_in_target",
            target_object_key="obj_1_pos",
            receptacle_object_key="basket_1_pos",
            max_attempts=2,
        )
        decision = verify_substep_completion(spec, {"target_to_receptacle_dist": 0.2})

        policy = should_retry_or_advance(
            decision,
            attempt=2,
            max_attempts=2,
            required=True,
        )

        self.assertEqual(policy["next"], "advance_with_required_unmet")
        self.assertFalse(policy["counts_as_pass"])


if __name__ == "__main__":
    unittest.main()
