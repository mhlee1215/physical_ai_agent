from unittest import TestCase

from scripts.run_libero_in_episode_smolvla_instrumented import (
    clamp_action_norm,
    distance,
    format_intervention_type,
    semantic_no_progress_trigger,
    should_trigger_verifier,
)


class LiberoInEpisodeSmolVLAInstrumentedTest(TestCase):
    def test_fixed_step_trigger(self) -> None:
        triggered, reason = should_trigger_verifier(
            step=3,
            action_norm=0.2,
            intervention_step=3,
            trigger_mode="fixed_step",
            action_norm_threshold=1.0,
            is_done=False,
        )

        self.assertTrue(triggered)
        self.assertEqual(reason, "fixed_step_threshold")

    def test_action_norm_trigger(self) -> None:
        triggered, reason = should_trigger_verifier(
            step=2,
            action_norm=1.2,
            intervention_step=10,
            trigger_mode="action_norm_threshold",
            action_norm_threshold=1.0,
            is_done=False,
        )

        self.assertTrue(triggered)
        self.assertEqual(reason, "action_norm_threshold")

    def test_clamp_action_norm_rejects_invalid_norm_without_torch(self) -> None:
        with self.assertRaises(ValueError):
            clamp_action_norm(object(), 0.0)

    def test_format_intervention_type_records_parameters(self) -> None:
        self.assertEqual(
            format_intervention_type(
                mode="clamp",
                intervention_scale=0.5,
                action_clamp_norm=0.75,
                smooth_alpha=0.2,
            ),
            "clamp_action_norm_0.75",
        )

    def test_format_policy_reset_intervention_type(self) -> None:
        self.assertEqual(
            format_intervention_type(
                mode="policy_reset",
                intervention_scale=1.0,
                action_clamp_norm=1.0,
                smooth_alpha=0.5,
            ),
            "policy_reset_reselect_action",
        )

    def test_format_semantic_push_intervention_type(self) -> None:
        self.assertEqual(
            format_intervention_type(
                mode="semantic_push_receptacle",
                intervention_scale=1.0,
                action_clamp_norm=1.0,
                smooth_alpha=0.5,
            ),
            "semantic_push_receptacle",
        )

    def test_semantic_no_progress_trigger(self) -> None:
        history = [{"target_pos": [0.0, 0.0, 0.0]} for _ in range(6)]

        triggered, reason = semantic_no_progress_trigger(
            step=5,
            semantic_history=history,
            min_step=3,
            window=3,
            progress_threshold=0.01,
        )

        self.assertTrue(triggered)
        self.assertIn("semantic_no_target_progress", reason)

    def test_semantic_progress_does_not_trigger(self) -> None:
        history = [
            {"target_pos": [0.0, 0.0, 0.0]},
            {"target_pos": [0.0, 0.0, 0.0]},
            {"target_pos": [0.0, 0.0, 0.0]},
            {"target_pos": [0.1, 0.0, 0.0]},
        ]

        triggered, reason = semantic_no_progress_trigger(
            step=4,
            semantic_history=history,
            min_step=3,
            window=3,
            progress_threshold=0.01,
        )

        self.assertFalse(triggered)
        self.assertIn("semantic_target_progress", reason)

    def test_distance_handles_missing_vectors(self) -> None:
        self.assertIsNone(distance(None, [0.0, 0.0, 0.0]))
        self.assertEqual(distance([0.0, 0.0, 0.0], [0.0, 3.0, 4.0]), 5.0)
