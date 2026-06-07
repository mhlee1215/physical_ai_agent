from unittest import TestCase

from scripts.run_libero_in_episode_smolvla_instrumented import (
    clamp_action_norm,
    format_intervention_type,
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
