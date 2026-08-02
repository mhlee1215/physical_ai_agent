from __future__ import annotations

from typing import Any


LOSS_OVERVIEW_PATTERNS = (
    r"^train/loss$",
    r"^val/loss$",
)

TRAIN_OBJECTIVE_LOSS_PATTERNS = (
    r"^train/action_loss$",
    r"^train/loss_unweighted$",
    r"^train/valid_mask_loss$",
    r"^train/action_chunk_consistency_loss$",
    r"^train/action_overlap_consistency_loss$",
    r"^train/action_requery_consistency_loss$",
    r"^train/action_smoothness_loss$",
    r"^train/action_wrist_roll_circular_loss$",
    r"^train/visual_servo_loss$",
)

TRAIN_DATASET_LOSS_PATTERNS = (
    r"^train/datasets/[^/]+/loss$",
)

VALIDATION_DATASET_LOSS_PATTERNS = (
    r"^val/datasets/[^/]+/loss$",
)

VALIDATION_OBJECTIVE_LOSS_PATTERNS = (
    r"^val(?:/datasets/[^/]+)?/action_loss$",
    r"^val(?:/datasets/[^/]+)?/loss_unweighted$",
    r"^val(?:/datasets/[^/]+)?/valid_mask_loss$",
    r"^val(?:/datasets/[^/]+)?/action_chunk_consistency_loss$",
    r"^val(?:/datasets/[^/]+)?/action_overlap_consistency_loss$",
    r"^val(?:/datasets/[^/]+)?/action_requery_consistency_loss$",
    r"^val(?:/datasets/[^/]+)?/action_smoothness_loss$",
    r"^val(?:/datasets/[^/]+)?/action_wrist_roll_circular_loss$",
    r"^val(?:/datasets/[^/]+)?/visual_servo_loss$",
)

_REPEATED_CONFIG_METRICS = frozenset(
    {
        "action_chunk_consistency_steps",
        "action_chunk_consistency_weight",
        "action_delta_loss_weight",
        "action_gripper_transition_loss_weight",
        "action_overlap_consistency_horizon",
        "action_overlap_consistency_offset",
        "action_overlap_consistency_weight",
        "action_requery_consistency_horizon",
        "action_requery_consistency_offset",
        "action_requery_consistency_weight",
        "action_smoothness_dims",
        "action_smoothness_include_gripper",
        "action_smoothness_loss_weight",
        "action_terminal_loss_steps",
        "action_terminal_loss_weight",
        "action_wrist_roll_circular_loss_weight",
        "action_wrist_roll_joint_index",
        "action_wrist_roll_period_normalized",
        "loss_prefix_steps",
        "loss_prefix_weight",
        "valid_mask_loss_weight",
        "visual_servo_loss_weight",
    }
)


def so101_loss_custom_scalars_layout() -> dict[str, dict[str, list[Any]]]:
    """Return the stable TensorBoard loss dashboard layout.

    Patterns intentionally omit duplicate ``important/*`` aliases, repeated
    configuration values, and the old ``losses_after_*`` diagnostics.
    """

    return {
        "Loss": {
            "Train vs validation": ["Multiline", list(LOSS_OVERVIEW_PATTERNS)],
            "Training objectives": ["Multiline", list(TRAIN_OBJECTIVE_LOSS_PATTERNS)],
            "Training datasets": ["Multiline", list(TRAIN_DATASET_LOSS_PATTERNS)],
            "Validation datasets": ["Multiline", list(VALIDATION_DATASET_LOSS_PATTERNS)],
            "Validation objectives": ["Multiline", list(VALIDATION_OBJECTIVE_LOSS_PATTERNS)],
        }
    }


def should_log_repeated_scalar_metric(key: str) -> bool:
    """Keep measurements while dropping per-step copies of static config/debug values."""

    if key.startswith("losses_after_"):
        return False
    return key not in _REPEATED_CONFIG_METRICS


def add_so101_loss_custom_scalars(writer: Any) -> None:
    add_custom_scalars = getattr(writer, "add_custom_scalars", None)
    if callable(add_custom_scalars):
        add_custom_scalars(so101_loss_custom_scalars_layout())
