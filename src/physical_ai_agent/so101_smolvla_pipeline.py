from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


ClosedLoopPolicy = Literal["off", "periodic", "best_only", "best_or_periodic"]


@dataclass(frozen=True)
class SmolVLASO101Contract:
    """Stable SO101 fine-tuning contract for SmolVLA-style policies."""

    image_shape: tuple[int, int, int] = (3, 256, 256)
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    camera_keys: tuple[str, ...] = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )
    state_key: str = "observation.state"
    action_key: str = "action"
    state_dim: int = 6
    action_dim: int = 6
    chunk_size: int = 50
    train_n_action_steps: int = 50
    rollout_n_action_steps: int = 15
    num_steps: int = 10
    tokenizer_max_length: int = 48
    runtime_camera_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.camera1": "egocentric_cam",
            "observation.images.camera2": "wrist_cam",
            "observation.images.camera3": "wrist_cam duplicate",
        }
    )

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_shape"] = list(self.image_shape)
        value["resize_imgs_with_padding"] = list(self.resize_imgs_with_padding)
        value["camera_keys"] = list(self.camera_keys)
        return value


@dataclass(frozen=True)
class SO101AugmentationContract:
    """Sample-time augmentation contract.

    The implementation may run on CUDA/MPS after the batch is moved to device,
    but these probabilities are part of the reproducible training recipe and are
    therefore tested independently of Torch/LeRobot.
    """

    prefer_device_backends: tuple[str, ...] = ("cuda", "mps")
    image_color_jitter: bool = True
    image_sharpness_jitter: bool = True
    image_affine_jitter: bool = True
    image_affine_degrees: float = 5.0
    image_affine_translate: float = 0.05
    image_camera_dropout_prob: float = 0.0
    image_patch_dropout_prob: float = 0.0
    image_patch_mask_ratio: float = 0.15
    state_jitter_std: float = 0.003
    state_dropout_prob: float = 0.02
    state_dropout_keep_gripper: bool = True
    run_after_batch_to_device: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name in (
            "image_camera_dropout_prob",
            "image_patch_dropout_prob",
            "image_patch_mask_ratio",
            "state_dropout_prob",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value >= 1.0:
                errors.append(f"{name} must be in [0, 1), got {value}")
        if self.state_jitter_std < 0.0:
            errors.append(f"state_jitter_std must be non-negative, got {self.state_jitter_std}")
        if self.image_affine_degrees < 0.0:
            errors.append(f"image_affine_degrees must be non-negative, got {self.image_affine_degrees}")
        if self.image_affine_translate < 0.0:
            errors.append(f"image_affine_translate must be non-negative, got {self.image_affine_translate}")
        if "cuda" not in self.prefer_device_backends and "mps" not in self.prefer_device_backends:
            errors.append("prefer_device_backends must include cuda or mps")
        return errors


@dataclass(frozen=True)
class SO101DatasetManifest:
    dataset_id: str
    split: Literal["train", "validation", "test"]
    episodes: int
    frames: int
    source_episode_count: int
    target_expansion_factor: float
    expected_frames_per_episode: int | None = None
    min_frames_per_episode: int | None = None
    max_frames_per_episode: int | None = None
    image_shape: tuple[int, int, int] = (3, 256, 256)
    state_dim: int = 6
    action_dim: int = 6
    camera_keys: tuple[str, ...] = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )
    teacher_uses_privileged_state: bool = True
    student_observation_contract: str = "visual_policy_inputs_only"
    includes_recovery_or_off_nominal_states: bool = False
    sticky_grasp_allowed: bool = False
    generator: str = ""
    notes: str = ""

    @property
    def expected_min_episodes(self) -> int:
        return int(round(float(self.source_episode_count) * float(self.target_expansion_factor)))

    def validate(self, contract: SmolVLASO101Contract | None = None) -> list[str]:
        contract = contract or SmolVLASO101Contract()
        errors: list[str] = []
        if self.episodes < self.expected_min_episodes:
            errors.append(
                f"{self.dataset_id} has {self.episodes} episodes; expected at least "
                f"{self.expected_min_episodes} for expansion factor {self.target_expansion_factor}"
            )
        if self.frames < self.episodes:
            errors.append(f"{self.dataset_id} has fewer frames than episodes")
        if self.expected_frames_per_episode is not None:
            expected_frames = int(self.episodes) * int(self.expected_frames_per_episode)
            if int(self.frames) != expected_frames:
                errors.append(
                    f"{self.dataset_id} has {self.frames} frames; expected "
                    f"{expected_frames} from {self.expected_frames_per_episode} frames/episode"
                )
        if self.min_frames_per_episode is not None:
            min_frames = int(self.episodes) * int(self.min_frames_per_episode)
            if int(self.frames) < min_frames:
                errors.append(
                    f"{self.dataset_id} has {self.frames} frames; expected at least "
                    f"{min_frames} from {self.min_frames_per_episode} min frames/episode"
                )
        if self.max_frames_per_episode is not None:
            max_frames = int(self.episodes) * int(self.max_frames_per_episode)
            if int(self.frames) > max_frames:
                errors.append(
                    f"{self.dataset_id} has {self.frames} frames; expected at most "
                    f"{max_frames} from {self.max_frames_per_episode} max frames/episode"
                )
        if tuple(self.image_shape) != contract.image_shape:
            errors.append(f"image_shape {self.image_shape} != required {contract.image_shape}")
        if self.state_dim != contract.state_dim:
            errors.append(f"state_dim {self.state_dim} != required {contract.state_dim}")
        if self.action_dim != contract.action_dim:
            errors.append(f"action_dim {self.action_dim} != required {contract.action_dim}")
        if tuple(self.camera_keys) != contract.camera_keys:
            errors.append(f"camera_keys {self.camera_keys} != required {contract.camera_keys}")
        if self.sticky_grasp_allowed:
            errors.append("sticky_grasp_allowed must be false for contact-realistic training data")
        if self.teacher_uses_privileged_state and not self.includes_recovery_or_off_nominal_states:
            errors.append(
                "privileged teacher dataset must include recovery/off-nominal states "
                "so the visual student sees closed-loop distribution drift"
            )
        return errors

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_shape"] = list(self.image_shape)
        value["camera_keys"] = list(self.camera_keys)
        value["expected_min_episodes"] = self.expected_min_episodes
        value["validation_errors"] = self.validate()
        return value


@dataclass(frozen=True)
class SO101TrainingSchedule:
    validation_every_checkpoints: int = 1
    closed_loop_policy: ClosedLoopPolicy = "best_only"
    closed_loop_every_epochs: int = 10
    steps_per_epoch: int = 149
    stop_on_overfit: bool = True
    overfit_patience_checkpoints: int = 3
    overfit_min_delta: float = 0.0

    def interval_steps(self) -> int:
        return int(self.closed_loop_every_epochs) * int(self.steps_per_epoch)


def validate_smolvla_train_config(
    config: dict[str, Any],
    contract: SmolVLASO101Contract | None = None,
) -> list[str]:
    contract = contract or SmolVLASO101Contract()
    policy = config.get("policy") or config
    errors: list[str] = []

    input_features = policy.get("input_features") or {}
    for camera_key in contract.camera_keys:
        shape = (input_features.get(camera_key) or {}).get("shape")
        if tuple(shape or ()) != contract.image_shape:
            errors.append(f"{camera_key}.shape {shape} != {list(contract.image_shape)}")
    state_shape = (input_features.get(contract.state_key) or {}).get("shape")
    if tuple(state_shape or ()) != (contract.state_dim,):
        errors.append(f"{contract.state_key}.shape {state_shape} != [{contract.state_dim}]")
    action_shape = ((policy.get("output_features") or {}).get(contract.action_key) or {}).get("shape")
    if tuple(action_shape or ()) != (contract.action_dim,):
        errors.append(f"{contract.action_key}.shape {action_shape} != [{contract.action_dim}]")

    if tuple(policy.get("resize_imgs_with_padding") or ()) != contract.resize_imgs_with_padding:
        errors.append(
            "resize_imgs_with_padding "
            f"{policy.get('resize_imgs_with_padding')} != {list(contract.resize_imgs_with_padding)}"
        )
    for key, expected in (
        ("chunk_size", contract.chunk_size),
        ("n_action_steps", contract.train_n_action_steps),
        ("num_steps", contract.num_steps),
        ("tokenizer_max_length", contract.tokenizer_max_length),
    ):
        if policy.get(key) != expected:
            errors.append(f"policy.{key} {policy.get(key)} != {expected}")
    if policy.get("pretrained_path") not in {"lerobot/smolvla_base", "lerobot/smolvla_libero"}:
        errors.append(f"unexpected pretrained_path {policy.get('pretrained_path')}")
    return errors


def is_best_validation_checkpoint(rows: list[dict[str, Any]], checkpoint: str) -> bool:
    valid_rows = [
        row for row in rows if row.get("loss") is not None and row.get("checkpoint") is not None
    ]
    if not valid_rows:
        return False
    best = min(valid_rows, key=lambda row: float(row["loss"]))
    return str(best.get("checkpoint")) == str(checkpoint)


def should_run_closed_loop(
    *,
    schedule: SO101TrainingSchedule,
    checkpoint: str,
    validation_rows: list[dict[str, Any]],
    closed_loop_rows: list[dict[str, Any]],
) -> bool:
    if schedule.closed_loop_policy == "off" or _closed_loop_recorded(closed_loop_rows, checkpoint):
        return False
    step = _checkpoint_to_step(checkpoint)
    if step is None:
        return False
    is_best = is_best_validation_checkpoint(validation_rows, checkpoint)
    is_periodic = schedule.interval_steps() > 0 and step % schedule.interval_steps() == 0
    if schedule.closed_loop_policy == "best_only":
        return is_best
    if schedule.closed_loop_policy == "periodic":
        return is_periodic
    if schedule.closed_loop_policy == "best_or_periodic":
        return is_best or is_periodic
    return False


def detect_overfit_stop(
    validation_rows: list[dict[str, Any]],
    *,
    patience_checkpoints: int,
    min_delta: float = 0.0,
) -> dict[str, Any]:
    rows = [
        row for row in validation_rows if row.get("loss") is not None and row.get("checkpoint") is not None
    ]
    if patience_checkpoints <= 0 or len(rows) <= patience_checkpoints:
        return {"should_stop": False, "reason": "insufficient_validation_history"}
    best_index, best_row = min(enumerate(rows), key=lambda pair: float(pair[1]["loss"]))
    trailing = rows[best_index + 1 :]
    if len(trailing) < patience_checkpoints:
        return {"should_stop": False, "reason": "patience_not_exceeded", "best": best_row}
    recent = trailing[-patience_checkpoints:]
    threshold = float(best_row["loss"]) + float(min_delta)
    if all(float(row["loss"]) > threshold for row in recent):
        return {
            "should_stop": True,
            "reason": "validation_loss_worse_than_best",
            "best": best_row,
            "recent": recent,
            "threshold": threshold,
        }
    return {"should_stop": False, "reason": "recent_validation_improved_or_tied", "best": best_row}


def write_dataset_manifest(path: Path, manifest: SO101DatasetManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_dataset_manifest(path: Path) -> SO101DatasetManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("expected_min_episodes", None)
    raw.pop("validation_errors", None)
    if "image_shape" in raw:
        raw["image_shape"] = tuple(raw["image_shape"])
    if "camera_keys" in raw:
        raw["camera_keys"] = tuple(raw["camera_keys"])
    return SO101DatasetManifest(**raw)


def _closed_loop_recorded(rows: list[dict[str, Any]], checkpoint: str) -> bool:
    return any(str(row.get("checkpoint")) == str(checkpoint) for row in rows)


def _checkpoint_to_step(checkpoint: str) -> int | None:
    try:
        return int(str(checkpoint))
    except ValueError:
        return None
