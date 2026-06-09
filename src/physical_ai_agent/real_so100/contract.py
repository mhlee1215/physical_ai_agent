from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


REAL_POLICY_CAMERA_INDEXES = (0, 1)
REAL_OBSERVER_CAMERA_INDEX = 3
LEGACY_CAMERA_INDEXES = (2,)
SIM_POLICY_CAMERA_NAMES = ("wrist_cam", "egocentric_cam")
SIM_DEBUG_CAMERA_NAMES = ("top_down",)
REAL_POLICY_CAMERA_FEATURE_KEYS = (
    "observation.images.camera_0",
    "observation.images.camera_1",
)
SIM_POLICY_CAMERA_FEATURE_KEYS = tuple(f"observation.images.{name}" for name in SIM_POLICY_CAMERA_NAMES)
SO100_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class CameraRoute:
    index: int | None
    name: str
    role: str
    policy_input: bool
    observer_input: bool
    feature_key: str | None
    notes: str


@dataclass(frozen=True)
class SO100ExecutionGate:
    physical_execution_allowed: bool
    send_action_called: bool
    reason: str
    required_before_motion: tuple[str, ...]


@dataclass(frozen=True)
class SO100Contract:
    robot: str
    policy_camera_routes: tuple[CameraRoute, ...]
    observer_camera_routes: tuple[CameraRoute, ...]
    debug_camera_routes: tuple[CameraRoute, ...]
    legacy_camera_indexes: tuple[int, ...]
    joint_order: tuple[str, ...]
    default_action_chunk_steps: int
    calibration_manifest: str
    calibration_file: str
    home_pose_file: str
    safety_notes: tuple[str, ...]

    @property
    def policy_camera_indexes(self) -> tuple[int, ...]:
        return tuple(route.index for route in self.policy_camera_routes if route.index is not None)

    @property
    def policy_feature_keys(self) -> tuple[str, ...]:
        return tuple(route.feature_key for route in self.policy_camera_routes if route.feature_key)


def current_so100_contract() -> SO100Contract:
    return SO100Contract(
        robot="SO-100 follower",
        policy_camera_routes=(
            CameraRoute(
                index=0,
                name="camera_0",
                role="policy_wrist",
                policy_input=True,
                observer_input=False,
                feature_key=REAL_POLICY_CAMERA_FEATURE_KEYS[0],
                notes="Innomaker U20CAM; SmolVLA policy input only.",
            ),
            CameraRoute(
                index=1,
                name="camera_1",
                role="policy_context",
                policy_input=True,
                observer_input=False,
                feature_key=REAL_POLICY_CAMERA_FEATURE_KEYS[1],
                notes="Innomaker U20CAM; object/context policy input.",
            ),
        ),
        observer_camera_routes=(
            CameraRoute(
                index=3,
                name="camera_3",
                role="codex_observer",
                policy_input=False,
                observer_input=True,
                feature_key=None,
                notes="iPhone observer/debug evidence. Never feed to SmolVLA.",
            ),
        ),
        debug_camera_routes=(
            CameraRoute(
                index=None,
                name="top_down",
                role="sim_debug",
                policy_input=False,
                observer_input=True,
                feature_key=None,
                notes="Simulation-only debug/observer view.",
            ),
        ),
        legacy_camera_indexes=LEGACY_CAMERA_INDEXES,
        joint_order=SO100_JOINT_ORDER,
        default_action_chunk_steps=10,
        calibration_manifest="_workspace/real_so100/calibration_manifest.json",
        calibration_file="_workspace/real_so100/calibration/so100_local.json",
        home_pose_file="_workspace/real_so100/home_pose/canonical_home_pose_2026_06_07.json",
        safety_notes=(
            "Do not feed observer camera index 3 to SmolVLA.",
            "Do not execute physical motion without calibration, clipping, observer evidence, and home-return plan.",
            "Do not convert raw SmolVLA chunk tensors to motor ticks without verified postprocessing or unnormalization.",
            "Return to canonical home pose and disable torque after every completed movement task.",
        ),
    )


def sim_policy_feature_keys() -> list[str]:
    return list(SIM_POLICY_CAMERA_FEATURE_KEYS)


def classify_sim_camera_names(camera_names: tuple[str, ...] | list[str]) -> tuple[list[str], list[str]]:
    policy = [name for name in SIM_POLICY_CAMERA_NAMES if name in camera_names]
    debug = [name for name in SIM_DEBUG_CAMERA_NAMES if name in camera_names]
    return policy, debug


def validate_policy_camera_routes(routes: tuple[CameraRoute, ...] | list[CameraRoute]) -> list[str]:
    errors = []
    for route in routes:
        if route.observer_input and route.policy_input:
            errors.append(f"{route.name} cannot be both observer and policy input")
        if route.index == REAL_OBSERVER_CAMERA_INDEX and route.policy_input:
            errors.append("camera index 3 is observer/debug only and must not be a policy input")
        if route.index in LEGACY_CAMERA_INDEXES and route.policy_input:
            errors.append(f"legacy camera index {route.index} must not be used as a current policy input")
    return errors


def build_no_actuation_gate(reason: str) -> SO100ExecutionGate:
    return SO100ExecutionGate(
        physical_execution_allowed=False,
        send_action_called=False,
        reason=reason,
        required_before_motion=(
            "workspace_clear_user_confirmed",
            "calibration_loaded",
            "observer_camera_3_evidence_available",
            "smolvla_postprocessor_or_unnormalization_verified",
            "joint_order_confirmed",
            "gripper_semantics_confirmed",
            "targets_clipped_to_calibration",
            "home_return_and_torque_off_plan_ready",
        ),
    )


def write_contract_artifact(path: Path, contract: SO100Contract | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(contract or current_so100_contract()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
