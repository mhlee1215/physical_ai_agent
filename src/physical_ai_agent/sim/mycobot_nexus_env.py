from __future__ import annotations

import json
import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MYCOBOT_MODEL_RELATIVE_PATH = Path("xml/mycobot_280jn_mujoco.xml")
OFFICIAL_GRIPPER_MESH_RELATIVE_PATH = Path("mycobot_description/urdf/parallel_gripper")
OFFICIAL_320_URDF_RELATIVE_PATH = Path(
    "mycobot_description/urdf/mycobot_320_m5_2022/new_mycobot_pro_320_m5_2022_gripper.urdf"
)
OFFICIAL_320_ADAPTIVE_URDF_RELATIVE_PATH = Path(
    "mycobot_description/urdf/mycobot_320_m5_2022/mycobot_320_m5_2022_adaptive_gripper.urdf"
)
OFFICIAL_320_MESH_RELATIVE_PATH = Path("mycobot_description/urdf/mycobot_320_m5_2022")
OFFICIAL_320_ADAPTIVE_GRIPPER_MESH_RELATIVE_PATH = Path(
    "mycobot_description/urdf/pro_adaptive_gripper"
)
OFFICIAL_280_PI_URDF_RELATIVE_PATH = Path(
    "mycobot_description/urdf/mycobot_280_pi/mycobot_280_pi.urdf"
)
OFFICIAL_280_PI_MESH_RELATIVE_PATH = Path("mycobot_description/urdf/mycobot_280_pi")
OFFICIAL_ADAPTIVE_GRIPPER_URDF_RELATIVE_PATH = Path(
    "mycobot_description/urdf/adaptive_gripper/mycobot_adaptive_gripper.urdf"
)
OFFICIAL_ADAPTIVE_GRIPPER_MESH_RELATIVE_PATH = Path(
    "mycobot_description/urdf/adaptive_gripper"
)
OFFICIAL_GRIPPER_MESH_NAMES = [
    "gripper_base",
    "gripper_left",
    "gripper_right",
]
OFFICIAL_320_LINK_NAMES = [
    "base",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "gripper_base",
    "gripper_left1",
    "gripper_left2",
    "gripper_left3",
    "gripper_right1",
    "gripper_right2",
    "gripper_right3",
]
OFFICIAL_320_ARM_LINK_NAMES = OFFICIAL_320_LINK_NAMES[:7]
OFFICIAL_320_GRIPPER_LINK_NAMES = OFFICIAL_320_LINK_NAMES[7:]
OFFICIAL_ADAPTIVE_GRIPPER_LINK_NAMES = [
    "gripper_base",
    "gripper_left1",
    "gripper_left2",
    "gripper_left3",
    "gripper_right1",
    "gripper_right2",
    "gripper_right3",
]
MYCOBOT_MODEL_JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint7_to_joint6",
]
MYCOBOT_320_MODEL_JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]
MYCOBOT_TEACHER_JOINT_NAMES = [
    *MYCOBOT_MODEL_JOINT_NAMES,
    "gripper_controller",
]
SYNTHETIC_GRIPPER_JOINT_NAMES = ["left_finger_slide", "right_finger_slide"]
OFFICIAL_GRIPPER_JOINT_NAMES = [
    "gripper_controller",
    "gripper_base_to_gripper_left",
]
OFFICIAL_GRIPPER_MIMIC = {
    "gripper_controller": 1.0,
    "gripper_base_to_gripper_left": -1.0,
}
OFFICIAL_320_GRIPPER_JOINT_NAMES = [
    "gripper_controller",
    "gripper_base_to_gripper_left2",
    "gripper_left3_to_gripper_left1",
    "gripper_base_to_gripper_right3",
    "gripper_base_to_gripper_right2",
    "gripper_right3_to_gripper_right1",
]
OFFICIAL_320_GRIPPER_MIMIC = {
    "gripper_controller": 1.0,
    "gripper_base_to_gripper_left2": 1.0,
    "gripper_left3_to_gripper_left1": -1.0,
    "gripper_base_to_gripper_right3": -1.0,
    "gripper_base_to_gripper_right2": -1.0,
    "gripper_right3_to_gripper_right1": 1.0,
}
TASK_CUBE_BODY = "task_cube_body"
TASK_CUBE_GEOM = "task_cube"
TASK_CUBE_POS = (-0.25, -0.13, 0.023)
TASK_CUBE_HALF_SIZE = 0.015
TCP_SITE = "mycobot_tcp_site"
MODEL_PROFILE_280_JN = "280-jn"
MODEL_PROFILE_320_GRIPPER = "320-m5-2022-gripper"
MODEL_PROFILE_320_ADAPTIVE_GRIPPER = "320-m5-2022-adaptive-gripper"
MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER = "280-pi-adaptive-gripper"
ADAPTIVE_LEFT_FINGER_PAD_PARENT = "gripper_left1"
ADAPTIVE_RIGHT_FINGER_PAD_PARENT = "gripper_right1"
ADAPTIVE_LEFT_FINGER_PAD_POS = (0.00093, 0.04795, 0.00381)
ADAPTIVE_RIGHT_FINGER_PAD_POS = (-0.00567, 0.04202, 0.00390)
ADAPTIVE_FINGER_PAD_SIZE = (0.014, 0.006, 0.006)
ADAPTIVE_FINGER_PAD_EULER = (0.0, 0.0, 0.0)
ADAPTIVE_FINGER_PAD_FRICTION = (80.0, 8.0, 8.0)
ADAPTIVE_FINGER_PAD_CONDIM = 6
ADAPTIVE_280_LEFT_FINGER_PAD_POS = ADAPTIVE_LEFT_FINGER_PAD_POS
ADAPTIVE_280_RIGHT_FINGER_PAD_POS = ADAPTIVE_RIGHT_FINGER_PAD_POS
ADAPTIVE_280_FINGER_PAD_SIZE = ADAPTIVE_FINGER_PAD_SIZE
ADAPTIVE_280_FINGER_PAD_FRICTION = ADAPTIVE_FINGER_PAD_FRICTION
ADAPTIVE_GATE7_TABLE_ARM_QPOS = (
    -0.655675253325397,
    0.810882674174539,
    0.8439829636723133,
    0.7977011570207353,
    -1.4127876896491962,
    -0.23321018739728538,
)
ADAPTIVE_GATE8_LIFT_ARM_QPOS = (
    -1.0817487541686395,
    1.2650681863929847,
    0.5655833702793307,
    1.2470797343323703,
    -1.3210836913320143,
    -0.6780218616382665,
)
ADAPTIVE_280_PI_GATE7_TABLE_ARM_QPOS = (
    2.209046727154287,
    1.7708220975518412,
    0.4366852271155825,
    -2.4021653896480166,
    -0.7968371300189514,
    0.387837875579959,
)
ADAPTIVE_280_PI_GATE8_LIFT_ARM_QPOS = (
    2.209046727154287,
    1.6908220975518411,
    0.5866852271155825,
    -2.1521653896480166,
    -0.7968371300189514,
    0.387837875579959,
)
ADAPTIVE_GATE7_CUBE_OFFSET = (0.0, 0.0, 0.0)
ADAPTIVE_280_PI_GATE7_CUBE_OFFSET = (-0.006, -0.012, 0.0)


@dataclass(frozen=True)
class MyCobotNexusConfig:
    asset_root: Path
    work_dir: Path
    official_gripper_root: Path | None = None
    model_profile: str = "280-jn"
    width: int = 640
    height: int = 360
    control_alpha: float = 0.35
    teacher_grasp_attachment_enabled: bool = True


@dataclass(frozen=True)
class MyCobotNexusStep:
    step: int
    observation: list[float]
    action: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class MyCobotNexusSmokeResult:
    status: str
    policy: str
    steps: int
    observation_dim: int
    action_dim: int
    initial_tcp_to_cube_dist: float
    final_tcp_to_cube_dist: float
    min_tcp_to_cube_dist: float
    approach_improved: bool
    initial_cube_z: float
    final_cube_z: float
    cube_lifted: bool
    gripper_cube_contacts: int
    gripper_cube_contact_pads: int
    grasp_success: bool
    trace_path: str
    frame_path: str
    report_path: str
    scene_path: str


@dataclass(frozen=True)
class MyCobotAdaptiveStaticContactResult:
    status: str
    steps: int
    placement_gripper_command: float
    final_gripper_command: float
    required_sustained_steps: int
    best_sustained_contact_steps: int
    gripper_cube_contacts: int
    gripper_cube_contact_pads: int
    initial_cube_position: list[float]
    final_cube_position: list[float]
    trace_path: str
    frame_path: str
    report_path: str
    scene_path: str


@dataclass(frozen=True)
class MyCobotAdaptiveGraspLiftResult:
    status: str
    steps: int
    close_gripper_command: float
    required_close_sustained_steps: int
    required_lift_sustained_steps: int
    required_final_lift: float
    close_best_sustained_contact_steps: int
    lift_best_sustained_contact_steps: int
    lift_two_pad_contact_steps: int
    final_gripper_cube_contact_pads: int
    final_gripper_cube_contacts: int
    initial_cube_position: list[float]
    final_cube_position: list[float]
    final_cube_lift: float
    trace_path: str
    frame_path: str
    report_path: str
    scene_path: str
    teacher_attachment_enabled: bool


class MyCobotNexusEnv:
    """Mac-local myCobot task environment with a Nexus-style MuJoCo world.

    This is a simulation POC, not only a renderer: it owns MuJoCo model/data,
    exposes reset/step/render, and records task state around a visible cube.
    The official myCobot MJCF has no actuator section, so step() currently uses
    deterministic qpos-target stepping. That keeps the env executable while the
    current cube-grasp POC uses an explicit teacher attachment proxy after
    gripper contact. The next POC can add calibrated actuators and contact-based
    force-closure success.
    """

    def __init__(self, config: MyCobotNexusConfig) -> None:
        try:
            import mujoco
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "MyCobotNexusEnv requires mujoco. Install a MuJoCo-capable Python "
                "or run the dry contract smoke first."
            ) from exc

        self._mujoco = mujoco
        self.config = config
        self.asset_root = config.asset_root.expanduser()
        self.work_dir = config.work_dir.expanduser()
        self.model_path = self.asset_root / MYCOBOT_MODEL_RELATIVE_PATH
        if config.model_profile in {
            MODEL_PROFILE_320_GRIPPER,
            MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
            MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
        }:
            self.model_path = Path("")
        elif not self.model_path.exists():
            raise FileNotFoundError(
                f"missing myCobot MuJoCo model: {self.model_path}. "
                "Clone https://github.com/elephantrobotics/mycobot_mujoco and pass --asset-root."
            )
        self.scene_path = self.work_dir / "mycobot_nexus_scene.xml"
        build_mycobot_nexus_scene_model(
            model_path=self.model_path,
            scene_path=self.scene_path,
            official_gripper_root=config.official_gripper_root,
            model_profile=config.model_profile,
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._step = 0
        self._grasp_attached = False
        self._cube_initial_pos = list(TASK_CUBE_POS)
        self._arm_joint_names = (
            MYCOBOT_320_MODEL_JOINT_NAMES
            if _has_all_joints(mujoco, self.model, MYCOBOT_320_MODEL_JOINT_NAMES)
            else MYCOBOT_MODEL_JOINT_NAMES
        )
        self._qpos_indices = _joint_qpos_indices(mujoco, self.model, self._arm_joint_names)
        self._dof_indices = _joint_dof_indices(mujoco, self.model, self._arm_joint_names)
        cube_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "task_cube_freejoint",
        )
        if cube_joint_id < 0:
            raise RuntimeError("missing task_cube_freejoint in myCobot Nexus scene")
        self._cube_freejoint_qpos_index = int(self.model.jnt_qposadr[cube_joint_id])
        self._cube_freejoint_qvel_index = int(self.model.jnt_dofadr[cube_joint_id])
        self._uses_official_gripper = _has_all_joints(
            mujoco,
            self.model,
            OFFICIAL_GRIPPER_JOINT_NAMES,
        )
        self._uses_official_320_gripper = _has_all_joints(
            mujoco,
            self.model,
            OFFICIAL_320_GRIPPER_JOINT_NAMES,
        )
        self._uses_official_320_adaptive_gripper = (
            config.model_profile in {
                MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
                MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
            }
            and self._uses_official_320_gripper
        )
        gripper_joint_names = (
            OFFICIAL_320_GRIPPER_JOINT_NAMES
            if self._uses_official_320_gripper
            else
            OFFICIAL_GRIPPER_JOINT_NAMES
            if self._uses_official_gripper
            else SYNTHETIC_GRIPPER_JOINT_NAMES
        )
        self._gripper_qpos_indices = _named_joint_qpos_indices(
            mujoco,
            self.model,
            gripper_joint_names,
        )
        self._gripper_low, self._gripper_high = _joint_ranges(
            self.model,
            self._gripper_qpos_indices,
        )
        self._arm_actuator_indices = (
            _named_actuator_indices(
                mujoco,
                self.model,
                [f"act_{name}" for name in self._arm_joint_names],
            )
            if self._uses_official_320_gripper
            else []
        )
        self._gripper_actuator_indices: list[int] = []
        self._low, self._high = _joint_ranges(self.model, self._qpos_indices)
        self._teacher_joint_names = [*self._arm_joint_names, "gripper_controller"]
        self.action_dim = len(self._teacher_joint_names)
        self.observation_dim = len(self._teacher_joint_names) + 3

    def reset(self, seed: int = 0) -> tuple[list[float], dict[str, Any]]:
        self._mujoco.mj_resetData(self.model, self.data)
        self._step = 0
        self._grasp_attached = False
        neutral = (
            [0.0, 0.45, 0.0, 0.0, 0.0, 0.0]
            if self._uses_official_320_gripper
            else _neutral_qpos(seed=seed, low=self._low, high=self._high)
        )
        for qpos_index, value in zip(self._qpos_indices, neutral, strict=True):
            self.data.qpos[qpos_index] = value
        if self._uses_official_320_gripper:
            for actuator_index, value in zip(self._arm_actuator_indices, neutral, strict=True):
                self.data.ctrl[actuator_index] = float(value)
        self._set_gripper(command=1.0)
        self._mujoco.mj_forward(self.model, self.data)
        self._cube_initial_pos = list(TASK_CUBE_POS)
        if self._uses_official_320_gripper:
            self._set_gripper(command=-0.4)
            self._mujoco.mj_forward(self.model, self.data)
            pad_midpoint = self._finger_pad_midpoint()
            self._set_gripper(command=1.0)
            self._cube_initial_pos = [
                float(pad_midpoint[0]),
                float(pad_midpoint[1]),
                float(pad_midpoint[2]),
            ]
            for axis, value in enumerate(self._cube_initial_pos):
                self.data.qpos[self._cube_freejoint_qpos_index + axis] = float(value)
            qvel_start = self._cube_freejoint_qvel_index
            self.data.qvel[qvel_start:qvel_start + 6] = 0.0
            self._mujoco.mj_forward(self.model, self.data)
        return self._observation(gripper=1.0), self._info(gripper=1.0)

    def step(self, action: list[float]) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        values = sanitize_teacher_action(action)
        arm_target = _clip(values[: len(self._qpos_indices)], self._low, self._high)
        gripper = values[-1]
        if self._uses_official_320_gripper:
            for target, actuator_index in zip(arm_target, self._arm_actuator_indices, strict=True):
                self.data.ctrl[actuator_index] = float(target)
        else:
            for target, qpos_index in zip(arm_target, self._qpos_indices, strict=True):
                current = float(self.data.qpos[qpos_index])
                delta = self.config.control_alpha * (target - current)
                self.data.qpos[qpos_index] = current + delta
        self._set_gripper(command=gripper)
        self._mujoco.mj_step(self.model, self.data)
        if self._uses_official_320_adaptive_gripper:
            self._set_gripper(command=gripper)
            self._mujoco.mj_forward(self.model, self.data)
        self._update_grasp_attachment(gripper=gripper)
        self._step += 1
        obs = self._observation(gripper=gripper)
        info = self._info(gripper=gripper)
        reward = -float(info["tcp_to_cube_dist"])
        terminated = bool(info["success"])
        return obs, reward, terminated, False, info

    def cube_approach_action(self) -> list[float]:
        """Choose a qpos target that moves the TCP proxy toward the cube."""
        import numpy as np

        target = np.asarray(self._cube_position(), dtype=float)
        target[2] += 0.08
        tcp = np.asarray(self._tcp_position(), dtype=float)
        error = target - tcp
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        self._mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.model.nbody - 1)
        arm_delta = jacp[:, self._dof_indices].T @ error
        arm_delta = np.clip(1.8 * arm_delta, -0.18, 0.18)
        current = [float(self.data.qpos[index]) for index in self._qpos_indices]
        target_qpos = [
            value + float(delta)
            for value, delta in zip(current, arm_delta, strict=True)
        ]
        return [*_clip(target_qpos, self._low, self._high), -0.2]

    def cube_grasp_lift_action(self, step: int, total_steps: int) -> list[float]:
        """Approach, close the native gripper, then lift above the cube."""
        import numpy as np

        if self._uses_official_320_gripper:
            home = [0.0, 0.22, 0.0, 0.12, 0.0, 0.0]
            pregrasp = [0.0, 0.45, 0.0, 0.08, 0.0, 0.0]
            lift = [0.0, -0.25, 0.0, 0.16, 0.0, 0.0]
            phase = step / max(1, total_steps - 1)
            if phase < 0.42:
                target_qpos = _lerp_vector(home, pregrasp, _smoothstep(phase / 0.42))
                gripper = 1.0
            elif phase < 0.64:
                target_qpos = pregrasp
                gripper = 1.0 - 2.0 * _smoothstep((phase - 0.42) / 0.22)
            else:
                target_qpos = _lerp_vector(pregrasp, lift, _smoothstep((phase - 0.64) / 0.36))
                gripper = -1.0
            return [*target_qpos, gripper]

        phase = step / max(1, total_steps - 1)
        cube = np.asarray(self._cube_position(), dtype=float)
        target = cube.copy()
        gripper = 1.0
        if phase < 0.45:
            target[2] += 0.065
        elif phase < 0.72:
            target[2] += 0.018
            gripper = -1.0
        else:
            target[2] += 0.18
            gripper = -1.0
        return [
            *self._jacobian_qpos_target(
                target.tolist(),
                gain=1.9,
                forward_target_xyz=cube.tolist(),
            ),
            gripper,
        ]

    def render(self) -> Any:
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                self.model,
                height=self.config.height,
                width=self.config.width,
            )
        camera = self._make_camera()
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _observation(self, *, gripper: float) -> list[float]:
        qpos = [float(self.data.qpos[index]) for index in self._qpos_indices]
        return [*qpos, float(gripper), *self._cube_position()]

    def _info(self, *, gripper: float) -> dict[str, Any]:
        tcp = self._tcp_position()
        cube = self._cube_position()
        contacts = self._gripper_cube_contacts()
        pad_contacts = self._gripper_cube_contact_pad_count()
        cube_lift = cube[2] - self._cube_initial_pos[2]
        contact_grasp_success = (
            self._uses_official_320_gripper
            and cube_lift > 0.025
            and pad_contacts >= 2
        )
        teacher_grasp_success = cube_lift > 0.025 and self._grasp_attached
        return {
            "step": self._step,
            "joint_names": self._teacher_joint_names,
            "cube_position": cube,
            "tcp_position": tcp,
            "tcp_to_cube_dist": _distance(tcp, cube),
            "gripper_command": float(gripper),
            "gripper_cube_contacts": contacts,
            "gripper_cube_contact_pads": pad_contacts,
            "grasp_attached": bool(self._grasp_attached),
            "cube_lift": cube_lift,
            "success": bool(contact_grasp_success or teacher_grasp_success),
            "success_label": (
                "contact_grasp_lift_success"
                if contact_grasp_success
                else "teacher_grasp_lift_success"
                if teacher_grasp_success
                else "not_success"
            ),
            "scene_path": str(self.scene_path),
        }

    def _update_grasp_attachment(self, *, gripper: float) -> None:
        if (
            self.config.model_profile == MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER
            and not self.config.teacher_grasp_attachment_enabled
        ):
            return
        uses_280_teacher_attachment = (
            self.config.model_profile == MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER
            and self.config.teacher_grasp_attachment_enabled
        )
        if self._uses_official_320_gripper and not uses_280_teacher_attachment:
            return
        cube = self._cube_position()
        tcp = self._tcp_position()
        if not self._grasp_attached:
            pad_midpoint = self._finger_pad_midpoint()
            pad_threshold = 0.14 if self._uses_official_320_gripper else 0.08
            pad_contacts = self._gripper_cube_contact_pad_count()
            close_enough = (
                _distance(cube, tcp) < 0.04
                or _distance(cube, pad_midpoint) < pad_threshold
                or self._gripper_cube_contacts() > 0
                or pad_contacts >= 2
            )
            attach_gripper_threshold = -0.1 if uses_280_teacher_attachment else -0.5
            verified_280_contact = not uses_280_teacher_attachment or pad_contacts >= 2
            if float(gripper) <= attach_gripper_threshold and close_enough and verified_280_contact:
                self._grasp_attached = True
        if not self._grasp_attached:
            return
        target = self._finger_pad_midpoint()
        for axis, value in enumerate(target):
            self.data.qpos[self._cube_freejoint_qpos_index + axis] = float(value)
        self.data.qvel[self._cube_freejoint_qvel_index:self._cube_freejoint_qvel_index + 6] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def _finger_pad_midpoint(self) -> list[float]:
        if self._uses_official_320_gripper:
            midpoint = self._geom_midpoint("left_finger_pad", "right_finger_pad")
            if midpoint is not None:
                return midpoint
        midpoint = self._geom_midpoint("left_finger_pad", "right_finger_pad")
        if midpoint is not None:
            return midpoint
        return self._tcp_position()

    def _body_local_point(self, body_name: str, local_xyz: list[float]) -> list[float] | None:
        import numpy as np

        body_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        if body_id < 0:
            return None
        pos = np.asarray(self.data.xpos[body_id], dtype=float)
        mat = np.asarray(self.data.xmat[body_id], dtype=float).reshape(3, 3)
        point = pos + mat @ np.asarray(local_xyz, dtype=float)
        return [float(value) for value in point]

    def _geom_midpoint(self, left_name: str, right_name: str) -> list[float] | None:
        left_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_GEOM,
            left_name,
        )
        right_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_GEOM,
            right_name,
        )
        if left_id < 0 or right_id < 0:
            return None
        midpoint = (self.data.geom_xpos[left_id] + self.data.geom_xpos[right_id]) * 0.5
        return [float(value) for value in midpoint]

    def _jacobian_qpos_target(
        self,
        target_xyz: list[float],
        *,
        gain: float,
        forward_target_xyz: list[float] | None = None,
    ) -> list[float]:
        import numpy as np

        target = np.asarray(target_xyz, dtype=float)
        tcp = np.asarray(self._tcp_position(), dtype=float)
        site_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        if site_id < 0:
            return self._single_step_jacobian_qpos_target(target, gain=gain)

        original_qpos = self.data.qpos.copy()
        original_qvel = self.data.qvel.copy()
        try:
            qpos_values = np.asarray(
                [float(self.data.qpos[index]) for index in self._qpos_indices],
                dtype=float,
            )
            for _ in range(8):
                for qpos_index, value in zip(self._qpos_indices, qpos_values, strict=True):
                    self.data.qpos[qpos_index] = float(value)
                self._mujoco.mj_forward(self.model, self.data)
                tcp = np.asarray(self.data.site_xpos[site_id], dtype=float)
                error = target - tcp
                jacp = np.zeros((3, self.model.nv), dtype=float)
                jacr = np.zeros((3, self.model.nv), dtype=float)
                self._mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
                arm_delta = jacp[:, self._dof_indices].T @ error
                if forward_target_xyz is not None:
                    orientation_error = self._gripper_forward_error(
                        site_id=site_id,
                        forward_target_xyz=forward_target_xyz,
                    )
                    if orientation_error is not None:
                        arm_delta += 0.14 * (
                            jacr[:, self._dof_indices].T @ orientation_error
                        )
                arm_delta = np.clip(gain * arm_delta, -0.075, 0.075)
                qpos_values = np.asarray(
                    _clip((qpos_values + arm_delta).tolist(), self._low, self._high),
                    dtype=float,
                )
                if float(np.linalg.norm(error)) < 0.01:
                    break
            return qpos_values.tolist()
        finally:
            self.data.qpos[:] = original_qpos
            self.data.qvel[:] = original_qvel
            self._mujoco.mj_forward(self.model, self.data)

    def _single_step_jacobian_qpos_target(self, target: Any, *, gain: float) -> list[float]:
        import numpy as np

        tcp = np.asarray(self._tcp_position(), dtype=float)
        error = target - tcp
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        body_id = self.model.nbody - 1
        self._mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        arm_delta = np.clip(gain * (jacp[:, self._dof_indices].T @ error), -0.18, 0.18)
        current = [float(self.data.qpos[index]) for index in self._qpos_indices]
        target_qpos = [
            value + float(delta)
            for value, delta in zip(current, arm_delta, strict=True)
        ]
        return _clip(target_qpos, self._low, self._high)

    def _gripper_forward_error(
        self,
        *,
        site_id: int,
        forward_target_xyz: list[float],
    ) -> Any | None:
        import numpy as np

        tcp = np.asarray(self.data.site_xpos[site_id], dtype=float)
        desired_forward = np.asarray(forward_target_xyz, dtype=float) - tcp
        desired_forward[2] = 0.0
        desired_norm = float(np.linalg.norm(desired_forward))
        if desired_norm <= 1e-6:
            return None
        desired_forward /= desired_norm
        site_xmat = np.asarray(self.data.site_xmat[site_id], dtype=float).reshape(3, 3)
        current_forward = site_xmat @ np.asarray([0.0, -1.0, 0.0], dtype=float)
        current_forward[2] = 0.0
        current_norm = float(np.linalg.norm(current_forward))
        if current_norm <= 1e-6:
            return None
        current_forward /= current_norm
        return np.cross(current_forward, desired_forward)

    def _set_gripper(self, *, command: float) -> None:
        if self._uses_official_320_gripper:
            open_value = 0.0 if self._uses_official_320_adaptive_gripper else -0.7
            closed_value = -1.11 if self._uses_official_320_adaptive_gripper else 0.3
            close_amount = (1.0 - max(-1.0, min(1.0, float(command)))) * 0.5
            base_value = open_value + close_amount * (closed_value - open_value)
            for index, (qpos_index, joint_name) in enumerate(
                zip(
                    self._gripper_qpos_indices,
                    OFFICIAL_320_GRIPPER_JOINT_NAMES,
                    strict=True,
                )
            ):
                raw_value = base_value * OFFICIAL_320_GRIPPER_MIMIC[joint_name]
                if self._uses_official_320_adaptive_gripper:
                    target = raw_value
                else:
                    target = max(
                        self._gripper_low[index],
                        min(self._gripper_high[index], raw_value),
                    )
                self.data.qpos[qpos_index] = target
            return
        if self._uses_official_gripper:
            open_value = -0.007
            closed_value = 0.0
            base_value = closed_value + (max(-1.0, min(1.0, float(command))) + 1.0) * 0.5 * (
                open_value - closed_value
            )
            for qpos_index, joint_name in zip(
                self._gripper_qpos_indices,
                OFFICIAL_GRIPPER_JOINT_NAMES,
                strict=True,
            ):
                self.data.qpos[qpos_index] = base_value * OFFICIAL_GRIPPER_MIMIC[joint_name]
            return
        close_amount = (1.0 - max(-1.0, min(1.0, float(command)))) * 0.5
        qpos_value = close_amount * 0.028
        for qpos_index in self._gripper_qpos_indices:
            self.data.qpos[qpos_index] = qpos_value

    def _cube_position(self) -> list[float]:
        return _body_position(self._mujoco, self.model, self.data, TASK_CUBE_BODY)

    def _tcp_position(self) -> list[float]:
        site_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        if site_id >= 0:
            pos = self.data.site_xpos[site_id]
            return [float(pos[0]), float(pos[1]), float(pos[2])]
        return _body_position(self._mujoco, self.model, self.data, "joint6_flange")

    def _gripper_cube_contacts(self) -> int:
        return _gripper_cube_contacts(self._mujoco, self.model, self.data)

    def _gripper_cube_contact_pad_count(self) -> int:
        return _gripper_cube_contact_pad_count(self._mujoco, self.model, self.data)

    def _make_camera(self) -> Any:
        camera = self._mujoco.MjvCamera()
        camera.type = self._mujoco.mjtCamera.mjCAMERA_FREE
        if self._uses_official_320_gripper:
            camera.lookat[:] = [-0.18, -0.06, 0.12]
            camera.distance = 0.92
            camera.azimuth = 38.0
            camera.elevation = -18.0
        else:
            camera.lookat[:] = [-0.13, -0.10, 0.11]
            camera.distance = 1.12
            camera.azimuth = 138.0
            camera.elevation = -34.0
        return camera


def run_mycobot_nexus_smoke(
    *,
    output_dir: Path,
    asset_root: Path,
    steps: int,
    seed: int,
    width: int,
    height: int,
    policy: str = "sample",
    official_gripper_root: Path | None = None,
    model_profile: str = "280-jn",
) -> MyCobotNexusSmokeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "mycobot_nexus_trace.jsonl"
    frame_path = output_dir / "mycobot_nexus_frame.bmp"
    report_path = output_dir / "mycobot_nexus_report.json"
    env = MyCobotNexusEnv(
        MyCobotNexusConfig(
            asset_root=asset_root,
            work_dir=output_dir,
            official_gripper_root=official_gripper_root,
            model_profile=model_profile,
            width=width,
            height=height,
        )
    )
    records: list[MyCobotNexusStep] = []
    obs, info = env.reset(seed=seed)
    initial_dist = float(info["tcp_to_cube_dist"])
    initial_cube_z = float(info["cube_position"][2])
    try:
        for step in range(steps):
            if policy == "cube-approach":
                action = env.cube_approach_action()
            elif policy == "grasp-lift":
                action = env.cube_grasp_lift_action(step, steps)
            elif policy == "sample":
                action = sample_mycobot_nexus_action(step, steps)
            else:
                raise ValueError(f"unsupported myCobot Nexus policy: {policy}")
            obs, reward, terminated, truncated, info = env.step(action)
            records.append(
                MyCobotNexusStep(
                    step=step,
                    observation=obs,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=_json_safe_info(info),
                )
            )
            if terminated or truncated:
                break
        _write_bmp(frame_path, env.render())
    finally:
        scene_path = str(env.scene_path)
        env.close()

    with trace_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    final_dist = float(records[-1].info["tcp_to_cube_dist"]) if records else initial_dist
    final_cube_z = float(records[-1].info["cube_position"][2]) if records else initial_cube_z
    max_contacts = (
        max(int(record.info["gripper_cube_contacts"]) for record in records)
        if records
        else 0
    )
    max_contact_pads = (
        max(int(record.info.get("gripper_cube_contact_pads", 0)) for record in records)
        if records
        else 0
    )
    grasp_success = bool(records and any(bool(record.info["success"]) for record in records))
    status = "failed" if policy == "grasp-lift" and not grasp_success else "passed"
    result = MyCobotNexusSmokeResult(
        status=status,
        policy=policy,
        steps=len(records),
        observation_dim=len(obs),
        action_dim=env.action_dim,
        initial_tcp_to_cube_dist=initial_dist,
        final_tcp_to_cube_dist=final_dist,
        min_tcp_to_cube_dist=min(
            float(record.info["tcp_to_cube_dist"]) for record in records
        )
        if records
        else initial_dist,
        approach_improved=bool(
            records and final_dist < initial_dist
        ),
        initial_cube_z=initial_cube_z,
        final_cube_z=final_cube_z,
        cube_lifted=bool(final_cube_z > initial_cube_z + 0.025),
        gripper_cube_contacts=max_contacts,
        gripper_cube_contact_pads=max_contact_pads,
        grasp_success=grasp_success,
        trace_path=str(trace_path),
        frame_path=str(frame_path),
        report_path=str(report_path),
        scene_path=scene_path,
    )
    report_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_mycobot_adaptive_static_contact_smoke(
    *,
    output_dir: Path,
    asset_root: Path,
    official_gripper_root: Path,
    model_profile: str = MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
    steps: int = 80,
    seed: int = 1,
    width: int = 640,
    height: int = 480,
    placement_gripper_command: float = 0.25,
    final_gripper_command: float = -1.0,
    required_sustained_steps: int = 15,
) -> MyCobotAdaptiveStaticContactResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "mycobot_adaptive_static_contact_trace.jsonl"
    frame_path = output_dir / "mycobot_adaptive_static_contact_frame.bmp"
    report_path = output_dir / "mycobot_adaptive_static_contact_report.json"
    env = MyCobotNexusEnv(
        MyCobotNexusConfig(
            asset_root=asset_root,
            work_dir=output_dir,
            official_gripper_root=official_gripper_root,
            model_profile=model_profile,
            width=width,
            height=height,
        )
    )
    records: list[MyCobotNexusStep] = []
    best_sustained_steps = 0
    sustained_steps = 0
    max_contacts = 0
    max_contact_pads = 0
    initial_cube_position: list[float] = []
    final_cube_position: list[float] = []
    try:
        env.reset(seed=seed)
        gate7_arm_qpos = _adaptive_gate7_arm_qpos(model_profile)
        for qpos_index, value in zip(env._qpos_indices, gate7_arm_qpos, strict=True):
            env.data.qpos[qpos_index] = float(value)
        if env._uses_official_320_gripper:
            for actuator_index, value in zip(
                env._arm_actuator_indices,
                gate7_arm_qpos,
                strict=True,
            ):
                env.data.ctrl[actuator_index] = float(value)
        env._set_gripper(command=placement_gripper_command)
        env._mujoco.mj_forward(env.model, env.data)
        pad_midpoint = env._finger_pad_midpoint()
        cube_offset = _adaptive_gate7_cube_offset(model_profile)
        initial_cube_position = [
            float(pad_midpoint[0] + cube_offset[0]),
            float(pad_midpoint[1] + cube_offset[1]),
            float(TASK_CUBE_POS[2] + cube_offset[2]),
        ]
        for axis, value in enumerate(initial_cube_position):
            env.data.qpos[env._cube_freejoint_qpos_index + axis] = float(value)
        qvel_start = env._cube_freejoint_qvel_index
        env.data.qvel[qvel_start:qvel_start + 6] = 0.0
        env._cube_initial_pos = list(initial_cube_position)
        env._mujoco.mj_forward(env.model, env.data)

        arm_target = list(gate7_arm_qpos)
        hold_arm_pose_each_step = model_profile == MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER
        denominator = max(steps - 1, 1)
        for step in range(steps):
            alpha = step / denominator
            gripper = placement_gripper_command + alpha * (
                final_gripper_command - placement_gripper_command
            )
            if hold_arm_pose_each_step:
                _set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
            obs, reward, terminated, truncated, info = env.step([*arm_target, gripper])
            if hold_arm_pose_each_step:
                _set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
            contacts = int(info["gripper_cube_contacts"])
            contact_pads = int(info["gripper_cube_contact_pads"])
            max_contacts = max(max_contacts, contacts)
            max_contact_pads = max(max_contact_pads, contact_pads)
            if contact_pads >= 2:
                sustained_steps += 1
            else:
                sustained_steps = 0
            best_sustained_steps = max(best_sustained_steps, sustained_steps)
            records.append(
                MyCobotNexusStep(
                    step=step,
                    observation=obs,
                    action=[*arm_target, gripper],
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=_json_safe_info(info),
                )
            )
        _write_bmp(frame_path, env.render())
        final_cube_position = env._cube_position()
    finally:
        scene_path = str(env.scene_path)
        env.close()

    with trace_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    status = (
        "passed"
        if max_contact_pads >= 2 and best_sustained_steps >= required_sustained_steps
        else "failed"
    )
    result = MyCobotAdaptiveStaticContactResult(
        status=status,
        steps=len(records),
        placement_gripper_command=float(placement_gripper_command),
        final_gripper_command=float(final_gripper_command),
        required_sustained_steps=int(required_sustained_steps),
        best_sustained_contact_steps=int(best_sustained_steps),
        gripper_cube_contacts=int(max_contacts),
        gripper_cube_contact_pads=int(max_contact_pads),
        initial_cube_position=initial_cube_position,
        final_cube_position=final_cube_position,
        trace_path=str(trace_path),
        frame_path=str(frame_path),
        report_path=str(report_path),
        scene_path=scene_path,
    )
    report_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_mycobot_adaptive_grasp_lift_smoke(
    *,
    output_dir: Path,
    asset_root: Path,
    official_gripper_root: Path,
    model_profile: str = MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
    seed: int = 1,
    width: int = 640,
    height: int = 480,
    pregrasp_steps: int = 20,
    close_steps: int = 80,
    lift_steps: int = 60,
    placement_gripper_command: float = 0.25,
    close_gripper_command: float = -0.7,
    required_close_sustained_steps: int = 15,
    required_lift_sustained_steps: int = 30,
    required_final_lift: float = 0.025,
    teacher_attachment_enabled: bool = True,
) -> MyCobotAdaptiveGraspLiftResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "mycobot_adaptive_grasp_lift_trace.jsonl"
    frame_path = output_dir / "mycobot_adaptive_grasp_lift_frame.bmp"
    report_path = output_dir / "mycobot_adaptive_grasp_lift_report.json"
    env = MyCobotNexusEnv(
        MyCobotNexusConfig(
            asset_root=asset_root,
            work_dir=output_dir,
            official_gripper_root=official_gripper_root,
            model_profile=model_profile,
            width=width,
            height=height,
            teacher_grasp_attachment_enabled=teacher_attachment_enabled,
        )
    )
    records: list[dict[str, Any]] = []
    initial_cube_position: list[float] = []
    final_cube_position: list[float] = []
    try:
        env.reset(seed=seed)
        gate7_arm_qpos = _adaptive_gate7_arm_qpos(model_profile)
        gate8_lift_arm_qpos = _adaptive_gate8_lift_arm_qpos(model_profile)
        _set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
        env._set_gripper(command=placement_gripper_command)
        env._mujoco.mj_forward(env.model, env.data)
        pad_midpoint = env._finger_pad_midpoint()
        cube_offset = _adaptive_gate7_cube_offset(model_profile)
        initial_cube_position = [
            float(pad_midpoint[0] + cube_offset[0]),
            float(pad_midpoint[1] + cube_offset[1]),
            float(TASK_CUBE_POS[2] + cube_offset[2]),
        ]
        for axis, value in enumerate(initial_cube_position):
            env.data.qpos[env._cube_freejoint_qpos_index + axis] = float(value)
        qvel_start = env._cube_freejoint_qvel_index
        env.data.qvel[qvel_start:qvel_start + 6] = 0.0
        env._cube_initial_pos = list(initial_cube_position)
        env._mujoco.mj_forward(env.model, env.data)

        step_index = 0
        hold_arm_pose_each_step = model_profile == MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER
        for _ in range(pregrasp_steps):
            if hold_arm_pose_each_step:
                _set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
            obs, reward, terminated, truncated, info = env.step(
                [*gate7_arm_qpos, placement_gripper_command]
            )
            if hold_arm_pose_each_step:
                _set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
            records.append(_phase_record("pregrasp", step_index, obs, reward, terminated, truncated, info))
            step_index += 1
        close_denominator = max(close_steps - 1, 1)
        for step in range(close_steps):
            alpha = step / close_denominator
            gripper = placement_gripper_command + alpha * (
                close_gripper_command - placement_gripper_command
            )
            if hold_arm_pose_each_step:
                _set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
            obs, reward, terminated, truncated, info = env.step(
                [*gate7_arm_qpos, gripper]
            )
            if hold_arm_pose_each_step:
                _set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
            records.append(_phase_record("close", step_index, obs, reward, terminated, truncated, info))
            step_index += 1
        lift_denominator = max(lift_steps - 1, 1)
        for step in range(lift_steps):
            alpha = _smoothstep(step / lift_denominator)
            arm = _lerp_vector(
                list(gate7_arm_qpos),
                list(gate8_lift_arm_qpos),
                alpha,
            )
            if hold_arm_pose_each_step:
                _set_adaptive_gate_arm_pose(env, tuple(arm))
            obs, reward, terminated, truncated, info = env.step([*arm, close_gripper_command])
            if hold_arm_pose_each_step:
                _set_adaptive_gate_arm_pose(env, tuple(arm))
            records.append(_phase_record("lift", step_index, obs, reward, terminated, truncated, info))
            step_index += 1
        _write_bmp(frame_path, env.render())
        final_cube_position = env._cube_position()
    finally:
        scene_path = str(env.scene_path)
        env.close()

    close_infos = [record["info"] for record in records if record["phase"] == "close"]
    lift_infos = [record["info"] for record in records if record["phase"] == "lift"]
    close_best = _best_sustained_two_pad_contact(close_infos)
    lift_best = _best_sustained_two_pad_contact(lift_infos)
    lift_two_pad_steps = sum(
        1 for info in lift_infos if int(info.get("gripper_cube_contact_pads", 0)) >= 2
    )
    final_info = records[-1]["info"] if records else {}
    final_lift = (
        float(final_cube_position[2]) - float(initial_cube_position[2])
        if final_cube_position and initial_cube_position
        else 0.0
    )
    status = (
        "passed"
        if close_best >= required_close_sustained_steps
        and lift_best >= required_lift_sustained_steps
        and int(final_info.get("gripper_cube_contact_pads", 0)) >= 2
        and final_lift >= required_final_lift
        else "failed"
    )
    with trace_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")
    result = MyCobotAdaptiveGraspLiftResult(
        status=status,
        steps=len(records),
        close_gripper_command=float(close_gripper_command),
        required_close_sustained_steps=int(required_close_sustained_steps),
        required_lift_sustained_steps=int(required_lift_sustained_steps),
        required_final_lift=float(required_final_lift),
        close_best_sustained_contact_steps=int(close_best),
        lift_best_sustained_contact_steps=int(lift_best),
        lift_two_pad_contact_steps=int(lift_two_pad_steps),
        final_gripper_cube_contact_pads=int(final_info.get("gripper_cube_contact_pads", 0)),
        final_gripper_cube_contacts=int(final_info.get("gripper_cube_contacts", 0)),
        initial_cube_position=initial_cube_position,
        final_cube_position=final_cube_position,
        final_cube_lift=float(final_lift),
        trace_path=str(trace_path),
        frame_path=str(frame_path),
        report_path=str(report_path),
        scene_path=scene_path,
        teacher_attachment_enabled=bool(teacher_attachment_enabled),
    )
    report_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def _adaptive_gate7_arm_qpos(model_profile: str) -> tuple[float, ...]:
    if model_profile == MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER:
        return ADAPTIVE_280_PI_GATE7_TABLE_ARM_QPOS
    return ADAPTIVE_GATE7_TABLE_ARM_QPOS


def _adaptive_gate8_lift_arm_qpos(model_profile: str) -> tuple[float, ...]:
    if model_profile == MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER:
        return ADAPTIVE_280_PI_GATE8_LIFT_ARM_QPOS
    return ADAPTIVE_GATE8_LIFT_ARM_QPOS


def _adaptive_gate7_cube_offset(model_profile: str) -> tuple[float, float, float]:
    if model_profile == MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER:
        return ADAPTIVE_280_PI_GATE7_CUBE_OFFSET
    return ADAPTIVE_GATE7_CUBE_OFFSET


def _set_adaptive_gate_arm_pose(env: MyCobotNexusEnv, qpos: tuple[float, ...]) -> None:
    for qpos_index, dof_index, value in zip(env._qpos_indices, env._dof_indices, qpos, strict=True):
        env.data.qpos[qpos_index] = float(value)
        env.data.qvel[dof_index] = 0.0
    if env._uses_official_320_gripper:
        for actuator_index, value in zip(env._arm_actuator_indices, qpos, strict=True):
            env.data.ctrl[actuator_index] = float(value)


def _phase_record(
    phase: str,
    step: int,
    observation: list[float],
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": phase,
        "step": step,
        "observation": observation,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": _json_safe_info(info),
    }


def _best_sustained_two_pad_contact(infos: list[dict[str, Any]]) -> int:
    best = 0
    current = 0
    for info in infos:
        if int(info.get("gripper_cube_contact_pads", 0)) >= 2:
            current += 1
        else:
            current = 0
        best = max(best, current)
    return best


def mycobot_nexus_contract() -> dict[str, Any]:
    return {
        "env": "MyCobotNexusEnv",
        "surface": ["reset(seed)", "step(action)", "render()", "close()"],
        "policies": ["sample", "cube-approach", "grasp-lift"],
        "model_profiles": [
            MODEL_PROFILE_280_JN,
            MODEL_PROFILE_320_GRIPPER,
            MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
            MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
        ],
        "asset_source": "https://github.com/elephantrobotics/mycobot_mujoco",
        "model_relative_path": str(MYCOBOT_MODEL_RELATIVE_PATH),
        "joint_order": MYCOBOT_TEACHER_JOINT_NAMES,
        "action_dim": len(MYCOBOT_TEACHER_JOINT_NAMES),
        "observation_dim": len(MYCOBOT_TEACHER_JOINT_NAMES) + 3,
        "task_objects": [
            "task_cube",
            "nexus_work_mat",
            "official_parallel_gripper",
            "official_320_m5_2022_gripper",
            "official_320_m5_2022_adaptive_gripper",
            "official_280_pi_adaptive_gripper",
            "official_320_m5_2022_friction_contact_gripper",
            "synthetic_parallel_gripper_fallback",
            "teacher_grasp_attachment_proxy",
        ],
        "official_gripper_asset_source": "https://github.com/elephantrobotics/mycobot_ros",
        "official_gripper_relative_path": str(OFFICIAL_GRIPPER_MESH_RELATIVE_PATH),
        "official_320_m5_2022_gripper_urdf": str(OFFICIAL_320_URDF_RELATIVE_PATH),
        "official_320_m5_2022_adaptive_gripper_urdf": str(
            OFFICIAL_320_ADAPTIVE_URDF_RELATIVE_PATH
        ),
        "official_320_m5_2022_adaptive_gripper_meshes": str(
            OFFICIAL_320_ADAPTIVE_GRIPPER_MESH_RELATIVE_PATH
        ),
        "official_280_pi_urdf": str(OFFICIAL_280_PI_URDF_RELATIVE_PATH),
        "official_280_pi_meshes": str(OFFICIAL_280_PI_MESH_RELATIVE_PATH),
        "official_adaptive_gripper_urdf": str(OFFICIAL_ADAPTIVE_GRIPPER_URDF_RELATIVE_PATH),
        "official_adaptive_gripper_meshes": str(OFFICIAL_ADAPTIVE_GRIPPER_MESH_RELATIVE_PATH),
        "real_robot_execution": "disabled",
        "poc_boundary": (
            "Kinematic qpos-target MuJoCo env. It steps a real myCobot model in a "
            "Nexus-style cube scene. The preferred legacy gripper path uses official "
            "mycobot_ros 280 JN parallel-gripper visual meshes with transparent "
            "contact pads. The 320 M5 2022 and 280 Pi adaptive profiles use official "
            "arm/adaptive-gripper URDF sources plus functional friction-contact "
            "finger pads; grasp-lift success requires both finger pads to contact "
            "the cube without teacher attachment."
        ),
    }


def write_dry_contract(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "mycobot_nexus_contract.json"
    path.write_text(
        json.dumps(mycobot_nexus_contract(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def sample_mycobot_nexus_action(step: int, total_steps: int) -> list[float]:
    denom = max(1, total_steps - 1)
    phase = step / denom
    values = []
    for index in range(len(MYCOBOT_TEACHER_JOINT_NAMES)):
        values.append(0.35 * math.sin(phase * 2.0 * math.pi + index * 0.61))
    return values


def sanitize_teacher_action(action: list[float]) -> list[float]:
    values = [0.0] * len(MYCOBOT_TEACHER_JOINT_NAMES)
    for index, raw in enumerate(action[: len(values)]):
        value = float(raw)
        values[index] = value if math.isfinite(value) else 0.0
    return values


def _smoothstep(value: float) -> float:
    clipped = max(0.0, min(1.0, float(value)))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _lerp_vector(start: list[float], end: list[float], amount: float) -> list[float]:
    return [
        float(a) + (float(b) - float(a)) * amount
        for a, b in zip(start, end, strict=True)
    ]


def build_mycobot_nexus_scene_model(
    *,
    model_path: Path,
    scene_path: Path,
    official_gripper_root: Path | None = None,
    model_profile: str = MODEL_PROFILE_280_JN,
) -> None:
    if model_profile in {MODEL_PROFILE_320_GRIPPER, MODEL_PROFILE_320_ADAPTIVE_GRIPPER}:
        _build_official_320_nexus_scene_model(
            scene_path=scene_path,
            official_gripper_root=official_gripper_root,
            model_profile=model_profile,
        )
        return
    if model_profile == MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER:
        _build_official_280_pi_adaptive_nexus_scene_model(
            scene_path=scene_path,
            official_gripper_root=official_gripper_root,
        )
        return
    if model_profile != MODEL_PROFILE_280_JN:
        raise ValueError(f"unsupported myCobot model profile: {model_profile}")
    tree = ET.parse(model_path)
    root = tree.getroot()
    official_gripper_dir = _official_gripper_mesh_dir(official_gripper_root)
    official_gripper_obj_dir = (
        _prepare_official_gripper_obj_assets(official_gripper_dir, scene_path.parent)
        if official_gripper_dir is not None
        else None
    )

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str((model_path.parent / "../meshes_mujoco").resolve()))

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        worldbody_index = _child_index(root, "worldbody")
        root.insert(worldbody_index if worldbody_index >= 0 else len(root), asset)
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "nexus_skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.78 0.82 0.86",
            "rgb2": "0.96 0.97 0.96",
            "width": "256",
            "height": "256",
        },
    )
    _add_material(asset, "nexus_floor", "0.78 0.80 0.77 1", specular="0.12")
    _add_material(asset, "nexus_mat", "0.37 0.43 0.45 1", specular="0.08")
    _add_material(asset, "task_cube", "0.92 0.24 0.15 1", specular="0.22")
    if official_gripper_obj_dir is not None:
        _add_official_gripper_assets(asset, official_gripper_obj_dir)

    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        worldbody_index = _child_index(root, "worldbody")
        root.insert(worldbody_index if worldbody_index >= 0 else len(root), visual)
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": "0.36 0.36 0.34",
            "diffuse": "0.76 0.74 0.68",
            "specular": "0.12 0.12 0.12",
        },
    )
    ET.SubElement(visual, "map", {"znear": "0.01", "zfar": "10"})

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"missing worldbody in MuJoCo model: {model_path}")
    for node in reversed(_nexus_scene_nodes()):
        worldbody.insert(0, node)
    flange = root.find(".//body[@name='joint6_flange']")
    if flange is None:
        raise ValueError(f"missing joint6_flange body in MuJoCo model: {model_path}")
    if official_gripper_obj_dir is not None:
        flange.append(_official_parallel_gripper_node())
    else:
        flange.append(_synthetic_parallel_gripper_node())

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(scene_path, encoding="utf-8", xml_declaration=True)


def _build_official_320_nexus_scene_model(
    *,
    scene_path: Path,
    official_gripper_root: Path | None,
    model_profile: str,
) -> None:
    if official_gripper_root is None:
        raise FileNotFoundError(
            f"official myCobot ROS root is required for model_profile={model_profile}"
        )
    ros_root = official_gripper_root.expanduser()
    urdf_relative_path = (
        OFFICIAL_320_ADAPTIVE_URDF_RELATIVE_PATH
        if model_profile == MODEL_PROFILE_320_ADAPTIVE_GRIPPER
        else OFFICIAL_320_URDF_RELATIVE_PATH
    )
    gripper_mesh_relative_path = (
        OFFICIAL_320_ADAPTIVE_GRIPPER_MESH_RELATIVE_PATH
        if model_profile == MODEL_PROFILE_320_ADAPTIVE_GRIPPER
        else OFFICIAL_320_MESH_RELATIVE_PATH
    )
    urdf_path = ros_root / urdf_relative_path
    arm_mesh_root = ros_root / OFFICIAL_320_MESH_RELATIVE_PATH
    gripper_mesh_root = ros_root / gripper_mesh_relative_path
    if not urdf_path.exists():
        raise FileNotFoundError(f"missing official 320 M5 2022 gripper URDF: {urdf_path}")
    missing_arm = [
        name for name in OFFICIAL_320_ARM_LINK_NAMES if not (arm_mesh_root / f"{name}.dae").exists()
    ]
    missing_gripper = [
        name
        for name in OFFICIAL_320_GRIPPER_LINK_NAMES
        if not (gripper_mesh_root / f"{name}.dae").exists()
    ]
    missing = [*missing_arm, *missing_gripper]
    if missing:
        raise FileNotFoundError(
            "missing official 320 M5 2022 meshes for "
            f"{model_profile}: {', '.join(missing)}"
        )

    obj_dir = scene_path.parent / "official_320_m5_2022_meshes"
    obj_dir.mkdir(parents=True, exist_ok=True)
    for name in OFFICIAL_320_ARM_LINK_NAMES:
        _convert_collada_mesh_to_obj(
            arm_mesh_root / f"{name}.dae",
            obj_dir / f"{name}.obj",
            bake_visual_scene=False,
        )
    for name in OFFICIAL_320_GRIPPER_LINK_NAMES:
        _convert_collada_mesh_to_obj(
            gripper_mesh_root / f"{name}.dae",
            obj_dir / f"{name}.obj",
            bake_visual_scene=False,
        )

    urdf = ET.parse(urdf_path).getroot()
    link_visuals = _urdf_link_visuals(urdf)
    joints = _urdf_joints(urdf)
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    for joint in joints:
        children_by_parent.setdefault(str(joint["parent"]), []).append(joint)

    root = ET.Element("mujoco", {"model": f"official_{model_profile}_nexus"})
    ET.SubElement(root, "compiler", {"angle": "radian", "eulerseq": "XYZ"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": "0.001",
            "cone": "elliptic",
            "impratio": "100",
            "iterations": "120",
            "ls_iterations": "40",
        },
    )
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "nexus_skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.78 0.82 0.86",
            "rgb2": "0.96 0.97 0.96",
            "width": "256",
            "height": "256",
        },
    )
    _add_material(asset, "nexus_floor", "0.78 0.80 0.77 1", specular="0.12")
    _add_material(asset, "nexus_mat", "0.37 0.43 0.45 1", specular="0.08")
    _add_material(asset, "task_cube", "0.92 0.24 0.15 1", specular="0.22")
    for name in OFFICIAL_320_LINK_NAMES:
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"official_320_{name}", "file": str((obj_dir / f"{name}.obj").resolve())},
        )

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": "0.36 0.36 0.34",
            "diffuse": "0.76 0.74 0.68",
            "specular": "0.12 0.12 0.12",
        },
    )
    ET.SubElement(visual, "map", {"znear": "0.01", "zfar": "10"})
    ET.SubElement(visual, "global", {"offwidth": "960", "offheight": "720"})

    worldbody = ET.SubElement(root, "worldbody")
    for node in _nexus_scene_nodes():
        worldbody.append(node)
    base = ET.SubElement(worldbody, "body", {"name": "base", "pos": "0 0 0"})
    _add_urdf_visual_geoms(base, "base", link_visuals, mesh_prefix="official_320")
    _append_urdf_children(
        base,
        "base",
        children_by_parent,
        link_visuals,
        mesh_prefix="official_320",
    )
    _add_320_official_gripper_contact_pads(base)
    _add_320_position_actuators(root)

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(scene_path, encoding="utf-8", xml_declaration=True)


def _build_official_280_pi_adaptive_nexus_scene_model(
    *,
    scene_path: Path,
    official_gripper_root: Path | None,
) -> None:
    if official_gripper_root is None:
        raise FileNotFoundError(
            "official myCobot ROS root is required for model_profile="
            f"{MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER}"
        )
    ros_root = official_gripper_root.expanduser()
    arm_urdf_path = ros_root / OFFICIAL_280_PI_URDF_RELATIVE_PATH
    gripper_urdf_path = ros_root / OFFICIAL_ADAPTIVE_GRIPPER_URDF_RELATIVE_PATH
    arm_mesh_root = ros_root / OFFICIAL_280_PI_MESH_RELATIVE_PATH
    gripper_mesh_root = ros_root / OFFICIAL_ADAPTIVE_GRIPPER_MESH_RELATIVE_PATH
    if not arm_urdf_path.exists():
        raise FileNotFoundError(f"missing official 280 Pi URDF: {arm_urdf_path}")
    if not gripper_urdf_path.exists():
        raise FileNotFoundError(f"missing official adaptive gripper URDF: {gripper_urdf_path}")

    arm_urdf = ET.parse(arm_urdf_path).getroot()
    gripper_urdf = ET.parse(gripper_urdf_path).getroot()
    urdf = _combined_280_pi_adaptive_urdf(arm_urdf, gripper_urdf)
    link_visuals = _urdf_link_visuals(urdf)
    joints = _urdf_joints(urdf)
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    for joint in joints:
        children_by_parent.setdefault(str(joint["parent"]), []).append(joint)

    mesh_names = sorted({visual["mesh"] for visuals in link_visuals.values() for visual in visuals})
    obj_dir = scene_path.parent / "official_280_pi_adaptive_meshes"
    obj_dir.mkdir(parents=True, exist_ok=True)
    for name in mesh_names:
        dae_path = _find_required_mesh(name, (arm_mesh_root, gripper_mesh_root))
        _convert_collada_mesh_to_obj(
            dae_path,
            obj_dir / f"{name}.obj",
            bake_visual_scene=False,
        )

    root = ET.Element("mujoco", {"model": "official_280_pi_adaptive_gripper_nexus"})
    ET.SubElement(root, "compiler", {"angle": "radian", "eulerseq": "XYZ"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": "0.001",
            "cone": "elliptic",
            "impratio": "100",
            "iterations": "120",
            "ls_iterations": "40",
        },
    )
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "nexus_skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.78 0.82 0.86",
            "rgb2": "0.96 0.97 0.96",
            "width": "256",
            "height": "256",
        },
    )
    _add_material(asset, "nexus_floor", "0.78 0.80 0.77 1", specular="0.12")
    _add_material(asset, "nexus_mat", "0.37 0.43 0.45 1", specular="0.08")
    _add_material(asset, "task_cube", "0.92 0.24 0.15 1", specular="0.22")
    for name in mesh_names:
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"official_280pi_{name}", "file": str((obj_dir / f"{name}.obj").resolve())},
        )

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": "0.36 0.36 0.34",
            "diffuse": "0.76 0.74 0.68",
            "specular": "0.12 0.12 0.12",
        },
    )
    ET.SubElement(visual, "map", {"znear": "0.01", "zfar": "10"})
    ET.SubElement(visual, "global", {"offwidth": "960", "offheight": "720"})

    root_link = "base" if "base" in link_visuals else _first_urdf_link_name(urdf)
    worldbody = ET.SubElement(root, "worldbody")
    for node in _nexus_scene_nodes():
        worldbody.append(node)
    base = ET.SubElement(worldbody, "body", {"name": root_link, "pos": "0 0 0"})
    _add_urdf_visual_geoms(base, root_link, link_visuals, mesh_prefix="official_280pi")
    _append_urdf_children(
        base,
        root_link,
        children_by_parent,
        link_visuals,
        mesh_prefix="official_280pi",
    )
    _add_320_official_gripper_contact_pads(
        base,
        left_pos=ADAPTIVE_280_LEFT_FINGER_PAD_POS,
        right_pos=ADAPTIVE_280_RIGHT_FINGER_PAD_POS,
        pad_size=ADAPTIVE_280_FINGER_PAD_SIZE,
        friction=ADAPTIVE_280_FINGER_PAD_FRICTION,
        solref="0.004 1",
        solimp="0.9 0.99 0.001",
        mass="0.005",
    )
    _add_280_pi_position_actuators(root)

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(scene_path, encoding="utf-8", xml_declaration=True)


def _combined_280_pi_adaptive_urdf(arm_urdf: ET.Element, gripper_urdf: ET.Element) -> ET.Element:
    combined = ET.Element("robot", {"name": "mycobot_280_pi_adaptive_gripper"})
    for source in (arm_urdf, gripper_urdf):
        for child in source:
            if child.tag in {"link", "joint"}:
                combined.append(ET.fromstring(ET.tostring(child, encoding="unicode")))
    parent_link = _preferred_existing_link(arm_urdf, ("joint6_flange", "link6", "joint6"))
    child_link = _preferred_existing_link(gripper_urdf, ("gripper_base",))
    joint = ET.SubElement(combined, "joint", {"name": "joint6_to_adaptive_gripper_base", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": parent_link})
    ET.SubElement(joint, "child", {"link": child_link})
    ET.SubElement(joint, "origin", {"xyz": "0 -0.007 0.056", "rpy": "1.5708 0 0"})
    return combined


def _preferred_existing_link(urdf: ET.Element, preferred: tuple[str, ...]) -> str:
    names = [link.attrib["name"] for link in urdf.findall("link") if "name" in link.attrib]
    for name in preferred:
        if name in names:
            return name
    if not names:
        raise ValueError("URDF has no links")
    return names[-1]


def _first_urdf_link_name(urdf: ET.Element) -> str:
    link = urdf.find("link")
    if link is None or "name" not in link.attrib:
        raise ValueError("URDF has no named root link")
    return link.attrib["name"]


def _find_required_mesh(name: str, mesh_roots: tuple[Path, ...]) -> Path:
    for root in mesh_roots:
        candidate = root / f"{name}.dae"
        if candidate.exists():
            return candidate
    roots = ", ".join(str(root) for root in mesh_roots)
    raise FileNotFoundError(f"missing official mesh {name}.dae under: {roots}")


def _add_280_pi_position_actuators(root: ET.Element) -> None:
    actuator = ET.SubElement(root, "actuator")
    arm_ranges = {
        "joint2_to_joint1": "-2.93 2.93",
        "joint3_to_joint2": "-2.35 2.35",
        "joint4_to_joint3": "-2.53 2.53",
        "joint5_to_joint4": "-2.53 2.53",
        "joint6_to_joint5": "-2.93 2.93",
        "joint6output_to_joint6": "-3.14 3.14",
    }
    for joint_name in MYCOBOT_320_MODEL_JOINT_NAMES:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"act_{joint_name}",
                "joint": joint_name,
                "kp": "60",
                "ctrlrange": arm_ranges[joint_name],
                "ctrllimited": "true",
            },
        )


def _add_320_position_actuators(root: ET.Element) -> None:
    actuator = ET.SubElement(root, "actuator")
    arm_ranges = {
        "joint2_to_joint1": "-2.93 2.93",
        "joint3_to_joint2": "-2.35 2.35",
        "joint4_to_joint3": "-2.53 2.53",
        "joint5_to_joint4": "-2.53 2.53",
        "joint6_to_joint5": "-2.93 2.93",
        "joint6output_to_joint6": "-3.14 3.14",
    }
    for joint_name in MYCOBOT_320_MODEL_JOINT_NAMES:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"act_{joint_name}",
                "joint": joint_name,
                "kp": "80",
                "ctrlrange": arm_ranges[joint_name],
                "ctrllimited": "true",
            },
        )

def _urdf_link_visuals(urdf: ET.Element) -> dict[str, list[dict[str, str]]]:
    visuals: dict[str, list[dict[str, str]]] = {}
    for link in urdf.findall("link"):
        link_name = link.attrib["name"]
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            origin = visual.find("origin")
            visuals.setdefault(link_name, []).append(
                {
                    "mesh": Path(mesh.attrib["filename"]).name.removesuffix(".dae"),
                    "xyz": origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                    "rpy": origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0",
                }
            )
    return visuals


def _urdf_joints(urdf: ET.Element) -> list[dict[str, Any]]:
    joints: list[dict[str, Any]] = []
    for joint in urdf.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib.get("type", "fixed"),
                "parent": parent.attrib["link"],
                "child": child.attrib["link"],
                "xyz": origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                "rpy": origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0",
                "axis": axis.attrib.get("xyz", "0 0 1") if axis is not None else "0 0 1",
                "lower": limit.attrib.get("lower", "-3.14159") if limit is not None else "-3.14159",
                "upper": limit.attrib.get("upper", "3.14159") if limit is not None else "3.14159",
            }
        )
    return joints


def _append_urdf_children(
    parent: ET.Element,
    parent_link: str,
    children_by_parent: dict[str, list[dict[str, Any]]],
    link_visuals: dict[str, list[dict[str, str]]],
    *,
    mesh_prefix: str = "official_320",
    skip_gripper_tree: bool = False,
) -> None:
    for joint in children_by_parent.get(parent_link, []):
        if skip_gripper_tree and str(joint["child"]).startswith("gripper"):
            continue
        attrib = {
            "name": str(joint["child"]),
            "pos": _clean_float_string(str(joint["xyz"])),
            "euler": _clean_float_string(str(joint["rpy"])),
        }
        body = ET.SubElement(parent, "body", attrib)
        if joint["type"] != "fixed":
            ET.SubElement(
                body,
                "joint",
                {
                    "name": str(joint["name"]),
                    "type": "hinge",
                    "axis": _clean_float_string(str(joint["axis"])),
                    "range": f"{float(joint['lower'])} {float(joint['upper'])}",
                    "limited": "true",
                    "damping": "0.35",
                },
            )
        _add_urdf_visual_geoms(
            body,
            str(joint["child"]),
            link_visuals,
            mesh_prefix=mesh_prefix,
        )
        _append_urdf_children(
            body,
            str(joint["child"]),
            children_by_parent,
            link_visuals,
            mesh_prefix=mesh_prefix,
            skip_gripper_tree=skip_gripper_tree,
        )


def _add_urdf_visual_geoms(
    body: ET.Element,
    link_name: str,
    link_visuals: dict[str, list[dict[str, str]]],
    *,
    mesh_prefix: str = "official_320",
) -> None:
    for index, visual in enumerate(link_visuals.get(link_name, [])):
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{link_name}_visual_{index}",
                "type": "mesh",
                "mesh": f"{mesh_prefix}_{visual['mesh']}",
                "pos": _clean_float_string(visual["xyz"]),
                "euler": _clean_float_string(visual["rpy"]),
                "contype": "0",
                "conaffinity": "0",
            },
        )


def _add_320_official_gripper_contact_pads(
    base: ET.Element,
    *,
    left_pos: tuple[float, float, float] = ADAPTIVE_LEFT_FINGER_PAD_POS,
    right_pos: tuple[float, float, float] = ADAPTIVE_RIGHT_FINGER_PAD_POS,
    pad_size: tuple[float, float, float] = ADAPTIVE_FINGER_PAD_SIZE,
    friction: tuple[float, float, float] = ADAPTIVE_FINGER_PAD_FRICTION,
    solref: str = "0.001 1",
    solimp: str = "0.995 0.999 0.0001",
    mass: str = "0.02",
) -> None:
    gripper_base = base.find(".//body[@name='gripper_base']")
    if gripper_base is not None:
        ET.SubElement(
            gripper_base,
            "site",
            {"name": TCP_SITE, "pos": "0 0.065 -0.018", "size": "0.006", "rgba": "0 0 0 0"},
        )
    left = base.find(f".//body[@name='{ADAPTIVE_LEFT_FINGER_PAD_PARENT}']")
    if left is not None:
        _add_320_official_finger_pad(
            left,
            "left_finger_pad",
            pos=left_pos,
            pad_size=pad_size,
            friction=friction,
            solref=solref,
            solimp=solimp,
            mass=mass,
        )
    right = base.find(f".//body[@name='{ADAPTIVE_RIGHT_FINGER_PAD_PARENT}']")
    if right is not None:
        _add_320_official_finger_pad(
            right,
            "right_finger_pad",
            pos=right_pos,
            pad_size=pad_size,
            friction=friction,
            solref=solref,
            solimp=solimp,
            mass=mass,
        )


def _add_320_official_finger_pad(
    parent: ET.Element,
    pad_name: str,
    *,
    pos: tuple[float, float, float],
    pad_size: tuple[float, float, float] = ADAPTIVE_FINGER_PAD_SIZE,
    friction: tuple[float, float, float] = ADAPTIVE_FINGER_PAD_FRICTION,
    solref: str = "0.001 1",
    solimp: str = "0.995 0.999 0.0001",
    mass: str = "0.02",
) -> None:
    ET.SubElement(
        parent,
        "geom",
        {
            "name": pad_name,
            "type": "box",
            "pos": _float_sequence(pos),
            "euler": _float_sequence(ADAPTIVE_FINGER_PAD_EULER),
            "size": _float_sequence(pad_size),
            "rgba": "0.08 0.08 0.08 0",
            "friction": _float_sequence(friction),
            "condim": str(ADAPTIVE_FINGER_PAD_CONDIM),
            "solref": solref,
            "solimp": solimp,
            "contype": "1",
            "conaffinity": "1",
            "mass": mass,
        },
    )


def _float_sequence(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def _clean_float_string(raw: str) -> str:
    return " ".join(str(float(value)) for value in raw.split())


def _nexus_scene_nodes() -> list[ET.Element]:
    return [
        ET.Element(
            "light",
            {
                "name": "nexus_key_light",
                "pos": "-0.55 -0.75 1.35",
                "dir": "0.35 0.45 -1",
                "directional": "true",
                "diffuse": "0.88 0.84 0.72",
            },
        ),
        ET.Element(
            "light",
            {
                "name": "nexus_fill_light",
                "pos": "0.65 0.35 0.95",
                "diffuse": "0.28 0.32 0.36",
            },
        ),
        ET.Element(
            "geom",
            {
                "name": "nexus_floor",
                "type": "plane",
                "pos": "0 0 -0.006",
                "size": "1.2 1.2 0.01",
                "material": "nexus_floor",
                "contype": "1",
                "conaffinity": "1",
            },
        ),
        ET.Element(
            "geom",
            {
                "name": "nexus_work_mat",
                "type": "box",
                "pos": "-0.18 -0.12 0.004",
                "size": "0.36 0.26 0.004",
                "material": "nexus_mat",
                "contype": "1",
                "conaffinity": "1",
            },
        ),
        _dynamic_cube_body_node(),
    ]


def _official_gripper_mesh_dir(official_gripper_root: Path | None) -> Path | None:
    if official_gripper_root is None:
        return None
    root = official_gripper_root.expanduser()
    mesh_dir = root / OFFICIAL_GRIPPER_MESH_RELATIVE_PATH
    missing = [
        name
        for name in OFFICIAL_GRIPPER_MESH_NAMES
        if not (mesh_dir / f"{name}.dae").exists()
    ]
    if missing:
        raise FileNotFoundError(
            "missing official myCobot parallel gripper meshes under "
            f"{mesh_dir}: {', '.join(missing)}"
        )
    return mesh_dir


def _prepare_official_gripper_obj_assets(mesh_dir: Path, output_root: Path) -> Path:
    obj_dir = output_root / "official_gripper_meshes"
    obj_dir.mkdir(parents=True, exist_ok=True)
    for name in OFFICIAL_GRIPPER_MESH_NAMES:
        _convert_collada_mesh_to_obj(mesh_dir / f"{name}.dae", obj_dir / f"{name}.obj")
    return obj_dir


def _convert_collada_mesh_to_obj(
    dae_path: Path,
    obj_path: Path,
    *,
    bake_visual_scene: bool = True,
) -> None:
    root = ET.parse(dae_path).getroot()
    namespace = {"c": root.tag.partition("}")[0].strip("{")}
    unit = root.find("c:asset/c:unit", namespace)
    scale = float(unit.attrib.get("meter", "1.0")) if unit is not None else 1.0
    geometries = _collada_geometry_meshes(root, namespace)
    library_nodes = {
        f"#{node.attrib['id']}": node
        for node in root.findall(".//c:library_nodes/c:node", namespace)
        if "id" in node.attrib
    }
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    if bake_visual_scene:
        for scene in root.findall(".//c:library_visual_scenes/c:visual_scene", namespace):
            for node in scene.findall("c:node", namespace):
                _append_collada_node_instances(
                    node,
                    namespace,
                    geometries,
                    library_nodes,
                    _identity_matrix4(),
                    scale,
                    vertices,
                    faces,
                )

    if not vertices:
        for mesh_vertices, mesh_faces in geometries.values():
            offset = len(vertices)
            vertices.extend(
                (x * scale, y * scale, z * scale)
                for x, y, z in mesh_vertices
            )
            faces.extend(
                (a + offset + 1, b + offset + 1, c + offset + 1)
                for a, b, c in mesh_faces
            )

    if not vertices or not faces:
        raise ValueError(f"could not extract triangles from official gripper mesh: {dae_path}")

    with obj_path.open("w", encoding="utf-8") as file:
        file.write(f"# converted from official myCobot ROS Collada mesh: {dae_path.name}\n")
        for x, y, z in vertices:
            file.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in faces:
            file.write(f"f {a} {b} {c}\n")


def _collada_geometry_meshes(
    root: ET.Element,
    namespace: dict[str, str],
) -> dict[str, tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]]:
    geometries: dict[str, tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]] = {}
    for geometry in root.findall(".//c:library_geometries/c:geometry", namespace):
        geometry_id = geometry.attrib.get("id")
        mesh = geometry.find("c:mesh", namespace)
        if geometry_id is None or mesh is None:
            continue
        position_source = _collada_position_source(mesh, namespace)
        if position_source is None:
            continue
        geometries[f"#{geometry_id}"] = (
            _collada_float_triplets(position_source, namespace),
            _collada_triangle_indices(mesh, namespace),
        )
    return geometries


def _append_collada_node_instances(
    node: ET.Element,
    namespace: dict[str, str],
    geometries: dict[str, tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]],
    library_nodes: dict[str, ET.Element],
    parent_matrix: list[list[float]],
    scale: float,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> None:
    matrix = _matrix_multiply(parent_matrix, _collada_node_matrix(node, namespace))
    for instance in node.findall("c:instance_geometry", namespace):
        geometry = geometries.get(instance.attrib.get("url", ""))
        if geometry is None:
            continue
        mesh_vertices, mesh_faces = geometry
        offset = len(vertices)
        vertices.extend(_transform_point(matrix, vertex, scale=scale) for vertex in mesh_vertices)
        faces.extend(
            (a + offset + 1, b + offset + 1, c + offset + 1)
            for a, b, c in mesh_faces
        )
    for instance in node.findall("c:instance_node", namespace):
        referenced = library_nodes.get(instance.attrib.get("url", ""))
        if referenced is not None:
            _append_collada_node_instances(
                referenced,
                namespace,
                geometries,
                library_nodes,
                matrix,
                scale,
                vertices,
                faces,
            )
    for child in node.findall("c:node", namespace):
        _append_collada_node_instances(
            child,
            namespace,
            geometries,
            library_nodes,
            matrix,
            scale,
            vertices,
            faces,
        )


def _collada_node_matrix(node: ET.Element, namespace: dict[str, str]) -> list[list[float]]:
    matrix_node = node.find("c:matrix", namespace)
    if matrix_node is None or matrix_node.text is None:
        return _identity_matrix4()
    values = [float(value) for value in matrix_node.text.split()]
    if len(values) != 16:
        return _identity_matrix4()
    return [values[index:index + 4] for index in range(0, 16, 4)]


def _identity_matrix4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matrix_multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def _transform_point(
    matrix: list[list[float]],
    point: tuple[float, float, float],
    *,
    scale: float,
) -> tuple[float, float, float]:
    x, y, z = point
    return (
        (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3]) * scale,
        (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3]) * scale,
        (matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3]) * scale,
    )


def _collada_position_source(mesh: ET.Element, namespace: dict[str, str]) -> ET.Element | None:
    for source in mesh.findall("c:source", namespace):
        if source.attrib.get("name") == "position" or "position" in source.attrib.get("id", ""):
            return source
    return None


def _collada_float_triplets(
    source: ET.Element,
    namespace: dict[str, str],
) -> list[tuple[float, float, float]]:
    array = source.find("c:float_array", namespace)
    if array is None or array.text is None:
        return []
    values = [float(value) for value in array.text.split()]
    return [
        (values[index], values[index + 1], values[index + 2])
        for index in range(0, len(values) - 2, 3)
    ]


def _collada_triangle_indices(
    mesh: ET.Element,
    namespace: dict[str, str],
) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    for triangle_node in mesh.findall("c:triangles", namespace):
        inputs = triangle_node.findall("c:input", namespace)
        stride = max(int(item.attrib.get("offset", "0")) for item in inputs) + 1
        vertex_offset = next(
            int(item.attrib.get("offset", "0"))
            for item in inputs
            if item.attrib.get("semantic") == "VERTEX"
        )
        p_node = triangle_node.find("c:p", namespace)
        if p_node is None or p_node.text is None:
            continue
        raw = [int(value) for value in p_node.text.split()]
        for index in range(0, len(raw), stride * 3):
            triangles.append(
                (
                    raw[index + vertex_offset],
                    raw[index + stride + vertex_offset],
                    raw[index + stride * 2 + vertex_offset],
                )
            )
    for polygon_node in mesh.findall("c:polygons", namespace):
        inputs = polygon_node.findall("c:input", namespace)
        if not inputs:
            continue
        stride = max(int(item.attrib.get("offset", "0")) for item in inputs) + 1
        vertex_offset = next(
            (
                int(item.attrib.get("offset", "0"))
                for item in inputs
                if item.attrib.get("semantic") == "VERTEX"
            ),
            0,
        )
        for p_node in polygon_node.findall("c:p", namespace):
            if p_node.text is None:
                continue
            raw = [int(value) for value in p_node.text.split()]
            polygon = [
                raw[index + vertex_offset]
                for index in range(0, len(raw), stride)
                if index + vertex_offset < len(raw)
            ]
            triangles.extend(_triangulate_polygon_indices(polygon))
    return triangles


def _triangulate_polygon_indices(indices: list[int]) -> list[tuple[int, int, int]]:
    if len(indices) < 3:
        return []
    return [
        (indices[0], indices[index], indices[index + 1])
        for index in range(1, len(indices) - 1)
    ]


def _add_official_gripper_assets(asset: ET.Element, mesh_dir: Path) -> None:
    for name in OFFICIAL_GRIPPER_MESH_NAMES:
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": f"official_{name}",
                "file": str((mesh_dir / f"{name}.obj").resolve()),
            },
        )


def _official_parallel_gripper_node() -> ET.Element:
    base = ET.Element(
        "body",
        {
            "name": "official_parallel_gripper",
            "pos": "0 0 0.034",
            "euler": "1.579 0 0",
        },
    )
    _add_mesh_geom(base, "gripper_base", pos="0 0 0", euler="-1.5708 0 0")
    ET.SubElement(
        base,
        "site",
        {"name": TCP_SITE, "pos": "0 0.035 0", "size": "0.006", "rgba": "0 0 0 0"},
    )

    left = _slide_body(
        base,
        name="gripper_left",
        joint="gripper_controller",
        axis="1 0 0",
        range_="-0.007 0",
    )
    _add_mesh_geom(left, "gripper_left", pos="0 0 0", euler="-1.5708 0 0")
    _add_proxy_pad(left, "left_finger_pad", pos="-0.018 0.035 0")

    right = _slide_body(
        base,
        name="gripper_right",
        joint="gripper_base_to_gripper_left",
        axis="1 0 0",
        range_="-0.007 0",
    )
    _add_mesh_geom(right, "gripper_right", pos="0 0 0", euler="-1.5708 0 0")
    _add_proxy_pad(right, "right_finger_pad", pos="0.018 0.035 0")
    return base


def _slide_body(
    parent: ET.Element,
    *,
    name: str,
    joint: str,
    axis: str,
    range_: str,
) -> ET.Element:
    body = ET.SubElement(parent, "body", {"name": name, "pos": "0 0 0"})
    ET.SubElement(
        body,
        "joint",
        {
            "name": joint,
            "type": "slide",
            "axis": axis,
            "range": range_,
            "limited": "true",
            "damping": "0.18",
        },
    )
    return body


def _add_mesh_geom(parent: ET.Element, mesh_name: str, *, pos: str, euler: str = "0 0 0") -> None:
    attrib = {
        "name": f"{mesh_name}_visual",
        "type": "mesh",
        "mesh": f"official_{mesh_name}",
        "pos": pos,
        "contype": "0",
        "conaffinity": "0",
    }
    if euler != "0 0 0":
        attrib["euler"] = euler
    ET.SubElement(parent, "geom", attrib)


def _add_proxy_pad(parent: ET.Element, name: str, *, pos: str) -> None:
    ET.SubElement(
        parent,
        "geom",
        {
            "name": name,
            "type": "box",
            "pos": pos,
            "size": "0.012 0.035 0.024",
            "rgba": "0.08 0.08 0.08 0",
            "friction": "1.5 0.1 0.1",
            "contype": "1",
            "conaffinity": "1",
            "mass": "0.01",
        },
    )


def _synthetic_parallel_gripper_node() -> ET.Element:
    base = ET.Element("body", {"name": "synthetic_parallel_gripper", "pos": "0 0 -0.055"})
    ET.SubElement(
        base,
        "geom",
        {
            "name": "gripper_palm",
            "type": "box",
            "pos": "0 0 0",
            "size": "0.032 0.018 0.012",
            "rgba": "0.18 0.18 0.17 1",
            "contype": "1",
            "conaffinity": "1",
            "mass": "0.03",
        },
    )
    ET.SubElement(
        base,
        "site",
        {"name": TCP_SITE, "pos": "0 0 -0.085", "size": "0.006", "rgba": "0 0 0 0"},
    )
    left = ET.SubElement(base, "body", {"name": "left_finger", "pos": "0 0.042 -0.05"})
    ET.SubElement(
        left,
        "joint",
        {
            "name": "left_finger_slide",
            "type": "slide",
            "axis": "0 -1 0",
            "range": "0 0.028",
            "limited": "true",
            "damping": "0.6",
        },
    )
    ET.SubElement(
        left,
        "geom",
        {
            "name": "left_finger_pad",
            "type": "box",
            "pos": "0 0 -0.035",
            "size": "0.012 0.006 0.04",
            "rgba": "0.08 0.08 0.08 1",
            "friction": "1.5 0.1 0.1",
            "contype": "1",
            "conaffinity": "1",
            "mass": "0.02",
        },
    )
    right = ET.SubElement(base, "body", {"name": "right_finger", "pos": "0 -0.042 -0.05"})
    ET.SubElement(
        right,
        "joint",
        {
            "name": "right_finger_slide",
            "type": "slide",
            "axis": "0 1 0",
            "range": "0 0.028",
            "limited": "true",
            "damping": "0.6",
        },
    )
    ET.SubElement(
        right,
        "geom",
        {
            "name": "right_finger_pad",
            "type": "box",
            "pos": "0 0 -0.035",
            "size": "0.012 0.006 0.04",
            "rgba": "0.08 0.08 0.08 1",
            "friction": "1.5 0.1 0.1",
            "contype": "1",
            "conaffinity": "1",
            "mass": "0.02",
        },
    )
    return base


def _dynamic_cube_body_node() -> ET.Element:
    cube = ET.Element(
        "body",
        {
            "name": TASK_CUBE_BODY,
            "pos": f"{TASK_CUBE_POS[0]} {TASK_CUBE_POS[1]} {TASK_CUBE_POS[2]}",
        },
    )
    ET.SubElement(cube, "freejoint", {"name": "task_cube_freejoint"})
    ET.SubElement(
        cube,
        "geom",
        {
            "name": TASK_CUBE_GEOM,
            "type": "box",
            "size": f"{TASK_CUBE_HALF_SIZE} {TASK_CUBE_HALF_SIZE} {TASK_CUBE_HALF_SIZE}",
            "material": "task_cube",
            "mass": "0.005",
            "friction": "60.0 6.0 6.0",
            "condim": "6",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    return cube


def _joint_qpos_indices(mujoco: Any, model: Any, joint_names: list[str]) -> list[int]:
    indices: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"joint not found in MuJoCo model: {name}")
        indices.append(int(model.jnt_qposadr[joint_id]))
    return indices


def _joint_dof_indices(mujoco: Any, model: Any, joint_names: list[str]) -> list[int]:
    indices: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"joint not found in MuJoCo model: {name}")
        indices.append(int(model.jnt_dofadr[joint_id]))
    return indices


def _named_joint_qpos_indices(mujoco: Any, model: Any, joint_names: list[str]) -> list[int]:
    indices: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"joint not found in MuJoCo model: {name}")
        indices.append(int(model.jnt_qposadr[joint_id]))
    return indices


def _named_actuator_indices(mujoco: Any, model: Any, actuator_names: list[str]) -> list[int]:
    indices: list[int] = []
    for name in actuator_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"actuator not found in MuJoCo model: {name}")
        indices.append(int(actuator_id))
    return indices


def _has_all_joints(mujoco: Any, model: Any, joint_names: list[str]) -> bool:
    return all(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        for name in joint_names
    )


def _joint_ranges(model: Any, qpos_indices: list[int]) -> tuple[list[float], list[float]]:
    low: list[float] = []
    high: list[float] = []
    for qpos_index in qpos_indices:
        joint_id = int((model.jnt_qposadr == qpos_index).nonzero()[0][0])
        if int(model.jnt_limited[joint_id]):
            low.append(float(model.jnt_range[joint_id][0]))
            high.append(float(model.jnt_range[joint_id][1]))
        else:
            low.append(-math.pi)
            high.append(math.pi)
    return low, high


def _neutral_qpos(*, seed: int, low: list[float], high: list[float]) -> list[float]:
    phase = seed * 0.017
    values = []
    for index, (lo, hi) in enumerate(zip(low, high, strict=True)):
        center = (lo + hi) / 2.0
        radius = min((hi - lo) * 0.08, 0.18)
        values.append(center + radius * math.sin(phase + index * 0.73))
    return values


def _clip(values: list[float], low: list[float], high: list[float]) -> list[float]:
    return [max(lo, min(hi, value)) for value, lo, hi in zip(values, low, high, strict=True)]


def _body_position(mujoco: Any, model: Any, data: Any, body_name: str) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"body not found in MuJoCo model: {body_name}")
    pos = data.xpos[body_id]
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _gripper_cube_contacts(mujoco: Any, model: Any, data: Any) -> int:
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TASK_CUBE_GEOM)
    finger_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad"),
    }
    finger_ids = {geom_id for geom_id in finger_ids if geom_id >= 0}
    if cube_id < 0 or not finger_ids:
        return 0
    contacts = 0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube_id in pair and pair.intersection(finger_ids):
            contacts += 1
    return contacts


def _gripper_cube_contact_pad_count(mujoco: Any, model: Any, data: Any) -> int:
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TASK_CUBE_GEOM)
    pad_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad"),
    }
    pad_ids = {geom_id for geom_id in pad_ids if geom_id >= 0}
    if cube_id < 0 or not pad_ids:
        return 0
    contacted_pad_ids: set[int] = set()
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube_id in pair:
            contacted_pad_ids.update(pair.intersection(pad_ids))
    return len(contacted_pad_ids)


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b, strict=True)))


def _add_material(
    asset: ET.Element,
    name: str,
    rgba: str,
    *,
    specular: str,
) -> None:
    ET.SubElement(
        asset,
        "material",
        {
            "name": name,
            "rgba": rgba,
            "specular": specular,
            "shininess": "0.28",
        },
    )


def _child_index(root: ET.Element, tag: str) -> int:
    for index, child in enumerate(list(root)):
        if child.tag == tag:
            return index
    return -1


def _write_bmp(path: Path, rgb: Any) -> None:
    height = int(rgb.shape[0])
    width = int(rgb.shape[1])
    row_stride = (width * 3 + 3) & ~3
    image_size = row_stride * height
    file_size = 14 + 40 + image_size
    with path.open("wb") as file:
        file.write(b"BM")
        file.write(struct.pack("<IHHI", file_size, 0, 0, 54))
        file.write(
            struct.pack(
                "<IIIHHIIIIII",
                40,
                width,
                height,
                1,
                24,
                0,
                image_size,
                2835,
                2835,
                0,
                0,
            )
        )
        padding = b"\x00" * (row_stride - width * 3)
        for y in range(height - 1, -1, -1):
            row = rgb[y, :, :3]
            file.write(row[:, ::-1].tobytes())
            file.write(padding)


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe
