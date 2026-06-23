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
OFFICIAL_GRIPPER_MESH_NAMES = [
    "gripper_base",
    "gripper_left",
    "gripper_right",
]
MYCOBOT_MODEL_JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint7_to_joint6",
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
TASK_CUBE_BODY = "task_cube_body"
TASK_CUBE_GEOM = "task_cube"
TASK_CUBE_POS = (-0.25, -0.13, 0.023)
TASK_CUBE_HALF_SIZE = 0.015
TCP_SITE = "mycobot_tcp_site"


@dataclass(frozen=True)
class MyCobotNexusConfig:
    asset_root: Path
    work_dir: Path
    official_gripper_root: Path | None = None
    width: int = 640
    height: int = 360
    control_alpha: float = 0.35


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
    grasp_success: bool
    trace_path: str
    frame_path: str
    report_path: str
    scene_path: str


class MyCobotNexusEnv:
    """Mac-local myCobot task environment with a Nexus-style MuJoCo world.

    This is a simulation POC, not only a renderer: it owns MuJoCo model/data,
    exposes reset/step/render, and records task state around a visible cube.
    The official myCobot MJCF has no actuator section, so step() currently uses
    deterministic qpos-target stepping. That keeps the env executable while the
    next POC can add calibrated actuators and contact-based success.
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
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"missing myCobot MuJoCo model: {self.model_path}. "
                "Clone https://github.com/elephantrobotics/mycobot_mujoco and pass --asset-root."
            )
        self.scene_path = self.work_dir / "mycobot_nexus_scene.xml"
        build_mycobot_nexus_scene_model(
            model_path=self.model_path,
            scene_path=self.scene_path,
            official_gripper_root=config.official_gripper_root,
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._step = 0
        self._qpos_indices = _joint_qpos_indices(mujoco, self.model)
        self._dof_indices = _joint_dof_indices(mujoco, self.model)
        self._uses_official_gripper = _has_all_joints(
            mujoco,
            self.model,
            OFFICIAL_GRIPPER_JOINT_NAMES,
        )
        gripper_joint_names = (
            OFFICIAL_GRIPPER_JOINT_NAMES
            if self._uses_official_gripper
            else SYNTHETIC_GRIPPER_JOINT_NAMES
        )
        self._gripper_qpos_indices = _named_joint_qpos_indices(
            mujoco,
            self.model,
            gripper_joint_names,
        )
        self._low, self._high = _joint_ranges(self.model, self._qpos_indices)
        self.action_dim = len(MYCOBOT_TEACHER_JOINT_NAMES)
        self.observation_dim = len(MYCOBOT_TEACHER_JOINT_NAMES) + 3

    def reset(self, seed: int = 0) -> tuple[list[float], dict[str, Any]]:
        self._mujoco.mj_resetData(self.model, self.data)
        self._step = 0
        neutral = _neutral_qpos(seed=seed, low=self._low, high=self._high)
        for qpos_index, value in zip(self._qpos_indices, neutral, strict=True):
            self.data.qpos[qpos_index] = value
        self._set_gripper(command=1.0)
        self._mujoco.mj_forward(self.model, self.data)
        return self._observation(gripper=1.0), self._info(gripper=1.0)

    def step(self, action: list[float]) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        values = sanitize_teacher_action(action)
        arm_target = _clip(values[: len(self._qpos_indices)], self._low, self._high)
        gripper = values[-1]
        for target, qpos_index in zip(arm_target, self._qpos_indices, strict=True):
            current = float(self.data.qpos[qpos_index])
            self.data.qpos[qpos_index] = current + self.config.control_alpha * (target - current)
        self._set_gripper(command=gripper)
        self._mujoco.mj_step(self.model, self.data)
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
        cube_lift = cube[2] - TASK_CUBE_POS[2]
        return {
            "step": self._step,
            "joint_names": MYCOBOT_TEACHER_JOINT_NAMES,
            "cube_position": cube,
            "tcp_position": tcp,
            "tcp_to_cube_dist": _distance(tcp, cube),
            "gripper_command": float(gripper),
            "gripper_cube_contacts": contacts,
            "cube_lift": cube_lift,
            "success": bool(cube_lift > 0.025 and contacts > 0),
            "success_label": (
                "contact_lift_success" if cube_lift > 0.025 and contacts > 0 else "not_success"
            ),
            "scene_path": str(self.scene_path),
        }

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

    def _make_camera(self) -> Any:
        camera = self._mujoco.MjvCamera()
        camera.type = self._mujoco.mjtCamera.mjCAMERA_FREE
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
    grasp_success = bool(records and any(bool(record.info["success"]) for record in records))
    result = MyCobotNexusSmokeResult(
        status="passed",
        policy=policy,
        steps=len(records),
        observation_dim=len(obs),
        action_dim=len(MYCOBOT_TEACHER_JOINT_NAMES),
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
        grasp_success=grasp_success,
        trace_path=str(trace_path),
        frame_path=str(frame_path),
        report_path=str(report_path),
        scene_path=scene_path,
    )
    report_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def mycobot_nexus_contract() -> dict[str, Any]:
    return {
        "env": "MyCobotNexusEnv",
        "surface": ["reset(seed)", "step(action)", "render()", "close()"],
        "policies": ["sample", "cube-approach", "grasp-lift"],
        "asset_source": "https://github.com/elephantrobotics/mycobot_mujoco",
        "model_relative_path": str(MYCOBOT_MODEL_RELATIVE_PATH),
        "joint_order": MYCOBOT_TEACHER_JOINT_NAMES,
        "action_dim": len(MYCOBOT_TEACHER_JOINT_NAMES),
        "observation_dim": len(MYCOBOT_TEACHER_JOINT_NAMES) + 3,
        "task_objects": [
            "task_cube",
            "nexus_work_mat",
            "official_parallel_gripper",
            "synthetic_parallel_gripper_fallback",
        ],
        "official_gripper_asset_source": "https://github.com/elephantrobotics/mycobot_ros",
        "official_gripper_relative_path": str(OFFICIAL_GRIPPER_MESH_RELATIVE_PATH),
        "real_robot_execution": "disabled",
        "poc_boundary": (
            "Kinematic qpos-target MuJoCo env. It steps a real myCobot model in a "
            "Nexus-style cube scene. The preferred gripper path uses official "
            "mycobot_ros 280 JN parallel-gripper visual meshes with transparent "
            "MuJoCo contact proxies; it does not yet claim calibrated force control."
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


def build_mycobot_nexus_scene_model(
    *,
    model_path: Path,
    scene_path: Path,
    official_gripper_root: Path | None = None,
) -> None:
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


def _convert_collada_mesh_to_obj(dae_path: Path, obj_path: Path) -> None:
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
    return triangles


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
            "mass": "0.02",
            "friction": "1.2 0.08 0.08",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    return cube


def _joint_qpos_indices(mujoco: Any, model: Any) -> list[int]:
    indices: list[int] = []
    for name in MYCOBOT_MODEL_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"joint not found in MuJoCo model: {name}")
        indices.append(int(model.jnt_qposadr[joint_id]))
    return indices


def _joint_dof_indices(mujoco: Any, model: Any) -> list[int]:
    indices: list[int] = []
    for name in MYCOBOT_MODEL_JOINT_NAMES:
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
    if cube_id < 0 or any(geom_id < 0 for geom_id in finger_ids):
        return 0
    contacts = 0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube_id in pair and pair.intersection(finger_ids):
            contacts += 1
    return contacts


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
