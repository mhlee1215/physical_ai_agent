from __future__ import annotations

import json
import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MYCOBOT_MODEL_RELATIVE_PATH = Path("xml/mycobot_280jn_mujoco.xml")
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


@dataclass(frozen=True)
class MyCobotNexusConfig:
    asset_root: Path
    work_dir: Path
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
    steps: int
    observation_dim: int
    action_dim: int
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
        build_mycobot_nexus_scene_model(model_path=self.model_path, scene_path=self.scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._step = 0
        self._qpos_indices = _joint_qpos_indices(mujoco, self.model)
        self._low, self._high = _joint_ranges(self.model, self._qpos_indices)
        self.action_dim = len(MYCOBOT_TEACHER_JOINT_NAMES)
        self.observation_dim = len(MYCOBOT_TEACHER_JOINT_NAMES) + 3

    def reset(self, seed: int = 0) -> tuple[list[float], dict[str, Any]]:
        self._mujoco.mj_resetData(self.model, self.data)
        self._step = 0
        neutral = _neutral_qpos(seed=seed, low=self._low, high=self._high)
        for qpos_index, value in zip(self._qpos_indices, neutral, strict=True):
            self.data.qpos[qpos_index] = value
        self._mujoco.mj_forward(self.model, self.data)
        return self._observation(gripper=0.0), self._info(gripper=0.0)

    def step(self, action: list[float]) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        values = sanitize_teacher_action(action)
        arm_target = _clip(values[: len(self._qpos_indices)], self._low, self._high)
        gripper = values[-1]
        for target, qpos_index in zip(arm_target, self._qpos_indices, strict=True):
            current = float(self.data.qpos[qpos_index])
            self.data.qpos[qpos_index] = current + self.config.control_alpha * (target - current)
        self._mujoco.mj_step(self.model, self.data)
        self._step += 1
        obs = self._observation(gripper=gripper)
        info = self._info(gripper=gripper)
        reward = -float(info["tcp_to_cube_dist"])
        terminated = bool(info["tcp_to_cube_dist"] < 0.08)
        return obs, reward, terminated, False, info

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
        return [*qpos, float(gripper), *_cube_position()]

    def _info(self, *, gripper: float) -> dict[str, Any]:
        tcp = _tcp_position_from_data(self.data)
        cube = _cube_position()
        return {
            "step": self._step,
            "joint_names": MYCOBOT_TEACHER_JOINT_NAMES,
            "cube_position": cube,
            "tcp_position": tcp,
            "tcp_to_cube_dist": _distance(tcp, cube),
            "gripper_command": float(gripper),
            "success": False,
            "success_label": "not_claimed_poc_kinematic_step",
            "scene_path": str(self.scene_path),
        }

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
) -> MyCobotNexusSmokeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "mycobot_nexus_trace.jsonl"
    frame_path = output_dir / "mycobot_nexus_frame.bmp"
    report_path = output_dir / "mycobot_nexus_report.json"
    env = MyCobotNexusEnv(
        MyCobotNexusConfig(
            asset_root=asset_root,
            work_dir=output_dir,
            width=width,
            height=height,
        )
    )
    records: list[MyCobotNexusStep] = []
    obs, info = env.reset(seed=seed)
    try:
        for step in range(steps):
            action = sample_mycobot_nexus_action(step, steps)
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
    result = MyCobotNexusSmokeResult(
        status="passed",
        steps=len(records),
        observation_dim=len(obs),
        action_dim=len(MYCOBOT_TEACHER_JOINT_NAMES),
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
        "asset_source": "https://github.com/elephantrobotics/mycobot_mujoco",
        "model_relative_path": str(MYCOBOT_MODEL_RELATIVE_PATH),
        "joint_order": MYCOBOT_TEACHER_JOINT_NAMES,
        "action_dim": len(MYCOBOT_TEACHER_JOINT_NAMES),
        "observation_dim": len(MYCOBOT_TEACHER_JOINT_NAMES) + 3,
        "task_objects": ["task_cube", "nexus_work_mat"],
        "real_robot_execution": "disabled",
        "poc_boundary": (
            "Kinematic qpos-target MuJoCo env. It steps a real myCobot model in a "
            "Nexus-style cube scene, but does not yet claim contact grasp success."
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


def build_mycobot_nexus_scene_model(*, model_path: Path, scene_path: Path) -> None:
    tree = ET.parse(model_path)
    root = tree.getroot()

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
                "contype": "0",
                "conaffinity": "0",
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
                "contype": "0",
                "conaffinity": "0",
            },
        ),
        ET.Element(
            "geom",
            {
                "name": "task_cube",
                "type": "box",
                "pos": "-0.34 -0.18 0.044",
                "size": "0.04 0.04 0.04",
                "material": "task_cube",
                "contype": "0",
                "conaffinity": "0",
            },
        ),
    ]


def _joint_qpos_indices(mujoco: Any, model: Any) -> list[int]:
    indices: list[int] = []
    for name in MYCOBOT_MODEL_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"joint not found in MuJoCo model: {name}")
        indices.append(int(model.jnt_qposadr[joint_id]))
    return indices


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


def _cube_position() -> list[float]:
    return [-0.34, -0.18, 0.044]


def _tcp_position_from_data(data: Any) -> list[float]:
    # The official model has no named TCP site. The last body is the closest
    # stable model-owned proxy for POC verifier distance.
    pos = data.xpos[-1]
    return [float(pos[0]), float(pos[1]), float(pos[2])]


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
