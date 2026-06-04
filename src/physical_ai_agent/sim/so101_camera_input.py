from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, sample_action


@dataclass(frozen=True)
class SO101CameraSpec:
    env_id: str
    observation_shape: list[int]
    camera_names: list[str]
    virtual_camera_names: list[str]
    info_keys: list[str]


@dataclass(frozen=True)
class SO101InputFrame:
    step: int
    observation: list[float]
    action: list[float]
    reward: float
    camera_frames: dict[str, str]


@dataclass(frozen=True)
class SO101InputCapture:
    env_id: str
    camera_specs: list[SO101CameraSpec]
    frames: list[SO101InputFrame]
    policy_input_names: list[str]
    debug_input_names: list[str]
    lerobot_policy_feature_keys: list[str]
    manifest_path: str
    preview_path: str
    preview_gif_path: str


DEFAULT_SO101_ENV_IDS = (
    "MuJoCoReach-v1",
    "MuJoCoMove-v1",
    "MuJoCoPickLift-v1",
    "MuJoCoPickAndPlace-v1",
)

POLICY_CAMERA_NAMES = ("wrist_cam", "egocentric_cam")
DEBUG_CAMERA_NAMES = ("top_down",)
DEFAULT_VIRTUAL_CAMERA_NAMES = ("egocentric_cam", "top_down")


def inspect_so101_camera_specs(env_ids: tuple[str, ...] = DEFAULT_SO101_ENV_IDS) -> list[SO101CameraSpec]:
    import gymnasium as gym
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    specs: list[SO101CameraSpec] = []
    for env_id in env_ids:
        env = gym.make(env_id, render_mode=None)
        try:
            obs, info = env.reset(seed=0)
            camera_names = [env.unwrapped.model.camera(index).name for index in range(env.unwrapped.model.ncam)]
            specs.append(
                SO101CameraSpec(
                    env_id=env_id,
                    observation_shape=list(getattr(obs, "shape", [])),
                    camera_names=camera_names,
                    virtual_camera_names=list(DEFAULT_VIRTUAL_CAMERA_NAMES),
                    info_keys=sorted(info.keys()),
                )
            )
        finally:
            env.close()
    return specs


def capture_so101_inputs(
    output_dir: Path,
    env_id: str = DEFAULT_SO101_ENV_ID,
    steps: int = 8,
    seed: int = 0,
    width: int = 640,
    height: int = 480,
    include_virtual_cameras: bool = False,
    camera_names: tuple[str, ...] | None = None,
) -> SO101InputCapture:
    import gymnasium as gym
    import mujoco
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    _clear_dir(frames_dir)

    specs = inspect_so101_camera_specs()
    env = gym.make(env_id, render_mode=None)
    renderers: dict[str, Any] = {}
    records: list[SO101InputFrame] = []
    try:
        obs, _info = env.reset(seed=seed)
        named_camera_names = [
            env.unwrapped.model.camera(index).name
            for index in range(env.unwrapped.model.ncam)
        ]
        input_names = list(camera_names) if camera_names is not None else named_camera_names[:]
        if include_virtual_cameras and camera_names is None:
            input_names.extend(DEFAULT_VIRTUAL_CAMERA_NAMES)
        renderers = {name: mujoco.Renderer(env.unwrapped.model, height=height, width=width) for name in input_names}
        for step in range(steps):
            action = sample_action(env.action_space, step / max(1, steps - 1))
            obs, reward, terminated, truncated, _info = env.step(action)
            camera_frames: dict[str, str] = {}
            for camera_name, renderer in renderers.items():
                camera = _make_camera(env, camera_name)
                renderer.update_scene(env.unwrapped.data, camera=camera)
                pixels = renderer.render()
                camera_path = frames_dir / f"step_{step:03d}_{camera_name}.png"
                _write_image(pixels, camera_path)
                camera_frames[camera_name] = str(camera_path)
            records.append(
                SO101InputFrame(
                    step=step,
                    observation=_as_float_list(obs),
                    action=[float(value) for value in action],
                    reward=float(reward),
                    camera_frames=camera_frames,
                )
            )
            if terminated or truncated:
                break
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()

    manifest_path = output_dir / "input_manifest.json"
    preview_path = output_dir / "input_preview.png"
    preview_gif_path = output_dir / "input_preview.gif"
    capture = SO101InputCapture(
        env_id=env_id,
        camera_specs=specs,
        frames=records,
        policy_input_names=[name for name in POLICY_CAMERA_NAMES if records and name in records[0].camera_frames],
        debug_input_names=[name for name in DEBUG_CAMERA_NAMES if records and name in records[0].camera_frames],
        lerobot_policy_feature_keys=[
            f"observation.images.{name}"
            for name in POLICY_CAMERA_NAMES
            if records and name in records[0].camera_frames
        ],
        manifest_path=str(manifest_path),
        preview_path=str(preview_path),
        preview_gif_path=str(preview_gif_path),
    )
    manifest_path.write_text(json.dumps(asdict(capture), indent=2, sort_keys=True), encoding="utf-8")
    _write_input_preview(records, preview_path, preview_gif_path)
    return capture


def _write_input_preview(records: list[SO101InputFrame], preview_path: Path, gif_path: Path) -> None:
    from PIL import Image, ImageDraw

    preview_frames = []
    for record in records:
        if not record.camera_frames:
            continue
        camera_names = _ordered_camera_names(record.camera_frames)
        canvas = Image.new("RGB", (960, 520), (245, 245, 240))
        draw = ImageDraw.Draw(canvas)
        for index, camera_name in enumerate(camera_names[:3]):
            camera_image = Image.open(record.camera_frames[camera_name]).convert("RGB").resize((320, 240))
            x = index * 320
            canvas.paste(camera_image, (x, 0))
            draw.rectangle((x, 0, x + 319, 24), fill=(245, 245, 240))
            draw.text((x + 10, 6), camera_name, fill=(25, 25, 25))
        draw.text((20, 344), f"step {record.step:03d}", fill=(25, 25, 25))
        draw.text((20, 370), f"visual inputs: {', '.join(camera_names)}", fill=(65, 65, 65))
        draw.text((660, 344), "state", fill=(65, 65, 65))
        for index, value in enumerate(record.observation[:12]):
            x0 = 670 + (index % 6) * 36
            y0 = 390 + (index // 6) * 58
            draw.line((x0, y0 - 32, x0, y0 + 32), fill=(210, 210, 210))
            bar = max(-1.0, min(1.0, float(value))) * 30
            color = (35, 115, 210) if bar >= 0 else (225, 105, 55)
            y_min, y_max = sorted((y0, y0 - bar))
            draw.rectangle((x0 - 8, y_min, x0 + 8, y_max), fill=color)
            draw.text((x0 - 9, y0 + 36), str(index), fill=(90, 90, 90))
        draw.text((20, 420), "action", fill=(65, 65, 65))
        for index, value in enumerate(record.action[:6]):
            x0 = 30 + index * 42
            y0 = 465
            fill = int(16 * max(-1.0, min(1.0, abs(float(value)))))
            draw.rectangle((x0 - 17, y0 - 7, x0 + 17, y0 + 7), outline=(160, 160, 160))
            draw.rectangle((x0 - fill, y0 - 6, x0 + fill, y0 + 6), fill=(55, 150, 100))
        preview_frames.append(canvas)
    if not preview_frames:
        preview_frames.append(Image.new("RGB", (960, 520), (245, 245, 240)))
    preview_frames[-1].save(preview_path)
    preview_frames[0].save(gif_path, save_all=True, append_images=preview_frames[1:], duration=140, loop=0)


def _write_image(pixels: Any, path: Path) -> None:
    from PIL import Image

    Image.fromarray(pixels).save(path)


def _ordered_camera_names(camera_frames: dict[str, str]) -> list[str]:
    preferred = ["wrist_cam", "egocentric_cam", "top_down"]
    ordered = [name for name in preferred if name in camera_frames]
    ordered.extend(sorted(name for name in camera_frames if name not in set(preferred)))
    return ordered


def _make_camera(env: Any, camera_name: str) -> Any:
    if camera_name not in {"egocentric_cam", "top_down"}:
        return camera_name

    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if camera_name == "egocentric_cam":
        camera.lookat[:] = _egocentric_lookat(env)
        camera.distance = 0.45
        camera.azimuth = 45
        camera.elevation = -22
    else:
        camera.lookat[:] = _top_down_lookat(env)
        camera.distance = 0.65
        camera.azimuth = 90
        camera.elevation = -90
    return camera


def _egocentric_lookat(env: Any) -> list[float]:
    model = env.unwrapped.model
    data = env.unwrapped.data
    names = {model.body(index).name: index for index in range(model.nbody)}
    candidates = [names[name] for name in ("target", "cube", "gripper") if name in names]
    if candidates:
        points = [data.xpos[index] for index in candidates]
        return [float(sum(point[axis] for point in points) / len(points)) for axis in range(3)]
    return [0.18, -0.30, 0.12]


def _top_down_lookat(env: Any) -> list[float]:
    model = env.unwrapped.model
    data = env.unwrapped.data
    names = {model.body(index).name: index for index in range(model.nbody)}
    candidates = [names[name] for name in ("target", "cube", "gripper") if name in names]
    if candidates:
        points = [data.xpos[index] for index in candidates]
        return [float(sum(point[axis] for point in points) / len(points)) for axis in range(3)]
    return [0.12, 0.0, 0.02]


def _as_float_list(obs: object) -> list[float]:
    import numpy as np

    return np.asarray(obs, dtype=float).reshape(-1).tolist()


def _clear_dir(path: Path) -> None:
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
