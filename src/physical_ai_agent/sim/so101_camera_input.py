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
    manifest_path: str
    preview_path: str
    preview_gif_path: str


DEFAULT_SO101_ENV_IDS = (
    "MuJoCoReach-v1",
    "MuJoCoMove-v1",
    "MuJoCoPickLift-v1",
    "MuJoCoPickAndPlace-v1",
)


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
        camera_names = [env.unwrapped.model.camera(index).name for index in range(env.unwrapped.model.ncam)]
        renderers = {
            camera_name: mujoco.Renderer(env.unwrapped.model, height=height, width=width)
            for camera_name in camera_names
        }
        for step in range(steps):
            action = sample_action(env.action_space, step / max(1, steps - 1))
            obs, reward, terminated, truncated, _info = env.step(action)
            camera_frames: dict[str, str] = {}
            for camera_name, renderer in renderers.items():
                renderer.update_scene(env.unwrapped.data, camera=camera_name)
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
        camera_name = sorted(record.camera_frames)[0]
        camera_image = Image.open(record.camera_frames[camera_name]).convert("RGB").resize((480, 360))
        canvas = Image.new("RGB", (760, 420), (245, 245, 240))
        canvas.paste(camera_image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((500, 18), f"step {record.step:03d}", fill=(25, 25, 25))
        draw.text((500, 42), f"camera {camera_name}", fill=(65, 65, 65))
        draw.text((500, 70), "state", fill=(65, 65, 65))
        for index, value in enumerate(record.observation[:12]):
            x0 = 510 + (index % 6) * 36
            y0 = 115 + (index // 6) * 72
            draw.line((x0, y0 - 32, x0, y0 + 32), fill=(210, 210, 210))
            bar = max(-1.0, min(1.0, float(value))) * 30
            color = (35, 115, 210) if bar >= 0 else (225, 105, 55)
            y_min, y_max = sorted((y0, y0 - bar))
            draw.rectangle((x0 - 8, y_min, x0 + 8, y_max), fill=color)
            draw.text((x0 - 9, y0 + 36), str(index), fill=(90, 90, 90))
        draw.text((500, 290), "action", fill=(65, 65, 65))
        for index, value in enumerate(record.action[:6]):
            x0 = 510 + index * 36
            y0 = 335
            fill = int(16 * max(-1.0, min(1.0, abs(float(value)))))
            draw.rectangle((x0 - 17, y0 - 7, x0 + 17, y0 + 7), outline=(160, 160, 160))
            draw.rectangle((x0 - fill, y0 - 6, x0 + fill, y0 + 6), fill=(55, 150, 100))
        preview_frames.append(canvas)
    if not preview_frames:
        preview_frames.append(Image.new("RGB", (760, 420), (245, 245, 240)))
    preview_frames[-1].save(preview_path)
    preview_frames[0].save(gif_path, save_all=True, append_images=preview_frames[1:], duration=140, loop=0)


def _write_image(pixels: Any, path: Path) -> None:
    from PIL import Image

    Image.fromarray(pixels).save(path)


def _as_float_list(obs: object) -> list[float]:
    import numpy as np

    return np.asarray(obs, dtype=float).reshape(-1).tolist()


def _clear_dir(path: Path) -> None:
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
