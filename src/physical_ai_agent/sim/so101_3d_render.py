from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, sample_action


@dataclass(frozen=True)
class SO101RenderAttempt:
    backend: str
    ok: bool
    error: str | None = None


@dataclass(frozen=True)
class SO101RenderResult:
    env_id: str
    status: str
    frame_path: str
    gif_path: str
    report_path: str
    blocker_path: str
    attempts: list[SO101RenderAttempt]
    width: int
    height: int
    frames: int


def render_so101_3d_rollout(
    output_dir: Path,
    env_id: str = DEFAULT_SO101_ENV_ID,
    steps: int = 24,
    seed: int = 0,
    width: int = 640,
    height: int = 360,
) -> SO101RenderResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "so101_3d_render.png"
    gif_path = output_dir / "so101_3d_render.gif"
    report_path = output_dir / "so101_3d_render_report.json"
    blocker_path = output_dir / "so101_3d_render_blocker.md"
    _unlink_if_exists(frame_path, gif_path, blocker_path)

    attempts: list[SO101RenderAttempt] = []
    frames: list[Any] = []

    try:
        frames = _render_with_gym_rgb(env_id, steps, seed)
        attempts.append(SO101RenderAttempt(backend="gymnasium_rgb_array", ok=True))
    except Exception as exc:  # noqa: BLE001
        attempts.append(SO101RenderAttempt("gymnasium_rgb_array", False, _short_error(exc)))

    if not frames:
        try:
            frames = _render_with_mujoco_renderer(env_id, steps, seed, width, height)
            attempts.append(SO101RenderAttempt(backend="mujoco.Renderer", ok=True))
        except Exception as exc:  # noqa: BLE001
            attempts.append(SO101RenderAttempt("mujoco.Renderer", False, _short_error(exc)))

    status = "passed" if frames else "blocked"
    if frames:
        from PIL import Image

        images = [Image.fromarray(frame) for frame in frames]
        images[-1].save(frame_path)
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=90, loop=0)
        blocker_path.write_text("", encoding="utf-8")
    else:
        _write_render_blocker(blocker_path, attempts)

    result = SO101RenderResult(
        env_id=env_id,
        status=status,
        frame_path=str(frame_path),
        gif_path=str(gif_path),
        report_path=str(report_path),
        blocker_path=str(blocker_path),
        attempts=attempts,
        width=width,
        height=height,
        frames=len(frames),
    )
    report_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def _render_with_gym_rgb(env_id: str, steps: int, seed: int) -> list[Any]:
    import gymnasium as gym
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    try:
        env.reset(seed=seed)
        first = env.render()
        if first is not None:
            frames.append(first)
        for step in range(steps):
            env.step(sample_action(env.action_space, step / max(1, steps - 1)))
            frame = env.render()
            if frame is not None:
                frames.append(frame)
    finally:
        env.close()
    if not frames:
        raise RuntimeError("Gymnasium render returned no RGB frames")
    return frames


def _render_with_mujoco_renderer(
    env_id: str,
    steps: int,
    seed: int,
    width: int,
    height: int,
) -> list[Any]:
    import gymnasium as gym
    import mujoco
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    env = gym.make(env_id, render_mode=None)
    renderer = None
    frames = []
    try:
        env.reset(seed=seed)
        renderer = mujoco.Renderer(env.unwrapped.model, height=height, width=width)
        for step in range(steps):
            env.step(sample_action(env.action_space, step / max(1, steps - 1)))
            renderer.update_scene(env.unwrapped.data)
            frames.append(renderer.render())
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
    if not frames:
        raise RuntimeError("MuJoCo renderer returned no RGB frames")
    return frames


def _write_render_blocker(path: Path, attempts: list[SO101RenderAttempt]) -> None:
    lines = [
        "# CP14 SO101 3D Render Blocker",
        "",
        "SO101-Nexus physics can reset and step, but this process could not create RGB frames.",
        "On headless macOS this commonly fails at the CoreGraphics/OpenGL context layer.",
        "",
        "## Attempts",
        "",
    ]
    for attempt in attempts:
        status = "ok" if attempt.ok else "failed"
        lines.append(f"- `{attempt.backend}`: {status}")
        if attempt.error:
            lines.append(f"  - `{attempt.error}`")
    lines.extend(
        [
            "",
            "## Local GUI Retry",
            "",
            "Run from a normal macOS terminal session, not a headless agent session:",
            "",
            "```bash",
            "sh scripts/view_so101_live.sh --browser-only --show-inputs --fps 2 --max-steps 1",
            "```",
            "",
            f"Observed `MUJOCO_GL={os.environ.get('MUJOCO_GL', '')}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _short_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text.replace("\n", " ")[:500]


def _unlink_if_exists(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
