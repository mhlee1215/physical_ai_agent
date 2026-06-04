from __future__ import annotations

import argparse
import io
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, sample_action


@dataclass(frozen=True)
class LiveViewerConfig:
    env_id: str = DEFAULT_SO101_ENV_ID
    fps: float = 30.0
    seed: int = 0
    max_steps: int | None = None
    show_inputs: bool = False
    input_width: int = 320
    input_height: int = 240
    input_port: int = 8765


def run_live_viewer(config: LiveViewerConfig) -> int:
    import gymnasium as gym
    import mujoco
    import mujoco.viewer
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    env = gym.make(config.env_id, render_mode=None)
    env.reset(seed=config.seed)
    sleep_s = 1.0 / max(1.0, config.fps)
    step = 0
    input_viewer = None

    try:
        if config.show_inputs:
            input_viewer = SO101LiveInputViewer(
                env=env,
                width=config.input_width,
                height=config.input_height,
                port=config.input_port,
            )
            input_viewer.start()
            print(f"SO101 camera inputs live at http://127.0.0.1:{config.input_port}")
        try:
            viewer_context = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)
        except RuntimeError as exc:
            if "mjpython" in str(exc):
                raise RuntimeError(
                    "MuJoCo live viewer on macOS requires `mjpython`. "
                    "Run through `scripts/view_so101_live.sh` for the repo-local preflight and setup hint."
                ) from exc
            raise
        with viewer_context as viewer:
            while viewer.is_running():
                if config.max_steps is not None and step >= config.max_steps:
                    break
                action = sample_action(env.action_space, (step % 120) / 119.0)
                obs, reward, terminated, truncated, _info = env.step(action)
                viewer.sync()
                if input_viewer is not None:
                    if not input_viewer.show(
                        step=step,
                        observation=[float(value) for value in obs],
                        action=[float(value) for value in action],
                        reward=float(reward),
                    ):
                        break
                step += 1
                if terminated or truncated:
                    env.reset()
                time.sleep(sleep_s)
    finally:
        if input_viewer is not None:
            input_viewer.close()
        env.close()
    return step


class SO101LiveInputViewer:
    camera_names = ("wrist_cam", "egocentric_cam", "top_down")

    def __init__(self, env: Any, width: int = 320, height: int = 240, port: int = 8765) -> None:
        import mujoco

        self._env = env
        self._width = width
        self._height = height
        self._port = port
        self._latest_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._renderers = {
            name: mujoco.Renderer(env.unwrapped.model, height=height, width=width)
            for name in self.camera_names
        }
        self._server = ThreadingHTTPServer(("127.0.0.1", port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def show(self, step: int, observation: list[float], action: list[float], reward: float) -> bool:
        from physical_ai_agent.sim.so101_camera_input import _make_camera

        frames = []
        for camera_name, renderer in self._renderers.items():
            renderer.update_scene(self._env.unwrapped.data, camera=_make_camera(self._env, camera_name))
            frames.append((camera_name, renderer.render()))
        image = self._compose_canvas(frames, step, observation, action, reward)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        with self._lock:
            self._latest_jpeg = buffer.getvalue()
        return True

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._server.shutdown()
        self._server.server_close()

    def _compose_canvas(
        self,
        frames: list[tuple[str, Any]],
        step: int,
        observation: list[float],
        action: list[float],
        reward: float,
    ) -> Any:
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", (self._width * 3, self._height + 120), (245, 245, 240))
        draw = ImageDraw.Draw(canvas)
        for index, (name, pixels) in enumerate(frames):
            image = Image.fromarray(pixels).convert("RGB")
            x = index * self._width
            canvas.paste(image, (x, 0))
            draw.rectangle((x, 0, x + self._width - 1, 24), fill=(245, 245, 240))
            draw.text((x + 10, 7), name, fill=(25, 25, 25))
        y = self._height + 26
        draw.text((16, y), f"step {step:04d}  reward {reward:.4f}", fill=(45, 45, 45))
        draw.text(
            (16, y + 28),
            "policy: wrist_cam + egocentric_cam    debug: top_down",
            fill=(45, 45, 45),
        )
        self._draw_bars(draw, "state", observation[:12], x=16, y=y + 76)
        self._draw_bars(draw, "action", action[:6], x=520, y=y + 76)
        return canvas

    def _draw_bars(self, draw: Any, label: str, values: list[float], x: int, y: int) -> None:
        draw.text((x, y - 32), label, fill=(45, 45, 45))
        for index, value in enumerate(values):
            x0 = x + 54 + index * 30
            clipped = max(-1.0, min(1.0, float(value)))
            length = int(abs(clipped) * 18)
            color = (35, 115, 210) if clipped >= 0 else (225, 105, 55)
            draw.rectangle((x0 - 19, y - 9, x0 + 19, y + 9), outline=(160, 160, 160))
            draw.rectangle((x0 - length, y - 8, x0 + length, y + 8), fill=color)

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        viewer = self

        class InputStreamHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path in {"/", "/index.html"}:
                    self._write_html()
                    return
                if self.path == "/stream.mjpg":
                    self._write_stream()
                    return
                self.send_error(404)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _write_html(self) -> None:
                body = b"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>SO101 Camera Inputs</title>
    <style>
      body { margin: 0; background: #f5f5f0; font-family: system-ui, sans-serif; }
      img { display: block; width: 100vw; height: auto; image-rendering: auto; }
    </style>
  </head>
  <body><img src="/stream.mjpg" alt="SO101 camera input stream"></body>
</html>"""
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_stream(self) -> None:
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with viewer._lock:
                            jpeg = viewer._latest_jpeg
                        if jpeg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return

        return InputStreamHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a live MuJoCo viewer for SO101-Nexus.")
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--show-inputs",
        action="store_true",
        help="Open a live camera-input window for wrist_cam, egocentric_cam, and top_down.",
    )
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--input-height", type=int, default=240)
    parser.add_argument("--input-port", type=int, default=8765)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional finite step count for smoke checks or scripted demos.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    steps = run_live_viewer(
        LiveViewerConfig(
            env_id=args.env_id,
            fps=args.fps,
            seed=args.seed,
            max_steps=args.max_steps,
            show_inputs=args.show_inputs,
            input_width=args.input_width,
            input_height=args.input_height,
            input_port=args.input_port,
        )
    )
    print(f"SO101 live viewer closed after {steps} steps")


if __name__ == "__main__":
    main()
