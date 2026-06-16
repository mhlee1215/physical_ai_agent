from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from physical_ai_agent.policies.smolvla_adapter import DEFAULT_SMOLVLA_MODEL_ID
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, sample_action


@dataclass(frozen=True)
class LiveViewerConfig:
    env_id: str = DEFAULT_SO101_ENV_ID
    fps: float = 30.0
    seed: int = 0
    max_steps: int | None = None
    policy: str = "sample"
    model_id: str = DEFAULT_SMOLVLA_MODEL_ID
    allow_download: bool = False
    smolvla_action_steps: int = 15
    smolvla_worker_python: str = ""
    visual_policy_checkpoint: str = "_workspace/so101_visual_rl/train/so101_visual_rl_policy.pt"
    visual_policy_camera: str = "wrist_cam"
    visual_reach_checkpoint: str = "_workspace/so101_visual_rl/reach_delta/so101_visual_reach_delta.pt"
    visual_reach_camera: str = "top_down"
    browser_only: bool = False
    show_inputs: bool = False
    input_width: int = 320
    input_height: int = 240
    input_port: int = 8765


def run_live_viewer(config: LiveViewerConfig) -> int:
    import gymnasium as gym
    import mujoco
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    env = gym.make(config.env_id, render_mode=None)
    obs, _info = env.reset(seed=config.seed)
    sleep_s = 1.0 / max(1.0, config.fps)
    step = 0
    input_viewer = None
    smolvla_worker = None
    visual_policy = None
    visual_reach_policy = None
    action_queue: list[list[float]] = []
    image_feature_mapping: dict[str, str] = {}
    chunk_status = ""

    try:
        if config.policy == "smolvla":
            smolvla_worker = SmolVLAWorkerClient(
                model_id=config.model_id,
                allow_download=config.allow_download,
                action_steps=config.smolvla_action_steps,
                python_executable=config.smolvla_worker_python,
            )
            print("Starting isolated SmolVLA worker process...", flush=True)
            smolvla_worker.start()
            print(
                "SmolVLA worker started; "
                f"executing {config.smolvla_action_steps} actions per predicted chunk.",
                flush=True,
            )
        if config.policy == "visual-rl":
            visual_policy = VisualRLPolicyClient(
                checkpoint_path=Path(config.visual_policy_checkpoint),
                camera_name=config.visual_policy_camera,
                width=config.input_width,
                height=config.input_height,
            )
            print(
                f"Loaded visual RL policy from {config.visual_policy_checkpoint} "
                f"using camera {config.visual_policy_camera}.",
                flush=True,
            )
        if config.policy == "visual-reach":
            visual_reach_policy = VisualReachPolicyClient(
                checkpoint_path=Path(config.visual_reach_checkpoint),
                camera_name=config.visual_reach_camera,
                width=config.input_width,
                height=config.input_height,
            )
            print(
                f"Loaded visual reach policy from {config.visual_reach_checkpoint} "
                f"using camera {config.visual_reach_camera}.",
                flush=True,
            )
        if config.show_inputs or config.policy == "smolvla" or config.browser_only:
            input_viewer = SO101LiveInputViewer(
                env=env,
                width=config.input_width,
                height=config.input_height,
                port=config.input_port,
                show_scene=config.browser_only,
            )
            input_viewer.start()
            print(f"SO101 live stream available at http://127.0.0.1:{config.input_port}")
        if config.browser_only:
            while True:
                if config.max_steps is not None and step >= config.max_steps:
                    break
                (
                    obs,
                    action,
                    reward,
                    terminated,
                    truncated,
                    inference_latency_s,
                    chunk_status,
                ) = _run_policy_step(
                    env=env,
                    obs=obs,
                    step=step,
                    config=config,
                    input_viewer=input_viewer,
                    smolvla_worker=smolvla_worker,
                    visual_policy=visual_policy,
                    visual_reach_policy=visual_reach_policy,
                    action_queue=action_queue,
                    image_feature_mapping=image_feature_mapping,
                    chunk_status=chunk_status,
                )
                if input_viewer is not None:
                    if not input_viewer.show(
                        step=step,
                        observation=[float(value) for value in obs],
                        action=[float(value) for value in action],
                        reward=float(reward),
                        policy_name=config.policy,
                        inference_latency_s=inference_latency_s,
                        image_feature_mapping=image_feature_mapping,
                        chunk_status=chunk_status
                        if config.policy in {"smolvla", "visual-rl", "visual-reach"}
                        else "",
                    ):
                        break
                step += 1
                if terminated or truncated:
                    obs, _info = env.reset()
                time.sleep(max(0.0, sleep_s - inference_latency_s))
        else:
            import mujoco.viewer

            try:
                viewer_context = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)
            except RuntimeError as exc:
                if "mjpython" in str(exc):
                    raise RuntimeError(
                        "MuJoCo live viewer on macOS requires `mjpython`. "
                        "Use `--browser-only` to run the live SmolVLA path without the native Cocoa viewer, "
                        "or run through `scripts/view_so101_live.sh` with a Homebrew/python.org viewer venv."
                    ) from exc
                raise
            with viewer_context as viewer:
                while viewer.is_running():
                    if config.max_steps is not None and step >= config.max_steps:
                        break
                    (
                        obs,
                        action,
                        reward,
                        terminated,
                        truncated,
                        inference_latency_s,
                        chunk_status,
                    ) = _run_policy_step(
                        env=env,
                        obs=obs,
                        step=step,
                        config=config,
                        input_viewer=input_viewer,
                        smolvla_worker=smolvla_worker,
                        visual_policy=visual_policy,
                        visual_reach_policy=visual_reach_policy,
                        action_queue=action_queue,
                        image_feature_mapping=image_feature_mapping,
                        chunk_status=chunk_status,
                    )
                    viewer.sync()
                    if input_viewer is not None:
                        if not input_viewer.show(
                            step=step,
                            observation=[float(value) for value in obs],
                            action=[float(value) for value in action],
                            reward=float(reward),
                            policy_name=config.policy,
                            inference_latency_s=inference_latency_s,
                            image_feature_mapping=image_feature_mapping,
                            chunk_status=chunk_status
                            if config.policy in {"smolvla", "visual-rl", "visual-reach"}
                            else "",
                        ):
                            break
                    step += 1
                    if terminated or truncated:
                        obs, _info = env.reset()
                    time.sleep(max(0.0, sleep_s - inference_latency_s))
    finally:
        if input_viewer is not None:
            input_viewer.close()
        if visual_policy is not None:
            visual_policy.close()
        if visual_reach_policy is not None:
            visual_reach_policy.close()
        if smolvla_worker is not None:
            smolvla_worker.close()
        env.close()
    return step


def _run_policy_step(
    env: Any,
    obs: Any,
    step: int,
    config: LiveViewerConfig,
    input_viewer: "SO101LiveInputViewer | None",
    smolvla_worker: "SmolVLAWorkerClient | None",
    visual_policy: "VisualRLPolicyClient | None",
    visual_reach_policy: "VisualReachPolicyClient | None",
    action_queue: list[list[float]],
    image_feature_mapping: dict[str, str],
    chunk_status: str,
) -> tuple[Any, list[float], float, bool, bool, float, str]:
    action_started_at = time.perf_counter()
    camera_pixels = input_viewer.render_policy_camera_pixels() if input_viewer is not None else {}
    if config.policy == "smolvla":
        if smolvla_worker is None:
            raise RuntimeError("SmolVLA worker was not started")
        if not action_queue:
            chunk = smolvla_worker.predict_chunk(
                observation=[float(value) for value in obs],
                camera_pixels=camera_pixels,
                action_dim=int(env.action_space.shape[0]),
            )
            action_queue[:] = chunk.actions[:]
            image_feature_mapping.clear()
            image_feature_mapping.update(chunk.image_feature_mapping)
            chunk_status = (
                f"chunk steps 0/{chunk.executed_action_steps} "
                f"from predicted {chunk.predicted_chunk_size}"
            )
        action = action_queue.pop(0)
        executed_steps = config.smolvla_action_steps
        predicted_steps = smolvla_worker.last_predicted_chunk_size or executed_steps
        used = executed_steps - len(action_queue)
        chunk_status = f"chunk steps {used}/{executed_steps} from predicted {predicted_steps}"
    elif config.policy == "reach":
        action = _reach_controller_action(env)
        chunk_status = _reach_status(env)
    elif config.policy == "visual-rl":
        if visual_policy is None:
            raise RuntimeError("Visual RL policy was not loaded")
        action = visual_policy.predict(env=env, state_obs=obs)
        chunk_status = f"visual RL checkpoint {Path(config.visual_policy_checkpoint).name}"
    elif config.policy == "visual-reach":
        if visual_reach_policy is None:
            raise RuntimeError("Visual reach policy was not loaded")
        action, predicted_error = visual_reach_policy.predict_action(env=env, state_obs=obs)
        chunk_status = (
            f"visual reach delta {float(sum(value * value for value in predicted_error) ** 0.5):.4f}m "
            f"checkpoint {Path(config.visual_reach_checkpoint).name}"
        )
    else:
        action = [float(value) for value in sample_action(env.action_space, (step % 120) / 119.0)]
    inference_latency_s = time.perf_counter() - action_started_at
    next_obs, reward, terminated, truncated, _info = env.step(action)
    return next_obs, action, float(reward), bool(terminated), bool(truncated), inference_latency_s, chunk_status


def _reach_controller_action(env: Any) -> list[float]:
    import numpy as np

    model = env.unwrapped.model
    data = env.unwrapped.data
    target_site = model.site("reach_target").id
    gripper_site = model.site("gripperframe").id
    error = data.site_xpos[target_site] - data.site_xpos[gripper_site]
    return _cartesian_error_controller_action(env, error)


def _cartesian_error_controller_action(env: Any, error: Any) -> list[float]:
    import mujoco
    import numpy as np

    model = env.unwrapped.model
    data = env.unwrapped.data
    gripper_site = model.site("gripperframe").id
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, gripper_site)
    bounded_error = np.clip(np.asarray(error, dtype=float), -0.25, 0.25)
    joint_delta = np.linalg.pinv(jacp, rcond=1e-3) @ (2.0 * bounded_error)
    action = np.asarray(data.qpos[: model.nu], dtype=float) + 0.05 * joint_delta[: model.nu]
    low = np.asarray(env.action_space.low, dtype=float)
    high = np.asarray(env.action_space.high, dtype=float)
    return np.clip(action, low, high).astype(float).tolist()


def _reach_status(env: Any) -> str:
    import numpy as np

    model = env.unwrapped.model
    data = env.unwrapped.data
    target = data.site_xpos[model.site("reach_target").id]
    gripper = data.site_xpos[model.site("gripperframe").id]
    return f"reach error {float(np.linalg.norm(target - gripper)):.4f}m"


class VisualRLPolicyClient:
    def __init__(
        self,
        checkpoint_path: Path,
        camera_name: str,
        width: int,
        height: int,
    ) -> None:
        import mujoco

        from physical_ai_agent.policies.so101_visual_actor_critic import (
            load_so101_visual_actor_critic_checkpoint,
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Visual RL checkpoint does not exist: {checkpoint_path}")
        self.model, self.metadata = load_so101_visual_actor_critic_checkpoint(checkpoint_path)
        model_config = self.metadata.get("config", {}) if isinstance(self.metadata, dict) else {}
        self.camera_name = camera_name or model_config.get("camera_name", "wrist_cam")
        self.width = int(model_config.get("width", width))
        self.height = int(model_config.get("height", height))
        self.include_state = bool(model_config.get("include_state", True))
        self.channel_first = bool(model_config.get("channel_first", True))
        self._renderer: Any | None = None
        self._mujoco = mujoco

    def predict(self, *, env: Any, state_obs: Any) -> list[float]:
        import numpy as np
        import torch

        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                env.unwrapped.model,
                height=self.height,
                width=self.width,
            )
        observation = {"image": self._render_pixels(env)}
        if self.include_state:
            observation["state"] = np.asarray(state_obs, dtype=np.float32).reshape(-1)
        with torch.no_grad():
            action_packet = self.model.act(observation, deterministic=True)
        return [float(value) for value in action_packet["action"].cpu().numpy()[0]]

    def _render_pixels(self, env: Any) -> Any:
        from physical_ai_agent.sim.so101_camera_input import _make_camera

        self._renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, self.camera_name))
        pixels = self._renderer.render()
        if self.channel_first:
            return pixels.transpose(2, 0, 1)
        return pixels

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()


class VisualReachPolicyClient:
    def __init__(
        self,
        checkpoint_path: Path,
        camera_name: str,
        width: int,
        height: int,
    ) -> None:
        import mujoco

        from physical_ai_agent.policies.so101_visual_reach_delta import (
            load_so101_visual_reach_delta_checkpoint,
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Visual reach checkpoint does not exist: {checkpoint_path}")
        self.model, self.metadata = load_so101_visual_reach_delta_checkpoint(checkpoint_path)
        model_config = self.metadata.get("config", {}) if isinstance(self.metadata, dict) else {}
        self.camera_name = camera_name or model_config.get("camera_name", "top_down")
        self.width = int(model_config.get("width", width))
        self.height = int(model_config.get("height", height))
        self.include_state = bool(model_config.get("include_state", True))
        self.channel_first = bool(model_config.get("channel_first", True))
        self._renderer: Any | None = None
        self._mujoco = mujoco

    def predict_action(self, *, env: Any, state_obs: Any) -> tuple[list[float], list[float]]:
        import numpy as np
        import torch

        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                env.unwrapped.model,
                height=self.height,
                width=self.width,
            )
        observation = {"image": self._render_pixels(env)}
        if self.include_state:
            observation["state"] = np.asarray(state_obs, dtype=np.float32).reshape(-1)
        with torch.no_grad():
            predicted_error = self.model(observation).detach().cpu().numpy()[0].astype(float)
        action = _cartesian_error_controller_action(env, predicted_error)
        return action, predicted_error.tolist()

    def _render_pixels(self, env: Any) -> Any:
        from physical_ai_agent.sim.so101_camera_input import _make_camera

        self._renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, self.camera_name))
        pixels = self._renderer.render()
        if self.channel_first:
            return pixels.transpose(2, 0, 1)
        return pixels

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()


class SmolVLAActionChunk:
    def __init__(
        self,
        actions: list[list[float]],
        predicted_chunk_size: int,
        executed_action_steps: int,
        image_feature_mapping: dict[str, str],
        latency_s: float,
    ) -> None:
        self.actions = actions
        self.predicted_chunk_size = predicted_chunk_size
        self.executed_action_steps = executed_action_steps
        self.image_feature_mapping = image_feature_mapping
        self.latency_s = latency_s


class SmolVLAWorkerClient:
    def __init__(
        self,
        model_id: str,
        allow_download: bool,
        action_steps: int,
        python_executable: str = "",
    ) -> None:
        self.model_id = model_id
        self.allow_download = allow_download
        self.action_steps = action_steps
        self.python_executable = python_executable or _default_worker_python()
        self.process: subprocess.Popen[str] | None = None
        self.last_predicted_chunk_size = 0

    def start(self) -> None:
        cmd = [
            self.python_executable,
            "-B",
            "-m",
            "physical_ai_agent.policies.smolvla_worker",
            "--model-id",
            self.model_id,
            "--action-steps",
            str(self.action_steps),
        ]
        if self.allow_download:
            cmd.append("--allow-download")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            close_fds=False,
        )

    def predict_chunk(
        self,
        observation: list[float],
        camera_pixels: dict[str, Any],
        action_dim: int,
    ) -> SmolVLAActionChunk:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("SmolVLA worker is not running")
        request = {
            "type": "predict",
            "observation": observation,
            "action_dim": action_dim,
            "camera_pixels": _encode_camera_pixels(camera_pixels),
        }
        self.process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("SmolVLA worker exited before returning an action chunk")
        response = json.loads(line)
        self.last_predicted_chunk_size = int(response["predicted_chunk_size"])
        return SmolVLAActionChunk(
            actions=[[float(value) for value in action] for action in response["actions"]],
            predicted_chunk_size=self.last_predicted_chunk_size,
            executed_action_steps=int(response["executed_action_steps"]),
            image_feature_mapping=dict(response["image_feature_mapping"]),
            latency_s=float(response["latency_s"]),
        )

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                self.process.stdin.flush()
            self.process.wait(timeout=2)
        except Exception:  # noqa: BLE001
            self.process.terminate()
        self.process = None


def _default_worker_python() -> str:
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _encode_camera_pixels(camera_pixels: dict[str, Any]) -> dict[str, str]:
    from PIL import Image

    encoded = {}
    for camera_name, pixels in camera_pixels.items():
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        encoded[camera_name] = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded


class SO101LiveInputViewer:
    camera_names = ("wrist_cam", "egocentric_cam", "top_down")

    def __init__(
        self,
        env: Any,
        width: int = 320,
        height: int = 240,
        port: int = 8765,
        show_scene: bool = False,
    ) -> None:
        import mujoco

        self._env = env
        self._width = width
        self._height = height
        self._port = port
        self._show_scene = show_scene
        self._latest_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._renderers = {
            name: mujoco.Renderer(env.unwrapped.model, height=height, width=width)
            for name in self.camera_names
        }
        self._scene_renderer = (
            mujoco.Renderer(env.unwrapped.model, height=height, width=width)
            if show_scene
            else None
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def render_camera_pixels(self) -> dict[str, Any]:
        frames = self.render_policy_camera_pixels()
        if self._scene_renderer is not None:
            self._scene_renderer.update_scene(self._env.unwrapped.data)
            frames = {"scene_3d": self._scene_renderer.render(), **frames}
        return frames

    def render_policy_camera_pixels(self) -> dict[str, Any]:
        from physical_ai_agent.sim.so101_camera_input import _make_camera

        frames = {}
        for camera_name, renderer in self._renderers.items():
            renderer.update_scene(self._env.unwrapped.data, camera=_make_camera(self._env, camera_name))
            frames[camera_name] = renderer.render()
        return frames

    def show(
        self,
        step: int,
        observation: list[float],
        action: list[float],
        reward: float,
        camera_pixels: dict[str, Any] | None = None,
        policy_name: str = "sample",
        inference_latency_s: float = 0.0,
        image_feature_mapping: dict[str, str] | None = None,
        chunk_status: str = "",
    ) -> bool:
        frames = camera_pixels or self.render_camera_pixels()
        image = self._compose_canvas(
            frames=frames,
            step=step,
            observation=observation,
            action=action,
            reward=reward,
            policy_name=policy_name,
            inference_latency_s=inference_latency_s,
            image_feature_mapping=image_feature_mapping or {},
            chunk_status=chunk_status,
        )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        with self._lock:
            self._latest_jpeg = buffer.getvalue()
        return True

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        if self._scene_renderer is not None:
            self._scene_renderer.close()
        self._server.shutdown()
        self._server.server_close()

    def _compose_canvas(
        self,
        frames: dict[str, Any],
        step: int,
        observation: list[float],
        action: list[float],
        reward: float,
        policy_name: str = "sample",
        inference_latency_s: float = 0.0,
        image_feature_mapping: dict[str, str] | None = None,
        chunk_status: str = "",
    ) -> Any:
        from PIL import Image, ImageDraw

        panel_names = ["scene_3d"] if self._show_scene else []
        panel_names.extend(self.camera_names)
        canvas = Image.new("RGB", (self._width * len(panel_names), self._height + 120), (245, 245, 240))
        draw = ImageDraw.Draw(canvas)
        for index, name in enumerate(panel_names):
            pixels = frames[name]
            image = Image.fromarray(pixels).convert("RGB")
            x = index * self._width
            canvas.paste(image, (x, 0))
            draw.rectangle((x, 0, x + self._width - 1, 24), fill=(245, 245, 240))
            draw.text((x + 10, 7), name, fill=(25, 25, 25))
        y = self._height + 26
        draw.text(
            (16, y),
            (
                f"step {step:04d}  reward {reward:.4f}  "
                f"policy {policy_name}  inference {inference_latency_s:.3f}s"
            ),
            fill=(45, 45, 45),
        )
        mapping_text = _mapping_text(image_feature_mapping or {})
        draw.text(
            (16, y + 28),
            _status_text(mapping_text, chunk_status),
            fill=(45, 45, 45),
        )
        self._draw_bars(draw, "state", observation[:12], x=16, y=y + 76)
        action_x = min(520, self._width * len(panel_names) - 260)
        self._draw_bars(draw, "action", action[:6], x=action_x, y=y + 76)
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


def _mapping_text(mapping: dict[str, str]) -> str:
    if not mapping:
        return "policy: wrist_cam + egocentric_cam    debug: top_down"
    values = ", ".join(f"{key.rsplit('.', 1)[-1]}<-{value}" for key, value in mapping.items())
    return f"SmolVLA images: {values}"


def _status_text(mapping_text: str, chunk_status: str) -> str:
    if mapping_text and chunk_status:
        return f"{mapping_text} | {chunk_status}"
    if mapping_text:
        return mapping_text
    if chunk_status:
        return chunk_status
    return "policy: wrist_cam + egocentric_cam    debug: top_down"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a live MuJoCo viewer for SO101-Nexus.")
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=["sample", "reach", "smolvla", "visual-rl", "visual-reach"],
        default="sample",
        help="Action source for the live simulation.",
    )
    parser.add_argument("--model-id", default=DEFAULT_SMOLVLA_MODEL_ID)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--smolvla-action-steps",
        type=int,
        default=15,
        help="Number of actions to execute from each SmolVLA predicted chunk.",
    )
    parser.add_argument(
        "--smolvla-worker-python",
        default="",
        help="Python executable for the isolated SmolVLA inference worker.",
    )
    parser.add_argument(
        "--visual-policy-checkpoint",
        default="_workspace/so101_visual_rl/train/so101_visual_rl_policy.pt",
        help="Torch checkpoint produced by scripts/train_so101_visual_rl.py.",
    )
    parser.add_argument(
        "--visual-policy-camera",
        default="wrist_cam",
        help="Camera name to render for the visual RL policy.",
    )
    parser.add_argument(
        "--visual-reach-checkpoint",
        default="_workspace/so101_visual_rl/reach_delta/so101_visual_reach_delta.pt",
        help="Torch checkpoint produced by scripts/train_so101_visual_reach_delta.py.",
    )
    parser.add_argument(
        "--visual-reach-camera",
        default="top_down",
        help="Camera name to render for the visual reach delta estimator.",
    )
    parser.add_argument(
        "--browser-only",
        action="store_true",
        help="Stream the live 3D scene, camera inputs, and action telemetry in the browser without mjpython.",
    )
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
            policy=args.policy,
            model_id=args.model_id,
            allow_download=args.allow_download,
            smolvla_action_steps=args.smolvla_action_steps,
            smolvla_worker_python=args.smolvla_worker_python,
            visual_policy_checkpoint=args.visual_policy_checkpoint,
            visual_policy_camera=args.visual_policy_camera,
            visual_reach_checkpoint=args.visual_reach_checkpoint,
            visual_reach_camera=args.visual_reach_camera,
            browser_only=args.browser_only,
            show_inputs=args.show_inputs,
            input_width=args.input_width,
            input_height=args.input_height,
            input_port=args.input_port,
        )
    )
    print(f"SO101 live viewer closed after {steps} steps")


if __name__ == "__main__":
    main()
