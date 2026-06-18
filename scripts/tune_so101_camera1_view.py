#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from physical_ai_agent.sim.so101_camera_input import EGOCENTRIC_CAMERA1_POSE, _make_camera, postprocess_camera_frame


@dataclass(frozen=True)
class Camera1Preset:
    lookat_x: float = float(EGOCENTRIC_CAMERA1_POSE["lookat"][0])
    lookat_y: float = float(EGOCENTRIC_CAMERA1_POSE["lookat"][1])
    lookat_z: float = float(EGOCENTRIC_CAMERA1_POSE["lookat"][2])
    distance: float = float(EGOCENTRIC_CAMERA1_POSE["distance"])
    azimuth: float = float(EGOCENTRIC_CAMERA1_POSE["azimuth"])
    elevation: float = float(EGOCENTRIC_CAMERA1_POSE["elevation"])
    rotation_degrees: int = int(EGOCENTRIC_CAMERA1_POSE["rotation_degrees"])
    width: int = 512
    height: int = 512


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive SO101 camera1 egocentric view tuner.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pose", choices=["home", "pregrasp", "elevated_top"], default="pregrasp")
    parser.add_argument("--output", type=Path, default=Path("_workspace/so101_camera_tuner/camera1_preset.json"))
    args = parser.parse_args()

    app = Camera1TunerApp(
        width=args.width,
        height=args.height,
        seed=args.seed,
        pose=args.pose,
        output_path=args.output,
    )
    server = ReusableHTTPServer((args.host, args.port), app.make_handler())
    print(f"[camera1-tuner] serving http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    finally:
        app.close()
        server.server_close()


class Camera1TunerApp:
    def __init__(self, *, width: int, height: int, seed: int, pose: str, output_path: Path) -> None:
        import mujoco

        from train_so101_wrist_ego_visual_servo import make_high_contrast_picklift_env

        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed)
        self.pose = pose
        self.output_path = output_path
        self.lock = threading.Lock()
        self.env = make_high_contrast_picklift_env()
        self.env.reset(seed=self.seed)
        self._set_pose(self.pose)
        self.renderers = {
            "camera1": mujoco.Renderer(self.env.unwrapped.model, height=self.height, width=self.width),
            "wrist": mujoco.Renderer(self.env.unwrapped.model, height=self.height, width=self.width),
            "top": mujoco.Renderer(self.env.unwrapped.model, height=self.height, width=self.width),
            "scene": mujoco.Renderer(self.env.unwrapped.model, height=self.height, width=self.width),
        }

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()
        self.env.close()

    def make_handler(self) -> type[BaseHTTPRequestHandler]:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path in {"/", "/index.html"}:
                    self._write_html()
                    return
                if parsed.path == "/api/render":
                    self._write_json(app.render_payload(parse_qs(parsed.query)))
                    return
                if parsed.path == "/api/preset":
                    self._write_json(app.current_preset_payload())
                    return
                self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != "/api/save":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body) if body else {}
                self._write_json(app.save_preset(payload))

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _write_json(self, payload: dict[str, Any]) -> None:
                data = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _write_html(self) -> None:
                data = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    def current_preset_payload(self) -> dict[str, Any]:
        preset = self._load_preset()
        return {"preset": asdict(preset), "output_path": str(self.output_path), "pose": self.pose, "seed": self.seed}

    def render_payload(self, query: dict[str, list[str]]) -> dict[str, Any]:
        preset = _preset_from_query(query, default=self._load_preset(), width=self.width, height=self.height)
        pose = _query_string(query, "pose", self.pose)
        seed = _query_int(query, "seed", self.seed)
        with self.lock:
            if pose != self.pose or seed != self.seed:
                self.seed = seed
                self.pose = pose
                self.env.reset(seed=self.seed)
                self._set_pose(self.pose)
            frames = self._render_frames(preset)
        return {
            "preset": asdict(preset),
            "pose": self.pose,
            "seed": self.seed,
            "frames": {name: _png_data_url(pixels) for name, pixels in frames.items()},
            "mujoco_camera": {
                "type": "free",
                "lookat": [preset.lookat_x, preset.lookat_y, preset.lookat_z],
                "distance": preset.distance,
                "azimuth": preset.azimuth,
                "elevation": preset.elevation,
            },
        }

    def save_preset(self, payload: dict[str, Any]) -> dict[str, Any]:
        preset = _preset_from_mapping(payload, default=self._load_preset(), width=self.width, height=self.height)
        data = {
            "camera_name": "egocentric_cam",
            "lerobot_feature": "observation.images.camera1",
            "preset": asdict(preset),
            "mujoco_camera": {
                "type": "free",
                "lookat": [preset.lookat_x, preset.lookat_y, preset.lookat_z],
                "distance": preset.distance,
                "azimuth": preset.azimuth,
                "elevation": preset.elevation,
            },
            "postprocess": {"rotation_degrees": preset.rotation_degrees},
            "seed": int(payload.get("seed", self.seed)),
            "pose": str(payload.get("pose", self.pose)),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"saved": True, "path": str(self.output_path), "preset": asdict(preset)}

    def _render_frames(self, preset: Camera1Preset) -> dict[str, Any]:
        camera = _make_free_camera(self.env, preset)
        self.renderers["camera1"].update_scene(self.env.unwrapped.data, camera=camera)
        camera1 = _rotate_pixels(self.renderers["camera1"].render(), preset.rotation_degrees)

        self.renderers["wrist"].update_scene(self.env.unwrapped.data, camera=_make_camera(self.env, "wrist_cam"))
        wrist = self.renderers["wrist"].render()

        self.renderers["top"].update_scene(self.env.unwrapped.data, camera=_make_camera(self.env, "top_down"))
        top = postprocess_camera_frame("top_down", self.renderers["top"].render())

        self.renderers["scene"].update_scene(self.env.unwrapped.data)
        scene = self.renderers["scene"].render()
        return {"camera1": camera1, "wrist": wrist, "top": top, "scene": scene}

    def _load_preset(self) -> Camera1Preset:
        if not self.output_path.exists():
            return Camera1Preset(width=self.width, height=self.height)
        try:
            data = json.loads(self.output_path.read_text(encoding="utf-8"))
            preset = data.get("preset", data)
            return _preset_from_mapping(preset, default=Camera1Preset(width=self.width, height=self.height), width=self.width, height=self.height)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return Camera1Preset(width=self.width, height=self.height)

    def _set_pose(self, pose: str) -> None:
        if pose == "home":
            return
        from train_so101_wrist_ego_visual_servo import _set_qpos, make_teacher_targets

        candidates = make_teacher_targets(self.env)
        overhead = [item for item in candidates if str(item["meta"].get("mode")) == "overhead"]
        if not overhead:
            return
        best = max(overhead, key=lambda item: float(item["meta"].get("score", -1e9)))
        qpos = np.asarray(best["q_open"], dtype=np.float32)
        if pose == "elevated_top":
            qpos = _offset_qpos_by_cartesian(self.env, qpos, np.asarray([0.0, 0.0, 0.7]))
        _set_qpos(self.env, qpos)


def _make_free_camera(env: Any, preset: Camera1Preset) -> Any:
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [preset.lookat_x, preset.lookat_y, preset.lookat_z]
    camera.distance = float(preset.distance)
    camera.azimuth = float(preset.azimuth)
    camera.elevation = float(preset.elevation)
    return camera


def _offset_qpos_by_cartesian(env: Any, qpos: np.ndarray, offset: np.ndarray, *, steps: int = 10) -> np.ndarray:
    from physical_ai_agent.sim.so101_live_viewer import _cartesian_error_controller_action
    from train_so101_wrist_ego_visual_servo import _current_qpos, _restore_sim_state, _set_qpos, _snapshot_sim_state

    snapshot = _snapshot_sim_state(env)
    try:
        target = np.clip(np.asarray(qpos, dtype=np.float32), env.action_space.low, env.action_space.high)
        gripper_value = float(target[-1])
        _set_qpos(env, target)
        per_step_offset = np.asarray(offset, dtype=float) / float(max(1, int(steps)))
        action = target.copy()
        for _ in range(max(1, int(steps))):
            action = np.asarray(_cartesian_error_controller_action(env, per_step_offset), dtype=np.float32)
            action[-1] = gripper_value
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            _obs, _reward, terminated, truncated, _info = env.step(np.asarray(action, dtype=float))
            if terminated or truncated:
                break
        result = _current_qpos(env).astype(np.float32)
        result[-1] = gripper_value
        return np.clip(result, env.action_space.low, env.action_space.high).astype(np.float32)
    finally:
        _restore_sim_state(env, snapshot)


def _rotate_pixels(pixels: Any, degrees: int) -> Any:
    quarter_turns = (int(degrees) // 90) % 4
    if quarter_turns == 0:
        return pixels
    return np.rot90(pixels, k=-quarter_turns).copy()


def _png_data_url(pixels: Any) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _preset_from_query(query: dict[str, list[str]], *, default: Camera1Preset, width: int, height: int) -> Camera1Preset:
    return Camera1Preset(
        lookat_x=_query_float(query, "lookat_x", default.lookat_x),
        lookat_y=_query_float(query, "lookat_y", default.lookat_y),
        lookat_z=_query_float(query, "lookat_z", default.lookat_z),
        distance=_query_float(query, "distance", default.distance),
        azimuth=_query_float(query, "azimuth", default.azimuth),
        elevation=_query_float(query, "elevation", default.elevation),
        rotation_degrees=_normalize_rotation(_query_int(query, "rotation_degrees", default.rotation_degrees)),
        width=width,
        height=height,
    )


def _preset_from_mapping(data: dict[str, Any], *, default: Camera1Preset, width: int, height: int) -> Camera1Preset:
    return Camera1Preset(
        lookat_x=float(data.get("lookat_x", default.lookat_x)),
        lookat_y=float(data.get("lookat_y", default.lookat_y)),
        lookat_z=float(data.get("lookat_z", default.lookat_z)),
        distance=float(data.get("distance", default.distance)),
        azimuth=float(data.get("azimuth", default.azimuth)),
        elevation=float(data.get("elevation", default.elevation)),
        rotation_degrees=_normalize_rotation(int(data.get("rotation_degrees", default.rotation_degrees))),
        width=width,
        height=height,
    )


def _query_float(query: dict[str, list[str]], name: str, default: float) -> float:
    try:
        return float(query.get(name, [default])[0])
    except (TypeError, ValueError):
        return float(default)


def _query_int(query: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(query.get(name, [default])[0])
    except (TypeError, ValueError):
        return int(default)


def _query_string(query: dict[str, list[str]], name: str, default: str) -> str:
    value = str(query.get(name, [default])[0])
    return value if value in {"home", "pregrasp", "elevated_top"} else default


def _normalize_rotation(value: int) -> int:
    return int(round(value / 90.0) * 90) % 360


HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>SO101 Camera1 Tuner</title>
    <style>
      :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      body { margin: 0; background: #f4f2ec; color: #202124; }
      main { min-height: 100vh; display: grid; grid-template-columns: 340px minmax(0, 1fr); }
      aside { border-right: 1px solid #cfcac0; padding: 16px; background: #eeebe3; overflow: auto; }
      h1 { font-size: 18px; margin: 0 0 14px; font-weight: 680; }
      .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; padding: 16px; }
      figure { margin: 0; background: #ffffff; border: 1px solid #d8d3ca; border-radius: 8px; overflow: hidden; min-width: 0; }
      figure.primary { grid-column: span 2; }
      figcaption { padding: 9px 10px; font-size: 13px; font-weight: 650; border-bottom: 1px solid #e4e0d8; background: #faf9f5; }
      img { display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #111; }
      label { display: grid; gap: 6px; font-size: 12px; font-weight: 620; margin: 12px 0; }
      .row { display: grid; grid-template-columns: minmax(0, 1fr) 86px; gap: 8px; align-items: center; }
      input[type="range"] { width: 100%; }
      input[type="number"], select { box-sizing: border-box; width: 100%; height: 30px; border: 1px solid #bab4aa; border-radius: 6px; padding: 4px 7px; background: #fff; color: #202124; }
      button { height: 34px; border: 1px solid #9f978a; border-radius: 7px; background: #fff; color: #202124; font-weight: 650; cursor: pointer; }
      button:active { transform: translateY(1px); }
      .buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 14px; }
      .status { min-height: 38px; margin-top: 12px; padding: 10px; border: 1px solid #d4cfc5; border-radius: 7px; background: #f9f7f1; font-size: 12px; line-height: 1.35; word-break: break-word; }
      .preset { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 10px 0 14px; }
      @media (max-width: 980px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid #cfcac0; } .grid { grid-template-columns: 1fr; } figure.primary { grid-column: span 1; } }
    </style>
  </head>
  <body>
    <main>
      <aside>
        <h1>SO101 Camera1 Tuner</h1>
        <label>Pose<select id="pose"><option value="pregrasp">pregrasp</option><option value="elevated_top">elevated_top</option><option value="home">home</option></select></label>
        <label>Seed<div class="row"><input id="seedRange" type="range" min="0" max="200" step="1"><input id="seed" type="number" step="1"></div></label>
        <div class="preset">
          <button data-preset="default">Default</button>
          <button data-preset="hardware">HW-ish</button>
          <button data-preset="front">Front</button>
        </div>
        <div id="controls"></div>
        <div class="buttons">
          <button id="reset">Reset</button>
          <button id="save">Save</button>
        </div>
        <div class="status" id="status"></div>
      </aside>
      <section class="grid">
        <figure class="primary"><figcaption>camera1 candidate</figcaption><img id="camera1" alt=""></figure>
        <figure><figcaption>camera2 wrist</figcaption><img id="wrist" alt=""></figure>
        <figure><figcaption>top/debug</figcaption><img id="top" alt=""></figure>
        <figure><figcaption>scene</figcaption><img id="scene" alt=""></figure>
      </section>
    </main>
    <script>
      const defaults = {lookat_x:0.245, lookat_y:0.11, lookat_z:0.035, distance:0.63, azimuth:270, elevation:-82, rotation_degrees:90};
      const fields = [
        ["lookat_x", -0.25, 0.45, 0.005],
        ["lookat_y", -0.35, 0.35, 0.005],
        ["lookat_z", -0.10, 0.35, 0.005],
        ["distance", 0.20, 1.60, 0.01],
        ["azimuth", 0, 360, 1],
        ["elevation", -89, 10, 1],
        ["rotation_degrees", 0, 270, 90],
      ];
      const presets = {
        default: defaults,
        hardware: {lookat_x:0.245, lookat_y:0.11, lookat_z:0.035, distance:0.63, azimuth:270, elevation:-82, rotation_degrees:90},
        front: {lookat_x:0.15, lookat_y:0.0, lookat_z:0.04, distance:0.70, azimuth:180, elevation:-35, rotation_degrees:0},
      };
      const state = {...defaults, seed:0, pose:"pregrasp"};
      const controls = document.getElementById("controls");
      const status = document.getElementById("status");
      let timer = null;

      function makeControl([name, min, max, step]) {
        const label = document.createElement("label");
        label.textContent = name;
        const row = document.createElement("div");
        row.className = "row";
        const range = document.createElement("input");
        range.type = "range"; range.min = min; range.max = max; range.step = step; range.value = state[name];
        const number = document.createElement("input");
        number.type = "number"; number.min = min; number.max = max; number.step = step; number.value = state[name];
        range.addEventListener("input", () => { number.value = range.value; state[name] = Number(range.value); scheduleRender(); });
        number.addEventListener("input", () => { range.value = number.value; state[name] = Number(number.value); scheduleRender(); });
        row.append(range, number);
        label.append(row);
        label.dataset.field = name;
        return label;
      }

      fields.forEach(field => controls.appendChild(makeControl(field)));
      const seed = document.getElementById("seed");
      const seedRange = document.getElementById("seedRange");
      const pose = document.getElementById("pose");
      seed.value = seedRange.value = state.seed;
      pose.value = state.pose;
      seed.addEventListener("input", () => { seedRange.value = seed.value; state.seed = Number(seed.value); scheduleRender(); });
      seedRange.addEventListener("input", () => { seed.value = seedRange.value; state.seed = Number(seedRange.value); scheduleRender(); });
      pose.addEventListener("change", () => { state.pose = pose.value; scheduleRender(); });

      document.querySelectorAll("[data-preset]").forEach(button => {
        button.addEventListener("click", () => applyPreset(presets[button.dataset.preset]));
      });
      document.getElementById("reset").addEventListener("click", () => applyPreset(defaults));
      document.getElementById("save").addEventListener("click", savePreset);

      function applyPreset(preset) {
        Object.assign(state, preset);
        fields.forEach(([name]) => {
          const label = document.querySelector(`[data-field="${name}"]`);
          const [range, number] = label.querySelectorAll("input");
          range.value = number.value = state[name];
        });
        scheduleRender();
      }

      function scheduleRender() {
        clearTimeout(timer);
        timer = setTimeout(render, 80);
      }

      async function render() {
        const params = new URLSearchParams();
        fields.forEach(([name]) => params.set(name, state[name]));
        params.set("seed", state.seed);
        params.set("pose", state.pose);
        const response = await fetch(`/api/render?${params.toString()}`);
        const payload = await response.json();
        for (const [name, src] of Object.entries(payload.frames)) document.getElementById(name).src = src;
        status.textContent = JSON.stringify(payload.mujoco_camera);
      }

      async function savePreset() {
        const payload = {...state};
        fields.forEach(([name]) => payload[name] = state[name]);
        const response = await fetch("/api/save", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
        const result = await response.json();
        status.textContent = `saved ${result.path}`;
      }

      fetch("/api/preset").then(r => r.json()).then(payload => {
        Object.assign(state, payload.preset || {});
        state.seed = payload.seed ?? state.seed;
        state.pose = payload.pose ?? state.pose;
        seed.value = seedRange.value = state.seed;
        pose.value = state.pose;
        applyPreset(state);
      });
    </script>
  </body>
</html>
"""


if __name__ == "__main__":
    main()
