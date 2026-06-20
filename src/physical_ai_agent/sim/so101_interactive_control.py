from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, SO101NexusEnv, sample_action


SO101_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@dataclass(frozen=True)
class InteractiveAction:
    values: list[float]
    source: str
    hardware_aligned_joint_order: list[str]
    real_robot_safe_to_execute: bool
    notes: list[str]


@dataclass(frozen=True)
class InteractiveSafetyReport:
    status: str
    candidate_safe_for_sim: bool
    execution_allowed_on_real_robot: bool
    action_dim: int
    expected_action_dim: int
    action_space_low: list[float]
    action_space_high: list[float]
    blockers: list[str]
    notes: list[str]


class SO101InteractiveSession:
    def __init__(self, *, env_id: str, output_dir: Path, seed: int = 0) -> None:
        self.env_id = env_id
        self.output_dir = output_dir
        self.seed = seed
        self.env = SO101NexusEnv(env_id=env_id, render_mode=None)
        self.observation, self.info = self.env.reset(seed=seed)
        self.step_index = 0
        self.total_reward = 0.0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.latest_observation_path = self.output_dir / "latest_observation.json"
        self.manifest_path = self.output_dir / "session_manifest.json"
        self._write_manifest(status="running")
        self._write_observation()

    def close(self, *, status: str = "closed") -> None:
        self._write_manifest(status=status)
        self.env.close()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        self.observation, self.info = self.env.reset(seed=self.seed)
        self.step_index = 0
        self.total_reward = 0.0
        event = {
            "type": "reset",
            "seed": self.seed,
            "observation": self.observation,
            "info": _json_safe(self.info),
        }
        self._append_event(event)
        self._write_observation()
        return event

    def observe(self) -> dict[str, Any]:
        event = {
            "type": "observe",
            "step": self.step_index,
            "observation": self.observation,
            "info": _json_safe(self.info),
            "hardware_alignment": hardware_alignment_contract(),
        }
        self._append_event(event)
        self._write_observation()
        return event

    def step(self, action: InteractiveAction) -> dict[str, Any]:
        safety = self.evaluate_action(action.values)
        if not safety.candidate_safe_for_sim:
            event = {
                "type": "blocked_step",
                "step": self.step_index,
                "action": asdict(action),
                "safety": asdict(safety),
            }
            self._append_event(event)
            return event

        previous_observation = self.observation
        observation, reward, terminated, truncated, info = self.env.step(action.values)
        self.observation = observation
        self.info = info
        self.total_reward += reward
        event = {
            "type": "step",
            "step": self.step_index,
            "previous_observation": previous_observation,
            "action": asdict(action),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "observation": observation,
            "info": _json_safe(info),
            "safety": asdict(safety),
        }
        self.step_index += 1
        if terminated or truncated:
            event["next_required_action"] = "reset"
        self._append_event(event)
        self._write_observation()
        return event

    def step_chunk(self, actions: list[InteractiveAction]) -> dict[str, Any]:
        events = []
        for action in actions:
            event = self.step(action)
            events.append(event)
            if event["type"] != "step" or event.get("terminated") or event.get("truncated"):
                break
        chunk_event = {
            "type": "chunk_summary",
            "requested_steps": len(actions),
            "executed_sim_steps": sum(1 for event in events if event["type"] == "step"),
            "events": events,
        }
        self._append_event(chunk_event)
        return chunk_event

    def evaluate_action(self, values: list[float]) -> InteractiveSafetyReport:
        low = [float(value) for value in self.env.action_space.low]
        high = [float(value) for value in self.env.action_space.high]
        blockers = validate_action_values(values, low=low, high=high)
        return InteractiveSafetyReport(
            status="passed" if not blockers else "blocked",
            candidate_safe_for_sim=not blockers,
            execution_allowed_on_real_robot=False,
            action_dim=len(values),
            expected_action_dim=len(low),
            action_space_low=low,
            action_space_high=high,
            blockers=blockers
            + [
                "Real SO-101 execution is intentionally disabled in this interactive sim path.",
                "Run a separate live-readonly hardware preflight before any real motor command.",
            ],
            notes=[
                "The command was evaluated against the SO101-Nexus action space only.",
                "The action order is aligned to the SO-100/SO-101 six-joint convention.",
                "These values are not Dynamixel raw ticks and are not real motor targets.",
            ],
        )

    def sample(self, fraction: float | None = None) -> InteractiveAction:
        fraction = (self.step_index % 120) / 119.0 if fraction is None else fraction
        return make_interactive_action(
            sample_action(self.env.action_space, fraction),
            source=f"sample fraction={fraction:.4f}",
        )

    def center(self) -> InteractiveAction:
        low = self.env.action_space.low
        high = self.env.action_space.high
        return make_interactive_action(
            [float((lo + hi) / 2.0) for lo, hi in zip(low, high, strict=True)],
            source="center",
        )

    def _write_manifest(self, *, status: str) -> None:
        payload = {
            "operation": "so101_interactive_control",
            "status": status,
            "env_id": self.env_id,
            "seed": self.seed,
            "step_index": self.step_index,
            "total_reward": round(self.total_reward, 6),
            "events_path": str(self.events_path),
            "latest_observation_path": str(self.latest_observation_path),
            "hardware_alignment": hardware_alignment_contract(),
        }
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_observation(self) -> None:
        payload = {
            "env_id": self.env_id,
            "step": self.step_index,
            "observation": self.observation,
            "info": _json_safe(self.info),
            "hardware_alignment": hardware_alignment_contract(),
        }
        self.latest_observation_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _append_event(self, event: dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            **event,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def make_interactive_action(values: list[float], *, source: str) -> InteractiveAction:
    return InteractiveAction(
        values=[float(value) for value in values],
        source=source,
        hardware_aligned_joint_order=SO101_JOINT_ORDER[:],
        real_robot_safe_to_execute=False,
        notes=[
            "Simulation-only action candidate.",
            "Use as a proposal artifact for a future real SO-101 adapter, not as a motor command.",
        ],
    )


def validate_action_values(
    values: list[float],
    *,
    low: list[float],
    high: list[float],
) -> list[str]:
    blockers: list[str] = []
    if len(values) != len(low):
        blockers.append(f"Expected {len(low)} action values, got {len(values)}.")
        return blockers
    for index, value in enumerate(values):
        joint = SO101_JOINT_ORDER[index] if index < len(SO101_JOINT_ORDER) else f"joint_{index}"
        if not math.isfinite(value):
            blockers.append(f"{joint}: action_not_finite")
        elif value < low[index] or value > high[index]:
            blockers.append(
                f"{joint}: action_outside_sim_bounds "
                f"({value:.6g} not in [{low[index]:.6g}, {high[index]:.6g}])"
            )
    return blockers


def hardware_alignment_contract() -> dict[str, Any]:
    return {
        "joint_order": SO101_JOINT_ORDER,
        "command_surface": "six_dimensional_so101_action_candidate",
        "real_robot_execution": "disabled",
        "send_action_called": False,
        "policy_actions_executed_on_hardware": False,
        "requires_before_real_execution": [
            "live SO-101 readback adapter",
            "calibration and joint-limit manifest",
            "camera contract for policy and observer views",
            "human confirmation",
            "bounded execution packet",
            "home-return and torque-off report",
        ],
    }


def parse_command(line: str, session: SO101InteractiveSession) -> tuple[bool, dict[str, Any]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True, {"type": "noop"}
    parts = stripped.split(maxsplit=2)
    command = parts[0].lower()
    if command in {"quit", "exit"}:
        return False, {"type": "quit"}
    if command == "help":
        return True, {"type": "help", "commands": command_help()}
    if command == "observe":
        return True, session.observe()
    if command == "reset":
        seed = int(parts[1]) if len(parts) > 1 else None
        return True, session.reset(seed)
    if command == "sample":
        fraction = float(parts[1]) if len(parts) > 1 else None
        return True, session.step(session.sample(fraction))
    if command == "center":
        return True, session.step(session.center())
    if command == "action":
        payload = stripped[len("action") :].strip()
        action = make_interactive_action(_parse_action_list(payload), source="manual action")
        return True, session.step(action)
    if command == "nudge":
        if len(parts) < 3:
            raise ValueError("nudge requires: nudge <joint> <value>")
        joint = parts[1]
        value = float(parts[2])
        return True, session.step(_nudge_action(session, joint, value))
    if command == "chunk":
        payload = stripped[len("chunk") :].strip()
        actions = [
            make_interactive_action([float(value) for value in row], source="manual chunk")
            for row in _parse_chunk(payload)
        ]
        return True, session.step_chunk(actions)
    raise ValueError(f"Unknown command {command!r}. Use `help` for commands.")


def command_help() -> list[str]:
    return [
        "observe",
        "sample [fraction]",
        "center",
        "action [a0,a1,a2,a3,a4,a5]",
        "nudge <joint_name> <value>",
        "chunk [[a0,...,a5],[a0,...,a5]]",
        "reset [seed]",
        "quit",
    ]


def run_scripted_session(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if args.dry_contract:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "hardware_alignment_contract.json"
        path.write_text(
            json.dumps(hardware_alignment_contract(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps({"status": "passed", "contract_path": str(path)}, sort_keys=True))
        return 0

    if args.gui:
        session = SO101InteractiveSession(env_id=args.env_id, output_dir=output_dir, seed=args.seed)
        server = SO101InteractiveGuiServer(session=session, host=args.host, port=args.port)
        try:
            server.serve()
        finally:
            session.close(status="passed")
        return 0

    commands: list[str] = []
    for command in args.command or []:
        commands.append(command)
    if args.script:
        commands.extend(Path(args.script).read_text(encoding="utf-8").splitlines())

    session = SO101InteractiveSession(env_id=args.env_id, output_dir=output_dir, seed=args.seed)
    exit_code = 0
    try:
        if commands:
            for line in commands:
                try:
                    keep_running, event = parse_command(line, session)
                    if event["type"] != "noop":
                        print(json.dumps(event, sort_keys=True))
                    if not keep_running:
                        break
                except Exception as exc:  # noqa: BLE001
                    exit_code = 2
                    print(
                        json.dumps(
                            {"type": "error", "command": line, "error": str(exc)},
                            sort_keys=True,
                        )
                    )
                    break
        else:
            print(
                "SO101 interactive sim ready. Type `help`, `observe`, `sample`, "
                "`nudge shoulder_pan 0.2`, or `quit`."
            )
            while True:
                try:
                    line = input("so101> ")
                except EOFError:
                    break
                try:
                    keep_running, event = parse_command(line, session)
                    if event["type"] == "help":
                        print("\n".join(event["commands"]))
                    elif event["type"] != "noop":
                        print(json.dumps(event, indent=2, sort_keys=True))
                    if not keep_running:
                        break
                except Exception as exc:  # noqa: BLE001
                    print(f"error: {exc}", file=sys.stderr)
    finally:
        session.close(status="passed" if exit_code == 0 else "blocked")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lightweight interactive SO101-Nexus control loop for Codex-driven simulation."
    )
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="_workspace/so101_interactive/latest")
    parser.add_argument("--script", type=Path)
    parser.add_argument("--command", action="append")
    parser.add_argument("--gui", action="store_true", help="Serve a lightweight browser control GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--dry-contract",
        action="store_true",
        help="Write the sim-to-real hardware alignment contract without importing SO101-Nexus.",
    )
    return parser


class SO101InteractiveGuiServer:
    def __init__(self, *, session: SO101InteractiveSession, host: str, port: int) -> None:
        self.session = session
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self.last_event: dict[str, Any] = session.observe()
        self.renderer = SO101GuiRenderer(session)
        self.server = ThreadingHTTPServer((host, port), self._make_handler())

    def serve(self) -> None:
        print(f"SO101 interactive GUI available at http://{self.host}:{self.port}", flush=True)
        try:
            self.server.serve_forever()
        finally:
            self.server.server_close()

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "manifest": {
                    "env_id": self.session.env_id,
                    "step": self.session.step_index,
                    "total_reward": round(self.session.total_reward, 6),
                    "output_dir": str(self.session.output_dir),
                },
                "latest_observation": self.session.observation,
                "latest_info": _json_safe(self.session.info),
                "latest_event": self.last_event,
                "hardware_alignment": hardware_alignment_contract(),
            }

    def command(self, command_line: str) -> dict[str, Any]:
        with self._lock:
            keep_running, event = parse_command(command_line, self.session)
            self.last_event = event
            if not keep_running:
                threading.Thread(target=self._shutdown_soon, daemon=True).start()
            return event

    def _shutdown_soon(self) -> None:
        time.sleep(0.1)
        self.server.shutdown()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        gui = self

        class SO101InteractiveGuiHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path in {"/", "/index.html"}:
                    self._write_html()
                    return
                if path == "/api/state":
                    self._write_json(gui.state())
                    return
                if path == "/scene.jpg":
                    self._write_scene()
                    return
                self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path != "/api/command":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                command_line = str(payload.get("command", "")).strip()
                try:
                    event = gui.command(command_line)
                    self._write_json({"ok": True, "event": event, "state": gui.state()})
                except Exception as exc:  # noqa: BLE001
                    self._write_json({"ok": False, "error": str(exc), "state": gui.state()}, status=400)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_scene(self) -> None:
                body = gui.renderer.render_jpeg()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_html(self) -> None:
                body = _gui_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return SO101InteractiveGuiHandler


class SO101GuiRenderer:
    def __init__(self, session: SO101InteractiveSession, width: int = 640, height: int = 480) -> None:
        self.session = session
        self.width = width
        self.height = height

    def render_jpeg(self) -> bytes:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (self.width, self.height), (238, 241, 235))
        draw = ImageDraw.Draw(image)
        obs = [float(value) for value in self.session.observation[:6]]
        points = _planar_arm_points(obs, origin=(self.width // 2, 330), scale=58.0)

        draw.rectangle((0, 0, self.width, 42), fill=(245, 247, 242))
        draw.text(
            (12, 10),
            f"SO101 sim  step {self.session.step_index}  reward {self.session.total_reward:.4f}",
            fill=(25, 32, 28),
        )
        draw.line((40, 360, self.width - 40, 360), fill=(199, 206, 196), width=2)
        target = _target_point(self.session.info, self.width, self.height)
        draw.ellipse((target[0] - 9, target[1] - 9, target[0] + 9, target[1] + 9), fill=(28, 126, 92))
        draw.text((target[0] + 12, target[1] - 7), "target", fill=(42, 86, 69))
        for start, end in zip(points[:-1], points[1:], strict=True):
            draw.line((*start, *end), fill=(38, 80, 134), width=11)
            draw.line((*start, *end), fill=(94, 145, 205), width=5)
        for index, point in enumerate(points):
            radius = 11 if index == 0 else 8
            draw.ellipse(
                (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                fill=(247, 248, 245),
                outline=(37, 70, 108),
                width=2,
            )
        tcp = points[-1]
        draw.ellipse((tcp[0] - 6, tcp[1] - 6, tcp[0] + 6, tcp[1] + 6), fill=(215, 94, 62))
        draw.text((24, 404), _sim_info_text(self.session.info), fill=(45, 55, 49))
        draw.text((24, 430), "2D lightweight view; real hardware execution disabled", fill=(92, 76, 56))
        return _encode_jpeg(image)


def _encode_jpeg(image: Any) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def _planar_arm_points(
    observation: list[float],
    *,
    origin: tuple[int, int],
    scale: float,
) -> list[tuple[int, int]]:
    angles = observation[:5] + [0.0] * max(0, 5 - len(observation))
    lengths = [1.0, 0.85, 0.68, 0.46, 0.28]
    points = [origin]
    heading = -math.pi / 2.0
    x, y = float(origin[0]), float(origin[1])
    for angle, length in zip(angles, lengths, strict=True):
        heading += float(angle) * 0.45
        x += math.cos(heading) * length * scale
        y += math.sin(heading) * length * scale
        points.append((int(round(x)), int(round(y))))
    return points


def _target_point(info: dict[str, Any], width: int, height: int) -> tuple[int, int]:
    distance = float(info.get("tcp_to_target_dist", 0.18) or 0.18)
    x = int(width * 0.72)
    y = int(height * 0.28 + min(0.3, distance) * 220)
    return x, y


def _sim_info_text(info: dict[str, Any]) -> str:
    distance = info.get("tcp_to_target_dist", "unknown")
    success = info.get("success", False)
    return f"tcp_to_target_dist={distance}  success={success}"


def _gui_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SO101 Interactive Control</title>
  <style>
    :root { color-scheme: light; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: #f4f6f2; color: #17201b; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }
    h1 { font-size: 24px; margin: 0 0 16px; font-weight: 720; }
    .status { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; color: #415148; }
    .layout { display: grid; grid-template-columns: 360px 1fr; gap: 16px; align-items: start; }
    section { background: #fff; border: 1px solid #d8ddd4; border-radius: 8px; padding: 14px; }
    h2 { font-size: 14px; margin: 0 0 12px; text-transform: uppercase; color: #526157; }
    .row { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }
    button { height: 34px; border: 1px solid #aeb8ad; background: #eef2ec; border-radius: 6px; padding: 0 12px; cursor: pointer; }
    button:hover { background: #e4eadf; }
    input { height: 32px; border: 1px solid #c5ccc2; border-radius: 6px; padding: 0 9px; min-width: 0; }
    input[type="number"] { width: 90px; }
    input.command { flex: 1; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    label { width: 96px; font-size: 13px; color: #526157; }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .scene { padding: 0; overflow: hidden; position: relative; background: #dfe6df; }
    .scene canvas { display: block; width: 100%; aspect-ratio: 16 / 10; background: linear-gradient(#eef3f0, #d7dfd6); }
    .scene .hud { position: absolute; left: 12px; top: 10px; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; color: #1d2b24; background: rgba(247,250,246,.78); border: 1px solid rgba(120,136,123,.35); border-radius: 6px; padding: 6px 8px; }
    .bars { display: grid; gap: 8px; }
    .bar { display: grid; grid-template-columns: 96px 1fr 80px; gap: 8px; align-items: center; }
    .track { height: 10px; background: #edf0eb; border: 1px solid #d5dbd2; border-radius: 999px; overflow: hidden; }
    .fill { height: 100%; width: 50%; background: #4f7fc7; }
    .negative { background: #d06b4a; }
    @media (max-width: 860px) { .layout { grid-template-columns: 1fr; } main { padding: 14px; } }
  </style>
</head>
<body>
<main>
  <header>
    <h1>SO101 Interactive Control</h1>
    <div class="status" id="status">connecting</div>
  </header>
  <div class="layout">
    <section>
      <h2>Commands</h2>
      <div class="row"><button onclick="send('observe')">Observe</button><button onclick="send('sample')">Sample Step</button><button onclick="send('center')">Center</button></div>
      <div class="row"><button id="autoButton" onclick="toggleAuto()">Auto Run</button><button onclick="send('sample 0.25')">Pose A</button><button onclick="send('sample 0.55')">Pose B</button><button onclick="send('sample 0.85')">Pose C</button></div>
      <div class="row"><label>Joint</label><input id="joint" value="shoulder_pan"><input id="nudge" type="number" step="0.01" value="0.06"><button onpointerdown="holdNudge(1)" onpointerup="stopHold()" onpointerleave="stopHold()">+ Hold</button><button onpointerdown="holdNudge(-1)" onpointerup="stopHold()" onpointerleave="stopHold()">- Hold</button></div>
      <div class="row"><input class="command" id="command" value="action [0,0,0,0,0,0]"><button onclick="sendCommand()">Run</button></div>
      <div class="row"><button onclick="send('reset')">Reset</button><button onclick="send('quit')">Stop Server</button></div>
    </section>
    <section class="scene">
      <canvas id="sceneCanvas" width="960" height="600" aria-label="SO101 3D simulation viewer"></canvas>
      <div class="hud" id="sceneHud">SO101 3D viewer</div>
    </section>
    <section>
      <h2>State</h2>
      <div class="bars" id="bars"></div>
    </section>
    <section>
      <h2>Last Event</h2>
      <pre id="event">{}</pre>
    </section>
    <section>
      <h2>Safety Boundary</h2>
      <pre id="contract">{}</pre>
    </section>
  </div>
</main>
<script>
async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
async function refresh() {
  const state = await api('/api/state');
  render(state);
}
async function send(command) {
  const data = await api('/api/command', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({command})});
  render(data.state);
}
function sendCommand() { send(document.getElementById('command').value); }
function nudge() { send(`nudge ${document.getElementById('joint').value} ${document.getElementById('nudge').value}`); }
let currentState = null;
let previousObs = null;
let targetObs = null;
let transitionStart = performance.now();
let holdTimer = null;
let autoTimer = null;
let autoPhase = 0;
function toggleAuto() {
  const button = document.getElementById('autoButton');
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
    button.textContent = 'Auto Run';
    return;
  }
  button.textContent = 'Stop Auto';
  autoTimer = setInterval(() => {
    autoPhase = (autoPhase + 0.025) % 1;
    send(`sample ${autoPhase.toFixed(3)}`).catch(() => toggleAuto());
  }, 90);
}
function holdNudge(sign) {
  stopHold();
  const step = () => {
    const joint = document.getElementById('joint').value;
    const value = Number(document.getElementById('nudge').value || 0.04) * sign;
    send(`nudge ${joint} ${value.toFixed(4)}`);
  };
  step();
  holdTimer = setInterval(step, 85);
}
function stopHold() {
  if (holdTimer) clearInterval(holdTimer);
  holdTimer = null;
}
function render(state) {
  currentState = state;
  previousObs = targetObs || state.latest_observation || [];
  targetObs = state.latest_observation || [];
  transitionStart = performance.now();
  document.getElementById('status').textContent = `step ${state.manifest.step}  reward ${state.manifest.total_reward}  ${state.manifest.env_id}`;
  document.getElementById('event').textContent = JSON.stringify(state.latest_event, null, 2);
  document.getElementById('contract').textContent = JSON.stringify(state.hardware_alignment, null, 2);
  const names = state.hardware_alignment.joint_order;
  const values = state.latest_observation || [];
  document.getElementById('bars').innerHTML = names.map((name, i) => {
    const value = Number(values[i] || 0);
    const pct = Math.max(0, Math.min(100, 50 + value * 25));
    const cls = value < 0 ? 'fill negative' : 'fill';
    return `<div class="bar"><span>${name}</span><div class="track"><div class="${cls}" style="width:${pct}%"></div></div><code>${value.toFixed(4)}</code></div>`;
  }).join('');
}
function animationLoop(now) {
  if (currentState) {
    const t = Math.min(1, (now - transitionStart) / 180);
    const eased = 1 - Math.pow(1 - t, 3);
    const blended = (targetObs || []).map((value, i) => {
      const start = Number((previousObs || [])[i] || value || 0);
      return start + (Number(value || 0) - start) * eased;
    });
    drawScene3D({...currentState, latest_observation: blended});
  }
  requestAnimationFrame(animationLoop);
}
function drawScene3D(state) {
  const canvas = document.getElementById('sceneCanvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const obs = state.latest_observation || [];
  ctx.clearRect(0, 0, w, h);
  const gradient = ctx.createLinearGradient(0, 0, 0, h);
  gradient.addColorStop(0, '#eef5f2');
  gradient.addColorStop(1, '#cfd9cf');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, w, h);
  drawGrid(ctx, w, h);
  const points = armPoints3D(obs);
  const target = {x: 1.9, y: 0.4, z: 0.35 + Math.min(0.35, Number(state.latest_info?.tcp_to_target_dist || 0.18))};
  drawTarget(ctx, project(target, w, h));
  for (let i = 0; i < points.length - 1; i++) {
    drawSegment(ctx, project(points[i], w, h), project(points[i + 1], w, h), i);
  }
  points.forEach((p, i) => drawJoint(ctx, project(p, w, h), i));
  const tip = project(points[points.length - 1], w, h);
  ctx.fillStyle = '#d75f43';
  ctx.beginPath();
  ctx.arc(tip.x, tip.y, 9, 0, Math.PI * 2);
  ctx.fill();
  document.getElementById('sceneHud').textContent = `step ${state.manifest.step} | tcp ${Number(state.latest_info?.tcp_to_target_dist || 0).toFixed(4)} | sim-only`;
}
function armPoints3D(obs) {
  const lengths = [0.72, 0.62, 0.46, 0.32, 0.22];
  const q = [0, 0, 0, 0, 0, 0].map((_, i) => Number(obs[i] || 0));
  const points = [{x:0, y:0, z:0}];
  let yaw = q[0] * 0.8;
  let pitch = -0.55 + q[1] * 0.42;
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < lengths.length; i++) {
    pitch += q[Math.min(i + 1, 4)] * 0.22;
    yaw += (i > 2 ? q[4] * 0.08 : 0);
    x += Math.cos(yaw) * Math.cos(pitch) * lengths[i];
    y += Math.sin(yaw) * Math.cos(pitch) * lengths[i];
    z += Math.sin(pitch) * lengths[i];
    points.push({x, y, z});
  }
  return points;
}
function project(p, w, h) {
  const camera = {x: 3.2, y: -5.4, z: 2.6};
  const dx = p.x - camera.x, dy = p.y - camera.y, dz = p.z - camera.z;
  const yaw = 0.52, pitch = -0.34;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * dx - sy * dy;
  const y1 = sy * dx + cy * dy;
  const z1 = dz;
  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1;
  const depth = Math.max(1.2, y2 + 6.0);
  const f = 520 / depth;
  return {x: w / 2 + x1 * f, y: h * 0.58 - z2 * f, depth};
}
function drawGrid(ctx, w, h) {
  ctx.strokeStyle = 'rgba(70,90,78,.22)';
  ctx.lineWidth = 1;
  for (let i = -5; i <= 5; i++) {
    const a = project({x:i * .45, y:-1.4, z:-.85}, w, h);
    const b = project({x:i * .45, y:1.8, z:-.85}, w, h);
    const c = project({x:-2.2, y:i * .32, z:-.85}, w, h);
    const d = project({x:2.2, y:i * .32, z:-.85}, w, h);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(d.x, d.y); ctx.stroke();
  }
}
function drawSegment(ctx, a, b, i) {
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'rgba(31, 53, 74, .22)';
  ctx.lineWidth = 22 - i * 2;
  ctx.beginPath(); ctx.moveTo(a.x + 4, a.y + 7); ctx.lineTo(b.x + 4, b.y + 7); ctx.stroke();
  ctx.strokeStyle = i % 2 ? '#5b8cc9' : '#2e6fae';
  ctx.lineWidth = 16 - i * 1.6;
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  ctx.strokeStyle = 'rgba(255,255,255,.45)';
  ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(a.x - 2, a.y - 3); ctx.lineTo(b.x - 2, b.y - 3); ctx.stroke();
}
function drawJoint(ctx, p, i) {
  ctx.fillStyle = '#f9faf6';
  ctx.strokeStyle = '#27445f';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(p.x, p.y, i === 0 ? 17 : 12, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
}
function drawTarget(ctx, p) {
  ctx.fillStyle = '#20936e';
  ctx.strokeStyle = '#0f5f47';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(p.x, p.y, 13, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
}
refresh().catch(err => document.getElementById('status').textContent = err.message);
requestAnimationFrame(animationLoop);
</script>
</body>
</html>"""


def main() -> int:
    return run_scripted_session(build_parser().parse_args())


def _nudge_action(session: SO101InteractiveSession, joint: str, value: float) -> InteractiveAction:
    if joint not in SO101_JOINT_ORDER:
        raise ValueError(f"Unknown joint {joint!r}; expected one of {SO101_JOINT_ORDER}")
    values = [0.0] * int(session.env.action_space.shape[0])
    values[SO101_JOINT_ORDER.index(joint)] = value
    return make_interactive_action(values, source=f"nudge {joint} {value}")


def _parse_action_list(payload: str) -> list[float]:
    values = json.loads(payload)
    if not isinstance(values, list) or any(isinstance(value, list) for value in values):
        raise ValueError("action payload must be a flat JSON list")
    return [float(value) for value in values]


def _parse_chunk(payload: str) -> list[list[float]]:
    values = json.loads(payload)
    if not isinstance(values, list) or not values:
        raise ValueError("chunk payload must be a non-empty JSON list of action lists")
    rows: list[list[float]] = []
    for row in values:
        if not isinstance(row, list):
            raise ValueError("chunk payload must contain only action lists")
        rows.append([float(value) for value in row])
    return rows


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
