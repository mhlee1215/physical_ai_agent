#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
    "gripper_controller",
]
TASK = "Use MoveIt/Gazebo teacher motion to reach the visible can with the myCobot gripper."
DEFAULT_SOURCE_NOTES = {
    "official_ros_repo": "https://github.com/elephantrobotics/mycobot_ros",
    "official_ros2_repo": "https://github.com/elephantrobotics/mycobot_ros2",
    "candidate_ros1_launch": (
        "roslaunch mycobot_280_gripper_moveit demo_gazebo.launch gazebo_gui:=false"
    ),
    "candidate_unofficial_table_launch": (
        "roslaunch mycobot_move_it_config demo_gazebo.launch gazebo_gui:=false"
    ),
    "official_ros1_moveit_doc": (
        "https://docs.elephantrobotics.com/docs/gitbook-en/12-ApplicationBaseROS/"
        "12.1-ROS1/12.1.5-Moveit/myCobot-280.html"
    ),
}


@dataclass(frozen=True)
class MyCobotTeacherFrame:
    episode_index: int
    frame_index: int
    timestamp: float
    observation_state: list[float]
    action: list[float]
    task: str
    top_image: str
    wrist_image: str
    source: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small myCobot ROS/Gazebo/MoveIt teacher-data POC. The default path "
            "is offline and deterministic; pass --input-trace with JSONL records captured "
            "from ROS topics to convert a real Gazebo/MoveIt run."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("_workspace/mycobot_ros_teacher_poc"))
    parser.add_argument("--input-trace", type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--repo-id", default="physical-ai-agent/mycobot-ros-teacher-poc")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = export_mycobot_ros_teacher_poc(
        root=args.root,
        input_trace=args.input_trace,
        episode_index=args.episode_index,
        frames=args.frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
        repo_id=args.repo_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def export_mycobot_ros_teacher_poc(
    *,
    root: Path,
    input_trace: Path | None,
    episode_index: int,
    frames: int,
    fps: int,
    width: int,
    height: int,
    repo_id: str,
    overwrite: bool,
) -> dict[str, Any]:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    root.mkdir(parents=True)
    (root / "data").mkdir()
    (root / "images" / "top").mkdir(parents=True)
    (root / "images" / "wrist").mkdir(parents=True)
    (root / "meta").mkdir()

    raw_records = (
        load_trace(input_trace)
        if input_trace
        else synthetic_moveit_trace(frames=frames, fps=fps)
    )
    teacher_frames = build_teacher_frames(
        raw_records,
        root=root,
        episode_index=episode_index,
        width=width,
        height=height,
    )
    if not teacher_frames:
        raise ValueError("trace did not produce any teacher frames")

    frames_path = root / "data" / "frames.jsonl"
    with frames_path.open("w", encoding="utf-8") as file:
        for frame in teacher_frames:
            file.write(json.dumps(asdict(frame), sort_keys=True) + "\n")

    episode = {
        "episode_index": episode_index,
        "length": len(teacher_frames),
        "from_frame": 0,
        "to_frame": len(teacher_frames) - 1,
        "task": TASK,
        "success": None,
        "success_label": "not_claimed_poc_trace_only",
    }
    episodes_path = root / "data" / "episodes.jsonl"
    episodes_path.write_text(json.dumps(episode, sort_keys=True) + "\n", encoding="utf-8")

    info = {
        "repo_id": repo_id,
        "robot_type": "mycobot_280_ros_gazebo_moveit_poc",
        "fps": fps,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [len(JOINT_NAMES)]},
            "action": {"dtype": "float32", "shape": [len(JOINT_NAMES)]},
            "observation.images.top": {"dtype": "image", "shape": [height, width, 3]},
            "observation.images.wrist": {"dtype": "image", "shape": [height, width, 3]},
            "task": {"dtype": "string"},
        },
        "joint_names": JOINT_NAMES,
        "source_notes": DEFAULT_SOURCE_NOTES,
        "poc_boundary": (
            "Offline POC adapter for ROS/Gazebo/MoveIt traces. It does not claim "
            "Gazebo task success until object pose/contact success checks are added."
        ),
    }
    info_path = root / "meta" / "info.json"
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    viewer_path = root / "viewer.html"
    report = {
        "operation": "export_mycobot_ros_teacher_poc",
        "status": "passed",
        "root": str(root),
        "input_trace": str(input_trace) if input_trace else "synthetic_moveit_trace",
        "frames": len(teacher_frames),
        "episodes": 1,
        "fps": fps,
        "image_size": [width, height],
        "dataset_files": {
            "frames": str(frames_path),
            "episodes": str(episodes_path),
            "info": str(info_path),
            "viewer": str(viewer_path),
        },
        "poc_boundary": info["poc_boundary"],
        "next_steps": [
            "Record /joint_states, MoveIt FollowJointTrajectory goals, and Gazebo camera topics.",
            "Add Gazebo model-state object pose and gripper/contact success oracle.",
            "Swap placeholder PPM images for decoded ROS image messages before training.",
        ],
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_viewer_html(viewer_path, info=info, report=report, frames=teacher_frames)
    return report


def load_trace(input_trace: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_trace.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_trace}:{line_number}: invalid JSONL record") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{input_trace}:{line_number}: expected object record")
            records.append(payload)
    return records


def synthetic_moveit_trace(*, frames: int, fps: int) -> list[dict[str, Any]]:
    count = max(2, int(frames))
    records = []
    for index in range(count):
        phase = index / float(max(1, count - 1))
        smooth = 0.5 - 0.5 * math.cos(math.pi * phase)
        positions = [
            -0.42 + 0.58 * smooth,
            0.18 - 0.36 * smooth,
            -0.28 + 0.42 * smooth,
            -0.12 - 0.30 * smooth,
            0.20 * math.sin(math.pi * phase),
            0.06 + 0.18 * smooth,
            -0.78 + 0.93 * min(1.0, phase * 1.4),
        ]
        next_phase = min(1.0, (index + 1) / float(max(1, count - 1)))
        next_smooth = 0.5 - 0.5 * math.cos(math.pi * next_phase)
        command = [
            -0.42 + 0.58 * next_smooth,
            0.18 - 0.36 * next_smooth,
            -0.28 + 0.42 * next_smooth,
            -0.12 - 0.30 * next_smooth,
            0.20 * math.sin(math.pi * next_phase),
            0.06 + 0.18 * next_smooth,
            -0.78 + 0.93 * min(1.0, next_phase * 1.4),
        ]
        records.append(
            {
                "timestamp": index / float(fps),
                "joint_state": {"name": JOINT_NAMES, "position": positions},
                "planned_action": command,
                "moveit_goal": "reach_coke_can_on_table",
                "gazebo_world": "table.world",
            }
        )
    return records


def build_teacher_frames(
    records: list[dict[str, Any]],
    *,
    root: Path,
    episode_index: int,
    width: int,
    height: int,
) -> list[MyCobotTeacherFrame]:
    frames: list[MyCobotTeacherFrame] = []
    for frame_index, record in enumerate(records):
        state = extract_joint_vector(record)
        action = extract_action_vector(record, fallback=state)
        top_path = (
            root / "images" / "top" / f"episode_{episode_index:03d}_frame_{frame_index:04d}.ppm"
        )
        wrist_path = (
            root
            / "images"
            / "wrist"
            / f"episode_{episode_index:03d}_frame_{frame_index:04d}.ppm"
        )
        write_state_image(top_path, state, width=width, height=height, view="top")
        write_state_image(wrist_path, action, width=width, height=height, view="wrist")
        frames.append(
            MyCobotTeacherFrame(
                episode_index=episode_index,
                frame_index=frame_index,
                timestamp=float(record.get("timestamp", frame_index)),
                observation_state=state,
                action=action,
                task=str(record.get("task", TASK)),
                top_image=str(top_path.relative_to(root)),
                wrist_image=str(wrist_path.relative_to(root)),
                source={
                    "moveit_goal": record.get("moveit_goal"),
                    "gazebo_world": record.get("gazebo_world"),
                    "source_record_index": frame_index,
                },
            )
        )
    return frames


def extract_joint_vector(record: dict[str, Any]) -> list[float]:
    joint_state = record.get("joint_state") or record.get("joint_states") or {}
    if not isinstance(joint_state, dict):
        raise ValueError("joint_state must be an object")
    positions = joint_state.get("position")
    names = joint_state.get("name") or JOINT_NAMES
    if not isinstance(positions, list):
        raise ValueError("joint_state.position must be a list")
    if not isinstance(names, list):
        raise ValueError("joint_state.name must be a list when provided")
    by_name = {str(name): float(value) for name, value in zip(names, positions)}
    if all(name in by_name for name in JOINT_NAMES):
        return [float(by_name[name]) for name in JOINT_NAMES]
    if len(positions) < len(JOINT_NAMES):
        raise ValueError(f"expected at least {len(JOINT_NAMES)} joint positions")
    return [float(value) for value in positions[: len(JOINT_NAMES)]]


def extract_action_vector(record: dict[str, Any], *, fallback: list[float]) -> list[float]:
    action = (
        record.get("planned_action")
        or record.get("commanded_joint_positions")
        or record.get("trajectory_point")
        or fallback
    )
    if isinstance(action, dict):
        positions = action.get("positions") or action.get("position")
        names = action.get("joint_names") or action.get("name") or JOINT_NAMES
        if not isinstance(positions, list):
            raise ValueError("action dict must include positions")
        by_name = {
            str(name): float(value) for name, value in zip(names, positions)
        }
        if all(name in by_name for name in JOINT_NAMES):
            return [float(by_name[name]) for name in JOINT_NAMES]
        action = positions
    if not isinstance(action, list):
        raise ValueError("planned action must be a list or trajectory point object")
    if len(action) < len(JOINT_NAMES):
        raise ValueError(f"expected at least {len(JOINT_NAMES)} action values")
    return [float(value) for value in action[: len(JOINT_NAMES)]]


def write_state_image(
    path: Path, values: list[float], *, width: int, height: int, view: str
) -> None:
    width = max(16, int(width))
    height = max(16, int(height))
    pixels = bytearray()
    tint = 45 if view == "top" else 85
    for y in range(height):
        for x in range(width):
            idx = min(len(values) - 1, int(x / max(1, width) * len(values)))
            value = max(-1.0, min(1.0, float(values[idx])))
            bar_height = int((value + 1.0) * 0.5 * (height - 1))
            active = y >= height - 1 - bar_height
            r = 40 + int(120 * active) + tint
            g = 42 + int(90 * active) + (idx * 17) % 60
            b = 48 + int(70 * active) + (x + y) % 30
            pixels.extend((min(255, r), min(255, g), min(255, b)))
    with path.open("wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        file.write(pixels)


def write_viewer_html(
    path: Path,
    *,
    info: dict[str, Any],
    report: dict[str, Any],
    frames: list[MyCobotTeacherFrame],
) -> None:
    payload = {
        "info": info,
        "report": report,
        "frames": [asdict(frame) for frame in frames],
    }
    payload_json = json.dumps(payload, sort_keys=True)
    escaped_payload = html.escape(payload_json, quote=False)
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>myCobot ROS Teacher POC</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #161a1d;
      --muted: #5d6670;
      --line: #d7dde3;
      --panel: #f7f9fb;
      --accent: #176b87;
      --accent-2: #b9472f;
      --ok: #22724a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 18px 24px 14px;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .subhead {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    main {{
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      min-height: calc(100vh - 72px);
    }}
    aside {{
      border-right: 1px solid var(--line);
      padding: 18px;
      background: var(--panel);
    }}
    section {{
      padding: 18px 22px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #ffffff;
    }}
    .metric strong {{
      display: block;
      font-size: 18px;
      line-height: 1.1;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .control-row {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 16px;
    }}
    button {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--ink);
      min-width: 40px;
      height: 36px;
      font-weight: 700;
      cursor: pointer;
    }}
    button.primary {{
      background: var(--accent);
      color: #ffffff;
      border-color: var(--accent);
      min-width: 70px;
    }}
    input[type="range"] {{
      width: 100%;
      margin-top: 14px;
      accent-color: var(--accent);
    }}
    .frame-meta {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
      line-height: 1.45;
    }}
    .visuals {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .visuals.render {{
      grid-template-columns: 1fr;
      margin-bottom: 16px;
    }}
    .visual-panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #ffffff;
    }}
    .visual-title {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }}
    canvas {{
      width: 100%;
      aspect-ratio: 16 / 10;
      display: block;
      background: #eef2f5;
    }}
    .render-frame {{
      width: 100%;
      aspect-ratio: 16 / 9;
      display: block;
      object-fit: contain;
      background: #e8edf1;
    }}
    .render-missing {{
      aspect-ratio: 16 / 7;
      background: #e8edf1;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      color: var(--muted);
      text-align: center;
      line-height: 1.45;
    }}
    .table-wrap {{
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    tr:last-child td {{ border-bottom: none; }}
    .boundary {{
      margin-top: 16px;
      border-left: 4px solid var(--accent-2);
      padding: 10px 12px;
      background: #fff8f5;
      color: #50261e;
      font-size: 13px;
      line-height: 1.45;
    }}
    .ok {{
      color: var(--ok);
      font-weight: 700;
    }}
    @media (max-width: 820px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: none; border-bottom: 1px solid var(--line); }}
      .visuals {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <script id="payload" type="application/json">{escaped_payload}</script>
  <header>
    <h1>myCobot ROS Teacher POC</h1>
    <div class="subhead">MoveIt/Gazebo trace adapter preview for Mac-local inspection</div>
  </header>
  <main>
    <aside>
      <div class="metrics">
        <div class="metric"><strong id="frameCount">0</strong><span>frames</span></div>
        <div class="metric"><strong id="fps">0</strong><span>fps</span></div>
        <div class="metric"><strong id="stateDim">0</strong><span>state dim</span></div>
        <div class="metric"><strong class="ok">POC</strong><span>claim</span></div>
      </div>
      <div class="control-row">
        <button id="prevBtn" aria-label="Previous frame">&lt;</button>
        <button id="playBtn" class="primary">Play</button>
        <button id="nextBtn" aria-label="Next frame">&gt;</button>
      </div>
      <input id="frameSlider" type="range" min="0" max="0" value="0" aria-label="Frame">
      <div id="frameMeta" class="frame-meta"></div>
      <div id="boundary" class="boundary"></div>
    </aside>
    <section>
      <div class="visuals render">
        <div class="visual-panel">
          <div class="visual-title">
            <span>MuJoCo Robot Render</span><span id="renderStatus">not generated</span>
          </div>
          <img id="renderImage" class="render-frame" alt="myCobot MuJoCo render frame">
          <div id="renderMissing" class="render-missing">
            Run the MuJoCo render command to generate real robot-arm frames from the
            official myCobot model.
          </div>
        </div>
      </div>
      <div class="visuals">
        <div class="visual-panel">
          <div class="visual-title"><span>Top Observation</span><span>state</span></div>
          <canvas id="topCanvas" width="960" height="600"></canvas>
        </div>
        <div class="visual-panel">
          <div class="visual-title"><span>Wrist Observation</span><span>action</span></div>
          <canvas id="wristCanvas" width="960" height="600"></canvas>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>joint</th><th>state</th><th>action</th><th>delta</th></tr>
          </thead>
          <tbody id="jointRows"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const frames = payload.frames;
    const joints = payload.info.joint_names;
    const slider = document.getElementById("frameSlider");
    const playBtn = document.getElementById("playBtn");
    let index = 0;
    let timer = null;

    document.getElementById("frameCount").textContent = frames.length;
    document.getElementById("fps").textContent = payload.info.fps;
    document.getElementById("stateDim").textContent = joints.length;
    document.getElementById("boundary").textContent = payload.info.poc_boundary;
    slider.max = Math.max(0, frames.length - 1);

    function drawBars(canvas, values, palette) {{
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#eef2f5";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#c8d0d8";
      ctx.lineWidth = 2;
      const left = 58;
      const bottom = height - 54;
      const chartWidth = width - 96;
      const chartHeight = height - 92;
      ctx.strokeRect(left, bottom - chartHeight, chartWidth, chartHeight);
      const barGap = 12;
      const barWidth = (chartWidth - barGap * (values.length + 1)) / values.length;
      values.forEach((raw, i) => {{
        const value = Math.max(-1, Math.min(1, raw));
        const normalized = (value + 1) / 2;
        const h = Math.max(4, normalized * chartHeight);
        const x = left + barGap + i * (barWidth + barGap);
        const y = bottom - h;
        ctx.fillStyle = palette[i % palette.length];
        ctx.fillRect(x, y, barWidth, h);
        ctx.fillStyle = "#44515c";
        ctx.font = "22px system-ui";
        ctx.textAlign = "center";
        ctx.fillText(String(i + 1), x + barWidth / 2, bottom + 32);
      }});
      ctx.fillStyle = "#5d6670";
      ctx.font = "20px system-ui";
      ctx.textAlign = "left";
      ctx.fillText("-1", 14, bottom + 4);
      ctx.fillText("+1", 14, bottom - chartHeight + 8);
    }}

    function updateRenderImage(frame) {{
      const image = document.getElementById("renderImage");
      const missing = document.getElementById("renderMissing");
      const status = document.getElementById("renderStatus");
      const framePath = `render/scene/frame_${{String(frame.frame_index).padStart(6, "0")}}.bmp`;
      image.style.display = "block";
      missing.style.display = "none";
      status.textContent = "loading";
      image.onload = () => {{ status.textContent = "real frame"; }};
      image.onerror = () => {{
        image.style.display = "none";
        missing.style.display = "flex";
        status.textContent = "not generated";
      }};
      image.src = framePath;
    }}

    function render(nextIndex) {{
      index = Math.max(0, Math.min(frames.length - 1, nextIndex));
      const frame = frames[index];
      slider.value = index;
      document.getElementById("frameMeta").textContent =
        `frame ${{frame.frame_index}} / ${{frames.length - 1}} | ` +
        `t=${{frame.timestamp.toFixed(3)}}s | ` +
        `${{frame.source.gazebo_world || "synthetic"}}`;
      updateRenderImage(frame);
      drawBars(
        document.getElementById("topCanvas"),
        frame.observation_state,
        ["#176b87", "#2f8f70", "#6f8f2f"]
      );
      drawBars(
        document.getElementById("wristCanvas"),
        frame.action,
        ["#b9472f", "#bf7a30", "#8d5fa6"]
      );
      const rows = joints.map((joint, i) => {{
        const state = frame.observation_state[i];
        const action = frame.action[i];
        return `<tr><td>${{joint}}</td><td>${{state.toFixed(3)}}</td>` +
          `<td>${{action.toFixed(3)}}</td>` +
          `<td>${{(action - state).toFixed(3)}}</td></tr>`;
      }}).join("");
      document.getElementById("jointRows").innerHTML = rows;
    }}

    function stop() {{
      if (timer) clearInterval(timer);
      timer = null;
      playBtn.textContent = "Play";
    }}

    playBtn.addEventListener("click", () => {{
      if (timer) {{
        stop();
        return;
      }}
      playBtn.textContent = "Pause";
      timer = setInterval(() => {{
        const next = index + 1 >= frames.length ? 0 : index + 1;
        render(next);
      }}, Math.max(80, 1000 / Number(payload.info.fps || 12)));
    }});
    document.getElementById("prevBtn").addEventListener("click", () => {{
      stop();
      render(index - 1);
    }});
    document.getElementById("nextBtn").addEventListener("click", () => {{
      stop();
      render(index + 1);
    }});
    slider.addEventListener("input", () => {{ stop(); render(Number(slider.value)); }});
    render(0);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
