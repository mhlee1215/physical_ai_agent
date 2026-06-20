#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


MEDIA_JOB_STATUS_FILENAME = ".generate_media_status.json"


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the loop-test analyzer UI.")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    media_job_lock = threading.Lock()
    media_job: dict[str, Any] = _load_media_job_status(export_dir)
    if media_job.get("status") == "running":
        media_job["status"] = "interrupted"
        media_job["finished_at"] = time.time()
        media_job["stderr"] = "Previous media generation server stopped before recording completion."
        _save_media_job_status(export_dir, media_job)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_html(_index_html())
                return
            if parsed.path == "/api/loop-tests":
                self._send_json(_loop_tests_payload(export_dir))
                return
            if parsed.path == "/api/loop-test":
                query = parse_qs(parsed.query)
                loop_test_id = (query.get("id") or [""])[0]
                self._send_json(_loop_test_detail(export_dir, loop_test_id))
                return
            if parsed.path == "/api/generate-media-status":
                with media_job_lock:
                    if media_job.get("status") == "idle":
                        media_job.update(_load_media_job_status(export_dir))
                    self._send_json(_with_media_progress(export_dir, dict(media_job)))
                return
            if parsed.path == "/artifact":
                query = parse_qs(parsed.query)
                self._send_artifact(export_dir, Path((query.get("path") or [""])[0]))
                return
            if parsed.path == "/vendor/chart.umd.min.js":
                self._send_vendor_file(Path(__file__).resolve().parents[1] / "third_party" / "chartjs" / "chart.umd.min.js")
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/generate-media":
                self._start_generate_media_job()
                return
            self.send_error(404)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[loop-test-analyzer] {self.address_string()} {fmt % args}", flush=True)

        def _send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_artifact(self, export_dir: Path, requested_path: Path) -> None:
            try:
                artifact_path = requested_path.resolve()
                artifact_path.relative_to(export_dir)
            except (OSError, ValueError):
                self.send_error(403)
                return
            if not artifact_path.is_file():
                self.send_error(404)
                return
            data = artifact_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(artifact_path))[0] or "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _start_generate_media_job(self) -> None:
            with media_job_lock:
                if media_job.get("status") == "running":
                    self._send_json(dict(media_job))
                    return
                generation_root = _media_generation_repo_root(repo_root, export_dir)
                command = _media_generation_command(generation_root, export_dir)
                media_job.clear()
                media_job.update(
                    {
                        "status": "running",
                        "started_at": time.time(),
                        "finished_at": None,
                        "command": command,
                        "repo_root": str(generation_root),
                        "run_dir": str(export_dir.parent),
                        "output_dir": str(export_dir),
                        "server_pid": os.getpid(),
                        "progress": _media_artifact_progress(export_dir),
                        "stdout": "",
                        "stderr": "",
                    }
                )
                _save_media_job_status(export_dir, media_job)

            def run_job() -> None:
                env = os.environ.copy()
                src_path = str(generation_root / "src")
                env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
                try:
                    completed = subprocess.run(
                        command,
                        cwd=generation_root,
                        env=env,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    status = "succeeded" if completed.returncode == 0 else "failed"
                    result = {
                        "status": status,
                        "finished_at": time.time(),
                        "returncode": completed.returncode,
                        "stdout": completed.stdout[-4000:],
                        "stderr": completed.stderr[-4000:],
                        "progress": _media_artifact_progress(export_dir),
                    }
                except OSError as exc:
                    result = {
                        "status": "failed",
                        "finished_at": time.time(),
                        "returncode": None,
                        "stdout": "",
                        "stderr": str(exc),
                        "progress": _media_artifact_progress(export_dir),
                    }
                with media_job_lock:
                    media_job.update(result)
                    _save_media_job_status(export_dir, media_job)

            threading.Thread(target=run_job, name="loop-test-generate-media", daemon=True).start()
            with media_job_lock:
                self._send_json(_with_media_progress(export_dir, dict(media_job)))

        def _send_vendor_file(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ReusableThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[loop-test-analyzer] serving http://{args.host}:{args.port}/ export_dir={export_dir}", flush=True)
    server.serve_forever()


def _media_generation_repo_root(server_repo_root: Path, export_dir: Path) -> Path:
    for candidate in (export_dir, *export_dir.parents):
        if (candidate / "scripts" / "build_loop_test_analyzer_export.py").is_file() and (candidate / "src").is_dir():
            return candidate
    return server_repo_root


def _media_generation_command(repo_root: Path, export_dir: Path, python_executable: str | None = None) -> list[str]:
    executable = _media_generation_python(repo_root, export_dir, python_executable=python_executable)
    return [
        executable,
        "-B",
        str(repo_root / "scripts" / "build_loop_test_analyzer_export.py"),
        "--run-dir",
        str(export_dir.parent),
        "--output-dir",
        str(export_dir),
        "--copy-source",
        "--generate-media",
    ]


def _media_generation_python(repo_root: Path, export_dir: Path, python_executable: str | None = None) -> str:
    for candidate in (export_dir, *export_dir.parents, repo_root):
        python_path = candidate / ".venv" / "bin" / "python"
        if python_path.exists():
            return str(python_path)
    return python_executable or sys.executable


def _media_job_status_path(export_dir: Path) -> Path:
    return export_dir / MEDIA_JOB_STATUS_FILENAME


def _load_media_job_status(export_dir: Path) -> dict[str, Any]:
    path = _media_job_status_path(export_dir)
    if not path.is_file():
        return {"status": "idle"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "idle"}
    if not isinstance(payload, dict):
        return {"status": "idle"}
    return payload


def _save_media_job_status(export_dir: Path, payload: dict[str, Any]) -> None:
    path = _media_job_status_path(export_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _with_media_progress(export_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") == "running":
        payload["progress"] = _media_artifact_progress(export_dir)
        _save_media_job_status(export_dir, payload)
    elif payload.get("progress") is None:
        payload["progress"] = _media_artifact_progress(export_dir)
    return payload


def _media_artifact_progress(export_dir: Path) -> dict[str, Any]:
    run_dir = export_dir.parent
    total_records = _source_rollout_record_count(run_dir)
    expected_png_files = total_records * 3 if total_records else 0
    media_root = export_dir / "loop_tests"
    png_files = _count_files(media_root, "*.png")
    gif_files = _count_files(media_root, "*.gif")
    mp4_files = _count_files(media_root, "*.mp4")
    expected = expected_png_files or png_files
    percent = None
    if expected:
        percent = min(100.0, round((png_files / expected) * 100.0, 1))
    stage = "waiting"
    if png_files:
        stage = "rendering frames"
    if gif_files or mp4_files:
        stage = "encoding videos"
    if percent == 100.0:
        stage = "finalizing export"
    return {
        "stage": stage,
        "percent": percent,
        "source_rollout_records": total_records,
        "expected_png_files": expected_png_files,
        "png_files": png_files,
        "gif_files": gif_files,
        "mp4_files": mp4_files,
    }


def _source_rollout_record_count(run_dir: Path) -> int:
    total = 0
    for report_path in sorted((run_dir / "closed_loop_evals").glob("**/*_report.json")):
        report = _read_json_safely(report_path)
        for episode in report.get("episodes") or []:
            trace_path = Path(episode.get("trace_path") or "")
            if not trace_path.is_file():
                trace_path = report_path.parent / trace_path.name
            total += _count_jsonl_records(trace_path)
    return total


def _read_json_safely(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _count_jsonl_records(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob(pattern) if path.is_file())


def _loop_tests_payload(export_dir: Path) -> dict[str, Any]:
    manifest = _read_json(export_dir / "manifest.json")
    return {
        "export_dir": str(export_dir),
        "schema_version": manifest.get("schema_version"),
        "summary": manifest.get("summary") or {},
        "loop_tests": manifest.get("loop_tests") or [],
    }


def _loop_test_detail(export_dir: Path, loop_test_id: str) -> dict[str, Any]:
    manifest = _read_json(export_dir / "manifest.json")
    row = next((item for item in manifest.get("loop_tests") or [] if item.get("loop_test_id") == loop_test_id), None)
    if row is None:
        return {"error": f"loop test not found: {loop_test_id}"}
    detail = _read_json(Path(row["manifest_path"]))
    episodes = []
    for episode in detail.get("episodes") or []:
        timeline_path = Path(episode["timeline_path"])
        episodes.append({**episode, "timeline": _read_jsonl(timeline_path)})
    return {"loop_test": detail, "episodes": episodes}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Loop Test Analyzer</title>
  <style>
    :root {
      color-scheme: light;
      --border:#d8dde7; --muted:#667085; --bg:#f5f7fb; --ink:#162033;
      --accent:#0f766e; --accent-soft:#d9f3ef; --policy:#5b4bdb; --policy-soft:#eceafe;
      --robot:#047857; --robot-soft:#dcfce7; --warn:#b45309; --warn-soft:#fff7ed;
      --fail:#b42318; --fail-soft:#fee4e2; --ok:#027a48; --ok-soft:#dcfae6;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; overflow: hidden; }
    body { margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    .app { display: grid; grid-template-columns: 330px minmax(0, 1fr); height: 100vh; overflow: hidden; }
    aside { border-right: 1px solid var(--border); background: #fff; padding: 14px; overflow: auto; min-height: 0; }
    main { overflow: auto; min-height: 0; padding: 16px 18px 32px; }
    h1 { font-size: 18px; margin: 0 0 12px; }
    h2 { font-size: 16px; margin: 0; }
    .summary, .toolbar { display: grid; gap: 8px; margin-bottom: 12px; }
    input, select { width: 100%; border: 1px solid var(--border); border-radius: 6px; padding: 8px; background: #fff; }
    button { border:1px solid var(--accent); border-radius:6px; background:var(--accent); color:#fff; font-weight:650; padding:7px 10px; cursor:pointer; }
    button:disabled { opacity:.62; cursor:not-allowed; }
    .test { border: 1px solid var(--border); border-radius: 6px; padding: 10px; margin: 8px 0; cursor: pointer; background: #fff; transition: border-color .12s, background .12s; }
    .test:hover { border-color:#9aa8bd; background:#fbfdff; }
    .test.active { border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }
    .row { display:flex; justify-content:space-between; gap:8px; }
    .muted { color: var(--muted); }
    .pill { display:inline-block; padding:2px 7px; border:1px solid var(--border); border-radius:999px; font-size:12px; background:#fff; }
    .pill.policy { color:#3f32b8; border-color:#c7c2fb; background:var(--policy-soft); }
    .pill.robot { color:#067647; border-color:#abefc6; background:var(--robot-soft); }
    .pill.fail { color:var(--fail); border-color:#fecdca; background:var(--fail-soft); }
    .pill.warn { color:var(--warn); border-color:#fedf89; background:var(--warn-soft); }
    .header { display:grid; gap:8px; margin-bottom:14px; }
    .metrics { display:flex; gap:8px; flex-wrap:wrap; }
    .media-actions { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .media-actions .status { color:var(--muted); }
    .media-progress { display:grid; gap:5px; min-width:min(420px, 100%); flex:1 1 280px; }
    .progress-track { width:100%; height:8px; border-radius:999px; background:#e9edf5; overflow:hidden; border:1px solid #dfe5ef; }
    .progress-bar { height:100%; width:0%; background:linear-gradient(90deg, var(--accent), #22c55e); transition:width .25s ease; }
    .progress-meta { display:flex; gap:8px; flex-wrap:wrap; color:var(--muted); font-size:12px; }
    .plan { background:#fff; border:1px solid var(--border); border-radius:6px; padding:10px; margin-bottom:14px; border-left:4px solid var(--policy); }
    .timeline { display:grid; gap:10px; }
    .event { display:grid; grid-template-columns:minmax(280px, 0.95fr) minmax(300px, 1.05fr); gap:10px; align-items:stretch; }
    .panel { background:#fff; border:1px solid var(--border); border-radius:6px; padding:10px; min-width:0; }
    .panel.policy { border-left:4px solid var(--policy); background:linear-gradient(90deg, var(--policy-soft), #fff 26%); }
    .panel.robot { border-left:4px solid var(--robot); background:linear-gradient(90deg, var(--robot-soft), #fff 26%); }
    .panel h3 { font-size:13px; margin:0 0 8px; }
    .iteration { margin:16px 0 8px; font-weight:700; color:#344054; display:flex; align-items:center; gap:8px; }
    .iteration::before { content:""; width:10px; height:10px; border-radius:50%; background:var(--accent); display:inline-block; }
    details { margin-top:8px; border:1px solid #edf0f5; border-radius:6px; background:#fff; }
    summary { cursor:pointer; padding:8px 9px; font-weight:650; color:#344054; }
    details > pre, details > .detail-body { margin:0 8px 8px; }
    .tool-call { border:1px solid #d9d6fe; background:#fafaff; border-radius:6px; padding:8px; margin-top:8px; }
    .step-list { display:grid; gap:7px; padding:0 8px 8px; }
    .step-card { border:1px solid #edf0f5; border-radius:5px; padding:7px; background:#fbfcfe; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; }
    pre { white-space: pre-wrap; word-break: break-word; max-height: 230px; overflow:auto; background:#f8fafc; border:1px solid #edf0f5; border-radius:5px; padding:8px; margin:6px 0 0; }
    .placeholder { height:112px; display:grid; place-items:center; border:1px dashed var(--border); border-radius:6px; color:var(--muted); background:#fbfcfe; text-align:center; padding:10px; }
    .thumb-row { display:grid; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); gap:8px; margin:8px 0; }
    .thumb { border:1px solid var(--border); border-radius:6px; overflow:hidden; background:#fff; }
    .thumb img { width:100%; display:block; aspect-ratio:4/3; object-fit:cover; }
    .thumb .label { padding:5px 7px; color:var(--muted); font-size:12px; }
    .video-link { display:inline-block; margin-top:8px; color:var(--accent); font-weight:650; }
    .diagnostics { background:#fff; border:1px solid var(--border); border-left:4px solid var(--warn); border-radius:6px; padding:10px; margin-bottom:14px; }
    .diagnostics-head { display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap; }
    .metric-help { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:8px; padding:0 8px 8px; }
    .metric-help div { border:1px solid #edf0f5; border-radius:6px; padding:8px; background:#fbfcfe; }
    .chart-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:10px; margin-top:8px; }
    .chart { border:1px solid #edf0f5; border-radius:6px; padding:8px; background:#fbfcfe; }
    .chart-head { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:6px; }
    .chart-head select { width:auto; min-width:86px; padding:4px 7px; }
    .chart-canvas { height:210px; }
    .chart canvas { width:100%; height:100%; display:block; }
    .chart-lib-warning { display:none; margin-top:8px; color:var(--fail); }
    .chart-lib-warning.show { display:block; }
    @media (max-width: 850px) { .event { grid-template-columns:1fr; } }
    @media (max-width: 640px) { .app { grid-template-columns: 1fr; grid-template-rows:minmax(160px, 42vh) minmax(0, 1fr); } aside { border-right:0; border-bottom:1px solid var(--border); } }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Loop Test Analyzer</h1>
      <div class="toolbar">
        <input id="filter" placeholder="filter checkpoint / policy / scenario">
        <select id="statusFilter">
          <option value="">all outcomes</option>
          <option value="success">task success</option>
          <option value="fail">task fail</option>
        </select>
      </div>
      <div id="summary" class="summary muted">loading...</div>
      <div id="tests"></div>
    </aside>
    <main>
      <div id="detail" class="muted">Select a loop test.</div>
    </main>
  </div>
  <script src="/vendor/chart.umd.min.js"></script>
  <script>
    let state = { tests: [], active: null, episode: 0, currentEpisode: null };
    let diagnosticCharts = [];
    const el = id => document.getElementById(id);
    const fmt = value => value === null || value === undefined ? "-" : (typeof value === "number" ? value.toFixed(4) : String(value));
    async function loadList() {
      const data = await (await fetch("/api/loop-tests")).json();
      state.tests = data.loop_tests || [];
      el("summary").innerHTML = `<div>${data.summary.loop_tests || 0} loop tests</div><div>latest: ${data.summary.latest_checkpoint || "-"}</div>`;
      renderList();
      const params = new URLSearchParams(location.search);
      const requested = params.get("loop");
      const episode = Number(params.get("episode") || 0);
      const fallback = state.tests.length ? state.tests[state.tests.length - 1].loop_test_id : null;
      if (requested || fallback) selectTest(requested || fallback, { episode, replace: true });
    }
    function renderList() {
      const q = el("filter").value.toLowerCase();
      const status = el("statusFilter").value;
      const rows = state.tests.filter(row => {
        const text = `${row.checkpoint} ${row.policy_type} ${row.scenario}`.toLowerCase();
        const statusOk = !status || (status === "success" ? row.success_rate > 0 : !(row.success_rate > 0));
        return text.includes(q) && statusOk;
      });
      el("tests").innerHTML = rows.map(row => `
        <div class="test ${row.loop_test_id === state.active ? "active" : ""}" onclick="selectTest('${row.loop_test_id}')">
          <div class="row"><strong>${row.checkpoint}</strong><span class="pill policy">${row.policy_type}</span></div>
          <div class="muted">step ${row.training_step ?? "-"} · val ${fmt(row.validation_loss)}</div>
          <div><span class="pill ${row.success_rate > 0 ? "robot" : "fail"}">success ${fmt(row.success_rate)}</span> <span class="muted">${row.status}</span></div>
        </div>`).join("");
    }
    async function selectTest(id, options = {}) {
      state.active = id;
      state.episode = Number(options.episode || 0);
      const nextUrl = `?loop=${encodeURIComponent(id)}&episode=${encodeURIComponent(state.episode)}`;
      if (options.replace) history.replaceState({ loop: id, episode: state.episode }, "", nextUrl);
      else history.pushState({ loop: id, episode: state.episode }, "", nextUrl);
      renderList();
      const data = await (await fetch(`/api/loop-test?id=${encodeURIComponent(id)}`)).json();
      renderDetail(data);
    }
    function renderDetail(data) {
      if (data.error) { el("detail").textContent = data.error; return; }
      const lt = data.loop_test;
      const episodes = data.episodes || [];
      const ep = episodes[Math.min(state.episode, Math.max(0, episodes.length - 1))] || { timeline: [] };
      el("detail").innerHTML = `
        <div class="header">
          <h2>${lt.policy_label} · ${lt.checkpoint}</h2>
          <div class="metrics">
            <span class="pill policy">scenario ${lt.scenario}</span>
            <span class="pill">step ${lt.training_step}</span>
            <span class="pill">val ${fmt(lt.validation_loss)}</span>
            <span class="pill ${lt.success_rate > 0 ? "robot" : "fail"}">task success ${fmt(lt.success_rate)}</span>
            <span class="pill warn">status means ${lt.status_meaning}</span>
          </div>
          <div class="media-actions">
            <button id="generateMediaBtn" onclick="generateMedia()">Generate / refresh media</button>
            <div class="media-progress">
              <span id="mediaJobStatus" class="status">frames and videos are generated locally on demand</span>
              <div class="progress-track"><div id="mediaJobProgressBar" class="progress-bar"></div></div>
              <div id="mediaJobProgressMeta" class="progress-meta"></div>
            </div>
          </div>
        </div>
          ${renderPlan(lt)}
        ${renderDiagnostics(ep)}
        <div class="timeline">${renderTimeline(groupTimeline(ep.timeline || []))}</div>`;
      state.currentEpisode = ep;
      renderDiagnosticCharts(ep);
      refreshMediaJobStatus();
    }
    function renderDiagnostics(ep) {
      const rows = (ep.timeline || []).filter(row => row.type === "policy_step");
      if (!rows.length) return "";
      const actionDimCount = rows.reduce((max, row) => Math.max(max, Array.isArray(row.policy_output?.action) ? row.policy_output.action.length : 0), 0);
      const finalInfo = ep.final_info || {};
      const heuristics = [];
      if (ep.final_success === false) heuristics.push("task did not reach success");
      if (typeof finalInfo.tcp_to_target_dist === "number" && finalInfo.tcp_to_target_dist > 0.08) heuristics.push(`final distance high: ${finalInfo.tcp_to_target_dist.toFixed(3)}`);
      const endedByMaxSteps = rows.length === Number(ep.steps || rows.length);
      if (endedByMaxSteps && ep.final_success === false) heuristics.push("rollout consumed full planned horizon");
      return `<section class="diagnostics">
        <div class="diagnostics-head">
          <strong>Diagnostics</strong>
          <span class="pill">charts: Chart.js</span>
        </div>
        <div class="metrics">${heuristics.map(item => `<span class="pill warn">${escapeHtml(item)}</span>`).join("") || `<span class="pill robot">no obvious failure heuristic</span>`}</div>
        <details open>
          <summary>Metrics explained</summary>
          <div class="metric-help">
            <div><strong>Reward</strong><br><span class="muted">Simulator reward returned after each executed env step. It is a progress signal, not the same thing as final task success.</span></div>
            <div><strong>TCP distance</strong><br><span class="muted"><code>info.tcp_to_target_dist</code>: distance from the robot tool center point / gripper to the target. Lower is better for pick-up style tasks.</span></div>
            <div><strong>Action dimension</strong><br><span class="muted">One selected component of the executed SmolVLA action vector. <code>action[0]</code> is only the first joint/control target, useful for spotting jitter or saturation, not a full policy score.</span></div>
          </div>
        </details>
        <div id="chartLibWarning" class="chart-lib-warning">Chart.js did not load, so interactive charts are unavailable. The raw values are still available in the rollout timeline below.</div>
        <div class="chart-grid">
          <div class="chart">
            <div class="chart-head"><strong>Reward</strong><span id="rewardLast" class="muted">-</span></div>
            <div class="chart-canvas"><canvas id="rewardChart"></canvas></div>
          </div>
          <div class="chart">
            <div class="chart-head"><strong>TCP distance</strong><span id="tcpDistanceLast" class="muted">-</span></div>
            <div class="chart-canvas"><canvas id="tcpDistanceChart"></canvas></div>
          </div>
          <div class="chart">
            <div class="chart-head">
              <strong>Action dimension</strong>
              <select id="diagActionDim" aria-label="action dimension">
                ${Array.from({ length: Math.max(actionDimCount, 1) }, (_, index) => `<option value="${index}">action[${index}]</option>`).join("")}
              </select>
              <span id="actionLast" class="muted">-</span>
            </div>
            <div class="chart-canvas"><canvas id="actionChart"></canvas></div>
          </div>
        </div>
      </section>`;
    }
    function renderDiagnosticCharts(ep) {
      destroyDiagnosticCharts();
      const rows = (ep.timeline || []).filter(row => row.type === "policy_step");
      const warning = el("chartLibWarning");
      if (!rows.length || !warning) return;
      if (!window.Chart) {
        warning.classList.add("show");
        return;
      }
      warning.classList.remove("show");
      const actionSelect = el("diagActionDim");
      const selectedActionDim = Number(actionSelect?.value || 0);
      const rewardSeries = rows.map(row => ({ x: Number(row.global_step ?? row.primitive_step ?? 0), y: Number(row.robot?.reward ?? 0) }));
      const distanceSeries = rows
        .filter(row => typeof row.robot?.info?.tcp_to_target_dist === "number")
        .map(row => ({ x: Number(row.global_step ?? row.primitive_step ?? 0), y: row.robot.info.tcp_to_target_dist }));
      const actionSeries = rows
        .filter(row => typeof row.policy_output?.action?.[selectedActionDim] === "number")
        .map(row => ({ x: Number(row.global_step ?? row.primitive_step ?? 0), y: row.policy_output.action[selectedActionDim] }));
      el("rewardLast").textContent = rewardSeries.length ? fmt(rewardSeries[rewardSeries.length - 1].y) : "-";
      el("tcpDistanceLast").textContent = distanceSeries.length ? fmt(distanceSeries[distanceSeries.length - 1].y) : "-";
      el("actionLast").textContent = actionSeries.length ? fmt(actionSeries[actionSeries.length - 1].y) : "-";
      actionSelect?.addEventListener("change", () => renderDiagnosticCharts(state.currentEpisode || ep), { once: true });
      diagnosticCharts.push(
        makeLineChart("rewardChart", "reward", rewardSeries, "#047857", "reward"),
        makeLineChart("tcpDistanceChart", "tcp distance", distanceSeries, "#b45309", "distance"),
        makeLineChart("actionChart", `action[${selectedActionDim}]`, actionSeries, "#5b4bdb", "action value"),
      );
    }
    function makeLineChart(canvasId, label, data, color, yTitle) {
      const canvas = el(canvasId);
      if (!canvas) return null;
      return new Chart(canvas, {
        type: "line",
        data: { datasets: [{ label, data, borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 0, pointHoverRadius: 3, tension: 0.18 }] },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          parsing: false,
          interaction: { intersect: false, mode: "nearest" },
          plugins: { legend: { display: false }, tooltip: { callbacks: { label: context => `${label}: ${fmt(context.parsed.y)}` } } },
          scales: {
            x: { type: "linear", title: { display: true, text: "global step" }, grid: { color: "#edf0f5" } },
            y: { title: { display: true, text: yTitle }, grid: { color: "#edf0f5" } }
          }
        }
      });
    }
    function destroyDiagnosticCharts() {
      for (const chart of diagnosticCharts) if (chart) chart.destroy();
      diagnosticCharts = [];
    }
    function renderPlan(lt) {
      const plan = lt.qwen_plan;
      if (!plan) return "";
      return `<div class="plan">
        <strong>Planner</strong>
        <div class="muted">${plan.model || "-"} · ${plan.task || "-"}</div>
        <details>
          <summary>System prompt</summary>
          <pre>${escapeHtml(lt.qwen_prompts?.system || "not recorded")}</pre>
        </details>
        <details open>
          <summary>Tool calls: function name + parameters</summary>
          <div class="detail-body">${(plan.calls || []).map(call => renderToolCall(call)).join("")}</div>
        </details>
        ${renderQwenRawLinks(lt.qwen_raw || {})}
      </div>`;
    }
    function renderQwenRawLinks(raw) {
      const links = [
        ["Raw request", raw.request_path],
        ["Raw response", raw.response_path],
        ["Parsed plan", raw.plan_path]
      ].filter(([, path]) => path);
      if (!links.length) return "";
      return `<details><summary>Raw Qwen payloads</summary><div class="detail-body">${links.map(([label, path]) => `<a class="video-link" href="${artifactUrl(path)}" target="_blank">${escapeHtml(label)}</a>`).join("<br>")}</div></details>`;
    }
    async function generateMedia() {
      const button = el("generateMediaBtn");
      if (button) button.disabled = true;
      setMediaJobStatus("starting media export...");
      try {
        const data = await (await fetch("/api/generate-media", { method: "POST" })).json();
        updateMediaJobStatus(data);
        pollMediaJob();
      } catch (error) {
        setMediaJobStatus(`media export request failed: ${error}`);
        if (button) button.disabled = false;
      }
    }
    async function pollMediaJob() {
      const data = await (await fetch("/api/generate-media-status")).json();
      updateMediaJobStatus(data);
      if (data.status === "running") {
        setTimeout(pollMediaJob, 1500);
        return;
      }
      const button = el("generateMediaBtn");
      if (button) button.disabled = false;
      if (data.status === "succeeded" && state.active) {
        await selectTest(state.active, { episode: state.episode, replace: true });
        setMediaJobStatus("media export complete; current loop reloaded");
      }
    }
    async function refreshMediaJobStatus() {
      try {
        const data = await (await fetch("/api/generate-media-status")).json();
        updateMediaJobStatus(data);
        if (data.status === "running") pollMediaJob();
      } catch (error) {
        setMediaJobStatus(`media status unavailable: ${error}`);
      }
    }
    function updateMediaJobStatus(data) {
      updateMediaProgress(data?.progress || null);
      const button = el("generateMediaBtn");
      if (button) button.disabled = data?.status === "running";
      if (!data || data.status === "idle") {
        if (typeof data?.progress?.percent === "number" && data.progress.percent >= 100) {
          setMediaJobStatus("media already available; generate again to refresh");
        } else {
          setMediaJobStatus("frames and videos are generated locally on demand");
        }
        return;
      }
      if (data.status === "running") {
        const stage = data.progress?.stage || "generating media";
        const percent = typeof data.progress?.percent === "number" ? ` ${data.progress.percent.toFixed(1)}%` : "";
        setMediaJobStatus(`${stage}${percent}`);
        return;
      }
      if (data.status === "succeeded") {
        setMediaJobStatus("media export complete");
        return;
      }
      if (data.status === "interrupted") {
        setMediaJobStatus(`media export interrupted: ${data.stderr || "server stopped before completion"}`);
        return;
      }
      setMediaJobStatus(`media export failed: ${data.stderr || data.returncode || "unknown error"}`);
    }
    function updateMediaProgress(progress) {
      const bar = el("mediaJobProgressBar");
      const meta = el("mediaJobProgressMeta");
      if (!bar || !meta) return;
      const percent = typeof progress?.percent === "number" ? progress.percent : 0;
      bar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
      if (!progress) {
        meta.textContent = "";
        return;
      }
      const expected = progress.expected_png_files || "-";
      meta.innerHTML = `
        <span>frames ${progress.png_files ?? 0}/${expected}</span>
        <span>gif ${progress.gif_files ?? 0}</span>
        <span>mp4 ${progress.mp4_files ?? 0}</span>
        <span>records ${progress.source_rollout_records ?? "-"}</span>`;
    }
    function setMediaJobStatus(text) {
      const node = el("mediaJobStatus");
      if (node) node.textContent = text;
    }
    function renderToolCall(call) {
      return `<div class="tool-call">
        <div><strong>${escapeHtml(call.fn || call.function || "-")}</strong></div>
        <pre>${escapeHtml(JSON.stringify({
          object: call.object ?? call.parameters?.object,
          primitive_id: call.primitive_id ?? call.parameters?.primitive_id,
          prompt: call.prompt ?? call.parameters?.prompt,
          max_steps: call.max_steps ?? call.parameters?.max_steps
        }, null, 2))}</pre>
      </div>`;
    }
    function groupTimeline(rows) {
      const groups = [];
      let current = null;
      for (const row of rows) {
        if (row.type === "planner_call") groups.push({ kind: "planner", row });
        else if (row.type === "tool_call_start") {
          current = { kind: "iteration", start: row, steps: [], end: null };
          groups.push(current);
        } else if (row.type === "policy_step" && current) current.steps.push(row);
        else if (row.type === "tool_call_end" && current) {
          current.end = row;
          current = null;
        } else if (row.type === "episode_end") groups.push({ kind: "episode_end", row });
      }
      return groups;
    }
    function renderTimeline(groups) {
      return groups.map(group => {
        if (group.kind === "planner") return `<div class="iteration">Planner</div>${renderEvent(group.row)}`;
        if (group.kind === "episode_end") return `<div class="iteration">Episode end</div>${renderEvent(group.row)}`;
        return renderIteration(group);
      }).join("");
    }
    function renderIteration(group) {
      const start = group.start || {};
      const steps = group.steps || [];
      const rewards = steps.map(row => row.robot?.reward).filter(value => typeof value === "number");
      const totalReward = rewards.reduce((sum, value) => sum + value, 0);
      const last = steps[steps.length - 1] || {};
      const firstMedia = steps.find(row => row.media?.available)?.media || {};
      const lastMedia = [...steps].reverse().find(row => row.media?.available)?.media || firstMedia;
      const contract = start.policy_output?.action_chunk_contract || steps[0]?.policy_output?.action_chunk || {};
      const rolloutConfig = start.policy_output?.rollout_config || firstRawRolloutConfig(steps) || {};
      const generated = rolloutConfig.chunk_size ?? contract.generated_count ?? "unknown";
      const usedPerChunk = rolloutConfig.n_action_steps ?? contract.used_per_chunk ?? "unknown";
      const chunks = groupActionChunks(steps);
      const confirmed = Boolean(contract.confirmed_in_rollout);
      const groupLabel = chunks.length === 1 ? "display group" : "display groups";
      return `
        <div class="iteration">Iteration ${start.iteration}: ${escapeHtml(start.tool_call || "-")}</div>
        <div class="event">
          <section class="panel policy">
            <h3>Policy tool call</h3>
            <div class="metrics">
              <span class="pill policy">function ${escapeHtml(start.tool_call || "-")}</span>
              <span class="pill">primitive ${escapeHtml(start.primitive_id || "-")}</span>
              <span class="pill ${confirmed ? "robot" : "warn"}">rollout n_action_steps ${usedPerChunk}</span>
              <span class="pill warn">rollout chunk_size ${generated}</span>
              <span class="pill">source ${escapeHtml(rolloutConfig.source || contract.rollout_config_source || "raw_record")}</span>
              <span class="pill robot">${steps.length} raw rollout records</span>
              <span class="pill">${chunks.length} ${groupLabel}</span>
            </div>
            <div class="muted">${escapeHtml(contract.note || "")}</div>
            ${renderPolicyInputImages(firstMedia.policy_input_images || {})}
            <div class="tool-call">
              <strong>function + parameters</strong>
              <pre>${escapeHtml(JSON.stringify({ function: start.tool_call, parameters: start.tool_parameters || {} }, null, 2))}</pre>
            </div>
            ${start.policy_input?.prompt ? `<div><strong>Prompt</strong><br>${escapeHtml(start.policy_input.prompt)}</div>` : ""}
            <details>
              <summary>Executed rollout records, grouped for display (${chunks.length})</summary>
              <div class="step-list">${chunks.map(renderActionChunk).join("")}</div>
            </details>
            <details open>
              <summary>Raw rollout config evidence</summary>
              <pre>${escapeHtml(JSON.stringify({
                tool_call_start_rollout_config: rolloutConfig,
                first_step_raw_policy_rollout_config: firstRawRolloutConfig(steps),
                analyzer_contract: contract,
                note: "Rows below preserve each raw rollout record under source.record."
              }, null, 2))}</pre>
            </details>
          </section>
          <section class="panel robot">
            <h3>Robot motion</h3>
            ${renderRobotMedia(lastMedia)}
            <div class="metrics">
              <span class="pill robot">total reward ${fmt(totalReward)}</span>
              <span class="pill">last global ${last.global_step ?? "-"}</span>
              <span class="pill ${last.robot?.info?.success ? "robot" : "fail"}">success ${String(Boolean(last.robot?.info?.success))}</span>
            </div>
            <details>
              <summary>Last robot state</summary>
              <pre>${escapeHtml(JSON.stringify(last.robot?.info || group.end?.robot?.last_info || {}, null, 2))}</pre>
            </details>
          </section>
        </div>
        <div class="iteration">Iteration ${start.iteration} end</div>`;
    }
    function renderPolicyInputImages(images) {
      const entries = Object.entries(images || {});
      if (!entries.length) return `<div class="placeholder">policy input images unavailable</div>`;
      return `<div class="thumb-row">${entries.map(([name, path]) => `
        <div class="thumb"><img src="${artifactUrl(path)}" alt="${escapeHtml(name)}"><div class="label">${escapeHtml(name)}</div></div>
      `).join("")}</div>`;
    }
    function renderRobotMedia(media) {
      if (!media?.robot_frame) return `<div class="placeholder">${media?.reason || "robot frames unavailable"}</div>`;
      const video = media.iteration_video_gif ? `<a class="video-link" href="${artifactUrl(media.iteration_video_gif)}" target="_blank">open iteration video</a>` : "";
      return `<div class="thumb"><img src="${artifactUrl(media.robot_frame)}" alt="robot frame"><div class="label">latest robot frame</div></div>${video}`;
    }
    function firstRawRolloutConfig(steps) {
      for (const row of steps || []) {
        const config = row.source?.record?.policy_rollout_config;
        if (config && typeof config === "object") return config;
      }
      return null;
    }
    function groupActionChunks(steps) {
      const byChunk = new Map();
      for (const row of steps) {
        const chunk = row.policy_output?.action_chunk || {};
        const key = Number.isInteger(chunk.chunk_index) ? chunk.chunk_index : 0;
        if (!byChunk.has(key)) byChunk.set(key, []);
        byChunk.get(key).push(row);
      }
      return Array.from(byChunk.entries()).sort((a, b) => a[0] - b[0]).map(([chunkIndex, rows]) => ({ chunkIndex, rows }));
    }
    function renderActionChunk(chunk) {
      const rows = chunk.rows || [];
      const first = rows[0] || {};
      const last = rows[rows.length - 1] || {};
      const actionChunk = first.policy_output?.action_chunk || {};
      const rawConfig = first.source?.record?.policy_rollout_config || {};
      const expected = rawConfig.n_action_steps ?? actionChunk.used_per_chunk ?? "unknown";
      const generated = rawConfig.chunk_size ?? actionChunk.generated_count ?? "unknown";
      return `<div class="step-card">
        <div class="metrics">
          <span class="pill policy">display group ${chunk.chunkIndex + 1}</span>
          <span class="pill warn">raw chunk_size ${generated}</span>
          <span class="pill robot">raw records ${rows.length} / rollout n_action_steps ${expected}</span>
          <span class="pill">global ${first.global_step ?? "-"}-${last.global_step ?? "-"}</span>
        </div>
        <div class="muted">Grouped from rollout rows for readability; raw records are preserved below.</div>
        <details>
          <summary>Raw rollout records: first + last</summary>
          <pre>${escapeHtml(JSON.stringify({ first: first.source?.record || null, last: last.source?.record || null }, null, 2))}</pre>
        </details>
        <details>
          <summary>Raw env action records (${rows.length})</summary>
          <div class="step-list">${rows.map(renderActionStep).join("")}</div>
        </details>
      </div>`;
    }
    function renderActionStep(row) {
      return `<div class="step-card">
        <div class="metrics">
          <span class="pill">global ${row.global_step ?? "-"}</span>
          <span class="pill">primitive ${row.primitive_step ?? "-"}</span>
          <span class="pill policy">chunk step ${row.policy_output?.action_chunk?.chunk_step_index ?? "-"}</span>
          <span class="pill robot">reward ${fmt(row.robot?.reward)}</span>
        </div>
        <details>
          <summary>action + motor state</summary>
          ${renderPolicyInputImages(row.media?.policy_input_images || {})}
          ${row.media?.robot_frame ? `<div class="thumb"><img src="${artifactUrl(row.media.robot_frame)}" alt="robot frame"><div class="label">robot frame</div></div>` : ""}
          <pre>${escapeHtml(JSON.stringify({ raw_rollout_record: row.source?.record || null, analyzer_row: { action: row.policy_output?.action, observation: row.policy_input?.observation, info: row.robot?.info } }, null, 2))}</pre>
        </details>
      </div>`;
    }
    function renderEvent(row) {
      const policy = row.policy_input || row.policy_output || row.policy ? `
        <h3>Policy · ${row.type}</h3>
        <div class="muted">${row.tool_call || ""} ${row.primitive_id || ""}</div>
        ${row.policy_input?.system_prompt ? `<details><summary>System prompt</summary><pre>${escapeHtml(row.policy_input.system_prompt)}</pre></details>` : ""}
        ${row.policy_input?.prompt ? `<div><strong>Prompt</strong><br>${escapeHtml(row.policy_input.prompt)}</div>` : ""}
        ${row.policy_output?.tool_calls ? `<details open><summary>Tool calls</summary><div class="detail-body">${row.policy_output.tool_calls.map(renderToolCall).join("")}</div></details>` : ""}
        ${row.policy_output?.action ? `<details><summary>Action</summary><pre>${escapeHtml(JSON.stringify(row.policy_output.action))}</pre></details>` : ""}
        ${row.policy_input?.observation ? `<div><strong>Motor state</strong><pre>${escapeHtml(JSON.stringify(row.policy_input.observation))}</pre></div>` : ""}
      ` : `<h3>Policy</h3><div class="muted">No policy event</div>`;
      const robot = row.robot ? `
        <h3>Robot motion</h3>
        <div class="placeholder">${row.media?.reason || "media unavailable"}</div>
        <div class="metrics">
          <span class="pill">global ${row.global_step ?? "-"}</span>
          <span class="pill">primitive ${row.primitive_step ?? "-"}</span>
          <span class="pill">reward ${fmt(row.robot.reward ?? row.robot.last_reward)}</span>
        </div>
        <pre>${escapeHtml(JSON.stringify(row.robot.info || row.robot.last_info || row.robot.final_info || {}, null, 2))}</pre>
      ` : `<h3>Robot motion</h3><div class="placeholder">${row.media?.reason || "waiting for motion data"}</div>`;
      return `<div class="event"><section class="panel policy">${policy}</section><section class="panel robot">${robot}</section></div>`;
    }
    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[ch]));
    }
    function artifactUrl(path) {
      return `/artifact?path=${encodeURIComponent(path)}`;
    }
    el("filter").addEventListener("input", renderList);
    el("statusFilter").addEventListener("change", renderList);
    window.addEventListener("popstate", () => {
      const params = new URLSearchParams(location.search);
      const id = params.get("loop");
      if (id) selectTest(id, { episode: Number(params.get("episode") || 0), replace: true });
    });
    loadList();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
