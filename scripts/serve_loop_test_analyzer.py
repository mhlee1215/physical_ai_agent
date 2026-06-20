#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the loop-test analyzer UI.")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()

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
            if parsed.path == "/artifact":
                query = parse_qs(parsed.query)
                self._send_artifact(export_dir, Path((query.get("path") or [""])[0]))
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

    server = ReusableThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[loop-test-analyzer] serving http://{args.host}:{args.port}/ export_dir={export_dir}", flush=True)
    server.serve_forever()


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
    :root { color-scheme: light; --border:#d8dde7; --muted:#667085; --bg:#f7f8fb; --ink:#162033; --accent:#0f766e; }
    * { box-sizing: border-box; }
    body { margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    .app { display: grid; grid-template-columns: 330px minmax(0, 1fr); min-height: 100vh; }
    aside { border-right: 1px solid var(--border); background: #fff; padding: 14px; overflow: auto; }
    main { overflow: auto; padding: 16px 18px 32px; }
    h1 { font-size: 18px; margin: 0 0 12px; }
    h2 { font-size: 16px; margin: 0; }
    .summary, .toolbar { display: grid; gap: 8px; margin-bottom: 12px; }
    input, select { width: 100%; border: 1px solid var(--border); border-radius: 6px; padding: 8px; background: #fff; }
    .test { border: 1px solid var(--border); border-radius: 6px; padding: 10px; margin: 8px 0; cursor: pointer; background: #fff; }
    .test.active { border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }
    .row { display:flex; justify-content:space-between; gap:8px; }
    .muted { color: var(--muted); }
    .pill { display:inline-block; padding:2px 6px; border:1px solid var(--border); border-radius:999px; font-size:12px; background:#fff; }
    .header { display:grid; gap:8px; margin-bottom:14px; }
    .metrics { display:flex; gap:8px; flex-wrap:wrap; }
    .plan { background:#fff; border:1px solid var(--border); border-radius:6px; padding:10px; margin-bottom:14px; }
    .timeline { display:grid; gap:10px; }
    .event { display:grid; grid-template-columns:minmax(280px, 0.95fr) minmax(300px, 1.05fr); gap:10px; align-items:stretch; }
    .panel { background:#fff; border:1px solid var(--border); border-radius:6px; padding:10px; min-width:0; }
    .panel h3 { font-size:13px; margin:0 0 8px; }
    .iteration { margin:16px 0 8px; font-weight:700; color:#344054; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; }
    pre { white-space: pre-wrap; word-break: break-word; max-height: 230px; overflow:auto; background:#f8fafc; border:1px solid #edf0f5; border-radius:5px; padding:8px; margin:6px 0 0; }
    .placeholder { height:112px; display:grid; place-items:center; border:1px dashed var(--border); border-radius:6px; color:var(--muted); background:#fbfcfe; text-align:center; padding:10px; }
    @media (max-width: 850px) { .app { grid-template-columns: 1fr; } aside { border-right:0; border-bottom:1px solid var(--border); max-height:45vh; } .event { grid-template-columns:1fr; } }
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
  <script>
    let state = { tests: [], active: null };
    const el = id => document.getElementById(id);
    const fmt = value => value === null || value === undefined ? "-" : (typeof value === "number" ? value.toFixed(4) : String(value));
    async function loadList() {
      const data = await (await fetch("/api/loop-tests")).json();
      state.tests = data.loop_tests || [];
      el("summary").innerHTML = `<div>${data.summary.loop_tests || 0} loop tests</div><div>latest: ${data.summary.latest_checkpoint || "-"}</div>`;
      renderList();
      if (state.tests.length) selectTest(state.tests[state.tests.length - 1].loop_test_id);
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
          <div class="row"><strong>${row.checkpoint}</strong><span class="pill">${row.policy_type}</span></div>
          <div class="muted">step ${row.training_step ?? "-"} · val ${fmt(row.validation_loss)}</div>
          <div class="muted">success ${fmt(row.success_rate)} · ${row.status}</div>
        </div>`).join("");
    }
    async function selectTest(id) {
      state.active = id;
      renderList();
      const data = await (await fetch(`/api/loop-test?id=${encodeURIComponent(id)}`)).json();
      renderDetail(data);
    }
    function renderDetail(data) {
      if (data.error) { el("detail").textContent = data.error; return; }
      const lt = data.loop_test;
      const ep = (data.episodes || [])[0] || { timeline: [] };
      el("detail").innerHTML = `
        <div class="header">
          <h2>${lt.policy_label} · ${lt.checkpoint}</h2>
          <div class="metrics">
            <span class="pill">scenario ${lt.scenario}</span>
            <span class="pill">step ${lt.training_step}</span>
            <span class="pill">val ${fmt(lt.validation_loss)}</span>
            <span class="pill">success ${fmt(lt.success_rate)}</span>
            <span class="pill">status means ${lt.status_meaning}</span>
          </div>
        </div>
        ${renderPlan(lt.qwen_plan)}
        <div class="timeline">${renderTimeline(ep.timeline || [])}</div>`;
    }
    function renderPlan(plan) {
      if (!plan) return "";
      return `<div class="plan"><strong>Planner</strong><div class="muted">${plan.model || "-"} · ${plan.task || "-"}</div><pre>${escapeHtml(JSON.stringify(plan.calls || [], null, 2))}</pre></div>`;
    }
    function renderTimeline(rows) {
      let lastIteration = null;
      return rows.map(row => {
        let label = "";
        if (row.type === "planner_call") label = `<div class="iteration">Planner</div>`;
        else if (row.type === "tool_call_end") label = `<div class="iteration">Iteration ${row.iteration} end</div>`;
        else if (row.type === "episode_end") label = `<div class="iteration">Episode end</div>`;
        else if (row.iteration !== lastIteration) label = `<div class="iteration">Iteration ${row.iteration}</div>`;
        lastIteration = row.iteration;
        return label + renderEvent(row);
      }).join("");
    }
    function renderEvent(row) {
      const policy = row.policy_input || row.policy_output || row.policy ? `
        <h3>Policy · ${row.type}</h3>
        <div class="muted">${row.tool_call || ""} ${row.primitive_id || ""}</div>
        ${row.policy_input?.prompt ? `<div><strong>Prompt</strong><br>${escapeHtml(row.policy_input.prompt)}</div>` : ""}
        ${row.policy_output?.action ? `<div><strong>Action</strong><pre>${escapeHtml(JSON.stringify(row.policy_output.action))}</pre></div>` : ""}
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
      return `<div class="event"><section class="panel">${policy}</section><section class="panel">${robot}</section></div>`;
    }
    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[ch]));
    }
    el("filter").addEventListener("input", renderList);
    el("statusFilter").addEventListener("change", renderList);
    loadList();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
