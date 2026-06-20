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
    :root {
      color-scheme: light;
      --border:#d8dde7; --muted:#667085; --bg:#f5f7fb; --ink:#162033;
      --accent:#0f766e; --accent-soft:#d9f3ef; --policy:#5b4bdb; --policy-soft:#eceafe;
      --robot:#047857; --robot-soft:#dcfce7; --warn:#b45309; --warn-soft:#fff7ed;
      --fail:#b42318; --fail-soft:#fee4e2; --ok:#027a48; --ok-soft:#dcfae6;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    .app { display: grid; grid-template-columns: 330px minmax(0, 1fr); min-height: 100vh; }
    aside { border-right: 1px solid var(--border); background: #fff; padding: 14px; overflow: auto; }
    main { overflow: auto; padding: 16px 18px 32px; }
    h1 { font-size: 18px; margin: 0 0 12px; }
    h2 { font-size: 16px; margin: 0; }
    .summary, .toolbar { display: grid; gap: 8px; margin-bottom: 12px; }
    input, select { width: 100%; border: 1px solid var(--border); border-radius: 6px; padding: 8px; background: #fff; }
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
    .chart-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:10px; margin-top:8px; }
    .chart { border:1px solid #edf0f5; border-radius:6px; padding:8px; background:#fbfcfe; }
    .chart svg { width:100%; height:96px; display:block; }
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
    let state = { tests: [], active: null, episode: 0 };
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
        </div>
          ${renderPlan(lt)}
        ${renderDiagnostics(ep)}
        <div class="timeline">${renderTimeline(groupTimeline(ep.timeline || []))}</div>`;
    }
    function renderDiagnostics(ep) {
      const rows = (ep.timeline || []).filter(row => row.type === "policy_step");
      if (!rows.length) return "";
      const rewards = rows.map(row => Number(row.robot?.reward || 0));
      const dists = rows.map(row => row.robot?.info?.tcp_to_target_dist).filter(value => typeof value === "number");
      const action0 = rows.map(row => row.policy_output?.action?.[0]).filter(value => typeof value === "number");
      const finalInfo = ep.final_info || {};
      const heuristics = [];
      if (ep.final_success === false) heuristics.push("task did not reach success");
      if (typeof finalInfo.tcp_to_target_dist === "number" && finalInfo.tcp_to_target_dist > 0.08) heuristics.push(`final distance high: ${finalInfo.tcp_to_target_dist.toFixed(3)}`);
      const endedByMaxSteps = rows.length === Number(ep.steps || rows.length);
      if (endedByMaxSteps && ep.final_success === false) heuristics.push("rollout consumed full planned horizon");
      return `<section class="diagnostics">
        <strong>Diagnostics</strong>
        <div class="metrics">${heuristics.map(item => `<span class="pill warn">${escapeHtml(item)}</span>`).join("") || `<span class="pill robot">no obvious failure heuristic</span>`}</div>
        <div class="chart-grid">
          ${renderSparkChart("reward", rewards, "#047857")}
          ${renderSparkChart("tcp distance", dists, "#b45309")}
          ${renderSparkChart("action[0]", action0, "#5b4bdb")}
        </div>
      </section>`;
    }
    function renderSparkChart(label, values, color) {
      if (!values.length) return `<div class="chart"><div class="muted">${escapeHtml(label)} unavailable</div></div>`;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max - min || 1;
      const points = values.map((value, index) => {
        const x = values.length === 1 ? 0 : (index / (values.length - 1)) * 100;
        const y = 88 - ((value - min) / span) * 76;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      }).join(" ");
      return `<div class="chart">
        <div class="row"><strong>${escapeHtml(label)}</strong><span class="muted">${fmt(values[values.length - 1])}</span></div>
        <svg viewBox="0 0 100 96" preserveAspectRatio="none">
          <polyline fill="none" stroke="${color}" stroke-width="2.5" points="${points}"></polyline>
        </svg>
        <div class="row muted"><span>min ${fmt(min)}</span><span>max ${fmt(max)}</span></div>
      </div>`;
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
      const generated = contract.generated_count ?? "unknown";
      const usedPerChunk = contract.used_per_chunk ?? "unknown";
      const chunks = groupActionChunks(steps);
      const confirmed = Boolean(contract.confirmed_in_rollout);
      const chunkLabel = chunks.length === 1 ? "chunk" : "chunks";
      return `
        <div class="iteration">Iteration ${start.iteration}: ${escapeHtml(start.tool_call || "-")}</div>
        <div class="event">
          <section class="panel policy">
            <h3>Policy tool call</h3>
            <div class="metrics">
              <span class="pill policy">function ${escapeHtml(start.tool_call || "-")}</span>
              <span class="pill">primitive ${escapeHtml(start.primitive_id || "-")}</span>
              <span class="pill ${confirmed ? "robot" : "warn"}">used per chunk (n_action_steps) ${usedPerChunk}</span>
              <span class="pill warn">generated horizon (chunk_size) ${generated}</span>
              <span class="pill robot">${chunks.length} ${chunkLabel}</span>
              <span class="pill">executed env actions ${steps.length}</span>
            </div>
            <div class="muted">${escapeHtml(contract.note || "")}</div>
            ${renderPolicyInputImages(firstMedia.policy_input_images || {})}
            <div class="tool-call">
              <strong>function + parameters</strong>
              <pre>${escapeHtml(JSON.stringify({ function: start.tool_call, parameters: start.tool_parameters || {} }, null, 2))}</pre>
            </div>
            ${start.policy_input?.prompt ? `<div><strong>Prompt</strong><br>${escapeHtml(start.policy_input.prompt)}</div>` : ""}
            <details>
              <summary>SmolVLA action chunks (${chunks.length})</summary>
              <div class="step-list">${chunks.map(renderActionChunk).join("")}</div>
            </details>
            <details>
              <summary>Rollout config evidence</summary>
              <pre>${escapeHtml(JSON.stringify({
                confirmed_in_rollout: confirmed,
                source: contract.rollout_config_source,
                generated_count: contract.generated_count,
                n_action_steps_used_per_chunk: contract.used_per_chunk,
                note: contract.note
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
      const expected = actionChunk.used_per_chunk ?? "unknown";
      const generated = actionChunk.generated_count ?? "unknown";
      return `<div class="step-card">
        <div class="metrics">
          <span class="pill policy">chunk ${chunk.chunkIndex + 1}</span>
          <span class="pill warn">chunk_size ${generated}</span>
          <span class="pill robot">used actions ${rows.length} / n_action_steps ${expected}</span>
          <span class="pill">global ${first.global_step ?? "-"}-${last.global_step ?? "-"}</span>
        </div>
        <details>
          <summary>Substeps (${rows.length})</summary>
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
          <pre>${escapeHtml(JSON.stringify({ action: row.policy_output?.action, observation: row.policy_input?.observation, info: row.robot?.info }, null, 2))}</pre>
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
