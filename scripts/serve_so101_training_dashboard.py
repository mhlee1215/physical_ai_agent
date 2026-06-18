#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from serve_so101_dataset_viewer import _datasets_payload, _frame_payload


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a lightweight SO101 SmolVLA training dashboard.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dataset-viewer-port", type=int, default=8768)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    repo_root = args.repo_root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_html(_index_html(run_dir))
                return
            if parsed.path == "/api/status":
                self._send_json(_status_payload(run_dir))
                return
            if parsed.path == "/api/datasets":
                self._send_json(_datasets_payload(repo_root))
                return
            if parsed.path == "/api/frame":
                query = parse_qs(parsed.query)
                split = (query.get("split") or ["train"])[0]
                episode = int((query.get("episode") or ["0"])[0])
                frame = int((query.get("frame") or ["0"])[0])
                self._send_json(_frame_payload(repo_root, split, episode, frame))
                return
            if parsed.path == "/artifact":
                query = parse_qs(parsed.query)
                path = Path((query.get("path") or [""])[0])
                self._send_artifact(run_dir, path)
                return
            self.send_error(404)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[dashboard] {self.address_string()} {fmt % args}", flush=True)

        def _send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_artifact(self, run_dir: Path, requested_path: Path) -> None:
            try:
                artifact_path = requested_path.resolve()
                artifact_path.relative_to(run_dir)
            except (OSError, ValueError):
                self.send_error(403)
                return
            if not artifact_path.is_file():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(str(artifact_path))[0] or "application/octet-stream"
            data = artifact_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ReusableThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dashboard] serving http://{args.host}:{args.port}/ run_dir={run_dir}", flush=True)
    server.serve_forever()


def _status_payload(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics" / "training_metrics.jsonl"
    validation_metrics_path = run_dir / "metrics" / "validation_metrics.jsonl"
    closed_loop_metrics_path = run_dir / "metrics" / "closed_loop_metrics.jsonl"
    monitor_events_path = run_dir / "metrics" / "monitor_events.jsonl"
    loss_summary_path = run_dir / "metrics" / "loss_summary.json"
    eval_report_path = run_dir / "eval_val24_seed98100_nact15" / "so101_picklift_smolvla_eval_report.json"
    checkpoints_dir = _checkpoints_dir(run_dir)
    checkpoints = []
    if checkpoints_dir.exists():
        checkpoints = sorted(path.name for path in checkpoints_dir.iterdir() if path.is_dir())
    metrics = _read_jsonl(metrics_path)
    validation_metrics = _read_jsonl(validation_metrics_path)
    closed_loop_metrics = _read_jsonl(closed_loop_metrics_path)
    monitor_events = _read_jsonl(monitor_events_path)
    latest_metric = metrics[-1] if metrics else None
    latest_validation_metric = validation_metrics[-1] if validation_metrics else None
    latest_closed_loop_metric = closed_loop_metrics[-1] if closed_loop_metrics else None
    latest_monitor_event = monitor_events[-1] if monitor_events else None
    return {
        "run_dir": str(run_dir),
        "train_output_dir": str(_train_output_dir(run_dir)),
        "metrics_path": str(metrics_path),
        "validation_metrics_path": str(validation_metrics_path),
        "closed_loop_metrics_path": str(closed_loop_metrics_path),
        "monitor_events_path": str(monitor_events_path),
        "loss_summary_path": str(loss_summary_path),
        "checkpoints": checkpoints,
        "latest_metric": latest_metric,
        "latest_validation_metric": latest_validation_metric,
        "latest_closed_loop_metric": latest_closed_loop_metric,
        "latest_monitor_event": latest_monitor_event,
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "closed_loop_metrics": closed_loop_metrics,
        "monitor_events": monitor_events,
        "loss_summary": _read_json(loss_summary_path),
        "eval_report": _summarize_eval(_read_json(eval_report_path)),
        "closed_loop_artifacts": _closed_loop_artifacts(run_dir),
        "system_status": _system_status(),
    }


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _train_output_dir(run_dir: Path) -> Path:
    nested = run_dir / "model"
    return nested if nested.exists() else run_dir


def _checkpoints_dir(run_dir: Path) -> Path:
    return _train_output_dir(run_dir) / "checkpoints"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _summarize_eval(report: Any | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    episodes = report.get("episodes") or []
    return {
        "success_rate": report.get("success_rate"),
        "grasp_rate": report.get("grasp_rate"),
        "episodes": len(episodes),
        "policy_rollout_config": report.get("policy_rollout_config"),
        "report_path": report.get("report_path"),
    }


def _closed_loop_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    root = run_dir / "closed_loop_evals"
    if not root.exists():
        return []
    rows = []
    media_suffixes = {".gif", ".mp4", ".webm", ".png", ".jpg", ".jpeg"}
    for path in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True):
        if not path.is_dir():
            continue
        report_path = path / "so101_picklift_smolvla_eval_report.json"
        report = _read_json(report_path)
        media = [
            {
                "name": media_path.name,
                "path": str(media_path),
                "url": f"/artifact?path={media_path}",
            }
            for media_path in sorted(path.rglob("*"))
            if media_path.is_file() and media_path.suffix.lower() in media_suffixes
        ]
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "report_path": str(report_path) if report_path.exists() else None,
                "report_url": f"/artifact?path={report_path}" if report_path.exists() else None,
                "success_rate": report.get("success_rate") if isinstance(report, dict) else None,
                "grasp_rate": report.get("grasp_rate") if isinstance(report, dict) else None,
                "episodes": len(report.get("episodes") or []) if isinstance(report, dict) else None,
                "media": media[:12],
                "rollout_episodes": _compact_rollout_episodes(report),
            }
        )
    return rows


def _compact_rollout_episodes(report: Any | None) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    rows = []
    for episode in report.get("episodes") or []:
        records = episode.get("records") or []
        compact_records = []
        min_dist = None
        max_lift = None
        for record in records:
            dist = _float_or_none(record.get("tcp_to_obj_dist"))
            lift = _float_or_none(record.get("lift_height"))
            if dist is not None:
                min_dist = dist if min_dist is None else min(min_dist, dist)
            if lift is not None:
                max_lift = lift if max_lift is None else max(max_lift, lift)
            compact_records.append(
                {
                    "step": record.get("step"),
                    "tcp_to_obj_dist": dist,
                    "lift_height": lift,
                    "is_grasped": bool(record.get("is_grasped")),
                    "success": bool(record.get("success")),
                }
            )
        rows.append(
            {
                "episode": episode.get("episode"),
                "seed": episode.get("seed"),
                "steps": episode.get("steps"),
                "success": bool(episode.get("success")),
                "final_is_grasped": bool(episode.get("final_is_grasped")),
                "final_lift_height": _float_or_none(episode.get("final_lift_height")),
                "final_tcp_to_obj_dist": _float_or_none(episode.get("final_tcp_to_obj_dist")),
                "min_tcp_to_obj_dist": min_dist,
                "max_lift_height": max_lift,
                "search_steps": episode.get("search_steps"),
                "records": compact_records,
            }
        )
    return rows


def _system_status() -> dict[str, Any]:
    mem = _memory_status()
    return {
        "memory": mem,
        "training_processes": _training_processes(),
        "gpu": _gpu_status(),
    }


def _memory_status() -> dict[str, Any]:
    linux_mem = _linux_memory_status()
    if linux_mem:
        return linux_mem
    total_bytes = _sysctl_int("hw.memsize")
    page_size = _sysctl_int("hw.pagesize") or 4096
    vm = _vm_stat()
    free_pages = vm.get("Pages free", 0) + vm.get("Pages speculative", 0)
    active_pages = vm.get("Pages active", 0)
    inactive_pages = vm.get("Pages inactive", 0)
    wired_pages = vm.get("Pages wired down", 0)
    compressed_pages = vm.get("Pages occupied by compressor", 0)
    used_bytes = max(0, total_bytes - free_pages * page_size) if total_bytes else None
    return {
        "physical_gb": _bytes_to_gb(total_bytes),
        "used_gb": _bytes_to_gb(used_bytes),
        "free_gb": _bytes_to_gb(free_pages * page_size),
        "active_gb": _bytes_to_gb(active_pages * page_size),
        "inactive_gb": _bytes_to_gb(inactive_pages * page_size),
        "wired_gb": _bytes_to_gb(wired_pages * page_size),
        "compressed_gb": _bytes_to_gb(compressed_pages * page_size),
        "swap": _swap_status(),
    }


def _linux_memory_status() -> dict[str, Any] | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    rows: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            rows[parts[0].rstrip(":")] = int(parts[1]) * 1024
    total = rows.get("MemTotal")
    available = rows.get("MemAvailable")
    free = rows.get("MemFree")
    swap_total = rows.get("SwapTotal")
    swap_free = rows.get("SwapFree")
    used = total - available if total is not None and available is not None else None
    return {
        "physical_gb": _bytes_to_gb(total),
        "used_gb": _bytes_to_gb(used),
        "free_gb": _bytes_to_gb(free),
        "available_gb": _bytes_to_gb(available),
        "swap": {
            "total_gb": _bytes_to_gb(swap_total),
            "used_gb": _bytes_to_gb(
                swap_total - swap_free if swap_total is not None and swap_free is not None else None
            ),
            "free_gb": _bytes_to_gb(swap_free),
        },
    }


def _gpu_status() -> dict[str, Any]:
    text = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    if text:
        rows = []
        for line in text.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                continue
            rows.append(
                {
                    "name": parts[0],
                    "memory_total_mb": _float_or_none(parts[1]),
                    "memory_used_mb": _float_or_none(parts[2]),
                    "memory_free_mb": _float_or_none(parts[3]),
                    "utilization_percent": _float_or_none(parts[4]),
                    "temperature_c": _float_or_none(parts[5]),
                    "power_w": _float_or_none(parts[6]),
                }
            )
        return {"backend": "NVIDIA CUDA", "gpus": rows}
    return {
        "backend": "Apple MPS / unified memory",
        "note": "macOS does not expose per-process MPS/GPU memory through this dashboard without privileged powermetrics.",
    }


def _training_processes() -> list[dict[str, Any]]:
    text = _run_text(["ps", "-axo", "pid,pcpu,pmem,rss,command"])
    rows = []
    for line in text.splitlines()[1:]:
        if "lerobot_train_so101_sampling_aug.py" not in line and "monitor_so101_training_dashboard.py" not in line:
            continue
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid, pcpu, pmem, rss_kb, command = parts
        rows.append(
            {
                "pid": int(pid),
                "cpu_percent": _float_or_none(pcpu),
                "mem_percent": _float_or_none(pmem),
                "rss_gb": _bytes_to_gb(int(rss_kb) * 1024),
                "command": Path(command.split()[0]).name if command else "",
                "role": "train" if "lerobot_train_so101_sampling_aug.py" in line else "monitor",
            }
        )
    return rows


def _vm_stat() -> dict[str, int]:
    text = _run_text(["vm_stat"])
    rows = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            rows[key.strip()] = int(digits)
    return rows


def _swap_status() -> dict[str, Any]:
    text = _run_text(["sysctl", "-n", "vm.swapusage"]).strip()
    result: dict[str, Any] = {"raw": text}
    for key, value in re.findall(r"(total|used|free)\s*=\s*([0-9.]+[MG])", text):
        if value.endswith("M"):
            result[f"{key.lower()}_gb"] = round(float(value[:-1]) / 1024, 2)
        elif value.endswith("G"):
            result[f"{key.lower()}_gb"] = round(float(value[:-1]), 2)
    return result


def _sysctl_int(name: str) -> int | None:
    text = _run_text(["sysctl", "-n", name]).strip()
    try:
        return int(text)
    except ValueError:
        return None


def _run_text(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=2).stdout
    except Exception:
        return ""


def _bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024**3), 2)


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _index_html(run_dir: Path) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SO101 Training</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f6f7f9; color: #15171a; }}
    header {{ padding: 18px 22px; background: #ffffff; border-bottom: 1px solid #d8dde6; }}
    h1 {{ margin: 0; font-size: 20px; font-weight: 650; letter-spacing: 0; }}
    main {{ padding: 18px 22px; display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    section {{ background: #ffffff; border: 1px solid #d8dde6; border-radius: 8px; padding: 14px; }}
    h2 {{ margin: 0 0 10px; font-size: 14px; color: #475467; font-weight: 650; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .metric {{ border: 1px solid #e1e6ef; border-radius: 6px; padding: 10px; min-height: 62px; }}
    .label {{ color: #667085; font-size: 12px; }}
    .value {{ margin-top: 6px; font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }}
    .chart {{ width: 100%; height: 260px; border: 1px solid #e1e6ef; border-radius: 6px; background: #fbfcfe; }}
    .legend {{ display: flex; gap: 14px; align-items: center; margin-top: 8px; color: #475467; font-size: 12px; }}
    .swatch {{ display: inline-block; width: 18px; height: 3px; border-radius: 999px; vertical-align: middle; margin-right: 5px; }}
    .tabs {{ display: flex; gap: 8px; padding: 10px 22px 0; background: #ffffff; border-bottom: 1px solid #d8dde6; }}
    .tab {{ border: 1px solid #cfd6e3; border-bottom: 0; border-radius: 8px 8px 0 0; padding: 9px 12px; background: #f6f7f9; color: #475467; font: inherit; font-weight: 650; cursor: pointer; }}
    .tab.active {{ background: #ffffff; color: #15171a; }}
    .panel {{ display: none; }}
    .panel.active {{ display: grid; }}
    .toolbar {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }}
    .button {{ display: inline-flex; align-items: center; justify-content: center; border: 1px solid #cfd6e3; border-radius: 6px; padding: 7px 10px; color: #1f2937; background: #fff; text-decoration: none; font-weight: 650; font-size: 13px; }}
    .dataset-controls {{ display: grid; grid-template-columns: 190px 1fr 1fr auto auto 96px auto auto; gap: 10px; align-items: end; }}
    label {{ display: grid; gap: 4px; font-size: 12px; color: #596273; }}
    select, input, button {{ font: inherit; }}
    select, input {{ border: 1px solid #cfd6e3; border-radius: 6px; padding: 7px 9px; background: #fff; }}
    .cameras {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; color: #596273; margin-bottom: 5px; }}
    img.dataset-image {{ width: 100%; image-rendering: pixelated; border: 1px solid #d9dee8; border-radius: 6px; background: #111; }}
    .kv {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .kv div {{ border: 1px solid #e1e6ef; border-radius: 6px; padding: 9px; }}
    .kv strong {{ display: block; font-size: 17px; margin-top: 4px; }}
    .dataset-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-top: 10px; }}
    .dataset-group {{ margin-top: 14px; }}
    .dataset-group:first-child {{ margin-top: 10px; }}
    .dataset-group-title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 4px; }}
    .dataset-group-title h3 {{ margin: 0; font-size: 13px; color: #344054; }}
    .dataset-group-title span {{ font-size: 12px; color: #667085; }}
    .dataset-card {{ border: 1px solid #e1e6ef; border-radius: 6px; padding: 10px; background: #fbfcfe; cursor: pointer; }}
    .dataset-card.unavailable {{ cursor: default; opacity: 0.76; background: #f8fafc; }}
    .dataset-card.active {{ border-color: #2563eb; box-shadow: inset 0 0 0 1px #2563eb; }}
    .dataset-card h3 {{ margin: 0 0 6px; font-size: 13px; overflow-wrap: anywhere; }}
    .dataset-card .row {{ display: flex; justify-content: space-between; gap: 8px; font-size: 12px; color: #475467; padding-top: 3px; }}
    .dataset-card .row strong {{ text-align: right; overflow-wrap: anywhere; }}
    .dataset-card .path {{ margin-top: 6px; font-size: 11px; color: #667085; overflow-wrap: anywhere; }}
    .dataset-badges {{ display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-bottom: 6px; }}
    .dataset-badge {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 2px 7px; font-size: 11px; font-weight: 700; background: #eef2ff; color: #3730a3; }}
    .dataset-badge.available {{ background: #dcfce7; color: #166534; }}
    .dataset-badge.incomplete, .dataset-badge.missing {{ background: #fef3c7; color: #92400e; }}
    .rollout-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; margin-top: 10px; }}
    .rollout-card {{ border: 1px solid #e1e6ef; border-radius: 6px; padding: 9px; background: #fbfcfe; }}
    .rollout-head {{ display: flex; justify-content: space-between; gap: 8px; align-items: center; font-size: 13px; font-weight: 650; }}
    .pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 2px 7px; font-size: 11px; font-weight: 700; }}
    .pill.ok {{ background: #dcfce7; color: #166534; }}
    .pill.warn {{ background: #fef3c7; color: #92400e; }}
    .pill.fail {{ background: #fee2e2; color: #991b1b; }}
    .rollout-svg {{ width: 100%; height: 118px; margin-top: 7px; border: 1px solid #edf0f5; border-radius: 5px; background: #fff; }}
    details.closed-eval {{ border: 1px solid #e1e6ef; border-radius: 6px; padding: 10px; margin-bottom: 10px; background: #fff; }}
    details.closed-eval summary {{ cursor: pointer; font-weight: 650; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #edf0f5; text-align: right; padding: 7px 5px; font-variant-numeric: tabular-nums; }}
    th:first-child, td:first-child {{ text-align: left; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f3f5f8; border: 1px solid #e1e6ef; border-radius: 6px; padding: 10px; max-height: 260px; overflow: auto; }}
    .wide {{ grid-column: 1 / -1; }}
    .muted {{ color: #667085; font-size: 12px; margin-top: 8px; }}
    @media (max-width: 820px) {{ main {{ grid-template-columns: 1fr; }} .metric-grid, .dataset-controls, .cameras {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header><h1>SO101 SmolVLA Training Dashboard</h1></header>
  <nav class="tabs" aria-label="Dashboard tabs">
    <button class="tab active" data-tab="datasetPanel">Dataset</button>
    <button class="tab" data-tab="closedLoopPanel">Closed-loop</button>
  </nav>
  <main id="trainingPanel" class="panel" style="display:none" aria-hidden="true">
    <section class="wide">
      <h2>Run</h2>
      <div class="metric-grid">
        <div class="metric"><div class="label">Step</div><div id="step" class="value">-</div></div>
        <div class="metric"><div class="label">Train Loss</div><div id="trainLoss" class="value">-</div></div>
        <div class="metric"><div class="label">Validation Loss</div><div id="valLoss" class="value">-</div></div>
        <div class="metric"><div class="label">Closed-loop Success</div><div id="closedSuccess" class="value">-</div></div>
        <div class="metric"><div class="label">Closed-loop Grasp</div><div id="closedGrasp" class="value">-</div></div>
        <div class="metric"><div class="label">Closed-loop Checkpoint</div><div id="closedCheckpoint" class="value">-</div></div>
      </div>
    </section>
    <section>
      <h2>Training Logs</h2>
      <table><thead><tr><th>Step</th><th>Train</th><th>Grad</th><th>LR</th></tr></thead><tbody id="metrics"></tbody></table>
    </section>
    <section>
      <h2>Loss Trend</h2>
      <svg id="lossChart" class="chart" viewBox="0 0 640 260" role="img" aria-label="Training and validation loss chart"></svg>
      <div class="legend">
        <span><span class="swatch" style="background:#2563eb"></span>train log loss</span>
        <span><span class="swatch" style="background:#dc2626"></span>validation sample loss</span>
      </div>
    </section>
    <section>
      <h2>Evaluation</h2>
      <pre id="eval">waiting for eval report...</pre>
    </section>
    <section>
      <h2>Monitor</h2>
      <table><thead><tr><th>Time</th><th>Kind</th><th>Detail</th></tr></thead><tbody id="monitor"></tbody></table>
      <div id="monitorNote" class="muted"></div>
    </section>
    <section class="wide">
      <h2>System</h2>
      <div id="systemKv" class="kv"></div>
      <pre id="systemRaw">loading...</pre>
    </section>
    <section class="wide">
      <h2>Artifacts</h2>
      <pre id="artifacts">loading...</pre>
    </section>
  </main>
  <main id="datasetPanel" class="panel active">
    <section class="wide">
      <h2>Dataset</h2>
      <div id="datasetSummary" class="muted"></div>
    </section>
    <section class="wide">
      <h2>Camera Frames</h2>
      <div class="dataset-controls">
        <label>Dataset<select id="datasetSplit"></select></label>
        <label>Episode<input id="datasetEpisode" type="range" min="0" max="0" value="0"></label>
        <label>Frame<input id="datasetFrame" type="range" min="0" max="0" value="0"></label>
        <button id="datasetPlay" class="button" type="button">Play</button>
        <label>FPS<select id="datasetFps"><option value="12" selected>12</option><option value="6">6</option><option value="24">24</option></select></label>
        <button id="datasetPrev" class="button" type="button">Prev</button>
        <button id="datasetNext" class="button" type="button">Next</button>
      </div>
      <div id="datasetMeta" class="muted"></div>
      <div id="datasetCameras" class="cameras"></div>
    </section>
    <section class="wide">
      <h2>State / Action</h2>
      <table>
        <thead><tr><th>Joint</th><th>State</th><th>Action</th></tr></thead>
        <tbody id="datasetJointRows"></tbody>
      </table>
    </section>
  </main>
  <main id="closedLoopPanel" class="panel">
    <section class="wide">
      <h2>Closed Loop</h2>
      <table><thead><tr><th>Step</th><th>Success</th><th>Grasp</th><th>Episodes</th></tr></thead><tbody id="closedLoop"></tbody></table>
      <div id="closedLoopNote" class="muted"></div>
    </section>
    <section class="wide">
      <h2>Closed-loop Artifacts</h2>
      <div id="closedLoopArtifacts" class="muted">waiting for closed-loop evaluation artifacts...</div>
    </section>
  </main>
  <script>
    const runDir = {json.dumps(str(run_dir))};
    const fmt = (v, n=4) => v === null || v === undefined ? "-" : Number(v).toFixed(n);
    let datasetInfo = null;
    let datasetGroups = [];
    let datasetTimer = null;
    let datasetLoading = false;
    let statusLoading = false;
    function artifactLink(item) {{
      if (!item || !item.url) return "";
      return `<a class="button" href="${{item.url}}" target="_blank" rel="noreferrer">${{item.name}}</a>`;
    }}
    document.querySelectorAll(".tab").forEach(button => {{
      button.addEventListener("click", () => {{
        document.querySelectorAll(".tab").forEach(tab => tab.classList.toggle("active", tab === button));
        document.querySelectorAll(".panel").forEach(panel => panel.classList.toggle("active", panel.id === button.dataset.tab));
        if (button.dataset.tab === "datasetPanel") loadDatasetFrame();
      }});
    }});
    async function refresh() {{
      if (statusLoading) return;
      const activePanel = document.querySelector(".panel.active")?.id;
      if (activePanel === "datasetPanel" && datasetTimer) return;
      statusLoading = true;
      try {{
      const res = await fetch("/api/status", {{ cache: "no-store" }});
      const data = await res.json();
      const latest = data.latest_metric || {{}};
      const loss = data.loss_summary || {{}};
      const latestClosed = data.latest_closed_loop_metric || {{}};
      document.getElementById("step").textContent = latest.step ?? "-";
      document.getElementById("trainLoss").textContent = fmt(loss.latest_train_loss ?? latest.loss);
      const latestVal = data.latest_validation_metric || {{}};
      document.getElementById("valLoss").textContent = fmt(latestVal.loss ?? loss.latest_val_loss);
      const closedEpisodes = latestClosed.episodes || null;
      const closedSuccessCount = closedEpisodes && latestClosed.success_rate !== undefined
        ? Math.round(Number(latestClosed.success_rate) * Number(closedEpisodes))
        : null;
      const closedGraspCount = closedEpisodes && latestClosed.grasp_rate !== undefined
        ? Math.round(Number(latestClosed.grasp_rate) * Number(closedEpisodes))
        : null;
      document.getElementById("closedSuccess").textContent = closedSuccessCount === null
        ? "-"
        : `${{closedSuccessCount}}/${{closedEpisodes}}`;
      document.getElementById("closedGrasp").textContent = closedGraspCount === null
        ? "-"
        : `${{closedGraspCount}}/${{closedEpisodes}}`;
      document.getElementById("closedCheckpoint").textContent = latestClosed.checkpoint ?? "-";
      document.getElementById("metrics").innerHTML = (data.metrics || []).slice(-12).reverse().map(row =>
        `<tr><td>${{row.step}}</td><td>${{fmt(row.loss)}}</td><td>${{fmt(row.grad_norm)}}</td><td>${{row.lr ?? "-"}}</td></tr>`
      ).join("");
      document.getElementById("eval").textContent = JSON.stringify(data.eval_report || "waiting for eval report...", null, 2);
      document.getElementById("closedLoop").innerHTML = (data.closed_loop_metrics || []).slice(-8).reverse().map(row =>
        `<tr><td>${{row.step ?? "-"}}</td><td>${{fmt(row.success_rate, 3)}}</td><td>${{fmt(row.grasp_rate, 3)}}</td><td>${{row.episodes ?? "-"}}</td></tr>`
      ).join("");
      document.getElementById("closedLoopNote").textContent = latestClosed.checkpoint
        ? `latest closed-loop checkpoint: ${{latestClosed.checkpoint}}`
        : "closed-loop validation will run every 5 epochs after checkpoint save";
      renderClosedLoopArtifacts(data.closed_loop_artifacts || []);
      document.getElementById("monitor").innerHTML = (data.monitor_events || []).slice(-8).reverse().map(row =>
        `<tr><td>${{row.checked_at_local ?? row.checked_at_utc ?? "-"}}</td><td>${{row.kind ?? "-"}}</td><td>${{row.detail ?? row.checkpoint ?? "-"}}</td></tr>`
      ).join("");
      renderSystem(data.system_status || {{}});
      const latestMonitor = data.latest_monitor_event || {{}};
      document.getElementById("monitorNote").textContent = latestMonitor.checked_at_local
        ? `last check: ${{latestMonitor.checked_at_local}}`
        : "waiting for first 10-minute monitor check";
      document.getElementById("artifacts").textContent = JSON.stringify({{
        run_dir: runDir,
        checkpoints: data.checkpoints,
        metrics_path: data.metrics_path,
        validation_metrics_path: data.validation_metrics_path,
        closed_loop_metrics_path: data.closed_loop_metrics_path,
        monitor_events_path: data.monitor_events_path,
        loss_summary_path: data.loss_summary_path
      }}, null, 2);
      renderLossChart(data.metrics || [], data.validation_metrics || []);
      }} finally {{
        statusLoading = false;
      }}
    }}
    function renderClosedLoopArtifacts(rows) {{
      const target = document.getElementById("closedLoopArtifacts");
      if (!rows.length) {{
        target.textContent = "waiting for closed-loop evaluation artifacts...";
        return;
      }}
      target.innerHTML = rows.slice(0, 6).map((row, index) => {{
        const report = row.report_url ? `<a class="button" href="${{row.report_url}}" target="_blank" rel="noreferrer">report</a>` : "";
        const media = (row.media || []).map(artifactLink).join("");
        const mediaNote = media || `<span class="muted">No GIF/MP4 recorded for this run. Showing rollout traces from the evaluation JSON.</span>`;
        const episodes = row.rollout_episodes || [];
        const successCount = episodes.filter(ep => ep.success).length;
        const graspCount = episodes.filter(ep => ep.final_is_grasped).length;
        const episodeGrid = episodes.length
          ? `<div class="rollout-grid">${{episodes.map(renderEpisodeRollout).join("")}}</div>`
          : `<div class="muted">No per-step rollout records found in this report.</div>`;
        return `
          <details class="closed-eval" ${{index === 0 ? "open" : ""}}>
            <summary>${{row.name}} · success ${{successCount}}/${{row.episodes ?? episodes.length}} · grasp ${{graspCount}}/${{row.episodes ?? episodes.length}}</summary>
            <div class="muted">success=${{fmt(row.success_rate, 3)}} grasp=${{fmt(row.grasp_rate, 3)}} episodes=${{row.episodes ?? "-"}}</div>
            <div class="toolbar" style="margin-top:8px">${{report}}${{mediaNote}}</div>
            <div class="muted">${{row.path}}</div>
            <div class="legend">
              <span><span class="swatch" style="background:#dc2626"></span>tcp-to-object distance</span>
              <span><span class="swatch" style="background:#16a34a"></span>lift height</span>
              <span><span class="swatch" style="background:#2563eb"></span>grasped/success markers</span>
            </div>
            ${{episodeGrid}}
          </details>
        `;
      }}).join("");
    }}
    function renderEpisodeRollout(ep) {{
      const records = (ep.records || []).filter(row => row && Number.isFinite(Number(row.step)));
      const statusClass = ep.success ? "ok" : (ep.final_is_grasped ? "warn" : "fail");
      const statusText = ep.success ? "SUCCESS" : (ep.final_is_grasped ? "GRASP" : "FAIL");
      const finalLift = fmt(ep.final_lift_height, 3);
      const minDist = fmt(ep.min_tcp_to_obj_dist, 3);
      const finalDist = fmt(ep.final_tcp_to_obj_dist, 3);
      const chart = records.length ? rolloutSvg(records) : `<svg class="rollout-svg" viewBox="0 0 260 118"><text x="16" y="62" fill="#667085" font-size="12">no records</text></svg>`;
      return `
        <div class="rollout-card">
          <div class="rollout-head">
            <span>episode ${{ep.episode ?? "-"}}</span>
            <span class="pill ${{statusClass}}">${{statusText}}</span>
          </div>
          <div class="muted">seed=${{ep.seed ?? "-"}} steps=${{ep.steps ?? records.length}} final_lift=${{finalLift}} min_dist=${{minDist}} final_dist=${{finalDist}}</div>
          ${{chart}}
        </div>
      `;
    }}
    function rolloutSvg(records) {{
      const w = 260, h = 118, pad = 18;
      const steps = records.map(row => Number(row.step));
      const minX = Math.min(...steps);
      const maxX = Math.max(...steps);
      const dists = records.map(row => Number(row.tcp_to_obj_dist)).filter(Number.isFinite);
      const lifts = records.map(row => Number(row.lift_height)).filter(Number.isFinite);
      const maxDist = Math.max(0.001, ...dists);
      const maxLift = Math.max(0.001, ...lifts);
      const sx = x => pad + (Number(x) - minX) / Math.max(1, maxX - minX) * (w - pad * 2);
      const syDist = y => h - pad - Math.max(0, Number(y)) / maxDist * (h - pad * 2);
      const syLift = y => h - pad - Math.max(0, Number(y)) / maxLift * (h - pad * 2);
      const pathFor = (key, sy) => records
        .filter(row => Number.isFinite(Number(row[key])))
        .map((row, i) => `${{i ? "L" : "M"}} ${{sx(row.step).toFixed(1)}} ${{sy(row[key]).toFixed(1)}}`)
        .join(" ");
      const graspMarkers = records
        .filter(row => row.is_grasped || row.success)
        .map(row => {{
          const x = sx(row.step).toFixed(1);
          const color = row.success ? "#2563eb" : "#93c5fd";
          return `<line x1="${{x}}" y1="${{pad}}" x2="${{x}}" y2="${{h - pad}}" stroke="${{color}}" stroke-width="${{row.success ? 2 : 1}}" opacity="0.75"/>`;
        }})
        .join("");
      return `
        <svg class="rollout-svg" viewBox="0 0 ${{w}} ${{h}}" role="img" aria-label="Closed-loop rollout trace">
          <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{h - pad}}" stroke="#98a2b3"/>
          <line x1="${{pad}}" y1="${{h - pad}}" x2="${{w - pad}}" y2="${{h - pad}}" stroke="#98a2b3"/>
          <line x1="${{pad}}" y1="${{pad}}" x2="${{w - pad}}" y2="${{pad}}" stroke="#edf0f5"/>
          <line x1="${{pad}}" y1="${{(h / 2).toFixed(1)}}" x2="${{w - pad}}" y2="${{(h / 2).toFixed(1)}}" stroke="#edf0f5"/>
          ${{graspMarkers}}
          <path d="${{pathFor("tcp_to_obj_dist", syDist)}}" fill="none" stroke="#dc2626" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="${{pathFor("lift_height", syLift)}}" fill="none" stroke="#16a34a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          <text x="${{pad}}" y="13" fill="#667085" font-size="10">d max ${{maxDist.toFixed(3)}}</text>
          <text x="${{w - pad - 70}}" y="13" fill="#667085" font-size="10">lift max ${{maxLift.toFixed(3)}}</text>
          <text x="${{w - pad - 48}}" y="${{h - 5}}" fill="#667085" font-size="10">step ${{maxX}}</text>
        </svg>
      `;
    }}
    async function initDataset() {{
      const payload = await fetch("/api/datasets", {{ cache: "no-store" }}).then(res => res.json());
      datasetInfo = payload.datasets || {{}};
      datasetGroups = payload.dataset_groups || [];
      const split = document.getElementById("datasetSplit");
      const current = split.value;
      split.innerHTML = Object.keys(datasetInfo).map(name => `<option value="${{name}}">${{name}}</option>`).join("");
      if (current && datasetInfo[current]) split.value = current;
      syncDatasetEpisodeRange();
      renderDatasetSummary();
      await loadDatasetFrame();
    }}
    function renderDatasetSummary() {{
      const target = document.getElementById("datasetSummary");
      if (!datasetInfo || !target) return;
      const active = document.getElementById("datasetSplit").value;
      const groups = datasetGroups.length
        ? datasetGroups
        : [{{ id: "datasets", title: "Datasets", description: "", items: Object.entries(datasetInfo).map(([name, summary]) => ({{ name, status: "available", summary }})) }}];
      target.innerHTML = groups.map(group => {{
        const items = group.items || [];
        if (!items.length) {{
          return `
            <div class="dataset-group">
              <div class="dataset-group-title"><h3>${{group.title}}</h3><span>0 datasets</span></div>
              <div class="muted">${{group.description || ""}}</div>
            </div>
          `;
        }}
        return `
          <div class="dataset-group">
            <div class="dataset-group-title"><h3>${{group.title}}</h3><span>${{items.length}} datasets</span></div>
            <div class="muted">${{group.description || ""}}</div>
            <div class="dataset-cards">${{items.map(item => renderDatasetCard(item, active)).join("")}}</div>
          </div>
        `;
      }}).join("");
      target.querySelectorAll(".dataset-card[data-split]").forEach(card => {{
        card.addEventListener("click", () => {{
          const name = card.dataset.split;
          if (!name || !datasetInfo[name]) return;
          const split = document.getElementById("datasetSplit");
          split.value = name;
          stopDatasetPlayback();
          syncDatasetEpisodeRange();
          renderDatasetSummary();
          loadDatasetFrame();
        }});
      }});
    }}
    function renderDatasetCard(item, active) {{
      const splitName = item.name;
      const data = item.summary || datasetInfo[splitName] || item;
      const status = item.status || (datasetInfo[splitName] ? "available" : "missing");
      const available = status === "available" && !!datasetInfo[splitName];
      const features = data.features || [];
        const shapes = data.image_shapes || {{}};
        const firstShape = features.length ? shapes[features[0]] : null;
        const shape = firstShape ? `${{firstShape[0]}}x${{firstShape[1]}}x${{firstShape[2]}}` : "no camera";
      const category = item.category || "dataset";
      const detail = item.detail && item.detail !== "ready" ? `<div class="path">${{item.detail}}</div>` : "";
      const root = data.root || item.root || "";
      return `
          <div class="dataset-card ${{splitName === active ? "active" : ""}} ${{available ? "" : "unavailable"}}" ${{available ? `data-split="${{splitName}}"` : ""}}>
            <div class="dataset-badges">
              <span class="dataset-badge">${{category}}</span>
              <span class="dataset-badge ${{status}}">${{status}}</span>
            </div>
            <h3>${{splitName}}</h3>
            <div class="row"><span>episodes</span><strong>${{data.episodes ?? "-"}}</strong></div>
            <div class="row"><span>frames</span><strong>${{data.frames ?? "-"}}</strong></div>
            <div class="row"><span>fps</span><strong>${{data.fps ?? "-"}}</strong></div>
            <div class="row"><span>image shape</span><strong>${{shape}}</strong></div>
            <div class="row"><span>dataset size</span><strong>${{data.size_human ?? "-"}}</strong></div>
            <div class="row"><span>data parquet</span><strong>${{data.data_human ?? "-"}}</strong></div>
            <div class="row"><span>image volume</span><strong>${{data.image_human ?? "-"}}</strong></div>
            <div class="path">${{root}}</div>
            ${{detail}}
          </div>
        `;
    }}
    function datasetEls() {{
      return {{
        split: document.getElementById("datasetSplit"),
        episode: document.getElementById("datasetEpisode"),
        frame: document.getElementById("datasetFrame"),
        meta: document.getElementById("datasetMeta"),
        cameras: document.getElementById("datasetCameras"),
        joints: document.getElementById("datasetJointRows"),
      }};
    }}
    function syncDatasetEpisodeRange() {{
      if (!datasetInfo) return;
      const el = datasetEls();
      const data = datasetInfo[el.split.value];
      if (!data) return;
      el.episode.max = String(Math.max(0, data.episodes - 1));
      el.episode.value = String(Math.min(Number(el.episode.value), data.episodes - 1));
      syncDatasetFrameRange();
    }}
    function syncDatasetFrameRange() {{
      if (!datasetInfo) return;
      const el = datasetEls();
      const data = datasetInfo[el.split.value];
      if (!data) return;
      const length = data.episode_lengths[Number(el.episode.value)] || 1;
      el.frame.max = String(Math.max(0, length - 1));
      el.frame.value = String(Math.min(Number(el.frame.value), length - 1));
    }}
    async function loadDatasetFrame() {{
      if (!datasetInfo) return;
      if (datasetLoading) return;
      datasetLoading = true;
      syncDatasetFrameRange();
      const el = datasetEls();
      const url = `/api/frame?split=${{el.split.value}}&episode=${{el.episode.value}}&frame=${{el.frame.value}}`;
      let row;
      try {{
        row = await fetch(url, {{ cache: "no-store" }}).then(res => res.json());
      }} finally {{
        datasetLoading = false;
      }}
      const totalEpisodes = datasetInfo[row.split]?.episodes ?? 0;
      el.meta.textContent = `${{row.split}} | episode ${{row.episode}}/${{Math.max(0, totalEpisodes - 1)}} | frame ${{row.frame}}/${{row.episode_length - 1}} | row ${{row.row_index}} | t=${{row.timestamp.toFixed(3)}}s | ${{row.task}}`;
      el.cameras.innerHTML = Object.entries(row.images || {{}}).map(([name, src]) => `
        <figure>
          <figcaption>${{name}}</figcaption>
          <img class="dataset-image" src="${{src}}" alt="${{name}}">
        </figure>
      `).join("");
      el.joints.innerHTML = Object.keys(row.state || {{}}).map(joint => `
        <tr><td>${{joint}}</td><td>${{fmt(row.state[joint])}}</td><td>${{fmt(row.action[joint])}}</td></tr>
      `).join("");
    }}
    function stopDatasetPlayback() {{
      if (!datasetTimer) return;
      clearInterval(datasetTimer);
      datasetTimer = null;
      document.getElementById("datasetPlay").textContent = "Play";
    }}
    function startDatasetPlayback() {{
      stopDatasetPlayback();
      document.getElementById("datasetPlay").textContent = "Pause";
      const fps = Number(document.getElementById("datasetFps").value || 12);
      datasetTimer = setInterval(() => {{
        const frame = document.getElementById("datasetFrame");
        if (Number(frame.value) >= Number(frame.max)) frame.value = "0";
        else frame.value = String(Number(frame.value) + 1);
        loadDatasetFrame();
      }}, Math.max(16, 1000 / fps));
    }}
    document.getElementById("datasetSplit").addEventListener("change", () => {{ stopDatasetPlayback(); syncDatasetEpisodeRange(); renderDatasetSummary(); loadDatasetFrame(); }});
    document.getElementById("datasetEpisode").addEventListener("input", () => {{ document.getElementById("datasetFrame").value = "0"; loadDatasetFrame(); }});
    document.getElementById("datasetFrame").addEventListener("input", loadDatasetFrame);
    document.getElementById("datasetFps").addEventListener("change", () => {{ if (datasetTimer) startDatasetPlayback(); }});
    document.getElementById("datasetPlay").addEventListener("click", () => {{
      if (datasetTimer) stopDatasetPlayback();
      else startDatasetPlayback();
    }});
    document.getElementById("datasetPrev").addEventListener("click", () => {{
      const frame = document.getElementById("datasetFrame");
      frame.value = String(Math.max(0, Number(frame.value) - 1));
      loadDatasetFrame();
    }});
    document.getElementById("datasetNext").addEventListener("click", () => {{
      const frame = document.getElementById("datasetFrame");
      frame.value = String(Math.min(Number(frame.max), Number(frame.value) + 1));
      loadDatasetFrame();
    }});
    function renderLossChart(metrics, validationMetrics) {{
      const svg = document.getElementById("lossChart");
      const w = 640, h = 260, pad = 34;
      svg.innerHTML = "";
      const rows = metrics.filter(row => Number.isFinite(Number(row.step)) && Number.isFinite(Number(row.loss)));
      const valRows = validationMetrics.filter(row => Number.isFinite(Number(row.step)) && Number.isFinite(Number(row.loss)));
      if (!rows.length) return;
      const allRows = rows.concat(valRows);
      const values = allRows.map(row => Number(row.loss));
      const minY = Math.max(0, Math.min(...values) * 0.85);
      const maxY = Math.max(...values) * 1.12;
      const minX = Math.min(...allRows.map(row => Number(row.step)));
      const maxX = Math.max(...allRows.map(row => Number(row.step)));
      const sx = x => pad + (Number(x) - minX) / Math.max(1, maxX - minX) * (w - pad * 2);
      const sy = y => h - pad - (Number(y) - minY) / Math.max(1e-9, maxY - minY) * (h - pad * 2);
      const make = (name, attrs) => {{
        const node = document.createElementNS("http://www.w3.org/2000/svg", name);
        Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, String(v)));
        svg.appendChild(node);
        return node;
      }};
      [0, 0.25, 0.5, 0.75, 1].forEach(t => {{
        const y = pad + t * (h - pad * 2);
        make("line", {{ x1: pad, y1: y, x2: w - pad, y2: y, stroke: "#e5e7eb", "stroke-width": 1 }});
      }});
      make("line", {{ x1: pad, y1: pad, x2: pad, y2: h - pad, stroke: "#98a2b3", "stroke-width": 1 }});
      make("line", {{ x1: pad, y1: h - pad, x2: w - pad, y2: h - pad, stroke: "#98a2b3", "stroke-width": 1 }});
      const path = rows.map((row, i) => `${{i ? "L" : "M"}} ${{sx(row.step).toFixed(2)}} ${{sy(row.loss).toFixed(2)}}`).join(" ");
      make("path", {{ d: path, fill: "none", stroke: "#2563eb", "stroke-width": 3, "stroke-linecap": "round", "stroke-linejoin": "round" }});
      rows.forEach(row => make("circle", {{ cx: sx(row.step), cy: sy(row.loss), r: 3, fill: "#2563eb" }}));
      if (valRows.length) {{
        const valPath = valRows.map((row, i) => `${{i ? "L" : "M"}} ${{sx(row.step).toFixed(2)}} ${{sy(row.loss).toFixed(2)}}`).join(" ");
        if (valRows.length > 1) {{
          make("path", {{ d: valPath, fill: "none", stroke: "#dc2626", "stroke-width": 3, "stroke-linecap": "round", "stroke-linejoin": "round" }});
        }}
        valRows.forEach(row => make("circle", {{ cx: sx(row.step), cy: sy(row.loss), r: 5, fill: "#dc2626" }}));
      }}
      make("text", {{ x: pad, y: 18, fill: "#475467", "font-size": 12 }}).textContent = `max ${{maxY.toFixed(3)}}`;
      make("text", {{ x: pad, y: h - 8, fill: "#475467", "font-size": 12 }}).textContent = `min ${{minY.toFixed(3)}}`;
      make("text", {{ x: w - pad - 72, y: h - 8, fill: "#475467", "font-size": 12 }}).textContent = `step ${{maxX}}`;
    }}
    function renderSystem(system) {{
      const mem = system.memory || {{}};
      const swap = mem.swap || {{}};
      const train = (system.training_processes || []).find(row => row.role === "train") || {{}};
      const cells = [
        ["Memory Used", mem.used_gb === undefined ? "-" : `${{mem.used_gb}} GB`],
        ["Physical", mem.physical_gb === undefined ? "-" : `${{mem.physical_gb}} GB`],
        ["Swap Used", swap.used_gb === undefined ? "-" : `${{swap.used_gb}} GB`],
        ["Train RSS", train.rss_gb === undefined ? "-" : `${{train.rss_gb}} GB`],
        ["Train CPU", train.cpu_percent === undefined ? "-" : `${{train.cpu_percent}}%`],
        ["Train Mem", train.mem_percent === undefined ? "-" : `${{train.mem_percent}}%`],
        ["Wired", mem.wired_gb === undefined ? "-" : `${{mem.wired_gb}} GB`],
        ["Compressed", mem.compressed_gb === undefined ? "-" : `${{mem.compressed_gb}} GB`],
      ];
      document.getElementById("systemKv").innerHTML = cells.map(([k, v]) =>
        `<div><span class="label">${{k}}</span><strong>${{v}}</strong></div>`
      ).join("");
      document.getElementById("systemRaw").textContent = JSON.stringify(system, null, 2);
    }}
    refresh();
    initDataset();
    setInterval(refresh, 10000);
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
