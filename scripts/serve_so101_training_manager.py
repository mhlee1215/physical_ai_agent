#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_ROOT = Path("_workspace/so101_training")
DEFAULT_LOCK = DEFAULT_ROOT / "active_local_training.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve local SO101 training history by stable training_id.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.repo_root.resolve()))
    print(f"[training-manager] serving http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


def make_handler(repo_root: Path):
    class TrainingManagerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_html(_index_html())
                elif parsed.path == "/api/runs":
                    self._send_json(_runs_payload(repo_root))
                elif parsed.path == "/api/run":
                    query = parse_qs(parsed.query)
                    training_id = (query.get("id") or [""])[0]
                    if not training_id:
                        raise ValueError("missing id")
                    self._send_json(_run_detail(repo_root, training_id))
                else:
                    self.send_error(404)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[training-manager] {self.address_string()} {format % args}", flush=True)

        def _send_html(self, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return TrainingManagerHandler


def _runs_payload(repo_root: Path) -> dict[str, Any]:
    active = _active_record(repo_root)
    active_run_dir = Path(str(active.get("run_dir"))).resolve() if active.get("run_dir") else None
    runs = []
    seen: set[str] = set()
    for summary_path in _summary_paths(repo_root):
        summary = _read_json(summary_path)
        if not isinstance(summary, dict):
            continue
        row = _run_row(repo_root, summary_path, summary, active_run_dir=active_run_dir)
        seen.add(row["training_id"])
        runs.append(row)
    for row in _registry_rows(repo_root):
        training_id = str(row.get("training_id") or "")
        if not training_id or training_id in seen:
            continue
        run_dir = Path(str(row.get("run_dir") or "")).resolve() if row.get("run_dir") else None
        row_payload = {
            "training_id": training_id,
            "run_dir": str(run_dir) if run_dir else row.get("run_dir"),
            "dataset_config_name": row.get("dataset_config_name"),
            "started_at_utc": row.get("started_at_utc"),
            "written_at_utc": None,
            "active": bool(active_run_dir and run_dir and run_dir == active_run_dir),
            "summary_path": row.get("training_run_summary_path"),
            "tensorboard_url": row.get("tensorboard_url"),
            "mobile_tensorboard_url": row.get("mobile_tensorboard_url"),
            "latest_train_loss": None,
            "latest_val_loss": None,
            "latest_closed_loop": None,
            "checkpoint_count": 0,
        }
        runs.append(row_payload)
    runs.sort(key=lambda row: str(row.get("started_at_utc") or row.get("written_at_utc") or ""), reverse=True)
    return {"active_training_id": _active_training_id(active), "runs": runs}


def _delete_runs(repo_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    training_ids = _unique_training_ids(payload.get("training_ids"))
    confirmation = str(payload.get("confirmation") or "").strip()
    expected_confirmation = f"DELETE {len(training_ids)} TRAINING RUNS"
    if confirmation != expected_confirmation:
        raise ValueError(
            f"bulk deletion confirmation must exactly match: {expected_confirmation}"
        )

    records = _training_run_records(repo_root)
    missing = [training_id for training_id in training_ids if training_id not in records]
    if missing:
        raise ValueError(f"unknown training runs: {', '.join(missing)}")

    active = _active_record(repo_root)
    active_id = _active_training_id(active)
    active_run_dir = (
        Path(str(active.get("run_dir"))).resolve()
        if active.get("run_dir")
        else None
    )
    runs_root = (repo_root / DEFAULT_ROOT / "runs").resolve()
    selected_ids = set(training_ids)
    target_dirs: dict[Path, dict[str, Any]] = {}
    ids_by_dir: dict[Path, set[str]] = {}
    for training_id, run_dirs in records.items():
        for run_dir in run_dirs:
            ids_by_dir.setdefault(run_dir, set()).add(training_id)

    for training_id in training_ids:
        run_dirs = records[training_id]
        if len(run_dirs) != 1:
            raise ValueError(
                f"training run id is ambiguous: {training_id} resolves to {len(run_dirs)} directories"
            )
        run_dir = next(iter(run_dirs))
        if training_id == active_id or (active_run_dir is not None and run_dir == active_run_dir):
            raise PermissionError(f"refusing to delete active training run: {training_id}")
        try:
            run_dir.relative_to(runs_root)
        except ValueError as exc:
            raise PermissionError(
                f"refusing to delete a training run outside {runs_root}: {run_dir}"
            ) from exc
        if run_dir == runs_root:
            raise PermissionError("refusing to delete the training runs root")
        aliases = ids_by_dir.get(run_dir, set()) - selected_ids
        if aliases:
            raise ValueError(
                f"training run directory is also registered as: {', '.join(sorted(aliases))}"
            )
        if run_dir.exists():
            if not run_dir.is_dir():
                raise ValueError(f"training run path is not a directory: {run_dir}")
            target_dirs[run_dir] = {
                "run_dir": str(run_dir),
                "size_bytes": _directory_size(run_dir),
            }

    selected_dirs = set(target_dirs)
    for run_dir in selected_dirs:
        for other_id, other_dirs in records.items():
            if other_id in selected_ids:
                continue
            for other_dir in other_dirs:
                if other_dir == run_dir:
                    continue
                try:
                    other_dir.relative_to(run_dir)
                except ValueError:
                    continue
                raise ValueError(
                    f"refusing to delete {run_dir}: it contains unselected run {other_id}"
                )

    total_size = sum(int(item["size_bytes"]) for item in target_dirs.values())
    for run_dir in sorted(target_dirs, key=lambda path: len(path.parts), reverse=True):
        shutil.rmtree(run_dir)
    _remove_training_registry_rows(
        repo_root,
        selected_ids=selected_ids,
        selected_dirs=selected_dirs,
    )
    return {
        "status": "deleted",
        "deleted_training_ids": training_ids,
        "deleted_directories": list(target_dirs.values()),
        "size_bytes": total_size,
        "size_human": _format_bytes(total_size),
    }


def _unique_training_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("training_ids must be a non-empty list")
    training_ids = sorted({str(item).strip() for item in value if str(item).strip()})
    if not training_ids:
        raise ValueError("training_ids must be a non-empty list")
    return training_ids


def _training_run_records(repo_root: Path) -> dict[str, set[Path]]:
    records: dict[str, set[Path]] = {}
    for summary_path in _summary_paths(repo_root):
        summary = _read_json(summary_path)
        if not isinstance(summary, dict):
            continue
        training_id = _summary_training_id(summary_path, summary)
        run_dir = Path(str(summary.get("run_dir") or summary_path.parent)).resolve()
        records.setdefault(training_id, set()).add(run_dir)
    for row in _registry_rows(repo_root):
        training_id = str(row.get("training_id") or "").strip()
        run_dir = str(row.get("run_dir") or "").strip()
        if training_id and run_dir:
            records.setdefault(training_id, set()).add(Path(run_dir).resolve())
    return records


def _remove_training_registry_rows(
    repo_root: Path,
    *,
    selected_ids: set[str],
    selected_dirs: set[Path],
) -> None:
    path = repo_root / DEFAULT_ROOT / "training_runs_index.json"
    payload = _read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        return
    retained = []
    for row in payload["runs"]:
        if not isinstance(row, dict):
            retained.append(row)
            continue
        training_id = str(row.get("training_id") or "")
        run_dir = Path(str(row.get("run_dir"))).resolve() if row.get("run_dir") else None
        if training_id in selected_ids or (run_dir is not None and run_dir in selected_dirs):
            continue
        retained.append(row)
    payload["runs"] = retained
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _directory_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _run_detail(repo_root: Path, training_id: str) -> dict[str, Any]:
    for summary_path in _summary_paths(repo_root):
        summary = _read_json(summary_path)
        if not isinstance(summary, dict):
            continue
        if _summary_training_id(summary_path, summary) == training_id:
            run_dir = Path(str(summary.get("run_dir") or summary_path.parent)).resolve()
            return {
                "training_id": training_id,
                "summary_path": str(summary_path),
                "summary": summary,
                "status": _status_for_run(repo_root, run_dir),
                "metrics": _metrics_payload(run_dir),
                "logs": _logs_payload(run_dir),
                "paths": {
                    "run_dir": str(run_dir),
                    "tensorboard": str(run_dir / "tensorboard"),
                    "checkpoints": str(_checkpoint_root(run_dir)),
                    "closed_loop_evals": str(run_dir / "closed_loop_evals"),
                },
            }
    raise ValueError(f"unknown training_id: {training_id}")


def _run_row(repo_root: Path, summary_path: Path, summary: dict[str, Any], *, active_run_dir: Path | None) -> dict[str, Any]:
    del repo_root
    run_dir = Path(str(summary.get("run_dir") or summary_path.parent)).resolve()
    metrics = _metrics_payload(run_dir)
    training_id = _summary_training_id(summary_path, summary)
    return {
        "training_id": training_id,
        "run_dir": str(run_dir),
        "dataset_config_name": _dataset_config_name(summary),
        "task": _dataset_config_value(summary, "task"),
        "started_at_utc": summary.get("started_at_utc"),
        "written_at_utc": summary.get("written_at_utc"),
        "active": bool(active_run_dir and run_dir == active_run_dir),
        "summary_path": str(summary_path),
        "tensorboard_url": summary.get("tensorboard_url"),
        "mobile_tensorboard_url": summary.get("mobile_tensorboard_url"),
        "latest_train_loss": _latest_metric(metrics["training"], "loss"),
        "latest_val_loss": _latest_metric(metrics["validation"], "loss"),
        "latest_closed_loop": metrics["closed_loop"][-1] if metrics["closed_loop"] else None,
        "checkpoint_count": len(_checkpoint_names(run_dir)),
    }


def _metrics_payload(run_dir: Path) -> dict[str, Any]:
    return {
        "training": _read_jsonl(run_dir / "metrics" / "training_metrics.jsonl"),
        "validation": _read_jsonl(run_dir / "metrics" / "validation_metrics.jsonl"),
        "closed_loop": _read_jsonl(run_dir / "metrics" / "closed_loop_metrics.jsonl"),
        "monitor_events": _read_jsonl(run_dir / "metrics" / "monitor_events.jsonl")[-50:],
        "loss_summary": _read_json(run_dir / "metrics" / "loss_summary.json") or {},
        "checkpoints": _checkpoint_names(run_dir),
    }


def _logs_payload(run_dir: Path) -> dict[str, Any]:
    return {
        "train_tail": _tail_text(run_dir / "logs" / "train.log", max_lines=80),
        "tensorboard_tail": _tail_text(run_dir / "logs" / "tensorboard.log", max_lines=40),
    }


def _status_for_run(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    active = _active_record(repo_root)
    active_run_dir = Path(str(active.get("run_dir"))).resolve() if active.get("run_dir") else None
    if active_run_dir and active_run_dir == run_dir:
        return {
            "active": True,
            "train": active.get("train"),
            "tensorboard": active.get("tensorboard"),
            "tensorboard_url": active.get("tensorboard_url"),
            "mobile_tensorboard_url": active.get("mobile_tensorboard_url"),
        }
    return {"active": False}


def _summary_paths(repo_root: Path) -> list[Path]:
    root = repo_root / DEFAULT_ROOT / "runs"
    return sorted(root.glob("**/training_run_summary.json"))


def _registry_rows(repo_root: Path) -> list[dict[str, Any]]:
    payload = _read_json(repo_root / DEFAULT_ROOT / "training_runs_index.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        return []
    return [row for row in payload["runs"] if isinstance(row, dict)]


def _active_record(repo_root: Path) -> dict[str, Any]:
    for path in (repo_root / DEFAULT_LOCK, repo_root / DEFAULT_ROOT / "active_training.json"):
        payload = _read_json(path)
        if isinstance(payload, dict) and payload:
            return _with_process_status(payload)
    return {}


def _active_training_id(active: dict[str, Any]) -> str | None:
    if not active:
        return None
    if active.get("training_id"):
        return str(active["training_id"])
    run_dir = active.get("run_dir")
    return _slug(Path(str(run_dir)).name) if run_dir else None


def _summary_training_id(summary_path: Path, summary: dict[str, Any]) -> str:
    if summary.get("training_id"):
        return str(summary["training_id"])
    run_dir = summary.get("run_dir")
    if run_dir:
        return _slug(Path(str(run_dir)).name)
    return _slug(summary_path.parent.name)


def _dataset_config_name(summary: dict[str, Any]) -> str | None:
    return _dataset_config_value(summary, "name")


def _dataset_config_value(summary: dict[str, Any], key: str) -> str | None:
    dataset_config = summary.get("dataset_config")
    if isinstance(dataset_config, dict) and dataset_config.get(key) is not None:
        return str(dataset_config[key])
    return None


def _latest_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _checkpoint_root(run_dir: Path) -> Path:
    for candidate in (run_dir / "model" / "checkpoints", run_dir / "checkpoints"):
        if candidate.exists():
            return candidate
    return run_dir / "model" / "checkpoints"


def _checkpoint_names(run_dir: Path) -> list[str]:
    root = _checkpoint_root(run_dir)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _with_process_status(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    updated["train"] = _process_status(record.get("train_pid"))
    updated["tensorboard"] = _process_status(record.get("tensorboard_pid"))
    return updated


def _process_status(pid: Any) -> dict[str, Any]:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return {"alive": False, "pid": None}
    alive = False
    try:
        os.kill(pid_int, 0)
        alive = True
    except OSError:
        alive = False
    return {"alive": alive, "pid": pid_int}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except Exception:
        return rows
    return rows


def _tail_text(path: Path, *, max_lines: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


def _slug(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return text.strip("._-") or "training"


def _index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SO101 Training Manager</title>
  <style>
    :root { color-scheme: light; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f7fb; color: #111827; }
    header { padding: 20px 24px; border-bottom: 1px solid #d8dee9; background: #fff; }
    h1 { margin: 0; font-size: 28px; }
    main { display: grid; grid-template-columns: minmax(320px, 430px) 1fr; gap: 16px; padding: 16px; }
    section { background: #fff; border: 1px solid #d8dee9; border-radius: 8px; overflow: hidden; }
    .section-title { padding: 12px 14px; font-weight: 700; border-bottom: 1px solid #e5e7eb; }
    .run-list { display: flex; flex-direction: column; }
    button.run { border: 0; background: #fff; text-align: left; padding: 12px 14px; border-bottom: 1px solid #edf0f5; cursor: pointer; }
    button.run:hover, button.run.active { background: #eef6ff; }
    .id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 700; font-size: 13px; overflow-wrap: anywhere; }
    .meta { color: #5b6475; font-size: 12px; margin-top: 4px; }
    .pill { display: inline-block; padding: 2px 7px; border-radius: 999px; background: #e8f7ee; color: #166534; font-size: 12px; margin-left: 6px; }
    .detail { padding: 14px; display: grid; gap: 14px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; }
    .metric { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }
    .metric-label { color: #6b7280; font-size: 12px; }
    .metric-value { font-size: 18px; font-weight: 700; margin-top: 3px; }
    pre { margin: 0; padding: 12px; background: #0f172a; color: #e5e7eb; border-radius: 8px; overflow: auto; max-height: 420px; }
    details { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }
    summary { cursor: pointer; font-weight: 700; }
    a { color: #0f66c2; text-decoration: none; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } .grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); } }
  </style>
</head>
<body>
  <header><h1>SO101 Training Manager</h1></header>
  <main>
    <section>
      <div class="section-title">Training IDs</div>
      <div id="runs" class="run-list"></div>
    </section>
    <section>
      <div class="section-title">Details</div>
      <div id="detail" class="detail">Select a training ID.</div>
    </section>
  </main>
<script>
const runsEl = document.getElementById("runs");
const detailEl = document.getElementById("detail");
let selectedId = null;

async function loadRuns() {
  const payload = await (await fetch("/api/runs")).json();
  runsEl.innerHTML = "";
  for (const run of payload.runs) {
    const btn = document.createElement("button");
    btn.className = "run" + (run.training_id === selectedId ? " active" : "");
    btn.innerHTML = `
      <div class="id">${escapeHtml(run.training_id)}${run.active ? '<span class="pill">active</span>' : ''}</div>
      <div class="meta">${escapeHtml(run.dataset_config_name || "")}</div>
      <div class="meta">train ${fmt(run.latest_train_loss)} · val ${fmt(run.latest_val_loss)} · checkpoints ${run.checkpoint_count || 0}</div>
    `;
    btn.onclick = () => selectRun(run.training_id);
    runsEl.appendChild(btn);
  }
  if (!selectedId && payload.active_training_id) selectRun(payload.active_training_id);
}

async function selectRun(id) {
  selectedId = id;
  document.querySelectorAll("button.run").forEach(btn => btn.classList.toggle("active", btn.textContent.includes(id)));
  const payload = await (await fetch(`/api/run?id=${encodeURIComponent(id)}`)).json();
  if (payload.error) {
    detailEl.textContent = payload.error;
    return;
  }
  const s = payload.summary || {};
  const d = s.dataset_config || {};
  const m = payload.metrics || {};
  const train = m.training || [];
  const val = m.validation || [];
  const closed = m.closed_loop || [];
  const latestClosed = closed[closed.length - 1] || {};
  const status = payload.status || {};
  detailEl.innerHTML = `
    <div>
      <div class="id">${escapeHtml(payload.training_id)}${status.active ? '<span class="pill">active</span>' : ''}</div>
      <div class="meta">${escapeHtml(payload.paths.run_dir)}</div>
      <div class="meta">${link(s.tensorboard_url, "TensorBoard")} ${link(s.mobile_tensorboard_url, "Mobile TensorBoard")}</div>
    </div>
    <div class="grid">
      ${metric("Dataset", d.name || "")}
      ${metric("Train Loss", fmt(lastValue(train, "loss")))}
      ${metric("Val Loss", fmt(lastValue(val, "loss")))}
      ${metric("Closed Loop", latestClosed.test_id ? `${latestClosed.test_id}: ${fmt(latestClosed.success_rate)}` : "n/a")}
      ${metric("Checkpoints", (m.checkpoints || []).length)}
      ${metric("Train Rows", train.length)}
      ${metric("Val Rows", val.length)}
      ${metric("Loop Rows", closed.length)}
    </div>
    <details open><summary>Run Identity</summary><pre>${escapeHtml(JSON.stringify(identityPayload(payload), null, 2))}</pre></details>
    <details><summary>Dataset Config</summary><pre>${escapeHtml(JSON.stringify(d, null, 2))}</pre></details>
    <details><summary>Training Command</summary><pre>${escapeHtml((s.train_cmd || []).join(" \\n"))}</pre></details>
    <details><summary>Metrics</summary><pre>${escapeHtml(JSON.stringify(m, null, 2))}</pre></details>
    <details><summary>Train Log Tail</summary><pre>${escapeHtml((payload.logs || {}).train_tail || "")}</pre></details>
  `;
}

function identityPayload(payload) {
  const s = payload.summary || {};
  return {
    training_id: payload.training_id,
    active: (payload.status || {}).active || false,
    run_dir: (payload.paths || {}).run_dir,
    summary_path: payload.summary_path,
    started_at_utc: s.started_at_utc,
    written_at_utc: s.written_at_utc,
    tensorboard_url: s.tensorboard_url,
    mobile_tensorboard_url: s.mobile_tensorboard_url,
  };
}
function lastValue(rows, key) {
  for (let i = rows.length - 1; i >= 0; i--) if (typeof rows[i][key] === "number") return rows[i][key];
  return null;
}
function metric(label, value) { return `<div class="metric"><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value">${escapeHtml(String(value))}</div></div>`; }
function fmt(value) { return typeof value === "number" ? value.toFixed(5) : "n/a"; }
function link(url, text) { return url ? `<a href="${escapeHtml(url)}" target="_blank">${escapeHtml(text)}</a>` : ""; }
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
}
loadRuns();
setInterval(loadRuns, 10000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
