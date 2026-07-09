#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.physical-ai-agent.dataset-viewer"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="$ROOT/_workspace/logs"
HOST="${SO101_DATASET_VIEWER_HOST:-0.0.0.0}"
PORT="${SO101_DATASET_VIEWER_PORT:-8768}"
PYTHON_BIN="${SO101_DATASET_VIEWER_PYTHON:-$ROOT/.venv/bin/python}"
UID_VALUE="$(id -u)"

write_plist() {
  mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
  /usr/bin/env python3 - "$PLIST" "$ROOT" "$PYTHON_BIN" "$HOST" "$PORT" "$LOG_DIR" <<'PY'
import plistlib
import sys
from pathlib import Path

plist_path, root, python_bin, host, port, log_dir = sys.argv[1:]
payload = {
    "Label": "com.physical-ai-agent.dataset-viewer",
    "ProgramArguments": [
        python_bin,
        f"{root}/scripts/serve_so101_dataset_viewer.py",
        "--host",
        host,
        "--port",
        str(port),
    ],
    "WorkingDirectory": root,
    "EnvironmentVariables": {
        "PYTHONPATH": "src",
    },
    "RunAtLoad": True,
    "KeepAlive": True,
    "StandardOutPath": f"{log_dir}/dataset_viewer_8768.launchd.log",
    "StandardErrorPath": f"{log_dir}/dataset_viewer_8768.launchd.err.log",
}
Path(plist_path).write_bytes(plistlib.dumps(payload, sort_keys=False))
PY
  xattr -c "$PLIST" 2>/dev/null || true
  chmod 644 "$PLIST"
}

bootout() {
  launchctl bootout "gui/$UID_VALUE/$LABEL" 2>/dev/null || true
}

start() {
  write_plist
  bootout
  launchctl bootstrap "gui/$UID_VALUE" "$PLIST"
  launchctl kickstart -k "gui/$UID_VALUE/$LABEL"
}

stop() {
  bootout
}

status() {
  launchctl print "gui/$UID_VALUE/$LABEL" 2>/dev/null | sed -n '1,90p' || true
  echo "--- port ---"
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN || true
  echo "--- api ---"
  curl -fsS "http://127.0.0.1:$PORT/api/datasets" >/dev/null && echo "DATASET_VIEWER_OK" || echo "DATASET_VIEWER_DOWN"
}

case "${1:-restart}" in
  start)
    start
    status
    ;;
  stop)
    stop
    ;;
  restart)
    start
    sleep 2
    status
    ;;
  status)
    status
    ;;
  *)
    echo "usage: $0 [start|stop|restart|status]" >&2
    exit 2
    ;;
esac
