#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.physical-ai-agent.dataset-viewer"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="$ROOT/_workspace/logs"
TUNNEL_LOG="${SO101_DATASET_VIEWER_TUNNEL_LOG:-$LOG_DIR/dataset_viewer_8768.tunnel.log}"
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
  launchctl bootout "gui/$UID_VALUE/$LABEL" 2>/dev/null \
    || launchctl bootout "gui/$UID_VALUE" "$PLIST" 2>/dev/null \
    || true
  for _ in $(seq 1 50); do
    if ! launchctl print "gui/$UID_VALUE/$LABEL" >/dev/null 2>&1; then
      return
    fi
    sleep 0.1
  done
}

bootstrap() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if launchctl bootstrap "gui/$UID_VALUE" "$PLIST"; then
      return
    fi
    sleep 1
  done
  echo "failed to bootstrap $LABEL after 5 attempts" >&2
  return 1
}

start() {
  write_plist
  bootout
  bootstrap
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
  links
}

links() {
  local iface=""
  local lan_ip=""
  local external_url=""

  iface="$(route -n get default 2>/dev/null | awk '/interface:/{print $2; exit}')"
  if [[ -n "$iface" ]]; then
    lan_ip="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
  fi
  if [[ -f "$TUNNEL_LOG" ]]; then
    external_url="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | tail -1 || true)"
  fi

  echo "--- links ---"
  echo "LOCAL_URL=http://127.0.0.1:$PORT/"
  if [[ -n "$lan_ip" ]]; then
    echo "MOBILE_URL=http://$lan_ip:$PORT/"
  else
    echo "MOBILE_URL=unavailable"
  fi
  if [[ -n "$external_url" ]] && curl -fsS --max-time 15 "$external_url/api/datasets" >/dev/null; then
    echo "EXTERNAL_URL=$external_url/"
  else
    echo "EXTERNAL_URL=unavailable"
  fi
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
  links)
    links
    ;;
  *)
    echo "usage: $0 [start|stop|restart|status|links]" >&2
    exit 2
    ;;
esac
