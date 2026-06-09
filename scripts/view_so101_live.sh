#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

PYTHON_BIN="${PHYSICAL_AI_PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

for arg in "$@"; do
  if [ "$arg" = "--browser-only" ]; then
    PYTHONPATH=src "$PYTHON_BIN" -B -m physical_ai_agent.sim.so101_live_viewer "$@"
    exit 0
  fi
done

if [ "$(uname -s)" = "Darwin" ]; then
  if [ -x ".venv/bin/mjpython" ]; then
    PYTHONPATH=src .venv/bin/mjpython -B -m physical_ai_agent.sim.so101_live_viewer "$@"
    exit 0
  fi
  cat >&2 <<'EOF'
SO101 live viewer on macOS requires MuJoCo's `mjpython`.

The repo's current `.venv/bin/mjpython` is not runnable. This usually happens
when the venv was created from a bundled/symlinked Python that does not expose
`libpython*.dylib` for MuJoCo's Cocoa viewer trampoline.

Fix from a normal macOS Terminal with a Homebrew or python.org Python:

  cd /Users/minhaeng/workspace/physical_ai_agent
  /opt/homebrew/bin/python3 -m venv .venv-viewer
  .venv-viewer/bin/python -m pip install -e ".[so101]"
  PYTHONPATH=src .venv-viewer/bin/mjpython -B -m physical_ai_agent.sim.so101_live_viewer

If `/opt/homebrew/bin/python3` does not exist, install Python with Homebrew or
python.org, then use that interpreter to create `.venv-viewer`.

For a headless browser-streamed smoke with the current venv, use:

  sh scripts/view_so101_live.sh --browser-only --show-inputs --fps 2 --max-steps 1
EOF
  exit 1
fi

PYTHONPATH=src "$PYTHON_BIN" -B -m physical_ai_agent.sim.so101_live_viewer "$@"
