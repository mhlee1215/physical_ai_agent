#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

if [ "$(uname -s)" = "Darwin" ]; then
  if [ -x ".venv/bin/mjpython" ]; then
    if PYTHONPATH=src .venv/bin/mjpython -c "print('mjpython-ok')" >/dev/null 2>&1; then
      PYTHONPATH=src .venv/bin/mjpython -B -m physical_ai_agent.sim.so101_live_viewer "$@"
      exit 0
    fi
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

For non-live 3D output with the current venv, use:

  sh scripts/checkpoint_14_15.sh --allow-download --require-3d-render --require-real-smolvla
EOF
  exit 1
fi

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PHYSICAL_AI_PYTHON:-python3}"
fi

PYTHONPATH=src "$PYTHON_BIN" -B -m physical_ai_agent.sim.so101_live_viewer "$@"
