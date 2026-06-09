#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_DIR="$(CDPATH= cd -- "$ROOT_DIR/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv-xvla-mps-py312}"
PYTHON_BIN="${PYTHON_BIN:-/Users/minhaeng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"

"$VENV_DIR/bin/python" - <<'PY'
import torch

print("torch", torch.__version__)
print("mps_built", torch.backends.mps.is_built())
print("mps_available", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    print(torch.ones(1, device="mps"))
PY
