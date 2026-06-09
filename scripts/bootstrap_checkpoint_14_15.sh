#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

sh scripts/bootstrap_checkpoint_07_13.sh

if [ -x ".venv/bin/python" ]; then
  .venv/bin/python -m pip install "lerobot[smolvla]>=0.5.1,<0.6"
else
  python3 -m pip install "lerobot[smolvla]>=0.5.1,<0.6"
fi
