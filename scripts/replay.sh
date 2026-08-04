#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: scripts/replay.sh <config.json>" >&2
  exit 2
fi

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  exec "$ROOT/.venv/bin/python" -m physical_ai_agent.real_so100.dataset_episode_replay "$1"
