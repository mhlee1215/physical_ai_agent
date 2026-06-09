#!/bin/sh
set -eu

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install -e ".[maniskill,smolvla]"
