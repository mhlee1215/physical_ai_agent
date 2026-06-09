#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$SCRIPT_DIR/eval_smolvla_lerobot_linux.sh" --benchmark metaworld "$@"
