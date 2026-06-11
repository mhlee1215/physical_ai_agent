#!/bin/sh
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec sh "$SCRIPT_DIR/install/recipes/bootstrap_runpod_risk1b_vlm_env.sh" "$@"
