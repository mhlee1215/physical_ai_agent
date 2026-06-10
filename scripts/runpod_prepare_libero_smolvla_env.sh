#!/bin/sh
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec sh "$SCRIPT_DIR/install/runpod_prepare_libero_smolvla_env.sh" "$@"
