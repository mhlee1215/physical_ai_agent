#!/bin/sh
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec sh "$SCRIPT_DIR/install/runpod_check_libero_env.sh" "$@"
