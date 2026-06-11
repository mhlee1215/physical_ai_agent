#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)}"
MODE="syntax"

usage() {
  cat <<'EOF'
Usage: sh scripts/install/local_check.sh [--mode syntax|venv|all]

Unified local install/check entrypoint. It is dependency-light by default.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="${2:?missing value for --mode}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

check_syntax() {
  find "$PROJECT_DIR/scripts/install" -type f -name "*.sh" | sort | while IFS= read -r path; do
    sh -n "$path"
  done
}

check_venv() {
  if [ ! -x "$PROJECT_DIR/.venv/bin/python" ]; then
    echo "local .venv is missing; run scripts/install/local_install.sh when dependencies are required" >&2
    exit 1
  fi
  "$PROJECT_DIR/.venv/bin/python" -V
}

case "$MODE" in
  syntax)
    check_syntax
    ;;
  venv)
    check_venv
    ;;
  all)
    check_syntax
    check_venv
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
