#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
COMPONENT="libero-smolvla"

usage() {
  cat <<'EOF'
Usage: sh scripts/install/runpod_check.sh [--component libero-smolvla|risk1b-vlm|all]

Unified RunPod install/check entrypoint.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --component)
      COMPONENT="${2:?missing value for --component}"
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

case "$COMPONENT" in
  libero-smolvla)
    exec sh "$SCRIPT_DIR/recipes/runpod_check_libero_env.sh"
    ;;
  risk1b-vlm)
    exec sh "$SCRIPT_DIR/recipes/runpod_check_risk1b_vlm_env.sh"
    ;;
  all)
    sh "$SCRIPT_DIR/recipes/runpod_check_libero_env.sh"
    sh "$SCRIPT_DIR/recipes/runpod_check_risk1b_vlm_env.sh"
    ;;
  *)
    echo "unknown component: $COMPONENT" >&2
    usage >&2
    exit 2
    ;;
esac
