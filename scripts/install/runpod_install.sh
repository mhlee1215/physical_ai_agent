#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
COMPONENT="libero-smolvla"

usage() {
  cat <<'EOF'
Usage: sh scripts/install/runpod_install.sh [--component libero-smolvla|risk1b-vlm|libero-config]

Unified RunPod install/bootstrap entrypoint. Existing RunPod setup script names
remain as compatibility shims or lower-level recipes.
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
    exec sh "$SCRIPT_DIR/runpod_prepare_libero_smolvla_env.sh"
    ;;
  risk1b-vlm)
    exec sh "$SCRIPT_DIR/bootstrap_runpod_risk1b_vlm_env.sh"
    ;;
  libero-config)
    exec sh "$SCRIPT_DIR/runpod_prepare_libero_config.sh"
    ;;
  *)
    echo "unknown component: $COMPONENT" >&2
    usage >&2
    exit 2
    ;;
esac
