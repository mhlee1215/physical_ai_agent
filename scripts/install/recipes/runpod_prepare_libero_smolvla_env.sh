#!/bin/sh
set -eu

# RunPod LIBERO + SmolVLA environment operator.
#
# This is the script agents should call before LIBERO diagnostics, probes, or
# benchmark runs. It separates hardware/storage policy from the package recipe:
#
# - profile=volume: keep env/cache under a persistent network volume and make
#   /root/physical-ai point at it so historical commands keep working.
# - profile=ephemeral: keep env under /root/physical-ai for throwaway Pods.
#
# A venv is never considered usable because the directory exists. The script
# first runs scripts/install/runpod_check.sh --component libero-smolvla. If the gate fails, it builds a
# fresh temporary venv, runs the bootstrap recipe against that temporary path,
# runs the hard gate again, and only then publishes it as the final venv.

PROFILE="${RUNPOD_ENV_PROFILE:-auto}"
PERSISTENT_WORK_ROOT="${RUNPOD_PERSISTENT_WORK_ROOT:-/workspace/physical-ai}"
EPHEMERAL_WORK_ROOT="${RUNPOD_EPHEMERAL_WORK_ROOT:-/root/physical-ai}"
PROJECT_DIR="${PROJECT_DIR:-}"
VENV_NAME="${RUNPOD_VENV_NAME:-lerobot_py312}"
FORCE_REBUILD="${RUNPOD_FORCE_REBUILD:-0}"
ALLOW_ROOT_REPLACE="${RUNPOD_ALLOW_ROOT_PHYSICAL_AI_REPLACE:-0}"

log() {
  printf '[runpod-libero-env-prepare] %s\n' "$*"
}

now_sec() {
  date -u +%s
}

timer_start() {
  stage="$1"
  eval "TIMER_${stage}=$(now_sec)"
  printf '[bootstrap-timer] %s start epoch_sec=%s\n' "$stage" "$(now_sec)"
}

timer_end() {
  stage="$1"
  end="$(now_sec)"
  eval "start=\${TIMER_${stage}:-$end}"
  duration=$((end - start))
  printf '[bootstrap-timer] %s end epoch_sec=%s duration_sec=%s\n' "$stage" "$end" "$duration"
}

timed_run() {
  stage="$1"
  shift
  timer_start "$stage"
  set +e
  "$@"
  status=$?
  set -e
  timer_end "$stage"
  return "$status"
}

die() {
  echo "[runpod-libero-env-prepare] error: $*" >&2
  exit 1
}

require_linux() {
  if [ "$(uname -s)" != "Linux" ]; then
    die "intended for Linux/RunPod, not $(uname -s)"
  fi
}

choose_profile() {
  if [ "$PROFILE" != "auto" ]; then
    return
  fi
  if [ -d /workspace ] && command -v df >/dev/null 2>&1; then
    PROFILE="volume"
  else
    PROFILE="ephemeral"
  fi
}

resolve_project_dir() {
  if [ -n "$PROJECT_DIR" ]; then
    return
  fi
  PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
}

prepare_paths() {
  case "$PROFILE" in
    volume)
      WORK_ROOT="$PERSISTENT_WORK_ROOT"
      mkdir -p "$WORK_ROOT" "$WORK_ROOT/envs"
      if [ -e /root/physical-ai ] && [ "$(readlink -f /root/physical-ai 2>/dev/null || true)" != "$WORK_ROOT" ]; then
        if [ "$ALLOW_ROOT_REPLACE" != "1" ]; then
          die "/root/physical-ai exists and is not $WORK_ROOT; set RUNPOD_ALLOW_ROOT_PHYSICAL_AI_REPLACE=1 only after preserving it"
        fi
        mv /root/physical-ai "$WORK_ROOT/root-physical-ai.backup.$(date -u +%Y%m%dT%H%M%SZ)"
      fi
      if [ ! -e /root/physical-ai ]; then
        ln -s "$WORK_ROOT" /root/physical-ai
      fi
      ;;
    ephemeral)
      WORK_ROOT="$EPHEMERAL_WORK_ROOT"
      mkdir -p "$WORK_ROOT" "$WORK_ROOT/envs"
      ;;
    *)
      die "unknown RUNPOD_ENV_PROFILE=$PROFILE; use auto, volume, or ephemeral"
      ;;
  esac

  FINAL_VENV="$WORK_ROOT/envs/$VENV_NAME"
  BUILD_VENV="$WORK_ROOT/envs/.$VENV_NAME.build.$$"
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$WORK_ROOT/libero_config}"
  LIBERO_ASSETS_DIR="${LIBERO_ASSETS_DIR:-$WORK_ROOT/libero_assets}"
  LOG_DIR="${RUNPOD_ENV_LOG_DIR:-$PROJECT_DIR/_workspace/runpod_results/env_prepare_$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$LOG_DIR"
  BOOTSTRAP_LOG="$LOG_DIR/bootstrap.log"
  GATE_LOG="$LOG_DIR/import_gate.log"
  MANIFEST="$LOG_DIR/env_manifest.json"
}

write_manifest() {
  status="$1"
  category="${2:-}"
  cat > "$MANIFEST" <<EOF
{
  "status": "$status",
  "blocker_category": "$category",
  "profile": "$PROFILE",
  "work_root": "$WORK_ROOT",
  "project_dir": "$PROJECT_DIR",
  "final_venv": "$FINAL_VENV",
  "libero_config_path": "$LIBERO_CONFIG_PATH",
  "libero_assets_dir": "$LIBERO_ASSETS_DIR",
  "build_venv": "$BUILD_VENV",
  "bootstrap_script": "$PROJECT_DIR/scripts/install/recipes/bootstrap_runpod_libero_smolvla_env.sh",
  "gate_script": "$PROJECT_DIR/scripts/install/recipes/runpod_check_libero_env.sh",
  "bootstrap_log": "$BOOTSTRAP_LOG",
  "gate_log": "$GATE_LOG"
}
EOF
}

classify_log() {
  path="$1"
  if [ ! -s "$path" ]; then
    echo "unknown"
    return
  fi
  if grep -m 1 'BLOCKER_CATEGORY=' "$path" >/dev/null 2>&1; then
    grep -m 1 'BLOCKER_CATEGORY=' "$path" | sed 's/.*BLOCKER_CATEGORY=//; s/[^A-Za-z0-9_.-].*$//'
    return
  fi
  if grep -E 'No space left on device|Disk quota exceeded' "$path" >/dev/null 2>&1; then
    echo "disk_space_blocked"
    return
  fi
  if grep -E 'Could not resolve host|Temporary failure in name resolution|Connection timed out|Read timed out|Network is unreachable|Name or service not known' "$path" >/dev/null 2>&1; then
    echo "network/download_blocked"
    return
  fi
  if grep -E 'No matching distribution found|ResolutionImpossible|Cannot install .* because these package versions have conflicting dependencies' "$path" >/dev/null 2>&1; then
    echo "resolver_drift"
    return
  fi
  if grep -E 'No module named .torch|torch CUDA is unavailable|unexpected torch CUDA version|torch.*drifted|torchvision.*drifted|torchaudio.*drifted' "$path" >/dev/null 2>&1; then
    echo "torch_install_failed"
    return
  fi
  if grep -E 'No module named .libero|libero.*failed|IMPORT_FAIL libero' "$path" >/dev/null 2>&1; then
    echo "libero_install_failed"
    return
  fi
  if grep -E 'No module named .lerobot|IMPORT_FAIL lerobot' "$path" >/dev/null 2>&1; then
    echo "lerobot_install_failed"
    return
  fi
  if grep -E 'Python 3\.11|Requires-Python.*<3\.12|requires a different Python' "$path" >/dev/null 2>&1; then
    echo "python_version_blocked"
    return
  fi
  echo "unknown"
}

run_gate() {
  venv="$1"
  WORK_ROOT="$WORK_ROOT" \
    PROJECT_DIR="$PROJECT_DIR" \
    LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
    LIBERO_ASSETS_DIR="$LIBERO_ASSETS_DIR" \
  PY312_VENV="$venv" \
    REQUIRE_CUDA="${REQUIRE_CUDA:-1}" \
    sh "$PROJECT_DIR/scripts/install/recipes/runpod_check_libero_env.sh"
}

publish_build() {
  if [ -d "$FINAL_VENV" ]; then
    backup="$WORK_ROOT/envs/$VENV_NAME.previous.$(date -u +%Y%m%dT%H%M%SZ)"
    log "moving previous venv to $backup"
    mv "$FINAL_VENV" "$backup"
  fi
  mv "$BUILD_VENV" "$FINAL_VENV"
}

main() {
  timer_start prepare_total
  require_linux
  choose_profile
  resolve_project_dir
  prepare_paths

  log "profile=$PROFILE"
  log "work_root=$WORK_ROOT"
  log "project_dir=$PROJECT_DIR"
  log "final_venv=$FINAL_VENV"
  log "log_dir=$LOG_DIR"
  write_manifest "started"

  if [ "$FORCE_REBUILD" != "1" ] && [ -x "$FINAL_VENV/bin/python" ]; then
    log "checking existing final venv"
    if timed_run existing_final_gate run_gate "$FINAL_VENV" > "$GATE_LOG" 2>&1; then
      log "existing env gate PASS"
      write_manifest "ready_existing"
      cat "$GATE_LOG"
      timer_end prepare_total
      exit 0
    fi
    log "existing env gate failed ($(classify_log "$GATE_LOG")); rebuilding. See $GATE_LOG"
  fi

  if [ -e "$BUILD_VENV" ]; then
    die "temporary build venv already exists: $BUILD_VENV"
  fi

  log "building temporary venv=$BUILD_VENV"
  if ! timed_run bootstrap_recipe env \
    WORK_ROOT="$WORK_ROOT" \
    PROJECT_DIR="$PROJECT_DIR" \
    PY312_VENV="$BUILD_VENV" \
    LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
    LIBERO_ASSETS_DIR="$LIBERO_ASSETS_DIR" \
    sh "$PROJECT_DIR/scripts/install/recipes/bootstrap_runpod_libero_smolvla_env.sh" > "$BOOTSTRAP_LOG" 2>&1; then
    category="$(classify_log "$BOOTSTRAP_LOG")"
    log "bootstrap failed ($category); see $BOOTSTRAP_LOG"
    write_manifest "bootstrap_failed" "$category"
    exit 1
  fi

  log "running hard gate against temporary venv"
  if ! timed_run temporary_env_gate run_gate "$BUILD_VENV" > "$GATE_LOG" 2>&1; then
    category="$(classify_log "$GATE_LOG")"
    log "temporary env gate failed ($category); see $GATE_LOG"
    write_manifest "gate_failed" "$category"
    exit 1
  fi

  timed_run publish_build publish_build
  log "running hard gate against published venv"
  if ! timed_run published_env_gate run_gate "$FINAL_VENV" > "$GATE_LOG" 2>&1; then
    category="$(classify_log "$GATE_LOG")"
    log "published env gate failed ($category); see $GATE_LOG"
    write_manifest "publish_gate_failed" "$category"
    exit 1
  fi
  write_manifest "ready"
  cat "$GATE_LOG"
  timer_end prepare_total
  log "environment ready"
}

main "$@"
