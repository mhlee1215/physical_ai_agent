#!/bin/sh
set -eu

# Reproduce the RunPod LIBERO + SmolVLA environment that worked on
# runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04.
#
# Key lesson from the 2026-06-09 run: do not let LeRobot dependency
# resolution pull torch 2.11/cu13 on driver 560. Pin torch 2.5.1+cu124
# first, install LeRobot editable with --no-deps, then add runtime deps
# without allowing torch-family upgrades.

WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
PROJECT_DIR="${PROJECT_DIR:-$WORK_ROOT/physical_ai_agent}"
PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"
LEROBOT_DIR="${LEROBOT_DIR:-$WORK_ROOT/vendor/lerobot}"
LEROBOT_REF="${LEROBOT_REF:-v0.5.2}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-$HOME/.libero}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$LIBERO_CONFIG_DIR}"
LIBERO_CONFIG_DIR="$LIBERO_CONFIG_PATH"
LIBERO_ASSETS_DIR="${LIBERO_ASSETS_DIR:-$WORK_ROOT/libero_assets}"
HF_HOME="${HF_HOME:-$WORK_ROOT/hf_home}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_ROOT/pip_cache}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.5.1+cu124}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.20.1+cu124}"
TORCHAUDIO_SPEC="${TORCHAUDIO_SPEC:-torchaudio==2.5.1+cu124}"
EXPECTED_TORCH_PREFIX="${EXPECTED_TORCH_PREFIX:-2.5.1+cu124}"
EXPECTED_TORCHVISION_PREFIX="${EXPECTED_TORCHVISION_PREFIX:-0.20.1+cu124}"
EXPECTED_TORCHAUDIO_PREFIX="${EXPECTED_TORCHAUDIO_PREFIX:-2.5.1+cu124}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy==2.2.6}"
FSSPEC_SPEC="${FSSPEC_SPEC:-fsspec==2026.2.0}"
SETUPTOOLS_SPEC="${SETUPTOOLS_SPEC:-setuptools>=71,<81}"
MUJOCO_SPEC="${MUJOCO_SPEC:-mujoco==3.3.2}"
ROBOSUITE_SPEC="${ROBOSUITE_SPEC:-robosuite==1.4.0}"
LIBERO_SPEC="${LIBERO_SPEC:-libero}"

export PIP_CACHE_DIR
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
export HF_HOME
export LIBERO_CONFIG_PATH
export LIBERO_CONFIG_DIR
export LIBERO_ASSETS_DIR
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

APT_PACKAGES="${APT_PACKAGES:-python3.12 python3.12-dev python3.12-venv build-essential cmake git ffmpeg libegl1 libgl1 libglib2.0-0 libglvnd0 libglx0 libopengl0 libosmesa6 libglfw3 libxrender1 libsm6 libxext6}"

log() {
  printf '[runpod-libero-bootstrap] %s\n' "$*"
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

run() {
  log "+ $*"
  "$@"
}

require_linux() {
  if [ "$(uname -s)" != "Linux" ]; then
    echo "This bootstrap is intended for Linux/RunPod, not $(uname -s)." >&2
    exit 2
  fi
}

install_apt_packages() {
  missing=""
  for package in $APT_PACKAGES; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then
      missing="$missing $package"
    fi
  done
  if [ -n "$missing" ]; then
    timed_run apt_update apt-get update
    # shellcheck disable=SC2086
    timed_run apt_install apt-get install -y $missing
  else
    log "apt packages already present"
  fi
}

ensure_python312_venv() {
  if [ ! -d "$PY312_VENV" ]; then
    timed_run python_venv_create python3.12 -m venv "$PY312_VENV"
  else
    log "venv already exists at $PY312_VENV"
  fi
  timed_run pip_upgrade_setup "$PY312_VENV/bin/python" -m pip install --upgrade pip wheel "$SETUPTOOLS_SPEC"
}

pin_torch_cu124() {
  # Remove any partial cu13/torch2.11 drift before pinning the known-good stack.
  "$PY312_VENV/bin/python" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
  timed_run torch_cu124_install "$PY312_VENV/bin/python" -m pip install \
    --index-url "$TORCH_INDEX_URL" \
    "$TORCH_SPEC" "$TORCHVISION_SPEC" "$TORCHAUDIO_SPEC"
}

write_torch_constraints() {
  mkdir -p "$WORK_ROOT"
  cat > "$WORK_ROOT/torch-cu124-constraints.txt" <<EOF
$TORCH_SPEC
$TORCHVISION_SPEC
$TORCHAUDIO_SPEC
$NUMPY_SPEC
$FSSPEC_SPEC
$SETUPTOOLS_SPEC
EOF
  log "constraints=$WORK_ROOT/torch-cu124-constraints.txt"
}

install_lerobot_and_runtime_deps() {
  mkdir -p "$(dirname "$LEROBOT_DIR")"
  if [ ! -d "$LEROBOT_DIR/.git" ]; then
    run git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
  fi
  run git -C "$LEROBOT_DIR" fetch --tags origin
  if ! git -C "$LEROBOT_DIR" checkout "$LEROBOT_REF"; then
    log "warning: LEROBOT_REF=$LEROBOT_REF unavailable; falling back to origin/main"
    run git -C "$LEROBOT_DIR" checkout origin/main
  fi

  # Editable registration only. Runtime deps are explicit below so torch stays cu124.
  timed_run lerobot_editable_install "$PY312_VENV/bin/python" -m pip install -e "$LEROBOT_DIR" --no-deps
  timed_run base_runtime_deps_install "$PY312_VENV/bin/python" -m pip install \
    --constraint "$WORK_ROOT/torch-cu124-constraints.txt" \
    "$NUMPY_SPEC" "$FSSPEC_SPEC" "$SETUPTOOLS_SPEC"
  timed_run sim_runtime_deps_install "$PY312_VENV/bin/python" -m pip install \
    --constraint "$WORK_ROOT/torch-cu124-constraints.txt" \
    "$MUJOCO_SPEC" "$ROBOSUITE_SPEC"
  timed_run auxiliary_deps_install "$PY312_VENV/bin/python" -m pip install \
    --constraint "$WORK_ROOT/torch-cu124-constraints.txt" \
    huggingface_hub draccus safetensors datasets transformers accelerate \
    imageio imageio-ffmpeg opencv-python-headless pyyaml einops num2words av
  timed_run libero_install "$PY312_VENV/bin/python" -m pip install \
    --constraint "$WORK_ROOT/torch-cu124-constraints.txt" \
    "$LIBERO_SPEC"
  # Re-pin torch family after dependency installs as a guard against resolver drift.
  timed_run torch_cu124_repin "$PY312_VENV/bin/python" -m pip install \
    --index-url "$TORCH_INDEX_URL" \
    "$TORCH_SPEC" "$TORCHVISION_SPEC" "$TORCHAUDIO_SPEC"
}

download_libero_assets() {
  "$PY312_VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lerobot/libero-assets", repo_type="dataset", local_dir="$LIBERO_ASSETS_DIR")
PY
}

write_libero_config() {
  mkdir -p "$LIBERO_CONFIG_DIR" "$LIBERO_ASSETS_DIR" "$HF_HOME"
  site_packages="$("$PY312_VENV/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
  libero_pkg="$site_packages/libero/libero"
  if [ -d "$libero_pkg" ]; then
    timed_run libero_assets_download download_libero_assets
    if [ ! -f "$PROJECT_DIR/scripts/install/runpod_prepare_libero_config.sh" ]; then
      echo "missing repo LIBERO config script: $PROJECT_DIR/scripts/install/runpod_prepare_libero_config.sh" >&2
      exit 1
    fi
    timed_run libero_config_prepare env \
      WORK_ROOT="$WORK_ROOT" \
      PROJECT_DIR="$PROJECT_DIR" \
      PY312_VENV="$PY312_VENV" \
      LIBERO_CONFIG_PATH="$LIBERO_CONFIG_DIR" \
      LIBERO_ASSETS_DIR="$LIBERO_ASSETS_DIR" \
      LIBERO_PACKAGE_DIR="$libero_pkg" \
      sh "$PROJECT_DIR/scripts/install/runpod_prepare_libero_config.sh"
    log "LIBERO config written to $LIBERO_CONFIG_DIR/config.yaml"
    if command -v du >/dev/null 2>&1; then
      log "LIBERO assets size=$(du -sh "$LIBERO_ASSETS_DIR" 2>/dev/null | awk '{print $1}') path=$LIBERO_ASSETS_DIR"
      log "HF cache size=$(du -sh "$HF_HOME" 2>/dev/null | awk '{print $1}') path=$HF_HOME"
    fi
  else
    log "warning: LIBERO package dir not found at $libero_pkg; import check will show the blocker"
  fi
}

print_checks() {
  log "hard import/CUDA gate"
  if [ ! -f "$PROJECT_DIR/scripts/install/runpod_check_libero_env.sh" ]; then
    echo "missing repo gate script: $PROJECT_DIR/scripts/install/runpod_check_libero_env.sh" >&2
    exit 1
  fi
  timed_run final_import_gate env \
    PY312_VENV="$PY312_VENV" \
    REQUIRE_CUDA="${REQUIRE_CUDA:-1}" \
    EXPECTED_TORCH_PREFIX="$EXPECTED_TORCH_PREFIX" \
    EXPECTED_TORCHVISION_PREFIX="$EXPECTED_TORCHVISION_PREFIX" \
    EXPECTED_TORCHAUDIO_PREFIX="$EXPECTED_TORCHAUDIO_PREFIX" \
    sh "$PROJECT_DIR/scripts/install/runpod_check_libero_env.sh"
  log "pip check (remaining LeRobot torch>=2.7 conflicts can be recorded if imports/CUDA are OK)"
  "$PY312_VENV/bin/python" -m pip check || true
}

main() {
  timer_start total_bootstrap
  require_linux
  log "project=$PROJECT_DIR"
  log "venv=$PY312_VENV"
  log "base_image_hint=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
  install_apt_packages
  ensure_python312_venv
  pin_torch_cu124
  write_torch_constraints
  install_lerobot_and_runtime_deps
  write_libero_config
  print_checks
  timer_end total_bootstrap
  log "bootstrap complete"
}

main "$@"
