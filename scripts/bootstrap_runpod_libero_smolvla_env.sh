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
LIBERO_ASSETS_DIR="${LIBERO_ASSETS_DIR:-$WORK_ROOT/libero_assets}"
HF_HOME="${HF_HOME:-$WORK_ROOT/hf_home}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_ROOT/pip_cache}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.5.1+cu124}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.20.1+cu124}"
TORCHAUDIO_SPEC="${TORCHAUDIO_SPEC:-torchaudio==2.5.1+cu124}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy==2.2.6}"
FSSPEC_SPEC="${FSSPEC_SPEC:-fsspec==2026.2.0}"
SETUPTOOLS_SPEC="${SETUPTOOLS_SPEC:-setuptools>=71,<81}"
MUJOCO_SPEC="${MUJOCO_SPEC:-mujoco==3.3.2}"
ROBOSUITE_SPEC="${ROBOSUITE_SPEC:-robosuite==1.4.0}"
LIBERO_SPEC="${LIBERO_SPEC:-libero}"

export PIP_CACHE_DIR
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

APT_PACKAGES="${APT_PACKAGES:-python3.12 python3.12-dev python3.12-venv build-essential cmake git ffmpeg libegl1 libgl1 libglib2.0-0 libglvnd0 libglx0 libxrender1 libsm6 libxext6}"

log() {
  printf '[runpod-libero-bootstrap] %s\n' "$*"
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
    run apt-get update
    # shellcheck disable=SC2086
    run apt-get install -y $missing
  else
    log "apt packages already present"
  fi
}

ensure_python312_venv() {
  if [ ! -d "$PY312_VENV" ]; then
    run python3.12 -m venv "$PY312_VENV"
  fi
  run "$PY312_VENV/bin/python" -m pip install --upgrade pip wheel "$SETUPTOOLS_SPEC"
}

pin_torch_cu124() {
  # Remove any partial cu13/torch2.11 drift before pinning the known-good stack.
  "$PY312_VENV/bin/python" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
  run "$PY312_VENV/bin/python" -m pip install \
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
  run "$PY312_VENV/bin/python" -m pip install -e "$LEROBOT_DIR" --no-deps
  run "$PY312_VENV/bin/python" -m pip install \
    --constraint "$WORK_ROOT/torch-cu124-constraints.txt" \
    "$NUMPY_SPEC" "$FSSPEC_SPEC" "$SETUPTOOLS_SPEC" \
    "$MUJOCO_SPEC" "$ROBOSUITE_SPEC" "$LIBERO_SPEC" \
    huggingface_hub draccus safetensors datasets transformers accelerate \
    imageio imageio-ffmpeg opencv-python-headless pyyaml einops num2words
  # Re-pin torch family after dependency installs as a guard against resolver drift.
  run "$PY312_VENV/bin/python" -m pip install \
    --index-url "$TORCH_INDEX_URL" \
    "$TORCH_SPEC" "$TORCHVISION_SPEC" "$TORCHAUDIO_SPEC"
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
    "$PY312_VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lerobot/libero-assets", repo_type="dataset", local_dir="$LIBERO_ASSETS_DIR")
PY
    cat > "$LIBERO_CONFIG_DIR/config.yaml" <<EOF
benchmark_root: $libero_pkg
assets: $LIBERO_ASSETS_DIR
bddl_files: $libero_pkg/bddl_files
datasets: $site_packages/libero/datasets
init_states: $libero_pkg/init_files
EOF
    log "LIBERO config written to $LIBERO_CONFIG_DIR/config.yaml"
  else
    log "warning: LIBERO package dir not found at $libero_pkg; import check will show the blocker"
  fi
}

print_checks() {
  log "version checks"
  "$PY312_VENV/bin/python" - <<'PY'
import importlib.metadata as md

import torch

print("python import check:")
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("torch.cuda.is_available", torch.cuda.is_available())
print("torch.cuda.device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA")
for package, module in [
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("lerobot", "lerobot"),
    ("libero", "libero"),
    ("robosuite", "robosuite"),
    ("mujoco", "mujoco"),
]:
    try:
        __import__(module)
        try:
            version = md.version(package)
        except md.PackageNotFoundError:
            version = "unknown"
        print(f"{module} OK {version}")
    except Exception as exc:
        print(f"{module} FAIL {type(exc).__name__}: {exc}")
PY
  log "pip check (remaining LeRobot torch>=2.7 conflicts can be recorded if imports/CUDA are OK)"
  "$PY312_VENV/bin/python" -m pip check || true
}

main() {
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
  log "bootstrap complete"
}

main "$@"
