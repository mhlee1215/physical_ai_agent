#!/bin/sh
set -eu

# Hard preflight gate for RunPod LIBERO/SmolVLA evaluation.
# This script must fail non-zero when the persistent venv is missing torch,
# CUDA is unavailable, or required LIBERO/LeRobot runtime imports are absent.

PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"
PYTHON_BIN="${PYTHON_BIN:-$PY312_VENV/bin/python}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
EXPECTED_TORCH_PREFIX="${EXPECTED_TORCH_PREFIX:-2.5.1+cu124}"
EXPECTED_TORCHVISION_PREFIX="${EXPECTED_TORCHVISION_PREFIX:-0.20.1+cu124}"
EXPECTED_TORCHAUDIO_PREFIX="${EXPECTED_TORCHAUDIO_PREFIX:-2.5.1+cu124}"

log() {
  printf '[runpod-libero-env-gate] %s\n' "$*"
}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "venv python is missing or not executable: $PYTHON_BIN" >&2
  exit 1
fi

log "python=$PYTHON_BIN"

REQUIRE_CUDA="$REQUIRE_CUDA" \
EXPECTED_TORCH_PREFIX="$EXPECTED_TORCH_PREFIX" \
EXPECTED_TORCHVISION_PREFIX="$EXPECTED_TORCHVISION_PREFIX" \
EXPECTED_TORCHAUDIO_PREFIX="$EXPECTED_TORCHAUDIO_PREFIX" \
"$PYTHON_BIN" - <<'PY'
import importlib
import importlib.metadata as md
import os
import sys


def version_for(package):
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return "unknown"


def require_prefix(label, actual, expected):
    if expected and not actual.startswith(expected):
        raise RuntimeError(f"{label} drifted: expected prefix {expected!r}, got {actual!r}")


print("python", sys.version.split()[0])
print("executable", sys.executable)

errors = []
loaded = {}
for package, module in [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("lerobot", "lerobot"),
    ("libero", "libero"),
    ("robosuite", "robosuite"),
    ("mujoco", "mujoco"),
    ("av", "av"),
    ("num2words", "num2words"),
]:
    try:
        loaded[module] = importlib.import_module(module)
        print(f"{module} OK {version_for(package)}")
    except Exception as exc:
        errors.append(f"{module}: {type(exc).__name__}: {exc}")

if errors:
    for error in errors:
        print(f"IMPORT_FAIL {error}", file=sys.stderr)
    raise SystemExit(1)

torch = loaded["torch"]
torchvision = loaded["torchvision"]
torchaudio = loaded["torchaudio"]

print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("torch.cuda.is_available", torch.cuda.is_available())
print("torch.cuda.device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA")

require_prefix("torch", torch.__version__, os.environ.get("EXPECTED_TORCH_PREFIX", ""))
require_prefix("torchvision", torchvision.__version__, os.environ.get("EXPECTED_TORCHVISION_PREFIX", ""))
require_prefix("torchaudio", torchaudio.__version__, os.environ.get("EXPECTED_TORCHAUDIO_PREFIX", ""))

if os.environ.get("REQUIRE_CUDA", "1") == "1":
    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA is unavailable; stop before LIBERO benchmark/probes")
    cuda_version = torch.version.cuda or ""
    if not cuda_version.startswith("12."):
        raise RuntimeError(f"unexpected torch CUDA version {cuda_version!r}; expected CUDA 12.x/cu124-compatible")

print("env gate OK")
PY
