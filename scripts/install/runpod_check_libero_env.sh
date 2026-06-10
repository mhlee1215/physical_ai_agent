#!/bin/sh
set -eu

# Hard preflight gate for RunPod LIBERO/SmolVLA evaluation.
# This script must fail non-zero when the persistent venv is missing torch,
# CUDA is unavailable, or required LIBERO/LeRobot runtime imports are absent.

PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"
PYTHON_BIN="${PYTHON_BIN:-$PY312_VENV/bin/python}"
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
PROJECT_DIR="${PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_CONFIG_DIR:-$HOME/.libero}}"
LIBERO_ASSETS_DIR="${LIBERO_ASSETS_DIR:-$WORK_ROOT/libero_assets}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
EXPECTED_TORCH_PREFIX="${EXPECTED_TORCH_PREFIX:-2.5.1+cu124}"
EXPECTED_TORCHVISION_PREFIX="${EXPECTED_TORCHVISION_PREFIX:-0.20.1+cu124}"
EXPECTED_TORCHAUDIO_PREFIX="${EXPECTED_TORCHAUDIO_PREFIX:-2.5.1+cu124}"

log() {
  printf '[runpod-libero-env-gate] %s\n' "$*"
}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "venv python is missing or not executable: $PYTHON_BIN" >&2
  echo "BLOCKER_CATEGORY=volume_path_mismatch" >&2
  exit 1
fi

log "python=$PYTHON_BIN"

if [ ! -f "$PROJECT_DIR/scripts/install/runpod_prepare_libero_config.sh" ]; then
  echo "missing repo LIBERO config script: $PROJECT_DIR/scripts/install/runpod_prepare_libero_config.sh" >&2
  echo "BLOCKER_CATEGORY=libero_config_missing" >&2
  exit 1
fi

log "ensuring LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH"
WORK_ROOT="$WORK_ROOT" \
PROJECT_DIR="$PROJECT_DIR" \
PY312_VENV="$PY312_VENV" \
PYTHON_BIN="$PYTHON_BIN" \
LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
LIBERO_CONFIG_DIR="$LIBERO_CONFIG_PATH" \
LIBERO_ASSETS_DIR="$LIBERO_ASSETS_DIR" \
sh "$PROJECT_DIR/scripts/install/runpod_prepare_libero_config.sh"

REQUIRE_CUDA="$REQUIRE_CUDA" \
EXPECTED_TORCH_PREFIX="$EXPECTED_TORCH_PREFIX" \
EXPECTED_TORCHVISION_PREFIX="$EXPECTED_TORCHVISION_PREFIX" \
EXPECTED_TORCHAUDIO_PREFIX="$EXPECTED_TORCHAUDIO_PREFIX" \
LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
"$PYTHON_BIN" - <<'PY'
import importlib
import importlib.metadata as md
import os
import sys
from pathlib import Path


def version_for(package):
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return "unknown"


def require_prefix(label, actual, expected):
    if expected and not actual.startswith(expected):
        raise RuntimeError(f"{label} drifted: expected prefix {expected!r}, got {actual!r}")


def blocker_for_import(module, exc):
    text = f"{type(exc).__name__}: {exc}"
    if module in {"torch", "torchvision", "torchaudio"}:
        return "torch_install_failed"
    if module == "lerobot":
        return "lerobot_install_failed"
    if module == "libero":
        return "libero_install_failed"
    if "numpy.dtype size changed" in text or "binary incompatibility" in text:
        return "resolver_drift"
    return "runtime_dependency_missing"


def fail(category, message):
    print(f"BLOCKER_CATEGORY={category}", file=sys.stderr)
    print(f"GATE_FAIL {message}", file=sys.stderr)
    raise SystemExit(1)


print("python", sys.version.split()[0])
print("executable", sys.executable)
config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))).expanduser()
config_path = config_dir / "config.yaml"
if not config_path.is_file():
    fail("libero_config_missing", f"LIBERO config.yaml missing at {config_path}")
config_text = config_path.read_text()
required_config_keys = ("benchmark_root:", "assets:", "bddl_files:", "datasets:", "init_states:")
missing_keys = [key for key in required_config_keys if key not in config_text]
if missing_keys:
    fail("libero_config_missing", f"LIBERO config missing keys: {', '.join(missing_keys)}")
print("LIBERO_CONFIG_PATH", str(config_dir))
print("LIBERO_CONFIG_FILE", str(config_path))

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
        errors.append((module, exc))

if errors:
    categories = []
    for module, exc in errors:
        category = blocker_for_import(module, exc)
        categories.append(category)
        print(f"IMPORT_FAIL {module}: {type(exc).__name__}: {exc}", file=sys.stderr)
    fail(categories[0], "required imports failed")

torch = loaded["torch"]
torchvision = loaded["torchvision"]
torchaudio = loaded["torchaudio"]

print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("torch.cuda.is_available", torch.cuda.is_available())
print("torch.cuda.device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA")

try:
    require_prefix("torch", torch.__version__, os.environ.get("EXPECTED_TORCH_PREFIX", ""))
    require_prefix("torchvision", torchvision.__version__, os.environ.get("EXPECTED_TORCHVISION_PREFIX", ""))
    require_prefix("torchaudio", torchaudio.__version__, os.environ.get("EXPECTED_TORCHAUDIO_PREFIX", ""))
except RuntimeError as exc:
    fail("resolver_drift", str(exc))

if os.environ.get("REQUIRE_CUDA", "1") == "1":
    if not torch.cuda.is_available():
        fail("cuda_mismatch", "torch CUDA is unavailable; stop before LIBERO benchmark/probes")
    cuda_version = torch.version.cuda or ""
    if not cuda_version.startswith("12."):
        fail("cuda_mismatch", f"unexpected torch CUDA version {cuda_version!r}; expected CUDA 12.x/cu124-compatible")

try:
    from libero.libero import get_libero_path

    for key in ("bddl_files", "init_states", "datasets"):
        print(f"get_libero_path.{key}", get_libero_path(key))
except EOFError as exc:
    fail("libero_config_prompt", f"LIBERO attempted an interactive config prompt: {exc}")
except Exception as exc:
    fail("libero_config_missing", f"LIBERO config path resolution failed: {type(exc).__name__}: {exc}")

print("env gate OK")
PY
