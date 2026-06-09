#!/bin/sh
set -eu

BENCHMARK="${EVAL_BENCHMARK:-libero}"
AGENTIC_LAYER="${AGENTIC_LAYER:-baseline}"
RETRY_BUDGET="${RETRY_BUDGET:-1}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    --agentic-layer)
      AGENTIC_LAYER="$2"
      shift 2
      ;;
    --retry-budget)
      RETRY_BUDGET="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$BENCHMARK" in
  libero|metaworld) ;;
  *)
    echo "unsupported benchmark: $BENCHMARK" >&2
    exit 2
    ;;
esac

PROJECT_DIR="${PROJECT_DIR:-/workspace/physical-ai/physical_ai_agent}"
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
LEROBOT_DIR="${LEROBOT_DIR:-$WORK_ROOT/vendor/lerobot}"
PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"
LEROBOT_REF="${LEROBOT_REF:-main}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_ROOT/pip_cache}"
export PIP_CACHE_DIR
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-$HOME/.libero}"
LIBERO_ASSETS_DIR="${LIBERO_ASSETS_DIR:-$WORK_ROOT/libero_assets}"
MUJOCO_VERSION="${MUJOCO_VERSION:-3.3.2}"

SMOLVLA_MODEL_ID="${SMOLVLA_MODEL_ID:-}"
LIBERO_TASKS="${LIBERO_TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-}"
LIBERO_N_EPISODES="${LIBERO_N_EPISODES:-10}"
LIBERO_BATCH_SIZE="${LIBERO_BATCH_SIZE:-1}"
LIBERO_MAX_PARALLEL_TASKS="${LIBERO_MAX_PARALLEL_TASKS:-1}"
LIBERO_USE_ASYNC_ENVS="${LIBERO_USE_ASYNC_ENVS:-false}"
if [ -z "${LIBERO_CAMERA_NAME_MAPPING+x}" ]; then
  LIBERO_CAMERA_NAME_MAPPING='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
fi
LIBERO_EXTRA_ARGS="${LIBERO_EXTRA_ARGS:-}"

METAWORLD_TASKS="${METAWORLD_TASKS:-easy,medium,hard,very_hard}"
METAWORLD_N_EPISODES="${METAWORLD_N_EPISODES:-10}"
METAWORLD_BATCH_SIZE="${METAWORLD_BATCH_SIZE:-1}"
METAWORLD_RENAME_MAP="${METAWORLD_RENAME_MAP:-{\"observation.image\":\"observation.images.camera1\"}}"
METAWORLD_SEED="${METAWORLD_SEED:-0}"
METAWORLD_EXTRA_ARGS="${METAWORLD_EXTRA_ARGS:-}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_USE_AMP="${POLICY_USE_AMP:-false}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-}"

POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TORCH_VERSION_SPEC="${TORCH_VERSION_SPEC:-}"
TORCHVISION_VERSION_SPEC="${TORCHVISION_VERSION_SPEC:-}"
TORCHAUDIO_VERSION_SPEC="${TORCHAUDIO_VERSION_SPEC:-}"

if [ -z "$SMOLVLA_MODEL_ID" ]; then
  if [ "$BENCHMARK" = "libero" ]; then
    SMOLVLA_MODEL_ID="lerobot/smolvla_libero"
  else
    SMOLVLA_MODEL_ID="lerobot/smolvla_metaworld"
  fi
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/_workspace/runpod_results/smolvla_${BENCHMARK}_$STAMP}"
LOG_PATH="$OUTPUT_ROOT/lerobot_eval.log"
REPORT_PATH="$OUTPUT_ROOT/smolvla_${BENCHMARK}_report.md"
COMMAND_SCRIPT="$OUTPUT_ROOT/run_command.sh"
ENV_PROBE_PATH="$OUTPUT_ROOT/debug_artifacts/environment_probe.txt"
RUNPOD_PREFLIGHT_PATH="$OUTPUT_ROOT/debug_artifacts/runpod_preflight.txt"
EVENTS_PATH="$OUTPUT_ROOT/debug_artifacts/events.jsonl"

mkdir -p "$OUTPUT_ROOT/debug_artifacts"

if [ ! -d "$PROJECT_DIR/.git" ]; then
  echo "PROJECT_DIR does not look like a git checkout: $PROJECT_DIR" >&2
  exit 2
fi

cd "$PROJECT_DIR"
GIT_COMMIT="$(git rev-parse --short HEAD)"

write_runpod_preflight() {
  {
    echo "benchmark=$BENCHMARK"
    echo "agentic_layer=$AGENTIC_LAYER"
    echo "project_dir=$PROJECT_DIR"
    echo "work_root=$WORK_ROOT"
    echo "lerobot_dir=$LEROBOT_DIR"
    echo "py312_venv=$PY312_VENV"
    echo "pip_cache_dir=$PIP_CACHE_DIR"
    echo "hf_home=${HF_HOME:-$WORK_ROOT/hf_home}"
    echo "require_cuda=$REQUIRE_CUDA"
    echo "policy_device=$POLICY_DEVICE"
    echo "mujoco_version=$MUJOCO_VERSION"
    echo "lerobot_ref=$LEROBOT_REF"
    echo
    echo "## OS"
    uname -a || true
    if [ -f /etc/os-release ]; then
      cat /etc/os-release
    fi
    echo
    echo "## Disk"
    df -h / /workspace 2>/dev/null || df -h || true
    echo
    echo "## Python"
    command -v python3.12 || true
    python3.12 --version 2>/dev/null || true
    if [ -x "$PY312_VENV/bin/python" ]; then
      "$PY312_VENV/bin/python" --version || true
    fi
    echo
    echo "## NVIDIA"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi || true
    else
      echo "nvidia-smi=missing"
    fi
  } > "$RUNPOD_PREFLIGHT_PATH"
}

write_runpod_preflight

bootstrap_common() {
  apt_packages="$1"
  missing_apt_packages=""
  for package in $apt_packages; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then
      missing_apt_packages="$missing_apt_packages $package"
    fi
  done

  if [ -n "$missing_apt_packages" ]; then
    apt-get update
    apt-get install -y $missing_apt_packages
  fi

  if [ ! -d "$PY312_VENV" ]; then
    if ! command -v python3.12 >/dev/null 2>&1; then
      echo "python3.12 is required for LeRobot SmolVLA but was not found after apt setup." >&2
      echo "See $RUNPOD_PREFLIGHT_PATH" >&2
      exit 3
    fi
    python3.12 -m venv "$PY312_VENV"
  fi

  "$PY312_VENV/bin/python" -m pip install --upgrade pip 'setuptools>=71,<81' wheel

  if [ -n "$TORCH_VERSION_SPEC" ]; then
    TORCH_PACKAGES="$TORCH_VERSION_SPEC"
    if [ -n "$TORCHVISION_VERSION_SPEC" ]; then
      TORCH_PACKAGES="$TORCH_PACKAGES $TORCHVISION_VERSION_SPEC"
    fi
    if [ -n "$TORCHAUDIO_VERSION_SPEC" ]; then
      TORCH_PACKAGES="$TORCH_PACKAGES $TORCHAUDIO_VERSION_SPEC"
    fi
    if [ -n "$TORCH_INDEX_URL" ]; then
      # shellcheck disable=SC2086
      "$PY312_VENV/bin/python" -m pip install --index-url "$TORCH_INDEX_URL" $TORCH_PACKAGES
    else
      # shellcheck disable=SC2086
      "$PY312_VENV/bin/python" -m pip install $TORCH_PACKAGES
    fi
  fi

  mkdir -p "$(dirname "$LEROBOT_DIR")"
  if [ ! -d "$LEROBOT_DIR/.git" ]; then
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
  fi
  git -C "$LEROBOT_DIR" fetch origin
  git -C "$LEROBOT_DIR" checkout "$LEROBOT_REF"
  git -C "$LEROBOT_DIR" pull --ff-only origin "$LEROBOT_REF" || true
}

bootstrap_libero_config() {
  if [ -n "$MUJOCO_VERSION" ]; then
    CURRENT_MUJOCO_VERSION="$("$PY312_VENV/bin/python" - <<'PY'
import importlib.metadata

try:
    print(importlib.metadata.version("mujoco"))
except importlib.metadata.PackageNotFoundError:
    print("")
PY
)"
    if [ "$CURRENT_MUJOCO_VERSION" != "$MUJOCO_VERSION" ]; then
      "$PY312_VENV/bin/python" -m pip install "mujoco==$MUJOCO_VERSION"
    fi
  fi

  LIBERO_SITE_PACKAGES="$("$PY312_VENV/bin/python" - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)"
  LIBERO_PACKAGE_DIR="$LIBERO_SITE_PACKAGES/libero/libero"

  if [ ! -f "$LIBERO_CONFIG_DIR/config.yaml" ]; then
    mkdir -p "$LIBERO_CONFIG_DIR"
    "$PY312_VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lerobot/libero-assets",
    repo_type="dataset",
    local_dir="$LIBERO_ASSETS_DIR",
)
PY
    cat > "$LIBERO_CONFIG_DIR/config.yaml" <<EOF
assets: $LIBERO_ASSETS_DIR
bddl_files: $LIBERO_PACKAGE_DIR/bddl_files
datasets: $LIBERO_PACKAGE_DIR/../datasets
init_states: $LIBERO_PACKAGE_DIR/init_files
EOF
  fi
  export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_DIR"
}

if [ "$SKIP_BOOTSTRAP" != "1" ]; then
  if [ "$BENCHMARK" = "libero" ]; then
    bootstrap_common "python3.12 python3.12-dev python3.12-venv build-essential cmake git ffmpeg libegl1 libgl1"
    "$PY312_VENV/bin/python" -m pip install -e "$LEROBOT_DIR[smolvla,libero]"
    bootstrap_libero_config
  else
    bootstrap_common "python3.12 python3.12-dev python3.12-venv build-essential cmake git ffmpeg libegl1 libgl1 libopengl0 libosmesa6 libglfw3"
    "$PY312_VENV/bin/python" -m pip install -e "$LEROBOT_DIR[smolvla,metaworld]"
  fi
else
  if [ ! -x "$PY312_VENV/bin/python" ]; then
    echo "SKIP_BOOTSTRAP=1 but Python venv is missing: $PY312_VENV" >&2
    echo "Unset SKIP_BOOTSTRAP or set PY312_VENV to a valid Python 3.12 venv." >&2
    exit 3
  fi
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HOME="${HF_HOME:-$WORK_ROOT/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

"$PY312_VENV/bin/python" - <<'PY' > "$ENV_PROBE_PATH"
import importlib.metadata
import platform
import sys

packages = ["torch", "lerobot", "libero", "metaworld", "mujoco"]
print(f"python={platform.python_version()}")
print(f"python_executable={sys.executable}")
for package in packages:
    try:
        print(f"{package}={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}=missing")
try:
    import torch

    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
except Exception as exc:
    print(f"torch_probe_error={type(exc).__name__}: {exc}")
PY

"$PY312_VENV/bin/python" - <<PY
import sys

if sys.version_info < (3, 12):
    raise SystemExit("Python >=3.12 is required for LeRobot SmolVLA evaluation")
PY

if [ "$REQUIRE_CUDA" = "1" ]; then
  "$PY312_VENV/bin/python" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"REQUIRE_CUDA=1 but torch import failed: {type(exc).__name__}: {exc}")

if not torch.cuda.is_available():
    raise SystemExit("REQUIRE_CUDA=1 but torch.cuda.is_available() is false; refusing CPU fallback")
PY
fi

write_runpod_preflight

if [ "$BENCHMARK" = "libero" ]; then
  if [ -n "$LIBERO_TASK_IDS" ]; then
    PYTHONPATH="$PROJECT_DIR/src" "$PY312_VENV/bin/python" -m physical_ai_agent.evaluation.lerobot_eval \
      --benchmark libero \
      --output-dir "$OUTPUT_ROOT/eval_logs" \
      --policy-path "$SMOLVLA_MODEL_ID" \
      --tasks "$LIBERO_TASKS" \
      --task-ids "$LIBERO_TASK_IDS" \
      --n-episodes "$LIBERO_N_EPISODES" \
      --batch-size "$LIBERO_BATCH_SIZE" \
      --use-async-envs "$LIBERO_USE_ASYNC_ENVS" \
      --max-parallel-tasks "$LIBERO_MAX_PARALLEL_TASKS" \
      --camera-name-mapping "$LIBERO_CAMERA_NAME_MAPPING" \
      --policy-empty-cameras "$POLICY_EMPTY_CAMERAS" \
      --agentic-layer "$AGENTIC_LAYER" \
      --retry-budget "$RETRY_BUDGET" \
      --artifact-root "$OUTPUT_ROOT" \
      --extra-args "$LIBERO_EXTRA_ARGS" \
      --write-command "$COMMAND_SCRIPT"
  else
    PYTHONPATH="$PROJECT_DIR/src" "$PY312_VENV/bin/python" -m physical_ai_agent.evaluation.lerobot_eval \
      --benchmark libero \
      --output-dir "$OUTPUT_ROOT/eval_logs" \
      --policy-path "$SMOLVLA_MODEL_ID" \
      --tasks "$LIBERO_TASKS" \
      --n-episodes "$LIBERO_N_EPISODES" \
      --batch-size "$LIBERO_BATCH_SIZE" \
      --use-async-envs "$LIBERO_USE_ASYNC_ENVS" \
      --max-parallel-tasks "$LIBERO_MAX_PARALLEL_TASKS" \
      --camera-name-mapping "$LIBERO_CAMERA_NAME_MAPPING" \
      --policy-empty-cameras "$POLICY_EMPTY_CAMERAS" \
      --agentic-layer "$AGENTIC_LAYER" \
      --retry-budget "$RETRY_BUDGET" \
      --artifact-root "$OUTPUT_ROOT" \
      --extra-args "$LIBERO_EXTRA_ARGS" \
      --write-command "$COMMAND_SCRIPT"
  fi
else
  ACTION_STEP_ARGS=""
  if [ -n "$POLICY_N_ACTION_STEPS" ]; then
    ACTION_STEP_ARGS="--n-action-steps $POLICY_N_ACTION_STEPS"
  fi
  # shellcheck disable=SC2086
  PYTHONPATH="$PROJECT_DIR/src" "$PY312_VENV/bin/python" -m physical_ai_agent.evaluation.lerobot_eval \
    --benchmark metaworld \
    --output-dir "$OUTPUT_ROOT/eval_logs" \
    --policy-path "$SMOLVLA_MODEL_ID" \
    --tasks "$METAWORLD_TASKS" \
    --n-episodes "$METAWORLD_N_EPISODES" \
    --batch-size "$METAWORLD_BATCH_SIZE" \
    --rename-map "$METAWORLD_RENAME_MAP" \
    --policy-empty-cameras "$POLICY_EMPTY_CAMERAS" \
    --policy-device "$POLICY_DEVICE" \
    --policy-use-amp "$POLICY_USE_AMP" \
    --seed "$METAWORLD_SEED" \
    --agentic-layer "$AGENTIC_LAYER" \
    --retry-budget "$RETRY_BUDGET" \
    --artifact-root "$OUTPUT_ROOT" \
    --extra-args "$METAWORLD_EXTRA_ARGS" \
    --write-command "$COMMAND_SCRIPT" \
    $ACTION_STEP_ARGS
fi

COMMAND="$(cat "$COMMAND_SCRIPT")"

cat > "$REPORT_PATH" <<EOF
# SmolVLA LeRobot Evaluation Report

- status: running
- benchmark: \`$BENCHMARK\`
- git_commit: \`$GIT_COMMIT\`
- model_id: \`$SMOLVLA_MODEL_ID\`
- agentic_layer: \`$AGENTIC_LAYER\`
- retry_budget: \`$RETRY_BUDGET\`
- mujoco_gl: \`$MUJOCO_GL\`
- output_root: \`$OUTPUT_ROOT\`
- environment_probe: \`$ENV_PROBE_PATH\`

## Command

\`\`\`bash
$COMMAND
\`\`\`

## Debug Artifacts

- eval_manifest: \`$OUTPUT_ROOT/debug_artifacts/eval_manifest.json\`
- command_argv: \`$OUTPUT_ROOT/debug_artifacts/command_argv.json\`
- agentic_layer: \`$OUTPUT_ROOT/debug_artifacts/agentic_layer.json\`
- events: \`$OUTPUT_ROOT/debug_artifacts/events.jsonl\`
EOF

printf '{"event":"run_started","benchmark":"%s","agentic_layer":"%s"}\n' "$BENCHMARK" "$AGENTIC_LAYER" >> "$EVENTS_PATH"

set +e
PATH="$PY312_VENV/bin:$PATH" sh "$COMMAND_SCRIPT" > "$LOG_PATH" 2>&1
eval_status=$?
cat "$LOG_PATH"
set -e

if [ "$eval_status" -eq 0 ] && [ "$BENCHMARK" = "libero" ] && [ "$AGENTIC_LAYER" = "episode_retry" ]; then
  mkdir -p "$OUTPUT_ROOT/agentic"
  PYTHONPATH="$PROJECT_DIR/src" "$PY312_VENV/bin/python" -m physical_ai_agent.agent_core.libero_agentic_retry \
    plan "$OUTPUT_ROOT/eval_logs/eval_info.json" \
    --task-group "$LIBERO_TASKS" \
    --retry-budget "$RETRY_BUDGET" \
    --output-json "$OUTPUT_ROOT/agentic/retry_plan.json" || true
fi

if [ "$eval_status" -eq 0 ]; then
  status_text="completed"
else
  status_text="failed"
fi
printf '{"event":"run_finished","benchmark":"%s","agentic_layer":"%s","exit_code":%s}\n' "$BENCHMARK" "$AGENTIC_LAYER" "$eval_status" >> "$EVENTS_PATH"

{
  echo
  echo "## Completion"
  echo
  echo "- status: $status_text"
  echo "- exit_code: $eval_status"
  echo "- log_path: \`$LOG_PATH\`"
  echo "- eval_logs: \`$OUTPUT_ROOT/eval_logs\`"
  echo
  echo "## Candidate Metric and Debug Files"
  echo
  find "$OUTPUT_ROOT" -maxdepth 4 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' -o -name '*.txt' -o -name '*.sh' \) | sort
} >> "$REPORT_PATH"

echo "report=$REPORT_PATH"
echo "log=$LOG_PATH"
echo "debug_artifacts=$OUTPUT_ROOT/debug_artifacts"
exit "$eval_status"
