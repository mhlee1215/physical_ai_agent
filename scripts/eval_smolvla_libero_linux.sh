#!/bin/sh
set -eu

PROJECT_DIR="${PROJECT_DIR:-/workspace/physical-ai/physical_ai_agent}"
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
LEROBOT_DIR="${LEROBOT_DIR:-$WORK_ROOT/vendor/lerobot}"
PY312_VENV="${PY312_VENV:-$WORK_ROOT/envs/lerobot_py312}"
LEROBOT_REF="${LEROBOT_REF:-main}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-$HOME/.libero}"
LIBERO_ASSETS_DIR="${LIBERO_ASSETS_DIR:-$WORK_ROOT/libero_assets}"

SMOLVLA_MODEL_ID="${SMOLVLA_MODEL_ID:-lerobot/smolvla_libero}"
LIBERO_TASKS="${LIBERO_TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-}"
LIBERO_N_EPISODES="${LIBERO_N_EPISODES:-10}"
LIBERO_BATCH_SIZE="${LIBERO_BATCH_SIZE:-1}"
LIBERO_MAX_PARALLEL_TASKS="${LIBERO_MAX_PARALLEL_TASKS:-1}"
LIBERO_USE_ASYNC_ENVS="${LIBERO_USE_ASYNC_ENVS:-false}"
LIBERO_CAMERA_NAME_MAPPING="${LIBERO_CAMERA_NAME_MAPPING:-{\"agentview_image\": \"camera1\", \"robot0_eye_in_hand_image\": \"camera2\"}}"
POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-1}"
LIBERO_EXTRA_ARGS="${LIBERO_EXTRA_ARGS:-}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"
APT_PACKAGES="python3.12 python3.12-dev python3.12-venv build-essential cmake git ffmpeg libegl1 libgl1"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/_workspace/runpod_results/smolvla_libero_$STAMP}"
LOG_PATH="$OUTPUT_ROOT/lerobot_eval.log"
REPORT_PATH="$OUTPUT_ROOT/smolvla_libero_report.md"

mkdir -p "$OUTPUT_ROOT"

if [ ! -d "$PROJECT_DIR/.git" ]; then
  echo "PROJECT_DIR does not look like a git checkout: $PROJECT_DIR" >&2
  exit 2
fi

cd "$PROJECT_DIR"
GIT_COMMIT="$(git rev-parse --short HEAD)"

if [ "$SKIP_BOOTSTRAP" != "1" ]; then
  missing_apt_packages=""
  for package in $APT_PACKAGES; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then
      missing_apt_packages="$missing_apt_packages $package"
    fi
  done

  if [ -n "$missing_apt_packages" ]; then
    apt-get update
    apt-get install -y $missing_apt_packages
  fi

  if [ ! -d "$PY312_VENV" ]; then
    python3.12 -m venv "$PY312_VENV"
  fi

  "$PY312_VENV/bin/python" -m pip install --upgrade pip 'setuptools>=71,<81' wheel

  mkdir -p "$(dirname "$LEROBOT_DIR")"
  if [ ! -d "$LEROBOT_DIR/.git" ]; then
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
  fi
  git -C "$LEROBOT_DIR" fetch origin
  git -C "$LEROBOT_DIR" checkout "$LEROBOT_REF"
  git -C "$LEROBOT_DIR" pull --ff-only origin "$LEROBOT_REF" || true

  "$PY312_VENV/bin/python" -m pip install -e "$LEROBOT_DIR[smolvla,libero]"
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

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HOME="${HF_HOME:-$WORK_ROOT/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

TASK_IDS_ARG=""
if [ -n "$LIBERO_TASK_IDS" ]; then
  TASK_IDS_ARG="--env.task_ids=$LIBERO_TASK_IDS"
fi

CAMERA_NAME_MAPPING_ARG=""
if [ "$LIBERO_CAMERA_NAME_MAPPING" != "none" ]; then
  CAMERA_NAME_MAPPING_ARG="--env.camera_name_mapping='$LIBERO_CAMERA_NAME_MAPPING'"
fi

COMMAND="lerobot-eval \
  --output_dir=$OUTPUT_ROOT/eval_logs \
  --policy.path=$SMOLVLA_MODEL_ID \
  --env.type=libero \
  --env.task=$LIBERO_TASKS \
  $TASK_IDS_ARG \
  $CAMERA_NAME_MAPPING_ARG \
  --eval.batch_size=$LIBERO_BATCH_SIZE \
  --eval.n_episodes=$LIBERO_N_EPISODES \
  --eval.use_async_envs=$LIBERO_USE_ASYNC_ENVS \
  --env.max_parallel_tasks=$LIBERO_MAX_PARALLEL_TASKS \
  --policy.empty_cameras=$POLICY_EMPTY_CAMERAS \
  $LIBERO_EXTRA_ARGS"

cat > "$REPORT_PATH" <<EOF
# SmolVLA LIBERO Evaluation Report

- status: running
- git_commit: \`$GIT_COMMIT\`
- model_id: \`$SMOLVLA_MODEL_ID\`
- tasks: \`$LIBERO_TASKS\`
- task_ids: \`${LIBERO_TASK_IDS:-all}\`
- episodes_per_task: \`$LIBERO_N_EPISODES\`
- batch_size: \`$LIBERO_BATCH_SIZE\`
- max_parallel_tasks: \`$LIBERO_MAX_PARALLEL_TASKS\`
- use_async_envs: \`$LIBERO_USE_ASYNC_ENVS\`
- camera_name_mapping: \`$LIBERO_CAMERA_NAME_MAPPING\`
- policy_empty_cameras: \`$POLICY_EMPTY_CAMERAS\`
- mujoco_gl: \`$MUJOCO_GL\`
- output_root: \`$OUTPUT_ROOT\`

## Command

\`\`\`bash
$COMMAND
\`\`\`

## Notes

- Paper-comparable protocol is 10 episodes per task over Spatial, Object, Goal,
  and Long/10 suites.
- Subset runs are comparable only against the same suite/task-id/episode subset.
- Quick smoke runs with fewer episodes are plumbing checks, not full-paper
  comparison numbers.
EOF

set +e
PATH="$PY312_VENV/bin:$PATH" sh -c "$COMMAND" > "$LOG_PATH" 2>&1
eval_status=$?
cat "$LOG_PATH"
set -e

if [ "$eval_status" -eq 0 ]; then
  status_text="completed"
else
  status_text="failed"
fi

{
  echo
  echo "## Completion"
  echo
  echo "- status: $status_text"
  echo "- exit_code: $eval_status"
  echo "- log_path: \`$LOG_PATH\`"
  echo "- eval_logs: \`$OUTPUT_ROOT/eval_logs\`"
  echo
  echo "## Candidate Metric Files"
  echo
  find "$OUTPUT_ROOT" -maxdepth 4 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' \) | sort
} >> "$REPORT_PATH"

echo "report=$REPORT_PATH"
echo "log=$LOG_PATH"
exit "$eval_status"
