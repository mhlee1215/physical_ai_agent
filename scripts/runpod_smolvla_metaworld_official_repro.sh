#!/usr/bin/env bash
set -euo pipefail

# Public LeRobot Meta-World reproduction runner for RunPod.
#
# Important: do not force the LeRobot docs' old `gymnasium==1.1.0` workaround
# here. On 2026-06-08, current public LeRobot main resolved to Gymnasium 1.3.0
# and ran a Meta-World rollout successfully, while downgrading to 1.1.0
# reproduced the render_modes assertion.

ROOT="${ROOT:-/tmp/metaworld_public_repro}"
RUN_ID="${RUN_ID:-metaworld_public_repro_$(date -u +%Y%m%dT%H%M%SZ)}"
WORK="$ROOT/$RUN_ID"
LERO="$WORK/lerobot"
OUT="${OUT:-/tmp/metaworld_public_repro_results/$RUN_ID}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
LERO_REF="${LERO_REF:-main}"
TASK="${TASK:-assembly-v3}"
EPISODES="${EPISODES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_metaworld}"
POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-0}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TORCH_VERSION_SPEC="${TORCH_VERSION_SPEC:-}"
TORCHVISION_VERSION_SPEC="${TORCHVISION_VERSION_SPEC:-}"
TORCHAUDIO_VERSION_SPEC="${TORCHAUDIO_VERSION_SPEC:-}"
RENAME_MAP="${RENAME_MAP:-{\"observation.image\":\"observation.images.camera1\"}}"
POLICY_N_ACTION_STEPS_ARGS=()
if [ -n "$POLICY_N_ACTION_STEPS" ]; then
  POLICY_N_ACTION_STEPS_ARGS=(--policy.n_action_steps="$POLICY_N_ACTION_STEPS")
fi

mkdir -p "$WORK" "$OUT"
exec > >(tee "$OUT/repro.log") 2>&1

echo "[meta] run_id=$RUN_ID"
echo "[meta] started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[meta] root=$ROOT"
echo "[meta] output=$OUT"
echo "[meta] lerobot_ref=$LERO_REF"
echo "[meta] python_version=$PYTHON_VERSION"
echo "[meta] task=$TASK"
echo "[meta] episodes=$EPISODES"
echo "[meta] batch_size=$BATCH_SIZE"
echo "[meta] policy_path=$POLICY_PATH"
echo "[meta] policy_n_action_steps=${POLICY_N_ACTION_STEPS:-checkpoint_default}"
echo "[meta] torch_index_url=${TORCH_INDEX_URL:-default}"
echo "[meta] torch_version_spec=${TORCH_VERSION_SPEC:-default}"
echo "[meta] torchvision_version_spec=${TORCHVISION_VERSION_SPEC:-default}"
echo "[meta] torchaudio_version_spec=${TORCHAUDIO_VERSION_SPEC:-default}"
echo "[meta] official_doc=https://huggingface.co/docs/lerobot/metaworld"
echo "[meta] note=Fresh public LeRobot clone with smolvla+metaworld extras; no Gymnasium downgrade/pin."

export HF_HOME="${HF_HOME:-/workspace/physical-ai/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git libegl1 libopengl0 libgl1 libosmesa6 libglfw3
fi

python3 -m pip install --upgrade pip uv
uv python install "$PYTHON_VERSION"

git clone https://github.com/huggingface/lerobot.git "$LERO"
git -C "$LERO" checkout "$LERO_REF"
git -C "$LERO" rev-parse HEAD > "$OUT/lerobot_commit.txt"
git -C "$LERO" describe --tags --always --dirty > "$OUT/lerobot_describe.txt"
git -C "$LERO" status --short > "$OUT/lerobot_status.txt"

uv venv --seed --python "$PYTHON_VERSION" "$WORK/.venv"
# shellcheck disable=SC1091
source "$WORK/.venv/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$LERO[smolvla,metaworld]"
if [ -n "$TORCH_VERSION_SPEC" ]; then
  torch_packages=("$TORCH_VERSION_SPEC")
  if [ -n "$TORCHVISION_VERSION_SPEC" ]; then
    torch_packages+=("$TORCHVISION_VERSION_SPEC")
  fi
  if [ -n "$TORCHAUDIO_VERSION_SPEC" ]; then
    torch_packages+=("$TORCHAUDIO_VERSION_SPEC")
  fi
  if [ -n "$TORCH_INDEX_URL" ]; then
    python -m pip install --index-url "$TORCH_INDEX_URL" "${torch_packages[@]}"
  else
    python -m pip install "${torch_packages[@]}"
  fi
fi

python - <<'PY' > "$OUT/environment_probe.txt"
import importlib.metadata as md
import sys

print("python", sys.version)
for name in ["lerobot", "torch", "torchvision", "torchaudio", "torchcodec", "metaworld", "gymnasium", "mujoco"]:
    try:
        print(name, md.version(name))
    except Exception as exc:
        print(name, f"unavailable:{type(exc).__name__}:{exc}")
try:
    import torch

    print("cuda_available", torch.cuda.is_available())
    print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
except Exception as exc:
    print("torch_probe_error", repr(exc))
PY

cat > "$OUT/run_command.txt" <<EOF
MUJOCO_GL=egl lerobot-eval \\
  --output_dir="$OUT/eval" \\
  --policy.path=$POLICY_PATH \\
  --env.type=metaworld \\
  --env.task=$TASK \\
  --eval.batch_size=$BATCH_SIZE \\
  --eval.n_episodes=$EPISODES \\
  --eval.use_async_envs=false \\
  --policy.device=cuda \\
  --policy.use_amp=false \\
  --policy.empty_cameras=$POLICY_EMPTY_CAMERAS \\
  ${POLICY_N_ACTION_STEPS:+--policy.n_action_steps=$POLICY_N_ACTION_STEPS \\}
  --rename_map='$RENAME_MAP' \\
  --seed=0
EOF

set +e
MUJOCO_GL=egl lerobot-eval \
  --output_dir="$OUT/eval" \
  --policy.path="$POLICY_PATH" \
  --env.type=metaworld \
  --env.task="$TASK" \
  --eval.batch_size="$BATCH_SIZE" \
  --eval.n_episodes="$EPISODES" \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.empty_cameras="$POLICY_EMPTY_CAMERAS" \
  "${POLICY_N_ACTION_STEPS_ARGS[@]}" \
  --rename_map="$RENAME_MAP" \
  --seed=0 \
  > "$OUT/eval.log" 2>&1
rc=$?
set -e
echo "$rc" > "$OUT/exit_code.txt"

python - <<PY > "$OUT/summary.txt"
from pathlib import Path
import json

out = Path("$OUT")
print("exit_code", (out / "exit_code.txt").read_text().strip())
for p in sorted((out / "eval").rglob("*.json")):
    print(f"--- {p.relative_to(out)} ---")
    try:
        data = json.loads(p.read_text())
        print(json.dumps(data, indent=2, sort_keys=True)[:12000])
    except Exception:
        print(p.read_text()[:12000])
PY

tar -C "$(dirname "$OUT")" -czf "$OUT/evidence.tgz" "$(basename "$OUT")" || true

echo "[meta] finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[meta] exit_code=$rc"
exit "$rc"
