#!/bin/sh
set -eu

if [ -z "${RUNPOD_SSH:-}" ]; then
  echo "Set RUNPOD_SSH first, for example:"
  echo "  export RUNPOD_SSH='user@ssh.runpod.io'"
  exit 2
fi

SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_PORT="${RUNPOD_SSH_PORT:-}"
SSH_PORT_ARGS=""
if [ -n "$SSH_PORT" ]; then
  SSH_PORT_ARGS="-p $SSH_PORT"
fi

cat <<'REMOTE_COMMANDS' | ssh -tt -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" $SSH_PORT_ARGS "$RUNPOD_SSH"
set -eu
echo "[runpod]"
hostname
df -h /workspace /
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python --version
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_version', torch.version.cuda)
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
PY
if [ -d /workspace/physical-ai/physical_ai_agent/.git ]; then
  cd /workspace/physical-ai/physical_ai_agent
  git rev-parse --short HEAD
  git status --short
fi
exit
REMOTE_COMMANDS
