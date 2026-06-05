#!/bin/sh
set -eu

if [ -z "${RUNPOD_SSH:-}" ]; then
  echo "Set RUNPOD_SSH first, for example:" >&2
  echo "  export RUNPOD_SSH='user@ssh.runpod.io'" >&2
  exit 2
fi

SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_RESULT_DIR="${RUNPOD_REMOTE_RESULT_DIR:-/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results}"
LOCAL_RESULT_ROOT="${RUNPOD_LOCAL_RESULT_ROOT:-_workspace/runpod_results}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_RESULT_DIR="$LOCAL_RESULT_ROOT/$STAMP"

mkdir -p "$LOCAL_RESULT_DIR"

ssh -tt -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" "$RUNPOD_SSH" \
  "test -d '$REMOTE_RESULT_DIR'"

scp -r -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" \
  "$RUNPOD_SSH:$REMOTE_RESULT_DIR/." "$LOCAL_RESULT_DIR/"

echo "fetched_results=$LOCAL_RESULT_DIR"
