#!/bin/sh
set -eu

usage() {
  cat <<'USAGE'
Usage:
  RUNPOD_SSH='root@host' RUNPOD_SSH_PORT='44000' sh scripts/runpod_reuse_live_oracle_probe.sh

Required:
  RUNPOD_SSH
    Existing reachable SSH target for an already-running reused RunPod.

  RUNPOD_SSH_PORT
    Optional SSH port.

Optional:
  RUNPOD_REMOTE_REPO
    Existing remote clone path. Default:
    /workspace/physical-ai/physical_ai_agent_affordance_probe

  RUNPOD_LIVE_OUTPUT_DIR
    Remote checkpoint output directory. Default:
    _workspace/checkpoints/live_oracle_probe_reuse_<UTC timestamp>

  LOCAL_FETCH_DIR
    Local directory for fetched artifacts. Default:
    _workspace/runpod_results/live_oracle_probe_reuse_<UTC timestamp>

  EPISODES
    Default: 1

  STEPS
    Default: 9

Rules:
  - This script never creates a Pod.
  - This script never starts a stopped Pod.
  - This script never uploads local workspace files to RunPod.
  - It only runs commands in an existing remote clone and fetches results back.
  - If the remote clone is stale, update it through your normal git workflow first.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ -z "${RUNPOD_SSH:-}" ]; then
  echo "RUNPOD_SSH is required. Refusing to create or start a Pod." >&2
  usage >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_REPO="${RUNPOD_REMOTE_REPO:-/workspace/physical-ai/physical_ai_agent_affordance_probe}"
REMOTE_OUTPUT="${RUNPOD_LIVE_OUTPUT_DIR:-_workspace/checkpoints/live_oracle_probe_reuse_$timestamp}"
LOCAL_DIR="${LOCAL_FETCH_DIR:-_workspace/runpod_results/live_oracle_probe_reuse_$timestamp}"
EPISODES="${EPISODES:-1}"
STEPS="${STEPS:-9}"
SSH_PORT="${RUNPOD_SSH_PORT:-}"

remote_cmd="
set -eu
cd '$REMOTE_REPO'
test -d src
test -x .venv/bin/python
PYTHONPATH=src .venv/bin/python -B -m physical_ai_agent.checkpoints.checkpoint_24 \
  --require-maniskill \
  --episodes '$EPISODES' \
  --steps '$STEPS' \
  --policy affordance_oracle_probe \
  --real-images \
  --output-dir '$REMOTE_OUTPUT'
if [ -d '$REMOTE_OUTPUT/affordance_oracle_probe_frames' ]; then
  PYTHONPATH=src .venv/bin/python -B scripts/build_oracle_overlay_gallery.py \
    --image-root '$REMOTE_OUTPUT/affordance_oracle_probe_frames' \
    --output-dir '$REMOTE_OUTPUT/affordance_oracle_probe_gallery' \
    --title 'Live ManiSkill Oracle Overlay Probe' \
    --min-frames 10 \
    --limit 40 || true
fi
find '$REMOTE_OUTPUT' -maxdepth 3 -type f | sort
"

mkdir -p "$LOCAL_DIR"
if [ -n "$SSH_PORT" ]; then
  ssh -p "$SSH_PORT" "$RUNPOD_SSH" "$remote_cmd"
  scp -P "$SSH_PORT" -r "$RUNPOD_SSH:$REMOTE_REPO/$REMOTE_OUTPUT/." "$LOCAL_DIR/"
else
  ssh "$RUNPOD_SSH" "$remote_cmd"
  scp -r "$RUNPOD_SSH:$REMOTE_REPO/$REMOTE_OUTPUT/." "$LOCAL_DIR/"
fi

echo "local_fetch_dir=$LOCAL_DIR"
echo "remote_output=$REMOTE_REPO/$REMOTE_OUTPUT"
echo "If this reused Pod should be stopped, run:"
echo "  RUNPOD_API_KEY=... RUNPOD_POD_ID=... sh scripts/runpod_pod.sh stop"
