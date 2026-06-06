#!/bin/sh
set -eu

usage() {
  cat <<'USAGE'
Usage:
  RUNPOD_SSH=... sh scripts/runpod_archive_results.sh
  RUNPOD_SSH=... RUNPOD_ACTIVE_RESULT_DIR=/workspace/.../active_run \
    sh scripts/runpod_archive_results.sh --delete-remote --yes-delete

Environment:
  RUNPOD_SSH                Required. SSH target, for example root@1.2.3.4.
  RUNPOD_SSH_KEY            Optional. Defaults to ~/.ssh/id_ed25519.
  RUNPOD_SSH_PORT           Optional. SSH port.
  RUNPOD_REMOTE_RESULT_DIR  Optional. Defaults to the repo runpod_results dir.
  RUNPOD_LOCAL_RESULT_ROOT  Optional. Defaults to _workspace/runpod_results.
  RUNPOD_ACTIVE_RESULT_DIR  Optional. Remote result dir to preserve.

Behavior:
  - Fetches completed remote result directories into a timestamped local archive.
  - A completed directory is one that contains eval_logs/eval_info.json.
  - With --delete-remote --yes-delete, deletes only completed remote result
    directories that contain eval_logs/eval_info.json.
  - Never deletes RUNPOD_ACTIVE_RESULT_DIR.
USAGE
}

DELETE_REMOTE=0
YES_DELETE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --delete-remote)
      DELETE_REMOTE=1
      ;;
    --yes-delete)
      YES_DELETE=1
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [ -z "${RUNPOD_SSH:-}" ]; then
  echo "Set RUNPOD_SSH first, for example:" >&2
  echo "  export RUNPOD_SSH='root@1.2.3.4'" >&2
  exit 2
fi

if [ "$DELETE_REMOTE" -eq 1 ] && [ "$YES_DELETE" -ne 1 ]; then
  echo "Refusing remote deletion without --yes-delete." >&2
  exit 2
fi

SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_PORT="${RUNPOD_SSH_PORT:-}"
SSH_PORT_ARGS=""
if [ -n "$SSH_PORT" ]; then
  SSH_PORT_ARGS="-P $SSH_PORT"
fi
SSH_PORT_ARGS_LOWER=""
if [ -n "$SSH_PORT" ]; then
  SSH_PORT_ARGS_LOWER="-p $SSH_PORT"
fi

REMOTE_RESULT_DIR="${RUNPOD_REMOTE_RESULT_DIR:-/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results}"
LOCAL_RESULT_ROOT="${RUNPOD_LOCAL_RESULT_ROOT:-_workspace/runpod_results/remote_archives}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_RESULT_DIR="$LOCAL_RESULT_ROOT/runpod_results_$STAMP"
ACTIVE_RESULT_DIR="${RUNPOD_ACTIVE_RESULT_DIR:-}"

mkdir -p "$LOCAL_RESULT_DIR"

ssh -T -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" $SSH_PORT_ARGS_LOWER "$RUNPOD_SSH" \
  "test -d '$REMOTE_RESULT_DIR'"

REMOTE_LIST="$LOCAL_RESULT_DIR/.remote_completed_dirs"
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" $SSH_PORT_ARGS_LOWER "$RUNPOD_SSH" \
  "REMOTE_RESULT_DIR='$REMOTE_RESULT_DIR' ACTIVE_RESULT_DIR='$ACTIVE_RESULT_DIR' sh -s" > "$REMOTE_LIST" <<'REMOTE'
set -eu

find "$REMOTE_RESULT_DIR" -mindepth 1 -maxdepth 1 -type d | while IFS= read -r path; do
  if [ -n "$ACTIVE_RESULT_DIR" ] && [ "$path" = "$ACTIVE_RESULT_DIR" ]; then
    continue
  fi
  if [ -f "$path/eval_logs/eval_info.json" ]; then
    echo "$path"
  fi
done
REMOTE

if [ ! -s "$REMOTE_LIST" ]; then
  echo "no_completed_remote_results=$REMOTE_RESULT_DIR"
  echo "fetched_results=$LOCAL_RESULT_DIR"
else
  while IFS= read -r remote_path; do
    scp -r -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" $SSH_PORT_ARGS \
      "$RUNPOD_SSH:$remote_path" "$LOCAL_RESULT_DIR/"
    echo "fetched_completed=$remote_path"
  done < "$REMOTE_LIST"
  echo "fetched_results=$LOCAL_RESULT_DIR"
fi

if [ "$DELETE_REMOTE" -eq 1 ]; then
  ssh -T -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" $SSH_PORT_ARGS_LOWER "$RUNPOD_SSH" \
    "REMOTE_RESULT_DIR='$REMOTE_RESULT_DIR' ACTIVE_RESULT_DIR='$ACTIVE_RESULT_DIR' sh -s" <<'REMOTE'
set -eu

find "$REMOTE_RESULT_DIR" -mindepth 1 -maxdepth 1 -type d | while IFS= read -r path; do
  if [ -n "$ACTIVE_RESULT_DIR" ] && [ "$path" = "$ACTIVE_RESULT_DIR" ]; then
    echo "preserved_active=$path"
    continue
  fi
  if [ -f "$path/eval_logs/eval_info.json" ]; then
    rm -rf "$path"
    echo "deleted_completed=$path"
  else
    echo "preserved_incomplete=$path"
  fi
done
REMOTE
fi
