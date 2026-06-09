#!/bin/sh
set -eu

API_BASE="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
NAME="${RUNPOD_NEW_POD_NAME:-physical_ai_pod}"
IMAGE_NAME="${RUNPOD_IMAGE_NAME:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA GeForce RTX 4090}"
GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
VCPU_COUNT="${RUNPOD_VCPU_COUNT:-16}"
CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-60}"
VOLUME_IN_GB="${RUNPOD_VOLUME_IN_GB:-0}"
VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH:-/workspace}"
PORTS_JSON="${RUNPOD_PORTS_JSON:-[\"8888/http\",\"22/tcp\"]}"
CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"
SUPPORT_PUBLIC_IP="${RUNPOD_SUPPORT_PUBLIC_IP:-true}"
PUBLIC_KEY_FILE="${RUNPOD_PUBLIC_KEY_FILE:-$HOME/.ssh/id_ed25519.pub}"
DOCKER_START_CMD_JSON="${RUNPOD_DOCKER_START_CMD_JSON:-[]}"
DOCKER_ENTRYPOINT_JSON="${RUNPOD_DOCKER_ENTRYPOINT_JSON:-[]}"

usage() {
  cat <<'USAGE'
Usage:
  RUNPOD_API_KEY=... RUNPOD_NETWORK_VOLUME_ID=... sh scripts/runpod_create_pod.sh
  RUNPOD_API_KEY=... RUNPOD_NETWORK_VOLUME_ID=... sh scripts/runpod_create_pod.sh --yes-create
  RUNPOD_API_KEY=... RUNPOD_NETWORK_VOLUME_ID=none sh scripts/runpod_create_pod.sh --yes-create

Defaults match the prior working Pod:
  image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
  gpu: NVIDIA GeForce RTX 4090 x1
  vcpu: 16
  container disk: 60 GB
  network volume mount: /workspace

Environment overrides:
  RUNPOD_NEW_POD_NAME
  RUNPOD_IMAGE_NAME
  RUNPOD_GPU_TYPE
  RUNPOD_GPU_COUNT
  RUNPOD_VCPU_COUNT
  RUNPOD_CONTAINER_DISK_GB
  RUNPOD_NETWORK_VOLUME_ID  Set to "none" to omit a network volume.
  RUNPOD_VOLUME_MOUNT_PATH
  RUNPOD_CLOUD_TYPE
  RUNPOD_SUPPORT_PUBLIC_IP
  RUNPOD_PUBLIC_KEY_FILE    Optional. Defaults to ~/.ssh/id_ed25519.pub and is
                            injected as SSH_PUBLIC_KEY/PUBLIC_KEY when the file
                            exists.
  RUNPOD_DOCKER_START_CMD_JSON
                            Optional JSON array for dockerStartCmd.
  RUNPOD_DOCKER_ENTRYPOINT_JSON
                            Optional JSON array for dockerEntrypoint.

Without --yes-create this prints the request body only and does not start billing.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
  echo "RUNPOD_API_KEY is required." >&2
  exit 2
fi

if [ -z "${RUNPOD_NETWORK_VOLUME_ID:-}" ]; then
  echo "RUNPOD_NETWORK_VOLUME_ID is required, for example: tchm4gxfvd" >&2
  echo "Set RUNPOD_NETWORK_VOLUME_ID=none to create an ephemeral Pod without a network volume." >&2
  exit 2
fi

BODY="$(
  python3 - "$PUBLIC_KEY_FILE" <<PY
import json
import sys

public_key_file = sys.argv[1]

body = {
    "cloudType": "$CLOUD_TYPE",
    "computeType": "GPU",
    "containerDiskInGb": int("$CONTAINER_DISK_GB"),
    "gpuCount": int("$GPU_COUNT"),
    "gpuTypeIds": ["$GPU_TYPE"],
    "gpuTypePriority": "availability",
    "imageName": "$IMAGE_NAME",
    "interruptible": False,
    "name": "$NAME",
    "ports": json.loads('$PORTS_JSON'),
    "supportPublicIp": "$SUPPORT_PUBLIC_IP".lower() == "true",
    "vcpuCount": int("$VCPU_COUNT"),
    "volumeInGb": int("$VOLUME_IN_GB"),
    "volumeMountPath": "$VOLUME_MOUNT_PATH",
}
docker_start_cmd = json.loads(r'''$DOCKER_START_CMD_JSON''')
docker_entrypoint = json.loads(r'''$DOCKER_ENTRYPOINT_JSON''')
if docker_entrypoint:
    body["dockerEntrypoint"] = docker_entrypoint
if docker_start_cmd:
    body["dockerStartCmd"] = docker_start_cmd
network_volume_id = "$RUNPOD_NETWORK_VOLUME_ID"
if network_volume_id and network_volume_id.lower() != "none":
    body["networkVolumeId"] = network_volume_id
try:
    with open(public_key_file, "r", encoding="utf-8") as handle:
        public_key = handle.read().strip()
except OSError:
    public_key = ""
if public_key:
    body["env"] = {"SSH_PUBLIC_KEY": public_key, "PUBLIC_KEY": public_key}
print(json.dumps(body, indent=2, sort_keys=True))
PY
)"

if [ "${1:-}" != "--yes-create" ]; then
  echo "$BODY"
  echo
  echo "dry_run=true"
  echo "Re-run with --yes-create to create the Pod and start billing."
  exit 0
fi

tmp="${TMPDIR:-/tmp}/runpod_create_pod_response.$$"
trap 'rm -f "$tmp"' EXIT HUP INT TERM

status="$(
  curl -sS \
    --request POST \
    --url "$API_BASE/pods" \
    --header "Authorization: Bearer $RUNPOD_API_KEY" \
    --header "Content-Type: application/json" \
    --data "$BODY" \
    --output "$tmp" \
    --write-out "%{http_code}"
)"

if [ -s "$tmp" ]; then
  python3 - "$tmp" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
for key in ("dockerStartCmd", "env", "containerRegistryAuth", "dockerArgs"):
    if key in data:
        data[key] = "<redacted>"
print(json.dumps(data, indent=2, sort_keys=True))
PY
fi
echo "http_status=$status"

case "$status" in
  2*) ;;
  *) exit 1 ;;
esac
