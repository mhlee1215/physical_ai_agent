#!/usr/bin/env sh
set -eu

# Probe multiple GPU types on RunPod and try to create one pod.
# Stops at the first successful creation.

API_KEY="${RUNPOD_API_KEY:-}"
CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"
NETWORK_VOLUME_ID="${RUNPOD_NETWORK_VOLUME_ID:-}"
IMAGE_NAME="${RUNPOD_IMAGE_NAME:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
VCPU_COUNT="${RUNPOD_VCPU_COUNT:-16}"
CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-60}"
VOLUME_IN_GB="${RUNPOD_VOLUME_IN_GB:-0}"
PORTS_JSON="${RUNPOD_PORTS_JSON:-[\"8888/http\",\"22/tcp\"]}"
NAME_PREFIX="${RUNPOD_NEW_POD_NAME_PREFIX:-physical_ai_probe}"
MAX_ATTEMPTS="${RUNPOD_MAX_ATTEMPTS:-40}"
DOCKER_START_CMD_JSON="${RUNPOD_DOCKER_START_CMD_JSON:-[]}"
DOCKER_ENTRYPOINT_JSON="${RUNPOD_DOCKER_ENTRYPOINT_JSON:-[]}"

usage() {
  cat <<'USAGE'
Usage:
  RUNPOD_API_KEY=... RUNPOD_NETWORK_VOLUME_ID=... sh scripts/runpod_probe_gpus.sh

Optional overrides:
  RUNPOD_CLOUD_TYPE          COMMUNITY or SECURE (default COMMUNITY)
  RUNPOD_VCPU_COUNT           vCPU override (default 16)
  RUNPOD_CONTAINER_DISK_GB    container disk (default 60)
  RUNPOD_MAX_ATTEMPTS         max types to probe before stop (default 40)
  RUNPOD_NEW_POD_NAME_PREFIX  prefix for temporary pod names
  RUNPOD_DOCKER_START_CMD_JSON
  RUNPOD_DOCKER_ENTRYPOINT_JSON

Notes:
- This script tries a curated set of GPU type IDs and prints API outcome per
  attempt.
- It keeps the first successful Pod running. Stop the Pod manually when you are done.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ -z "$API_KEY" ]; then
  echo "RUNPOD_API_KEY is required." >&2
  exit 2
fi

if [ -z "$NETWORK_VOLUME_ID" ]; then
  echo "RUNPOD_NETWORK_VOLUME_ID is required." >&2
  echo "Example: export RUNPOD_NETWORK_VOLUME_ID=tchm4gxfvd" >&2
  exit 2
fi

if [ "$CLOUD_TYPE" != "COMMUNITY" ] && [ "$CLOUD_TYPE" != "SECURE" ]; then
  echo "RUNPOD_CLOUD_TYPE must be COMMUNITY or SECURE" >&2
  exit 2
fi

set -- \
  "NVIDIA GeForce RTX 4090" \
  "NVIDIA GeForce RTX 3090" \
  "NVIDIA GeForce RTX 4070 Ti" \
  "NVIDIA GeForce RTX 3080 Ti" \
  "NVIDIA GeForce RTX 3080" \
  "NVIDIA GeForce RTX 3070" \
  "NVIDIA L4" \
  "NVIDIA L40S" \
  "NVIDIA H100 PCIe" \
  "NVIDIA H100 NVL" \
  "NVIDIA H200" \
  "NVIDIA H100 80GB HBM3" \
  "NVIDIA A40" \
  "NVIDIA RTX A4000" \
  "NVIDIA RTX A5000" \
  "NVIDIA RTX A4500" \
  "NVIDIA RTX 4000 Ada Generation" \
  "NVIDIA RTX 5000 Ada Generation" \
  "NVIDIA RTX 2000 Ada Generation" \
  "NVIDIA A100 80GB PCIe" \
  "NVIDIA A100-SXM4-80GB"

attempts=0
created_pod_id=""
created_gpu=""

while [ "$#" -gt 0 ] && [ "$attempts" -lt "$MAX_ATTEMPTS" ]; do
  gpu=$1
  shift
  attempts=$((attempts + 1))

  stamp="$(date +%s)"
  pod_name="${NAME_PREFIX}-${stamp}"
  echo ""
  echo "== TRY [$attempts] $gpu on $CLOUD_TYPE =="

  set +e
  response="$(
    RUNPOD_API_KEY="$API_KEY" \
    RUNPOD_NETWORK_VOLUME_ID="$NETWORK_VOLUME_ID" \
    RUNPOD_IMAGE_NAME="$IMAGE_NAME" \
    RUNPOD_GPU_TYPE="$gpu" \
    RUNPOD_GPU_COUNT=1 \
    RUNPOD_VCPU_COUNT="$VCPU_COUNT" \
    RUNPOD_CONTAINER_DISK_GB="$CONTAINER_DISK_GB" \
    RUNPOD_VOLUME_IN_GB="$VOLUME_IN_GB" \
    RUNPOD_PORTS_JSON="$PORTS_JSON" \
    RUNPOD_DOCKER_START_CMD_JSON="$DOCKER_START_CMD_JSON" \
    RUNPOD_DOCKER_ENTRYPOINT_JSON="$DOCKER_ENTRYPOINT_JSON" \
    RUNPOD_CLOUD_TYPE="$CLOUD_TYPE" \
    RUNPOD_NEW_POD_NAME="$pod_name" \
    sh scripts/runpod_create_pod.sh --yes-create 2>&1
  )"
  set -e
  set +e
  status="$(printf '%s\n' "$response" | awk -F= '/^http_status=/{print $2}')"
  set -e
  status="${status:-unknown}"

  echo "$response" | sed -n '1,12p'

  if [ "$status" = "200" ] || [ "$status" = "201" ]; then
    created_pod_id="$(printf '%s\n' "$response" | awk -F'"' '/"id":/{print $4; exit}')"
    created_gpu="$gpu"
    echo ""
    echo "Success. Pod created:"
    echo "   cloud=$CLOUD_TYPE gpu=$created_gpu pod_id=$created_pod_id"
    break
  fi

  if [ "$status" = "400" ]; then
    echo "   schema/enum error (skip or fix GPU spelling)"
  else
    echo "   unavailable for now (API returned $status)"
  fi
done

if [ -n "$created_pod_id" ]; then
  echo ""
  echo "KEEP pod_id=$created_pod_id for next steps:"
  echo "  RUNPOD_POD_ID=$created_pod_id"
  exit 0
fi

echo ""
echo "No GPU type from the probe list could be created under cloud=$CLOUD_TYPE right now."
echo "Try again later or add more GPU entries."
exit 3
