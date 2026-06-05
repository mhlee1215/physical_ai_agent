#!/bin/sh
set -eu

API_BASE="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"

usage() {
  cat <<'USAGE'
Usage:
  RUNPOD_API_KEY=... RUNPOD_POD_ID=... sh scripts/runpod_pod.sh status
  RUNPOD_API_KEY=... RUNPOD_POD_ID=... sh scripts/runpod_pod.sh stop
  RUNPOD_API_KEY=... RUNPOD_POD_ID=... sh scripts/runpod_pod.sh start
  RUNPOD_API_KEY=... sh scripts/runpod_pod.sh list

Dangerous:
  RUNPOD_API_KEY=... RUNPOD_POD_ID=... sh scripts/runpod_pod.sh terminate --yes-terminate

Environment:
  RUNPOD_API_KEY   Required. RunPod REST API key.
  RUNPOD_POD_ID    Required except for "list".
  RUNPOD_API_BASE  Optional. Defaults to https://rest.runpod.io/v1.

Notes:
  "stop" releases the GPU when RunPod allows the Pod to stop.
  RunPod may reject stop for Pods attached to network volumes; use the console
  or explicit terminate only after confirming workspace data is on the network volume.
USAGE
}

require_api_key() {
  if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "RUNPOD_API_KEY is required." >&2
    echo "Create one in RunPod account settings, then export it locally." >&2
    exit 2
  fi
}

require_pod_id() {
  if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "RUNPOD_POD_ID is required for action: $ACTION" >&2
    echo "Example: export RUNPOD_POD_ID='v605dhuhdkjbfm'" >&2
    exit 2
  fi
}

request() {
  method="$1"
  path="$2"
  tmp="${TMPDIR:-/tmp}/runpod_pod_response.$$"
  trap 'rm -f "$tmp"' EXIT HUP INT TERM

  set +e
  status="$(
    curl -sS \
      --request "$method" \
      --url "$API_BASE$path" \
      --header "Authorization: Bearer $RUNPOD_API_KEY" \
      --output "$tmp" \
      --write-out "%{http_code}"
  )"
  curl_status=$?
  set -e

  if [ -s "$tmp" ]; then
    cat "$tmp"
    printf '\n'
  fi
  echo "http_status=$status"

  if [ "$curl_status" -ne 0 ]; then
    exit "$curl_status"
  fi

  case "$status" in
    2*) ;;
    *)
      echo "RunPod API request failed: $method $path" >&2
      exit 1
      ;;
  esac
}

ACTION="${1:-}"

case "$ACTION" in
  ""|-h|--help|help)
    usage
    ;;
  list)
    require_api_key
    request GET "/pods"
    ;;
  status|get)
    require_api_key
    require_pod_id
    request GET "/pods/$RUNPOD_POD_ID"
    ;;
  stop)
    require_api_key
    require_pod_id
    request POST "/pods/$RUNPOD_POD_ID/stop"
    ;;
  start)
    require_api_key
    require_pod_id
    request POST "/pods/$RUNPOD_POD_ID/start"
    ;;
  terminate|delete)
    require_api_key
    require_pod_id
    if [ "${2:-}" != "--yes-terminate" ]; then
      echo "Refusing to terminate without --yes-terminate." >&2
      echo "Terminate deletes the Pod and non-network-volume data." >&2
      exit 2
    fi
    request DELETE "/pods/$RUNPOD_POD_ID"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    usage >&2
    exit 2
    ;;
esac
