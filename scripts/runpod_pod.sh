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
    print_response "$tmp"
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

print_response() {
  response_path="$1"

  if [ "${RUNPOD_RAW_RESPONSE:-0}" = "1" ]; then
    cat "$response_path"
    printf '\n'
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$response_path" <<'PY'
import json
import sys

path = sys.argv[1]
redacted_keys = {
    "api_key",
    "apiKey",
    "authorization",
    "env",
    "jupyter_password",
    "password",
    "public_key",
    "secret",
    "token",
}


def redact(value):
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key in redacted_keys or key.lower() in redacted_keys:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


with open(path, "r", encoding="utf-8") as handle:
    body = handle.read()

try:
    parsed = json.loads(body)
except json.JSONDecodeError:
    print(body)
else:
    print(json.dumps(redact(parsed), indent=2, sort_keys=True))
PY
    return
  fi

  echo "[response redacted: install python3 or set RUNPOD_RAW_RESPONSE=1 to print raw JSON]"
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
