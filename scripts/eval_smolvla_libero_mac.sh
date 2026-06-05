#!/bin/sh
set -eu

PROJECT_DIR="${PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/_workspace/local_results/smolvla_libero_mac_preflight_$STAMP}"
REPORT_PATH="$OUTPUT_ROOT/smolvla_libero_mac_preflight.md"

mkdir -p "$OUTPUT_ROOT"

cd "$PROJECT_DIR"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
UNAME="$(uname -s)"

cat > "$REPORT_PATH" <<EOF
# SmolVLA LIBERO Mac Preflight

- status: preflight
- git_commit: \`$GIT_COMMIT\`
- platform: \`$UNAME\`
- python_bin: \`$PYTHON_BIN\`

## Result

This is not a paper-comparable LIBERO evaluation.

LeRobot LIBERO evaluation requires Linux. On macOS, use this script only to
check local readiness and to document why the comparable benchmark must run on
RunPod or another Linux GPU machine.

## Comparable Linux Command

\`\`\`bash
sh scripts/eval_smolvla_libero_linux.sh
\`\`\`

## Smoke Linux Command

\`\`\`bash
LIBERO_TASKS=libero_spatial LIBERO_N_EPISODES=1 sh scripts/eval_smolvla_libero_linux.sh
\`\`\`

## Local Checks

EOF

{
  echo "### uname"
  echo
  echo '```text'
  uname -a
  echo '```'
  echo
  echo "### Python"
  echo
  echo '```text'
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    "$PYTHON_BIN" --version
  else
    echo "$PYTHON_BIN not found"
  fi
  echo '```'
  echo
  echo "### Optional Imports"
  echo
  echo '```text'
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    "$PYTHON_BIN" - <<'PY'
import importlib.util
for name in ("lerobot", "torch", "libero", "mujoco"):
    print(f"{name}: {bool(importlib.util.find_spec(name))}")
PY
  else
    echo "skipped"
  fi
  echo '```'
} >> "$REPORT_PATH"

cat "$REPORT_PATH"
echo "report=$REPORT_PATH"
