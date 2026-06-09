#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LEROBOT_REF="${LEROBOT_REF:-d9ec3a6}"

cd "$ROOT"

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if [ ! -d vendor/lerobot/.git ]; then
  mkdir -p vendor
  git clone https://github.com/huggingface/lerobot.git vendor/lerobot
fi

cd vendor/lerobot
git fetch --tags --quiet
git checkout "$LEROBOT_REF"

python -m pip install -e ".[smolvla,feetech]"

echo "Done."
echo "Activate with: source $ROOT/.venv/bin/activate"

