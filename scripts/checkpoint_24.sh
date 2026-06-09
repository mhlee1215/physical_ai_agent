#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

if [ "$(uname -s)" = "Darwin" ]; then
  HOMEBREW_VULKAN="/opt/homebrew/opt/vulkan-loader/lib/libvulkan.1.dylib"
  HOMEBREW_MOLTENVK_ICD="/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
  if [ -f "$HOMEBREW_VULKAN" ] && [ -f "$HOMEBREW_MOLTENVK_ICD" ]; then
    export SAPIEN_VULKAN_LIBRARY_PATH="${SAPIEN_VULKAN_LIBRARY_PATH:-$HOMEBREW_VULKAN}"
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$HOMEBREW_MOLTENVK_ICD}"
  fi
fi

PYTHONPATH=src "$PYTHON_BIN" -B -m physical_ai_agent.checkpoints.checkpoint_24 "$@"
