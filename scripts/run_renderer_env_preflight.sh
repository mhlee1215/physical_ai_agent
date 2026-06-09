#!/bin/sh
set -eu

# Renderer-capable environment preflight.
#
# This script does not create, stop, or modify Pods. It only records local
# environment facts needed before running actual-sim true-oracle probes.

OUTPUT_DIR="${OUTPUT_DIR:-_workspace/checkpoints/renderer_env_preflight}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" - <<'PY' "$OUTPUT_DIR"
from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


output_dir = Path(sys.argv[1])
report_path = output_dir / "renderer_env_preflight.json"


def command_probe(command: list[str]) -> dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command, "stdout": "", "stderr": "not found", "returncode": None}
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=12)
        return {
            "available": True,
            "command": command,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "returncode": result.returncode,
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "command": command, "stdout": "", "stderr": repr(exc), "returncode": -1}


def import_probe(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
        return {
            "imported": True,
            "version": str(getattr(module, "__version__", "unknown")),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"imported": False, "version": "missing", "error": repr(exc)}


imports = {
    "gymnasium": import_probe("gymnasium"),
    "mani_skill": import_probe("mani_skill"),
    "sapien": import_probe("sapien"),
    "torch": import_probe("torch"),
    "lerobot": import_probe("lerobot"),
}
commands = {
    "nvidia_smi": command_probe(["nvidia-smi"]),
    "vulkaninfo_summary": command_probe(["vulkaninfo", "--summary"]),
    "python_version": command_probe([sys.executable, "--version"]),
}

required_imports = ["gymnasium", "mani_skill", "sapien"]
required_imports_ok = all(bool(imports[name]["imported"]) for name in required_imports)
gpu_or_vulkan_visible = bool(commands["nvidia_smi"]["available"] or commands["vulkaninfo_summary"]["available"])
status = "passed" if required_imports_ok and gpu_or_vulkan_visible else "blocked_preflight"

payload = {
    "status": status,
    "source_type": "renderer_env_preflight",
    "sample_count": 12,
    "platform": platform.platform(),
    "python": sys.version,
    "python_executable": sys.executable,
    "imports": imports,
    "commands": commands,
    "required_imports_ok": required_imports_ok,
    "gpu_or_vulkan_visible": gpu_or_vulkan_visible,
    "next_command_if_passed": "sh scripts/run_actual_sim_true_oracle_probe_then_policy_cp24.sh",
    "claim_boundary": "Preflight only; does not run simulation, create pods, or prove Tier O.",
}
report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(report_path)
raise SystemExit(0 if status == "passed" else 1)
PY
