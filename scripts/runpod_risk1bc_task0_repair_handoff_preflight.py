#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


POD_LOCAL_PREREQ_CODE = r'''
import ctypes.util
import json
import os
import sys

required_libs = {
    "OSMesa": ctypes.util.find_library("OSMesa"),
    "EGL": ctypes.util.find_library("EGL"),
    "GL": ctypes.util.find_library("GL"),
}
payload = {
    "status": "PASS",
    "python_executable": sys.executable,
    "python_version": sys.version.split()[0],
    "required_python": "/usr/bin/python3.12",
    "required_python_ok": sys.executable == "/usr/bin/python3.12",
    "required_libs": required_libs,
    "mujoco_gl": os.environ.get("MUJOCO_GL"),
    "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
}
missing = [name for name, value in required_libs.items() if not value]
if not payload["required_python_ok"] or missing:
    payload["status"] = "BLOCKED"
    payload["blocker_category"] = "RUNPOD_TASK0_REPAIR_POD_LOCAL_PREREQ_BLOCKED"
    payload["missing_gl_libs"] = missing
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["status"] == "PASS" else 2)
'''


QWEN7B_READINESS_CODE = r'''
import json
import os
import sys
import time

model_id = sys.argv[1]
started = time.time()
payload = {
    "status": "PASS",
    "model_id": model_id,
    "backend": "transformers_model_load",
    "hf_home": os.environ.get("HF_HOME"),
    "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
    "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
}
try:
    import torch
    import transformers
    from transformers import AutoProcessor
except Exception as exc:
    payload.update({
        "status": "BLOCKED",
        "blocker_category": "RUNPOD_QWEN7B_READINESS_IMPORT_BLOCKED",
        "error_type": type(exc).__name__,
        "error": str(exc)[:800],
    })
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(2)

payload["torch_version"] = getattr(torch, "__version__", "unknown")
payload["cuda_available"] = bool(torch.cuda.is_available())
payload["transformers_version"] = getattr(transformers, "__version__", "unknown")

loader_attempts = []
processor = None
model = None
try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    payload["processor_class"] = type(processor).__name__
except Exception as exc:
    payload.update({
        "status": "BLOCKED",
        "blocker_category": "RUNPOD_QWEN7B_READINESS_PROCESSOR_LOAD_BLOCKED",
        "error_type": type(exc).__name__,
        "error": str(exc)[:800],
    })
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(2)

for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
    try:
        model_cls = getattr(transformers, class_name)
    except Exception as exc:
        loader_attempts.append({"class": class_name, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        continue
    try:
        model = model_cls.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        loader_attempts.append({"class": class_name, "ok": True})
        payload["model_loader_class"] = class_name
        payload["model_class"] = type(model).__name__
        break
    except Exception as exc:
        loader_attempts.append({"class": class_name, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})

payload["model_loader_attempts"] = loader_attempts
if model is None:
    payload.update({
        "status": "BLOCKED",
        "blocker_category": "RUNPOD_QWEN7B_READINESS_MODEL_LOAD_BLOCKED",
    })
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(2)

payload["elapsed_sec"] = round(time.time() - started, 3)
print(json.dumps(payload, sort_keys=True))
'''


@dataclass(frozen=True)
class Phase:
    name: str
    argv: list[str]
    timeout_sec: int
    env: dict[str, str]
    blocker_category: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one bounded RunPod Manager preflight for the Risk1-B/C task0 repair-only lane. "
            "It prepares handoff readiness only; it never runs the Researcher experiment."
        )
    )
    parser.add_argument("--project-dir", default="/workspace/physical-ai/physical_ai_agent")
    parser.add_argument("--work-root", default="/workspace/physical-ai")
    parser.add_argument(
        "--output-dir",
        default="/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/ita_risk_probes/risk1bc_task0_repair_handoff_preflight",
    )
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--policy-num-steps", type=int, default=10)
    parser.add_argument("--policy-n-action-steps", type=int, default=15)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--system-python", default="/usr/bin/python3.12")
    parser.add_argument("--canonical-venv", default="/workspace/physical-ai/envs/lerobot_py312")
    parser.add_argument("--vlm-venv", default="/workspace/physical-ai/envs/risk1b_vlm_py312")
    parser.add_argument("--libero-config-path", default="/workspace/physical-ai/libero_config")
    parser.add_argument("--libero-assets-dir", default="/workspace/physical-ai/libero_assets")
    parser.add_argument("--hf-home", default="/workspace/physical-ai/hf_home")
    parser.add_argument("--renderer-backend", choices=("osmesa", "egl", "auto"), default="osmesa")
    parser.add_argument("--phase-timeout-sec", type=int, default=900)
    parser.add_argument("--qwen-readiness-timeout-sec", type=int, default=2400)
    parser.add_argument("--context-timeout-sec", type=int, default=900)
    parser.add_argument(
        "--qwen-readiness-mode",
        choices=("dependency-check", "model-load"),
        default="model-load",
        help="Use dependency-check for a light class gate, or model-load for Qwen 7B cache/model readiness.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def build_base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    project_dir = str(Path(args.project_dir))
    work_root = str(Path(args.work_root))
    canonical_python = str(Path(args.canonical_venv) / "bin" / "python")
    vlm_python = str(Path(args.vlm_venv) / "bin" / "python")
    env.update(
        {
            "PROJECT_DIR": project_dir,
            "WORK_ROOT": work_root,
            "RUNPOD_ENV_PROFILE": "volume",
            "PYTHONPATH": str(Path(project_dir) / "src"),
            "PY312_VENV": str(Path(args.canonical_venv)),
            "PYTHON_BIN": canonical_python,
            "LIBERO_CONFIG_PATH": str(Path(args.libero_config_path)),
            "LIBERO_CONFIG_DIR": str(Path(args.libero_config_path)),
            "LIBERO_ASSETS_DIR": str(Path(args.libero_assets_dir)),
            "VLM_VENV": str(Path(args.vlm_venv)),
            "VLM_PYTHON_BIN": vlm_python,
            "MODEL_ID": args.model_id,
            "HF_HOME": str(Path(args.hf_home)),
            "HF_HUB_CACHE": str(Path(args.hf_home) / "hub"),
            "TRANSFORMERS_CACHE": str(Path(args.hf_home) / "transformers"),
            "RISK1B_VLM_HF_HOME": str(Path(args.hf_home)),
            "MUJOCO_GL": args.renderer_backend,
        }
    )
    if args.renderer_backend == "osmesa":
        env["PYOPENGL_PLATFORM"] = "osmesa"
    elif args.renderer_backend == "egl":
        env["PYOPENGL_PLATFORM"] = "egl"
    return env


def build_phase_specs(args: argparse.Namespace) -> list[Phase]:
    project_dir = Path(args.project_dir)
    base_env = build_base_env(args)
    canonical_python = str(Path(args.canonical_venv) / "bin" / "python")
    vlm_python = str(Path(args.vlm_venv) / "bin" / "python")
    output_dir = Path(args.output_dir)
    context_preflight_dir = output_dir / "context_capture_preflight"
    install_script = str(project_dir / "scripts" / "install" / "runpod_install.sh")
    check_script = str(project_dir / "scripts" / "install" / "runpod_check.sh")
    generator_script = str(project_dir / "scripts" / "generate_risk1b_vlm_subgoals.py")
    context_preflight_script = str(project_dir / "scripts" / "preflight_risk1b_context_capture.py")

    if args.qwen_readiness_mode == "model-load":
        qwen_argv = [vlm_python, "-B", "-c", QWEN7B_READINESS_CODE, args.model_id]
        qwen_timeout = args.qwen_readiness_timeout_sec
    else:
        qwen_argv = [
            vlm_python,
            "-B",
            generator_script,
            "--backend",
            "transformers",
            "--dependency-check-only",
            "--model-id",
            args.model_id,
            "--json",
        ]
        qwen_timeout = args.phase_timeout_sec

    return [
        Phase(
            name="pod_local_prereq",
            argv=[args.system_python, "-B", "-c", POD_LOCAL_PREREQ_CODE],
            timeout_sec=180,
            env=base_env,
            blocker_category="RUNPOD_TASK0_REPAIR_POD_LOCAL_PREREQ_BLOCKED",
        ),
        Phase(
            name="canonical_env_install",
            argv=["sh", install_script, "--component", "libero-smolvla"],
            timeout_sec=args.phase_timeout_sec,
            env=base_env,
            blocker_category="RUNPOD_TASK0_REPAIR_CANONICAL_ENV_INSTALL_BLOCKED",
        ),
        Phase(
            name="canonical_env_gate",
            argv=["sh", check_script, "--component", "libero-smolvla"],
            timeout_sec=args.phase_timeout_sec,
            env=base_env,
            blocker_category="RUNPOD_TASK0_REPAIR_CANONICAL_ENV_GATE_BLOCKED",
        ),
        Phase(
            name="vlm_env_install",
            argv=["sh", install_script, "--component", "risk1b-vlm"],
            timeout_sec=args.phase_timeout_sec,
            env=base_env,
            blocker_category="RUNPOD_TASK0_REPAIR_VLM_ENV_INSTALL_BLOCKED",
        ),
        Phase(
            name="vlm_env_gate",
            argv=["sh", check_script, "--component", "risk1b-vlm"],
            timeout_sec=args.phase_timeout_sec,
            env=base_env,
            blocker_category="RUNPOD_TASK0_REPAIR_VLM_ENV_GATE_BLOCKED",
        ),
        Phase(
            name="qwen7b_readiness",
            argv=qwen_argv,
            timeout_sec=qwen_timeout,
            env=base_env,
            blocker_category="RUNPOD_TASK0_REPAIR_QWEN7B_READINESS_BLOCKED",
        ),
        Phase(
            name="shallow_osmesa_context_preflight",
            argv=[
                canonical_python,
                "-B",
                context_preflight_script,
                "--python-bin",
                canonical_python,
                "--suite",
                args.suite,
                "--task-id",
                str(args.task_id),
                "--seed",
                str(args.seed),
                "--policy-path",
                args.policy_path,
                "--policy-num-steps",
                str(args.policy_num_steps),
                "--policy-n-action-steps",
                str(args.policy_n_action_steps),
                "--renderer-backend",
                args.renderer_backend,
                "--actual-timeout-sec",
                str(args.context_timeout_sec),
                "--output-dir",
                str(context_preflight_dir),
                "--json",
            ],
            timeout_sec=args.context_timeout_sec + 60,
            env=base_env,
            blocker_category="RUNPOD_TASK0_REPAIR_CONTEXT_PREFLIGHT_BLOCKED",
        ),
    ]


def parse_phase_payload(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def cleanup_instruction(status: str) -> str:
    if status == "ENV_READY_HANDOFF_READY":
        return (
            "Do not run the Researcher experiment from this wrapper. Hand off the same pod/env to Researcher; "
            "after the repair-only task0 experiment artifacts are fetched and no active run remains, stop the pod."
        )
    return "Fetch this preflight output directory and stop the pod; do not hand off to Researcher."


def run_phase(phase: Phase, output_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    phase_dir = output_dir / phase.name
    phase_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = phase_dir / "stdout.log"
    stderr_path = phase_dir / "stderr.log"
    started = time.time()
    result: dict[str, Any] = {
        "phase": phase.name,
        "argv": phase.argv,
        "timeout_sec": phase.timeout_sec,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "blocker_category": phase.blocker_category,
    }
    if dry_run:
        result.update({"status": "DRY_RUN", "returncode": None, "elapsed_sec": 0.0})
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return result
    try:
        completed = subprocess.run(
            phase.argv,
            cwd=Path.cwd(),
            env=phase.env,
            capture_output=True,
            text=True,
            timeout=phase.timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else f"timeout after {phase.timeout_sec}s"
        completed = subprocess.CompletedProcess(phase.argv, 124, stdout=stdout, stderr=stderr)

    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    payload = parse_phase_payload(completed.stdout or "")
    ok = completed.returncode == 0
    result.update(
        {
            "status": "PASS" if ok else "BLOCKED",
            "returncode": completed.returncode,
            "elapsed_sec": round(time.time() - started, 3),
            "payload": payload,
        }
    )
    if not ok and isinstance(payload, dict) and payload.get("blocker_category"):
        result["blocker_category"] = payload["blocker_category"]
    return result


def run_preflight(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phases = build_phase_specs(args)
    phase_results: list[dict[str, Any]] = []
    final_status = "ENV_READY_HANDOFF_READY"
    blocked_phase: dict[str, Any] | None = None

    for phase in phases:
        result = run_phase(phase, output_dir, dry_run=args.dry_run)
        phase_results.append(result)
        if not args.dry_run and result["status"] != "PASS":
            final_status = "BLOCKED"
            blocked_phase = result
            break

    if args.dry_run:
        final_status = "DRY_RUN"

    report: dict[str, Any] = {
        "operation": "risk1bc_task0_repair_handoff_preflight",
        "status": final_status,
        "project_dir": args.project_dir,
        "work_root": args.work_root,
        "suite": args.suite,
        "task_id": args.task_id,
        "seed": args.seed,
        "renderer_backend": args.renderer_backend,
        "model_id": args.model_id,
        "qwen_readiness_mode": args.qwen_readiness_mode,
        "phase_results": phase_results,
        "cleanup_instruction": cleanup_instruction(final_status),
        "claim_boundary": (
            "Infra handoff readiness only. This wrapper does not run Qwen generation, "
            "SmolVLA Risk1-B/C probes, benchmark evaluation, or any paper-facing experiment."
        ),
    }
    if blocked_phase is not None:
        report["blocked_phase"] = blocked_phase["phase"]
        report["blocker_category"] = blocked_phase.get("blocker_category")
        report["blocked_phase_logs"] = {
            "stdout": blocked_phase.get("stdout_log"),
            "stderr": blocked_phase.get("stderr_log"),
        }
    return (0 if final_status in {"ENV_READY_HANDOFF_READY", "DRY_RUN"} else 2), report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, report = run_preflight(args)
    report_path = Path(args.output_dir) / "risk1bc_task0_repair_handoff_preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"status={report['status']}")
        print(f"report_path={report_path}")
        if report.get("blocked_phase"):
            print(f"blocked_phase={report['blocked_phase']}")
            print(f"blocker_category={report.get('blocker_category')}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
