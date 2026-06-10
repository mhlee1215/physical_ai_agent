#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from physical_ai_agent.imagine_then_act.risk_probes import (
    RISK1B_CANDIDATE_MODELS,
    RISK1B_REQUIRED_FIELDS,
    validate_risk1b_subgoal_records,
)


PROMPT_TEMPLATE = """You are generating grounded subgoal candidates for a frozen SmolVLA LIBERO policy.

Return only valid JSON with this schema:
{{
  "subgoals": [
    {{
      "subgoal_text": "...",
      "strategy_axis": "...",
      "target_object": "...",
      "target_region_or_point": "...",
      "stop_condition": "...",
      "confidence": 0.0,
      "rationale": "optional short explanation"
    }}
  ]
}}

Rules:
- Produce exactly {num_subgoals} subgoals.
- The first subgoal must be the baseline/original task.
- Every subgoal must preserve the same final task goal.
- Prefer grounded, visually actionable axes such as alignment_before_contact,
  object_centric_direction, gripper_alignment, and contact_first.
- Do not invent a different object or target.

Context:
suite={suite}
task_id={task_id}
seed={seed}
task_description={task_description}
context_summary={context_summary}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Risk1-B external VLM subgoal JSON. The mock/fixture backends are "
            "contract-only and cannot count as Risk1-B PASS evidence."
        )
    )
    parser.add_argument("--backend", choices=("mock", "fixture", "transformers"), default="mock")
    parser.add_argument("--model-id", choices=RISK1B_CANDIDATE_MODELS, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--num-subgoals", type=int, default=5)
    parser.add_argument("--task-description", default="Complete the LIBERO task from the current observation.")
    parser.add_argument("--context-image", default=None, help="Optional start observation/contact sheet image for VLM input.")
    parser.add_argument("--context-json", default=None, help="Optional JSON context summary to include in the prompt.")
    parser.add_argument("--fixture-output", default=None, help="Raw model output fixture for dependency-light local tests.")
    parser.add_argument("--output-dir", default="_workspace/runpod_results/ita_risk_probes")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument(
        "--dependency-check-only",
        action="store_true",
        help="For RunPod diagnostics: check transformers backend imports/classes without loading model weights.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def default_output_path(output_dir: str, model_id: str, suite: str, task_id: int, seed: int) -> Path:
    model_slug = sanitize_slug(model_id.split("/")[-1])
    return Path(output_dir) / f"risk1b_subgoals_{model_slug}_{sanitize_slug(suite)}_task{task_id}_seed{seed}.json"


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "unknown"


def read_context_summary(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"context_error": f"{type(exc).__name__}: {str(exc)[:200]}", "context_json": path}
    return payload if isinstance(payload, dict) else {"context": payload}


def is_actual_context(context_summary: dict[str, Any]) -> bool:
    provenance = context_summary.get("provenance")
    if isinstance(provenance, dict) and provenance.get("actual_context") is True:
        return True
    return context_summary.get("actual_context") is True


def build_generation_prompt(args: argparse.Namespace, context_summary: dict[str, Any]) -> str:
    compact_context = json.dumps(context_summary, sort_keys=True)[:2000] if context_summary else "none"
    return PROMPT_TEMPLATE.format(
        num_subgoals=args.num_subgoals,
        suite=args.suite,
        task_id=args.task_id,
        seed=args.seed,
        task_description=args.task_description,
        context_summary=compact_context,
    )


def generate_mock_output(args: argparse.Namespace) -> str:
    subgoals = [
        {
            "subgoal_text": args.task_description,
            "strategy_axis": "baseline",
            "target_object": "task object",
            "target_region_or_point": "task target region",
            "stop_condition": "original task success condition is reached",
            "confidence": 1.0,
            "rationale": "mock baseline for schema plumbing only",
        },
        {
            "subgoal_text": "Align the gripper with the task object before first contact while preserving the final goal.",
            "strategy_axis": "alignment_before_contact",
            "target_object": "task object",
            "target_region_or_point": "pre-contact alignment region",
            "stop_condition": "gripper is aligned and close enough for stable contact",
            "confidence": 0.75,
            "rationale": "mock contract branch",
        },
        {
            "subgoal_text": "Approach the task object from the most open visible side before manipulation.",
            "strategy_axis": "object_centric_direction",
            "target_object": "task object",
            "target_region_or_point": "object-relative open side",
            "stop_condition": "object can be manipulated without collision from that side",
            "confidence": 0.72,
            "rationale": "mock contract branch",
        },
        {
            "subgoal_text": "Prioritize precise gripper pose alignment before closing or pushing.",
            "strategy_axis": "gripper_alignment",
            "target_object": "gripper and task object",
            "target_region_or_point": "pre-grasp or pre-push alignment point",
            "stop_condition": "gripper pose is aligned with the intended contact direction",
            "confidence": 0.7,
            "rationale": "mock contract branch",
        },
        {
            "subgoal_text": "Make short-horizon contact that visibly improves progress toward the same final goal.",
            "strategy_axis": "contact_first",
            "target_object": "task object",
            "target_region_or_point": "nearest useful contact point",
            "stop_condition": "first contact produces visible progress toward the target",
            "confidence": 0.65,
            "rationale": "mock contract branch",
        },
    ][: args.num_subgoals]
    return json.dumps({"subgoals": subgoals}, indent=2)


def generate_transformers_output(args: argparse.Namespace, prompt: str) -> str:
    components = resolve_transformers_components(args.model_id)
    torch = components["torch"]
    image_module = components["pil_image"]
    processor_cls = components["processor_cls"]
    model_cls = components["model_cls"]

    processor = processor_cls.from_pretrained(args.model_id, trust_remote_code=True)
    model = model_cls.from_pretrained(
        args.model_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=True,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if args.context_image:
        content.insert(0, {"type": "image", "image": image_module.open(args.context_image).convert("RGB")})
    messages = [{"role": "user", "content": content}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[content[0]["image"]] if args.context_image else None, return_tensors="pt")
    except TypeError:
        text = prompt
        inputs = processor(text=text, images=content[0]["image"] if args.context_image else None, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return decoded


def resolve_transformers_components(model_id: str | None = None) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    missing: list[str] = []
    try:
        torch = importlib.import_module("torch")
        diagnostics["torch_version"] = getattr(torch, "__version__", "unknown")
        diagnostics["torch_cuda_available"] = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
    except Exception as exc:  # noqa: BLE001
        torch = None
        missing.append(f"torch ({type(exc).__name__}: {str(exc)[:200]})")
    try:
        pil_image = importlib.import_module("PIL.Image")
        diagnostics["pil_import"] = "PIL.Image"
    except Exception as exc:  # noqa: BLE001
        pil_image = None
        missing.append(f"PIL.Image ({type(exc).__name__}: {str(exc)[:200]})")
    try:
        transformers = importlib.import_module("transformers")
        diagnostics["transformers_version"] = getattr(transformers, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        transformers = None
        missing.append(f"transformers ({type(exc).__name__}: {str(exc)[:200]})")
    if missing:
        raise RuntimeError(
            "RUNPOD_VLM_ENV_OR_MODEL_LOAD_BLOCKED: missing python import(s): "
            + "; ".join(missing)
        )
    compatibility = diagnose_torch_transformers_compatibility(torch, transformers)
    diagnostics.update(compatibility)
    if compatibility.get("compatibility_blocker"):
        raise RuntimeError(str(compatibility["compatibility_blocker"]))
    processor_cls, processor_name, processor_attempts = select_transformers_processor_class(transformers, model_id or "")
    model_cls, class_name = select_transformers_model_class(transformers)
    diagnostics["processor_loader_class"] = processor_name
    diagnostics["processor_loader_attempts"] = processor_attempts
    diagnostics["model_loader_class"] = class_name
    return {
        "torch": torch,
        "pil_image": pil_image,
        "processor_cls": processor_cls,
        "model_cls": model_cls,
        "diagnostics": diagnostics,
    }


def diagnose_torch_transformers_compatibility(torch_module: Any, transformers_module: Any) -> dict[str, Any]:
    torch_version = str(getattr(torch_module, "__version__", "unknown"))
    transformers_version = str(getattr(transformers_module, "__version__", "unknown"))
    has_float8_e8m0fnu = hasattr(torch_module, "float8_e8m0fnu")
    has_float8_e5m2 = hasattr(torch_module, "float8_e5m2")
    diagnostics: dict[str, Any] = {
        "torch_float8_e8m0fnu_available": bool(has_float8_e8m0fnu),
        "torch_float8_e5m2_available": bool(has_float8_e5m2),
        "torch_transformers_compatibility": "unknown",
    }
    version_tuple = parse_version_prefix(transformers_version)
    if version_tuple >= (5, 10) and not has_float8_e8m0fnu:
        diagnostics["torch_transformers_compatibility"] = "blocked"
        diagnostics["compatibility_blocker"] = (
            "RUNPOD_VLM_ENV_OR_MODEL_LOAD_BLOCKED_COMPATIBILITY: "
            f"transformers {transformers_version} is installed with torch {torch_version}, "
            "but torch.float8_e8m0fnu is absent. This Transformers lazy processor path is "
            "incompatible with the canonical torch==2.5.1+cu124 LeRobot/LIBERO env. "
            "Use the separate Risk1-B VLM env pinned to transformers==4.49.0, or change "
            "torch only in that separate VLM env after validation."
        )
    else:
        diagnostics["torch_transformers_compatibility"] = "pass"
    return diagnostics


def parse_version_prefix(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def select_transformers_processor_class(transformers_module: Any, model_id: str) -> tuple[Any, str, list[dict[str, Any]]]:
    model_lower = model_id.lower()
    candidates = ["AutoProcessor"]
    if "qwen2.5-vl" in model_lower or "qwen2_5_vl" in model_lower:
        candidates.extend(("Qwen2_5_VLProcessor", "Qwen2VLProcessor"))
    if "gemma-3" in model_lower or "gemma3" in model_lower:
        candidates.append("Gemma3Processor")
    attempts: list[dict[str, Any]] = []
    for name in dict.fromkeys(candidates):
        try:
            cls = getattr(transformers_module, name)
        except Exception as exc:  # noqa: BLE001 - Transformers lazy loader raises here for optional deps.
            attempts.append(
                {
                    "class": name,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            continue
        if cls is not None:
            attempts.append({"class": name, "ok": True})
            return cls, name, attempts
        attempts.append({"class": name, "ok": False, "error_type": "AttributeError", "error": "attribute is None"})
    raise RuntimeError(
        "RUNPOD_VLM_ENV_OR_MODEL_LOAD_BLOCKED_PROCESSOR_LOADER: transformers import passed, but no supported "
        "processor class could be resolved. Attempts: "
        + json.dumps(attempts, sort_keys=True)
        + ". This usually means a Transformers lazy-loader optional dependency failure or an incompatible "
        "Transformers/model combination."
    )


def select_transformers_model_class(transformers_module: Any) -> tuple[Any, str]:
    candidates = (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    )
    for name in candidates:
        try:
            cls = getattr(transformers_module, name)
        except Exception:  # noqa: BLE001 - keep trying less specific loaders.
            cls = None
        if cls is not None:
            return cls, name
    raise RuntimeError(
        "RUNPOD_VLM_ENV_OR_MODEL_LOAD_BLOCKED: transformers import passed, but none of "
        f"{', '.join(candidates)} is available. Upgrade transformers or install the model-specific VLM loader."
    )


def extract_json_payload(raw_output: str) -> Any:
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw_output, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return json.loads(fence_match.group(1))
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw_output[start : end + 1])
    raise ValueError("model output did not contain a JSON object")


def build_output_payload(
    *,
    args: argparse.Namespace,
    prompt: str,
    context_summary: dict[str, Any],
    raw_output: str,
    parsed: Any,
    latency_ms: int,
    memory_mb: float | None,
    raw_output_path: Path,
) -> dict[str, Any]:
    records = parsed.get("subgoals", parsed) if isinstance(parsed, dict) else parsed
    subgoals, errors = validate_risk1b_subgoal_records(records, limit=args.num_subgoals)
    valid = not errors and len(subgoals) >= 2
    provenance = "external_vlm_json" if args.backend == "transformers" else f"{args.backend}_contract"
    return {
        "model": args.model_id,
        "generator_backend": args.backend,
        "provenance": provenance,
        "source_context": {
            "suite": args.suite,
            "task_id": args.task_id,
            "seed": args.seed,
            "task_description": args.task_description,
            "context_image": args.context_image,
            "context_json": args.context_json,
            "context_summary": context_summary,
        },
        "prompt_template": prompt,
        "raw_output_path": str(raw_output_path),
        "raw_output_preview": raw_output[:2000],
        "latency_ms": latency_ms,
        "memory_mb": memory_mb,
        "cost_usd": 0.0,
        "schema_validation": {
            "valid": valid,
            "errors": errors,
            "required_fields": list(RISK1B_REQUIRED_FIELDS),
        },
        "subgoals": subgoals,
        "boundary": (
            "mock/fixture generation is schema plumbing only and cannot count as Risk1-B PASS; "
            "PASS requires external VLM provenance plus actual SmolVLA policy-generated chunks."
        ),
    }


def current_memory_mb() -> float | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            usage = usage / (1024 * 1024)
        else:
            usage = usage / 1024
        return round(float(usage), 3)
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_subgoals < 2:
        print("config_error: --num-subgoals must be at least 2", file=sys.stderr)
        return 2
    context_summary = read_context_summary(args.context_json)
    prompt = build_generation_prompt(args, context_summary)
    if args.dependency_check_only:
        try:
            components = resolve_transformers_components(args.model_id)
        except Exception as exc:  # noqa: BLE001
            print(f"dependency_check_error: {type(exc).__name__}: {str(exc)[:500]}", file=sys.stderr)
            return 2
        result = {"status": "PASS", **components["diagnostics"]}
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else f"status=PASS {result}")
        return 0
    if args.backend == "transformers" and not is_actual_context(context_summary):
        print(
            "context_error: transformers generation requires actual Risk1-B context JSON "
            "with provenance.actual_context=true; do not generate from mock/fabricated context",
            file=sys.stderr,
        )
        return 2
    output_path = Path(args.output_path) if args.output_path else default_output_path(
        args.output_dir, args.model_id, args.suite, args.task_id, args.seed
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_path.with_suffix(".raw.txt")
    start = time.perf_counter()
    try:
        if args.backend == "mock":
            raw_output = generate_mock_output(args)
        elif args.backend == "fixture":
            if not args.fixture_output:
                print("config_error: --fixture-output is required for backend=fixture", file=sys.stderr)
                return 2
            raw_output = Path(args.fixture_output).read_text(encoding="utf-8")
        else:
            raw_output = generate_transformers_output(args, prompt)
        latency_ms = int(round((time.perf_counter() - start) * 1000))
        raw_output_path.write_text(raw_output + "\n", encoding="utf-8")
        parsed = extract_json_payload(raw_output)
        payload = build_output_payload(
            args=args,
            prompt=prompt,
            context_summary=context_summary,
            raw_output=raw_output,
            parsed=parsed,
            latency_ms=latency_ms,
            memory_mb=current_memory_mb(),
            raw_output_path=raw_output_path,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"generation_error: {type(exc).__name__}: {str(exc)[:500]}", file=sys.stderr)
        return 2
    validation = payload["schema_validation"]
    if not validation["valid"]:
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"schema_validation_error: {validation['errors']}", file=sys.stderr)
        return 2
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "status": "PASS",
        "output_path": str(output_path),
        "raw_output_path": str(raw_output_path),
        "schema_validation": validation,
        "provenance": payload["provenance"],
        "model": args.model_id,
        "generator_backend": args.backend,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"status=PASS output_path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
