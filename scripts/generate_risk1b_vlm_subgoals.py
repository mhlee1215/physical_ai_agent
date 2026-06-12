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
    validate_risk1b_task_relation,
    validate_risk1b_subgoal_records,
)


PROMPT_TEMPLATE = """You are generating a strategy portfolio for a frozen SmolVLA LIBERO policy.

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
- Produce exactly {num_subgoals} alternative prompt candidates.
- All candidates must describe the SAME immediate next subgoal, not sequential
  steps in a plan.
- Do NOT decompose the task over time. Do NOT output "pick up, then move, then
  place" as separate entries.
- Candidate 0 must preserve the original task wording as the baseline.
- Candidates 1..N must keep the same target object, same target relation, and
  same stop condition as the baseline, while varying only the approach strategy.
- Each non-baseline candidate must use a distinct strategy axis from this set
  when possible: object_centric_open_side, pre_contact_alignment,
  gripper_pose_precision, short_horizon_contact, collision_avoidant_approach.
- For bowl/container placement tasks, prefer concrete mode-shift axes when
  visible and task-preserving: high_clearance_over_rim, vertical_drop_centering,
  near_side_entry, far_side_entry, rim_avoidance_arc, gentle_release_inside_bowl.
- Do not vary only verbs such as put/place/move/transfer. Each non-baseline
  candidate must include a behaviorally distinct motion cue that a frozen
  SmolVLA executor can condition on, such as approach side, centering before
  contact, wrist/gripper orientation, short first contact, or clearance around
  the bowl rim.
- The strategy cue must still preserve X-in-Y semantics. For a task like
  "put cream cheese in bowl", every candidate must move the cream cheese
  toward/into the bowl; none may become a bowl pickup task.
- Make each candidate directly usable as one prompt to the frozen SmolVLA
  executor for the next action chunk.
- Do not introduce cleanup, repetition, extra objects, or any task not visible
  in the context.

Context:
suite={suite}
task_id={task_id}
seed={seed}
task_goal={task_description}
task_goal_source={task_goal_source}
context_summary={context_summary}

Task-grounding rules:
- Treat task_goal as the authority over object roles.
- Candidate 0 subgoal_text must preserve task_goal as closely as possible.
- Candidate 0 strategy_axis must be exactly "baseline".
- If task_goal says to put/place/move X in/on/to Y, X is the manipulated
  object and Y is the destination/target region. Do not invert those roles.
- Do not choose a different object just because it appears first in state keys
  or is visually salient in the image.
- For cream-cheese-to-bowl tasks, every candidate must keep cream cheese as
  the manipulated object and bowl as the target, while making the motion cue
  concrete enough to change the next action chunk.
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
    task_description, task_goal_source = effective_task_description(args, context_summary)
    compact_context = json.dumps(compact_context_for_prompt(context_summary), sort_keys=True) if context_summary else "none"
    return PROMPT_TEMPLATE.format(
        num_subgoals=args.num_subgoals,
        suite=args.suite,
        task_id=args.task_id,
        seed=args.seed,
        task_description=task_description,
        task_goal_source=task_goal_source,
        context_summary=compact_context,
    )


def effective_task_description(args: argparse.Namespace, context_summary: dict[str, Any]) -> tuple[str, str]:
    context_task = context_summary.get("task_description")
    if isinstance(context_task, str) and context_task.strip():
        return context_task.strip(), "context_json.task_description"
    task_descriptions = context_summary.get("task_descriptions")
    if isinstance(task_descriptions, list):
        for candidate in task_descriptions:
            if isinstance(candidate, str) and candidate.strip() and candidate.strip() != args.suite:
                return candidate.strip(), "context_json.task_descriptions"
    return str(args.task_description).strip(), "cli.task_description"


def compact_context_for_prompt(context_summary: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "task_description",
        "task_descriptions",
        "suite",
        "task_id",
        "seed",
        "camera_count",
        "camera_mapping",
        "observation_source",
        "info_summary",
        "provenance",
    ):
        if key in context_summary:
            compact[key] = context_summary[key]
    observation_summary = context_summary.get("observation_summary")
    if isinstance(observation_summary, dict):
        image_keys: list[str] = []
        object_state_keys: list[str] = []
        for key, value in sorted(observation_summary.items()):
            if "image" in key:
                image_keys.append(key)
            elif not key.startswith("robot0_"):
                object_state_keys.append(key)
        compact["observation_summary_compact"] = {
            "image_keys": image_keys,
            "object_state_keys": object_state_keys[:30],
            "object_state_key_count": len(object_state_keys),
        }
    camera_images = context_summary.get("camera_images")
    if isinstance(camera_images, list):
        compact["camera_images"] = [
            {
                "camera_key": item.get("camera_key"),
                "source": item.get("source"),
            }
            for item in camera_images
            if isinstance(item, dict)
        ]
    return compact


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
            "subgoal_text": "Approach the same task object from the most open visible side while preserving the original goal.",
            "strategy_axis": "object_centric_open_side",
            "target_object": "task object",
            "target_region_or_point": "task target region",
            "stop_condition": "original task success condition is reached",
            "confidence": 0.75,
            "rationale": "same-subgoal strategy branch for schema plumbing only",
        },
        {
            "subgoal_text": "Pause at a pre-contact alignment pose before executing the same original task.",
            "strategy_axis": "pre_contact_alignment",
            "target_object": "task object",
            "target_region_or_point": "task target region",
            "stop_condition": "original task success condition is reached",
            "confidence": 0.72,
            "rationale": "same-subgoal strategy branch for schema plumbing only",
        },
        {
            "subgoal_text": "Use a precise gripper pose for the same task object and same final placement relation.",
            "strategy_axis": "gripper_pose_precision",
            "target_object": "task object",
            "target_region_or_point": "task target region",
            "stop_condition": "original task success condition is reached",
            "confidence": 0.7,
            "rationale": "same-subgoal strategy branch for schema plumbing only",
        },
        {
            "subgoal_text": "Make the smallest useful contact that advances the same original goal without changing the target.",
            "strategy_axis": "short_horizon_contact",
            "target_object": "task object",
            "target_region_or_point": "task target region",
            "stop_condition": "original task success condition is reached",
            "confidence": 0.65,
            "rationale": "same-subgoal strategy branch for schema plumbing only",
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
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw_output[start : end + 1])
        except json.JSONDecodeError:
            pass
    repaired = extract_wrapped_subgoal_records(raw_output)
    if repaired:
        return {
            "subgoals": repaired,
            "_schema_repair": {
                "strategy": "extract_wrapped_subgoal_objects",
                "record_count": len(repaired),
                "reason": "model output contained malformed JSON but valid nested subgoal objects",
            },
        }
    raise ValueError("model output did not contain a JSON object")


def balanced_json_object_at(text: str, start: int) -> str | None:
    if start < 0 or start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_wrapped_subgoal_records(raw_output: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for match in re.finditer(r'"subgoal"\s*:', raw_output):
        object_start = raw_output.find("{", match.end())
        object_text = balanced_json_object_at(raw_output, object_start)
        if not object_text:
            continue
        try:
            record = json.loads(object_text)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and any(field in record for field in RISK1B_REQUIRED_FIELDS):
            records.append({"subgoal": record})
    return records


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
    task_description, task_goal_source = effective_task_description(args, context_summary)
    errors = [*errors, *validate_risk1b_task_relation(subgoals, task_description)]
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
            "task_description": task_description,
            "task_goal_source": task_goal_source,
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
            "repair": parsed.get("_schema_repair") if isinstance(parsed, dict) else None,
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
