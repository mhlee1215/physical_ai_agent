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
    RISK1B_CANDIDATE_PROMPT_SEMANTICS,
    RISK1B_CANDIDATE_MODELS,
    RISK1B_LEGACY_SCHEMA_NOTE,
    RISK1B_REQUIRED_FIELDS,
    parse_simple_manipulation_relation,
    parse_risk1b_manipulation_relations,
    validate_risk1b_task_relation,
    validate_risk1b_subgoal_records,
)


PROMPT_TEMPLATE = """You are generating a strategy portfolio for a frozen SmolVLA LIBERO policy.

Return only valid JSON with this schema:
{{
  "candidate_prompt_semantics": "same_immediate_goal_strategy_variants_first_action_chunk",
  "locked_task_fields": {{
    "original_task": "...",
    "manipulated_object": "...",
    "target_region": "...",
    "relation": "..."
  }},
  "subgoals": [
    {{
      "locked_task_fields": {{
        "original_task": "...",
        "manipulated_object": "...",
        "target_region": "...",
        "relation": "..."
      }},
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
- The key "subgoals" is a legacy compatibility key. These records are
  alternative goal-conditioned candidate prompts / strategy variants for the
  same immediate objective, not temporal subgoal-completion plan steps.
- Produce exactly {num_subgoals} alternative prompt candidates.
- All candidates must describe the SAME immediate next subgoal, not sequential
  steps in a plan.
- Do NOT decompose the task over time. Do NOT output "pick up, then move, then
  place" as separate entries.
- The following LOCKED_TASK_FIELDS are immutable. Copy each value exactly into
  the top-level locked_task_fields object and into every candidate's
  locked_task_fields object:
{locked_task_fields_json}
- Candidate 0 must preserve locked_task_fields.original_task exactly as the
  baseline subgoal_text.
- Every candidate must keep locked_task_fields.manipulated_object as the
  manipulated object and locked_task_fields.target_region as the destination or
  relation target. Do not swap, omit, rename, abbreviate, or replace them.
- Never command the robot to move, align, center, pick up, grasp, lift, slide,
  push, pull, reposition, or bring locked_task_fields.target_region. The target
  region is the destination, not the manipulated object.
- If locked_task_fields.manipulated_object contains " and " or
  locked_task_fields.relation contains ";", this is a multi-object task. Every
  candidate must preserve the complete object group and every relation pair in
  the same candidate. Do not split one object per candidate. Do not set
  target_object to only one member of the group.
- This means every candidate keeps the same target object, same target relation, and
  same stop condition as the baseline while varying only the approach strategy.
- Candidates 1..N may vary only strategy_axis, subgoal_text motion cue,
  confidence, and rationale. The object, target region, relation, and stop
  condition must stay task-equivalent to the locked fields.
- Every candidate subgoal_text must be a completed-goal command, not an
  intermediate motion command. Valid form: "place/put [all locked objects] on/in
  [all locked targets] using [strategy cue]". Invalid forms include "align X
  with Y", "approach X toward Y", "center X above Y", "make first contact with
  Y", or "move X close to Y" because those describe only intermediate progress.
- Every candidate stop_condition must describe the final task-complete state
  from locked_task_fields.relation. It must not stop at aligned/approaching/
  centered/close/first-contact states.
- Strategy-axis words such as pre_contact_alignment, object_centric_open_side,
  short_horizon_contact, and collision_avoidant_approach may appear only as a
  motion cue inside a final placement command. Example: "place both moka pots on
  the stove using a pre-contact alignment cue"; not "align both moka pots with
  the stove".
- Each candidate's target_object must be exactly
  locked_task_fields.manipulated_object.
- Each candidate's target_region_or_point must be exactly
  locked_task_fields.target_region.
- Each candidate's subgoal_text and stop_condition must mention all objects and
  all destination relations encoded in locked_task_fields.relation. For example,
  if the locked relation is "alphabet soup -> basket; tomato sauce -> basket",
  every candidate must keep both alphabet soup and tomato sauce going to the
  basket in that same candidate.
- If locked_task_fields.relation includes an action clause such as
  "turn on stove" or "close microwave", every candidate must preserve that
  action clause in the same candidate. Do not drop non-placement success
  conditions from long-horizon tasks.
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
locked_task_fields={locked_task_fields_compact_json}
context_summary={context_summary}

Task-grounding rules:
- Treat task_goal as the authority over object roles.
- Treat locked_task_fields as the authority over output fields. If the image,
  state keys, or your intuition conflicts with locked_task_fields, obey
  locked_task_fields.
- Candidate 0 subgoal_text must preserve task_goal as closely as possible.
- Candidate 0 strategy_axis must be exactly "baseline".
- If task_goal says to put/place/move X in/on/to Y, X is the manipulated
  object and Y is the destination/target region. Do not invert those roles.
- Do not choose a different object just because it appears first in state keys
  or is visually salient in the image.
- For cream-cheese-to-bowl tasks, every candidate must keep cream cheese as
  the manipulated object and bowl as the target, while making the motion cue
  concrete enough to change the next action chunk.
- For multi-object basket tasks, every candidate must keep the full "put both
  X and Y in the basket" objective; vary only the approach/order/alignment cue,
  not which object is included.
- For stove tasks, every candidate must preserve both parts when present:
  turn on the stove and place the object on the stove.
{required_action_prompt_rules}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Risk1-B external VLM alternative goal / strategy-variant JSON. "
            "The legacy filename/key still says subgoals for compatibility; these are "
            "not temporal subgoal-completion plans. Mock/fixture backends are contract-only "
            "and cannot count as Risk1-B PASS evidence."
        )
    )
    parser.add_argument("--backend", choices=("mock", "fixture", "transformers"), default="mock")
    parser.add_argument("--model-id", choices=RISK1B_CANDIDATE_MODELS, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument(
        "--num-subgoals",
        type=int,
        default=5,
        help="Number of strategy variants / candidate prompts to generate. Name is legacy-compatible.",
    )
    parser.add_argument("--task-description", default="Complete the LIBERO task from the current observation.")
    parser.add_argument("--context-image", default=None, help="Optional start observation/contact sheet image for VLM input.")
    parser.add_argument("--context-json", default=None, help="Optional JSON context summary to include in the prompt.")
    parser.add_argument("--fixture-output", default=None, help="Raw model output fixture for dependency-light local tests.")
    parser.add_argument("--output-dir", default="_workspace/runpod_results/ita_risk_probes")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=0,
        help=(
            "Retry invalid VLM outputs with a validation-error repair prompt before failing. "
            "For transformers, the model is loaded once and reused across repair attempts."
        ),
    )
    parser.add_argument(
        "--fallback-on-validation-error",
        choices=("none", "deterministic_locked_fields"),
        default="none",
        help=(
            "If validation still fails after repair attempts, emit deterministic locked-field "
            "strategy variants instead of stopping the row. Provenance records this fallback."
        ),
    )
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
    locked_task_fields = build_locked_task_fields(task_description)
    return PROMPT_TEMPLATE.format(
        num_subgoals=args.num_subgoals,
        suite=args.suite,
        task_id=args.task_id,
        seed=args.seed,
        task_description=task_description,
        task_goal_source=task_goal_source,
        locked_task_fields_json=indent_json_for_prompt(locked_task_fields),
        locked_task_fields_compact_json=json.dumps(locked_task_fields, sort_keys=True),
        required_action_prompt_rules=build_required_action_prompt_rules(locked_task_fields),
        context_summary=compact_context,
    )


def build_locked_task_fields(task_description: str) -> dict[str, str]:
    original_task = str(task_description).strip()
    relations = parse_risk1b_manipulation_relations(original_task.lower())
    if relations:
        manipulated_object = dedupe_relation_parts(object_name for object_name, _target in relations)
        target_region = dedupe_relation_parts(target for _object_name, target in relations)
        relation = "; ".join(f"{object_name} -> {target}" for object_name, target in relations)
    else:
        relation = parse_simple_manipulation_relation(original_task.lower())
        manipulated_object = relation[0] if relation else "copy original_task exactly"
        target_region = relation[1] if relation else "copy original_task exactly"
        relation = f"{manipulated_object} -> {target_region}" if relation else "unparsed: preserve original_task exactly"
    if re.search(r"\band\s+close\s+it\b|\bclose\s+(?:the\s+)?microwave\b", original_task.lower()):
        relation = f"{relation}; close microwave"
    if re.search(r"\bturn\s+on\s+(?:the\s+)?stove\b|\bturn\s+(?:the\s+)?stove\s+on\b", original_task.lower()):
        relation = f"{relation}; turn on stove"
    return {
        "original_task": original_task,
        "manipulated_object": manipulated_object,
        "target_region": target_region,
        "relation": relation,
    }


def dedupe_relation_parts(parts: Any) -> str:
    unique: list[str] = []
    for part in parts:
        cleaned = str(part).strip()
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return " and ".join(unique)


def indent_json_for_prompt(payload: dict[str, str]) -> str:
    return "\n".join(f"  {line}" for line in json.dumps(payload, indent=2, sort_keys=True).splitlines())


def build_required_action_prompt_rules(locked_fields: dict[str, str]) -> str:
    relation = str(locked_fields.get("relation", "")).lower()
    manipulated_object = str(locked_fields.get("manipulated_object", "")).strip()
    target_region = str(locked_fields.get("target_region", "")).strip()
    rules: list[str] = []
    relation_pairs = [
        tuple(part.strip() for part in relation_part.split("->", 1))
        for relation_part in str(locked_fields.get("relation", "")).split(";")
        if "->" in relation_part
    ]
    if len(relation_pairs) > 1:
        pair_text = "; ".join(f"{obj} -> {target}" for obj, target in relation_pairs)
        natural_pairs = " AND ".join(f"{obj} to {target}" for obj, target in relation_pairs)
        rules.extend(
            [
                f"- MULTI-RELATION LOCK: every candidate must preserve all relation pairs in the same candidate: {pair_text}.",
                f"- Every candidate subgoal_text must include the complete combined objective: {natural_pairs}.",
                "- Do not split relation pairs across candidates. A candidate that handles only one object-target pair is invalid.",
                "- Every candidate target_object must be the full locked_task_fields.manipulated_object string, not a single object from the relation.",
                "- Every candidate target_region_or_point must be the full locked_task_fields.target_region string, not a single destination from the relation.",
            ]
        )
    if "turn on stove" in relation:
        rules.extend(
            [
                "- STOVE ACTION LOCK: every candidate subgoal_text must include both turning on the stove and placing the "
                f"{manipulated_object} on the {target_region}.",
                "- Do not make candidate 0 only 'turn on the stove' and later candidates only 'place the object'. That is temporal decomposition and invalid.",
                "- A valid candidate form is: 'Turn on the stove and [strategy cue] place the "
                f"{manipulated_object} on the {target_region}.'",
                "- Every stop_condition must include both 'stove is turned on' and '"
                f"{manipulated_object} is on the {target_region}'.",
            ]
        )
    if "close microwave" in relation:
        rules.extend(
            [
                "- MICROWAVE ACTION LOCK: every candidate subgoal_text must include both placing the object in the microwave and closing the microwave.",
                "- Do not make separate candidates for placing versus closing. That is temporal decomposition and invalid.",
                "- Every stop_condition must include both the object in the microwave and the microwave closed.",
            ]
        )
    return "\n".join(rules)


def build_repair_prompt(
    *,
    base_prompt: str,
    task_description: str,
    validation_errors: list[str],
    previous_output: str,
) -> str:
    locked_fields = build_locked_task_fields(task_description)
    previous_excerpt = previous_output[:1200].strip()
    return (
        base_prompt
        + "\n\nVALIDATION FAILED. Repair the JSON now.\n"
        + "Return ONLY corrected JSON. Do not explain.\n"
        + "You must preserve these LOCKED TASK FIELDS exactly:\n"
        + json.dumps(locked_fields, indent=2, sort_keys=True)
        + "\nValidation errors to fix:\n"
        + json.dumps(validation_errors, indent=2, sort_keys=True)
        + "\nPrevious output was invalid. Do not copy it if it split, omitted, "
        + "renamed, or swapped any locked object/relation.\n"
        + ("Previous invalid excerpt for reference only:\n" + previous_excerpt + "\n" if previous_excerpt else "")
        + "\nRepair requirements:\n"
        + "- Keep candidate 0 as locked_task_fields.original_task.\n"
        + "- Every candidate must copy locked_task_fields exactly.\n"
        + "- Every candidate target_object must exactly equal locked_task_fields.manipulated_object.\n"
        + "- Every candidate target_region_or_point must exactly equal locked_task_fields.target_region.\n"
        + "- If locked_task_fields.manipulated_object contains ' and ', every candidate must keep the entire object group; do not make one-object candidates.\n"
        + "- If locked_task_fields.relation contains ';', every candidate subgoal_text and stop_condition must preserve every relation pair in that same candidate.\n"
        + "- Do not tell the robot to move, align, center, pick up, grasp, lift, slide, push, pull, reposition, or bring locked_task_fields.target_region.\n"
        + "- Repair intermediate-only candidates by rewriting them as final placement commands with the same strategy cue. For example, replace 'align X with Y' with 'place X on/in Y using pre-contact alignment'.\n"
        + "- Every candidate subgoal_text must start from the completed locked task relation, such as 'put/place [all locked objects] on/in [all locked targets] ...'.\n"
        + "- Every stop_condition must describe the final relation complete, not aligned, approaching, centered, close, or first-contact.\n"
        + (build_required_action_prompt_rules(locked_fields) + "\n" if build_required_action_prompt_rules(locked_fields) else "")
        + "- Vary only strategy_axis and motion cue. Do not introduce new objects. Do not swap object roles.\n"
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
    processor, model, image_module, torch = load_transformers_generator(args, components)
    return generate_transformers_text(args, prompt, processor=processor, model=model, image_module=image_module, torch=torch)


def load_transformers_generator(args: argparse.Namespace, components: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
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
    return processor, model, image_module, torch


def generate_transformers_text(
    args: argparse.Namespace,
    prompt: str,
    *,
    processor: Any,
    model: Any,
    image_module: Any,
    torch: Any,
) -> str:
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
    input_token_count = getattr(getattr(inputs, "input_ids", None), "shape", [0, 0])[-1]
    try:
        generated_outputs = outputs[:, input_token_count:]
    except Exception:  # noqa: BLE001
        generated_outputs = outputs
    decoded = processor.batch_decode(generated_outputs, skip_special_tokens=True)[0]
    return decoded


def deterministic_locked_field_output(args: argparse.Namespace, context_summary: dict[str, Any]) -> str:
    task_description, _source = effective_task_description(args, context_summary)
    locked_fields = build_locked_task_fields(task_description)
    manipulated_object = locked_fields["manipulated_object"]
    target_region = locked_fields["target_region"]
    relation = locked_fields["relation"]
    final_state = f"{manipulated_object} satisfies {relation}"
    base_command = task_description.rstrip(".")
    axes = [
        ("baseline", base_command),
        (
            "pre_contact_alignment",
            f"{base_command} using a pre-contact alignment cue before final placement at {target_region}",
        ),
        (
            "object_centric_open_side",
            f"{base_command} from the clearest visible side while preserving the final relation with {target_region}",
        ),
        (
            "gripper_pose_precision",
            f"{base_command} using a precise gripper pose on {manipulated_object} before release at {target_region}",
        ),
        (
            "short_horizon_contact",
            f"{base_command} using the shortest contact path that still completes the relation with {target_region}",
        ),
        (
            "collision_avoidant_approach",
            f"{base_command} along a collision-avoidant path that finishes with {manipulated_object} at {target_region}",
        ),
    ]
    records = []
    for axis, text in axes[: args.num_subgoals]:
        records.append(
            {
                "subgoal_text": text,
                "strategy_axis": axis,
                "locked_task_fields": locked_fields,
                "target_object": manipulated_object,
                "target_region_or_point": target_region,
                "stop_condition": final_state,
                "confidence": 1.0 if axis == "baseline" else 0.55,
                "rationale": "deterministic locked-field fallback; not external VLM generation",
            }
        )
    return json.dumps({"locked_task_fields": locked_fields, "subgoals": records}, indent=2)


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
    generation_attempts: list[dict[str, Any]] | None = None,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = (
        parsed.get("candidate_prompts")
        or parsed.get("strategy_variants")
        or parsed.get("subgoals")
        or parsed
        if isinstance(parsed, dict)
        else parsed
    )
    subgoals, errors = validate_risk1b_subgoal_records(records, limit=args.num_subgoals)
    task_description, task_goal_source = effective_task_description(args, context_summary)
    locked_task_fields = build_locked_task_fields(task_description)
    errors = [*errors, *validate_risk1b_task_relation(subgoals, task_description)]
    valid = not errors and len(subgoals) >= 2
    if fallback:
        provenance = "deterministic_locked_fields_fallback"
    else:
        provenance = "external_vlm_json" if args.backend == "transformers" else f"{args.backend}_contract"
    return {
        "model": args.model_id,
        "generator_backend": args.backend,
        "provenance": provenance,
        "candidate_prompt_semantics": RISK1B_CANDIDATE_PROMPT_SEMANTICS,
        "legacy_schema_note": RISK1B_LEGACY_SCHEMA_NOTE,
        "source_context": {
            "suite": args.suite,
            "task_id": args.task_id,
            "seed": args.seed,
            "task_description": task_description,
            "task_goal_source": task_goal_source,
            "locked_task_fields": locked_task_fields,
            "context_image": args.context_image,
            "context_json": args.context_json,
            "context_summary": context_summary,
        },
        "locked_task_fields": locked_task_fields,
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
        "generation_attempts": generation_attempts or [],
        "fallback": fallback,
        "subgoals": subgoals,
        "strategy_variants": subgoals,
        "candidate_prompts": subgoals,
        "boundary": (
            "mock/fixture generation is schema plumbing only and cannot count as Risk1-B PASS; "
            "PASS requires external VLM provenance plus actual SmolVLA policy-generated first action chunks. "
            "This is alternative-goal/strategy-variant candidate generation, not temporal subgoal completion "
            "or full-task benchmark success."
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
    if args.repair_attempts < 0:
        print("config_error: --repair-attempts must be >= 0", file=sys.stderr)
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
    generation_attempts: list[dict[str, Any]] = []
    fallback: dict[str, Any] | None = None
    start = time.perf_counter()
    try:
        if args.backend in {"mock", "fixture"}:
            if args.backend == "mock":
                raw_output = generate_mock_output(args)
            else:
                if not args.fixture_output:
                    print("config_error: --fixture-output is required for backend=fixture", file=sys.stderr)
                    return 2
                raw_output = Path(args.fixture_output).read_text(encoding="utf-8")
            raw_output_path.write_text(raw_output + "\n", encoding="utf-8")
            parsed = extract_json_payload(raw_output)
            payload = build_output_payload(
                args=args,
                prompt=prompt,
                context_summary=context_summary,
                raw_output=raw_output,
                parsed=parsed,
                latency_ms=int(round((time.perf_counter() - start) * 1000)),
                memory_mb=current_memory_mb(),
                raw_output_path=raw_output_path,
                generation_attempts=generation_attempts,
            )
        else:
            components = resolve_transformers_components(args.model_id)
            processor, model, image_module, torch = load_transformers_generator(args, components)
            task_description, _task_goal_source = effective_task_description(args, context_summary)
            current_prompt = prompt
            payload = None
            raw_output = ""
            parsed: Any = None
            for attempt_index in range(args.repair_attempts + 1):
                attempt_started = time.perf_counter()
                raw_output = generate_transformers_text(
                    args,
                    current_prompt,
                    processor=processor,
                    model=model,
                    image_module=image_module,
                    torch=torch,
                )
                attempt_raw_path = raw_output_path if attempt_index == 0 else output_path.with_suffix(f".repair{attempt_index}.raw.txt")
                attempt_raw_path.write_text(raw_output + "\n", encoding="utf-8")
                try:
                    parsed = extract_json_payload(raw_output)
                    payload = build_output_payload(
                        args=args,
                        prompt=current_prompt,
                        context_summary=context_summary,
                        raw_output=raw_output,
                        parsed=parsed,
                        latency_ms=int(round((time.perf_counter() - start) * 1000)),
                        memory_mb=current_memory_mb(),
                        raw_output_path=attempt_raw_path,
                        generation_attempts=generation_attempts,
                    )
                    validation = payload["schema_validation"]
                    generation_attempts.append(
                        {
                            "attempt": attempt_index,
                            "kind": "initial" if attempt_index == 0 else "repair",
                            "valid": bool(validation["valid"]),
                            "errors": validation["errors"],
                            "raw_output_path": str(attempt_raw_path),
                            "elapsed_ms": int(round((time.perf_counter() - attempt_started) * 1000)),
                        }
                    )
                    payload["generation_attempts"] = generation_attempts
                    if validation["valid"]:
                        break
                    current_prompt = build_repair_prompt(
                        base_prompt=prompt,
                        task_description=task_description,
                        validation_errors=validation["errors"],
                        previous_output=raw_output,
                    )
                except Exception as attempt_exc:  # noqa: BLE001
                    generation_attempts.append(
                        {
                            "attempt": attempt_index,
                            "kind": "initial" if attempt_index == 0 else "repair",
                            "valid": False,
                            "errors": [f"{type(attempt_exc).__name__}: {str(attempt_exc)[:500]}"],
                            "raw_output_path": str(attempt_raw_path),
                            "elapsed_ms": int(round((time.perf_counter() - attempt_started) * 1000)),
                        }
                    )
                    current_prompt = build_repair_prompt(
                        base_prompt=prompt,
                        task_description=task_description,
                        validation_errors=generation_attempts[-1]["errors"],
                        previous_output=raw_output,
                    )
            if payload is None or not payload["schema_validation"]["valid"]:
                if args.fallback_on_validation_error == "deterministic_locked_fields":
                    fallback = {
                        "strategy": "deterministic_locked_fields",
                        "reason": "validation_failed_after_repair_attempts",
                        "attempt_count": len(generation_attempts),
                        "last_errors": generation_attempts[-1]["errors"] if generation_attempts else [],
                    }
                    raw_output = deterministic_locked_field_output(args, context_summary)
                    fallback_raw_path = output_path.with_suffix(".fallback.raw.txt")
                    fallback_raw_path.write_text(raw_output + "\n", encoding="utf-8")
                    parsed = extract_json_payload(raw_output)
                    payload = build_output_payload(
                        args=args,
                        prompt=prompt,
                        context_summary=context_summary,
                        raw_output=raw_output,
                        parsed=parsed,
                        latency_ms=int(round((time.perf_counter() - start) * 1000)),
                        memory_mb=current_memory_mb(),
                        raw_output_path=fallback_raw_path,
                        generation_attempts=generation_attempts,
                        fallback=fallback,
                    )
                elif payload is None:
                    raise ValueError("model output did not produce a valid payload")
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
