#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import physical_ai_agent.sim.mycobot_nexus_env as nexus  # noqa: E402


@dataclass(frozen=True)
class RawLiftCandidate:
    index: int
    pad_size: tuple[float, float, float]
    pad_offset_y: float
    pad_offset_z: float
    pad_euler: tuple[float, float, float]
    cube_offset_x: float
    cube_offset_y: float
    cube_offset_z: float
    close_gripper_command: float
    close_steps: int
    lift_steps: int
    cube_mass: float
    lift_delta: tuple[float, float, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep focused 280 Pi adaptive raw-contact Gate 8 candidates and "
            "log contact-normal metrics for lift-retention tuning."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/raw280_tuning/side_only_pad_gate8_sweep_002"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--max-candidates", type=int, default=0, help="Stop after N candidates; 0 means run all.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Skip candidates already present in results.jsonl.")
    parser.add_argument("--pad-size-x", type=float, nargs="+", default=[0.014, 0.020, 0.028, 0.036])
    parser.add_argument("--pad-size-y", type=float, nargs="+", default=[0.006, 0.009, 0.012, 0.016, 0.020])
    parser.add_argument("--pad-size-z", type=float, nargs="+", default=[0.002, 0.003, 0.004, 0.006])
    parser.add_argument("--pad-offset-y", type=float, nargs="+", default=[0.0, 0.002])
    parser.add_argument("--pad-offset-z", type=float, nargs="+", default=[-0.012, -0.008, -0.004])
    parser.add_argument("--pad-euler-x", type=float, nargs="+", default=[0.0])
    parser.add_argument("--pad-euler-y", type=float, nargs="+", default=[0.0])
    parser.add_argument("--pad-euler-z", type=float, nargs="+", default=[0.0])
    parser.add_argument("--cube-offset-x", type=float, nargs="+", default=[nexus.ADAPTIVE_280_PI_GATE7_CUBE_OFFSET[0]])
    parser.add_argument("--cube-offset-y", type=float, nargs="+", default=[nexus.ADAPTIVE_280_PI_GATE7_CUBE_OFFSET[1]])
    parser.add_argument("--cube-offset-z", type=float, nargs="+", default=[0.006, 0.012, 0.018])
    parser.add_argument("--close-gripper-command", type=float, nargs="+", default=[-0.085, -0.12, -0.18])
    parser.add_argument("--close-steps", type=int, nargs="+", default=[180, 240])
    parser.add_argument("--lift-steps", type=int, default=120)
    parser.add_argument("--cube-mass", type=float, nargs="+", default=[0.005, 0.02, 0.05])
    parser.add_argument("--lift-delta-j2", type=float, nargs="+", default=[-0.08])
    parser.add_argument("--lift-delta-j3", type=float, nargs="+", default=[0.15])
    parser.add_argument("--lift-delta-j4", type=float, nargs="+", default=[0.25])
    return parser


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    completed = _completed_indices(results_path) if args.resume else set()
    candidates = list(_candidate_grid(args))
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    best: dict[str, Any] | None = None
    for ordinal, candidate in enumerate(candidates, start=1):
        if candidate.index in completed:
            continue
        row = run_candidate(args, candidate)
        with results_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")
        best = _better(row, best)
        should_print = (
            row["status"] == "passed"
            or row["final_gripper_cube_contact_pads"] >= 2
            or row["lift_best_sustained_contact_steps"] >= 10
            or ordinal % max(args.print_every, 1) == 0
        )
        if should_print:
            print(json.dumps({"candidate": ordinal, "row": row, "best": best}, sort_keys=True), flush=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"candidate_count": len(candidates), "best": best}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if best is not None:
        print(json.dumps({"best": best, "summary_path": str(summary_path)}, indent=2, sort_keys=True))


def _candidate_grid(args: argparse.Namespace) -> Iterable[RawLiftCandidate]:
    index = 0
    for sx, sy, sz, dy, dz, ex, ey, ez, cx, cy, cz, cmd, close_steps, mass, dj2, dj3, dj4 in itertools.product(
        args.pad_size_x,
        args.pad_size_y,
        args.pad_size_z,
        args.pad_offset_y,
        args.pad_offset_z,
        args.pad_euler_x,
        args.pad_euler_y,
        args.pad_euler_z,
        args.cube_offset_x,
        args.cube_offset_y,
        args.cube_offset_z,
        args.close_gripper_command,
        args.close_steps,
        args.cube_mass,
        args.lift_delta_j2,
        args.lift_delta_j3,
        args.lift_delta_j4,
    ):
        yield RawLiftCandidate(
            index=index,
            pad_size=(sx, sy, sz),
            pad_offset_y=dy,
            pad_offset_z=dz,
            pad_euler=(ex, ey, ez),
            cube_offset_x=cx,
            cube_offset_y=cy,
            cube_offset_z=cz,
            close_gripper_command=cmd,
            close_steps=close_steps,
            lift_steps=args.lift_steps,
            cube_mass=mass,
            lift_delta=(dj2, dj3, dj4),
        )
        index += 1


def run_candidate(args: argparse.Namespace, candidate: RawLiftCandidate) -> dict[str, Any]:
    _install_overrides(candidate)
    _install_lift_override(candidate)
    output_dir = args.output_dir / f"candidate_{candidate.index:05d}"
    result = nexus.run_mycobot_adaptive_grasp_lift_smoke(
        output_dir=output_dir,
        asset_root=args.asset_root,
        official_gripper_root=args.official_gripper_root,
        model_profile=nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
        seed=1,
        width=64,
        height=48,
        pregrasp_steps=20,
        close_steps=candidate.close_steps,
        lift_steps=candidate.lift_steps,
        placement_gripper_command=1.0,
        close_gripper_command=candidate.close_gripper_command,
        required_close_sustained_steps=15,
        required_lift_sustained_steps=30,
        required_final_lift=0.025,
        teacher_attachment_enabled=False,
    )
    metrics = _trace_metrics(Path(result.trace_path))
    result_dict = asdict(result)
    row = {
        **asdict(candidate),
        **result_dict,
        **metrics,
        **_result_stability_metrics(result_dict),
        "output_dir": str(output_dir),
    }
    return row


def _install_overrides(candidate: RawLiftCandidate) -> None:
    left = nexus.ADAPTIVE_LEFT_FINGER_PAD_POS
    right = nexus.ADAPTIVE_RIGHT_FINGER_PAD_POS
    nexus.ADAPTIVE_280_LEFT_FINGER_PAD_POS = (left[0], left[1] + candidate.pad_offset_y, left[2] + candidate.pad_offset_z)
    nexus.ADAPTIVE_280_RIGHT_FINGER_PAD_POS = (right[0], right[1] + candidate.pad_offset_y, right[2] + candidate.pad_offset_z)
    nexus.ADAPTIVE_280_FINGER_PAD_SIZE = candidate.pad_size
    nexus.ADAPTIVE_FINGER_PAD_EULER = candidate.pad_euler
    cube_x, cube_y, _cube_z = nexus.ADAPTIVE_280_PI_GATE7_CUBE_OFFSET
    nexus.ADAPTIVE_280_PI_GATE7_CUBE_OFFSET = (cube_x, cube_y, candidate.cube_offset_z)
    _install_cube_mass_override(candidate.cube_mass)



def _install_lift_override(candidate: RawLiftCandidate) -> None:
    gate7 = nexus.ADAPTIVE_280_PI_GATE7_TABLE_ARM_QPOS
    nexus.ADAPTIVE_280_PI_GATE8_LIFT_ARM_QPOS = (
        gate7[0],
        gate7[1] + candidate.lift_delta[0],
        gate7[2] + candidate.lift_delta[1],
        gate7[3] + candidate.lift_delta[2],
        gate7[4],
        gate7[5],
    )

def _install_cube_mass_override(cube_mass: float) -> None:
    original_build = nexus.build_mycobot_nexus_scene_model
    if getattr(original_build, "_raw280_mass_wrapper", False):
        original_build = original_build._raw280_original_build  # type: ignore[attr-defined]

    def wrapper(*wrapper_args: Any, **wrapper_kwargs: Any) -> None:
        original_build(*wrapper_args, **wrapper_kwargs)
        scene_path = wrapper_kwargs.get("scene_path") if "scene_path" in wrapper_kwargs else wrapper_args[1]
        tree = ET.parse(scene_path)
        root = tree.getroot()
        for geom in root.findall(".//geom"):
            if geom.attrib.get("name") == nexus.TASK_CUBE_GEOM:
                geom.set("mass", str(cube_mass))
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)

    wrapper._raw280_mass_wrapper = True  # type: ignore[attr-defined]
    wrapper._raw280_original_build = original_build  # type: ignore[attr-defined]
    nexus.build_mycobot_nexus_scene_model = wrapper


def _trace_metrics(trace_path: Path) -> dict[str, Any]:
    lift_records: list[dict[str, Any]] = []
    close_records: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record["phase"] == "close":
                close_records.append(record)
            elif record["phase"] == "lift":
                lift_records.append(record)
    close_tail = [record["info"] for record in close_records[-24:]]
    lift_head = [record["info"] for record in lift_records[:24]]
    return {
        "close_tail_two_pad_steps": sum(1 for info in close_tail if int(info.get("gripper_cube_contact_pads", 0)) >= 2),
        "lift_head_two_pad_steps": sum(1 for info in lift_head if int(info.get("gripper_cube_contact_pads", 0)) >= 2),
        "lift_head_avg_cube_lift": _avg(info.get("cube_lift", 0.0) for info in lift_head),
    }


def _completed_indices(results_path: Path) -> set[int]:
    if not results_path.exists():
        return set()
    completed: set[int] = set()
    with results_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                completed.add(int(json.loads(line)["index"]))
    return completed


def _better(row: dict[str, Any], best: dict[str, Any] | None) -> dict[str, Any]:
    if best is None:
        return row
    row_score = _score(row)
    best_score = _score(best)
    return row if row_score > best_score else best


def _result_stability_metrics(result: dict[str, Any]) -> dict[str, Any]:
    initial = np.asarray(result.get("initial_cube_position", [0.0, 0.0, 0.0])[:2], dtype=float)
    final = np.asarray(result.get("final_cube_position", [0.0, 0.0, 0.0])[:2], dtype=float)
    planar_drift = float(np.linalg.norm(final - initial))
    final_lift = float(result.get("final_cube_lift", 0.0))
    return {
        "final_planar_drift": planar_drift,
        "stable_final_cube": planar_drift <= 0.25 and -0.05 <= final_lift <= 0.15,
    }


def _score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    if not row.get("stable_final_cube", False):
        return (
            1000.0 if row["status"] == "passed" else 0.0,
            float(row["lift_best_sustained_contact_steps"]),
            -float(row.get("final_planar_drift", 0.0)),
            -abs(float(row["final_cube_lift"])),
        )
    return (
        1000.0 if row["status"] == "passed" else 0.0,
        float(row["final_gripper_cube_contact_pads"]) * 100.0 + float(row["lift_best_sustained_contact_steps"]),
        float(row["final_cube_lift"]),
        float(row["close_best_sustained_contact_steps"]),
    )


def _avg(values: Iterable[Any]) -> float:
    numbers = [float(value) for value in values]
    return float(np.mean(numbers)) if numbers else 0.0


if __name__ == "__main__":
    main()
