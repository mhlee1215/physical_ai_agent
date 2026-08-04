"""Typed configuration and reporting helpers for SO101 grasp workspace probes."""

from __future__ import annotations

from collections import Counter
from math import atan2, cos, degrees, hypot, radians, sin
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AxisGrid(StrictModel):
    min_m: float
    max_m: float
    step_m: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_range(self) -> AxisGrid:
        if self.max_m <= self.min_m:
            raise ValueError("max_m must be greater than min_m")
        steps = (self.max_m - self.min_m) / self.step_m
        if abs(steps - round(steps)) > 1e-7:
            raise ValueError("(max_m - min_m) must be divisible by step_m")
        return self

    def values(self) -> list[float]:
        count = int(round((self.max_m - self.min_m) / self.step_m))
        return [float(self.min_m + index * self.step_m) for index in range(count + 1)]


class WorkspaceGrid(StrictModel):
    x: AxisGrid
    y: AxisGrid


class AngleGrid(StrictModel):
    min_deg: float
    max_deg: float
    step_deg: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_range(self) -> AngleGrid:
        if self.max_deg <= self.min_deg:
            raise ValueError("max_deg must be greater than min_deg")
        steps = (self.max_deg - self.min_deg) / self.step_deg
        if abs(steps - round(steps)) > 1e-7:
            raise ValueError("(max_deg - min_deg) must be divisible by step_deg")
        return self

    def values(self) -> list[float]:
        count = int(round((self.max_deg - self.min_deg) / self.step_deg))
        return [
            float(self.min_deg + index * self.step_deg)
            for index in range(count + 1)
        ]


class PolarWorkspaceGrid(StrictModel):
    radius: AxisGrid
    angle: AngleGrid

    @model_validator(mode="after")
    def validate_radius(self) -> PolarWorkspaceGrid:
        if self.radius.min_m < 0.0:
            raise ValueError("polar radius cannot be negative")
        return self


class TeacherContract(StrictModel):
    skill_mode: Literal[
        "grip_the_cube_v1",
        "grip_the_cube_near_v1",
        "grip_the_cube_continuous_v1",
    ] = "grip_the_cube_v1"
    trajectory_variant: Literal[
        "standard", "roll_first", "direct_align", "auto"
    ] = "standard"
    task_prompt: str = "grip the green cube and lift"
    approach_steps: int = Field(default=34, gt=0)
    settle_steps: int = Field(default=10, ge=0)
    close_steps: int = Field(default=42, gt=0)
    lift_steps: int = Field(default=70, gt=0)
    terminal_hold_steps: int = Field(default=12, ge=0)
    lift_target_height_m: float = Field(default=0.065, gt=0.0)
    operational_lift_height_m: float = Field(default=0.06, gt=0.0)
    lift_controller_z_error_m: float = Field(default=0.015, gt=0.0)
    move_target_z_offset_m: float = Field(default=0.075, gt=0.0)
    edge_contact_xy_threshold_m: float = Field(default=0.012, gt=0.0)
    edge_parallel_threshold_deg: float = Field(default=3.0, gt=0.0, le=90.0)
    min_gripper_floor_clearance_m: float = Field(default=0.01, ge=0.0)
    camera2_gate_mode: Literal[
        "geometry_only", "preclose_and_early_trace", "strict_image_trace"
    ] = "preclose_and_early_trace"
    camera2_pre_close_max_deg: float = Field(default=25.0, gt=0.0, le=90.0)
    camera2_close_25_max_deg: float = Field(default=25.0, gt=0.0, le=90.0)
    camera2_close_50_max_deg: float = Field(default=25.0, gt=0.0, le=90.0)

    @model_validator(mode="after")
    def validate_lift_thresholds(self) -> TeacherContract:
        if self.operational_lift_height_m > self.lift_target_height_m:
            raise ValueError(
                "operational_lift_height_m cannot exceed lift_target_height_m"
            )
        return self

    def camera2_limits(self) -> dict[str, float] | None:
        if self.camera2_gate_mode == "geometry_only":
            return None
        return {
            "pre_close_image_alignment_error_deg": self.camera2_pre_close_max_deg,
            "close_25_image_alignment_error_deg": self.camera2_close_25_max_deg,
            "close_50_image_alignment_error_deg": self.camera2_close_50_max_deg,
        }


class WorkspaceProbeConfig(StrictModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    camera_rig_config: str
    base_reference: Literal["shoulder_pan_axis_projection"] = (
        "shoulder_pan_axis_projection"
    )
    home_qpos: tuple[float, float, float, float, float, float]
    object_half_size_m: float = Field(default=0.015, gt=0.0)
    object_color: Literal["green", "red", "blue"] = "green"
    object_yaw_mode: Literal["fixed", "radial_face_normal"] = "fixed"
    object_yaw_degrees: list[float] = Field(default_factory=list)
    radial_yaw_offsets_degrees: list[float] = Field(default_factory=list)
    seed_base: int = Field(ge=0)
    grid: WorkspaceGrid | None = None
    polar_grid: PolarWorkspaceGrid | None = None
    evaluate_dataset_contract: bool = True
    teacher: TeacherContract = Field(default_factory=TeacherContract)

    @model_validator(mode="after")
    def validate_sampling_grid(self) -> WorkspaceProbeConfig:
        if (self.grid is None) == (self.polar_grid is None):
            raise ValueError("exactly one of grid or polar_grid must be configured")
        if self.object_yaw_mode == "fixed" and not self.object_yaw_degrees:
            raise ValueError("fixed object_yaw_mode requires object_yaw_degrees")
        if self.object_yaw_mode == "radial_face_normal":
            if self.polar_grid is None:
                raise ValueError(
                    "radial_face_normal object yaw requires polar_grid"
                )
            if not self.radial_yaw_offsets_degrees:
                raise ValueError(
                    "radial_face_normal object yaw requires "
                    "radial_yaw_offsets_degrees"
                )
        return self


def grid_points(config: WorkspaceProbeConfig) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    if config.grid is not None:
        for y_index, y_m in enumerate(config.grid.y.values()):
            for x_index, x_m in enumerate(config.grid.x.values()):
                for yaw_index, yaw_deg in enumerate(config.object_yaw_degrees):
                    points.append(
                        {
                            "point_id": (
                                f"x{x_index:03d}_y{y_index:03d}"
                                f"_yaw{yaw_index:02d}"
                            ),
                            "x_index": x_index,
                            "y_index": y_index,
                            "yaw_index": yaw_index,
                            "base_x_m": x_m,
                            "base_y_m": y_m,
                            "yaw_deg": float(yaw_deg),
                            "point_cell_area_m2": (
                                config.grid.x.step_m * config.grid.y.step_m
                            ),
                        }
                    )
        return points

    assert config.polar_grid is not None
    angle_step_rad = radians(config.polar_grid.angle.step_deg)
    for angle_index, angle_deg in enumerate(config.polar_grid.angle.values()):
        angle_rad = radians(angle_deg)
        if config.object_yaw_mode == "fixed":
            yaw_values = config.object_yaw_degrees
        else:
            yaw_values = [
                angle_deg + offset
                for offset in config.radial_yaw_offsets_degrees
            ]
        for radius_index, radius_m in enumerate(
            config.polar_grid.radius.values()
        ):
            for yaw_index, yaw_deg in enumerate(yaw_values):
                points.append(
                    {
                        "point_id": (
                            f"r{radius_index:03d}_a{angle_index:03d}"
                            f"_yaw{yaw_index:02d}"
                        ),
                        "radius_index": radius_index,
                        "angle_index": angle_index,
                        "yaw_index": yaw_index,
                        "base_x_m": float(radius_m * cos(angle_rad)),
                        "base_y_m": float(radius_m * sin(angle_rad)),
                        "yaw_deg": float(yaw_deg),
                        "point_cell_area_m2": float(
                            radius_m
                            * config.polar_grid.radius.step_m
                            * angle_step_rad
                        ),
                    }
                )
    return points


def base_relative_to_world_xy(
    base_xy: tuple[float, float],
    base_rotation_world_xy: np.ndarray,
    relative_xy: tuple[float, float],
) -> tuple[float, float]:
    rotation = np.asarray(base_rotation_world_xy, dtype=float).reshape(2, 2)
    relative = np.asarray(relative_xy, dtype=float).reshape(2)
    world = np.asarray(base_xy, dtype=float).reshape(2) + rotation @ relative
    return float(world[0]), float(world[1])


def physical_outcome_metrics(
    result: dict[str, Any],
    *,
    target_lift_height_m: float,
    operational_lift_height_m: float,
) -> dict[str, Any]:
    final_infos: list[dict[str, Any]] = []
    final_info = result.get("final_info")
    if isinstance(final_info, dict):
        final_infos.append(final_info)
    for failure in result.get("candidate_failures", []):
        candidate_final = failure.get("final_info")
        if isinstance(candidate_final, dict):
            final_infos.append(candidate_final)

    grasped_lifts = [
        float(info.get("lift_height", float("-inf")))
        for info in final_infos
        if bool(info.get("is_grasped"))
    ]
    max_grasped_lift = max(grasped_lifts, default=None)
    writer_success = bool(result.get("success"))
    grasped = bool(grasped_lifts) or writer_success
    return {
        "teacher_geometry_contract_success": writer_success,
        "grasp_success": grasped,
        "max_grasped_lift_height_m": max_grasped_lift,
        "operational_lift_success": (
            writer_success
            or (
                max_grasped_lift is not None
                and max_grasped_lift >= operational_lift_height_m
            )
        ),
        "target_lift_success": (
            writer_success
            or (
                max_grasped_lift is not None
                and max_grasped_lift >= target_lift_height_m
            )
        ),
    }


def annotate_physical_outcomes(
    record: dict[str, Any],
    *,
    target_lift_height_m: float,
    operational_lift_height_m: float,
) -> dict[str, Any]:
    annotated = dict(record)
    result = annotated.get("physical_result")
    if not isinstance(result, dict):
        return annotated
    metrics = physical_outcome_metrics(
        result,
        target_lift_height_m=target_lift_height_m,
        operational_lift_height_m=operational_lift_height_m,
    )
    annotated["physical_teacher_contract_success"] = metrics[
        "teacher_geometry_contract_success"
    ]
    annotated["physical_grasp_success"] = metrics["grasp_success"]
    annotated["physical_operational_lift_success"] = metrics[
        "operational_lift_success"
    ]
    annotated["physical_target_lift_success"] = metrics["target_lift_success"]
    annotated["physical_max_grasped_lift_height_m"] = metrics[
        "max_grasped_lift_height_m"
    ]
    annotated["physical_success"] = metrics["target_lift_success"]
    if annotated["physical_success"]:
        annotated["physical_failure_reason"] = None
    return annotated


def summarize_workspace_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    physical = [row for row in records if bool(row.get("physical_success"))]
    grasped = [row for row in records if bool(row.get("physical_grasp_success"))]
    operational = [
        row
        for row in records
        if bool(row.get("physical_operational_lift_success"))
    ]
    teacher_geometry = [
        row
        for row in records
        if bool(row.get("physical_teacher_contract_success"))
    ]
    dataset_ready = [row for row in records if bool(row.get("dataset_contract_success"))]
    preflight = [row for row in records if bool(row.get("preflight_passed"))]

    def bounds(rows: list[dict[str, Any]]) -> dict[str, float] | None:
        if not rows:
            return None
        xs = [float(row["base_x_m"]) for row in rows]
        ys = [float(row["base_y_m"]) for row in rows]
        radii = [hypot(x, y) for x, y in zip(xs, ys, strict=True)]
        return {
            "x_min_m": min(xs),
            "x_max_m": max(xs),
            "x_span_m": max(xs) - min(xs),
            "y_min_m": min(ys),
            "y_max_m": max(ys),
            "y_span_m": max(ys) - min(ys),
            "radial_min_m": min(radii),
            "radial_max_m": max(radii),
        }

    failure_counts = Counter(
        str(row.get("physical_failure_reason") or "unknown")
        for row in records
        if not bool(row.get("physical_success"))
    )
    dataset_failure_counts = Counter(
        str(row.get("dataset_contract_failure_reason") or "unknown")
        for row in records
        if bool(row.get("physical_success"))
        and not bool(row.get("dataset_contract_success"))
    )
    return {
        "points": len(records),
        "preflight_passed": len(preflight),
        "physical_grasp_successes": len(grasped),
        "physical_operational_lift_successes": len(operational),
        "physical_teacher_geometry_contract_successes": len(teacher_geometry),
        "physical_successes": len(physical),
        "physical_success_rate": len(physical) / len(records) if records else 0.0,
        "dataset_contract_successes": len(dataset_ready),
        "dataset_contract_success_rate": (
            len(dataset_ready) / len(records) if records else 0.0
        ),
        "physical_success_bounds": bounds(physical),
        "dataset_contract_success_bounds": bounds(dataset_ready),
        "physical_failure_reasons": dict(sorted(failure_counts.items())),
        "dataset_contract_failure_reasons": dict(
            sorted(dataset_failure_counts.items())
        ),
        "radial_envelope": radial_envelope(physical),
    }


def radial_envelope(
    records: list[dict[str, Any]], *, sector_width_deg: float = 15.0
) -> list[dict[str, float | int]]:
    sectors: dict[int, list[float]] = {}
    for row in records:
        x = float(row["base_x_m"])
        y = float(row["base_y_m"])
        angle = degrees(atan2(y, x))
        sector = int(np.floor((angle + 180.0) / sector_width_deg))
        sectors.setdefault(sector, []).append(hypot(x, y))
    return [
        {
            "angle_min_deg": -180.0 + sector * sector_width_deg,
            "angle_max_deg": -180.0 + (sector + 1) * sector_width_deg,
            "samples": len(radii),
            "radius_min_m": min(radii),
            "radius_max_m": max(radii),
        }
        for sector, radii in sorted(sectors.items())
    ]


def camera_grid_bin(
    normalized_centroid: list[float] | tuple[float, float],
    *,
    grid_size: int = 4,
) -> int:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if len(normalized_centroid) != 2:
        raise ValueError("normalized centroid must contain x and y")
    bx = min(
        grid_size - 1,
        max(0, int(float(normalized_centroid[0]) * grid_size)),
    )
    by = min(
        grid_size - 1,
        max(0, int(float(normalized_centroid[1]) * grid_size)),
    )
    return int(by * grid_size + bx)


def successful_workspace_cells(
    records: list[dict[str, Any]],
    *,
    success_key: str = "dataset_contract_success",
    camera_grid_size: int = 4,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for row in records:
        if not bool(row.get(success_key)):
            continue
        centroid = row.get("initial_camera1_centroid")
        camera_bin = (
            None
            if not isinstance(centroid, (list, tuple)) or len(centroid) != 2
            else camera_grid_bin(centroid, grid_size=camera_grid_size)
        )
        cells.append(
            {
                "point_id": str(row["point_id"]),
                "base_xy_m": [
                    float(row["base_x_m"]),
                    float(row["base_y_m"]),
                ],
                "world_xy_m": [
                    float(row["world_x_m"]),
                    float(row["world_y_m"]),
                ],
                "radius_from_base_m": float(row["radius_from_base_m"]),
                "angle_from_base_deg": float(row["angle_from_base_deg"]),
                "object_yaw_deg": float(row["yaw_deg"]),
                "camera1_normalized_centroid": (
                    None
                    if camera_bin is None
                    else [float(centroid[0]), float(centroid[1])]
                ),
                "camera1_grid_bin": camera_bin,
                "cell_area_m2": float(row.get("point_cell_area_m2", 0.0)),
            }
        )
    cells.sort(
        key=lambda cell: (
            round(float(cell["angle_from_base_deg"]), 9),
            round(float(cell["radius_from_base_m"]), 9),
            str(cell["point_id"]),
        )
    )
    total_area = sum(float(cell["cell_area_m2"]) for cell in cells)
    for cell in cells:
        cell["uniform_area_weight"] = (
            float(cell["cell_area_m2"]) / total_area
            if total_area > 0.0
            else 0.0
        )
    return cells
