"""Typed workspace spawn catalogs for deterministic SO101 dataset generation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceSpawnCandidate(_StrictModel):
    candidate_id: str = Field(min_length=1)
    source_cell_id: str = Field(min_length=1)
    stage: Literal["primary", "backup"]
    world_xy_m: list[float] = Field(min_length=2, max_length=2)
    base_xy_m: list[float] = Field(min_length=2, max_length=2)
    radius_from_base_m: float = Field(gt=0.0)
    angle_from_base_deg: float
    object_yaw_deg: float = Field(ge=-180.0, le=180.0)
    sampling_weight: float = Field(gt=0.0)
    camera1_grid_bin: int | None = Field(default=None, ge=0)
    radial_offset_m: float = 0.0
    angular_offset_deg: float = 0.0


class IndependentObjectYawDistribution(_StrictModel):
    mode: Literal["independent_stratified"] = "independent_stratified"
    reference_frame: Literal["robot_relative", "world_absolute"] = "robot_relative"
    min_deg: float = Field(ge=-180.0, le=180.0)
    max_deg: float = Field(ge=-180.0, le=180.0)
    strata: int = Field(gt=0)
    periodicity_deg: float = Field(default=90.0, gt=0.0, le=360.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> IndependentObjectYawDistribution:
        if self.max_deg <= self.min_deg:
            raise ValueError("object yaw max_deg must exceed min_deg")
        if self.max_deg - self.min_deg > self.periodicity_deg + 1e-9:
            raise ValueError(
                "object yaw range cannot exceed its declared symmetry periodicity"
            )
        return self


class ContinuousAreaDistribution(_StrictModel):
    coordinate_system: Literal["base_xy"] = "base_xy"
    radius_min_m: float = Field(gt=0.0)
    radius_max_m: float = Field(gt=0.0)
    angle_min_deg: float = Field(ge=-180.0, le=180.0)
    angle_max_deg: float = Field(ge=-180.0, le=180.0)
    radial_strata: int = Field(gt=0)
    angular_strata: int = Field(gt=0)
    far_to_near_area_density_ratio: float = Field(gt=0.0, le=1.0)
    minimum_spacing_m: float = Field(gt=0.0)
    candidate_pool_minimum_spacing_m: float | None = Field(default=None, gt=0.0)
    evidence_radius_half_range_m: float | None = Field(default=None, gt=0.0)
    evidence_angle_half_range_deg: float | None = Field(default=None, gt=0.0)
    excluded_radius_relative_yaw_pairs: list[list[float]] = Field(default_factory=list)
    object_yaw: IndependentObjectYawDistribution | None = None
    max_robot_relative_yaw_count_cv: float = Field(default=0.10, ge=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> ContinuousAreaDistribution:
        if self.radius_max_m <= self.radius_min_m:
            raise ValueError("continuous radius_max_m must exceed radius_min_m")
        if self.angle_max_deg <= self.angle_min_deg:
            raise ValueError("continuous angle_max_deg must exceed angle_min_deg")
        if (
            self.candidate_pool_minimum_spacing_m is not None
            and self.candidate_pool_minimum_spacing_m > self.minimum_spacing_m
        ):
            raise ValueError(
                "candidate pool spacing cannot exceed accepted dataset spacing"
            )
        return self


class WorkspaceSpawnCellQuota(_StrictModel):
    cell_id: str = Field(min_length=1)
    radial_index: int = Field(ge=0)
    angular_index: int = Field(ge=0)
    radius_bounds_m: list[float] = Field(min_length=2, max_length=2)
    angle_bounds_deg: list[float] = Field(min_length=2, max_length=2)
    yaw_index: int | None = Field(default=None, ge=0)
    yaw_bounds_deg: list[float] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
    )
    area_m2: float = Field(gt=0.0)
    target_probability: float = Field(gt=0.0, le=1.0)
    primary_target_count: int = Field(gt=0)
    backup_count: int = Field(gt=0)


class WorkspaceSpawnShard(_StrictModel):
    shard: str = Field(min_length=1)
    start_index: int = Field(ge=0)
    candidate_count: int = Field(gt=0)
    primary_target_count: int = Field(gt=0)


class WorkspaceSpawnCatalog(_StrictModel):
    format: Literal[
        "so101_workspace_spawn_catalog_v1",
        "so101_workspace_spawn_catalog_v2",
    ]
    catalog_id: str = Field(min_length=1)
    source_workspace_catalog: str
    source_workspace_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    camera_rig_config: str
    home_qpos: list[float] = Field(min_length=6, max_length=6)
    object_color: str
    object_half_size_m: float = Field(gt=0.0)
    base_origin_world_xy_m: list[float] = Field(min_length=2, max_length=2)
    candidate_count: int = Field(gt=0)
    primary_target_count: int = Field(gt=0)
    backup_count: int = Field(ge=0)
    source_cell_count: int = Field(gt=0)
    distance_decay_rate_per_m: float = Field(gt=0.0)
    angular_jitter_max_deg: float = Field(ge=0.0)
    radial_jitter_max_m: float = Field(default=0.0, ge=0.0)
    candidate_sequence_offset: int = Field(default=0, ge=0)
    preserve_evidence_object_yaw: bool = False
    object_yaw_jitter_half_range_deg: float = Field(default=0.0, ge=0.0)
    sampling_strategy: Literal[
        "angular_golden_v1",
        "polar_stratified_v2",
        "evidence_local_pose_jitter_v1",
        "continuous_area_stratified_v3",
        "continuous_area_yaw_stratified_v4",
        "continuous_joint_feasible_yaw_stratified_v5",
    ] = "angular_golden_v1"
    enforce_cell_local_quota: bool = False
    continuous_distribution: ContinuousAreaDistribution | None = None
    cell_quotas: list[WorkspaceSpawnCellQuota] = Field(default_factory=list)
    candidates: list[WorkspaceSpawnCandidate] = Field(min_length=1)
    shards: list[WorkspaceSpawnShard] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog(self) -> WorkspaceSpawnCatalog:
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must match candidates")
        if self.primary_target_count + self.backup_count != self.candidate_count:
            raise ValueError("primary_target_count + backup_count must match candidate_count")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("workspace candidate_id values must be unique")
        positions = [
            (round(candidate.world_xy_m[0], 12), round(candidate.world_xy_m[1], 12))
            for candidate in self.candidates
        ]
        if len(positions) != len(set(positions)):
            raise ValueError("workspace candidate world positions must be unique")
        covered: set[int] = set()
        primary_total = 0
        for shard in self.shards:
            indices = set(range(shard.start_index, shard.start_index + shard.candidate_count))
            if covered & indices:
                raise ValueError("workspace shard candidate ranges must not overlap")
            covered.update(indices)
            primary_total += shard.primary_target_count
            rows = self.candidates[
                shard.start_index : shard.start_index + shard.candidate_count
            ]
            if len(rows) != shard.candidate_count:
                raise ValueError(f"workspace shard range is out of bounds: {shard.shard}")
            if sum(row.stage == "primary" for row in rows) < shard.primary_target_count:
                raise ValueError(
                    f"workspace shard has fewer primary candidates than target: {shard.shard}"
                )
        if covered != set(range(self.candidate_count)):
            raise ValueError("workspace shard ranges must cover every candidate exactly once")
        if primary_total != self.primary_target_count:
            raise ValueError("workspace shard primary targets must match primary_target_count")
        if self.format == "so101_workspace_spawn_catalog_v2":
            if self.sampling_strategy not in {
                "continuous_area_stratified_v3",
                "continuous_area_yaw_stratified_v4",
                "continuous_joint_feasible_yaw_stratified_v5",
            }:
                raise ValueError(
                    "workspace catalog v2 requires a continuous-area strategy"
                )
            if not self.enforce_cell_local_quota:
                raise ValueError("workspace catalog v2 requires cell-local quota enforcement")
            if self.continuous_distribution is None or not self.cell_quotas:
                raise ValueError("workspace catalog v2 requires continuous distribution and quotas")
            if len(self.cell_quotas) != self.source_cell_count:
                raise ValueError("source_cell_count must match cell_quotas")
            quota_ids = [row.cell_id for row in self.cell_quotas]
            if len(quota_ids) != len(set(quota_ids)):
                raise ValueError("workspace cell quota ids must be unique")
            quota_primary = {row.cell_id: row.primary_target_count for row in self.cell_quotas}
            quota_backup = {row.cell_id: row.backup_count for row in self.cell_quotas}
            actual_primary = Counter(
                row.source_cell_id for row in self.candidates if row.stage == "primary"
            )
            actual_backup = Counter(
                row.source_cell_id for row in self.candidates if row.stage == "backup"
            )
            if dict(actual_primary) != quota_primary:
                raise ValueError("primary candidates must match cell quotas exactly")
            if dict(actual_backup) != quota_backup:
                raise ValueError("backup candidates must match cell quotas exactly")
            radius_min = self.continuous_distribution.radius_min_m
            radius_max = self.continuous_distribution.radius_max_m
            angle_min = self.continuous_distribution.angle_min_deg
            angle_max = self.continuous_distribution.angle_max_deg
            yaw_distribution = self.continuous_distribution.object_yaw
            if self.sampling_strategy in {
                "continuous_area_yaw_stratified_v4",
                "continuous_joint_feasible_yaw_stratified_v5",
            }:
                if yaw_distribution is None:
                    raise ValueError(
                        "yaw-stratified catalog requires an object-yaw distribution"
                    )
                if any(
                    row.yaw_index is None or row.yaw_bounds_deg is None
                    for row in self.cell_quotas
                ):
                    raise ValueError(
                        "yaw-stratified catalog requires yaw bounds on every quota"
                    )
            elif yaw_distribution is not None:
                raise ValueError(
                    "continuous_area_stratified_v3 cannot declare object yaw strata"
                )
            for row in self.candidates:
                if not radius_min < row.radius_from_base_m < radius_max:
                    raise ValueError("continuous candidate lies on or outside radius boundary")
                if not angle_min < row.angle_from_base_deg < angle_max:
                    raise ValueError("continuous candidate lies on or outside angle boundary")
                yaw_coordinate = (
                    _object_yaw_distribution_coordinate(row, yaw_distribution)
                    if yaw_distribution is not None
                    else None
                )
                if yaw_coordinate is not None and not (
                    yaw_distribution.min_deg
                    < yaw_coordinate
                    < yaw_distribution.max_deg
                ):
                    raise ValueError(
                        "continuous candidate lies on or outside object-yaw boundary"
                    )
        return self


def load_workspace_spawn_catalog(path: Path) -> WorkspaceSpawnCatalog:
    return WorkspaceSpawnCatalog.model_validate_json(path.read_text(encoding="utf-8"))


def build_workspace_spawn_catalog(
    *,
    source_path: Path,
    catalog_id: str,
    primary_target_count: int,
    backup_count: int,
    shard_count: int,
    distance_decay_rate_per_m: float,
    angular_jitter_max_deg: float,
    radial_jitter_max_m: float = 0.0,
    candidate_sequence_offset: int = 0,
    preserve_evidence_object_yaw: bool = False,
    object_yaw_jitter_half_range_deg: float = 0.0,
) -> WorkspaceSpawnCatalog:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("format") != "so101_grasp_workspace_catalog_v1":
        raise ValueError(f"unsupported workspace source format: {source_path}")
    cells = list(source.get("cells") or [])
    if not cells:
        raise ValueError("workspace source catalog contains no cells")
    if primary_target_count < len(cells):
        raise ValueError("primary target count must cover every workspace cell")
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if candidate_sequence_offset < 0:
        raise ValueError("candidate_sequence_offset must be nonnegative")
    if object_yaw_jitter_half_range_deg < 0.0:
        raise ValueError("object yaw jitter half range must be nonnegative")
    if preserve_evidence_object_yaw and any(
        "object_yaw_deg" not in cell for cell in cells
    ):
        raise ValueError(
            "preserve_evidence_object_yaw requires object_yaw_deg on every source cell"
        )
    if primary_target_count % shard_count or backup_count % shard_count:
        raise ValueError("primary and backup counts must divide evenly across shards")

    origin = [float(value) for value in source["base_contract"]["world_xyz_m"][:2]]
    weights = _normalized_cell_weights(
        cells,
        distance_decay_rate_per_m=distance_decay_rate_per_m,
    )
    primary_counts = _allocate_counts(
        primary_target_count,
        weights,
        minimum_per_cell=1,
    )
    backup_counts = _allocate_counts(
        backup_count,
        weights,
        minimum_per_cell=0,
    )
    primary = _make_candidates(
        cells,
        counts=primary_counts,
        stage="primary",
        origin=origin,
        weights=weights,
        angular_jitter_max_deg=angular_jitter_max_deg,
        radial_jitter_max_m=radial_jitter_max_m,
        sequence_offset=candidate_sequence_offset,
        preserve_evidence_object_yaw=preserve_evidence_object_yaw,
        object_yaw_jitter_half_range_deg=object_yaw_jitter_half_range_deg,
    )
    backup = _make_candidates(
        cells,
        counts=backup_counts,
        stage="backup",
        origin=origin,
        weights=weights,
        angular_jitter_max_deg=angular_jitter_max_deg,
        radial_jitter_max_m=radial_jitter_max_m,
        sequence_offset=(candidate_sequence_offset + max(primary_counts) + 1),
        preserve_evidence_object_yaw=preserve_evidence_object_yaw,
        object_yaw_jitter_half_range_deg=object_yaw_jitter_half_range_deg,
    )
    primary_shards = _stratified_partition(primary, shard_count)
    backup_shards = _stratified_partition(backup, shard_count)

    candidates: list[WorkspaceSpawnCandidate] = []
    shards: list[WorkspaceSpawnShard] = []
    primary_per_shard = primary_target_count // shard_count
    for shard_index in range(shard_count):
        start_index = len(candidates)
        rows = [*primary_shards[shard_index], *backup_shards[shard_index]]
        candidates.extend(rows)
        shards.append(
            WorkspaceSpawnShard(
                shard=f"workspace_{shard_index:02d}",
                start_index=start_index,
                candidate_count=len(rows),
                primary_target_count=primary_per_shard,
            )
        )

    return WorkspaceSpawnCatalog(
        format="so101_workspace_spawn_catalog_v1",
        catalog_id=catalog_id,
        source_workspace_catalog=str(source_path),
        source_workspace_catalog_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        camera_rig_config=str(source["camera_rig_config"]),
        home_qpos=[float(value) for value in source["home_qpos"]],
        object_color=str(source["object_color"]),
        object_half_size_m=float(source["object_half_size_m"]),
        base_origin_world_xy_m=origin,
        candidate_count=len(candidates),
        primary_target_count=primary_target_count,
        backup_count=backup_count,
        source_cell_count=len(cells),
        distance_decay_rate_per_m=distance_decay_rate_per_m,
        angular_jitter_max_deg=angular_jitter_max_deg,
        radial_jitter_max_m=radial_jitter_max_m,
        candidate_sequence_offset=candidate_sequence_offset,
        preserve_evidence_object_yaw=preserve_evidence_object_yaw,
        object_yaw_jitter_half_range_deg=object_yaw_jitter_half_range_deg,
        sampling_strategy=(
            "evidence_local_pose_jitter_v1"
            if preserve_evidence_object_yaw
            else (
                "polar_stratified_v2"
                if radial_jitter_max_m > 0.0
                else "angular_golden_v1"
            )
        ),
        candidates=candidates,
        shards=shards,
    )


def build_joint_feasible_workspace_spawn_catalog(
    *,
    source_path: Path,
    catalog_id: str,
    radial_primary_counts: list[int],
    yaw_primary_counts: list[int],
    radial_backup_counts: list[int],
    yaw_backup_counts: list[int],
    shard_count: int,
    radius_min_m: float,
    radius_max_m: float,
    angle_min_deg: float,
    angle_max_deg: float,
    minimum_spacing_m: float,
    candidate_pool_minimum_spacing_m: float | None = None,
    evidence_radius_half_range_m: float = 0.004,
    evidence_angle_half_range_deg: float = 2.0,
    object_yaw_center_offset_deg: float = 2.5,
    object_yaw_jitter_half_range_deg: float = 2.0,
    object_yaw_periodicity_deg: float = 90.0,
    max_robot_relative_yaw_count_cv: float = 0.10,
    excluded_radius_relative_yaw_pairs: list[tuple[float, float]] | None = None,
    candidate_sequence_offset: int = 0,
) -> WorkspaceSpawnCatalog:
    """Build a continuous catalog only over jointly teacher-feasible pose cells.

    The source probe defines feasible radius/angle/robot-relative-yaw tuples.
    Marginal quotas are solved before candidate generation, so a common easy
    orientation cannot consume quota intended for another feasible orientation.
    """

    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("format") != "so101_grasp_workspace_catalog_v1":
        raise ValueError(f"unsupported workspace source format: {source_path}")
    evidence_cells = list(source.get("cells") or [])
    if not evidence_cells:
        raise ValueError("workspace source catalog contains no cells")
    excluded_pairs = list(excluded_radius_relative_yaw_pairs or [])
    evidence_cells = [
        row
        for row in evidence_cells
        if not any(
            math.isclose(
                float(row["radius_from_base_m"]),
                float(radius_m),
                abs_tol=1e-8,
            )
            and math.isclose(
                _normalize_periodic_coordinate(
                    float(row["object_yaw_deg"])
                    - float(row["angle_from_base_deg"]),
                    object_yaw_periodicity_deg,
                ),
                _normalize_periodic_coordinate(
                    float(relative_yaw_deg),
                    object_yaw_periodicity_deg,
                ),
                abs_tol=1e-8,
            )
            for radius_m, relative_yaw_deg in excluded_pairs
        )
    ]
    if not evidence_cells:
        raise ValueError("joint-feasible exclusions removed every evidence cell")
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if candidate_sequence_offset < 0:
        raise ValueError("candidate_sequence_offset must be nonnegative")
    if object_yaw_jitter_half_range_deg <= 0.0:
        raise ValueError("object yaw jitter half range must be positive")
    if evidence_radius_half_range_m <= 0.0:
        raise ValueError("evidence radius half range must be positive")
    if evidence_angle_half_range_deg <= 0.0:
        raise ValueError("evidence angle half range must be positive")
    pool_spacing = (
        minimum_spacing_m
        if candidate_pool_minimum_spacing_m is None
        else candidate_pool_minimum_spacing_m
    )
    if pool_spacing <= 0.0 or pool_spacing > minimum_spacing_m:
        raise ValueError(
            "candidate pool spacing must be positive and no greater than accepted spacing"
        )

    radii = _unique_sorted_coordinates(
        float(row["radius_from_base_m"]) for row in evidence_cells
    )
    angles = _unique_sorted_coordinates(
        float(row["angle_from_base_deg"]) for row in evidence_cells
    )
    relative_yaws = _unique_periodic_coordinates(
        (
            (
                float(row["object_yaw_deg"])
                - float(row["angle_from_base_deg"])
            )
            % object_yaw_periodicity_deg
            for row in evidence_cells
        ),
        period=object_yaw_periodicity_deg,
    )
    if len(radial_primary_counts) != len(radii):
        raise ValueError("radial primary counts must match probe radii")
    if len(radial_backup_counts) != len(radii):
        raise ValueError("radial backup counts must match probe radii")
    if len(yaw_primary_counts) != len(relative_yaws):
        raise ValueError("yaw primary counts must match probe yaw strata")
    if len(yaw_backup_counts) != len(relative_yaws):
        raise ValueError("yaw backup counts must match probe yaw strata")
    if any(value <= 0 for value in [*radial_primary_counts, *radial_backup_counts]):
        raise ValueError("every radial quota must be positive")
    if any(value <= 0 for value in [*yaw_primary_counts, *yaw_backup_counts]):
        raise ValueError("every yaw quota must be positive")
    primary_target_count = sum(radial_primary_counts)
    backup_count = sum(radial_backup_counts)
    if primary_target_count != sum(yaw_primary_counts):
        raise ValueError("primary radial and yaw margins must have equal totals")
    if backup_count != sum(yaw_backup_counts):
        raise ValueError("backup radial and yaw margins must have equal totals")
    if primary_target_count % len(angles) or backup_count % len(angles):
        raise ValueError("primary and backup totals must divide across probe angles")

    radius_index = {round(value, 9): index for index, value in enumerate(radii)}
    angle_index = {round(value, 9): index for index, value in enumerate(angles)}
    yaw_index = {round(value, 9): index for index, value in enumerate(relative_yaws)}
    feasible_by_angle: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for row in evidence_cells:
        radial = radius_index[round(float(row["radius_from_base_m"]), 9)]
        angular = angle_index[round(float(row["angle_from_base_deg"]), 9)]
        relative_yaw = _normalize_periodic_coordinate((
            float(row["object_yaw_deg"])
            - float(row["angle_from_base_deg"])
        ), object_yaw_periodicity_deg)
        yaw = yaw_index[round(relative_yaw, 9)]
        feasible_by_angle[angular].add((radial, yaw))
    expected_pattern = feasible_by_angle[0]
    if not expected_pattern or any(
        feasible_by_angle[index] != expected_pattern
        for index in range(len(angles))
    ):
        raise ValueError(
            "joint-feasible builder requires the same radius/yaw mask at every angle"
        )

    primary_pair_counts = _allocate_sparse_matrix(
        radial_primary_counts,
        yaw_primary_counts,
        allowed=expected_pattern,
    )
    backup_pair_counts = _allocate_sparse_matrix(
        radial_backup_counts,
        yaw_backup_counts,
        allowed=expected_pattern,
    )
    primary_counts_by_cell = _distribute_pairs_across_angles(
        primary_pair_counts,
        angle_target=primary_target_count // len(angles),
        angle_count=len(angles),
    )
    backup_counts_by_cell = _distribute_pairs_across_angles(
        backup_pair_counts,
        angle_target=backup_count // len(angles),
        angle_count=len(angles),
    )
    active_keys = sorted(
        set(primary_counts_by_cell) | set(backup_counts_by_cell)
    )
    if any(
        primary_counts_by_cell.get(key, 0) <= 0
        or backup_counts_by_cell.get(key, 0) <= 0
        for key in active_keys
    ):
        raise ValueError("every active joint-feasible cell needs primary and backup quota")

    radius_bounds = [
        (
            max(radius_min_m, center - evidence_radius_half_range_m),
            min(radius_max_m, center + evidence_radius_half_range_m),
        )
        for center in radii
    ]
    angle_bounds = [
        (
            max(angle_min_deg, center - evidence_angle_half_range_deg),
            min(angle_max_deg, center + evidence_angle_half_range_deg),
        )
        for center in angles
    ]
    active_yaws_by_radius: dict[int, set[int]] = defaultdict(set)
    for radial, _, yaw in active_keys:
        active_yaws_by_radius[radial].add(yaw)
    cells: list[dict[str, Any]] = []
    for radial, angular, yaw in active_keys:
        radius_lo, radius_hi = radius_bounds[radial]
        angle_lo, angle_hi = angle_bounds[angular]
        yaw_center = (
            relative_yaws[yaw] + object_yaw_center_offset_deg
        ) % object_yaw_periodicity_deg
        yaw_lo = yaw_center - object_yaw_jitter_half_range_deg
        yaw_hi = yaw_center + object_yaw_jitter_half_range_deg
        if not 0.0 < yaw_lo < yaw_hi < object_yaw_periodicity_deg:
            raise ValueError("jittered object-yaw bounds must remain in the open period")
        spatial_area = (
            0.5
            * (radius_hi**2 - radius_lo**2)
            * math.radians(angle_hi - angle_lo)
        )
        cells.append(
            {
                "cell_id": f"r{radial:02d}_a{angular:02d}_y{yaw:02d}",
                "radial_index": radial,
                "angular_index": angular,
                "evidence_radius_center_m": radii[radial],
                "evidence_angle_center_deg": angles[angular],
                "radius_bounds_m": [radius_lo, radius_hi],
                "angle_bounds_deg": [angle_lo, angle_hi],
                "yaw_index": yaw,
                "yaw_bounds_deg": [yaw_lo, yaw_hi],
                # Yaw is an attribute of the same XY support, not extra floor area.
                "area_m2": spatial_area / len(active_yaws_by_radius[radial]),
                "target_probability": (
                    primary_counts_by_cell[(radial, angular, yaw)]
                    / primary_target_count
                ),
            }
        )

    radial_areas: dict[int, float] = defaultdict(float)
    for cell in cells:
        radial_areas[int(cell["radial_index"])] += float(cell["area_m2"])
    radial_densities = [
        radial_primary_counts[index] / radial_areas[index]
        for index in range(len(radii))
    ]
    if any(
        left + 1e-9 < right
        for left, right in zip(radial_densities, radial_densities[1:])
    ):
        raise ValueError("radial quotas do not define nonincreasing area density")
    far_to_near_ratio = radial_densities[-1] / radial_densities[0]
    decay_rate = max(
        1e-9,
        -math.log(far_to_near_ratio) / (radius_max_m - radius_min_m),
    )
    object_yaw = IndependentObjectYawDistribution(
        reference_frame="robot_relative",
        min_deg=0.0,
        max_deg=object_yaw_periodicity_deg,
        strata=len(relative_yaws),
        periodicity_deg=object_yaw_periodicity_deg,
    )
    distribution = ContinuousAreaDistribution(
        radius_min_m=radius_min_m,
        radius_max_m=radius_max_m,
        angle_min_deg=angle_min_deg,
        angle_max_deg=angle_max_deg,
        radial_strata=len(radii),
        angular_strata=len(angles),
        far_to_near_area_density_ratio=far_to_near_ratio,
        minimum_spacing_m=minimum_spacing_m,
        candidate_pool_minimum_spacing_m=pool_spacing,
        evidence_radius_half_range_m=evidence_radius_half_range_m,
        evidence_angle_half_range_deg=evidence_angle_half_range_deg,
        excluded_radius_relative_yaw_pairs=[
            [float(radius_m), float(relative_yaw_deg)]
            for radius_m, relative_yaw_deg in excluded_pairs
        ],
        object_yaw=object_yaw,
        max_robot_relative_yaw_count_cv=max_robot_relative_yaw_count_cv,
    )
    _validate_continuous_domain_evidence(distribution, evidence_cells)
    origin = [float(value) for value in source["base_contract"]["world_xyz_m"][:2]]
    primary_counts = [
        primary_counts_by_cell[
            (int(cell["radial_index"]), int(cell["angular_index"]), int(cell["yaw_index"]))
        ]
        for cell in cells
    ]
    backup_counts = [
        backup_counts_by_cell[
            (int(cell["radial_index"]), int(cell["angular_index"]), int(cell["yaw_index"]))
        ]
        for cell in cells
    ]
    used_positions: list[tuple[float, float]] = []
    primary = _make_continuous_candidates(
        cells,
        counts=primary_counts,
        stage="primary",
        origin=origin,
        distribution=distribution,
        evidence_cells=evidence_cells,
        minimum_spacing_m=minimum_spacing_m,
        used_positions=used_positions,
        sequence_offset=candidate_sequence_offset,
    )
    backup = _make_continuous_candidates(
        cells,
        counts=backup_counts,
        stage="backup",
        origin=origin,
        distribution=distribution,
        evidence_cells=evidence_cells,
        minimum_spacing_m=pool_spacing,
        used_positions=used_positions,
        sequence_offset=candidate_sequence_offset + 10_000_000,
    )
    primary_shards = _cell_stratified_partition(primary, shard_count)
    backup_shards = _cell_stratified_partition(backup, shard_count)
    candidates: list[WorkspaceSpawnCandidate] = []
    shards: list[WorkspaceSpawnShard] = []
    for shard_index in range(shard_count):
        start_index = len(candidates)
        rows = [*primary_shards[shard_index], *backup_shards[shard_index]]
        primary_cells = {
            row.source_cell_id for row in primary_shards[shard_index]
        }
        backup_cells = {
            row.source_cell_id for row in backup_shards[shard_index]
        }
        if primary_cells - backup_cells:
            raise ValueError(
                f"workspace shard {shard_index} lacks cell-local backups"
            )
        candidates.extend(rows)
        shards.append(
            WorkspaceSpawnShard(
                shard=f"workspace_{shard_index:02d}",
                start_index=start_index,
                candidate_count=len(rows),
                primary_target_count=len(primary_shards[shard_index]),
            )
        )
    quotas = [
        WorkspaceSpawnCellQuota(
            cell_id=str(cell["cell_id"]),
            radial_index=int(cell["radial_index"]),
            angular_index=int(cell["angular_index"]),
            radius_bounds_m=[float(value) for value in cell["radius_bounds_m"]],
            angle_bounds_deg=[float(value) for value in cell["angle_bounds_deg"]],
            yaw_index=int(cell["yaw_index"]),
            yaw_bounds_deg=[float(value) for value in cell["yaw_bounds_deg"]],
            area_m2=float(cell["area_m2"]),
            target_probability=float(cell["target_probability"]),
            primary_target_count=primary_counts[index],
            backup_count=backup_counts[index],
        )
        for index, cell in enumerate(cells)
    ]
    radial_widths = [hi - lo for lo, hi in radius_bounds]
    angular_widths = [hi - lo for lo, hi in angle_bounds]
    return WorkspaceSpawnCatalog(
        format="so101_workspace_spawn_catalog_v2",
        catalog_id=catalog_id,
        source_workspace_catalog=str(source_path),
        source_workspace_catalog_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        camera_rig_config=str(source["camera_rig_config"]),
        home_qpos=[float(value) for value in source["home_qpos"]],
        object_color=str(source["object_color"]),
        object_half_size_m=float(source["object_half_size_m"]),
        base_origin_world_xy_m=origin,
        candidate_count=len(candidates),
        primary_target_count=primary_target_count,
        backup_count=backup_count,
        source_cell_count=len(cells),
        distance_decay_rate_per_m=decay_rate,
        angular_jitter_max_deg=max(angular_widths) / 2.0,
        radial_jitter_max_m=max(radial_widths) / 2.0,
        candidate_sequence_offset=candidate_sequence_offset,
        sampling_strategy="continuous_joint_feasible_yaw_stratified_v5",
        enforce_cell_local_quota=True,
        continuous_distribution=distribution,
        cell_quotas=quotas,
        candidates=candidates,
        shards=shards,
    )


def build_continuous_area_workspace_spawn_catalog(
    *,
    source_path: Path,
    catalog_id: str,
    primary_target_count: int,
    backup_count: int,
    shard_count: int,
    radius_min_m: float,
    radius_max_m: float,
    angle_min_deg: float,
    angle_max_deg: float,
    radial_strata: int,
    angular_strata: int,
    far_to_near_area_density_ratio: float,
    minimum_spacing_m: float,
    object_yaw_min_deg: float | None = None,
    object_yaw_max_deg: float | None = None,
    object_yaw_strata: int = 0,
    object_yaw_periodicity_deg: float = 90.0,
    object_yaw_reference_frame: Literal[
        "robot_relative", "world_absolute"
    ] = "robot_relative",
) -> WorkspaceSpawnCatalog:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("format") != "so101_grasp_workspace_catalog_v1":
        raise ValueError(f"unsupported workspace source format: {source_path}")
    evidence_cells = list(source.get("cells") or [])
    if not evidence_cells:
        raise ValueError("workspace source catalog contains no cells")
    if primary_target_count % shard_count or backup_count % shard_count:
        raise ValueError("primary and backup counts must divide evenly across shards")
    yaw_values = (object_yaw_min_deg, object_yaw_max_deg, object_yaw_strata)
    yaw_enabled = any(
        value is not None and value != 0
        for value in yaw_values
    )
    if yaw_enabled and (
        object_yaw_min_deg is None
        or object_yaw_max_deg is None
        or int(object_yaw_strata) <= 0
    ):
        raise ValueError(
            "independent object yaw requires min, max, and positive strata"
        )
    object_yaw = (
        IndependentObjectYawDistribution(
            reference_frame=object_yaw_reference_frame,
            min_deg=float(object_yaw_min_deg),
            max_deg=float(object_yaw_max_deg),
            strata=int(object_yaw_strata),
            periodicity_deg=float(object_yaw_periodicity_deg),
        )
        if yaw_enabled
        else None
    )
    distribution = ContinuousAreaDistribution(
        radius_min_m=radius_min_m,
        radius_max_m=radius_max_m,
        angle_min_deg=angle_min_deg,
        angle_max_deg=angle_max_deg,
        radial_strata=radial_strata,
        angular_strata=angular_strata,
        far_to_near_area_density_ratio=far_to_near_area_density_ratio,
        minimum_spacing_m=minimum_spacing_m,
        object_yaw=object_yaw,
    )
    _validate_continuous_domain_evidence(distribution, evidence_cells)

    cells = _continuous_area_cells(distribution)
    weights = [float(row["target_probability"]) for row in cells]
    primary_counts = _allocate_counts(
        primary_target_count,
        weights,
        minimum_per_cell=1,
    )
    backup_counts = _allocate_counts(
        backup_count,
        weights,
        minimum_per_cell=1,
    )
    origin = [float(value) for value in source["base_contract"]["world_xyz_m"][:2]]
    used_positions: list[tuple[float, float]] = []
    primary = _make_continuous_candidates(
        cells,
        counts=primary_counts,
        stage="primary",
        origin=origin,
        distribution=distribution,
        evidence_cells=evidence_cells,
        minimum_spacing_m=minimum_spacing_m,
        used_positions=used_positions,
        sequence_offset=0,
    )
    backup = _make_continuous_candidates(
        cells,
        counts=backup_counts,
        stage="backup",
        origin=origin,
        distribution=distribution,
        evidence_cells=evidence_cells,
        minimum_spacing_m=minimum_spacing_m,
        used_positions=used_positions,
        sequence_offset=10_000_000,
    )
    primary_shards = _cell_stratified_partition(primary, shard_count)
    backup_shards = _cell_stratified_partition(backup, shard_count)

    candidates: list[WorkspaceSpawnCandidate] = []
    shards: list[WorkspaceSpawnShard] = []
    for shard_index in range(shard_count):
        start_index = len(candidates)
        rows = [*primary_shards[shard_index], *backup_shards[shard_index]]
        primary_cells = Counter(
            row.source_cell_id for row in primary_shards[shard_index]
        )
        backup_cells = Counter(
            row.source_cell_id for row in backup_shards[shard_index]
        )
        missing_backups = sorted(set(primary_cells) - set(backup_cells))
        if missing_backups:
            raise ValueError(
                f"workspace shard {shard_index} lacks cell-local backups: "
                + ", ".join(missing_backups)
            )
        candidates.extend(rows)
        shards.append(
            WorkspaceSpawnShard(
                shard=f"workspace_{shard_index:02d}",
                start_index=start_index,
                candidate_count=len(rows),
                primary_target_count=len(primary_shards[shard_index]),
            )
        )

    quotas = [
        WorkspaceSpawnCellQuota(
            cell_id=str(cell["cell_id"]),
            radial_index=int(cell["radial_index"]),
            angular_index=int(cell["angular_index"]),
            radius_bounds_m=[float(value) for value in cell["radius_bounds_m"]],
            angle_bounds_deg=[float(value) for value in cell["angle_bounds_deg"]],
            yaw_index=(
                None
                if cell.get("yaw_index") is None
                else int(cell["yaw_index"])
            ),
            yaw_bounds_deg=(
                None
                if cell.get("yaw_bounds_deg") is None
                else [float(value) for value in cell["yaw_bounds_deg"]]
            ),
            area_m2=float(cell["area_m2"]),
            target_probability=float(cell["target_probability"]),
            primary_target_count=int(primary_counts[index]),
            backup_count=int(backup_counts[index]),
        )
        for index, cell in enumerate(cells)
    ]
    radial_width = (radius_max_m - radius_min_m) / radial_strata
    angular_width = (angle_max_deg - angle_min_deg) / angular_strata
    decay_rate = -math.log(far_to_near_area_density_ratio) / (
        radius_max_m - radius_min_m
    )
    return WorkspaceSpawnCatalog(
        format="so101_workspace_spawn_catalog_v2",
        catalog_id=catalog_id,
        source_workspace_catalog=str(source_path),
        source_workspace_catalog_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        camera_rig_config=str(source["camera_rig_config"]),
        home_qpos=[float(value) for value in source["home_qpos"]],
        object_color=str(source["object_color"]),
        object_half_size_m=float(source["object_half_size_m"]),
        base_origin_world_xy_m=origin,
        candidate_count=len(candidates),
        primary_target_count=primary_target_count,
        backup_count=backup_count,
        source_cell_count=len(cells),
        distance_decay_rate_per_m=decay_rate,
        angular_jitter_max_deg=angular_width / 2.0,
        radial_jitter_max_m=radial_width / 2.0,
        sampling_strategy=(
            "continuous_area_yaw_stratified_v4"
            if object_yaw is not None
            else "continuous_area_stratified_v3"
        ),
        enforce_cell_local_quota=True,
        continuous_distribution=distribution,
        cell_quotas=quotas,
        candidates=candidates,
        shards=shards,
    )


class WorkspaceCellQuotaScheduler:
    """Select candidates while preserving each shard's primary cell quotas."""

    def __init__(
        self,
        candidates: list[WorkspaceSpawnCandidate],
        *,
        accepted_minimum_spacing_m: float = 0.0,
        forbidden_positions: list[tuple[float, float]] | None = None,
    ) -> None:
        if accepted_minimum_spacing_m < 0.0:
            raise ValueError("accepted minimum spacing cannot be negative")
        primary = Counter(
            row.source_cell_id for row in candidates if row.stage == "primary"
        )
        if not primary:
            raise ValueError("cell-local scheduler requires primary candidates")
        grouped: dict[str, list[WorkspaceSpawnCandidate]] = defaultdict(list)
        for row in candidates:
            grouped[row.source_cell_id].append(row)
        self._targets = dict(sorted(primary.items()))
        self._queues = {
            cell_id: deque(
                sorted(
                    grouped[cell_id],
                    key=lambda row: (row.stage != "primary", row.candidate_id),
                )
            )
            for cell_id in self._targets
        }
        missing = [
            cell_id
            for cell_id, rows in self._queues.items()
            if len(rows) <= self._targets[cell_id]
        ]
        if missing:
            raise ValueError(
                "cell-local scheduler requires at least one backup per cell: "
                + ", ".join(missing)
            )
        self._accepted: Counter[str] = Counter()
        self._attempted: Counter[str] = Counter()
        self._spacing_skipped: Counter[str] = Counter()
        self._accepted_positions: list[tuple[float, float]] = []
        self._forbidden_positions = [
            (float(position[0]), float(position[1]))
            for position in (forbidden_positions or [])
        ]
        self._accepted_minimum_spacing_m = float(accepted_minimum_spacing_m)
        self._cell_ids = list(self._targets)
        self._cursor = 0

    @property
    def complete(self) -> bool:
        return all(
            self._accepted[cell_id] >= target
            for cell_id, target in self._targets.items()
        )

    def next_candidate(self) -> WorkspaceSpawnCandidate:
        if self.complete:
            raise StopIteration
        for _ in range(len(self._cell_ids)):
            cell_id = self._cell_ids[self._cursor % len(self._cell_ids)]
            self._cursor += 1
            if self._accepted[cell_id] >= self._targets[cell_id]:
                continue
            queue = self._queues[cell_id]
            while queue:
                row = queue.popleft()
                position = tuple(float(value) for value in row.world_xy_m)
                if any(
                    math.hypot(position[0] - prior[0], position[1] - prior[1])
                    < self._accepted_minimum_spacing_m - 1e-12
                    for prior in (
                        self._forbidden_positions + self._accepted_positions
                    )
                ):
                    self._spacing_skipped[cell_id] += 1
                    continue
                self._attempted[cell_id] += 1
                return row
            raise RuntimeError(
                f"workspace cell exhausted before quota: {cell_id} "
                f"accepted={self._accepted[cell_id]} target={self._targets[cell_id]}"
            )
        raise RuntimeError("workspace quota scheduler could not find an unfinished cell")

    def record_success(self, candidate: WorkspaceSpawnCandidate) -> None:
        cell_id = candidate.source_cell_id
        if cell_id not in self._targets:
            raise ValueError(f"unknown workspace quota cell: {cell_id}")
        if self._accepted[cell_id] >= self._targets[cell_id]:
            raise ValueError(f"workspace quota already full: {cell_id}")
        self._accepted[cell_id] += 1
        self._accepted_positions.append(
            tuple(float(value) for value in candidate.world_xy_m)
        )

    def summary(self) -> dict[str, Any]:
        cells = {
            cell_id: {
                "target": int(self._targets[cell_id]),
                "accepted": int(self._accepted[cell_id]),
                "attempted": int(self._attempted[cell_id]),
                "spacing_skipped": int(self._spacing_skipped[cell_id]),
                "remaining_candidates": len(self._queues[cell_id]),
            }
            for cell_id in self._cell_ids
        }
        return {
            "complete": self.complete,
            "target_total": sum(self._targets.values()),
            "accepted_total": sum(self._accepted.values()),
            "attempted_total": sum(self._attempted.values()),
            "spacing_skipped_total": sum(self._spacing_skipped.values()),
            "forbidden_position_count": len(self._forbidden_positions),
            "cells": cells,
        }


def _unique_sorted_coordinates(values: Any) -> list[float]:
    return sorted({round(float(value), 9) for value in values})


def _normalize_periodic_coordinate(value: float, period: float) -> float:
    normalized = float(value) % float(period)
    return 0.0 if math.isclose(normalized, period, abs_tol=1e-8) else normalized


def _unique_periodic_coordinates(values: Any, *, period: float) -> list[float]:
    normalized = []
    for value in values:
        normalized.append(_normalize_periodic_coordinate(value, period))
    return _unique_sorted_coordinates(normalized)


def _coordinate_bounds(
    centers: list[float],
    lower: float,
    upper: float,
) -> list[tuple[float, float]]:
    if not centers or lower > centers[0] or upper < centers[-1]:
        raise ValueError("coordinate bounds must contain every evidence center")
    boundaries = [float(lower)]
    boundaries.extend(
        (left + right) / 2.0
        for left, right in zip(centers, centers[1:])
    )
    boundaries.append(float(upper))
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def _allocate_sparse_matrix(
    row_targets: list[int],
    column_targets: list[int],
    *,
    allowed: set[tuple[int, int]],
    edge_capacity: int | None = None,
) -> dict[tuple[int, int], int]:
    """Solve integer row/column margins over an explicit sparse support."""

    if sum(row_targets) != sum(column_targets):
        raise ValueError("sparse matrix margins must have equal totals")
    row_count = len(row_targets)
    column_count = len(column_targets)
    if any(not any(row == index for row, _ in allowed) for index in range(row_count)):
        raise ValueError("every sparse matrix row must have allowed support")
    if any(
        not any(column == index for _, column in allowed)
        for index in range(column_count)
    ):
        raise ValueError("every sparse matrix column must have allowed support")

    source = ("source", -1)
    sink = ("sink", -1)
    graph: dict[tuple[str, int], dict[tuple[str, int], int]] = defaultdict(dict)

    def add_edge(
        left: tuple[str, int],
        right: tuple[str, int],
        capacity: int,
    ) -> None:
        graph[left][right] = capacity
        graph[right].setdefault(left, 0)

    for row, target in enumerate(row_targets):
        add_edge(source, ("row", row), int(target))
    for row, column in sorted(allowed):
        add_edge(
            ("row", row),
            ("column", column),
            sum(row_targets) if edge_capacity is None else edge_capacity,
        )
    for column, target in enumerate(column_targets):
        add_edge(("column", column), sink, int(target))

    flow = 0
    while True:
        parents: dict[tuple[str, int], tuple[str, int] | None] = {source: None}
        queue = deque([source])
        while queue and sink not in parents:
            left = queue.popleft()
            for right, capacity in graph[left].items():
                if capacity > 0 and right not in parents:
                    parents[right] = left
                    queue.append(right)
        if sink not in parents:
            break
        amount = sum(row_targets)
        node = sink
        while parents[node] is not None:
            parent = parents[node]
            amount = min(amount, graph[parent][node])
            node = parent
        node = sink
        while parents[node] is not None:
            parent = parents[node]
            graph[parent][node] -= amount
            graph[node][parent] += amount
            node = parent
        flow += amount
    if flow != sum(row_targets):
        raise ValueError("sparse feasibility mask cannot satisfy requested margins")

    result: dict[tuple[int, int], int] = {}
    for row, column in sorted(allowed):
        value = graph[("column", column)][("row", row)]
        if value > 0:
            result[(row, column)] = value
    return result


def _distribute_pairs_across_angles(
    pair_counts: dict[tuple[int, int], int],
    *,
    angle_target: int,
    angle_count: int,
) -> dict[tuple[int, int, int], int]:
    pairs = sorted(pair_counts)
    base_counts = [pair_counts[pair] // angle_count for pair in pairs]
    pair_targets = [pair_counts[pair] % angle_count for pair in pairs]
    base_angle_count = sum(base_counts)
    angle_targets = [angle_target - base_angle_count] * angle_count
    allowed = {
        (pair_index, angle_index)
        for pair_index in range(len(pairs))
        for angle_index in range(angle_count)
    }
    allocation = _allocate_sparse_matrix(
        pair_targets,
        angle_targets,
        allowed=allowed,
        edge_capacity=1,
    )
    result: dict[tuple[int, int, int], int] = {}
    for pair_index, (radial, yaw) in enumerate(pairs):
        for angle_index in range(angle_count):
            count = base_counts[pair_index] + allocation.get(
                (pair_index, angle_index), 0
            )
            if count > 0:
                result[(radial, angle_index, yaw)] = count
    missing_angles = {
        angle
        for angle in range(angle_count)
        if sum(
            count
            for (_, candidate_angle, _), count in result.items()
            if candidate_angle == angle
        )
        != angle_target
    }
    if missing_angles:
        raise RuntimeError("failed to satisfy angular quotas")
    return result


def _validate_continuous_domain_evidence(
    distribution: ContinuousAreaDistribution,
    evidence_cells: list[dict[str, Any]],
) -> None:
    evidence_radii = [float(row["radius_from_base_m"]) for row in evidence_cells]
    evidence_angles = [float(row["angle_from_base_deg"]) for row in evidence_cells]
    tolerance = 1e-9
    if min(evidence_radii) > distribution.radius_min_m + tolerance:
        raise ValueError("continuous radius minimum is outside feasible evidence")
    if max(evidence_radii) < distribution.radius_max_m - tolerance:
        raise ValueError("continuous radius maximum is outside feasible evidence")
    if min(evidence_angles) > distribution.angle_min_deg + tolerance:
        raise ValueError("continuous angle minimum is outside feasible evidence")
    if max(evidence_angles) < distribution.angle_max_deg - tolerance:
        raise ValueError("continuous angle maximum is outside feasible evidence")


def _continuous_area_cells(
    distribution: ContinuousAreaDistribution,
) -> list[dict[str, Any]]:
    radial_width = (
        distribution.radius_max_m - distribution.radius_min_m
    ) / distribution.radial_strata
    angular_width = (
        distribution.angle_max_deg - distribution.angle_min_deg
    ) / distribution.angular_strata
    decay_rate = -math.log(distribution.far_to_near_area_density_ratio) / (
        distribution.radius_max_m - distribution.radius_min_m
    )
    rows: list[dict[str, Any]] = []
    raw_weights: list[float] = []
    yaw_distribution = distribution.object_yaw
    yaw_rows: list[tuple[int | None, list[float] | None]]
    if yaw_distribution is None:
        yaw_rows = [(None, None)]
    else:
        yaw_width = (
            yaw_distribution.max_deg - yaw_distribution.min_deg
        ) / yaw_distribution.strata
        yaw_rows = [
            (
                yaw_index,
                [
                    yaw_distribution.min_deg + yaw_index * yaw_width,
                    yaw_distribution.min_deg + (yaw_index + 1) * yaw_width,
                ],
            )
            for yaw_index in range(yaw_distribution.strata)
        ]
    for radial_index in range(distribution.radial_strata):
        radius_lo = distribution.radius_min_m + radial_index * radial_width
        radius_hi = radius_lo + radial_width
        radial_weight = _radial_density_integral(
            radius_lo,
            radius_hi,
            radius_origin=distribution.radius_min_m,
            decay_rate=decay_rate,
        )
        for angular_index in range(distribution.angular_strata):
            angle_lo = distribution.angle_min_deg + angular_index * angular_width
            angle_hi = angle_lo + angular_width
            angle_width_rad = math.radians(angle_hi - angle_lo)
            area = 0.5 * (radius_hi**2 - radius_lo**2) * angle_width_rad
            for yaw_index, yaw_bounds in yaw_rows:
                yaw_suffix = "" if yaw_index is None else f"_y{yaw_index:02d}"
                raw_weight = (
                    radial_weight * angle_width_rad / len(yaw_rows)
                )
                rows.append(
                    {
                        "cell_id": (
                            f"r{radial_index:02d}_a{angular_index:02d}"
                            f"{yaw_suffix}"
                        ),
                        "radial_index": radial_index,
                        "angular_index": angular_index,
                        "radius_bounds_m": [radius_lo, radius_hi],
                        "angle_bounds_deg": [angle_lo, angle_hi],
                        "yaw_index": yaw_index,
                        "yaw_bounds_deg": yaw_bounds,
                        "area_m2": area / len(yaw_rows),
                    }
                )
                raw_weights.append(raw_weight)
    total = sum(raw_weights)
    for row, raw_weight in zip(rows, raw_weights, strict=True):
        row["target_probability"] = raw_weight / total
    return rows


def _make_continuous_candidates(
    cells: list[dict[str, Any]],
    *,
    counts: list[int],
    stage: Literal["primary", "backup"],
    origin: list[float],
    distribution: ContinuousAreaDistribution,
    evidence_cells: list[dict[str, Any]],
    minimum_spacing_m: float,
    used_positions: list[tuple[float, float]],
    sequence_offset: int,
) -> list[WorkspaceSpawnCandidate]:
    result: list[WorkspaceSpawnCandidate] = []
    for cell_index, (cell, count) in enumerate(zip(cells, counts, strict=True)):
        radius_lo, radius_hi = [float(value) for value in cell["radius_bounds_m"]]
        angle_lo, angle_hi = [float(value) for value in cell["angle_bounds_deg"]]
        yaw_bounds = cell.get("yaw_bounds_deg")
        radius_origin = distribution.radius_min_m
        decay_rate = -math.log(
            distribution.far_to_near_area_density_ratio
        ) / (distribution.radius_max_m - distribution.radius_min_m)
        cursor = sequence_offset + cell_index * 100_000 + 1
        for local_index in range(count):
            for _ in range(100_000):
                angle_unit = _radical_inverse(cursor, base=2)
                radius_unit = _radical_inverse(cursor, base=3)
                yaw_unit = _radical_inverse(cursor, base=5)
                cursor += 1
                angle = angle_lo + (angle_hi - angle_lo) * angle_unit
                radius = _inverse_radial_density_cdf(
                    radius_unit,
                    lower=radius_lo,
                    upper=radius_hi,
                    radius_origin=radius_origin,
                    decay_rate=decay_rate,
                )
                angle_rad = math.radians(angle)
                base_xy = (radius * math.cos(angle_rad), radius * math.sin(angle_rad))
                world_xy = (origin[0] + base_xy[0], origin[1] + base_xy[1])
                if all(
                    math.hypot(world_xy[0] - prior[0], world_xy[1] - prior[1])
                    >= minimum_spacing_m
                    for prior in used_positions
                ):
                    break
            else:
                raise RuntimeError(
                    f"unable to satisfy minimum spacing in {cell['cell_id']}"
                )
            used_positions.append(world_xy)
            center_radius = float(
                cell.get("evidence_radius_center_m", (radius_lo + radius_hi) / 2.0)
            )
            center_angle = float(
                cell.get("evidence_angle_center_deg", (angle_lo + angle_hi) / 2.0)
            )
            object_yaw_deg = _wrap_degrees(angle - 80.0)
            if yaw_bounds is not None:
                yaw_lo, yaw_hi = [float(value) for value in yaw_bounds]
                sampled_yaw = yaw_lo + (yaw_hi - yaw_lo) * yaw_unit
                yaw_distribution = distribution.object_yaw
                if (
                    yaw_distribution is not None
                    and yaw_distribution.reference_frame == "robot_relative"
                ):
                    object_yaw_deg = _wrap_degrees(angle + sampled_yaw)
                else:
                    object_yaw_deg = _wrap_degrees(sampled_yaw)
            result.append(
                WorkspaceSpawnCandidate(
                    candidate_id=(
                        f"{stage}-{cell['cell_id']}-{local_index:03d}"
                    ),
                    source_cell_id=str(cell["cell_id"]),
                    stage=stage,
                    world_xy_m=list(world_xy),
                    base_xy_m=list(base_xy),
                    radius_from_base_m=radius,
                    angle_from_base_deg=angle,
                    object_yaw_deg=object_yaw_deg,
                    sampling_weight=float(cell["target_probability"]),
                    camera1_grid_bin=_nearest_evidence_camera_bin(
                        radius,
                        angle,
                        evidence_cells,
                    ),
                    radial_offset_m=radius - center_radius,
                    angular_offset_deg=angle - center_angle,
                )
            )
    return result


def _radial_density_integral(
    lower: float,
    upper: float,
    *,
    radius_origin: float,
    decay_rate: float,
) -> float:
    if math.isclose(decay_rate, 0.0, abs_tol=1e-12):
        return 0.5 * (upper**2 - lower**2)

    def primitive(radius: float) -> float:
        return -(
            radius / decay_rate + 1.0 / decay_rate**2
        ) * math.exp(-decay_rate * (radius - radius_origin))

    return primitive(upper) - primitive(lower)


def _inverse_radial_density_cdf(
    unit: float,
    *,
    lower: float,
    upper: float,
    radius_origin: float,
    decay_rate: float,
) -> float:
    target = float(unit) * _radial_density_integral(
        lower,
        upper,
        radius_origin=radius_origin,
        decay_rate=decay_rate,
    )
    lo, hi = lower, upper
    for _ in range(60):
        mid = (lo + hi) / 2.0
        value = _radial_density_integral(
            lower,
            mid,
            radius_origin=radius_origin,
            decay_rate=decay_rate,
        )
        if value < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _radical_inverse(index: int, *, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    while index > 0:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def _nearest_evidence_camera_bin(
    radius: float,
    angle: float,
    evidence_cells: list[dict[str, Any]],
) -> int | None:
    candidates = [
        row for row in evidence_cells if row.get("camera1_grid_bin") is not None
    ]
    if not candidates:
        return None
    nearest = min(
        candidates,
        key=lambda row: (
            (float(row["radius_from_base_m"]) - radius) ** 2
            + (math.radians(float(row["angle_from_base_deg"]) - angle) * radius) ** 2
        ),
    )
    return int(nearest["camera1_grid_bin"])


def _normalized_cell_weights(
    cells: list[dict],
    *,
    distance_decay_rate_per_m: float,
) -> list[float]:
    min_radius = min(float(cell["radius_from_base_m"]) for cell in cells)
    raw = [
        float(cell["uniform_area_weight"])
        * math.exp(
            -float(distance_decay_rate_per_m)
            * (float(cell["radius_from_base_m"]) - min_radius)
        )
        for cell in cells
    ]
    total = sum(raw)
    return [value / total for value in raw]


def _allocate_counts(
    total: int,
    weights: list[float],
    *,
    minimum_per_cell: int,
) -> list[int]:
    required = minimum_per_cell * len(weights)
    if total < required:
        raise ValueError("allocation total is below the per-cell minimum")
    remaining = total - required
    exact = [remaining * weight for weight in weights]
    floors = [int(math.floor(value)) for value in exact]
    counts = [minimum_per_cell + value for value in floors]
    leftovers = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (exact[index] - floors[index], weights[index], -index),
        reverse=True,
    )
    for index in order[:leftovers]:
        counts[index] += 1
    return counts


def _make_candidates(
    cells: list[dict],
    *,
    counts: list[int],
    stage: Literal["primary", "backup"],
    origin: list[float],
    weights: list[float],
    angular_jitter_max_deg: float,
    radial_jitter_max_m: float,
    sequence_offset: int,
    preserve_evidence_object_yaw: bool,
    object_yaw_jitter_half_range_deg: float,
) -> list[WorkspaceSpawnCandidate]:
    result: list[WorkspaceSpawnCandidate] = []
    min_angle = min(float(cell["angle_from_base_deg"]) for cell in cells)
    max_angle = max(float(cell["angle_from_base_deg"]) for cell in cells)
    min_radius = min(float(cell["radius_from_base_m"]) for cell in cells)
    max_radius = max(float(cell["radius_from_base_m"]) for cell in cells)
    for cell_index, (cell, count) in enumerate(zip(cells, counts, strict=True)):
        center_angle = float(cell["angle_from_base_deg"])
        center_radius = float(cell["radius_from_base_m"])
        for local_index in range(count):
            is_tested_cell_center = (
                sequence_offset == 0 and stage == "primary" and local_index == 0
            )
            sequence_index = (
                sequence_offset
                + 1
                + cell_index * 997
                + local_index
            )
            angular_offset = (
                0.0
                if is_tested_cell_center
                else _quasi_random_offset(
                    sequence_index,
                    maximum=angular_jitter_max_deg,
                    irrational=0.6180339887498949,
                )
            )
            radial_offset = (
                0.0
                if is_tested_cell_center
                else _quasi_random_offset(
                    sequence_index,
                    maximum=radial_jitter_max_m,
                    irrational=0.7548776662466927,
                )
            )
            object_yaw_offset = (
                0.0
                if is_tested_cell_center
                else _quasi_random_offset(
                    sequence_index,
                    maximum=object_yaw_jitter_half_range_deg,
                    irrational=0.5698402909980532,
                )
            )
            if math.isclose(center_angle, min_angle, abs_tol=1e-9):
                angular_offset = abs(angular_offset)
            elif math.isclose(center_angle, max_angle, abs_tol=1e-9):
                angular_offset = -abs(angular_offset)
            if math.isclose(center_radius, min_radius, abs_tol=1e-9):
                radial_offset = abs(radial_offset)
            elif math.isclose(center_radius, max_radius, abs_tol=1e-9):
                radial_offset = -abs(radial_offset)
            angle = center_angle + angular_offset
            radius = center_radius + radial_offset
            angle_rad = math.radians(angle)
            base_xy = [radius * math.cos(angle_rad), radius * math.sin(angle_rad)]
            world_xy = [origin[0] + base_xy[0], origin[1] + base_xy[1]]
            object_yaw_deg = _wrap_degrees(angle - 80.0)
            if preserve_evidence_object_yaw:
                object_yaw_deg = _wrap_degrees(
                    float(cell["object_yaw_deg"]) + object_yaw_offset
                )
            result.append(
                WorkspaceSpawnCandidate(
                    candidate_id=(
                        f"{stage}-{cell_index:03d}-"
                        f"{local_index + sequence_offset:09d}"
                    ),
                    source_cell_id=str(cell["point_id"]),
                    stage=stage,
                    world_xy_m=world_xy,
                    base_xy_m=base_xy,
                    radius_from_base_m=radius,
                    angle_from_base_deg=angle,
                    object_yaw_deg=object_yaw_deg,
                    sampling_weight=float(weights[cell_index]),
                    camera1_grid_bin=int(cell["camera1_grid_bin"]),
                    radial_offset_m=float(radial_offset),
                    angular_offset_deg=float(angular_offset),
                )
            )
    return result


def _quasi_random_offset(
    sequence_index: int,
    *,
    maximum: float,
    irrational: float,
) -> float:
    if maximum <= 0.0:
        return 0.0
    unit = ((float(sequence_index) + 0.5) * float(irrational)) % 1.0
    return (2.0 * unit - 1.0) * float(maximum)


def _angular_offsets(count: int, *, maximum: float, sequence_offset: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1 and sequence_offset == 0:
        return [0.0]
    golden = 0.6180339887498949
    result = []
    for index in range(count):
        sequence_index = index + sequence_offset
        unit = ((sequence_index + 0.5) * golden) % 1.0
        offset = (2.0 * unit - 1.0) * maximum
        if sequence_index == 0:
            offset = 0.0
        result.append(offset)
    return result


def _stratified_partition(
    candidates: list[WorkspaceSpawnCandidate],
    shard_count: int,
) -> list[list[WorkspaceSpawnCandidate]]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            row.radius_from_base_m,
            row.angle_from_base_deg,
            row.source_cell_id,
            row.candidate_id,
        ),
    )
    shards: list[list[WorkspaceSpawnCandidate]] = [[] for _ in range(shard_count)]
    for index, candidate in enumerate(ordered):
        shards[index % shard_count].append(candidate)
    return shards


def _cell_stratified_partition(
    candidates: list[WorkspaceSpawnCandidate],
    shard_count: int,
) -> list[list[WorkspaceSpawnCandidate]]:
    """Spread every cell across shards while keeping shard sizes balanced."""

    grouped: dict[str, list[WorkspaceSpawnCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.source_cell_id].append(candidate)
    shards: list[list[WorkspaceSpawnCandidate]] = [[] for _ in range(shard_count)]
    for cell_index, cell_id in enumerate(sorted(grouped)):
        rows = sorted(grouped[cell_id], key=lambda row: row.candidate_id)
        per_shard, remainder = divmod(len(rows), shard_count)
        cursor = 0
        for shard in shards:
            shard.extend(rows[cursor : cursor + per_shard])
            cursor += per_shard
        remainder_order = sorted(
            range(shard_count),
            key=lambda shard_index: (
                len(shards[shard_index]),
                (shard_index - cell_index) % shard_count,
            ),
        )
        for shard_index in remainder_order[:remainder]:
            shards[shard_index].append(rows[cursor])
            cursor += 1
        if cursor != len(rows):
            raise RuntimeError(f"failed to partition workspace cell: {cell_id}")
    return shards


def _wrap_degrees(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _object_yaw_distribution_coordinate(
    candidate: WorkspaceSpawnCandidate,
    distribution: IndependentObjectYawDistribution,
) -> float:
    value = float(candidate.object_yaw_deg)
    if distribution.reference_frame == "robot_relative":
        value -= float(candidate.angle_from_base_deg)
    return distribution.min_deg + (
        (value - distribution.min_deg) % distribution.periodicity_deg
    )
