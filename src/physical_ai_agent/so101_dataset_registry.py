"""Canonical registry for recipe-backed SO101 datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from physical_ai_agent.so101_dataset_generation_schema import DatasetGenerationRecipe


REGISTRY_SCHEMA_VERSION = 1
SUPPORTED_RECIPE_SCHEMA_VERSIONS = {1, 2}
DATASET_RECIPE_DIR = Path("configs/so101/dataset_generation")
DATASET_ROOT = Path("_workspace/so101_lerobot")
REQUIRED_CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
)
REQUIRED_JOINT_KEYS = ("observation.state", "action")


class DatasetRegistryError(ValueError):
    """Raised when a recipe violates the canonical dataset registry contract."""


@dataclass(frozen=True)
class RegistryIssue:
    code: str
    message: str
    recipe: str | None = None
    split: str | None = None


@dataclass(frozen=True)
class DatasetRegistryEntry:
    dataset_id: str
    split: str
    catalog_name: str
    recipe_path: str
    output_root: str
    absolute_root: str
    repo_id: str | None
    expected_episodes: int | None
    status: str
    episodes: int | None
    frames: int | None
    fps: int | float | None
    size_bytes: int | None
    audit_status: str | None
    grid_sidecar: str | None
    closed_loop_start: str | None
    training_ready: bool
    readiness_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["readiness_errors"] = list(self.readiness_errors)
        return payload


@dataclass(frozen=True)
class DatasetRegistry:
    repo_root: str
    recipe_dir: str
    dataset_root: str
    entries: tuple[DatasetRegistryEntry, ...]
    issues: tuple[RegistryIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    @property
    def training_ready(self) -> bool:
        return self.valid and bool(self.entries) and all(entry.training_ready for entry in self.entries)

    @property
    def training_manifests(self) -> dict[str, dict[str, Any]]:
        manifests: dict[str, dict[str, Any]] = {}
        for dataset_id in sorted({entry.dataset_id for entry in self.entries}):
            entries = [entry for entry in self.entries if entry.dataset_id == dataset_id]
            train_entries = [entry for entry in entries if entry.split == "train"]
            validation_entries = [entry for entry in entries if entry.split in {"validation", "val"}]
            manifests[dataset_id] = {
                "name": dataset_id,
                "source_recipe": entries[0].recipe_path,
                "training_ready": bool(entries) and all(entry.training_ready for entry in entries),
                "train_datasets": [_training_dataset_row(entry) for entry in train_entries],
                "validation_dataset": (
                    _training_dataset_row(validation_entries[0]) if validation_entries else None
                ),
            }
        return manifests

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "repo_root": self.repo_root,
            "recipe_dir": self.recipe_dir,
            "dataset_root": self.dataset_root,
            "valid": self.valid,
            "training_ready": self.training_ready,
            "summary": {
                "recipes": len({entry.recipe_path for entry in self.entries}),
                "splits": len(self.entries),
                "available": sum(entry.status == "available" for entry in self.entries),
                "training_ready": sum(entry.training_ready for entry in self.entries),
                "issues": len(self.issues),
            },
            "issues": [asdict(issue) for issue in self.issues],
            "datasets": [entry.to_dict() for entry in self.entries],
            "training_manifests": self.training_manifests,
        }


def scan_dataset_registry(
    repo_root: Path,
    *,
    inspect_artifacts: bool = True,
    recipe_paths: Iterable[Path] | None = None,
) -> DatasetRegistry:
    repo_root = repo_root.resolve()
    recipe_dir = (repo_root / DATASET_RECIPE_DIR).resolve()
    dataset_root = (repo_root / DATASET_ROOT).resolve()
    paths = sorted(recipe_paths or recipe_dir.glob("*.json"))
    entries: list[DatasetRegistryEntry] = []
    issues: list[RegistryIssue] = []
    seen_ids: dict[str, Path] = {}
    seen_catalog_names: dict[str, Path] = {}
    seen_roots: dict[Path, Path] = {}

    for raw_path in paths:
        recipe_path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        recipe_path = recipe_path.resolve()
        relative_recipe = _relative_or_absolute(recipe_path, repo_root)
        if not _is_relative_to(recipe_path, recipe_dir):
            issues.append(
                RegistryIssue(
                    "recipe_outside_registry",
                    f"recipe must be stored under {DATASET_RECIPE_DIR}",
                    relative_recipe,
                )
            )
            continue
        try:
            raw_recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            recipe = (
                DatasetGenerationRecipe.model_validate(raw_recipe).as_dict()
                if "exporter" in raw_recipe
                else raw_recipe
            )
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            issues.append(RegistryIssue("invalid_recipe_json", str(exc), relative_recipe))
            continue

        recipe_schema_version = int(recipe.get("schema_version", 0))
        if recipe_schema_version not in SUPPORTED_RECIPE_SCHEMA_VERSIONS:
            issues.append(
                RegistryIssue(
                    "unsupported_schema_version",
                    "recipe schema_version must be one of "
                    f"{sorted(SUPPORTED_RECIPE_SCHEMA_VERSIONS)}",
                    relative_recipe,
                )
            )

        dataset_id = str(recipe.get("name") or "").strip()
        if not dataset_id:
            issues.append(RegistryIssue("missing_dataset_id", "recipe.name is required", relative_recipe))
            continue
        if recipe_path.stem != dataset_id:
            issues.append(
                RegistryIssue(
                    "recipe_name_mismatch",
                    f"recipe filename '{recipe_path.stem}' must match recipe.name '{dataset_id}'",
                    relative_recipe,
                )
            )
        if dataset_id in seen_ids:
            issues.append(
                RegistryIssue(
                    "duplicate_dataset_id",
                    f"dataset id '{dataset_id}' is also declared by {_relative_or_absolute(seen_ids[dataset_id], repo_root)}",
                    relative_recipe,
                )
            )
        else:
            seen_ids[dataset_id] = recipe_path

        splits = recipe.get("splits")
        if not isinstance(splits, dict) or not splits:
            issues.append(RegistryIssue("missing_splits", "recipe.splits must be a non-empty object", relative_recipe))
            continue
        for split_name, split_spec in splits.items():
            if not isinstance(split_spec, dict) or not split_spec.get("output_root"):
                issues.append(
                    RegistryIssue(
                        "missing_output_root",
                        "split.output_root is required",
                        relative_recipe,
                        str(split_name),
                    )
                )
                continue
            output_root = Path(str(split_spec["output_root"]))
            if output_root.is_absolute():
                issues.append(
                    RegistryIssue(
                        "absolute_output_root",
                        "output_root must be repository-relative for portability",
                        relative_recipe,
                        str(split_name),
                    )
                )
                absolute_root = output_root.resolve()
            else:
                absolute_root = (repo_root / output_root).resolve()
            if not _is_relative_to(absolute_root, dataset_root):
                issues.append(
                    RegistryIssue(
                        "output_outside_dataset_root",
                        f"output_root must be under {DATASET_ROOT}: {output_root}",
                        relative_recipe,
                        str(split_name),
                    )
                )
            if absolute_root in seen_roots:
                issues.append(
                    RegistryIssue(
                        "duplicate_output_root",
                        f"output_root is also declared by {_relative_or_absolute(seen_roots[absolute_root], repo_root)}",
                        relative_recipe,
                        str(split_name),
                    )
                )
            else:
                seen_roots[absolute_root] = recipe_path

            catalog_name = absolute_root.name
            expected_catalog_name = _expected_catalog_name(dataset_id, str(split_name))
            if catalog_name != expected_catalog_name:
                issues.append(
                    RegistryIssue(
                        "output_name_mismatch",
                        f"{split_name} output root must end with '{expected_catalog_name}', got '{catalog_name}'",
                        relative_recipe,
                        str(split_name),
                    )
                )
            if catalog_name in seen_catalog_names:
                issues.append(
                    RegistryIssue(
                        "duplicate_catalog_name",
                        f"catalog name '{catalog_name}' is also declared by {_relative_or_absolute(seen_catalog_names[catalog_name], repo_root)}",
                        relative_recipe,
                        str(split_name),
                    )
                )
            else:
                seen_catalog_names[catalog_name] = recipe_path

            expected_episodes = _expected_episodes(split_spec)
            entries.append(
                _build_entry(
                    repo_root=repo_root,
                    dataset_id=dataset_id,
                    split_name=str(split_name),
                    split_spec=split_spec,
                    recipe_path=recipe_path,
                    output_root=output_root,
                    absolute_root=absolute_root,
                    expected_episodes=expected_episodes,
                    inspect_artifacts=inspect_artifacts,
                )
            )

    return DatasetRegistry(
        repo_root=str(repo_root),
        recipe_dir=str(recipe_dir),
        dataset_root=str(dataset_root),
        entries=tuple(entries),
        issues=tuple(issues),
    )


def registered_dataset_roots(repo_root: Path, *, existing_only: bool = True) -> dict[str, Path]:
    registry = scan_dataset_registry(repo_root, inspect_artifacts=False)
    if not registry.valid:
        raise DatasetRegistryError(_format_issues(registry.issues))
    roots: dict[str, Path] = {}
    for entry in registry.entries:
        root = Path(entry.absolute_root)
        if existing_only and not root.exists():
            continue
        roots[entry.catalog_name] = root
    return roots


def validate_registered_recipe(repo_root: Path, recipe_path: Path) -> DatasetRegistry:
    repo_root = repo_root.resolve()
    resolved = recipe_path if recipe_path.is_absolute() else repo_root / recipe_path
    resolved = resolved.resolve()
    registry = scan_dataset_registry(repo_root, inspect_artifacts=False)
    errors = list(registry.issues)
    registered_paths = {(repo_root / entry.recipe_path).resolve() for entry in registry.entries}
    if resolved not in registered_paths:
        errors.append(
            RegistryIssue(
                "recipe_not_registered",
                f"recipe is not a valid member of {DATASET_RECIPE_DIR}",
                _relative_or_absolute(resolved, repo_root),
            )
        )
    if errors:
        raise DatasetRegistryError(_format_issues(errors))
    return registry


def require_recipe_training_ready(
    repo_root: Path,
    recipe_path: Path,
    *,
    splits: Iterable[str] | None = None,
) -> DatasetRegistry:
    repo_root = repo_root.resolve()
    resolved = recipe_path if recipe_path.is_absolute() else repo_root / recipe_path
    resolved = resolved.resolve()
    registry = scan_dataset_registry(repo_root, inspect_artifacts=True)
    selected = set(splits or ())
    entries = [
        entry
        for entry in registry.entries
        if (repo_root / entry.recipe_path).resolve() == resolved
        and (not selected or entry.split in selected)
    ]
    errors = list(registry.issues)
    if not entries:
        errors.append(RegistryIssue("no_selected_splits", "no selected recipe splits were found", str(recipe_path)))
    for entry in entries:
        for message in entry.readiness_errors:
            errors.append(RegistryIssue("not_training_ready", message, entry.recipe_path, entry.split))
    if errors:
        raise DatasetRegistryError(_format_issues(errors))
    return DatasetRegistry(
        repo_root=registry.repo_root,
        recipe_dir=registry.recipe_dir,
        dataset_root=registry.dataset_root,
        entries=tuple(entries),
        issues=(),
    )


def _build_entry(
    *,
    repo_root: Path,
    dataset_id: str,
    split_name: str,
    split_spec: dict[str, Any],
    recipe_path: Path,
    output_root: Path,
    absolute_root: Path,
    expected_episodes: int | None,
    inspect_artifacts: bool,
) -> DatasetRegistryEntry:
    status = "missing"
    episodes: int | None = None
    frames: int | None = None
    fps: int | float | None = None
    size_bytes: int | None = None
    audit_status: str | None = None
    grid_sidecar: str | None = None
    closed_loop_start: str | None = None
    readiness_errors: list[str] = []
    info: dict[str, Any] = {}

    if absolute_root.exists():
        info_path = absolute_root / "meta" / "info.json"
        data_files = list((absolute_root / "data").glob("**/*.parquet"))
        status = "available" if info_path.exists() and data_files else "incomplete"
        if inspect_artifacts:
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    readiness_errors.append("meta/info.json is not valid JSON")
            episodes = _optional_int(info.get("total_episodes"))
            frames = _optional_int(info.get("total_frames"))
            fps = info.get("fps")
            size_bytes = _dir_size(absolute_root)
            audit_status = _read_status(absolute_root / "so101_lerobot_audit.json")
            sidecars = sorted((absolute_root / "meta" / "camera_grid_bins").glob("*.parquet"))
            if sidecars:
                grid_sidecar = _relative_or_absolute(sidecars[0], repo_root)
            closed_loop = split_spec.get("closed_loop")
            if isinstance(closed_loop, dict) and closed_loop.get("output"):
                candidate = absolute_root / str(closed_loop["output"])
                if candidate.exists():
                    closed_loop_start = _relative_or_absolute(candidate, repo_root)
            readiness_errors.extend(
                _training_readiness_errors(
                    root=absolute_root,
                    info=info,
                    status=status,
                    split_name=split_name,
                    split_spec=split_spec,
                    expected_episodes=expected_episodes,
                    episodes=episodes,
                    audit_status=audit_status,
                    grid_sidecar=grid_sidecar,
                    closed_loop_start=closed_loop_start,
                )
            )

    if inspect_artifacts and status == "missing":
        readiness_errors.append(f"dataset root does not exist: {_relative_or_absolute(absolute_root, repo_root)}")
    elif inspect_artifacts and status == "incomplete":
        readiness_errors.append("dataset root is missing meta/info.json or parquet data")

    return DatasetRegistryEntry(
        dataset_id=dataset_id,
        split=split_name,
        catalog_name=absolute_root.name,
        recipe_path=_relative_or_absolute(recipe_path, repo_root),
        output_root=str(output_root),
        absolute_root=str(absolute_root),
        repo_id=str(split_spec["repo_id"]) if split_spec.get("repo_id") else None,
        expected_episodes=expected_episodes,
        status=status,
        episodes=episodes,
        frames=frames,
        fps=fps,
        size_bytes=size_bytes,
        audit_status=audit_status,
        grid_sidecar=grid_sidecar,
        closed_loop_start=closed_loop_start,
        training_ready=inspect_artifacts and not readiness_errors,
        readiness_errors=tuple(readiness_errors),
    )


def _training_readiness_errors(
    *,
    root: Path,
    info: dict[str, Any],
    status: str,
    split_name: str,
    split_spec: dict[str, Any],
    expected_episodes: int | None,
    episodes: int | None,
    audit_status: str | None,
    grid_sidecar: str | None,
    closed_loop_start: str | None,
) -> list[str]:
    errors: list[str] = []
    if status != "available":
        return errors
    for relative in (
        "meta/stats.json",
        "meta/tasks.parquet",
        "so101_lerobot_export_report.json",
        "so101_lerobot_merge_report.json",
        "so101_lerobot_audit.json",
    ):
        if not (root / relative).exists():
            errors.append(f"missing required artifact: {relative}")
    if audit_status != "passed":
        errors.append(f"dataset audit status must be passed, got {audit_status or 'missing'}")
    if expected_episodes is not None and episodes != expected_episodes:
        errors.append(f"episode count mismatch: expected {expected_episodes}, got {episodes}")
    features = info.get("features") if isinstance(info.get("features"), dict) else {}
    for key in REQUIRED_CAMERA_KEYS:
        feature = features.get(key) if isinstance(features, dict) else None
        shape = feature.get("shape") if isinstance(feature, dict) else None
        if shape not in ([256, 256, 3], [3, 256, 256]):
            errors.append(f"{key} must be 256x256 RGB, got {shape}")
    for key in REQUIRED_JOINT_KEYS:
        feature = features.get(key) if isinstance(features, dict) else None
        shape = feature.get("shape") if isinstance(feature, dict) else None
        if shape != [6]:
            errors.append(f"{key} must have shape [6], got {shape}")
    if split_name == "train" and grid_sidecar is None:
        errors.append("train split is missing camera-grid sidecar")
    if isinstance(split_spec.get("closed_loop"), dict) and closed_loop_start is None:
        errors.append("validation split declares closed_loop but its start report is missing")
    return errors


def _expected_episodes(split_spec: dict[str, Any]) -> int | None:
    bins = split_spec.get("bins")
    if not isinstance(bins, list) or not bins:
        return _optional_int(split_spec.get("expected_episodes"))
    counts = [row.get("episodes") for row in bins if isinstance(row, dict)]
    if not counts or any(not isinstance(value, int) for value in counts):
        return None
    return sum(counts)


def _expected_catalog_name(dataset_id: str, split_name: str) -> str:
    if split_name == "train":
        return dataset_id
    if split_name in {"validation", "val"}:
        return f"{dataset_id}_validation"
    return f"{dataset_id}_{split_name}"


def _read_status(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid"
    value = payload.get("status") or payload.get("audit_status")
    return str(value) if value is not None else None


def _dir_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _format_issues(issues: Iterable[RegistryIssue]) -> str:
    return "\n".join(
        f"[{issue.code}] {issue.recipe or '-'}:{issue.split or '-'} {issue.message}"
        for issue in issues
    )


def _training_dataset_row(entry: DatasetRegistryEntry) -> dict[str, Any]:
    return {
        "name": entry.catalog_name,
        "repo_id": entry.repo_id,
        "root": entry.output_root,
        "grid_bin_sidecar": entry.grid_sidecar,
        "expected_episodes": entry.episodes,
        "expected_frames": entry.frames,
        "closed_loop_start": entry.closed_loop_start,
    }
