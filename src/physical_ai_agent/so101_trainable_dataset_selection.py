from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DatasetRole = Literal["training", "validation", "loop_test"]
DATASET_ROLE_SELECTION_PATH = Path(
    "_workspace/so101_training/dataset_role_selection.json"
)
LEGACY_TRAINABLE_DATASET_SELECTION_PATH = Path(
    "_workspace/so101_training/trainable_dataset_selection.json"
)
_ROLE_ORDER: dict[str, int] = {
    "training": 0,
    "validation": 1,
    "loop_test": 2,
}
_SELECTION_LOCK = threading.Lock()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetRoleSelectionEntry(_StrictModel):
    role: DatasetRole
    catalog_name: str = Field(min_length=1)
    root: str = Field(min_length=1)
    repo_id: str = Field(min_length=1)
    marked_at: str = Field(min_length=1)
    expected_episodes: int | None = Field(default=None, gt=0)
    expected_frames: int | None = Field(default=None, gt=0)
    grid_bin_sidecar: str | None = None
    loop_test_case: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_role_payload(self) -> DatasetRoleSelectionEntry:
        if self.role == "loop_test" and not self.loop_test_case:
            raise ValueError("loop-test selections require a complete loop_test_case")
        if self.role != "loop_test" and self.loop_test_case is not None:
            raise ValueError("loop_test_case is only valid for loop-test selections")
        return self


class DatasetRoleSelection(_StrictModel):
    schema_version: Literal[2] = 2
    updated_at: str = Field(min_length=1)
    datasets: list[DatasetRoleSelectionEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_assignments(self) -> DatasetRoleSelection:
        keys = [(entry.role, entry.catalog_name) for entry in self.datasets]
        if len(keys) != len(set(keys)):
            raise ValueError("dataset role selection contains duplicate assignments")

        dataset_roots = [
            (entry.role, entry.root)
            for entry in self.datasets
            if entry.role != "loop_test"
        ]
        if len(dataset_roots) != len(set(dataset_roots)):
            raise ValueError("training/validation roles contain duplicate dataset roots")
        return self


def dataset_role_selection_path(repo_root: Path) -> Path:
    return (repo_root.resolve() / DATASET_ROLE_SELECTION_PATH).resolve()


def trainable_dataset_selection_path(repo_root: Path) -> Path:
    """Compatibility alias for callers written before role-aware selection."""

    return dataset_role_selection_path(repo_root)


def load_dataset_role_selection(repo_root: Path) -> DatasetRoleSelection:
    repo_root = repo_root.resolve()
    path = dataset_role_selection_path(repo_root)
    if path.is_file():
        return DatasetRoleSelection.model_validate_json(path.read_text(encoding="utf-8"))

    legacy_path = (repo_root / LEGACY_TRAINABLE_DATASET_SELECTION_PATH).resolve()
    if not legacy_path.is_file():
        return DatasetRoleSelection(updated_at=_utc_now())
    return _migrate_legacy_selection(
        json.loads(legacy_path.read_text(encoding="utf-8"))
    )


def load_trainable_dataset_selection(repo_root: Path) -> DatasetRoleSelection:
    """Compatibility alias returning the role-aware selection."""

    return load_dataset_role_selection(repo_root)


def update_dataset_role_selection(
    repo_root: Path,
    *,
    additions: Iterable[dict[str, Any]] = (),
    removals: Iterable[dict[str, Any]] = (),
    remove_roots: Iterable[str | Path] = (),
) -> DatasetRoleSelection:
    repo_root = repo_root.resolve()
    with _SELECTION_LOCK:
        current = load_dataset_role_selection(repo_root)
        by_key = {
            (entry.role, entry.catalog_name): entry
            for entry in current.datasets
        }

        roots_to_remove = {
            _canonical_root(repo_root, root)
            for root in remove_roots
        }
        if roots_to_remove:
            by_key = {
                key: entry
                for key, entry in by_key.items()
                if _canonical_root(repo_root, entry.root) not in roots_to_remove
            }

        for raw_removal in removals:
            role = _validated_role(raw_removal.get("role"))
            catalog_name = str(raw_removal.get("catalog_name") or "").strip()
            if not catalog_name:
                raise ValueError("dataset role removal requires catalog_name")
            by_key.pop((role, catalog_name), None)

        now = _utc_now()
        for raw_entry in additions:
            payload = copy.deepcopy(dict(raw_entry))
            role = _validated_role(payload.get("role", "training"))
            catalog_name = str(payload.get("catalog_name") or "").strip()
            if not catalog_name:
                raise ValueError("dataset role addition requires catalog_name")
            canonical_root = _canonical_root(repo_root, str(payload["root"]))
            key = (role, catalog_name)
            if role != "loop_test":
                by_key = {
                    existing_key: entry
                    for existing_key, entry in by_key.items()
                    if not (
                        entry.role == role
                        and _canonical_root(repo_root, entry.root) == canonical_root
                        and existing_key != key
                    )
                }
            previous = by_key.get(key)
            payload["role"] = role
            payload["catalog_name"] = catalog_name
            payload["root"] = _portable_path(repo_root, canonical_root)
            payload["grid_bin_sidecar"] = _portable_optional_path(
                repo_root,
                payload.get("grid_bin_sidecar"),
            )
            payload["loop_test_case"] = _portable_loop_test_case(
                repo_root,
                payload.get("loop_test_case"),
            )
            payload["marked_at"] = previous.marked_at if previous else now
            by_key[key] = DatasetRoleSelectionEntry.model_validate(payload)

        selection = DatasetRoleSelection(
            updated_at=now,
            datasets=sorted(
                by_key.values(),
                key=lambda entry: (
                    _ROLE_ORDER[entry.role],
                    entry.catalog_name.casefold(),
                ),
            ),
        )
        _write_selection(repo_root, selection)
        return selection


def update_trainable_dataset_selection(
    repo_root: Path,
    *,
    additions: Iterable[dict[str, Any]] = (),
    remove_roots: Iterable[str | Path] = (),
) -> DatasetRoleSelection:
    """Compatibility wrapper for the original training-only API."""

    training_additions = []
    for raw_entry in additions:
        payload = dict(raw_entry)
        payload.setdefault("role", "training")
        training_additions.append(payload)
    return update_dataset_role_selection(
        repo_root,
        additions=training_additions,
        remove_roots=remove_roots,
    )


def selected_dataset_roots(repo_root: Path, role: DatasetRole) -> set[Path]:
    repo_root = repo_root.resolve()
    return {
        _canonical_root(repo_root, entry.root)
        for entry in load_dataset_role_selection(repo_root).datasets
        if entry.role == role
    }


def trainable_dataset_roots(repo_root: Path) -> set[Path]:
    return selected_dataset_roots(repo_root, "training")


def selected_catalog_names(repo_root: Path, role: DatasetRole) -> set[str]:
    return {
        entry.catalog_name
        for entry in load_dataset_role_selection(repo_root).datasets
        if entry.role == role
    }


def dataset_role_counts(repo_root: Path) -> dict[str, int]:
    counts = {role: 0 for role in _ROLE_ORDER}
    for entry in load_dataset_role_selection(repo_root).datasets:
        counts[entry.role] += 1
    return counts


def dataset_entries_from_selection(
    repo_root: Path,
    role: Literal["training", "validation"],
) -> list[dict[str, Any]]:
    repo_root = repo_root.resolve()
    selection = load_dataset_role_selection(repo_root)
    selected = [entry for entry in selection.datasets if entry.role == role]
    if not selected:
        raise ValueError(f"no datasets are marked for the {role} set")

    rows: list[dict[str, Any]] = []
    for entry in selected:
        root = _canonical_root(repo_root, entry.root)
        if not root.is_dir():
            raise ValueError(
                f"marked {role} dataset no longer exists: "
                f"{entry.catalog_name} ({root})"
            )
        row: dict[str, Any] = {
            "name": entry.catalog_name,
            "repo_id": entry.repo_id,
            "root": str(root),
        }
        if entry.expected_episodes is not None:
            row["expected_episodes"] = entry.expected_episodes
        if entry.expected_frames is not None:
            row["expected_frames"] = entry.expected_frames
        if entry.grid_bin_sidecar:
            row["grid_bin_sidecar"] = str(
                _canonical_root(repo_root, entry.grid_bin_sidecar)
            )
        rows.append(row)
    return rows


def training_dataset_entries_from_selection(repo_root: Path) -> list[dict[str, Any]]:
    return dataset_entries_from_selection(repo_root, "training")


def validation_dataset_entries_from_selection(repo_root: Path) -> list[dict[str, Any]]:
    return dataset_entries_from_selection(repo_root, "validation")


def loop_test_cases_from_selection(repo_root: Path) -> list[dict[str, Any]]:
    repo_root = repo_root.resolve()
    selection = load_dataset_role_selection(repo_root)
    selected = [entry for entry in selection.datasets if entry.role == "loop_test"]
    if not selected:
        raise ValueError("no datasets are marked for the loop-test set")

    test_cases = []
    for entry in selected:
        root = _canonical_root(repo_root, entry.root)
        if not root.is_dir():
            raise ValueError(
                f"marked loop-test dataset no longer exists: "
                f"{entry.catalog_name} ({root})"
            )
        test_case = copy.deepcopy(entry.loop_test_case or {})
        report = test_case.get("start_report_path")
        if report:
            report_path = _canonical_root(repo_root, str(report))
            if not report_path.is_file():
                raise ValueError(
                    f"marked loop-test start report no longer exists: {report_path}"
                )
            test_case["start_report_path"] = str(report_path)
        start_dataset = test_case.get("start_dataset")
        if isinstance(start_dataset, dict) and start_dataset.get("root"):
            start_dataset["root"] = str(
                _canonical_root(repo_root, str(start_dataset["root"]))
            )
        test_cases.append(test_case)
    return test_cases


def _migrate_legacy_selection(payload: dict[str, Any]) -> DatasetRoleSelection:
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("legacy trainable dataset selection has invalid datasets")
    migrated = []
    for raw_entry in datasets:
        entry = dict(raw_entry)
        entry["role"] = "training"
        migrated.append(DatasetRoleSelectionEntry.model_validate(entry))
    return DatasetRoleSelection(
        updated_at=str(payload.get("updated_at") or _utc_now()),
        datasets=migrated,
    )


def _write_selection(repo_root: Path, selection: DatasetRoleSelection) -> None:
    path = dataset_role_selection_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(selection.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _portable_loop_test_case(
    repo_root: Path,
    value: Any,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("loop_test_case must be an object")
    test_case = copy.deepcopy(value)
    if test_case.get("start_report_path"):
        test_case["start_report_path"] = _portable_optional_path(
            repo_root,
            test_case["start_report_path"],
        )
    start_dataset = test_case.get("start_dataset")
    if isinstance(start_dataset, dict) and start_dataset.get("root"):
        start_dataset["root"] = _portable_path(
            repo_root,
            _canonical_root(repo_root, str(start_dataset["root"])),
        )
    return test_case


def _validated_role(value: Any) -> DatasetRole:
    role = str(value or "").strip()
    if role not in _ROLE_ORDER:
        raise ValueError(
            "dataset role must be one of: training, validation, loop_test"
        )
    return role  # type: ignore[return-value]


def _canonical_root(repo_root: Path, root: str | Path) -> Path:
    path = Path(root)
    return (path if path.is_absolute() else repo_root / path).resolve()


def _portable_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _portable_optional_path(repo_root: Path, value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return _portable_path(repo_root, _canonical_root(repo_root, str(value)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
