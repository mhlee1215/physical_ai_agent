from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    required: bool = True


@dataclass(frozen=True)
class CheckpointReport:
    checkpoint: str
    status: str
    duration_s: float
    python: str
    platform: str
    config: str
    results: list[CheckResult]


OPTIONAL_IMPORTS = {
    "mujoco": "MuJoCo simulation runtime",
    "robosuite": "robosuite environment layer used by LIBERO",
    "libero": "LIBERO benchmark package",
    "lerobot": "LeRobot policy/evaluation integration",
}


def _import_check(module_name: str, detail: str, required: bool) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - smoke reports should include import failures.
        return CheckResult(
            name=f"import:{module_name}",
            status="failed" if required else "skipped",
            detail=f"{detail} unavailable: {exc.__class__.__name__}: {exc}",
            required=required,
        )

    version = getattr(module, "__version__", "unknown")
    return CheckResult(
        name=f"import:{module_name}",
        status="passed",
        detail=f"{detail} import OK, version={version}",
        required=required,
    )


def _load_yaml_config(path: Path) -> CheckResult:
    try:
        text = path.read_text(encoding="utf-8")
        data = _parse_yaml_like_config(text)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="config:libero",
            status="failed",
            detail=f"could not load {path}: {exc.__class__.__name__}: {exc}",
        )

    required_keys = {"name", "suite", "render", "episode"}
    missing = sorted(required_keys.difference(data or {}))
    if missing:
        return CheckResult(
            name="config:libero",
            status="failed",
            detail=f"{path} missing required keys: {', '.join(missing)}",
        )

    return CheckResult(
        name="config:libero",
        status="passed",
        detail=f"loaded {path}",
    )


def _parse_yaml_like_config(text: str) -> dict[str, Any]:
    """Load simple YAML configs without requiring PyYAML for scaffold smoke tests."""
    try:
        yaml = importlib.import_module("yaml")
    except Exception:  # noqa: BLE001
        data: dict[str, Any] = {}
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line or line.startswith(" "):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip() or {}
        return data

    loaded = yaml.safe_load(text)
    return loaded if isinstance(loaded, dict) else {}


def _libero_env_probe(enabled: bool, required: bool) -> CheckResult:
    if not enabled:
        return CheckResult(
            name="libero:env_probe",
            status="skipped",
            detail="use --probe-libero-env to attempt an environment reset",
            required=False,
        )

    try:
        benchmark_module = importlib.import_module("libero.libero.benchmark")
        benchmark_factory = getattr(benchmark_module, "get_benchmark")
        benchmark = benchmark_factory("libero_spatial")()
        task = benchmark.get_task(0)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="libero:env_probe",
            status="failed" if required else "skipped",
            detail=f"could not create LIBERO benchmark/task: {exc.__class__.__name__}: {exc}",
            required=required,
        )

    return CheckResult(
        name="libero:env_probe",
        status="passed",
        detail=f"created LIBERO task: {getattr(task, 'name', 'unknown')}",
    )


def run_checkpoint(
    config_path: Path,
    strict_sim_deps: bool,
    probe_libero_env: bool,
) -> CheckpointReport:
    start = perf_counter()
    results: list[CheckResult] = [
        CheckResult(
            name="python",
            status="passed",
            detail=f"running Python {platform.python_version()}",
        ),
        _load_yaml_config(config_path),
    ]

    results.extend(
        _import_check(module_name, detail, required=strict_sim_deps)
        for module_name, detail in OPTIONAL_IMPORTS.items()
    )
    results.append(_libero_env_probe(probe_libero_env, required=strict_sim_deps))

    required_failures = [
        result
        for result in results
        if result.required and result.status not in {"passed", "skipped"}
    ]
    skipped_required = [
        result for result in results if result.required and result.status == "skipped"
    ]
    if required_failures:
        status = "failed"
    elif skipped_required:
        status = "blocked"
    else:
        status = "passed"

    return CheckpointReport(
        checkpoint="checkpoint_01_libero_smoke",
        status=status,
        duration_s=round(perf_counter() - start, 4),
        python=platform.python_version(),
        platform=platform.platform(),
        config=str(config_path),
        results=results,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Checkpoint 01: verify the local LIBERO/LeRobot smoke-test path.",
    )
    parser.add_argument(
        "--config",
        default="configs/sim/libero.yaml",
        help="Path to the LIBERO simulation config.",
    )
    parser.add_argument(
        "--strict-sim-deps",
        action="store_true",
        help="Fail if MuJoCo, robosuite, LIBERO, or LeRobot are not importable.",
    )
    parser.add_argument(
        "--probe-libero-env",
        action="store_true",
        help="Attempt to instantiate a LIBERO benchmark task.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    return parser


def _print_human(report: CheckpointReport) -> None:
    print(f"{report.checkpoint}: {report.status}")
    print(f"python={report.python}")
    print(f"platform={report.platform}")
    print(f"config={report.config}")
    for result in report.results:
        marker = {
            "passed": "PASS",
            "failed": "FAIL",
            "skipped": "SKIP",
            "blocked": "BLOCK",
        }.get(result.status, result.status.upper())
        required = "required" if result.required else "optional"
        print(f"- {marker} {result.name} ({required}): {result.detail}")


def main() -> None:
    args = build_parser().parse_args()
    report = run_checkpoint(
        config_path=Path(args.config),
        strict_sim_deps=args.strict_sim_deps,
        probe_libero_env=args.probe_libero_env,
    )

    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        _print_human(report)

    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
