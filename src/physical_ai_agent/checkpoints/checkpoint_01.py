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
    strict_local_sim: bool
    strict_sim_deps: bool
    probe_mujoco: bool
    probe_libero_env: bool
    results: list[CheckResult]


OPTIONAL_IMPORTS = {
    "mujoco": "MuJoCo simulation runtime",
    "robosuite": "robosuite environment layer used by LIBERO",
    "libero": "LIBERO benchmark package",
    "lerobot": "LeRobot policy/evaluation integration",
}

MIN_PYTHON = (3, 11)


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


def _platform_check(strict_sim_deps: bool) -> CheckResult:
    is_linux = sys.platform.startswith("linux")
    if is_linux:
        return CheckResult(
            name="platform:libero",
            status="passed",
            detail=f"LIBERO strict gate running on supported platform: {sys.platform}",
            required=strict_sim_deps,
        )

    return CheckResult(
        name="platform:libero",
        status="failed" if strict_sim_deps else "skipped",
        detail=(
            "LeRobot's LIBERO integration currently documents Linux as required; "
            f"current platform is {sys.platform}"
        ),
        required=strict_sim_deps,
    )


def _mujoco_probe(enabled: bool, required: bool) -> CheckResult:
    if not enabled:
        return CheckResult(
            name="mujoco:step_probe",
            status="skipped",
            detail="use --probe-mujoco to instantiate and step a tiny MuJoCo model",
            required=False,
        )

    try:
        mujoco = importlib.import_module("mujoco")
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="box" pos="0 0 0.1">
                  <freejoint/>
                  <geom type="box" size="0.05 0.05 0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="mujoco:step_probe",
            status="failed" if required else "skipped",
            detail=f"could not step tiny MuJoCo model: {exc.__class__.__name__}: {exc}",
            required=required,
        )

    return CheckResult(
        name="mujoco:step_probe",
        status="passed",
        detail=f"stepped tiny MuJoCo model: nq={model.nq}, nv={model.nv}",
        required=required,
    )


def run_checkpoint(
    config_path: Path,
    strict_local_sim: bool,
    strict_sim_deps: bool,
    probe_mujoco: bool,
    probe_libero_env: bool,
) -> CheckpointReport:
    start = perf_counter()
    python_status = "passed" if sys.version_info >= MIN_PYTHON else "failed"
    results: list[CheckResult] = [
        CheckResult(
            name="python",
            status=python_status,
            detail=(
                f"running Python {platform.python_version()}, "
                f"requires >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
            ),
        ),
        _load_yaml_config(config_path),
    ]

    results.extend(
        _import_check(
            module_name,
            detail,
            required=strict_sim_deps or (strict_local_sim and module_name == "mujoco"),
        )
        for module_name, detail in OPTIONAL_IMPORTS.items()
    )
    results.append(_mujoco_probe(probe_mujoco, required=strict_local_sim))
    results.append(_platform_check(strict_sim_deps))
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
        strict_local_sim=strict_local_sim,
        strict_sim_deps=strict_sim_deps,
        probe_mujoco=probe_mujoco,
        probe_libero_env=probe_libero_env,
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
        "--strict-local-sim",
        action="store_true",
        help="Fail if the Mac-local MuJoCo smoke path is not runnable.",
    )
    parser.add_argument(
        "--strict-sim-deps",
        action="store_true",
        help="Fail if the full LIBERO/LeRobot simulation dependency path is not runnable.",
    )
    parser.add_argument(
        "--probe-mujoco",
        action="store_true",
        help="Instantiate and step a tiny MuJoCo model.",
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
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON checkpoint report.",
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
        strict_local_sim=args.strict_local_sim,
        strict_sim_deps=args.strict_sim_deps,
        probe_mujoco=args.probe_mujoco,
        probe_libero_env=args.probe_libero_env,
    )

    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        _print_human(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")

    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
