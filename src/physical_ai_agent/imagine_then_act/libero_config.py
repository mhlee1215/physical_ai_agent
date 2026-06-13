from __future__ import annotations

import argparse
import json
import os
import sysconfig
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LiberoConfigResult:
    config_dir: str
    config_path: str
    assets: str
    benchmark_root: str
    bddl_files: str
    datasets: str
    init_states: str
    wrote_config: bool


def resolve_config_dir(raw_path: str | None = None) -> Path:
    raw = raw_path or os.environ.get("LIBERO_CONFIG_PATH") or os.environ.get("LIBERO_CONFIG_DIR")
    if not raw:
        raw = str(Path.home() / ".libero")
    path = Path(raw).expanduser()
    return path.parent if path.name == "config.yaml" else path


def resolve_libero_package_dir(raw_path: str | None = None) -> Path:
    if raw_path:
        return Path(raw_path).expanduser()
    env_path = os.environ.get("LIBERO_PACKAGE_DIR")
    if env_path:
        return Path(env_path).expanduser()
    site_packages = Path(sysconfig.get_paths()["purelib"])
    return site_packages / "libero" / "libero"


def config_text(*, assets: Path, libero_package_dir: Path) -> str:
    datasets = libero_package_dir.parent / "datasets"
    lines = [
        f"benchmark_root: {libero_package_dir}",
        f"assets: {assets}",
        f"bddl_files: {libero_package_dir / 'bddl_files'}",
        f"datasets: {datasets}",
        f"init_states: {libero_package_dir / 'init_files'}",
    ]
    return "\n".join(lines) + "\n"


def parse_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def ensure_noninteractive_libero_config(
    *,
    config_dir: str | None = None,
    assets_dir: str | None = None,
    libero_package_dir: str | None = None,
) -> LiberoConfigResult:
    resolved_config_dir = resolve_config_dir(config_dir)
    resolved_assets = Path(
        assets_dir
        or os.environ.get("LIBERO_ASSETS_DIR")
        or os.environ.get("LIBERO_DATASET_DIR")
        or "/workspace/physical-ai/libero_assets"
    ).expanduser()
    resolved_package = resolve_libero_package_dir(libero_package_dir)
    if not resolved_package.is_dir():
        raise FileNotFoundError(f"LIBERO package directory not found: {resolved_package}")

    resolved_config_dir.mkdir(parents=True, exist_ok=True)
    resolved_assets.mkdir(parents=True, exist_ok=True)
    config_path = resolved_config_dir / "config.yaml"
    desired = config_text(assets=resolved_assets, libero_package_dir=resolved_package)
    wrote_config = not config_path.exists() or config_path.read_text() != desired
    if wrote_config:
        config_path.write_text(desired)

    values = parse_simple_yaml(config_path)
    required = {"benchmark_root", "assets", "bddl_files", "datasets", "init_states"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"LIBERO config missing required keys: {', '.join(missing)}")

    os.environ["LIBERO_CONFIG_PATH"] = str(resolved_config_dir)
    os.environ["LIBERO_CONFIG_DIR"] = str(resolved_config_dir)
    os.environ["LIBERO_ASSETS_DIR"] = str(resolved_assets)
    return LiberoConfigResult(
        config_dir=str(resolved_config_dir),
        config_path=str(config_path),
        assets=values["assets"],
        benchmark_root=values["benchmark_root"],
        bddl_files=values["bddl_files"],
        datasets=values["datasets"],
        init_states=values["init_states"],
        wrote_config=wrote_config,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a non-interactive LIBERO config.yaml for headless RunPod probes.")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--assets-dir", default=None)
    parser.add_argument("--libero-package-dir", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = ensure_noninteractive_libero_config(
        config_dir=args.config_dir,
        assets_dir=args.assets_dir,
        libero_package_dir=args.libero_package_dir,
    )
    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
