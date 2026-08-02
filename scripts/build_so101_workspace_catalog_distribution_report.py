#!/usr/bin/env python3
"""Audit a continuous SO101 workspace catalog before teacher export."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from physical_ai_agent.so101_workspace_spawn_catalog import (
    load_workspace_spawn_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_workspace_catalog_distribution_report(
        catalog_path=args.catalog,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["gate"]["status"] != "passed":
        raise SystemExit(2)


def build_workspace_catalog_distribution_report(
    *,
    catalog_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    catalog = load_workspace_spawn_catalog(catalog_path)
    distribution = catalog.continuous_distribution
    if distribution is None or not catalog.enforce_cell_local_quota:
        raise ValueError("catalog is not a continuous cell-local quota catalog")

    primary = [row for row in catalog.candidates if row.stage == "primary"]
    source_counts = Counter(row.source_cell_id for row in primary)
    expected_counts = {
        row.cell_id: int(row.primary_target_count) for row in catalog.cell_quotas
    }
    points = [list(row.world_xy_m) for row in primary]
    all_points = [list(row.world_xy_m) for row in catalog.candidates]
    nearest = _nearest_neighbor_stats(points)
    pool_nearest = _nearest_neighbor_stats(all_points)
    pool_minimum_spacing = (
        distribution.candidate_pool_minimum_spacing_m
        or distribution.minimum_spacing_m
    )
    boundary_hits = sum(
        not (
            distribution.radius_min_m < row.radius_from_base_m < distribution.radius_max_m
            and distribution.angle_min_deg
            < row.angle_from_base_deg
            < distribution.angle_max_deg
        )
        for row in catalog.candidates
    )

    radial: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "area_m2": 0.0}
    )
    angular_counts: Counter[int] = Counter()
    configured_yaw_counts: Counter[int] = Counter()
    for quota in catalog.cell_quotas:
        radial[quota.radial_index]["count"] += quota.primary_target_count
        radial[quota.radial_index]["area_m2"] += quota.area_m2
        angular_counts[quota.angular_index] += quota.primary_target_count
        if quota.yaw_index is not None:
            configured_yaw_counts[quota.yaw_index] += quota.primary_target_count
    radial_rows = []
    for radial_index, values in sorted(radial.items()):
        area = float(values["area_m2"])
        count = int(values["count"])
        radial_rows.append(
            {
                "radial_index": radial_index,
                "count": count,
                "area_m2": area,
                "samples_per_m2": count / area,
            }
        )
    densities = [float(row["samples_per_m2"]) for row in radial_rows]
    density_ratio = densities[-1] / densities[0]
    angular_values = list(angular_counts.values())
    angular_cv = statistics.pstdev(angular_values) / statistics.fmean(
        angular_values
    )
    yaw_distribution = distribution.object_yaw
    yaw_bins = yaw_distribution.strata if yaw_distribution is not None else 4
    yaw_period_deg = (
        yaw_distribution.periodicity_deg
        if yaw_distribution is not None
        else 90.0
    )
    absolute_yaw_counts = _periodic_yaw_counts(
        [float(row.object_yaw_deg) for row in primary],
        bins=yaw_bins,
        period_deg=yaw_period_deg,
    )
    absolute_yaw_values = [
        absolute_yaw_counts[index] for index in range(yaw_bins)
    ]
    absolute_yaw_coverage = (
        sum(value > 0 for value in absolute_yaw_values) / yaw_bins
    )
    absolute_yaw_mean = statistics.fmean(absolute_yaw_values)
    yaw_cv = (
        statistics.pstdev(absolute_yaw_values) / absolute_yaw_mean
        if absolute_yaw_mean > 0.0
        else None
    )
    relative_yaw_counts: Counter[int] = Counter()
    relative_yaw_cv: float | None = None
    relative_yaw_coverage: float | None = None
    if primary:
        sampled_yaw_values = [
            (
                float(row.object_yaw_deg) - float(row.angle_from_base_deg)
                if yaw_distribution is not None
                and yaw_distribution.reference_frame == "robot_relative"
                else float(row.object_yaw_deg)
            )
            - (yaw_distribution.min_deg if yaw_distribution is not None else 0.0)
            for row in primary
        ]
        relative_yaw_counts = _periodic_yaw_counts(
            [
                float(row.object_yaw_deg) - float(row.angle_from_base_deg)
                for row in primary
            ],
            bins=yaw_bins,
            period_deg=yaw_period_deg,
        )
        relative_values = [
            relative_yaw_counts[index]
            for index in range(yaw_bins)
        ]
        relative_yaw_coverage = (
            sum(value > 0 for value in relative_values)
            / yaw_bins
        )
        relative_mean = statistics.fmean(relative_values)
        relative_yaw_cv = (
            statistics.pstdev(relative_values) / relative_mean
            if relative_mean > 0.0
            else None
        )
    yaw_boundary_hits = (
        0
        if yaw_distribution is None
        else sum(
            not (
                0.0
                < (sampled_yaw % yaw_distribution.periodicity_deg)
                < (yaw_distribution.max_deg - yaw_distribution.min_deg)
            )
            for sampled_yaw in sampled_yaw_values
        )
    )
    quota_tv = _count_total_variation(source_counts, expected_counts)
    density_nonincreasing = all(
        left >= right for left, right in zip(densities, densities[1:])
    )
    density_ratio_error = abs(
        density_ratio - distribution.far_to_near_area_density_ratio
    )
    gates = {
        "primary_count": {
            "passed": len(primary) == catalog.primary_target_count,
            "actual": len(primary),
            "expected": catalog.primary_target_count,
        },
        "cell_quota_exact": {
            "passed": quota_tv == 0.0,
            "total_variation": quota_tv,
        },
        "all_cells_populated": {
            "passed": len(source_counts) == catalog.source_cell_count,
            "actual": len(source_counts),
            "expected": catalog.source_cell_count,
        },
        "open_domain_no_boundary_clipping": {
            "passed": boundary_hits == 0,
            "boundary_hits": boundary_hits,
        },
        "minimum_spacing": {
            "passed": float(nearest["min_m"] or 0.0)
            >= distribution.minimum_spacing_m - 1e-12,
            "actual_m": nearest["min_m"],
            "minimum_m": distribution.minimum_spacing_m,
            "scope": "primary",
        },
        "candidate_pool_minimum_spacing": {
            "passed": float(pool_nearest["min_m"] or 0.0)
            >= pool_minimum_spacing - 1e-12,
            "actual_m": pool_nearest["min_m"],
            "minimum_m": pool_minimum_spacing,
            "scope": "primary_and_backup",
        },
        "radial_area_density_nonincreasing": {
            "passed": density_nonincreasing,
            "actual_far_to_near_ratio": density_ratio,
        },
        "radial_area_density_ratio": {
            "passed": density_ratio_error <= 0.05,
            "actual": density_ratio,
            "target": distribution.far_to_near_area_density_ratio,
            "absolute_error": density_ratio_error,
            "maximum_absolute_error": 0.05,
        },
        "angular_balance": {
            "passed": angular_cv <= 0.05,
            "count_cv": angular_cv,
            "maximum_cv": 0.05,
        },
    }
    gates.update(
        {
            "object_yaw_coverage": {
                "passed": absolute_yaw_coverage == 1.0,
                "coverage_ratio": absolute_yaw_coverage,
                "minimum_ratio": 1.0,
            },
            "object_yaw_balance": {
                "passed": yaw_cv is not None and yaw_cv <= 0.10,
                "count_cv": yaw_cv,
                "maximum_cv": 0.10,
            },
            "robot_relative_object_yaw_coverage": {
                "passed": relative_yaw_coverage == 1.0,
                "coverage_ratio": relative_yaw_coverage,
                "minimum_ratio": 1.0,
            },
            "robot_relative_object_yaw_balance": {
                "passed": relative_yaw_cv is not None
                and relative_yaw_cv
                <= distribution.max_robot_relative_yaw_count_cv,
                "count_cv": relative_yaw_cv,
                "maximum_cv": distribution.max_robot_relative_yaw_count_cv,
            },
        }
    )
    if yaw_distribution is not None:
        gates.update(
            {
                "independent_object_yaw_strata": {
                    "passed": len(configured_yaw_counts)
                    == yaw_distribution.strata,
                    "actual": len(configured_yaw_counts),
                    "expected": yaw_distribution.strata,
                },
                "object_yaw_open_domain_no_boundary_clipping": {
                    "passed": yaw_boundary_hits == 0,
                    "boundary_hits": yaw_boundary_hits,
                },
            }
        )
    failed = [name for name, check in gates.items() if not check["passed"]]
    report = {
        "format": "so101_workspace_catalog_distribution_report_v1",
        "catalog": str(catalog_path),
        "catalog_id": catalog.catalog_id,
        "summary": {
            "primary_candidates": len(primary),
            "backup_candidates": catalog.backup_count,
            "cells": catalog.source_cell_count,
            "cell_count_min": min(source_counts.values()),
            "cell_count_max": max(source_counts.values()),
            "angular_count_cv": angular_cv,
            "far_to_near_area_density_ratio": density_ratio,
            "nearest_neighbor_min_m": nearest["min_m"],
            "nearest_neighbor_median_m": nearest["median_m"],
            "boundary_hits": boundary_hits,
            "object_yaw_strata": yaw_bins,
            "object_yaw_count_cv": yaw_cv,
            "object_yaw_boundary_hits": yaw_boundary_hits,
            "robot_relative_object_yaw_coverage_ratio": relative_yaw_coverage,
            "robot_relative_object_yaw_count_cv": relative_yaw_cv,
        },
        "continuous_distribution": distribution.model_dump(mode="json"),
        "source_cell_counts": dict(sorted(source_counts.items())),
        "expected_source_cell_counts": dict(sorted(expected_counts.items())),
        "radial_area_density": radial_rows,
        "angular_counts": dict(sorted(angular_counts.items())),
        "configured_yaw_stratum_counts": dict(
            sorted(configured_yaw_counts.items())
        ),
        "object_yaw_counts": dict(sorted(absolute_yaw_counts.items())),
        "robot_relative_object_yaw_counts": dict(
            sorted(relative_yaw_counts.items())
        ),
        "nearest_neighbor_primary_m": nearest,
        "nearest_neighbor_all_candidates_m": pool_nearest,
        "gate": {
            "status": "passed" if not failed else "failed",
            "failed": failed,
            "checks": gates,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "distribution.json"
    md_path = output_dir / "distribution.md"
    html_path = output_dir / "distribution.html"
    markdown = _markdown(report)
    markdown_sha256 = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    report["artifacts"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
        "markdown_sha256": markdown_sha256,
    }
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(
        _html(report, points, markdown_sha256),
        encoding="utf-8",
    )
    return report


def _count_total_variation(
    actual: Counter[str], expected: dict[str, int]
) -> float:
    actual_total = sum(actual.values())
    expected_total = sum(expected.values())
    return 0.5 * sum(
        abs(actual.get(key, 0) / actual_total - expected[key] / expected_total)
        for key in expected
    )


def _periodic_yaw_counts(
    values: list[float],
    *,
    bins: int,
    period_deg: float,
) -> Counter[int]:
    if bins <= 0 or period_deg <= 0.0:
        raise ValueError("periodic yaw bins and period must be positive")
    result: Counter[int] = Counter()
    for value in values:
        unit = (float(value) % float(period_deg)) / float(period_deg)
        result[min(bins - 1, int(math.floor(unit * bins)))] += 1
    return result


def _nearest_neighbor_stats(points: list[list[float]]) -> dict[str, float | int | None]:
    if len(points) < 2:
        return {"count": 0, "min_m": None, "median_m": None, "max_m": None}
    values = [
        min(
            math.dist(point, other)
            for other_index, other in enumerate(points)
            if index != other_index
        )
        for index, point in enumerate(points)
    ]
    return {
        "count": len(values),
        "min_m": min(values),
        "median_m": statistics.median(values),
        "max_m": max(values),
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = [
        f"# SO101 Workspace Catalog Distribution: {report['catalog_id']}",
        "",
        f"**Gate:** `{report['gate']['status']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Primary candidates | {summary['primary_candidates']} |",
        f"| Backup candidates | {summary['backup_candidates']} |",
        f"| Continuous cells | {summary['cells']} |",
        f"| Per-cell count range | {summary['cell_count_min']}..{summary['cell_count_max']} |",
        f"| Angular count CV | {summary['angular_count_cv']:.6f} |",
        f"| Object yaw strata | {summary['object_yaw_strata']} |",
        f"| Object yaw count CV | {_optional_number(summary['object_yaw_count_cv'])} |",
        f"| Robot-relative yaw coverage | {_optional_number(summary['robot_relative_object_yaw_coverage_ratio'])} |",
        f"| Robot-relative yaw count CV | {_optional_number(summary['robot_relative_object_yaw_count_cv'])} |",
        f"| Far/near area-density ratio | {summary['far_to_near_area_density_ratio']:.6f} |",
        f"| Minimum spacing | {summary['nearest_neighbor_min_m'] * 1000:.3f} mm |",
        f"| Median spacing | {summary['nearest_neighbor_median_m'] * 1000:.3f} mm |",
        f"| Boundary hits | {summary['boundary_hits']} |",
        "",
        "## Radial Area Density",
        "",
        "| Stratum | Samples | Area (m²) | Samples/m² |",
        "|---:|---:|---:|---:|",
    ]
    rows.extend(
        f"| {row['radial_index']} | {row['count']} | {row['area_m2']:.8f} | "
        f"{row['samples_per_m2']:.2f} |"
        for row in report["radial_area_density"]
    )
    rows.extend(
        [
            "",
            "## Gate Checks",
            "",
            "| Check | Result |",
            "|---|---|",
            *[
                f"| `{name}` | {'PASS' if check['passed'] else 'FAIL'} |"
                for name, check in report["gate"]["checks"].items()
            ],
            "",
        ]
    )
    return "\n".join(rows)


def _optional_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _html(
    report: dict[str, Any],
    points: list[list[float]],
    markdown_sha256: str,
) -> str:
    status = report["gate"]["status"]
    cards = "".join(
        f"<div><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in [
            ("Primary", str(report["summary"]["primary_candidates"])),
            ("Cells", str(report["summary"]["cells"])),
            ("Cell range", f"{report['summary']['cell_count_min']}..{report['summary']['cell_count_max']}"),
            ("Far / near", f"{report['summary']['far_to_near_area_density_ratio']:.3f}"),
            ("Min spacing", f"{report['summary']['nearest_neighbor_min_m'] * 1000:.2f} mm"),
            ("Boundary hits", str(report["summary"]["boundary_hits"])),
        ]
    )
    gate_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td class={'pass' if check['passed'] else 'fail'}>"
        f"{'PASS' if check['passed'] else 'FAIL'}</td></tr>"
        for name, check in report["gate"]["checks"].items()
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Workspace Distribution</title>
<style>body{{margin:0;background:#f3f6fa;color:#172033;font:15px system-ui}}main{{max-width:1050px;margin:auto;padding:28px}}
h1{{margin:0 0 18px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.cards div,.panel{{background:white;border:1px solid #dbe4f0;border-radius:8px;padding:16px}}span{{display:block;color:#64748b;font-size:12px;text-transform:uppercase}}strong{{font-size:24px}}.panel{{margin-top:14px}}svg{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}td{{padding:8px;border-bottom:1px solid #edf2f7}}td:last-child{{text-align:right}}.pass{{color:#15803d;font-weight:800}}.fail{{color:#c2410c;font-weight:800}}@media(max-width:650px){{.cards{{grid-template-columns:1fr 1fr}}}}</style></head>
<body data-markdown-sha256="{markdown_sha256}"><main><h1>Continuous Workspace <span class="{'pass' if status == 'passed' else 'fail'}">{status.upper()}</span></h1>
<section class="cards">{cards}</section><section class="panel"><h2>Primary spawn coordinates</h2>{_scatter_svg(points)}</section>
<section class="panel"><h2>Pre-export gates</h2><table>{gate_rows}</table></section></main></body></html>"""


def _scatter_svg(points: list[list[float]]) -> str:
    xs = [row[0] for row in points]
    ys = [row[1] for row in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width, height, pad = 760.0, 420.0, 22.0
    def project(point: list[float]) -> tuple[float, float]:
        x = pad + (point[0] - x_min) / max(1e-12, x_max - x_min) * (width - 2 * pad)
        y = height - pad - (point[1] - y_min) / max(1e-12, y_max - y_min) * (height - 2 * pad)
        return x, y
    circles = "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.1" fill="#2563eb" fill-opacity=".72"/>'
        for x, y in map(project, points)
    )
    return f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img"><rect width="100%" height="100%" fill="#f8fafc"/>{circles}</svg>'


if __name__ == "__main__":
    main()
