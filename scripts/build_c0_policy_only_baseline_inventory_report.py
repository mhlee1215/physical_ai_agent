#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineRow:
    checkpoint_dir: str
    status: str
    env_id: str
    executed_env_id: str
    policies: list[str]
    episodes_requested: int
    rollout_episodes: int
    rollout_steps: int
    success_count: int
    success_rate: str
    real_images: bool
    smolvla_ready: str
    paper_ready: bool
    paper_gap: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-root", default="_workspace/checkpoints")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=18)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _rows(Path(args.checkpoints_root))[: args.limit]
    paper_ready_count = sum(1 for row in rows if row.paper_ready)
    manifest = {
        "status": "passed_inventory_policy_only_not_claim_ready",
        "source_type": "c0_policy_only_baseline_inventory",
        "sample_count": len(rows),
        "paper_ready_count": paper_ready_count,
        "claim_ready": False,
        "final_success_source": "environment_success_flags_only",
        "rows": [asdict(row) for row in rows],
        "claim_boundary": (
            "Inventory of existing CP24 baseline artifacts. It does not prove the C0 paper baseline "
            "until a fixed-seed/fixed-budget SmolVLA policy-only run is selected."
        ),
    }
    (output_dir / "c0_policy_only_baseline_inventory_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    html_path = output_dir / "c0_policy_only_baseline_inventory.html"
    html_path.write_text(_html(rows, manifest), encoding="utf-8")
    print(html_path)
    return 0


def _rows(root: Path) -> list[BaselineRow]:
    rows: list[BaselineRow] = []
    for report_path in sorted(root.glob("checkpoint_24*/checkpoint_report.json")):
        report = _load_json(report_path)
        metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
        policies = metrics.get("policies_requested", metrics.get("policies", []))
        if not isinstance(policies, list):
            policies = [str(policies)]
        checkpoint_dir = report_path.parent
        rollout_episodes = _int(metrics.get("rollout_episodes", metrics.get("episodes", 0)))
        rollout_steps = _int(metrics.get("rollout_steps", metrics.get("steps", 0)))
        success_count = _int(metrics.get("rollout_success_count", metrics.get("success_count", 0)))
        success_rate = str(metrics.get("success_rate", "missing"))
        real_images = bool(metrics.get("real_images", False))
        smolvla_ready = str(metrics.get("smolvla_ready", "missing"))
        paper_ready, gap = _paper_readiness(policies, rollout_episodes, rollout_steps, real_images, report.get("status"))
        rows.append(
            BaselineRow(
                checkpoint_dir=str(checkpoint_dir),
                status=str(report.get("status", "missing")),
                env_id=str(metrics.get("env_id", metrics.get("requested_env_id", "missing"))),
                executed_env_id=str(metrics.get("executed_env_id", metrics.get("env_id", "missing"))),
                policies=[str(policy) for policy in policies],
                episodes_requested=_int(metrics.get("episodes_requested", metrics.get("episodes_per_policy", 0))),
                rollout_episodes=rollout_episodes,
                rollout_steps=rollout_steps,
                success_count=success_count,
                success_rate=success_rate,
                real_images=real_images,
                smolvla_ready=smolvla_ready,
                paper_ready=paper_ready,
                paper_gap=gap,
            )
        )
    rows.sort(key=lambda row: (not row.paper_ready, "smolvla" not in ",".join(row.policies), row.checkpoint_dir))
    return rows


def _paper_readiness(
    policies: list[Any],
    rollout_episodes: int,
    rollout_steps: int,
    real_images: bool,
    status: object,
) -> tuple[bool, str]:
    policy_names = {str(policy) for policy in policies}
    if status != "passed":
        return False, "checkpoint did not pass"
    if not rollout_episodes or not rollout_steps:
        return False, "no rollout episodes/steps"
    if "smolvla_real" not in policy_names:
        return False, "not SmolVLA policy-only"
    if not real_images:
        return False, "not actual image policy input"
    if rollout_episodes < 10:
        return False, "fewer than 10 episodes"
    return True, "candidate C0 baseline; still needs fixed seed/budget selection"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _int(value: Any) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return 0


def _html(rows: list[BaselineRow], manifest: dict[str, Any]) -> str:
    cards = "\n".join(_card(row, index + 1) for index, row in enumerate(rows))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>C0 Policy-only Baseline Inventory</title>
  <style>
    :root {{ --ink:#172018; --panel:#fffaf0; --line:#d1bd91; --green:#2f7d59; --red:#b13f31; --gold:#ad7c1d; --muted:#65705f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Avenir Next", "Trebuchet MS", sans-serif; color:var(--ink); background:radial-gradient(circle at 10% 8%, rgba(47,125,89,.18), transparent 30%), radial-gradient(circle at 88% 8%, rgba(177,63,49,.15), transparent 28%), linear-gradient(135deg,#fcf5e4,#ead7ad); }}
    header {{ padding:44px clamp(18px,5vw,72px) 24px; }}
    h1 {{ max-width:1120px; margin:0; font-size:clamp(38px,6vw,78px); line-height:.94; letter-spacing:-.06em; }}
    .lead {{ max-width:980px; color:var(--muted); font-size:clamp(17px,2vw,22px); line-height:1.45; margin-top:18px; }}
    main {{ padding:0 clamp(18px,5vw,72px) 70px; }}
    .summary,.card {{ background:rgba(255,250,240,.88); border:1px solid var(--line); border-radius:24px; padding:18px; box-shadow:0 18px 52px rgba(48,35,10,.11); }}
    .summary {{ max-width:1180px; margin-bottom:20px; }}
    .grid {{ max-width:1180px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
    .card {{ min-height:260px; border-width:2px; }}
    .ready {{ border-color:var(--green); }} .gap {{ border-color:var(--gold); }}
    .tag {{ display:inline-block; padding:6px 10px; border-radius:999px; background:var(--ink); color:#fff9e8; font-size:12px; font-weight:800; }}
    h2 {{ margin:15px 0 8px; font-size:21px; letter-spacing:-.025em; overflow-wrap:anywhere; }}
    p {{ color:var(--muted); line-height:1.43; overflow-wrap:anywhere; }}
    code {{ background:rgba(23,32,24,.08); padding:2px 5px; border-radius:6px; }}
    @media (max-width:920px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>C0 Policy-only Baseline Inventory</h1>
    <p class="lead">Inventory of existing CP24 baseline artifacts. It separates candidate SmolVLA policy-only evidence from runs that are useful but not paper-ready C0 baselines.</p>
  </header>
  <main>
    <section class="summary">
      <p><strong>Status:</strong> <code>{html.escape(str(manifest['status']))}</code></p>
      <p><strong>Rows:</strong> <code>{manifest['sample_count']}</code>. <strong>Paper-ready candidates:</strong> <code>{manifest['paper_ready_count']}</code>. <strong>Claim ready:</strong> <code>{manifest['claim_ready']}</code>.</p>
      <p>{html.escape(str(manifest['claim_boundary']))}</p>
    </section>
    <section class="grid">{cards}</section>
  </main>
</body>
</html>
"""


def _card(row: BaselineRow, number: int) -> str:
    klass = "ready" if row.paper_ready else "gap"
    return f"""
      <article class="card {klass}">
        <span class="tag">{number}</span>
        <h2>{html.escape(Path(row.checkpoint_dir).name)}</h2>
        <p><strong>Status:</strong> <code>{html.escape(row.status)}</code>; <strong>paper_ready:</strong> <code>{row.paper_ready}</code></p>
        <p><strong>Policy:</strong> <code>{html.escape(', '.join(row.policies))}</code></p>
        <p><strong>Env:</strong> <code>{html.escape(row.executed_env_id)}</code>; <strong>real_images:</strong> <code>{row.real_images}</code></p>
        <p><strong>Episodes/steps:</strong> <code>{row.rollout_episodes}/{row.rollout_steps}</code>; <strong>success:</strong> <code>{row.success_count}</code>; <strong>rate:</strong> <code>{html.escape(row.success_rate)}</code></p>
        <p><strong>Gap:</strong> {html.escape(row.paper_gap)}</p>
      </article>
    """


if __name__ == "__main__":
    raise SystemExit(main())
