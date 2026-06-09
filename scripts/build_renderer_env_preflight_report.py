#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-json", default="_workspace/checkpoints/renderer_env_preflight/renderer_env_preflight.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    preflight_path = Path(args.preflight_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = _load_json(preflight_path)
    if not preflight:
        preflight = {
            "status": "not_run",
            "source_type": "renderer_env_preflight",
            "sample_count": 12,
            "claim_boundary": "Preflight report scaffold only; no preflight JSON was found.",
        }
    manifest = {
        "status": preflight.get("status", "not_run"),
        "source_type": "renderer_env_preflight_report",
        "sample_count": 12,
        "preflight_json": str(preflight_path),
        "true_oracle_projection": False,
        "claim_boundary": "Preflight only; does not run simulation, create pods, or prove Tier O.",
    }
    (output_dir / "renderer_env_preflight_report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    html_path = output_dir / "renderer_env_preflight_report.html"
    html_path.write_text(_html(preflight, manifest), encoding="utf-8")
    print(html_path)
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _html(preflight: dict[str, Any], manifest: dict[str, Any]) -> str:
    imports = preflight.get("imports", {}) if isinstance(preflight.get("imports"), dict) else {}
    commands = preflight.get("commands", {}) if isinstance(preflight.get("commands"), dict) else {}
    cards = [
        ("1", "Preflight status", str(preflight.get("status", "not_run"))),
        ("2", "Platform", str(preflight.get("platform", "pending"))),
        ("3", "Python executable", str(preflight.get("python_executable", "pending"))),
        ("4", "Gymnasium import", _import_status(imports, "gymnasium")),
        ("5", "ManiSkill import", _import_status(imports, "mani_skill")),
        ("6", "SAPIEN import", _import_status(imports, "sapien")),
        ("7", "Torch import", _import_status(imports, "torch")),
        ("8", "LeRobot import", _import_status(imports, "lerobot")),
        ("9", "nvidia-smi", _command_status(commands, "nvidia_smi")),
        ("10", "vulkaninfo", _command_status(commands, "vulkaninfo_summary")),
        ("11", "Next if passed", str(preflight.get("next_command_if_passed", "run two-stage true-oracle gate"))),
        ("12", "Claim boundary", str(manifest.get("claim_boundary"))),
    ]
    card_html = "\n".join(
        f"""
        <article class="card">
          <span class="tag">{html.escape(number)}</span>
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(text)}</p>
        </article>
        """
        for number, title, text in cards
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Renderer Environment Preflight</title>
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
    .card {{ min-height:185px; }}
    .tag {{ display:inline-block; padding:6px 10px; border-radius:999px; background:var(--ink); color:#fff9e8; font-size:12px; font-weight:800; }}
    h2 {{ margin:15px 0 8px; font-size:23px; letter-spacing:-.025em; }}
    p {{ color:var(--muted); line-height:1.43; overflow-wrap:anywhere; }}
    code {{ background:rgba(23,32,24,.08); padding:2px 5px; border-radius:6px; }}
    @media (max-width:920px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Renderer Environment Preflight</h1>
    <p class="lead">A non-mutating preflight for renderer-capable environments. It checks imports and GPU/Vulkan visibility before running the two-stage true-oracle gate.</p>
  </header>
  <main>
    <section class="summary">
      <p><strong>Status:</strong> <code>{html.escape(str(preflight.get('status', 'not_run')))}</code></p>
      <p><strong>Required imports OK:</strong> <code>{html.escape(str(preflight.get('required_imports_ok', 'pending')))}</code></p>
      <p><strong>GPU or Vulkan visible:</strong> <code>{html.escape(str(preflight.get('gpu_or_vulkan_visible', 'pending')))}</code></p>
    </section>
    <section class="grid">{card_html}</section>
  </main>
</body>
</html>
"""


def _import_status(imports: dict[str, Any], name: str) -> str:
    item = imports.get(name, {})
    if not isinstance(item, dict):
        return "pending"
    return f"imported={item.get('imported')}; version={item.get('version')}; error={item.get('error', '')}"


def _command_status(commands: dict[str, Any], name: str) -> str:
    item = commands.get(name, {})
    if not isinstance(item, dict):
        return "pending"
    return f"available={item.get('available')}; returncode={item.get('returncode')}; stderr={item.get('stderr', '')[:180]}"


if __name__ == "__main__":
    raise SystemExit(main())
