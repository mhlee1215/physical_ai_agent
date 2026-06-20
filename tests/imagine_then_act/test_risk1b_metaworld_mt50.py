from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.run_risk1b_alt_goal_full_metaworld_mt50 import (
    aggregate,
    build_config,
    build_generation_argv,
    build_parser,
    build_rollout_argv,
    load_manifest,
    qwen_json_path,
    resolve_context_artifacts,
)


class Risk1BMetaWorldMt50Test(unittest.TestCase):
    def test_manifest_covers_mt50_group_counts(self) -> None:
        manifest = load_manifest(Path("configs/eval/risk1b_metaworld_mt50_manifest.json"))
        rows = manifest["rows"]

        self.assertEqual(len(rows), 50)
        self.assertEqual(
            Counter(row["task_group"] for row in rows),
            {"easy": 28, "medium": 11, "hard": 6, "very_hard": 5},
        )
        self.assertEqual(len({row["task_name"] for row in rows}), 50)

    def test_plan_defaults_to_zero_fallback_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(["--output-dir", tmpdir])
            config = build_config(args)

        self.assertEqual(config.repair_attempts, 3)
        self.assertEqual(config.fallback_on_validation_error, "none")
        self.assertEqual(config.episodes_per_task, 10)
        self.assertEqual(config.policy_path, "lerobot/smolvla_metaworld")
        self.assertEqual(config.policy_n_action_steps, 20)

    def test_generation_and_rollout_commands_use_metaworld_task_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(["--output-dir", tmpdir])
            config = build_config(args)
            row = {
                "row_id": "metaworld_drawer_open_v3_seed0",
                "task_group": "easy",
                "task_name": "drawer-open-v3",
                "seed": 0,
                "task_description": "Complete the Meta-World drawer open task.",
            }
            row_dir = Path(tmpdir) / row["row_id"]
            qwen_path = qwen_json_path(row_dir, config.model_id, row["task_name"], row["seed"])

            generation = build_generation_argv(
                config,
                row,
                row_dir,
                context_json=Path("/tmp/context.json"),
                context_image=Path("/tmp/contact_sheet.png"),
            )
            rollout = build_rollout_argv(config, row, row_dir, qwen_path)

        self.assertIn("--suite", generation)
        self.assertIn("metaworld", generation)
        self.assertIn("--fallback-on-validation-error", generation)
        self.assertIn("none", generation)
        self.assertIn("--repair-attempts", generation)
        self.assertIn("3", generation)
        self.assertIn("--context-json", generation)
        self.assertIn("/tmp/context.json", generation)
        self.assertIn("--context-image", generation)
        self.assertIn("/tmp/contact_sheet.png", generation)
        self.assertIn("--env.type=metaworld", rollout)
        self.assertIn("--env.task=drawer-open-v3", rollout)
        self.assertIn("--policy.path=lerobot/smolvla_metaworld", rollout)
        self.assertIn("--policy.n_action_steps=20", rollout)
        self.assertIn(f"--ita-candidate-prompts-json", rollout)
        self.assertIn(str(qwen_path), rollout)

    def test_planned_rows_are_not_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(["--output-dir", tmpdir])
            config = build_config(args)
            manifest = json.loads(Path("configs/eval/risk1b_metaworld_mt50_manifest.json").read_text())
            rows = [
                {
                    "row_id": "metaworld_assembly_v3_seed0",
                    "task_group": "easy",
                    "task_name": "assembly-v3",
                    "status": "planned",
                }
            ]

            summary = aggregate(rows, config, manifest)

        self.assertEqual(summary["planned_rows"], ["metaworld_assembly_v3_seed0"])
        self.assertEqual(summary["failed_rows"], [])

    def test_context_artifacts_resolve_from_context_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            row_dir = root / "metaworld_drawer_open_v3_seed0"
            row_dir.mkdir()
            context_json = row_dir / "context.json"
            context_image = row_dir / "contact_sheet.png"
            context_json.write_text(
                json.dumps({"provenance": {"actual_context": True}, "contact_sheet": "contact_sheet.png"}),
                encoding="utf-8",
            )
            context_image.write_bytes(b"not-a-real-png-for-path-resolution")
            args = build_parser().parse_args(["--output-dir", str(root / "out"), "--context-root", str(root)])
            config = build_config(args)
            row = {
                "row_id": "metaworld_drawer_open_v3_seed0",
                "task_name": "drawer-open-v3",
                "seed": 0,
            }

            artifacts = resolve_context_artifacts(config, row)

        self.assertEqual(artifacts["json"], context_json)
        self.assertEqual(artifacts["image"], context_image)


if __name__ == "__main__":
    unittest.main()
