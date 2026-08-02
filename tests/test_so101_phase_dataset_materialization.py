from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

from physical_ai_agent.so101_dataset_generation_schema import (
    DatasetGenerationRecipe,
    PhaseSubsetSpec,
)

SCRIPTS = Path("scripts").resolve()
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_so101_dataset_recipe import build_stages  # noqa: E402
from materialize_so101_phase_dataset import (  # noqa: E402
    SourceSpec,
    _episode_matches_source,
    _observation_state_replay_rmse,
    _phase_start_policy_camera_visibility,
    _source_dataset_manifest,
    phase_frame_window,
    rewrite_phase_table,
)

RECIPE_ROOT = Path("configs/so101/dataset_generation")
PHASE_RECIPES = {
    "approach": (
        RECIPE_ROOT
        / "grip_the_cube_v3_hardware_locked_photoreal_approach_phase_v1.json",
        "Open the gripper and move it above the visible green cube.",
        300,
        50,
    ),
    "alignment": (
        RECIPE_ROOT
        / "grip_the_cube_v3_hardware_locked_photoreal_alignment_phase_v1.json",
        "Align the open gripper jaws with the visible green cube edge.",
        500,
        50,
    ),
    "grip_lift": (
        RECIPE_ROOT
        / "grip_the_cube_v3_hardware_locked_photoreal_grip_lift_phase_v1.json",
        "Close the aligned gripper on the visible green cube and lift it.",
        500,
        50,
    ),
}
V44_PHASE_RECIPES = {
    "move": (
        RECIPE_ROOT / "grip_the_cube_v4_4_move_phase_v2.json",
        "Open the gripper and move it above the visible green cube.",
        500,
        50,
    ),
    "align": (
        RECIPE_ROOT / "grip_the_cube_v4_4_align_phase_v2.json",
        "Fine-align the open gripper jaws with the visible green cube.",
        800,
        50,
    ),
    "grip_lift": (
        RECIPE_ROOT / "grip_the_cube_v4_4_grip_lift_phase_v2.json",
        "Close the aligned gripper on the visible green cube and lift it.",
        800,
        50,
    ),
}


class SO101PhaseDatasetMaterializationTests(unittest.TestCase):
    def test_replay_state_contract_uses_motor_ctrl_not_physical_qpos(self) -> None:
        replay = {
            "qpos": [9.0] * 6,
            "ctrl": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
        expected = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        self.assertEqual(_observation_state_replay_rmse(replay, expected), 0.0)

    def test_phase_start_visibility_is_recomputed_from_selected_images(self) -> None:
        green = np.zeros((16, 16, 3), dtype=np.uint8)
        green[4:12, 5:13] = (0, 180, 0)
        black = np.zeros((16, 16, 3), dtype=np.uint8)

        def encoded(image: np.ndarray) -> bytes:
            output = io.BytesIO()
            Image.fromarray(image).save(output, format="PNG")
            return output.getvalue()

        image_type = pa.struct(
            [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
        )
        selected = pa.table(
            {
                "observation.images.camera1": pa.array(
                    [{"bytes": encoded(green), "path": "camera1.png"}],
                    type=image_type,
                ),
                "observation.images.camera2": pa.array(
                    [{"bytes": encoded(black), "path": "camera2.png"}],
                    type=image_type,
                ),
            }
        )

        visibility = _phase_start_policy_camera_visibility(selected)

        self.assertTrue(visibility["camera1"]["visible"])
        self.assertEqual(visibility["camera1"]["area"], 64)
        self.assertFalse(visibility["camera2"]["visible"])
        self.assertEqual(visibility["camera2"]["area"], 0)

    def test_variant_filtered_sources_may_repeat_a_root_when_disjoint(self) -> None:
        payload = {
            "phase_id": "move",
            "prompt": "Move above the cube.",
            "sources": [
                {
                    "source_dataset_root": "dataset",
                    "phase_order": ["move", "align"],
                    "phases": ["move"],
                    "trajectory_variants": ["standard"],
                },
                {
                    "source_dataset_root": "dataset",
                    "phase_order": ["move_and_align", "grip"],
                    "phases": ["move_and_align"],
                    "trajectory_variants": ["direct_align"],
                },
            ],
        }
        spec = PhaseSubsetSpec.model_validate(payload)
        self.assertEqual(len(spec.sources), 2)

        payload["sources"][1]["trajectory_variants"] = ["standard"]
        with self.assertRaisesRegex(ValueError, "disjoint trajectory_variants"):
            PhaseSubsetSpec.model_validate(payload)

    def test_materializer_filters_recorded_trajectory_variants(self) -> None:
        source = SourceSpec(
            root=Path("dataset"),
            phase_order=("move", "align"),
            phases=("move",),
            trajectory_variants=("standard",),
        )
        self.assertTrue(
            _episode_matches_source({"trajectory_variant": "standard"}, source)
        )
        self.assertFalse(
            _episode_matches_source({"trajectory_variant": "direct_align"}, source)
        )

    def test_non_photoreal_source_gets_a_phase_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "total_episodes": 2,
                        "total_frames": 5,
                        "features": {
                            "observation.images.camera1": {"dtype": "image"},
                            "observation.images.camera2": {"dtype": "image"},
                            "observation.state": {"dtype": "float32"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = _source_dataset_manifest(root)
        self.assertEqual(
            manifest["camera_keys"],
            ["observation.images.camera1", "observation.images.camera2"],
        )
        self.assertEqual(manifest["episodes"], 2)

    def test_phase_window_must_be_contiguous(self) -> None:
        counts = {"approach": 3, "align": 2, "grip": 4}

        self.assertEqual(
            phase_frame_window(
                counts,
                phase_order=("approach", "align", "grip"),
                phases=("align", "grip"),
            ),
            (3, 9),
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            phase_frame_window(
                counts,
                phase_order=("approach", "align", "grip"),
                phases=("approach", "grip"),
            )

    def test_rewrite_preserves_payload_and_reindexes_episode(self) -> None:
        image_type = pa.struct(
            [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
        )
        image_values = pa.array(
            [
                {"bytes": b"frame-a", "path": "old/a.png"},
                {"bytes": b"frame-b", "path": "old/b.png"},
            ],
            type=image_type,
        )
        table = pa.table(
            {
                "observation.images.camera1": image_values,
                "observation.images.camera2": image_values,
                "observation.images.camera3": image_values,
                "observation.state": pa.array(
                    [[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32(), 2)
                ),
                "action": pa.array(
                    [[5.0, 6.0], [7.0, 8.0]], type=pa.list_(pa.float32(), 2)
                ),
                "timestamp": pa.array([4.0, 5.0], type=pa.float32()),
                "frame_index": pa.array([4, 5], type=pa.int64()),
                "episode_index": pa.array([9, 9], type=pa.int64()),
                "index": pa.array([90, 91], type=pa.int64()),
                "task_index": pa.array([7, 7], type=pa.int64()),
            }
        )

        result = rewrite_phase_table(
            table,
            output_episode_index=2,
            output_global_start=12,
            fps=12,
            prompt="phase prompt",
        )

        self.assertEqual(
            result["observation.state"].to_pylist(),
            table["observation.state"].to_pylist(),
        )
        self.assertEqual(result["action"].to_pylist(), table["action"].to_pylist())
        self.assertEqual(result["frame_index"].to_pylist(), [0, 1])
        self.assertEqual(result["episode_index"].to_pylist(), [2, 2])
        self.assertEqual(result["index"].to_pylist(), [12, 13])
        self.assertEqual(result["task_index"].to_pylist(), [0, 0])
        self.assertEqual(
            [
                value["bytes"]
                for value in result["observation.images.camera1"].to_pylist()
            ],
            [b"frame-a", b"frame-b"],
        )
        self.assertEqual(
            [
                value["path"]
                for value in result["observation.images.camera1"].to_pylist()
            ],
            [
                "images/observation_images_camera1/episode_000002_frame_000000.png",
                "images/observation_images_camera1/episode_000002_frame_000001.png",
            ],
        )

    def test_phase_recipes_have_independent_prompts_and_counts(self) -> None:
        for phase_id, (
            path,
            expected_prompt,
            train_episodes,
            validation_episodes,
        ) in PHASE_RECIPES.items():
            with self.subTest(phase_id=phase_id):
                recipe = DatasetGenerationRecipe.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                self.assertEqual(recipe.source.operation, "phase_subset")
                self.assertEqual(recipe.splits["train"].expected_episodes, train_episodes)
                self.assertEqual(
                    recipe.splits["validation"].expected_episodes,
                    validation_episodes,
                )
                self.assertEqual(
                    recipe.splits["train"].phase_subset.prompt,
                    expected_prompt,
                )
                self.assertEqual(
                    recipe.splits["validation"].phase_subset.prompt,
                    expected_prompt,
                )
                self.assertEqual(recipe.audit.expected_prompt, expected_prompt)
                self.assertNotEqual(
                    recipe.splits["train"].output_root,
                    recipe.splits["validation"].output_root,
                )

    def test_v44_phase_recipes_have_exact_counts_prompts_and_no_loop_starts(self) -> None:
        for phase_id, (
            path,
            expected_prompt,
            train_episodes,
            validation_episodes,
        ) in V44_PHASE_RECIPES.items():
            with self.subTest(phase_id=phase_id):
                recipe = DatasetGenerationRecipe.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                self.assertEqual(recipe.splits["train"].expected_episodes, train_episodes)
                self.assertEqual(
                    recipe.splits["validation"].expected_episodes,
                    validation_episodes,
                )
                self.assertEqual(
                    recipe.splits["train"].phase_subset.phase_id,
                    phase_id,
                )
                self.assertEqual(
                    recipe.splits["train"].phase_subset.prompt,
                    expected_prompt,
                )
                self.assertEqual(recipe.audit.expected_prompt, expected_prompt)
                self.assertIsNone(recipe.splits["validation"].closed_loop)

                stages = build_stages(
                    recipe.as_dict(),
                    python="python",
                    split="all",
                    overwrite=False,
                    recipe_path=path,
                )
                names = [stage["name"] for stage in stages]
                self.assertNotIn("closed-loop-starts:validation", names)

    def test_v44_parent_recipe_owns_the_single_end_to_end_loop(self) -> None:
        recipe = DatasetGenerationRecipe.model_validate_json(
            (RECIPE_ROOT / "grip_the_cube_v4_4.json").read_text(encoding="utf-8")
        )
        closed_loop = recipe.splits["validation"].closed_loop
        self.assertIsNotNone(closed_loop)
        self.assertEqual(closed_loop.episodes, 10)
        self.assertEqual(closed_loop.success_metric, "env_success")
        self.assertEqual(
            closed_loop.output,
            "meta/closed_loop/grip_the_cube_v4_4_validation_start10.json",
        )

    def test_v44_primitive_contract_has_only_move_align_and_grip_lift(self) -> None:
        self.assertEqual(set(V44_PHASE_RECIPES), {"move", "align", "grip_lift"})

        move = DatasetGenerationRecipe.model_validate_json(
            V44_PHASE_RECIPES["move"][0].read_text(encoding="utf-8")
        )
        move_sources = move.splits["train"].phase_subset.sources
        self.assertEqual(len(move_sources), 2)
        self.assertEqual(
            {tuple(source.phases) for source in move_sources},
            {
                ("open_from_hardware_start", "move_to_cube"),
                ("open_from_hardware_start", "move_and_align"),
            },
        )

        align = DatasetGenerationRecipe.model_validate_json(
            V44_PHASE_RECIPES["align"][0].read_text(encoding="utf-8")
        )
        align_sources = align.splits["train"].phase_subset.sources
        self.assertEqual(len(align_sources), 3)
        self.assertTrue(
            all(
                "move_to_cube" not in source.phases
                and "move_and_align" not in source.phases
                for source in align_sources
            )
        )

    def test_phase_recipe_builds_materialize_audit_and_completion_stages(self) -> None:
        path = PHASE_RECIPES["alignment"][0]
        recipe = DatasetGenerationRecipe.model_validate_json(
            path.read_text(encoding="utf-8")
        ).as_dict()

        stages = build_stages(
            recipe,
            python="python",
            split="all",
            overwrite=False,
            recipe_path=path,
        )
        names = [stage["name"] for stage in stages]

        self.assertEqual(names.count("phase-subset:train"), 1)
        self.assertEqual(names.count("phase-subset:validation"), 1)
        self.assertIn("audit:train-vs-validation", names)
        self.assertEqual(names[-1], "completion:registry-viewer")
        train_command = next(
            stage["command"]
            for stage in stages
            if stage["name"] == "phase-subset:train"
        )
        self.assertIn("--reconstruct-sim-snapshots", train_command)
        self.assertEqual(train_command.count("--source-spec"), 2)
        self.assertNotIn("--overwrite", train_command)


if __name__ == "__main__":
    unittest.main()
