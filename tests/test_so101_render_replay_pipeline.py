from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from physical_ai_agent.so101_dataset_generation_schema import (
    DatasetGenerationRecipe,
    load_dataset_generation_recipe,
)
from physical_ai_agent.so101_render_replay import (
    merge_render_replay_sidecars,
    sidecar_scene_items,
    write_captured_render_replay_sidecar,
)

RECIPE = Path("configs/so101/dataset_generation/grip_the_cube_v2_5_photoreal_preview.json")


class SO101RenderReplayPipelineTests(unittest.TestCase):
    def test_photoreal_preview_preserves_v25_teacher_contract(self) -> None:
        base = json.loads(
            Path("configs/so101/dataset_generation/grip_the_cube_v2_5.json").read_text(
                encoding="utf-8"
            )
        )
        preview = json.loads(RECIPE.read_text(encoding="utf-8"))
        for key, value in base["common"].items():
            with self.subTest(key=key):
                self.assertEqual(preview["common"].get(key), value)

    def test_all_registered_generation_recipes_validate_with_pydantic(self) -> None:
        for path in Path("configs/so101/dataset_generation").glob("*.json"):
            with self.subTest(path=path):
                self.assertEqual(load_dataset_generation_recipe(path).schema_version, 1)

    def test_schema_forbids_unknown_render_fields(self) -> None:
        payload = json.loads(RECIPE.read_text(encoding="utf-8"))
        broken = copy.deepcopy(payload)
        broken["splits"]["train"]["render"]["mystery_option"] = True
        with self.assertRaises(ValidationError):
            DatasetGenerationRecipe.model_validate(broken)

    def test_schema_rejects_environment_drift(self) -> None:
        payload = json.loads(RECIPE.read_text(encoding="utf-8"))
        broken = copy.deepcopy(payload)
        broken["render_replay"]["environment"]["target_object_color"] = "red"
        with self.assertRaisesRegex(ValidationError, "target_object_color"):
            DatasetGenerationRecipe.model_validate(broken)
        broken = copy.deepcopy(payload)
        broken["render_replay"]["environment"]["object_half_sizes"] = [0.02]
        broken["render_replay"]["environment"]["object_pool_order"] = [
            {"slot": 0, "color": "green", "half_size": 0.02}
        ]
        with self.assertRaisesRegex(ValidationError, "object_half_sizes"):
            DatasetGenerationRecipe.model_validate(broken)

    def test_camera_contract_has_explicit_intrinsics_and_extrinsics(self) -> None:
        import sys

        sys.path.insert(0, str(Path("scripts").resolve()))
        from render_so101_dataset_blender_preview import _with_explicit_camera_contract

        camera = _with_explicit_camera_contract(
            {
                "location": [1.0, 2.0, 3.0],
                "forward": [0.0, 0.0, -1.0],
                "up": [0.0, 1.0, 0.0],
                "fovy": 90.0,
                "clip_start": 0.001,
            },
            role="wrist_cam",
            width=256,
            height=256,
        )
        self.assertEqual(len(camera["world_from_camera"]), 16)
        self.assertAlmostEqual(camera["intrinsics"]["fx"], 128.0)
        self.assertEqual(camera["intrinsics"]["cx"], 128.0)
        self.assertEqual(camera["role"], "wrist_cam")

    def test_photoreal_builder_indexes_png_and_jpeg_profiles(self) -> None:
        import sys

        sys.path.insert(0, str(Path("scripts").resolve()))
        from build_so101_photoreal_lerobot_dataset import _rendered_index

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "episode_0000_frame_0000_camera1.jpg").write_bytes(b"jpeg")
            (root / "episode_0000_frame_0000_camera2.png").write_bytes(b"png")
            index = _rendered_index(root)
            self.assertIn((0, 0, "observation.images.camera1"), index)
            self.assertIn((0, 0, "observation.images.camera2"), index)

    def test_recipe_builds_capture_replay_render_and_derivative_stages(self) -> None:
        import sys

        sys.path.insert(0, str(Path("scripts").resolve()))
        from generate_so101_dataset_recipe import build_stages

        recipe = load_dataset_generation_recipe(RECIPE).as_dict()
        stages = build_stages(
            recipe,
            python="python",
            split="all",
            overwrite=False,
            recipe_path=RECIPE,
        )
        names = [stage["name"] for stage in stages]
        self.assertLess(names.index("merge:source"), names.index("render-replay:source"))
        self.assertLess(names.index("render-replay:source"), names.index("render:train"))
        self.assertLess(names.index("render:train"), names.index("build-derivative:train"))
        export = next(
            stage["command"] for stage in stages if stage["name"].startswith("export:source:")
        )
        render = next(stage["command"] for stage in stages if stage["name"] == "render:train")
        self.assertIn("--capture-render-replay", export)
        self.assertIn("--render-replay-sidecar", render)
        self.assertEqual(render[render.index("--width") + 1], "256")
        self.assertEqual(render[render.index("--height") + 1], "256")
        self.assertEqual(render[render.index("--output-format") + 1], "PNG")
        self.assertEqual(render[render.index("--lighting-profile") + 1], "studio_small_08")
        self.assertEqual(render[render.index("--exposure") + 1], "-1.3")

    def test_sidecar_scene_uses_frame_transform_and_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh = root / "assets" / "meshes" / "cube.ply"
            mesh.parent.mkdir(parents=True)
            mesh.write_bytes(b"ply")
            manifest = {
                "model": {"ngeom": 2},
                "geom_manifest": [
                    {
                        "geom_id": 0,
                        "name": "arm",
                        "body_name": "arm",
                        "type": "mesh",
                        "size": [0.0, 0.0, 0.0],
                        "asset_path": "assets/meshes/cube.ply",
                    },
                    {
                        "geom_id": 1,
                        "name": "pick_slot_0_geom",
                        "body_name": "cube",
                        "type": "box",
                        "size": [0.01, 0.01, 0.01],
                        "asset_path": None,
                    },
                ],
            }
            frame = {
                "geom_positions": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "geom_quaternions_wxyz": [1.0, 0.0, 0.0, 0.0] * 2,
                "geom_rgba": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                "geom_visible": [True, False],
            }
            meshes, primitives = sidecar_scene_items(root, manifest, frame)
            self.assertEqual(meshes[0]["position"], [1.0, 2.0, 3.0])
            self.assertEqual(meshes[0]["quaternion_wxyz"], [1.0, 0.0, 0.0, 0.0])
            self.assertEqual(primitives, [])

    def test_episode_snapshot_uses_exact_first_frame_metadata(self) -> None:
        import mujoco
        import numpy as np

        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><freejoint/>"
            "<geom name='pick_slot_2_geom' type='box' size='.01 .01 .01'/>"
            "</body></worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
        frame = {
            "episode_index": 0,
            "frame_index": 0,
            "timestamp": 0.0,
            "integration_state": np.zeros(state_size, dtype=float).tolist(),
            "qpos": np.zeros(model.nq, dtype=float).tolist(),
            "qvel": np.zeros(model.nv, dtype=float).tolist(),
            "ctrl": np.zeros(model.nu, dtype=float).tolist(),
            "act": np.zeros(model.na, dtype=float).tolist(),
            "mocap_pos": [1.0, 2.0, 3.0],
            "mocap_quat": [1.0, 0.0, 0.0, 0.0],
            "rng_state_json": '{"state": 7}',
            "collision_state_json": '{"pick_slot_2_geom": {"conaffinity": 1, "contype": 1}}',
            "active_object_slots": [2],
            "geom_positions": np.asarray(data.geom_xpos, dtype=float).reshape(-1).tolist(),
            "geom_quaternions_wxyz": [1.0, 0.0, 0.0, 0.0] * model.ngeom,
            "geom_rgba": np.asarray(model.geom_rgba, dtype=float).reshape(-1).tolist(),
            "geom_visible": [True] * model.ngeom,
            "camera_specs_json": "{}",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_captured_render_replay_sidecar(
                root,
                model=model,
                episode_captures=[
                    {
                        "episode_index": 0,
                        "seed": 11,
                        "target_slot_index": 2,
                        "initial_object_z": 0.01,
                        "frames": [frame],
                    }
                ],
                environment={"factory": "test"},
            )
            snapshot = pq.read_table(
                root / "render_replay" / "episode_snapshots.parquet"
            ).to_pylist()[0]
            self.assertEqual(snapshot["mocap_pos"], [1.0, 2.0, 3.0])
            self.assertEqual(snapshot["active_object_slots"], [2])
            self.assertEqual(json.loads(snapshot["rng_state_json"]), {"state": 7})

    def test_merge_sidecars_offsets_episode_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shards = [root / "shard0", root / "shard1"]
            for index, shard in enumerate(shards):
                sidecar = shard / "render_replay"
                (sidecar / "assets" / "meshes").mkdir(parents=True)
                (sidecar / "assets" / "meshes" / "mesh.ply").write_bytes(b"same")
                manifest = {
                    "schema_version": 1,
                    "capture_mode": "teacher_time_exact",
                    "source_episodes": 1,
                    "source_frames": 1,
                    "model": {"ngeom": 1},
                    "geom_manifest": [],
                    "files": {
                        "episode_snapshots": "episode_snapshots.parquet",
                        "frame_world_state": "frame_world_state.parquet",
                        "frame_camera_specs": "frame_camera_specs.parquet",
                        "asset_checksums": "asset_checksums.json",
                    },
                    "validation": {"passed": True},
                }
                (sidecar / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                (sidecar / "asset_checksums.json").write_text(
                    json.dumps({"assets/meshes/mesh.ply": "same"}), encoding="utf-8"
                )
                pq.write_table(
                    pa.Table.from_pylist([{"episode_index": 0, "seed": index}]),
                    sidecar / "episode_snapshots.parquet",
                )
                pq.write_table(
                    pa.Table.from_pylist([{"episode_index": 0, "frame_index": 0}]),
                    sidecar / "frame_world_state.parquet",
                )
                pq.write_table(
                    pa.Table.from_pylist([{"episode_index": 0, "frame_index": 0}]),
                    sidecar / "frame_camera_specs.parquet",
                )
            output = root / "merged"
            output.mkdir()
            manifest = merge_render_replay_sidecars(shards, output)
            self.assertIsNotNone(manifest)
            rows = pq.read_table(output / "render_replay" / "frame_world_state.parquet").to_pylist()
            self.assertEqual([row["episode_index"] for row in rows], [0, 1])


if __name__ == "__main__":
    unittest.main()
