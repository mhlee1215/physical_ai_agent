from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OracleAffordanceOverlay:
    camera_name: str
    point_xy: list[int]
    image_shape: list[int]
    confidence: float
    mode: str
    label: str
    overlay_image_path: str | None = None
    object_pose_xyz: list[float] | None = None
    camera_metadata_keys: list[str] | None = None
    projection_source: str = "unknown"

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def build_oracle_affordance_overlay(
    obs: Any,
    output_path: Path | None = None,
    preferred_camera: str = "base_camera",
    label: str = "grasp point",
) -> tuple[dict[str, Any], OracleAffordanceOverlay]:
    """Build a zero-parameter affordance overlay for ManiSkill-style RGB obs.

    The first implementation is intentionally conservative: try to project an
    object/target pose when common simulator keys are present, otherwise use a
    deterministic image-center fallback so Mac-local visualization still works.
    """

    images = extract_rgb_images(obs)
    if not images:
        raise RuntimeError("No RGB camera observations found for oracle affordance overlay")

    camera_name = preferred_camera if preferred_camera in images else next(iter(images))
    pixels = images[camera_name]
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    object_pose_xyz = _find_pose_xyz(obs)
    camera_metadata_keys = _camera_metadata_keys(obs, camera_name)
    projected = _project_known_pose(obs, camera_name, width, height)
    if projected is None:
        point_xy = [width // 2, height // 2]
        confidence = 0.5
        mode = "image_center_fallback"
    else:
        point_xy = projected
        confidence = 1.0
        mode = "projected_object_pose"

    overlay = _draw_overlay(pixels, point_xy, label)
    if output_path is not None:
        _write_image(overlay, output_path)
    return (
        {camera_name: overlay},
        OracleAffordanceOverlay(
            camera_name=camera_name,
            point_xy=point_xy,
            image_shape=[height, width, 3],
            confidence=confidence,
            mode=mode,
            label=label,
            overlay_image_path=str(output_path) if output_path is not None else None,
            object_pose_xyz=object_pose_xyz,
            camera_metadata_keys=camera_metadata_keys,
            projection_source="sim_pose_camera" if mode == "projected_object_pose" else "fallback_center",
        ),
    )


def build_center_overlay_from_image_path(
    input_path: Path,
    output_path: Path,
    label: str = "fallback point",
) -> OracleAffordanceOverlay:
    from PIL import Image
    import numpy as np

    pixels = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8)
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    point_xy = [width // 2, height // 2]
    overlay = _draw_overlay(pixels, point_xy, label)
    _write_image(overlay, output_path)
    return OracleAffordanceOverlay(
        camera_name=input_path.name,
        point_xy=point_xy,
        image_shape=[height, width, 3],
        confidence=0.5,
        mode="image_center_file_fallback",
        label=label,
        overlay_image_path=str(output_path),
    )


def extract_rgb_images(obs: Any) -> dict[str, Any]:
    images: dict[str, Any] = {}
    if not isinstance(obs, dict):
        return images
    sensor_data = obs.get("sensor_data", {})
    if not isinstance(sensor_data, dict):
        return images
    for camera_name, camera_data in sorted(sensor_data.items(), key=lambda item: str(item[0])):
        if not isinstance(camera_data, dict) or "rgb" not in camera_data:
            continue
        try:
            images[str(camera_name)] = _rgb_tensor_to_hwc_uint8(camera_data["rgb"])
        except Exception:  # noqa: BLE001
            continue
    return images


def _project_known_pose(obs: Any, camera_name: str, width: int, height: int) -> list[int] | None:
    import numpy as np

    point_world = _find_pose_xyz(obs)
    intrinsic = _find_camera_matrix(obs, camera_name, ("intrinsic_cv", "intrinsic", "cam_intrinsic"))
    if point_world is None or intrinsic is None:
        return None

    transforms = _candidate_camera_transforms(obs, camera_name)
    if not transforms:
        return None

    point_h = np.asarray([point_world[0], point_world[1], point_world[2], 1.0], dtype=float)
    k = np.asarray(intrinsic, dtype=float)
    if k.shape != (3, 3):
        return None
    for transform in transforms:
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4):
            continue
        for candidate in (matrix, _safe_inverse(matrix)):
            if candidate is None:
                continue
            point_cam = candidate @ point_h
            z = float(point_cam[2])
            if z <= 1e-6:
                continue
            uv_h = k @ point_cam[:3]
            u = int(round(float(uv_h[0] / uv_h[2])))
            v = int(round(float(uv_h[1] / uv_h[2])))
            if 0 <= u < width and 0 <= v < height:
                return [u, v]
    return None


def _find_pose_xyz(value: Any) -> list[float] | None:
    import numpy as np

    preferred_names = (
        "obj_pose",
        "cube_pose",
        "object_pose",
        "target_pose",
        "goal_pose",
        "tcp_pose",
    )

    def visit(node: Any, key_hint: str = "") -> list[float] | None:
        if isinstance(node, dict):
            for key in preferred_names:
                if key in node:
                    found = _pose_array_to_xyz(node[key])
                    if found is not None:
                        return found
            for key, child in sorted(node.items(), key=lambda item: str(item[0])):
                found = visit(child, str(key))
                if found is not None:
                    return found
            return None
        if any(name in key_hint for name in preferred_names):
            try:
                array = np.asarray(node, dtype=float).reshape(-1)
                if array.size >= 3:
                    return [float(array[0]), float(array[1]), float(array[2])]
            except Exception:  # noqa: BLE001
                return None
        return None

    return visit(value)


def _pose_array_to_xyz(value: Any) -> list[float] | None:
    import numpy as np

    if isinstance(value, dict):
        for key in ("p", "position", "translation", "xyz", "pos"):
            if key in value:
                found = _pose_array_to_xyz(value[key])
                if found is not None:
                    return found
        for key in ("raw_pose", "pose", "matrix"):
            if key in value:
                found = _pose_array_to_xyz(value[key])
                if found is not None:
                    return found
        return None

    try:
        array = np.asarray(value, dtype=float)
    except Exception:  # noqa: BLE001
        return None
    if array.shape == (4, 4):
        return [float(array[0, 3]), float(array[1, 3]), float(array[2, 3])]
    flat = array.reshape(-1)
    if flat.size >= 3:
        return [float(flat[0]), float(flat[1]), float(flat[2])]
    return None


def _find_camera_matrix(obs: Any, camera_name: str, names: tuple[str, ...]) -> Any | None:
    for camera_params in _camera_data_candidates(obs, camera_name):
        for name in names:
            if isinstance(camera_params, dict) and name in camera_params:
                return camera_params[name]
    return None


def _candidate_camera_transforms(obs: Any, camera_name: str) -> list[Any]:
    transforms = []
    for camera_params in _camera_data_candidates(obs, camera_name):
        if not isinstance(camera_params, dict):
            continue
        for name in ("extrinsic_cv", "cam2world_gl", "cam2world", "extrinsic"):
            if name in camera_params:
                transforms.append(camera_params[name])
    return transforms


def _camera_data_candidates(obs: Any, camera_name: str) -> list[Any]:
    if not isinstance(obs, dict):
        return []
    params = obs.get("sensor_param", {}) if isinstance(obs, dict) else {}
    sensor_data = obs.get("sensor_data", {}) if isinstance(obs, dict) else {}
    candidates = []
    if isinstance(params, dict) and camera_name in params:
        candidates.append(params[camera_name])
    if isinstance(sensor_data, dict) and camera_name in sensor_data:
        candidates.append(sensor_data[camera_name])
    return candidates


def _camera_metadata_keys(obs: Any, camera_name: str) -> list[str]:
    keys: set[str] = set()
    for camera_params in _camera_data_candidates(obs, camera_name):
        if isinstance(camera_params, dict):
            keys.update(str(key) for key in camera_params)
    return sorted(keys)


def _safe_inverse(matrix: Any) -> Any | None:
    import numpy as np

    try:
        return np.linalg.inv(matrix)
    except Exception:  # noqa: BLE001
        return None


def _draw_overlay(pixels: Any, point_xy: list[int], label: str) -> Any:
    from PIL import Image, ImageDraw
    import numpy as np

    image = Image.fromarray(pixels).convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y = point_xy
    radius = max(5, min(image.size) // 32)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 255, 90), width=4)
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=(0, 255, 90), width=2)
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=(0, 255, 90), width=2)
    text = f"oracle {label}"
    draw.rectangle((4, 4, 4 + 8 * len(text), 26), fill=(245, 245, 240))
    draw.text((8, 9), text, fill=(25, 25, 25))
    return np.asarray(image, dtype=np.uint8)


def _rgb_tensor_to_hwc_uint8(value: Any) -> Any:
    import numpy as np

    try:
        import torch

        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
    except Exception:  # noqa: BLE001
        array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected RGB image with shape HxWxC, got {array.shape}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        max_value = float(array.max()) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _write_image(pixels: Any, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
