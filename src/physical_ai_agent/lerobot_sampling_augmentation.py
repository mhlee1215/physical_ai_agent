from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SamplingAugmentationConfig:
    state_jitter_std: float = 0.0
    state_jitter_arm_only: bool = True
    state_dropout_prob: float = 0.0
    state_dropout_keep_gripper: bool = True
    image_camera_dropout_prob: float = 0.0
    image_patch_dropout_prob: float = 0.0
    image_patch_mask_ratio: float = 0.0
    image_blur_prob: float = 0.0
    image_blur_kernel_size: int = 5
    image_motion_blur_prob: float = 0.0
    image_motion_blur_kernel_size: int = 7
    image_noise_std: float = 0.0
    image_color_jitter: bool = False
    image_color_jitter_strength: float = 0.08
    image_sharpness_jitter: bool = False
    image_affine_degrees: float = 0.0
    image_affine_translate: float = 0.0
    gpu_image_augmentation: bool = False
    state_key: str = "observation.state"
    action_key: str = "action"
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "SamplingAugmentationConfig":
        state_jitter_std = float(os.environ.get("SO101_STATE_JITTER_STD", "0.0"))
        state_jitter_arm_only = os.environ.get("SO101_STATE_JITTER_ARM_ONLY", "1") not in {
            "0",
            "false",
            "False",
        }
        gpu_image_augmentation = os.environ.get("SO101_GPU_IMAGE_AUGMENTATION", "0") in {
            "1",
            "true",
            "True",
        }
        state_dropout_prob = float(os.environ.get("SO101_STATE_DROPOUT_PROB", "0.0"))
        image_camera_dropout_prob = float(os.environ.get("SO101_IMAGE_CAMERA_DROPOUT_PROB", "0.0"))
        image_patch_dropout_prob = float(os.environ.get("SO101_IMAGE_PATCH_DROPOUT_PROB", "0.0"))
        image_patch_mask_ratio = float(os.environ.get("SO101_IMAGE_PATCH_MASK_RATIO", "0.0"))
        image_blur_prob = float(os.environ.get("SO101_IMAGE_BLUR_PROB", "0.0"))
        image_blur_kernel_size = int(os.environ.get("SO101_IMAGE_BLUR_KERNEL_SIZE", "5"))
        image_motion_blur_prob = float(os.environ.get("SO101_IMAGE_MOTION_BLUR_PROB", "0.0"))
        image_motion_blur_kernel_size = int(os.environ.get("SO101_IMAGE_MOTION_BLUR_KERNEL_SIZE", "7"))
        image_noise_std = float(os.environ.get("SO101_IMAGE_NOISE_STD", "0.0"))
        image_color_jitter = os.environ.get("SO101_IMAGE_COLOR_JITTER", "0") in {
            "1",
            "true",
            "True",
        }
        image_color_jitter_strength = float(os.environ.get("SO101_IMAGE_COLOR_JITTER_STRENGTH", "0.08"))
        image_sharpness_jitter = os.environ.get("SO101_IMAGE_SHARPNESS_JITTER", "0") in {
            "1",
            "true",
            "True",
        }
        image_affine_degrees = float(os.environ.get("SO101_IMAGE_AFFINE_DEGREES", "0.0"))
        image_affine_translate = float(os.environ.get("SO101_IMAGE_AFFINE_TRANSLATE", "0.0"))
        state_dropout_keep_gripper = os.environ.get("SO101_STATE_DROPOUT_KEEP_GRIPPER", "1") not in {
            "0",
            "false",
            "False",
        }
        return cls(
            state_jitter_std=state_jitter_std,
            state_jitter_arm_only=state_jitter_arm_only,
            state_dropout_prob=state_dropout_prob,
            state_dropout_keep_gripper=state_dropout_keep_gripper,
            image_camera_dropout_prob=image_camera_dropout_prob,
            image_patch_dropout_prob=image_patch_dropout_prob,
            image_patch_mask_ratio=image_patch_mask_ratio,
            image_blur_prob=image_blur_prob,
            image_blur_kernel_size=image_blur_kernel_size,
            image_motion_blur_prob=image_motion_blur_prob,
            image_motion_blur_kernel_size=image_motion_blur_kernel_size,
            image_noise_std=image_noise_std,
            image_color_jitter=image_color_jitter,
            image_color_jitter_strength=image_color_jitter_strength,
            image_sharpness_jitter=image_sharpness_jitter,
            image_affine_degrees=image_affine_degrees,
            image_affine_translate=image_affine_translate,
            gpu_image_augmentation=gpu_image_augmentation,
            enabled=(
                state_jitter_std > 0.0
                or state_dropout_prob > 0.0
                or image_camera_dropout_prob > 0.0
                or image_patch_dropout_prob > 0.0
                or image_patch_mask_ratio > 0.0
                or image_blur_prob > 0.0
                or image_motion_blur_prob > 0.0
                or image_noise_std > 0.0
                or image_color_jitter
                or image_sharpness_jitter
                or image_affine_degrees > 0.0
                or image_affine_translate > 0.0
                or gpu_image_augmentation
            ),
        )


class SamplingAugmentedDataset:
    """Dataset wrapper that applies SO101 augmentation when a sample is fetched."""

    def __init__(self, dataset: Any, config: SamplingAugmentationConfig) -> None:
        self.dataset = dataset
        self.config = config

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        if self.config.enabled:
            self._augment_state(item)
        return item

    def _augment_state(self, item: dict[str, Any]) -> None:
        import torch

        value = item.get(self.config.state_key)
        if not torch.is_tensor(value):
            return
        item[self.config.state_key] = augment_state_tensor(value, self.config)

    def __getattr__(self, name: str) -> Any:
        if name == "dataset":
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)


class PredecodedImageCacheDataset:
    """LeRobotDataset wrapper that reads image tensors from a local uint8 memmap cache."""

    def __init__(self, dataset: Any, cache_dir: Path) -> None:
        self.dataset = dataset
        self.cache_dir = cache_dir
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"SO101 image cache manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.image_keys = list(self.manifest["image_keys"])
        self.cache_arrays = self._load_arrays(cache_dir, self.image_keys)
        self.tabular_dataset = dataset.hf_dataset.remove_columns(self.image_keys)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        reader = self.dataset._ensure_reader()
        if reader.hf_dataset is None:
            reader.load_and_activate()
        item = dict(self.tabular_dataset[index])
        ep_index = _scalar_int(item["episode_index"])
        abs_index = _scalar_int(item["index"])

        query_indices = None
        if reader.delta_indices is not None:
            query_indices, padding = reader._get_query_indices(abs_index, ep_index)
            item = {**item, **padding, **self._query_cached_dataset(reader, query_indices)}
            for key in self.image_keys:
                if key not in item:
                    item[key] = self._cached_image_tensor(abs_index, key)
        else:
            for key in self.image_keys:
                item[key] = self._cached_image_tensor(abs_index, key)

        task_index = _scalar_int(item["task_index"])
        item["task"] = self.dataset.meta.tasks.iloc[task_index].name
        if "subtask_index" in self.dataset.meta.features and self.dataset.meta.subtasks is not None:
            subtask_index = _scalar_int(item["subtask_index"])
            item["subtask"] = self.dataset.meta.subtasks.iloc[subtask_index].name
        return item

    def _query_cached_dataset(self, reader: Any, query_indices: dict[str, list[int]]) -> dict[str, Any]:
        import torch

        result = {}
        for key, query_index in query_indices.items():
            relative_indices = (
                query_index
                if reader._absolute_to_relative_idx is None
                else [reader._absolute_to_relative_idx[idx] for idx in query_index]
            )
            if key in self.image_keys:
                result[key] = torch.stack([self._cached_image_tensor(abs_index, key) for abs_index in query_index])
                continue
            try:
                result[key] = torch.stack(self.tabular_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.tabular_dataset[relative_indices][key])
        return result

    def _cached_image_tensor(self, abs_index: int, key: str) -> Any:
        import numpy as np
        import torch

        cached = self.cache_arrays[key][abs_index]
        return torch.from_numpy(np.array(cached, copy=True)).to(torch.float32).div_(255.0)

    def _load_arrays(self, cache_dir: Path, image_keys: list[str]) -> dict[str, Any]:
        import numpy as np

        arrays = {}
        for key in image_keys:
            name = key.replace(".", "_") + ".npy"
            arrays[key] = np.load(cache_dir / name, mmap_mode="r")
        return arrays

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def _scalar_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def patch_lerobot_train_make_dataset(config: SamplingAugmentationConfig) -> None:
    """Patch LeRobot train.py so augmentation happens at dataset sample time."""

    import lerobot.scripts.lerobot_train as lerobot_train

    original_make_dataset = lerobot_train.make_dataset

    def make_dataset_with_sampling_augmentation(cfg: Any) -> Any:
        dataset = original_make_dataset(cfg)
        cache_dir = os.environ.get("SO101_IMAGE_CACHE_DIR")
        if cache_dir:
            dataset = PredecodedImageCacheDataset(dataset, Path(cache_dir))
        if not config.enabled or config.gpu_image_augmentation:
            return dataset
        return SamplingAugmentedDataset(dataset, config)

    lerobot_train.make_dataset = make_dataset_with_sampling_augmentation


def patch_lerobot_train_gpu_augmentation(config: SamplingAugmentationConfig) -> None:
    """Patch LeRobot train.py so SO101 augmentation runs after the batch is moved to device."""

    if not config.enabled or not config.gpu_image_augmentation:
        return

    import lerobot.scripts.lerobot_train as lerobot_train

    original_update_policy = lerobot_train.update_policy

    def update_policy_with_gpu_augmentation(train_tracker: Any, policy: Any, batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        augment_batch_on_device(batch, config)
        return original_update_policy(train_tracker, policy, batch, *args, **kwargs)

    lerobot_train.update_policy = update_policy_with_gpu_augmentation


def augment_batch_on_device(batch: dict[str, Any], config: SamplingAugmentationConfig) -> None:
    _augment_state_on_device(batch, config)
    _augment_images_on_device(batch, config)


def _augment_state_on_device(batch: dict[str, Any], config: SamplingAugmentationConfig) -> None:
    import torch

    value = batch.get(config.state_key)
    if not torch.is_tensor(value):
        return
    batch[config.state_key] = augment_state_tensor(value, config)


def augment_state_tensor(value: Any, config: SamplingAugmentationConfig) -> Any:
    import torch

    result = value.to(torch.float32)
    if config.state_jitter_std > 0.0:
        noise = torch.randn_like(result, dtype=torch.float32) * float(config.state_jitter_std)
        if config.state_jitter_arm_only and noise.shape[-1] >= 6:
            noise = noise.clone()
            noise[..., 5] = 0.0
        result = result + noise
    if config.state_dropout_prob > 0.0:
        mask = torch.rand_like(result, dtype=torch.float32) >= float(config.state_dropout_prob)
        if config.state_dropout_keep_gripper and mask.shape[-1] >= 6:
            mask = mask.clone()
            mask[..., 5] = True
        result = result * mask.to(result.dtype)
    return result.to(value.dtype)


def _augment_images_on_device(batch: dict[str, Any], config: SamplingAugmentationConfig) -> None:
    import torch
    import torch.nn.functional as F

    for key, value in list(batch.items()):
        if not key.startswith("observation.images.") or not torch.is_tensor(value):
            continue
        normalized = _images_to_nchw_batch(value)
        if normalized is None:
            continue
        image, restore = normalized
        if image.ndim != 4 or image.shape[1] not in (1, 3, 4):
            continue
        image = image.to(torch.float32).clamp(0.0, 1.0)
        image = _camera_dropout(image, config.image_camera_dropout_prob)
        image = _patch_dropout(image, config.image_patch_dropout_prob)
        if config.image_color_jitter:
            image = _color_jitter(image, strength=config.image_color_jitter_strength)
        image = _random_blur(image, config.image_blur_prob, config.image_blur_kernel_size)
        image = _motion_blur(image, config.image_motion_blur_prob, config.image_motion_blur_kernel_size)
        image = _additive_noise(image, config.image_noise_std)
        if config.image_sharpness_jitter:
            image = _sharpness_jitter(image)
        image, theta = _affine_jitter_with_theta(image, F, config)
        _transform_visual_servo_labels_by_affine(batch, key, theta)
        image = _patch_mask_ratio(image, config.image_patch_mask_ratio)
        batch[key] = restore(image.to(value.dtype).clamp(0.0, 1.0))


def _images_to_nchw_batch(value: Any) -> tuple[Any, Any] | None:
    import math

    if value.ndim < 3:
        return None
    shape = tuple(value.shape)
    if shape[-3] in (1, 3, 4):
        prefix = shape[:-3]
        channels, height, width = shape[-3:]
        batch_size = int(math.prod(prefix)) if prefix else 1
        flat = value.reshape(batch_size, channels, height, width)

        def restore(image: Any) -> Any:
            return image.reshape(*prefix, channels, height, width) if prefix else image.reshape(channels, height, width)

        return flat, restore
    if shape[-1] in (1, 3, 4):
        prefix = shape[:-3]
        height, width, channels = shape[-3:]
        batch_size = int(math.prod(prefix)) if prefix else 1
        flat = value.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()

        def restore(image: Any) -> Any:
            restored = image.permute(0, 2, 3, 1).contiguous()
            return restored.reshape(*prefix, height, width, channels) if prefix else restored.reshape(height, width, channels)

        return flat, restore
    return None


def _dropout_tensor(value: Any, probability: float) -> Any:
    import torch

    if probability <= 0.0:
        return value
    mask = torch.rand_like(value, dtype=torch.float32) >= float(probability)
    return (value.to(torch.float32) * mask.to(torch.float32)).to(value.dtype)


def _camera_dropout(image: Any, probability: float) -> Any:
    import torch

    probability = float(probability)
    if probability <= 0.0:
        return image
    batch_size = image.shape[0]
    mask = (torch.rand((batch_size, 1, 1, 1), device=image.device) >= probability).to(image.dtype)
    return image * mask


def _patch_dropout(image: Any, probability: float) -> Any:
    import torch

    probability = float(probability)
    if probability <= 0.0:
        return image
    batch_size, channels, height, width = image.shape
    patch_h = max(1, height // 8)
    patch_w = max(1, width // 8)
    result = image.clone()
    drop_mask = torch.rand((batch_size,), device=image.device) < probability
    for index in torch.nonzero(drop_mask, as_tuple=False).flatten().tolist():
        top = int(torch.randint(0, max(1, height - patch_h + 1), (1,), device=image.device).item())
        left = int(torch.randint(0, max(1, width - patch_w + 1), (1,), device=image.device).item())
        result[index, :, top : top + patch_h, left : left + patch_w] = 0.0
    return result


def _patch_mask_ratio(image: Any, ratio: float) -> Any:
    import torch

    ratio = float(ratio)
    if ratio <= 0.0:
        return image
    ratio = min(ratio, 1.0)
    batch_size, channels, height, width = image.shape
    grid = 8
    patch_h = max(1, height // grid)
    patch_w = max(1, width // grid)
    patches = grid * grid
    masked_patches = max(1, int(round(patches * ratio)))
    result = image.clone()
    for index in range(batch_size):
        selected = torch.randperm(patches, device=image.device)[:masked_patches]
        rows = torch.div(selected, grid, rounding_mode="floor")
        cols = selected % grid
        for row, col in zip(rows.tolist(), cols.tolist()):
            top = int(row) * patch_h
            left = int(col) * patch_w
            result[index, :, top : min(height, top + patch_h), left : min(width, left + patch_w)] = 0.0
    return result


def _color_jitter(image: Any, *, strength: float = 0.08) -> Any:
    import torch

    strength = max(0.0, float(strength))
    if strength <= 0.0:
        return image
    brightness_span = min(0.2, strength)
    contrast_span = min(0.2, strength)
    saturation_span = min(0.25, strength)
    batch_size = image.shape[0]
    device = image.device
    dtype = image.dtype
    brightness = _rand_factor(batch_size, device, dtype, 1.0 - brightness_span, 1.0 + brightness_span)
    contrast = _rand_factor(batch_size, device, dtype, 1.0 - contrast_span, 1.0 + contrast_span)
    saturation = _rand_factor(batch_size, device, dtype, 1.0 - saturation_span, 1.0 + saturation_span)

    image = image * brightness
    mean = image.mean(dim=(2, 3), keepdim=True)
    image = (image - mean) * contrast + mean
    gray = (image[:, 0:1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114)
    image = (image - gray) * saturation + gray
    return image.clamp(0.0, 1.0)


def _random_blur(image: Any, probability: float, kernel_size: int) -> Any:
    import torch
    import torch.nn.functional as F

    probability = float(probability)
    if probability <= 0.0:
        return image
    batch_size, channels, _height, _width = image.shape
    mask = (torch.rand((batch_size, 1, 1, 1), device=image.device) < probability).to(image.dtype)
    if not bool(mask.any().item()):
        return image
    size = max(3, int(kernel_size))
    if size % 2 == 0:
        size += 1
    radius = size // 2
    coords = torch.arange(size, device=image.device, dtype=image.dtype) - radius
    sigma = torch.empty((batch_size, 1, 1, 1), device=image.device, dtype=image.dtype).uniform_(0.6, 1.4)
    kernel_1d = torch.exp(-0.5 * (coords.view(1, 1, 1, size) / sigma).pow(2))
    kernel_1d = kernel_1d / kernel_1d.sum(dim=-1, keepdim=True)
    padded = F.pad(image, (radius, radius, 0, 0), mode="replicate")
    horizontal = torch.empty_like(image)
    for index in range(batch_size):
        kernel = kernel_1d[index].view(1, 1, 1, size).repeat(channels, 1, 1, 1)
        horizontal[index : index + 1] = F.conv2d(padded[index : index + 1], kernel, groups=channels)
    padded = F.pad(horizontal, (0, 0, radius, radius), mode="replicate")
    blurred = torch.empty_like(image)
    for index in range(batch_size):
        kernel = kernel_1d[index].view(1, 1, size, 1).repeat(channels, 1, 1, 1)
        blurred[index : index + 1] = F.conv2d(padded[index : index + 1], kernel, groups=channels)
    return torch.where(mask.bool(), blurred, image).clamp(0.0, 1.0)


def _motion_blur(image: Any, probability: float, kernel_size: int) -> Any:
    import torch
    import torch.nn.functional as F

    probability = float(probability)
    if probability <= 0.0:
        return image
    batch_size, channels, _height, _width = image.shape
    mask = (torch.rand((batch_size, 1, 1, 1), device=image.device) < probability).to(image.dtype)
    if not bool(mask.any().item()):
        return image
    size = max(3, int(kernel_size))
    if size % 2 == 0:
        size += 1
    radius = size // 2
    blurred = torch.empty_like(image)
    horizontal_kernel = torch.ones((channels, 1, 1, size), device=image.device, dtype=image.dtype) / float(size)
    vertical_kernel = torch.ones((channels, 1, size, 1), device=image.device, dtype=image.dtype) / float(size)
    diagonal_kernel = torch.eye(size, device=image.device, dtype=image.dtype).view(1, 1, size, size)
    diagonal_kernel = (diagonal_kernel / diagonal_kernel.sum()).repeat(channels, 1, 1, 1)
    anti_diagonal_kernel = torch.flip(diagonal_kernel, dims=(2,))
    padded_line = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    padded_h = F.pad(image, (radius, radius, 0, 0), mode="replicate")
    padded_v = F.pad(image, (0, 0, radius, radius), mode="replicate")
    for index in range(batch_size):
        direction = int(torch.randint(0, 4, (1,), device=image.device).item())
        if direction == 0:
            blurred[index : index + 1] = F.conv2d(padded_h[index : index + 1], horizontal_kernel, groups=channels)
        elif direction == 1:
            blurred[index : index + 1] = F.conv2d(padded_v[index : index + 1], vertical_kernel, groups=channels)
        elif direction == 2:
            blurred[index : index + 1] = F.conv2d(padded_line[index : index + 1], diagonal_kernel, groups=channels)
        else:
            blurred[index : index + 1] = F.conv2d(padded_line[index : index + 1], anti_diagonal_kernel, groups=channels)
    return torch.where(mask.bool(), blurred, image).clamp(0.0, 1.0)


def _additive_noise(image: Any, std: float) -> Any:
    import torch

    std = float(std)
    if std <= 0.0:
        return image
    noise = torch.randn_like(image, dtype=torch.float32) * std
    return (image.to(torch.float32) + noise).clamp(0.0, 1.0).to(image.dtype)


def _sharpness_jitter(image: Any) -> Any:
    import torch
    import torch.nn.functional as F

    batch_size = image.shape[0]
    channels = image.shape[1]
    device = image.device
    dtype = image.dtype
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    blurred = F.conv2d(image, kernel, padding=1, groups=channels)
    factor = _rand_factor(batch_size, device, dtype, 0.5, 1.5)
    return (blurred + (image - blurred) * factor).clamp(0.0, 1.0)


def _affine_jitter(image: Any, functional: Any, config: SamplingAugmentationConfig) -> Any:
    image, _theta = _affine_jitter_with_theta(image, functional, config)
    return image


def _affine_jitter_with_theta(image: Any, functional: Any, config: SamplingAugmentationConfig) -> tuple[Any, Any | None]:
    import math
    import torch

    max_degrees = float(config.image_affine_degrees)
    max_translate = float(config.image_affine_translate)
    if max_degrees <= 0.0 and max_translate <= 0.0:
        return image, None
    batch_size, channels, height, width = image.shape
    device = image.device
    dtype = image.dtype
    degrees = (torch.rand(batch_size, device=device, dtype=dtype) * (2.0 * max_degrees) - max_degrees) * (math.pi / 180.0)
    cos = torch.cos(degrees)
    sin = torch.sin(degrees)
    translate_x = torch.rand(batch_size, device=device, dtype=dtype) * (2.0 * max_translate) - max_translate
    translate_y = torch.rand(batch_size, device=device, dtype=dtype) * (2.0 * max_translate) - max_translate
    theta = torch.zeros((batch_size, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 0, 2] = translate_x
    theta[:, 1, 2] = translate_y
    grid = functional.affine_grid(theta, size=(batch_size, channels, height, width), align_corners=False)
    image = functional.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="zeros" if image.device.type == "mps" else "border",
        align_corners=False,
    )
    return image, theta


def _transform_visual_servo_labels_by_affine(batch: dict[str, Any], image_key: str, theta: Any | None) -> None:
    if theta is None:
        return
    camera = image_key.rsplit(".", maxsplit=1)[-1]
    label_key = f"visual_servo.{camera}"
    visible_key = f"{label_key}_visible"
    target = batch.get(label_key)
    visible = batch.get(visible_key)
    if target is None or visible is None:
        return
    import math
    import torch

    if not torch.is_tensor(target) or target.ndim != 2 or target.shape[-1] < 3:
        return
    target = target.to(device=theta.device, dtype=theta.dtype)
    visible_mask = visible.to(device=theta.device).bool()
    if target.shape[0] != theta.shape[0]:
        return

    rotation = theta[:, :, :2]
    translation = theta[:, :, 2]
    # affine_grid theta maps output coords -> input coords, so labels need inverse theta.
    inv_rotation = rotation.transpose(1, 2)
    point = torch.bmm(inv_rotation, (target[:, :2] - translation).unsqueeze(-1)).squeeze(-1)
    angle = target[:, 2] * (math.pi * 0.5)
    rotation_angle = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    angle = torch.remainder(angle - rotation_angle + (math.pi * 0.5), math.pi) - (math.pi * 0.5)

    transformed = target.clone()
    transformed[:, :2] = point.clamp(-1.0, 1.0)
    transformed[:, 2] = (angle / (math.pi * 0.5)).clamp(-1.0, 1.0)
    batch[label_key] = torch.where(visible_mask.unsqueeze(-1), transformed, target).to(batch[label_key].dtype)


def _rand_factor(batch_size: int, device: Any, dtype: Any, low: float, high: float) -> Any:
    import torch

    return (torch.rand((batch_size, 1, 1, 1), device=device, dtype=dtype) * (high - low)) + low


def write_sampling_augmentation_report(path: Path, config: SamplingAugmentationConfig, argv: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": "so101_lerobot_sampling_augmentation",
        "config": asdict(config),
        "argv": argv,
        "notes": [
            "When gpu_image_augmentation=true, image augmentation runs after LeRobot preprocessing on the training device.",
            "The default SO101 image augmentation recipe mirrors the LeRobot SmolVLA image_transforms list: color jitter, sharpness jitter, affine jitter, and patch masking when enabled.",
            "When gpu_image_augmentation=false, image augmentation is delegated to LeRobot dataset.image_transforms and is applied in __getitem__.",
            "State augmentation adds observation.state noise during training; stored teacher data is unchanged.",
            "Actions are not noised.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
