from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from physical_ai_agent.so101_training_config_schema import (
    TRAINING_CONFIG_DIR,
    SO101TrainingConfig,
    parse_so101_training_config,
)


HYDRA_CONFIG_DIR = Path("configs/so101/hydra")


class SO101HydraLauncherDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_dir: str
    host: str
    tensorboard_port: PositiveInt
    dashboard_port: PositiveInt
    tensorboard_enabled: bool
    tensorboard_tunnel_enabled: bool
    dashboard_enabled: bool
    gpu_monitor_enabled: bool
    progress_monitor_enabled: bool
    allow_incomplete_monitoring: bool
    hf_dataset_cache_root: str
    skip_hf_dataset_download: bool
    use_local_dataset_roots: bool
    hf_local_files_only: bool
    gpu_monitor_interval_s: PositiveFloat
    progress_monitor_interval_s: PositiveInt
    progress_monitor_batch_size: PositiveInt
    progress_monitor_validation_max_batches: PositiveInt
    runtime_platform: Literal["auto", "macos", "linux"]
    training_device: Literal["auto", "cpu", "mps", "cuda"]
    closed_loop_every_epochs: PositiveInt
    closed_loop_episodes: PositiveInt
    closed_loop_steps: PositiveInt
    closed_loop_env_id: str | None
    closed_loop_mujoco_gl: Literal["auto", "glfw", "egl", "osmesa"]
    max_monitored_checkpoints: PositiveInt
    closed_loop_policy: Literal["off", "periodic", "best_only", "best_or_periodic"]
    closed_loop_runner: Literal["auto", "picklift", "qwen_chain"]
    closed_loop_eval_skill_mode: str | None
    closed_loop_task_prompt: str | None
    closed_loop_action_contract_mode: (
        Literal[
            "processor",
            "legacy",
            "processor_dataset_clamp",
            "processor_gripper_snap",
            "processor_delta_q",
            "visual_servo_delta_q",
            "visual_servo_gt_delta_q",
        ]
        | None
    )
    closed_loop_record_rollout_gif: bool
    record_loop_artifacts: bool
    render_loop_media: bool
    loop_artifact_width: PositiveInt
    loop_artifact_height: PositiveInt
    loop_artifact_fps: PositiveInt
    loop_artifact_every_n_steps: PositiveInt
    qwen_model: str
    qwen_base_url: str | None
    qwen_api_key: str | None
    qwen_object: str | None
    qwen_env_object_color: str | None
    closed_loop_subgoal_chain_mode: Literal["off", "fixed", "valid-mask"]
    closed_loop_subgoal_sequence: str | None
    closed_loop_fixed_subgoal_chunks: PositiveInt
    closed_loop_valid_mask_checkpoint: str | None
    closed_loop_valid_mask_threshold: float = Field(ge=0, le=1)
    closed_loop_valid_mask_consecutive: PositiveInt
    closed_loop_policy_n_action_steps: PositiveInt
    closed_loop_policy_num_steps: PositiveInt
    validation_interval_steps: PositiveInt | None
    validation_interval_epochs: PositiveInt
    python: str


class SO101HydraTrainingEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    training_config: str = Field(description="Repo-relative path to a configs/so101/training/*.json file.")
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    launcher: SO101HydraLauncherDefaults
    training_args: list[str] = Field(
        default_factory=list,
        description="Default args forwarded to lerobot_train_so101_lightning.py when the CLI supplies none.",
    )

    def training_config_path(self, repo_root: Path) -> Path:
        path = Path(self.training_config)
        resolved = path if path.is_absolute() else repo_root / path
        training_root = (repo_root / TRAINING_CONFIG_DIR).resolve()
        try:
            resolved.resolve().relative_to(training_root)
        except ValueError as exc:
            raise ValueError(
                f"Hydra training_config must live under {TRAINING_CONFIG_DIR}: {self.training_config}"
            ) from exc
        return resolved


def load_so101_hydra_training_entry(
    config_name: str,
    *,
    repo_root: Path | None = None,
) -> SO101HydraTrainingEntry:
    root = repo_root or Path.cwd()
    config_dir = (root / HYDRA_CONFIG_DIR).resolve()
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise RuntimeError("hydra-core is required to load SO101 Hydra configs") from exc

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Hydra config {config_name!r} must resolve to a mapping")
    return SO101HydraTrainingEntry.model_validate(payload)


def load_so101_hydra_training_config(
    config_name: str,
    *,
    repo_root: Path | None = None,
) -> tuple[SO101HydraTrainingEntry, SO101TrainingConfig]:
    root = repo_root or Path.cwd()
    entry = load_so101_hydra_training_entry(config_name, repo_root=root)
    path = entry.training_config_path(root)
    import json

    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: training config must be a JSON object")
    return entry, parse_so101_training_config(payload)
