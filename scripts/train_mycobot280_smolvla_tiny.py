#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from scripts.run_mycobot_280_pi_smolvla_tiny_smoke import (
    DEFAULT_POLICY_PATH,
    DEFAULT_REPO_ID,
    audit_native_lerobot_dataset_root,
)


TinyTrainer = Callable[..., dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run, or explicitly block, a true tiny myCobot 280 SmolVLA "
            "fine-tune smoke. Unlike the supervised-loss smoke, this performs "
            "backward(), optimizer.step(), and checkpoint/log artifact writes."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--torch-seed", type=int, default=1001)
    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help="Fail instead of writing a blocked report when LeRobot/SmolVLA runtime imports are unavailable.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_mycobot280_smolvla_tiny_finetune(
        dataset_root=args.dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        policy_path=args.policy_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        device=args.device,
        local_files_only=args.local_files_only,
        torch_seed=args.torch_seed,
        require_runtime=args.require_runtime,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] in {"passed", "blocked"} else 1)


def run_mycobot280_smolvla_tiny_finetune(
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    policy_path: str,
    output_dir: Path,
    batch_size: int,
    steps: int,
    learning_rate: float,
    num_workers: int,
    device: str,
    local_files_only: bool,
    torch_seed: int = 1001,
    require_runtime: bool = False,
    trainer: TinyTrainer | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "tiny_finetune.json"
    dataset_audit = audit_native_lerobot_dataset_root(dataset_root)
    if dataset_audit["status"] != "passed":
        report = _blocked_report(
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            policy_path=policy_path,
            output_dir=output_dir,
            blocker="native LeRobotDataset root is incomplete",
            dataset_audit=dataset_audit,
        )
        _write_report(report_path, report)
        if require_runtime:
            raise RuntimeError(report["blocker"])
        return report

    if trainer is None:
        trainer = _run_actual_tiny_finetune

    try:
        training_report = trainer(
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            policy_path=policy_path,
            output_dir=output_dir,
            batch_size=int(batch_size),
            steps=int(steps),
            learning_rate=float(learning_rate),
            num_workers=int(num_workers),
            device=device,
            local_files_only=bool(local_files_only),
            torch_seed=int(torch_seed),
        )
    except Exception as exc:  # noqa: BLE001
        report = _blocked_report(
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            policy_path=policy_path,
            output_dir=output_dir,
            blocker=f"SmolVLA tiny fine-tune failed: {_short_error(exc)}",
            dataset_audit=dataset_audit,
        )
        _write_report(report_path, report)
        if require_runtime:
            raise
        return report

    status = "passed" if training_report.get("status") == "passed" else "blocked"
    report = {
        "operation": "train_mycobot280_smolvla_tiny",
        "status": status,
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "policy_path": policy_path,
        "output_dir": str(output_dir),
        "batch_size": int(batch_size),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "num_workers": int(num_workers),
        "device": device,
        "local_files_only": bool(local_files_only),
        "torch_seed": int(torch_seed),
        "dataset_audit": dataset_audit,
        "training_report": training_report,
        "claim_boundary": (
            "This is a true tiny optimizer-step fine-tune smoke. It proves "
            "dataset/model/training/checkpoint plumbing only; it does not prove "
            "closed-loop myCobot task success or publication-level learning gain."
        ),
    }
    _write_report(report_path, report)
    return report


def _run_actual_tiny_finetune(
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    policy_path: str,
    output_dir: Path,
    batch_size: int,
    steps: int,
    learning_rate: float,
    num_workers: int,
    device: str,
    local_files_only: bool,
    torch_seed: int,
) -> dict[str, Any]:
    started = perf_counter()
    if steps <= 0:
        raise ValueError("--steps must be positive for a true fine-tune smoke")

    import torch
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from physical_ai_agent.policies.mycobot280_smolvla_contract import (
        CONTRACT_FILENAME,
        make_mycobot280_pre_post_processors,
    )
    from physical_ai_agent.policies.smolvla_real import (
        _load_pretrained_policy,
        _policy_device_metadata,
    )

    torch.manual_seed(int(torch_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(torch_seed))

    log_path = output_dir / "train.log"
    tensorboard_dir = output_dir / "tensorboard"
    checkpoint_root = output_dir / "checkpoints" / "latest"
    checkpoint_model_dir = checkpoint_root / "pretrained_model"
    optimizer_state_path = checkpoint_root / "optimizer_state.pt"
    training_state_path = checkpoint_root / "training_state.json"
    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)

    policy = _load_pretrained_policy(
        model_id=policy_path,
        local_files_only=local_files_only,
        device=device,
    )
    selected_device = str(_policy_device_metadata(policy).get("device_selected") or device)
    if selected_device != "auto":
        if hasattr(policy, "config"):
            policy.config.device = selected_device
        if hasattr(policy, "to"):
            policy.to(selected_device)
    policy.train()

    preprocessor, postprocessor, contract_report = make_mycobot280_pre_post_processors(
        policy=policy,
        dataset_meta=metadata,
        policy_path=policy_path,
        selected_device=selected_device,
    )
    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
    )
    if int(dataset.num_frames) <= 0:
        raise RuntimeError("Native LeRobotDataset has no frames")

    data_generator = torch.Generator()
    data_generator.manual_seed(int(torch_seed))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        generator=data_generator,
    )
    trainable_params = [param for param in policy.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("SmolVLA policy exposes no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(learning_rate))
    writer, tensorboard_error = _try_summary_writer(tensorboard_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []
    step_records: list[dict[str, Any]] = []
    data_iter = iter(dataloader)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            json.dumps(
                {
                    "event": "start",
                    "operation": "train_mycobot280_smolvla_tiny",
                    "dataset_root": str(dataset_root),
                    "dataset_repo_id": dataset_repo_id,
                    "policy_path": policy_path,
                    "steps": int(steps),
                    "batch_size": int(batch_size),
                },
                sort_keys=True,
            )
            + "\n"
        )
        for step_index in range(int(steps)):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            torch.manual_seed(int(torch_seed) + step_index)
            batch = preprocessor(batch)
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).detach().cpu().item())
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            record = {
                "event": "step",
                "step": step_index + 1,
                "loss": loss_value,
                "grad_norm_clipped": grad_norm,
                "loss_dict": _loss_dict_to_scalars(loss_dict),
            }
            step_records.append(record)
            log_file.write(json.dumps(record, sort_keys=True) + "\n")
            log_file.flush()
            if writer is not None:
                writer.add_scalar("train/loss", loss_value, step_index + 1)
                writer.add_scalar("train/grad_norm_clipped", grad_norm, step_index + 1)

    if writer is not None:
        writer.flush()
        writer.close()

    _save_policy_checkpoint(
        policy,
        checkpoint_model_dir,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    contract_config_path = checkpoint_model_dir / CONTRACT_FILENAME
    contract_config_path.write_text(
        json.dumps(contract_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    preprocessor_config_path = checkpoint_model_dir / "policy_preprocessor.json"
    postprocessor_config_path = checkpoint_model_dir / "policy_postprocessor.json"
    missing_processor_configs = [
        str(path)
        for path in (preprocessor_config_path, postprocessor_config_path, contract_config_path)
        if not path.is_file()
    ]
    if missing_processor_configs:
        raise RuntimeError(
            "Checkpoint is not reloadable because processor configs are missing: "
            + ", ".join(missing_processor_configs)
        )
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": int(steps),
            "learning_rate": float(learning_rate),
        },
        optimizer_state_path,
    )
    training_state = {
        "operation": "train_mycobot280_smolvla_tiny",
        "status": "passed",
        "steps": int(steps),
        "loss_initial": float(losses[0]),
        "loss_final": float(losses[-1]),
        "losses": losses,
        "checkpoint_model_dir": str(checkpoint_model_dir),
        "preprocessor_config_path": str(preprocessor_config_path),
        "postprocessor_config_path": str(postprocessor_config_path),
        "contract": contract_report,
        "contract_config_path": str(contract_config_path),
        "dataloader_shuffle": True,
        "optimizer_state_path": str(optimizer_state_path),
        "train_log": str(log_path),
        "tensorboard_dir": str(tensorboard_dir),
        "tensorboard_error": tensorboard_error,
    }
    training_state_path.write_text(json.dumps(training_state, indent=2, sort_keys=True), encoding="utf-8")

    return {
        **training_state,
        "dataset_num_frames": int(dataset.num_frames),
        "dataset_num_episodes": int(dataset.num_episodes),
        "batch_size": int(batch_size),
        "device": _policy_device_metadata(policy),
        "delta_timestamps": delta_timestamps,
        "step_records": step_records,
        "duration_s": round(perf_counter() - started, 4),
    }


def _save_policy_checkpoint(
    policy: Any,
    checkpoint_model_dir: Path,
    *,
    preprocessor: Any | None = None,
    postprocessor: Any | None = None,
) -> None:
    checkpoint_model_dir.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(policy, "save_pretrained", None)
    if callable(save_pretrained):
        save_pretrained(checkpoint_model_dir)
    else:
        import torch

        torch.save(policy.state_dict(), checkpoint_model_dir / "pytorch_model.bin")

    for processor_name, processor in (
        ("preprocessor", preprocessor),
        ("postprocessor", postprocessor),
    ):
        if processor is None:
            continue
        save_processor = getattr(processor, "save_pretrained", None)
        if not callable(save_processor):
            raise RuntimeError(f"SmolVLA {processor_name} cannot be saved with save_pretrained()")
        save_processor(checkpoint_model_dir)


def _try_summary_writer(tensorboard_dir: Path) -> tuple[Any | None, str | None]:
    try:
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(tensorboard_dir)), None
    except Exception as exc:  # noqa: BLE001
        return None, _short_error(exc)


def _loss_dict_to_scalars(loss_dict: Any) -> dict[str, float]:
    if not isinstance(loss_dict, dict):
        return {}
    scalars: dict[str, float] = {}
    for key, value in loss_dict.items():
        try:
            scalars[str(key)] = float(value.detach().cpu().item())
        except AttributeError:
            try:
                scalars[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        except (RuntimeError, ValueError):
            continue
    return scalars


def _blocked_report(
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    policy_path: str,
    output_dir: Path,
    blocker: str,
    dataset_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "operation": "train_mycobot280_smolvla_tiny",
        "status": "blocked",
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "policy_path": policy_path,
        "output_dir": str(output_dir),
        "blocker": blocker,
        "dataset_audit": dataset_audit,
        "install_command": "sh scripts/install/local_install.sh --checkpoint 05-06",
        "approval_required": True,
        "native_conversion_command": (
            "PYTHONPATH=src:. python3 scripts/convert_mycobot_280_pi_adaptive_jsonl_to_lerobot.py "
            "--source-root _workspace/mycobot280_lerobot/ground_pickup_tiny_smoke "
            "--output-root _workspace/mycobot280_lerobot/ground_pickup_tiny_smoke_native "
            "--repo-id physical-ai-agent/mycobot-280-ground-pickup-tiny-smoke "
            "--require-lerobot"
        ),
        "next_step": (
            "After approval, use the LeRobot/SmolVLA runtime and native dataset to rerun "
            "this script with --require-runtime so backward/optimizer/checkpoint artifacts are produced."
        ),
        "claim_boundary": "No SmolVLA fine-tune smoke was completed.",
    }


def _write_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _short_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text.replace("\n", " ")[:800]


if __name__ == "__main__":
    main()
