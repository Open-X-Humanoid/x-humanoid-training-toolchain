"""WandB logging for ``xhum.train.train_multi`` (no ``TrainPipelineConfig`` required)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from lerobot.rl.wandb_utils import get_safe_wandb_artifact_name, get_wandb_run_id_from_filesystem
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from termcolor import colored

logger = logging.getLogger(__name__)


def _truncate_tag(tag: str, max_length: int = 64) -> str:
    return tag if len(tag) <= max_length else tag[:max_length]


def build_wandb_tags(
    *,
    policy_type: str,
    seed: int,
    dataset_repo_ids: list[str],
    truncate: bool = True,
) -> list[str]:
    tags = [
        f"policy:{policy_type}",
        f"seed:{seed}",
        f"datasets:{len(dataset_repo_ids)}",
    ]
    for repo_id in dataset_repo_ids:
        tags.append(f"dataset:{repo_id}")
    if truncate:
        tags = [_truncate_tag(t) for t in tags]
    return tags


def build_wandb_group(policy_type: str, seed: int) -> str:
    return f"policy:{policy_type}-seed:{seed}"


class MultiTrainWandbLogger:
    """Thin wrapper around ``wandb.init`` / ``wandb.log`` for multi-dataset training."""

    def __init__(
        self,
        wandb_cfg: dict,
        *,
        job_name: str,
        output_dir: Path,
        policy_type: str,
        dataset_repo_ids: list[str],
        seed: int,
        resume: bool,
        full_config: dict,
    ) -> None:
        self.cfg = wandb_cfg
        self.log_dir = output_dir
        self.job_name = job_name
        self._group = build_wandb_group(policy_type, seed)

        os.environ["WANDB_SILENT"] = "True"
        import wandb

        wandb_run_id = wandb_cfg.get("run_id")
        if wandb_run_id is None and resume:
            try:
                wandb_run_id = get_wandb_run_id_from_filesystem(self.log_dir)
            except RuntimeError:
                logger.warning("resume=True but no prior wandb run found under %s", self.log_dir)
                wandb_run_id = None

        run_name = wandb_cfg.get("run_name") or job_name
        tags = (
            build_wandb_tags(
                policy_type=policy_type,
                seed=seed,
                dataset_repo_ids=dataset_repo_ids,
            )
            if wandb_cfg.get("add_tags", True)
            else None
        )

        mode = wandb_cfg.get("mode") or "online"
        if mode not in ("online", "offline", "disabled"):
            mode = "online"

        wandb.init(
            id=wandb_run_id,
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            name=run_name,
            group=wandb_cfg.get("group") or self._group,
            notes=wandb_cfg.get("notes"),
            tags=tags,
            dir=str(self.log_dir),
            config=full_config,
            save_code=False,
            job_type="train",
            resume="must" if resume and wandb_run_id else None,
            mode=mode,
        )
        self._wandb = wandb
        logger.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logger.info("Track this run --> %s", colored(wandb.run.get_url(), "yellow", attrs=["bold"]))

    def log_dict(self, metrics: dict[str, Any], step: int) -> None:
        for key, value in metrics.items():
            if not isinstance(value, (int, float, str)):
                continue
            self._wandb.log({f"train/{key}": value}, step=step)

    def log_policy(self, checkpoint_dir: Path) -> None:
        if self.cfg.get("disable_artifact"):
            return

        from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

        step_id = checkpoint_dir.name
        artifact_name = get_safe_wandb_artifact_name(f"{self._group}-{step_id}")
        artifact = self._wandb.Artifact(artifact_name, type="model")
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

        adapter_model_file = pretrained_model_dir / "adapter_model.safetensors"
        standard_model_file = pretrained_model_dir / SAFETENSORS_SINGLE_FILE

        if adapter_model_file.exists():
            artifact.add_file(adapter_model_file)
            for extra in ("adapter_config.json", "config.json"):
                path = pretrained_model_dir / extra
                if path.exists():
                    artifact.add_file(path)
        elif standard_model_file.exists():
            artifact.add_file(standard_model_file)
        else:
            logger.warning(
                "No model weights found in %s; skipping wandb artifact upload.",
                pretrained_model_dir,
            )
            return

        self._wandb.log_artifact(artifact)

    def finish(self) -> None:
        if self._wandb.run is not None:
            self._wandb.finish()


def maybe_create_wandb_logger(
    cfg: dict,
    *,
    output_dir: Path,
    policy_type: str,
    dataset_repo_ids: list[str],
    seed: int,
    resume: bool,
    is_main: bool,
) -> MultiTrainWandbLogger | None:
    wandb_cfg = cfg.get("wandb") or {}
    if not wandb_cfg.get("enable"):
        return None
    if not is_main:
        return None
    project = wandb_cfg.get("project")
    if not project:
        logger.warning("wandb.enable=true but wandb.project is empty; skipping wandb.")
        return None

    job_name = cfg.get("training", {}).get("job_name") or f"multi_{policy_type}"
    return MultiTrainWandbLogger(
        wandb_cfg,
        job_name=job_name,
        output_dir=output_dir,
        policy_type=policy_type,
        dataset_repo_ids=dataset_repo_ids,
        seed=seed,
        resume=resume,
        full_config=cfg,
    )
