"""Multi-dataset training script for LeRobot policies.

Usage (from repo root)::

    ./scripts/xhum-run xhum.train.train_multi --config path/to/multi_train.json

This script bypasses upstream ``lerobot-train`` (which blocks multi-dataset use)
and directly orchestrates:

1. Loading multiple ``LeRobotDataset`` instances (each at its own root path).
2. Wrapping them in ``MultiRobotDataset`` + ``MultiDatasetMetaAdapter``.
3. Creating a policy via ``make_policy`` with the aggregated metadata.
4. Running the standard training loop (reusing ``update_policy`` from lerobot).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from pprint import pformat

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_training_state,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, has_method, init_logging, inside_slurm

from xhum.train.meta_adapter import build_multi_dataset

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_DIR = "pretrained_model"


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    policy,
    optimizer,
    scheduler,
    preprocessor,
    postprocessor,
    train_cfg: dict,
) -> None:
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)

    with open(pretrained_dir / "multi_train_config.json", "w") as f:
        json.dump(train_cfg, f, indent=2)

    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)

    save_training_state(checkpoint_dir, step, optimizer, scheduler)


def train(cfg: dict) -> None:
    datasets_cfg = cfg["datasets"]
    policy_cfg = cfg["policy"]
    train_cfg = cfg["training"]
    use_imagenet_stats = cfg.get("use_imagenet_stats", True)
    video_backend = cfg.get("video_backend", None)

    output_dir = Path(train_cfg["output_dir"])
    steps = train_cfg["steps"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 4)
    save_freq = train_cfg.get("save_freq", 20_000)
    log_freq = train_cfg.get("log_freq", 200)
    seed = train_cfg.get("seed", 1000)
    resume = train_cfg.get("resume", False)
    resume_dir = train_cfg.get("resume_dir", None)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )
    init_logging(accelerator=accelerator)
    is_main = accelerator.is_main_process

    if is_main:
        logging.info("Multi-dataset training config:\n%s", pformat(cfg))

    set_seed(seed, accelerator=accelerator)
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- dataset ---
    if is_main:
        logging.info("Loading %d datasets …", len(datasets_cfg))

    first_meta = LeRobotDatasetMetadata(
        datasets_cfg[0]["repo_id"], root=datasets_cfg[0]["root"],
    )

    if policy_cfg.get("path"):
        pol_cfg = PreTrainedConfig.from_pretrained(policy_cfg["path"])
    else:
        from lerobot.policies.factory import make_policy_config
        pol_cfg = make_policy_config(policy_cfg["type"])

    delta_timestamps = resolve_delta_timestamps(pol_cfg, first_meta)

    dataset = build_multi_dataset(
        datasets_cfg,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )
    accelerator.wait_for_everyone()

    if use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stat_name, stat_val in IMAGENET_STATS.items():
                dataset.meta.stats[key][stat_name] = torch.tensor(stat_val, dtype=torch.float32)

    if is_main:
        logging.info("Combined dataset: %s", dataset)
        logging.info("Repo-id → index: %s", dataset.repo_id_to_index)

    # --- policy ---
    if is_main:
        logging.info("Creating policy (type=%s)", policy_cfg["type"])

    pol_cfg.push_to_hub = policy_cfg.get("push_to_hub", False)
    if policy_cfg.get("path"):
        pol_cfg.pretrained_path = Path(policy_cfg["path"])

    policy = make_policy(cfg=pol_cfg, ds_meta=dataset.meta)

    # --- pre/post processors ---
    proc_kwargs: dict = {"dataset_stats": dataset.meta.stats}
    postproc_kwargs: dict = {}
    if policy_cfg.get("path"):
        proc_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        postproc_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=pol_cfg,
        pretrained_path=pol_cfg.pretrained_path if hasattr(pol_cfg, "pretrained_path") else None,
        **proc_kwargs,
        **postproc_kwargs,
    )

    # --- optimizer / scheduler ---
    optimizer_cfg = pol_cfg.get_optimizer_preset()
    scheduler_cfg = pol_cfg.get_scheduler_preset()
    params = policy.get_optim_params() if hasattr(policy, "get_optim_params") else policy.parameters()
    optimizer = optimizer_cfg.build(params)
    lr_scheduler = scheduler_cfg.build(optimizer, steps) if scheduler_cfg else None
    grad_clip_norm = getattr(optimizer_cfg, "grad_clip_norm", 10.0)

    step = 0
    if resume and resume_dir:
        step, optimizer, lr_scheduler = load_training_state(Path(resume_dir), optimizer, lr_scheduler)
        if is_main:
            logging.info("Resumed from step %d", step)

    # --- info ---
    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    if is_main:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")
        logging.info("steps=%d (%s)", steps, format_big_number(steps))
        logging.info("num_frames=%d (%s)", dataset.num_frames, format_big_number(dataset.num_frames))
        logging.info("num_episodes=%d", dataset.num_episodes)
        eff_bs = batch_size * accelerator.num_processes
        logging.info("Effective batch size: %d x %d = %d", batch_size, accelerator.num_processes, eff_bs)
        logging.info("num_learnable_params=%d (%s)", num_learnable, format_big_number(num_learnable))
        logging.info("num_total_params=%d (%s)", num_total, format_big_number(num_total))

    # --- dataloader ---
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler,
    )
    dl_iter = cycle(dataloader)
    policy.train()

    # --- metrics ---
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main:
        progbar = tqdm(
            total=steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info("Start multi-dataset training, effective batch size: %d", batch_size * accelerator.num_processes)

    # --- loop ---
    for _ in range(step, steps):
        t0 = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - t0

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        if is_main:
            progbar.update(1)
        train_tracker.step()

        is_log = log_freq > 0 and step % log_freq == 0 and is_main
        is_save = step % save_freq == 0 or step == steps

        if is_log:
            logging.info(train_tracker)
            train_tracker.reset_averages()

        if is_save and is_main:
            logging.info("Saving checkpoint at step %d", step)
            ckpt_dir = get_step_checkpoint_dir(output_dir, steps, step)
            _save_checkpoint(
                ckpt_dir,
                step,
                accelerator.unwrap_model(policy),
                optimizer,
                lr_scheduler,
                preprocessor,
                postprocessor,
                cfg,
            )
            update_last_checkpoint(ckpt_dir)
            accelerator.wait_for_everyone()

    if is_main:
        progbar.close()
        logging.info("Training complete.")

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset LeRobot training")
    parser.add_argument("--config", required=True, help="Path to multi_train JSON config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
