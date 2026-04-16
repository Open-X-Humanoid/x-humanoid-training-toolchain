"""Adapter that makes a collection of LeRobotDatasets quack like a single dataset
with a ``.meta`` attribute compatible with ``make_policy`` / ``make_pre_post_processors``.

All datasets are assumed to have identical action / state dimensions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import torch
import torch.utils.data

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


class MultiDatasetMetaAdapter:
    """Drop-in replacement for ``LeRobotDatasetMetadata`` built from multiple datasets.

    ``make_policy(ds_meta=...)`` accesses ``ds_meta.features``, ``ds_meta.stats``,
    ``ds_meta.fps``, and ``ds_meta.camera_keys``.  This adapter provides all of those
    by combining info from the underlying sub-datasets.
    """

    def __init__(
        self,
        datasets: list[LeRobotDataset],
        disabled_features: set[str] | None = None,
    ) -> None:
        first = datasets[0].meta
        disabled = disabled_features or set()

        self.info = dict(first.info)
        self.info["features"] = {
            k: v for k, v in first.info["features"].items() if k not in disabled
        }
        self.info["total_episodes"] = sum(d.num_episodes for d in datasets)
        self.info["total_frames"] = sum(d.num_frames for d in datasets)

        common_keys = set(self.info["features"].keys())

        # Validate shapes match across datasets
        for key in common_keys:
            shapes = set()
            for ds in datasets:
                if key in ds.meta.stats and "mean" in ds.meta.stats[key]:
                    shapes.add(np.array(ds.meta.stats[key]["mean"]).shape)
            if len(shapes) > 1:
                raise ValueError(
                    f"Feature '{key}' has different shapes across datasets: {shapes}. "
                    f"All datasets must have the same dimensions for common features."
                )

        filtered_stats = [
            {k: v for k, v in d.meta.stats.items() if k in common_keys}
            for d in datasets
        ]
        self.stats = aggregate_stats(filtered_stats)
        self.episodes = None

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def camera_keys(self) -> list[str]:
        return [k for k, ft in self.features.items() if ft["dtype"] in ("video", "image")]

    @property
    def shapes(self) -> dict[str, tuple]:
        return {k: tuple(ft["shape"]) for k, ft in self.features.items()}


class MultiRobotDataset(torch.utils.data.Dataset):
    """Concatenates multiple ``LeRobotDataset`` instances, each with its own root.

    Unlike upstream ``MultiLeRobotDataset`` (which computes ``root / repo_id``
    internally), this class accepts pre-built datasets so every dataset can live
    at an arbitrary absolute path.
    """

    def __init__(
        self,
        datasets: list[LeRobotDataset],
        image_transforms: Callable | None = None,
    ) -> None:
        super().__init__()
        self._datasets = datasets

        intersection = set(datasets[0].features)
        for ds in datasets[1:]:
            intersection &= set(ds.features)
        if not intersection:
            raise RuntimeError(
                "The provided datasets share no common feature keys. "
                "Multi-dataset training requires at least overlapping state/action/image features."
            )

        self.disabled_features: set[str] = set()
        for ds in datasets:
            extra = set(ds.features) - intersection
            if extra:
                logger.warning("Disabled non-common features %s from %s", extra, ds.repo_id)
                self.disabled_features.update(extra)

        self.image_transforms = image_transforms
        self.meta = MultiDatasetMetaAdapter(datasets, self.disabled_features)

    @property
    def repo_id_to_index(self) -> dict[str, int]:
        return {ds.repo_id: i for i, ds in enumerate(self._datasets)}

    @property
    def num_frames(self) -> int:
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        return sum(d.num_episodes for d in self._datasets)

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds (total {len(self)}).")
        offset = 0
        for ds_idx, ds in enumerate(self._datasets):
            if idx < offset + ds.num_frames:
                item = ds[idx - offset]
                item["dataset_index"] = torch.tensor(ds_idx)
                for k in self.disabled_features:
                    item.pop(k, None)
                return item
            offset += ds.num_frames
        raise AssertionError("Index within bounds but no dataset matched.")

    def __repr__(self) -> str:
        ids = [ds.repo_id for ds in self._datasets]
        return (
            f"{self.__class__.__name__}(\n"
            f"  repo_ids={ids},\n"
            f"  num_frames={self.num_frames},\n"
            f"  num_episodes={self.num_episodes},\n"
            f")"
        )


def build_multi_dataset(
    dataset_entries: list[dict],
    *,
    delta_timestamps: dict[str, list[float]] | None = None,
    image_transforms: Callable | None = None,
    video_backend: str | None = None,
    tolerance_s: float = 1e-4,
) -> MultiRobotDataset:
    """Build a ``MultiRobotDataset`` from a list of config dicts.

    Each dict must contain ``repo_id`` and ``root``; optionally ``episodes``.
    """
    datasets: list[LeRobotDataset] = []
    for entry in dataset_entries:
        root = entry["root"]
        repo_id = entry["repo_id"]
        episodes = entry.get("episodes")
        logger.info("Loading dataset %s from %s", repo_id, root)
        ds = LeRobotDataset(
            repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            tolerance_s=tolerance_s,
            video_backend=video_backend,
        )
        datasets.append(ds)
    return MultiRobotDataset(datasets, image_transforms=image_transforms)
