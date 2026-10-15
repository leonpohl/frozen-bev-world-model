# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
bev_dataloader.py -- BEV World Model: DataLoader factory
====================================================================

Creates Train/Val/Test DataLoaders with two modes:

  MODE 1 -- split (mini / development):
    One or more PKLs are deterministically split on the scene level
    into Train/Val/Test. BEVLatentDataset is instantiated with the
    respective scene_indices -- no duplicated code.

    Usage:
        loaders = make_dataloaders_split(
            sources=[("infos_val.pkl", "/Latents/Val")],
            train_ratio=0.70, val_ratio=0.15,
            batch_size=16, seed=42,
        )

  MODE 2 -- explicit (full nuScenes / production):
    Separate PKLs per split. DataLoaders are built directly from the
    respective BEVDatasetConfig objects -- no split needed.
    Optional: one HDF5 cache per split (train_h5_path/val_h5_path/
    test_h5_path) -- recommended for full nuScenes data volumes where
    .npy direct access on tens of thousands of individual files is too slow.

    Usage (.npy direct):
        loaders = make_dataloaders_explicit(
            train_sources=[("infos_train.pkl", "/Latents/Train")],
            val_sources  =[("infos_val.pkl",   "/Latents/Val")],
            test_sources =[("infos_test.pkl",  "/Latents/Test")],
            batch_size=16,
        )

    Usage (HDF5 cache, recommended for the full dataset):
        loaders = make_dataloaders_explicit(
            train_sources=[("infos_train.pkl", "/Latents/Train")],
            val_sources  =[("infos_val.pkl",   "/Latents/Val")],
            batch_size=16,
            train_h5_path="/Cache/bev_latents_train.h5",
            val_h5_path  ="/Cache/bev_latents_val.h5",
        )

Both functions return:
    {"train": DataLoader, "val": DataLoader, "test": DataLoader | None}
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from bev_dataset import BEVDatasetConfig, BEVLatentDataset


# ---------------------------------------------------------------------------
# Sampling weights for the train loader
# ---------------------------------------------------------------------------

def _build_train_sampler(ds, mode, alpha, gt_masks_path, dynamics_path, seed):
    """WeightedRandomSampler over the train windows.
    mode=rare:     w ~ (rare-class pixel share of the target + 1e-3)^alpha
    mode=dynamics: w ~ (dynamics score of the target pair / median)^alpha
    replacement=True, num_samples=len(ds) -> epoch length unchanged."""
    import json

    import numpy as np
    from torch.utils.data import WeightedRandomSampler

    tokens = [w[-1]["token"] for w in ds._windows]
    if mode == "rare":
        assert gt_masks_path, "sampler_mode=rare requires gt_masks_train_path"
        npz = np.load(gt_masks_path)
        idx = [BEVLatentDataset._GT_CLASSES.index(c) for c in ds.config.rare_classes]
        shares = np.array([
            np.unpackbits(npz[t]).reshape(6, 128, 128)[idx].any(axis=0).mean()
            for t in tokens], dtype=np.float64)
        weights = (shares + 1e-3) ** alpha
    elif mode == "dynamics":
        assert dynamics_path, "sampler_mode=dynamics requires dynamics_scores_path"
        with open(dynamics_path) as f:
            scores = json.load(f)
        s = np.array([scores[t] for t in tokens], dtype=np.float64)
        weights = (s / np.median(s)) ** alpha
    else:
        raise ValueError(f"sampler_mode must be off|rare|dynamics, was '{mode}'")

    weights = weights / weights.mean()
    print(f"[sampler:{mode}] alpha={alpha}, n={len(weights)}, "
          f"w-min/median/max = {weights.min():.3f}/{np.median(weights):.3f}/"
          f"{weights.max():.3f}")
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(weights), replacement=True,
                                 generator=g)


# ---------------------------------------------------------------------------
# Helper: shuffle and split scenes deterministically
# ---------------------------------------------------------------------------

def _split_scene_indices(
    n_scenes: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits n_scenes scene indices deterministically into Train/Val/Test.

    The split happens on the scene level: all sequences of a scene always
    land in the same split -- no leakage from overlapping windows.

    Uses random.Random(seed) -- isolated RNG, no global state.
    """
    assert train_ratio + val_ratio <= 1.0

    indices = list(range(n_scenes))
    random.Random(seed).shuffle(indices)

    n_train = max(1, int(n_scenes * train_ratio))
    n_val   = max(1, int(n_scenes * val_ratio))
    n_test  = max(0, n_scenes - n_train - n_val)

    # Catch rounding errors
    if n_train + n_val + n_test != n_scenes:
        n_train = n_scenes - n_val - n_test

    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


# ---------------------------------------------------------------------------
# Internal DataLoader builder
# ---------------------------------------------------------------------------

def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    prefetch_factor: int = 4,
    pin_memory: bool = True,   # Loader diagnostics: the ONE pin thread is the remaining
                               # bottleneck candidate (data% ~58 after 1d); False only for
                               # cluster-diagnosis configs, default unchanged.
    sampler=None,              # WeightedRandomSampler (train)
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    # prefetch_factor may ONLY be set when num_workers > 0,
    # otherwise PyTorch throws a ValueError. With num_workers=0 (Windows /
    # .npy debug) the DataLoader falls back to synchronous loading.
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        generator=generator,
    )
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


# ---------------------------------------------------------------------------
# MODE 1 -- split
# ---------------------------------------------------------------------------

def make_dataloaders_split(
    sources: List[Tuple[str, str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 16,
    n_input_frames: int = 3,
    scene_gap_sec: float = 2.0,
    num_workers: int = 4,
    seed: int = 42,
    transform=None,
    h5_path: str = None,
    packed_dir: str = None,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    latent_scale: float = 1.0,   # multiplicative normalization (det->seg scale)
) -> Dict[str, Optional[DataLoader]]:
    """
    Loads all scenes from 'sources', shuffles deterministically and splits
    on the scene level into Train/Val/Test. BEVLatentDataset is instantiated
    per split with the matching scene_indices.

    Args:
        sources:        [(pkl_path, latent_dir), ...]
        train_ratio:    share of training scenes (default: 0.70)
        val_ratio:      share of validation scenes (default: 0.15)
        batch_size:     batch size for all DataLoaders
        n_input_frames: number of input frames per sequence
        scene_gap_sec:  threshold for scene separation in seconds
        num_workers:    DataLoader worker processes
        seed:           RNG seed for split and DataLoader
        transform:      optional transformation for input tensors

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader | None}
    """
    # Create the config once -- shared by all three splits.
    # BEVLatentDataset internally loads all scenes from the sources and
    # then filters to the desired split via scene_indices.
    cfg = BEVDatasetConfig(
        sources=sources,
        n_input_frames=n_input_frames,
        scene_gap_sec=scene_gap_sec,
        h5_path=h5_path,
        packed_dir=packed_dir,
        latent_scale=latent_scale,
    )

    # Determine the number of scenes without instantiating the Dataset:
    # For this, briefly create a dummy instance with scene_indices=[] -- loads only metadata.
    # Alternative: instantiate once and read n_scenes from _all_scenes.
    _probe = BEVLatentDataset(cfg, scene_indices=[])
    n_scenes = len(_probe._all_scenes)

    train_idx, val_idx, test_idx = _split_scene_indices(
        n_scenes, train_ratio, val_ratio, seed
    )

    print(f"[make_dataloaders_split] {n_scenes} scenes, seed={seed}:")
    print(f"  Train: {len(train_idx)} scene(s), indices {sorted(train_idx)}")
    print(f"  Val:   {len(val_idx)} scene(s), indices {sorted(val_idx)}")
    print(f"  Test:  {len(test_idx)} scene(s), indices {sorted(test_idx)}")

    train_ds = BEVLatentDataset(cfg, scene_indices=train_idx, transform=transform)
    val_ds   = BEVLatentDataset(cfg, scene_indices=val_idx,   transform=transform)

    loaders: Dict[str, Optional[DataLoader]] = {
        "train": _make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, seed=seed, prefetch_factor=prefetch_factor, pin_memory=pin_memory),
        "val":   _make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers, seed=seed, prefetch_factor=prefetch_factor, pin_memory=pin_memory),
        "test":  None,
    }

    if test_idx:
        test_ds = BEVLatentDataset(cfg, scene_indices=test_idx, transform=transform)
        loaders["test"] = _make_loader(test_ds, batch_size, shuffle=False, num_workers=num_workers, seed=seed, prefetch_factor=prefetch_factor, pin_memory=pin_memory)

    return loaders


# ---------------------------------------------------------------------------
# MODE 2 -- explicit
# ---------------------------------------------------------------------------

def make_dataloaders_explicit(
    train_sources: List[Tuple[str, str]],
    val_sources:   List[Tuple[str, str]],
    test_sources:  Optional[List[Tuple[str, str]]] = None,
    batch_size: int = 16,
    n_input_frames: int = 3,
    scene_gap_sec: float = 2.0,
    num_workers: int = 4,
    seed: int = 42,
    transform=None,
    train_h5_path: str = None,
    val_h5_path: str = None,
    test_h5_path: str = None,
    train_packed_dir: str = None,
    val_packed_dir: str = None,
    test_packed_dir: str = None,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    latent_scale: float = 1.0,   # normalization at load time (1.0 = off)
    # --- Sampler/weighting options (all defaults OFF = bit-identical) -------------------------
    gt_masks_train_path: str = None,   # npz -> train samples with cell_weight
    rare_classes: tuple = ("stop_line", "ped_crossing", "divider"),
    rare_weight: float = 1.0,
    sampler_mode: str = "off",         # off | rare | dynamics
    sampler_alpha: float = 0.5,        # exponent of the sampling weights
    dynamics_scores_path: str = None,  # JSON token->score (train)
) -> Dict[str, Optional[DataLoader]]:
    """
    Builds DataLoaders from explicitly separated PKL files.
    No further split needed -- nuScenes already provides separated
    train/val/test info files.

    Suited for: full nuScenes (700 train + 150 val + 150 test scenes).

    Args:
        train_sources:  [(pkl_path, latent_dir), ...] for training
        val_sources:    [(pkl_path, latent_dir), ...] for validation
        test_sources:   [(pkl_path, latent_dir), ...] for test (optional)
        batch_size:     batch size
        n_input_frames: number of input frames per sequence
        scene_gap_sec:  threshold for scene separation in seconds
        num_workers:    DataLoader worker processes
        seed:           RNG seed for the DataLoader generator
        transform:      optional transformation for input tensors
        train_h5_path:  optional path to the HDF5 cache for train
                        (created with build_hdf5_cache.py).
                        None -> mode A: load directly from .npy (train_sources)
        val_h5_path:    optional path to the HDF5 cache for val.
        test_h5_path:   optional path to the HDF5 cache for test.

        NOTE: Unlike make_dataloaders_split() (ONE shared
        h5_path for all splits, since there a single source is split
        internally), there is one h5_path PER SPLIT here --
        train and val are already separate PKLs/latent folders and thus
        typically also separate HDF5 files (e.g.
        bev_latents_train.h5 / bev_latents_val.h5).

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader | None}
    """
    def _build(sources, shuffle, h5_path, packed_dir, is_train=False):
        cfg = BEVDatasetConfig(
            sources=sources,
            n_input_frames=n_input_frames,
            scene_gap_sec=scene_gap_sec,
            h5_path=h5_path,
            packed_dir=packed_dir,
            latent_scale=latent_scale,
            # Weight maps ONLY in the train split (the val loss remains
            # unweighted -> early stopping/selection comparable across sweeps).
            gt_masks_path=(gt_masks_train_path if is_train else None),
            rare_classes=tuple(rare_classes),
            rare_weight=rare_weight,
        )
        ds = BEVLatentDataset(cfg, transform=transform)
        sampler = None
        if is_train and sampler_mode != "off":
            sampler = _build_train_sampler(ds, sampler_mode, sampler_alpha,
                                           gt_masks_train_path,
                                           dynamics_scores_path, seed)
            shuffle = False   # sampler and shuffle are mutually exclusive
        return _make_loader(ds, batch_size, shuffle=shuffle, num_workers=num_workers,
                            seed=seed, prefetch_factor=prefetch_factor,
                            pin_memory=pin_memory, sampler=sampler)

    loaders: Dict[str, Optional[DataLoader]] = {
        "train": _build(train_sources, shuffle=True,  h5_path=train_h5_path, packed_dir=train_packed_dir, is_train=True),
        "val":   _build(val_sources,   shuffle=False, h5_path=val_h5_path,   packed_dir=val_packed_dir),
        "test":  _build(test_sources,  shuffle=False, h5_path=test_h5_path,  packed_dir=test_packed_dir) if test_sources else None,
    }

    print(f"[make_dataloaders_explicit] Train={len(loaders['train'])} Batches, "
          f"Val={len(loaders['val'])} Batches"
          + (f", Test={len(loaders['test'])} Batches" if loaders["test"] else ""))

    return loaders
