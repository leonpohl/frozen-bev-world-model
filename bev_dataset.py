# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
bev_dataset.py -- BEV World Model: dataset class
===================================================================

Loads precomputed BEV latents and returns sequences of length
n_input_frames+1: n input frames + 1 target frame.

Two load modes (controlled via BEVDatasetConfig.h5_path):

  MODE A -- .npy (default, no h5_path):
    Every __getitem__ call opens the .npy files directly.
    Good for: small datasets, development, when no HDF5 cache
    is available.

  MODE B -- HDF5 (h5_path given):
    Latents are read from a single HDF5 file created beforehand
    with build_hdf5_cache.py. Per worker process
    a separate h5py.File instance is opened (thread-safe).
    Good for: full nuScenes, training on HDD/NFS, many DataLoader workers.

Usage:
    # Mode A -- directly from .npy
    cfg = BEVDatasetConfig(
        sources=[("infos_val.pkl", "/Latents/Val")]
    )

    # Mode B -- from HDF5 cache
    cfg = BEVDatasetConfig(
        sources=[("infos_val.pkl", "/Latents/Val")],
        h5_path="/Cache/bev_latents_val.h5",
    )

    dataset = BEVLatentDataset(cfg)
    dataset = BEVLatentDataset(cfg, scene_indices=[0, 2, 5])
"""

import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class BEVDatasetConfig:
    """
    Configuration for BEVLatentDataset.

    sources:         list of (pkl_path, latent_dir) pairs.
                     Each pair describes a PKL and the corresponding
                     folder with .npy BEV latents. Multiple pairs are
                     pooled into a single dataset.

    n_input_frames:  number of input frames per sequence (default: 3).
                     Target is always the immediately following frame.

    scene_gap_sec:   time gap in seconds from which the start of a new scene
                     is assumed. At nuScenes 2Hz, 2.0s suffices.

    h5_path:         optional path to an HDF5 cache file
                     (created with build_hdf5_cache.py).
                     None  -> mode A: load directly from .npy
                     path  -> mode B: load from the HDF5 cache

    packed_dir:      optional path to a memmap pack directory
                     (created with pack_latents.py). Expects the
                     file 'packed.npy' inside -- a single memmap [N,C,H,W]
                     with all frames of this split in scene order.
                     None  -> mode A/B as above
                     path  -> mode C: one read per frame from the memmap,
                              instead of thousands of individual opens.
                     Takes precedence over h5_path if both are set.
    """
    sources:        List[Tuple[str, str]]   # [(pkl_path, latent_dir), ...]
    n_input_frames: int   = 3
    scene_gap_sec:  float = 2.0
    h5_path:        Optional[str] = None    # None = .npy mode
    packed_dir:     Optional[str] = None    # None = no memmap pack (mode C)
    latent_scale:   float = 1.0             # multiplicative normalization
                                            # at load time (det: 0.3641 -> seg scale,
                                            # so the smooth-L1 operating point beta/sigma
                                            # stays identical). 1.0 = bit-identical.
    # --- spatially weighted latent loss (default OFF) ----------
    gt_masks_path:  Optional[str] = None    # npz from mk_gt_masks_seg.py
                                            # (keys=token, packbits 6x128x128).
                                            # None = no cell_weight in the sample
                                            # -> bit-identical to before.
    rare_classes:   Tuple[str, ...] = ("stop_line", "ped_crossing", "divider")
    rare_weight:    float = 1.0             # cell weight 1+(w-1)*rare_mask;
                                            # 1.0 = weight map constant 1.


# ---------------------------------------------------------------------------
# Helper: read one PKL and split into scenes
# ---------------------------------------------------------------------------

def _load_scenes_from_pkl(
    pkl_path: str,
    latent_dir: str,
    scene_gap_threshold_sec: float = 2.0,
) -> List[List[Dict]]:
    """
    Reads a BEVFusion info PKL, sorts frames by timestamp, and
    separates them into individual scenes based on large time gaps.

    Each frame dict gets '_latent_dir' recorded for the .npy lookup
    in mode A. In mode B this field is ignored -- the lookup
    then runs via the token key in the HDF5.
    """
    pkl_path   = Path(pkl_path)
    latent_dir = Path(latent_dir)

    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL not found: {pkl_path}")
    if not latent_dir.exists():
        raise FileNotFoundError(f"latent directory not found: {latent_dir}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    infos: List[Dict] = sorted(data["infos"], key=lambda x: x["timestamp"])

    # Filter out missing .npy files -- the PKL can have more entries
    # than actual existing latents (e.g. if the BEVFusion extraction
    # failed for individual frames). Missing frames are skipped.
    n_before = len(infos)
    infos = [i for i in infos
             if (latent_dir / ("bev_latent_" + i["token"] + ".npy")).exists()]
    n_missing = n_before - len(infos)
    if n_missing > 0:
        print(f"  [Info] {n_missing}/{n_before} frames without .npy skipped "
              f"({latent_dir.name})")

    for info in infos:
        info["_latent_dir"] = str(latent_dir)

    threshold_us = scene_gap_threshold_sec * 1e6
    scenes: List[List[Dict]] = []
    current: List[Dict] = [infos[0]]

    for prev, curr in zip(infos[:-1], infos[1:]):
        if curr["timestamp"] - prev["timestamp"] > threshold_us:
            scenes.append(current)
            current = [curr]
        else:
            current.append(curr)

    scenes.append(current)
    return scenes


# ---------------------------------------------------------------------------
# HDF5 handle cache -- one handle per worker thread (thread-safe)
# ---------------------------------------------------------------------------

class _H5HandleCache:
    """
    Manages one h5py.File handle per worker thread.

    Why needed:
        h5py.File objects are NOT thread-safe when multiple threads
        read simultaneously. PyTorch DataLoader starts its own
        process/thread per worker. Each worker therefore needs its own
        h5py.File instance.

        Solution: threading.local() -- each thread gets its own
        slot. On the first access of a thread the file is opened
        and stored in the thread-local slot.
    """

    def __init__(self, h5_path: str):
        self._path  = h5_path
        self._local = threading.local()

    def get(self):
        """Returns the h5py.File handle for the current thread."""
        if not hasattr(self._local, "handle") or self._local.handle is None:
            try:
                import h5py
            except ImportError:
                raise ImportError(
                    "h5py not installed. Please: pip install h5py\n"
                    "Or use mode A (no h5_path in BEVDatasetConfig)."
                )
            # swmr=True: single-writer-multiple-reader -- allows simultaneous
            # reading from multiple processes without locks
            self._local.handle = h5py.File(self._path, "r", swmr=True)
        return self._local.handle


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class BEVLatentDataset(Dataset):
    """
    PyTorch dataset for BEV world model training.

    Creates sliding-window sequences:
        Input:  [t-n, ..., t]  -> tensor [n_input_frames, 256, 128, 128]
        Target: [t+1]          -> tensor [256, 128, 128]

    Sequences are NEVER formed across scene boundaries or PKL boundaries.

    Args:
        config:        BEVDatasetConfig with paths, parameters and
                       optional h5_path for the HDF5 mode.
        scene_indices: optional -- use only sequences of these scenes.
                       None = all scenes. Set by make_dataloaders_split()
                       for train/val/test splits.
        transform:     optional transformation on the input tensor.
    """

    def __init__(
        self,
        config: BEVDatasetConfig,
        scene_indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.config    = config
        self.transform = transform

        # Initialize the HDF5 handle cache if h5_path is given
        self._h5: Optional[_H5HandleCache] = None
        if config.h5_path is not None:
            h5_path = Path(config.h5_path)
            if not h5_path.exists():
                raise FileNotFoundError(
                    f"HDF5 cache not found: {h5_path}\n"
                    f"Please run build_hdf5_cache.py first."
                )
            self._h5 = _H5HandleCache(str(h5_path))

        # Initialize the memmap pack if packed_dir is given (mode C).
        # The handle is opened LAZY at the first _load_latent in the respective
        # worker process (see _load_latent) -- under fork each
        # worker gets its own mapping this way. Here only store the path +
        # check existence (fail-fast), do NOT open yet.
        self._latent_scale: float = float(getattr(config, "latent_scale", 1.0))
        self._packed_dir: Optional[str] = config.packed_dir
        self._packed_path: Optional[Path] = None
        self._packed_array = None
        self._packed_is_f32 = True     # set correctly at lazy open
        if config.packed_dir is not None:
            self._packed_path = Path(config.packed_dir) / "packed.npy"
            if not self._packed_path.exists():
                raise FileNotFoundError(
                    f"Memmap pack not found: {self._packed_path}\n"
                    f"Please run pack_latents.py for this split first."
                )

        window_size = config.n_input_frames + 1

        # Load all PKL sources
        self._all_scenes: List[List[Dict]] = []
        self._source_info: List[Dict] = []

        for pkl_path, latent_dir in config.sources:
            scenes = _load_scenes_from_pkl(
                pkl_path, latent_dir, config.scene_gap_sec
            )
            self._all_scenes.extend(scenes)
            self._source_info.append({
                "pkl":           pkl_path,
                "latent_dir":    latent_dir,
                "n_scenes":      len(scenes),
                "scene_lengths": [len(s) for s in scenes],
            })

        # Assign the global pack row number per frame: running counter over
        # ALL scenes in construction order -- IDENTICAL to the write
        # order of pack_latents.py (which uses the same dataset instance).
        # Deliberately BEFORE the scene_indices filtering: the row of a frame is
        # thus globally stable, independent of which split is currently active
        # (all splits read from the same packed.npy). No separate
        # index file -- the mapping is deterministic from the config.
        # Always set (also without packed_dir) so pack_latents.py can
        # read the order in .npy mode; otherwise a harmless plain int counter.
        row = 0
        for scene in self._all_scenes:
            for info in scene:
                info["_packed_row"] = row
                row += 1
        self._n_packed_frames = row

        # Determine active scenes
        if scene_indices is None:
            active_scenes = self._all_scenes
        else:
            if scene_indices and max(scene_indices) >= len(self._all_scenes):
                raise IndexError(
                    f"scene_indices contains index {max(scene_indices)}, "
                    f"but there are only {len(self._all_scenes)} scenes."
                )
            active_scenes = [self._all_scenes[i] for i in scene_indices]

        # Create sliding-window sequences
        self._windows: List[List[Dict]] = []

        for scene in active_scenes:
            n_frames  = len(scene)
            n_windows = n_frames - window_size + 1

            if n_windows <= 0:
                print(
                    f"[BEVLatentDataset] Warning: scene with {n_frames} frames "
                    f"too short for window size {window_size}. Skipped."
                )
                continue

            for i in range(n_windows):
                self._windows.append(scene[i : i + window_size])

        if config.packed_dir:
            mode = f"packed-memmap ({self._packed_path})"
        elif config.h5_path:
            mode = f"HDF5 ({config.h5_path})"
        else:
            mode = ".npy"
        self._print_summary(scene_indices, mode)

    # ------------------------------------------------------------------
    # Load latent -- mode A (.npy) or mode B (HDF5)
    # ------------------------------------------------------------------

    def _load_latent(self, frame: Dict) -> np.ndarray:
        """Loads a single latent frame as float32 numpy array [256,128,128].

        BEVFusion stores latents with a batch dimension: (1, 256, 128, 128).
        squeeze(axis=0) removes this dimension deterministically.

        Precedence: packed_dir (mode C) > h5_path (mode B) > .npy (mode A).
        """
        if self._packed_dir is not None:
            # Mode C: one read from the memmap via the global row number.
            # Lazy open PER WORKER PROCESS: self._packed_array starts as None
            # (also after fork), the first access in the worker maps the file.
            if self._packed_array is None:
                self._packed_array = np.load(self._packed_path, mmap_mode="r")
                # Guard: gross mismatch (wrong frame count) -> hard abort,
                # instead of silently reading wrong rows. Does NOT catch a
                # reordering at the same count -- against that, the fact that
                # pack_latents.py uses the same dataset instance for the order protects.
                if self._packed_array.shape[0] != self._n_packed_frames:
                    raise RuntimeError(
                        f"Pack mismatch: {self._packed_path} has "
                        f"{self._packed_array.shape[0]} rows, dataset expects "
                        f"{self._n_packed_frames}. Config/pack diverge "
                        f"-- regenerate packed.npy with the current config."
                    )
                # Remember the storage dtype once (perf: not checked per frame).
                self._packed_is_f32 = (self._packed_array.dtype == np.float32)
            # We ALWAYS need an in-RAM copy (no memmap view to the model,
            # otherwise each batch holds the whole file mapped).
            #  - the latent is returned in the STORAGE dtype
            #    (fp16 stays fp16). The upcast fp16->fp32 is deliberately NOT
            #    done here anymore, but only GPU-side at the respective h2d point
            #    (train_linux.py / eval_full_val.py) -- so only half as many
            #    bytes go over the PCIe bus (h2d was 24% of step time, stage-0 profile).
            #    Numerically neutral: fp16->fp32 is an exact widening, just relocated.
            #  - np.array() ALWAYS returns an in-RAM copy (no memmap view to
            #    the model), independent of the dtype.
            row = self._packed_array[frame["_packed_row"]]
            arr = np.array(row)                      # native dtype (fp16 OR fp32), pure copy
            if self._latent_scale != 1.0:            # normalization (dtype-preserving)
                arr = arr * arr.dtype.type(self._latent_scale)
            return arr

        if self._h5 is not None:
            # Mode B: HDF5 lookup via token string
            handle = self._h5.get()
            arr = handle["latents"][frame["token"]][:]
        else:
            # Mode A: directly from .npy
            path = Path(frame["_latent_dir"]) / f"bev_latent_{frame['token']}.npy"
            arr = np.load(path).astype("float32")

        # BEVFusion stores (1, 256, 128, 128) -- remove batch dimension
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr.squeeze(axis=0)   # -> [256, 128, 128]

        if self._latent_scale != 1.0:   # normalization (modes A/B)
            arr = arr * arr.dtype.type(self._latent_scale)
        return arr

    # ------------------------------------------------------------------
    # Pack order (single source of truth for pack_latents.py)
    # ------------------------------------------------------------------

    def iter_pack_order(self):
        """Yields (row, frame) for ALL frames in global pack order.

        Exactly the set and order that _packed_row defines. pack_latents.py
        uses this iterator to write the memmap -- thereby there is only ONE
        implementation of the frame order (that of the dataset); pack and runtime
        cannot diverge. Independent of scene_indices (runs over
        _all_scenes, not over the active windows).
        """
        for scene in self._all_scenes:
            for frame in scene:
                yield frame["_packed_row"], frame

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    @staticmethod
    def _rel_ego_delta(fi: Dict, fj: Dict, scale: float = 10.0):
        """Relative ego motion fi->fj in fi's ego coordinates:
        [dx/scale, dy/scale, cos(dyaw), sin(dyaw)]. Ego conditioning.
        Missing pose -> zero vector (backwards compatible)."""
        ti = fi.get("ego2global_translation"); ri = fi.get("ego2global_rotation")
        tj = fj.get("ego2global_translation"); rj = fj.get("ego2global_rotation")
        if ti is None or tj is None or ri is None or rj is None:
            return [0.0, 0.0, 1.0, 0.0]
        def yaw(q):  # q=[w,x,y,z]
            w, x, y, z = q
            return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yi = float(yaw(ri)); dyaw = float(yaw(rj)) - yi
        dxg, dyg = tj[0] - ti[0], tj[1] - ti[1]              # global
        c, s = np.cos(-yi), np.sin(-yi)                      # rotate into fi's ego frame
        px = c * dxg - s * dyg
        py = s * dxg + c * dyg
        return [px / scale, py / scale, float(np.cos(dyaw)), float(np.sin(dyaw))]

    def __getitem__(self, idx: int) -> Dict:
        window       = self._windows[idx]
        input_frames = window[:-1]
        target_frame = window[-1]

        input_tensor = torch.from_numpy(
            np.stack([self._load_latent(f) for f in input_frames], axis=0)
        )
        target_tensor = torch.from_numpy(self._load_latent(target_frame))

        if self.transform is not None:
            input_tensor = self.transform(input_tensor)

        # Ego deltas along the sequence: [d12, d23, ..., d(n-1)n, d(n->target)]
        # -> flat [ (n_input) * 4 ]. The model cuts depending on ego_cond_mode:
        #    action = last 4 (F_last->target), state = first (n_input-1)*4.
        chain = list(input_frames) + [target_frame]
        ego = []
        for a, b in zip(chain[:-1], chain[1:]):
            ego.extend(self._rel_ego_delta(a, b))
        ego_delta = torch.tensor(ego, dtype=torch.float32)   # [n_input * 4]

        sample = {
            "input":        input_tensor,   # [n_input_frames, 256, 128, 128]
            "target":       target_tensor,  # [256, 128, 128]
            "ego_delta":    ego_delta,      # [n_input*4] 
            "input_tokens": [f["token"] for f in input_frames],
            "target_token": target_frame["token"],
        }
        # Cell weights from the GT map of the TARGET frame.
        if self.config.gt_masks_path is not None:
            sample["cell_weight"] = torch.from_numpy(
                self._cell_weight(target_frame["token"]))   # [128,128] float32
        return sample

    # Class order from mk_gt_masks_seg.py (= seg config)
    _GT_CLASSES = ("drivable_area", "ped_crossing", "walkway", "stop_line",
                   "carpark_area", "divider")

    def _cell_weight(self, token: str) -> np.ndarray:
        """1+(w-1)*rare_mask from the packed 6-class GT mask.
        The npz is opened lazily per worker (fork-safe)."""
        if not hasattr(self, "_gt_masks_npz"):
            self._gt_masks_npz = np.load(self.config.gt_masks_path)
        packed = self._gt_masks_npz[token]
        masks = np.unpackbits(packed).reshape(6, 128, 128)
        idx = [self._GT_CLASSES.index(c) for c in self.config.rare_classes]
        rare = masks[idx].any(axis=0).astype(np.float32)
        w = float(self.config.rare_weight)
        return (1.0 + (w - 1.0) * rare).astype(np.float32)

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    def _print_summary(self, scene_indices, mode: str) -> None:
        total  = len(self._all_scenes)
        active = len(scene_indices) if scene_indices is not None else total
        print(
            f"[BEVLatentDataset] {len(self._source_info)} source(s) | "
            f"{active}/{total} scene(s) | "
            f"{len(self)} sequences | "
            f"mode: {mode}"
        )
