# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
pack_latents.py -- BEV World Model: memmap packing
=================================================================

Packs the precomputed .npy BEV latents of a split into ONE
contiguous memmap file ({packed_dir}/packed.npy). At runtime
BEVLatentDataset then reads line by line from this single file
instead of thousands of individual opens -- measured ~2.6x faster on
BeeGFS (measured on the cluster filesystem).

CORE PRINCIPLE (single source of truth):
    The write order does NOT come from an independent PKL logic here,
    but from a real BEVLatentDataset instance (packed_dir=None,
    scene_indices=None). The exact same instance assigns the
    row numbers at runtime (_packed_row). As a result there is only ONE
    implementation of the frame order -> pack and runtime cannot diverge.

MODES:
    explicit config:  --split {train,val,test}
        sources  <- data.<split>_sources
        out_dir  <- data.<split>_packed_dir
    split config:     --split all
        sources  <- data.sources
        out_dir  <- data.packed_dir   (ONE pack for all internal splits)

USAGE:
    # float32 (default, lossless):
    python -u pack_latents.py --config config_nuscenes_full.yaml --split val
    python -u pack_latents.py --config config_nuscenes_full.yaml --split train

    # float16:
    python -u pack_latents.py --config config_nuscenes_full.yaml --split val --dtype float16

NOTE:
    Run on val (small) first, measure runtime + size, then
    extrapolate to train -- do not guess. The original .npy files remain
    untouched as reference/fallback.
"""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from bev_dataset import BEVDatasetConfig, BEVLatentDataset


DTYPES = {"float32": np.float32, "float16": np.float16}


# =============================================================================
# Manifest helper -- audit artifact, NOT a runtime index.
# Row numbers remain deterministic and are computed at runtime
# (BEVLatentDataset._packed_row) -- the manifest only serves to check
# against silent config/order mismatches and as reproducibility
# documentation for the thesis.
# =============================================================================

def _config_hash(cfg: dict) -> str:
    """Hash over the CONTENT of the loaded cfg dict (not over the file --
    detects content changes of the effective config, ignores pure
    formatting/comments in the YAML). sort_keys ensures determinism
    independent of the key order in the dict.
    """
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _git_commit() -> Optional[str]:
    """Best-effort git commit hash. Neither server nor cluster currently
    use git versioning for this code -- no repo found is the
    EXPECTED normal case here, not an error. Returns None instead of
    throwing; the manifest value is then explicitly 'null' (JSON), not
    silently omitted.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _scene_order_hash(ds) -> str:
    """Hash over the token sequence in pack order (iter_pack_order).
    Two packs with identical hash are guaranteed to have the same frame
    order -- changes to sorting/filtering/scene separation
    show up immediately and visibly.
    """
    hasher = hashlib.sha256()
    for _row, frame in ds.iter_pack_order():
        hasher.update(frame["token"].encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


def _source_root(sources) -> str:
    """Best-effort common root path of all latent_dirs of a split,
    only for the manifest documentation (no functional use). Falls back to
    the full list if no common path exists (e.g.
    different drives).
    """
    import os
    latent_dirs = [str(Path(ld).resolve()) for _pkl, ld in sources]
    if len(latent_dirs) == 1:
        return latent_dirs[0]
    try:
        return os.path.commonpath(latent_dirs)
    except ValueError:
        return "; ".join(latent_dirs)


def _sources_and_outdir(data_cfg: dict, split: str):
    """Determines (sources, out_dir) from the data block depending on mode/split."""
    mode = data_cfg.get("mode", "split")

    def _as_tuples(key):
        return [(s["pkl_path"], s["latent_dir"]) for s in data_cfg.get(key, [])]

    if mode == "explicit":
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"explicit config: --split must be train/val/test, not '{split}'."
            )
        sources = _as_tuples(f"{split}_sources")
        out_dir = data_cfg.get(f"{split}_packed_dir")
    elif mode == "split":
        if split != "all":
            raise ValueError(
                "split config: --split must be 'all' (ONE shared pack for "
                "the internally split source)."
            )
        sources = _as_tuples("sources")
        out_dir = data_cfg.get("packed_dir")
    else:
        raise ValueError(f"data.mode must be 'explicit' or 'split', not '{mode}'.")

    if not sources:
        raise ValueError(f"No sources found for split='{split}' in the config.")
    if not out_dir:
        raise ValueError(
            f"No packed_dir set for split='{split}' in the config "
            f"(expected: {'%s_packed_dir' % split if mode == 'explicit' else 'packed_dir'})."
        )
    return sources, out_dir


def pack_split(cfg: dict, split: str, dtype_str: str = "float32", force: bool = False,
               verify_samples: int = 500, log_every: int = 2000):
    """Packs ONE split from an already-loaded cfg dict.

    Used equally by train_linux.py (auto-pack at startup) and by the CLI
    wrapper pack() -- one implementation, no second code path.
    """
    if dtype_str not in DTYPES:
        raise ValueError(f"packed_dtype must be float32 or float16, not '{dtype_str}'.")
    store_dtype = DTYPES[dtype_str]

    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    n_input_frames = model_cfg.get("n_frames", 3)
    scene_gap_sec = data_cfg.get("scene_gap_sec", 2.0)

    sources, out_dir = _sources_and_outdir(data_cfg, split)
    out_dir = Path(out_dir)
    out_path = out_dir / "packed.npy"

    print(f"[pack] Split  : {split}  (mode={data_cfg.get('mode', 'split')})", flush=True)
    print(f"[pack] dtype  : {dtype_str}", flush=True)
    print(f"[pack] Target : {out_path}", flush=True)

    manifest_path = out_dir / "manifest.json"

    # Idempotency: if the pack already exists, skip the step (cheap, BEFORE
    # the dataset build). Only read the header (mmap_mode='r' reads no data).
    # The runtime guard (shape[0] == N) catches an old pack that does not
    # match the current config during training anyway. --force forces a rebuild.
    if out_path.exists() and not force:
        existing = np.load(str(out_path), mmap_mode="r")
        print(f"[pack] SKIP: already exists "
              f"(shape={tuple(existing.shape)}, dtype={existing.dtype}).", flush=True)
        print(f"[pack]       Create anew with --force.", flush=True)
        del existing
        if not manifest_path.exists():
            print(f"[pack] WARNING: {manifest_path.name} is missing next to an existing "
                  f"packed.npy (probably packed before the manifest was introduced). "
                  f"Re-generate with --force if the audit artifact is needed.",
                  flush=True)
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset instance WITHOUT packed_dir -> reads .npy, assigns _packed_row.
    ds_cfg = BEVDatasetConfig(
        sources=sources,
        n_input_frames=n_input_frames,
        scene_gap_sec=scene_gap_sec,
        h5_path=None,
        packed_dir=None,          # deliberately: here we WRITE, we do not read
    )
    ds = BEVLatentDataset(ds_cfg, scene_indices=None)

    n_frames = ds._n_packed_frames
    if n_frames == 0:
        raise RuntimeError("Dataset has 0 frames -- nothing to pack.")

    # Determine shape/dtype from the first frame (do not hard-assume [256,128,128]).
    _, first_frame = next(ds.iter_pack_order())
    sample = ds._load_latent(first_frame)          # [C,H,W] float32
    frame_shape = sample.shape
    print(f"[pack] Frames : {n_frames}", flush=True)
    print(f"[pack] Shape  : {frame_shape} per frame", flush=True)

    total_bytes = n_frames * int(np.prod(frame_shape)) * np.dtype(store_dtype).itemsize
    print(f"[pack] Size   : ~{total_bytes / 1e9:.1f} GB", flush=True)

    mm = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=store_dtype,
        shape=(n_frames, *frame_shape),
    )

    written = 0
    seen_rows = set()
    t0 = time.time()
    for row, frame in ds.iter_pack_order():
        arr = ds._load_latent(frame)               # .npy-Pfad, [C,H,W] float32
        if arr.shape != frame_shape:
            raise RuntimeError(
                f"Inconsistent frame shape in row {row}: {arr.shape} != {frame_shape}"
            )
        mm[row] = arr.astype(store_dtype)
        seen_rows.add(row)
        written += 1
        if written % log_every == 0:
            dt = time.time() - t0
            rate = written / dt
            eta = (n_frames - written) / rate if rate > 0 else float("nan")
            print(f"[pack]   {written}/{n_frames}  ({rate:.0f} f/s, ETA {eta/60:.1f} min)",
                  flush=True)

    mm.flush()
    del mm  # close the memmap before re-opening for verification

    # Completeness: was each row 0..N-1 written exactly once?
    if len(seen_rows) != n_frames or min(seen_rows) != 0 or max(seen_rows) != n_frames - 1:
        raise RuntimeError(
            f"Row gap: wrote {len(seen_rows)} unique rows, "
            f"expected {n_frames} gapless 0..{n_frames - 1}."
        )
    print(f"[pack] Done   : {written} frames in {(time.time() - t0)/60:.1f} min", flush=True)

    # ------------------------------------------------------------------
    # Inner correctness check (sampling): .npy path vs. packed path
    # ------------------------------------------------------------------
    print(f"[verify] Sampling {verify_samples} frames (.npy vs packed)...", flush=True)
    packed_arr = np.load(str(out_path), mmap_mode="r")
    rng = np.random.default_rng(0)
    all_rows = [row for row, _ in ds.iter_pack_order()]
    frames_by_row = {row: frame for row, frame in ds.iter_pack_order()}
    check_rows = rng.choice(all_rows, size=min(verify_samples, n_frames), replace=False)

    max_abs = 0.0
    n_exact = 0
    for row in check_rows:
        frame = frames_by_row[int(row)]
        ref = ds._load_latent(frame)                          # .npy -> float32
        got = packed_arr[int(row)].astype("float32")          # packed -> float32
        diff = float(np.max(np.abs(ref - got))) if ref.size else 0.0
        max_abs = max(max_abs, diff)
        if diff == 0.0:
            n_exact += 1

    print(f"[verify] max |Delta| = {max_abs:.3e}", flush=True)
    print(f"[verify] exact match: {n_exact}/{len(check_rows)}", flush=True)
    if store_dtype == np.float32:
        if max_abs != 0.0:
            raise RuntimeError(
                f"float32 pack is NOT bit-identical (max|Delta|={max_abs:.3e}) "
                f"-- this must not happen with lossless copying."
            )
        print("[verify] float32: bit-identical. OK.", flush=True)
    else:
        print(f"[verify] float16: rounding error expected (max|Delta|={max_abs:.3e}). "
              f"The outer anchor run decides.", flush=True)

    # ------------------------------------------------------------------
    # Manifest -- ONLY after a passed verify,
    # i.e. manifest.json exists only for a validated pack.
    # Audit/doc artifact: NOT read as a runtime index (see
    # module header) -- purely informational + to detect silent mismatches.
    # ------------------------------------------------------------------
    print(f"[manifest] Computing scene-order hash...", flush=True)
    manifest = {
        "split":            split,
        "dtype":            dtype_str,
        "shape":            [n_frames, *frame_shape],
        "num_frames":       n_frames,
        "config_hash":      _config_hash(cfg),
        "git_commit":       _git_commit(),          # None (JSON: null) if no git repo
        "source_root":      _source_root(sources),
        "scene_order_hash": _scene_order_hash(ds),
        "created_at":       datetime.now(timezone.utc).isoformat(),
        # Extra fields beyond the work-plan minimum set, same
        # effort, increase the audit value:
        "verify_samples":   int(len(check_rows)),
        "verify_max_abs_diff": max_abs,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] Written: {manifest_path}", flush=True)
    if manifest["git_commit"] is None:
        print(f"[manifest]   git_commit=null (no git repo -- server/cluster "
              f"currently run without versioning, expected)", flush=True)

    print(f"\n[pack] packed_dir for the config: {out_dir}", flush=True)
    return out_path


def pack(config_path: str, split: str, dtype_str: str = "float32", force: bool = False,
         verify_samples: int = 500):
    """CLI wrapper: loads the YAML and packs one split (standalone usage)."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return pack_split(cfg, split, dtype_str, force=force, verify_samples=verify_samples)


def _configured_splits(data_cfg: dict):
    """Which splits need to be packed for the current mode?"""
    mode = data_cfg.get("mode", "split")
    if mode == "explicit":
        # test only when both sources and packed_dir are set
        splits = ["train", "val"]
        if data_cfg.get("test_sources") and data_cfg.get("test_packed_dir"):
            splits.append("test")
        return splits
    return ["all"]


def ensure_packed(cfg: dict, dtype_str: str = "float32", force: bool = False,
                  verify_samples: int = 500):
    """Packs all splits required by the current data.mode if they do not yet
    exist (idempotent -- the skip check applies per split). Called at
    training start so that packing is a switchable part of the train cycle
    instead of a separate pipeline. If the pack already exists, the call is quasi
    free (header read only, no dataset build).
    """
    data_cfg = cfg["data"]
    for split in _configured_splits(data_cfg):
        pack_split(cfg, split, dtype_str, force=force, verify_samples=verify_samples)


def main():
    ap = argparse.ArgumentParser(description="Pack BEV latents into a memmap container.")
    ap.add_argument("--config", required=True, help="Path to the YAML config")
    ap.add_argument("--split", required=True,
                    help="explicit: train/val/test  |  split config: all")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES.keys()),
                    help="storage dtype (float32=lossless, float16=half the storage)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing packed.npy instead of skipping")
    ap.add_argument("--verify_samples", type=int, default=500,
                    help="number of frames for the inner correctness check")
    args = ap.parse_args()
    pack(args.config, args.split, args.dtype, force=args.force,
         verify_samples=args.verify_samples)


if __name__ == "__main__":
    main()
