# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
build_hdf5_cache.py -- BEV World Model: HDF5 cache builder
======================================================================

Converts precomputed BEV latents (.npy files) into a single HDF5 file.
The HDF5 format enables fast random access to individual frames without
having to keep all latents in RAM.

HDF5 structure:
    latents/<token>   float32 [256, 128, 128]   -- one dataset per frame
    meta/tokens       byte array of all tokens   -- for completeness check

Invocation (val set):
    python data/build_hdf5_cache.py \\
        --pkl   nuscenes_infos_val.pkl \\
        --npy   /data/Latents/Val \\
        --out   /data/Cache/bev_latents_val.h5

Invocation (multiple sources in one step):
    python data/build_hdf5_cache.py \\
        --pkl   infos_train.pkl  infos_val.pkl \\
        --npy   /Latents/Train   /Latents/Val \\
        --out   /Cache/bev_latents_all.h5

Why HDF5 instead of individual .npy files?
    - A single open() call instead of thousands of np.load() calls
    - OS cache works better with sequential access
    - Compression reduces storage requirements (gzip ~30-50% for BEV latents)
    - Portable: share one file instead of thousands of individual files
    - h5py access is thread-safe with swmr=True (important for DataLoaders
      with num_workers > 0)
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# h5py is lazily imported in build_cache() and verify_cache()


# ---------------------------------------------------------------------------
# Helper: collect all frame tokens from the PKL(s)
# ---------------------------------------------------------------------------

def _collect_tokens(
    pkl_paths: List[str],
    npy_dirs: List[str],
) -> List[Tuple[str, Path]]:
    """
    Reads all PKLs and returns a list of (token, npy_path) pairs.
    Tokens are deduplicated -- if the same PKL is accidentally given
    twice, each token appears only once.

    Args:
        pkl_paths: list of paths to nuscenes_infos_*.pkl files
        npy_dirs:  list of folders with .npy latents, parallel to pkl_paths

    Returns:
        List of (token, npy_path) pairs, sorted by token
    """
    seen   = set()
    result = []

    for pkl_path, npy_dir in zip(pkl_paths, npy_dirs):
        npy_dir = Path(npy_dir)

        with open(pkl_path, "rb") as f:
            infos = pickle.load(f)["infos"]

        for info in infos:
            token = info["token"]
            if token in seen:
                continue
            seen.add(token)

            npy_path = npy_dir / f"bev_latent_{token}.npy"
            if not npy_path.exists():
                print(f"  [Warning] .npy not found: {npy_path} -- skipped")
                continue

            result.append((token, npy_path))

    return sorted(result, key=lambda x: x[0])  # sorted deterministically


# ---------------------------------------------------------------------------
# Main function: build the HDF5 cache
# ---------------------------------------------------------------------------

def build_cache(
    pkl_paths:    List[str],
    npy_dirs:     List[str],
    out_path:     str,
    compression:  str = "gzip",
    comp_level:   int = 4,
    chunk_shape:  Tuple[int, int, int] = (256, 128, 128),
    overwrite:    bool = False,
) -> None:
    """
    Builds an HDF5 cache file from .npy BEV latents.

    Args:
        pkl_paths:   list of PKL files (defines which tokens are expected)
        npy_dirs:    list of .npy folders, parallel to pkl_paths
        out_path:    target path for the .h5 file
        compression: HDF5 compression algorithm ('gzip', 'lzf', None)
                     gzip: best compression, somewhat slower
                     lzf:  faster, somewhat worse compression
                     None: no overhead, maximum I/O speed
        comp_level:  compression level for gzip (1=fast, 9=small)
        chunk_shape: HDF5 chunk size. One chunk = one frame [256,128,128].
                     Important: the chunk size determines the minimal read amount.
                     For frame-wise access, optimal = one frame per chunk.
        overwrite:   overwrite an existing file?
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        if overwrite:
            print(f"[build_cache] Overwriting: {out_path}")
            out_path.unlink()
        else:
            raise FileExistsError(
                f"{out_path} already exists. Use --overwrite to overwrite."
            )

    # Collect all tokens
    print("[build_cache] Collecting tokens from PKL(s)...")
    token_paths = _collect_tokens(pkl_paths, npy_dirs)
    n = len(token_paths)
    print(f"[build_cache] {n} frames found. Starting conversion...")
    print(f"  Compression: {compression or 'none'} (level {comp_level})")
    print(f"  Chunk-Shape: {chunk_shape}")
    print(f"  Target:      {out_path}")
    print()

    try:
        import h5py
    except ImportError:
        raise ImportError("h5py not installed. Please: pip install h5py")

    t_start = time.time()
    compress_kwargs = {}
    if compression:
        compress_kwargs = {"compression": compression, "compression_opts": comp_level}

    with h5py.File(out_path, "w") as f:

        # Group for latents
        grp = f.create_group("latents")

        # Group for metadata
        meta = f.create_group("meta")
        meta.attrs["n_frames"]    = n
        meta.attrs["latent_shape"] = list(chunk_shape)
        meta.attrs["dtype"]       = "float32"
        meta.attrs["source_pkls"] = str(pkl_paths)

        # Store all tokens as byte array (completeness check)
        token_bytes = [t.encode("utf-8") for t, _ in token_paths]
        meta.create_dataset(
            "tokens",
            data=np.array(token_bytes, dtype=h5py.special_dtype(vlen=bytes)),
        )

        # Write frames
        for i, (token, npy_path) in enumerate(token_paths):
            arr = np.load(npy_path).astype("float32")

            # BEVFusion stores (1, 256, 128, 128) -- remove batch dimension
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr.squeeze(axis=0)   # -> [256, 128, 128]

            if arr.shape != chunk_shape:
                raise ValueError(
                    f"Unexpected shape {arr.shape} for token {token}. "
                    f"Expected: {chunk_shape}"
                )

            grp.create_dataset(
                token,
                data=arr,
                chunks=chunk_shape,
                **compress_kwargs,
            )

            # Progress
            if (i + 1) % 10 == 0 or i == n - 1:
                elapsed = time.time() - t_start
                fps     = (i + 1) / elapsed
                eta     = (n - i - 1) / fps if fps > 0 else 0
                mb_done = (i + 1) * 16.0   # ~16 MB per frame (uncompressed)
                print(
                    f"  [{i+1:4d}/{n}]  "
                    f"{elapsed:5.1f}s  "
                    f"{fps:.1f} frames/s  "
                    f"ETA {eta:.0f}s  "
                    f"~{mb_done/1024:.1f} GB processed"
                )

    # Final report
    elapsed   = time.time() - t_start
    h5_size   = out_path.stat().st_size / 1e9
    raw_size  = n * 16.0 / 1024   # GB uncompressed
    ratio     = raw_size / h5_size if h5_size > 0 else 1.0

    print()
    print(f"[build_cache] Done in {elapsed:.1f}s")
    print(f"  Frames:          {n}")
    print(f"  Uncompressed:    {raw_size:.2f} GB")
    print(f"  HDF5 size:       {h5_size:.2f} GB")
    print(f"  Compression ratio:{ratio:.1f}x")
    print(f"  Saved:           {out_path}")


# ---------------------------------------------------------------------------
# Completeness check: verify the HDF5 against the PKLs
# ---------------------------------------------------------------------------

def verify_cache(h5_path: str, pkl_paths: List[str], npy_dirs: List[str]) -> bool:
    """
    Checks whether the HDF5 file contains all expected tokens.
    To do this, reads the metadata from the HDF5 and compares it with the PKLs.

    Returns:
        True when complete, False when tokens are missing.
    """
    print(f"[verify_cache] Checking: {h5_path}")

    expected_tokens = set(t for t, _ in _collect_tokens(pkl_paths, npy_dirs))

    try:
        import h5py
    except ImportError:
        raise ImportError("h5py not installed. Please: pip install h5py")

    with h5py.File(h5_path, "r") as f:
        cached_tokens = set(f["latents"].keys())
        n_frames      = f["meta"].attrs["n_frames"]
        latent_shape  = tuple(f["meta"].attrs["latent_shape"])

    print(f"  Expected:  {len(expected_tokens)} tokens")
    print(f"  In cache:  {len(cached_tokens)} tokens")
    print(f"  Latent-Shape: {latent_shape}")

    missing = expected_tokens - cached_tokens
    extra   = cached_tokens - expected_tokens

    if missing:
        print(f"  [ERROR] {len(missing)} tokens missing in the cache!")
        for t in sorted(missing)[:5]:
            print(f"    {t}")
        return False

    if extra:
        print(f"  [Warning] {len(extra)} unexpected tokens in the cache (from another PKL?)")

    print(f"  [OK] Cache complete.")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts BEV latent .npy files into an HDF5 cache."
    )
    parser.add_argument(
        "--pkl", nargs="+", required=True,
        help="Path(s) to nuscenes_infos_*.pkl (space-separated for multiple)"
    )
    parser.add_argument(
        "--npy", nargs="+", required=True,
        help="Path(s) to .npy latent folders (parallel to --pkl)"
    )
    parser.add_argument(
        "--out", required=True,
        help="Target path for the .h5 file (e.g. /data/Cache/bev_latents_val.h5)"
    )
    parser.add_argument(
        "--compression", default="gzip", choices=["gzip", "lzf", "none"],
        help="Compression algorithm (default: gzip)"
    )
    parser.add_argument(
        "--level", type=int, default=4,
        help="Compression level for gzip 1-9 (default: 4)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing .h5 file"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only run the completeness check, no rebuild"
    )

    args = parser.parse_args()

    if len(args.pkl) != len(args.npy):
        parser.error("Number of --pkl and --npy entries must be equal.")

    compression = None if args.compression == "none" else args.compression

    if args.verify_only:
        ok = verify_cache(args.out, args.pkl, args.npy)
        exit(0 if ok else 1)

    build_cache(
        pkl_paths=args.pkl,
        npy_dirs=args.npy,
        out_path=args.out,
        compression=compression,
        comp_level=args.level,
        overwrite=args.overwrite,
    )

    # Verify automatically after building
    verify_cache(args.out, args.pkl, args.npy)
