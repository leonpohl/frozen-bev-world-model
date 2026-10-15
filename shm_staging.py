# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
shm_staging.py -- BEV World Model: node-local /dev/shm staging
==============================================================================

Copies the packed memmap files (packed.npy + manifest.json) once per
job start from BeeGFS (network) into the node-local RAM filesystem /dev/shm and
redirects the packed_dir paths of the config there. Afterwards the DataLoader
reads at RAM speed instead of over the shared BeeGFS network link.

WHY (cluster finding, anchor run):
    Despite memmap packing, the data_time share was ~89% (epoch time ~3100s
    vs. local ~860s) -- the A100 idles because BeeGFS does not deliver the
    data fast enough. /dev/shm takes the network entirely out of the loop.

INTEGRATION (one switch, analogous to use_packed):
    data.stage_to_shm: true   -> train_linux.py calls stage_and_rewrite();
                                 files are copied to /dev/shm (idempotent),
                                 the *_packed_dir keys of the config point
                                 to /dev/shm afterwards. Requires data.use_packed: true.
    data.stage_to_shm: false/absent -> no staging, config unchanged.

CLEANUP STRATEGY (deliberately modular -- approach A now, B/C extensible later):
    The strategy is interchangeable via CLEANUP_STRATEGIES + data.shm_cleanup,
    without touching the copy/redirect logic:
      "job_local" (approach A, DEFAULT): each job copies into its OWN
          directory /dev/shm/<user>_<jobid>/ and deletes ONLY that at
          job end (trap in the sbatch script calls cleanup()). Conflict-free,
          no sharing -> with several parallel fp32 jobs on one node
          it may not fit into /dev/shm (2x535GB > 1008GB); with fp16 it does.
      "shared_refcount" (approach B, NOT implemented): shared directory
          with reference counting, only the last job deletes.
      "shared_manual" (approach C, NOT implemented): shared directory,
          no auto-cleanup, separate cleanup command.
    To activate B/C later: register a new strategy function and
    set data.shm_cleanup -- train_linux.py and the rest remain unchanged.

USAGE (by train_linux.py, not directly):
    import shm_staging
    shm_staging.stage_and_rewrite(cfg)          # redirects cfg["data"][*_packed_dir]
    ...
    shm_staging.cleanup(cfg)                     # at job end (also via sbatch-trap)

STANDALONE STAGING (pre-stage before parallel runs):
    python -u shm_staging.py --stage --config <cfg.yaml>   # copies once to /dev/shm

STANDALONE CLEANUP (for sbatch-trap / manual cleanup):
    python -u shm_staging.py --cleanup --config <cfg.yaml>
    python -u shm_staging.py --cleanup --job-dir /dev/shm/vima_123456   # targeted
"""

import argparse
import getpass
import os
import shutil
import time
from pathlib import Path

import yaml


SHM_ROOT = Path("/dev/shm")

# Files staged per split (packed.npy mandatory, manifest.json
# best-effort -- not needed at runtime, but convenient to take along).
_STAGE_FILES = ["packed.npy", "manifest.json"]


# ---------------------------------------------------------------------------
# Path conventions (one place -- cleanup and staging must be consistent)
# ---------------------------------------------------------------------------

def _job_tag() -> str:
    """Unique tag for this job/session.

    On the cluster, SLURM_JOB_ID provides a per-job unique value. Without
    SLURM (local/interactive) fall back to PID -- then the copy is bound to the
    process, which is exactly right for approach A.
    """
    return os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"


def _job_local_dir() -> Path:
    """Approach A: job-own /dev/shm directory /dev/shm/<user>_<tag>/."""
    return SHM_ROOT / f"{getpass.getuser()}_{_job_tag()}"


# Cleanup strategies return the directory(ies) to delete.
# Register new strategies (B/C) here -- the rest of the module only calls
# _target_root(strategy) and the cleanup logic generically.
CLEANUP_STRATEGIES = {
    "job_local": _job_local_dir,
    # Parallel runs: several processes in the SAME job share the one
    # /dev/shm copy (same SLURM_JOB_ID -> same path; memmap pages
    # are shared read-only, ONE RAM copy for all). The cleanup() guard
    # below treats every non-job_local strategy as a no-op -> NO
    # process-end cleanup; the sbatch-trap cleans up at JOB end. Staging must
    # run once sequentially BEFORE the parallel start (the skip check is not
    # race-safe during simultaneous cold starts).
    "job_shared": _job_local_dir,
    # "shared_refcount": _shared_dir,   # approach B -- later
    # "shared_manual":   _shared_dir,   # approach C -- later
}


def _target_root(strategy: str) -> Path:
    if strategy not in CLEANUP_STRATEGIES:
        raise ValueError(
            f"Unknown shm_cleanup strategy '{strategy}'. "
            f"Available: {sorted(CLEANUP_STRATEGIES)}."
        )
    return CLEANUP_STRATEGIES[strategy]()


# ---------------------------------------------------------------------------
# Config helper: which packed_dirs need staging?
# ---------------------------------------------------------------------------

def _packed_dir_keys(data_cfg: dict):
    """Returns the *_packed_dir config keys relevant in the current mode."""
    mode = data_cfg.get("mode", "split")
    if mode == "explicit":
        keys = ["train_packed_dir", "val_packed_dir"]
        if data_cfg.get("test_packed_dir"):
            keys.append("test_packed_dir")
        return keys
    return ["packed_dir"]


def _free_bytes(path: Path) -> int:
    st = os.statvfs(str(path))
    return st.f_bavail * st.f_frsize


# ---------------------------------------------------------------------------
# Core operation: copy + redirect config paths
# ---------------------------------------------------------------------------

def stage_and_rewrite(cfg: dict) -> dict:
    """Stages the packed.npy files to /dev/shm and redirects cfg IN-PLACE.

    Idempotent: if a valid copy exists (packed.npy present, size
    matches), it is NOT copied again -- this way a second job on the same
    node (with shared strategies) or a restart can use the same copy.
    With approach A the target directory is job-own, i.e. practically always fresh.

    Returns cfg (the same instance, changed IN-PLACE).
    """
    if not cfg["data"].get("stage_to_shm", False):
        return cfg

    if not cfg["data"].get("use_packed", False):
        raise ValueError(
            "data.stage_to_shm=true requires data.use_packed=true "
            "(the packed files are staged, not the individual .npy files)."
        )

    data_cfg = cfg["data"]
    strategy = data_cfg.get("shm_cleanup", "job_local")
    target_root = _target_root(strategy)

    if not SHM_ROOT.exists():
        raise RuntimeError(f"{SHM_ROOT} does not exist -- no tmpfs on this node?")

    keys = _packed_dir_keys(data_cfg)

    # --- Size pre-check (fail-fast, before anything is copied) ---
    # /dev/shm is node-local and multi-tenant -> check beforehand whether there is enough free.
    need = 0
    srcs = {}
    for key in keys:
        src_dir = Path(data_cfg[key])
        src_file = src_dir / "packed.npy"
        if not src_file.exists():
            raise FileNotFoundError(
                f"stage_to_shm: {src_file} not found (use_packed should have "
                f"created it -- did ensure_packed run before staging?)."
            )
        size = src_file.stat().st_size
        # Fix: splits already fully staged (job_shared
        # pre-staging in sbatch) do NOT count toward space requirements -- otherwise
        # the later skip-copy case fails at the pre-check as soon as
        # 2x demand > /dev/shm (det packs 567G; never triggered with seg at 286G).
        dst_probe = target_root / src_dir.name / "packed.npy"
        if not (dst_probe.exists() and dst_probe.stat().st_size == size):
            need += size
        srcs[key] = (src_dir, src_file, size)

    free = _free_bytes(SHM_ROOT)
    print(f"[shm] Strategy: {strategy} | Target: {target_root}", flush=True)
    print(f"[shm] Needed: ~{need/1e9:.1f} GB | free in {SHM_ROOT}: "
          f"~{free/1e9:.1f} GB", flush=True)
    # 5% safety buffer so we do not fill the node down to 0
    if need > free * 0.95:
        raise RuntimeError(
            f"stage_to_shm: not enough space in {SHM_ROOT} "
            f"(need ~{need/1e9:.1f} GB, free ~{free/1e9:.1f} GB). "
            f"Node may be occupied by other users. Options: float16 storage "
            f"(half the size), retry later, or request another node."
        )

    # --- Copy (per split), redirect the config key to /dev/shm ---
    for key in keys:
        src_dir, src_file, size = srcs[key]
        # Target subfolder per split so multiple packed.npy do not collide
        dst_dir = target_root / src_dir.name
        dst_file = dst_dir / "packed.npy"

        if dst_file.exists() and dst_file.stat().st_size == size:
            print(f"[shm] {key}: already staged ({dst_file}), skip-copy.", flush=True)
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            for fname in _STAGE_FILES:
                s = src_dir / fname
                if s.exists():
                    t0 = time.time()
                    shutil.copy2(str(s), str(dst_dir / fname))
                    if fname == "packed.npy":
                        dt = time.time() - t0
                        rate = size / dt / 1e9 if dt > 0 else float("nan")
                        print(f"[shm] {key}: {fname} copied "
                              f"({size/1e9:.1f} GB in {dt:.0f}s, {rate:.2f} GB/s)",
                              flush=True)

        # Redirect the config path: runtime now reads from /dev/shm
        data_cfg[key] = str(dst_dir)

    print(f"[shm] Staging done. packed_dir keys point to {target_root}.", flush=True)
    return cfg


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup(cfg: dict = None, strategy: str = None, job_dir: str = None):
    """Deletes this job's staged /dev/shm data.

    Call paths:
      - from train_linux.py at normal job end: cleanup(cfg)
      - from the sbatch-trap (also on scancel/timeout/OOM): via
        `python shm_staging.py --cleanup --config <cfg>` OR
        `--job-dir <path>` (if the config is not available there)

    Approach A ("job_local"): deletes the job-own directory -> conflict-free,
    because no other job uses the same copy. With future shared
    strategies (B/C) the refcount/no-op logic of the
    respective strategy would apply here instead (then solved via the strategy registry).
    """
    if job_dir:
        target = Path(job_dir)
    else:
        if strategy is None:
            strategy = (cfg or {}).get("data", {}).get("shm_cleanup", "job_local") \
                if cfg else "job_local"
        # For shared strategies, the "may I delete?" decision would
        # be made here. Approach A: always yes (job-own).
        if strategy != "job_local":
            print(f"[shm cleanup] Strategy '{strategy}' has no auto-cleanup logic "
                  f"(only 'job_local' implemented) -- nothing deleted.", flush=True)
            return
        target = _job_local_dir()

    if target.exists():
        # Safety net: only delete below /dev/shm, never elsewhere
        try:
            target.resolve().relative_to(SHM_ROOT.resolve())
        except ValueError:
            print(f"[shm cleanup] ABORT: {target} is not below {SHM_ROOT} "
                  f"-- as a safety measure nothing deleted.", flush=True)
            return
        shutil.rmtree(str(target), ignore_errors=True)
        print(f"[shm cleanup] Deleted: {target}", flush=True)
    else:
        print(f"[shm cleanup] Nothing to delete ({target} does not exist).", flush=True)


def main():
    ap = argparse.ArgumentParser(description="/dev/shm staging")
    ap.add_argument("--stage", action="store_true",
                    help="copy the packed files ONCE to /dev/shm "
                         "(pre-stage before parallel runs). Needs --config.")
    ap.add_argument("--cleanup", action="store_true", help="delete staged data")
    ap.add_argument("--config", type=str, default=None,
                    help="config from which paths/cleanup strategy are read")
    ap.add_argument("--job-dir", type=str, default=None,
                    help="concrete /dev/shm directory (overrides strategy; --cleanup only)")
    args = ap.parse_args()

    # Exactly ONE action per call (XOR) -- protects against accidental duplication.
    if args.stage == args.cleanup:
        ap.error("Specify exactly ONE of --stage or --cleanup.")

    cfg = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    if args.stage:
        # one-time, sequential pre-staging BEFORE the parallel start.
        # Fills /dev/shm/<user>_<jobid>/ (same SLURM_JOB_ID as the
        # subsequent training runs -> they then hit the skip-copy, race-free).
        # The returned (redirected) cfg is discarded: each
        # training process redirects its OWN in-memory config itself.
        if cfg is None:
            ap.error("--stage needs --config (paths + target strategy).")
        stage_and_rewrite(cfg)
    else:
        cleanup(cfg=cfg, job_dir=args.job_dir)


if __name__ == "__main__":
    main()
