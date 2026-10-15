# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap

DATA_ROOT = "/bevfusion/data/nuscenes"
OUT_DIR   = Path("/output/gt_masks_seg")
CLASSES   = ["drivable_area", "ped_crossing", "walkway", "stop_line",
             "carpark_area", "divider"]
LOCATIONS = ["singapore-onenorth", "singapore-hollandvillage",
             "singapore-queenstown", "boston-seaport"]

# Latent grid (production) vs. metric grid (gate)
GRID_LATENT = dict(half=51.2, step=0.8)   # -> 128x128
GRID_PIPE   = dict(half=50.0, step=0.5)   # -> 200x200


def build_mappings():
    m = {}
    for name in CLASSES:
        if name == "divider":
            m[name] = ["road_divider", "lane_divider"]
        else:
            m[name] = [name]
    return m


def rasterize(nmaps, info, grid):
    l2e = np.eye(4)
    l2e[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    l2e[:3, 3]  = info["lidar2ego_translation"]
    e2g = np.eye(4)
    e2g[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    e2g[:3, 3]  = info["ego2global_translation"]
    lidar2global = e2g @ l2e                      # lidar_aug = identity

    map_pose = lidar2global[:2, 3]
    patch = 2 * grid["half"]
    patch_box = (map_pose[0], map_pose[1], patch, patch)
    v = lidar2global[:3, :3] @ np.array([1, 0, 0])
    patch_angle = np.arctan2(v[1], v[0]) / np.pi * 180
    canvas = int(patch / grid["step"])

    mappings = build_mappings()
    layer_names = sorted({l for ls in mappings.values() for l in ls})
    masks = nmaps[info["location"]].get_map_mask(
        patch_box=patch_box, patch_angle=patch_angle,
        layer_names=layer_names, canvas_size=(canvas, canvas))
    masks = masks.transpose(0, 2, 1).astype(bool)

    labels = np.zeros((len(CLASSES), canvas, canvas), dtype=np.uint8)
    for k, name in enumerate(CLASSES):
        for layer in mappings[name]:
            labels[k][masks[layer_names.index(layer)]] = 1
    return labels


def run_gate(n, nmaps, infos):
    sys.path.insert(0, "/bevfusion")
    from mmdet3d.datasets.pipelines.loading import LoadBEVSegmentation
    pipe = LoadBEVSegmentation(dataset_root=DATA_ROOT,
                               xbound=[-50.0, 50.0, 0.5],
                               ybound=[-50.0, 50.0, 0.5], classes=CLASSES)
    worst = 1.0
    for info in infos[:n]:
        l2e = np.eye(4)
        l2e[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        l2e[:3, 3]  = info["lidar2ego_translation"]
        e2g = np.eye(4)
        e2g[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        e2g[:3, 3]  = info["ego2global_translation"]
        data = {"lidar_aug_matrix": np.eye(4), "lidar2ego": l2e,
                "ego2global": e2g, "location": info["location"]}
        ref = pipe(data)["gt_masks_bev"].astype(np.uint8)
        own = rasterize(nmaps, info, GRID_PIPE)
        agree = float((ref == own).mean())
        worst = min(worst, agree)
    print(f"[gate] {n} frames, worst pixel agreement: {worst:.6f}")
    if worst < 0.9999:
        sys.exit("[gate] FAILED — geometry deviates, do NOT produce.")
    print("[gate] PASSED — geometry identical to the pipeline.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--split", choices=["val", "train"])
    ap.add_argument("--grid", choices=["latent", "pipe"], default="latent",
                    help="'pipe' = pipeline grid 200x200 (+-50m/0.5m) "
                         "for head training; default 'latent' as before")
    args = ap.parse_args()

    nmaps = {loc: NuScenesMap(DATA_ROOT, loc) for loc in LOCATIONS}

    if args.gate:
        with open(f"{DATA_ROOT}/nuscenes_infos_val.pkl", "rb") as f:
            infos = sorted(pickle.load(f)["infos"], key=lambda x: x["timestamp"])
        run_gate(args.gate, nmaps, infos)
        return

    assert args.split, "pass --gate N or --split val|train"
    with open(f"{DATA_ROOT}/nuscenes_infos_{args.split}.pkl", "rb") as f:
        infos = sorted(pickle.load(f)["infos"], key=lambda x: x["timestamp"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = GRID_PIPE if args.grid == "pipe" else GRID_LATENT
    out, t0 = {}, time.time()
    for i, info in enumerate(infos):
        labels = rasterize(nmaps, info, grid)   # (6,128,128) or (6,200,200)
        out[info["token"]] = np.packbits(labels)
        if (i + 1) % 500 == 0 or (i + 1) == len(infos):
            r = (i + 1) / (time.time() - t0)
            print(f"  [{i+1:>6}/{len(infos)}] {r:5.1f} f/s", flush=True)
    suffix = "_pipe" if args.grid == "pipe" else ""
    path = OUT_DIR / f"gt_masks{suffix}_{args.split}.npz"
    np.savez_compressed(path, **out)
    print(f"[done] {len(out)} masks -> {path}")


if __name__ == "__main__":
    main()
