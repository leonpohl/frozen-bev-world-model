# SPDX-FileCopyrightText: 2026 Leon Pohl <leon.pohl@unibw.de>
#
# SPDX-License-Identifier: MIT
PURE_MAPS = False   # set via --pure-maps (segmentation-only display)
"""render_hero.py — README hero video: 6-camera strip + LiDAR-BEV duett.

Layout (10.0 x 6.15 in @ 180 dpi):
  top     : 6 camera thumbnails (nuScenes standard order, labeled)
  bottom  : left  = world-model forecast (mask overlay on the TARGET frame's
                    LiDAR cloud, GT boxes into the target frame, mIoU badge)
            right = real latent (reference), same cloud/boxes
Convention (validated against mmdet3d reference renders, see the paper
demo): all frames in nuScenes LiDAR coordinates (x = right, y = forward,
z = up); GT boxes w = b[3] (X extent), l = b[4] (Y extent), yaw CCW on z;
BEV mask shown with np.rot90(mask, k=1) over extent +-54.4 m (200 px seg
grid, 0.544 m/px). Ground truth = target-frame boxes (lidar frame).
Bonus overlays (honest by design): the LiDAR cloud is drawn in grayscale
by INTENSITY and labeled as a real measurement (the world model forecasts
only the colored BEV semantics); the forecast panel also shows the model's
gate-alpha map (per-cell predict intensity, inferno) and a mIoU trend
inset; scene selection = disclosed gate-alpha window rule, full horizon
decay visible.

Input: NPZ from scripts/rollout_compute.py.
Output: <out>/hero_<name>.mp4 (2 fps, H.264 yuv420p) + frames dir.

    python3 scripts/render_hero.py --npz iv_rollout.npz \
        --data-root /data/nuscenes/complete --out img --name dynamic
"""
import argparse
import pickle
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"]
EXT = 54.4
CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT = CAM_ORDER[:3]
CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT = CAM_ORDER[3:]
BOX_COLOR = "#FF3D3D"
VEH_CLASSES = {"car", "truck", "bus", "trailer", "construction_vehicle",
               "pedestrian", "motorcycle", "bicycle"}
BG = "#ffffff"


def mask_img(mask, alpha):
    rgba = np.zeros((mask.shape[1], mask.shape[2], 4))
    for c_ in range(mask.shape[0]):
        rgb = np.array(matplotlib.colors.to_rgb(OI[c_]))
        rgba[mask[c_]] = np.append(rgb, alpha)
    return np.rot90(rgba, k=1)


def gate_img(gate_map, alpha):
    """Per-cell gate alpha [128,128] -> RGBA heatmap (inferno).

    Self-normalizing: opacity and intensity scale with the map's own 95th
    percentile so the layer only lights up where the model actually
    predicts (background alpha stays transparent)."""
    p95 = np.percentile(gate_map, 95)
    norm = np.clip(gate_map / max(p95, 1e-3), 0, 1)
    cmap = plt.get_cmap("inferno")                             # workstation mpl 3.x
    rgba = np.zeros(gate_map.shape + (4,))
    rgba[..., :3] = cmap(norm)[..., :3]
    rgba[..., 3] = np.where(norm > 0.12, alpha * norm, 0.0)
    return np.rot90(rgba, k=1)


def render(npz, name, data_root, out_dir):
    d = np.load(npz, allow_pickle=True)
    tokens = [str(t) for t in d[f"{name}_tokens"]]
    cams = [str(c) for c in d[f"{name}_cams"]]
    n_input, K = int(d["n_input"]), int(d["k"])
    L = n_input + K
    infos = pickle.load(open(Path(data_root) / "nuscenes_infos_val.pkl", "rb"))["infos"]
    BY = {i["token"]: i for i in infos}

    frames = Path(out_dir) / f"hero_frames_{name}"
    frames.mkdir(exist_ok=True, parents=True)
    for f in frames.glob("frame*.png"):
        f.unlink()

    fig = plt.figure(figsize=(11.0, 10.6), dpi=180)
    fig.patch.set_facecolor(BG)
    # camera positions mirror the physical rig: front-left and front-right
    # flank the big front view; the back trio sits below in spatial order.
    ax_front = fig.add_axes((0.295, 0.735, 0.41, 0.208))
    ax_fl = fig.add_axes((0.012, 0.770, 0.235, 0.138))
    ax_fr = fig.add_axes((0.753, 0.770, 0.235, 0.138))
    ax_bl = fig.add_axes((0.012, 0.545, 0.306, 0.1685))
    ax_b = fig.add_axes((0.347, 0.545, 0.306, 0.1685))
    ax_br = fig.add_axes((0.682, 0.545, 0.306, 0.1685))
    gsb = fig.add_gridspec(1, 2, left=0.004, right=0.996, top=0.528,
                           bottom=0.01, wspace=0.06)
    ax_f = fig.add_subplot(gsb[0])
    ax_r = fig.add_subplot(gsb[1])
    title = fig.suptitle("", fontsize=23, y=0.99)

    # mIoU trend inset (dashboard style, top-right of the forecast panel)
    ins = fig.add_axes((0.30, 0.225, 0.19, 0.115))
    ins.set_facecolor("white")
    ins.patch.set_alpha(0.88)
    ins.set_xticks([]); ins.set_yticks([])
    miou_full = d[f"{name}_mious"]
    ins.plot(np.arange(1, K + 1), miou_full, color="#B25A00", lw=1.4)
    ins.set_xlim(1, K)
    ins.set_ylim(0, max(0.05, float(miou_full.max()) * 1.15))
    ins.text(0.03, 0.85, "mIoU (proxy)", transform=ins.transAxes,
             fontsize=10.5, color="#774400")

    for step in range(L):
        is_ctx = step < n_input
        tok = tokens[step]
        info = BY[tok]

        rig_labels = {"CAM_FRONT": "front", "CAM_FRONT_LEFT": "front left",
                      "CAM_FRONT_RIGHT": "front right", "CAM_BACK": "back",
                      "CAM_BACK_LEFT": "back left", "CAM_BACK_RIGHT": "back right"}
        rig = [(ax_front, CAM_FRONT), (ax_fl, CAM_FRONT_LEFT),
               (ax_fr, CAM_FRONT_RIGHT), (ax_bl, CAM_BACK_LEFT),
               (ax_b, CAM_BACK), (ax_br, CAM_BACK_RIGHT)]
        for ax, cam in rig:
            rel = info["cams"][cam]["data_path"]
            if rel.startswith("./data/nuscenes/"):
                rel = rel[len("./data/nuscenes/"):]
            ax.clear()
            ax.imshow(plt.imread(Path(data_root) / rel))
            axs_title = rig_labels[cam]
            ax.set_title(axs_title, fontsize=12.5 if ax is not ax_front else 15,
                         pad=3.5, color="#333333")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#222222"); sp.set_linewidth(1.0)
            for sp in ax.spines.values():
                sp.set_color("#222222"); sp.set_linewidth(1.0)

        p = np.fromfile(Path(data_root) / info["lidar_path"].replace("./data/nuscenes/", ""),
                        dtype=np.float32).reshape(-1, 5)
        sel = (np.abs(p[:, 0]) < EXT) & (np.abs(p[:, 1]) < EXT)
        P = p[sel]

        for ax, panel_name in ((ax_f, "forecast"), (ax_r, "real")):
            ax.clear()
            if panel_name != "forecast" and not PURE_MAPS:
                # LiDAR = measured ground truth; pure-maps mode hides it too
                ax.add_patch(plt.Rectangle((-EXT, -EXT), 2 * EXT, 2 * EXT,
                             fill=True, facecolor="#0d0d12", zorder=0))
                ax.scatter(P[:, 0], P[:, 1], c=1.0 - np.clip(P[:, 3], 0, 1),
                           cmap="gray", s=0.40,
                           vmin=0.0, vmax=1.0, zorder=1)
            if is_ctx:
                mask = d[f"{name}_ctx"][step]
            else:
                k = step - n_input + 1
                mask = d[f"{name}_pred"][k - 1] if panel_name == "forecast" \
                    else d[f"{name}_real"][k - 1]
            ax.imshow(mask_img(mask, 0.50), extent=(-EXT, EXT, -EXT, EXT), zorder=3)
            # GT boxes only on the real/reference panel and NOT in
            # pure-maps mode (there: only decoded maps are shown).
            for b_idx, (b, keep) in enumerate([(b_, panel_name == "real")
                                               for b_ in info["gt_boxes"]]):
                if PURE_MAPS or b[0] < -1e9 or not keep \
                        or info["gt_names"][b_idx] not in VEH_CLASSES:
                    continue
                c_, s_ = np.cos(b[6]), np.sin(b[6])
                Rz = np.array([[c_, -s_], [s_, c_]])
                poly = (np.array([(-b[3]/2, -b[4]/2), (b[3]/2, -b[4]/2),
                                  (b[3]/2, b[4]/2), (-b[3]/2, b[4]/2)]) @ Rz.T
                        + b[:2])
                ax.plot(np.append(poly[:, 0], poly[0, 0]),
                        np.append(poly[:, 1], poly[0, 1]),
                        color=BOX_COLOR, lw=1.4, zorder=5)
            # ego anchor (front = up)
            ax.plot([0, 0], [0, 2.2], color="#222222", lw=2.0, zorder=6)
            ax.add_patch(plt.Rectangle((-0.95, -1.9), 1.9, 4.4,
                         fill=False, edgecolor="#222222", lw=1.8, zorder=6))
            ax.set_xlim(-EXT, EXT); ax.set_ylim(-EXT, EXT)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#222222"); sp.set_linewidth(1.4)

        if is_ctx:
            ins.set_visible(False)
            for ax in (ax_f, ax_r):
                ax.text(0.03, 0.05, "observed BEV\nLiDAR = real measurement",
                        transform=ax.transAxes,
                        fontsize=15.5, color="#111111",
                        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.88))
            title.set_text("BEV world model (nuScenes) — observation  |  "
                           f"t = {0.5 * step:.1f} s")
        else:
            k = step - n_input + 1
            mm = float(d[f"{name}_mious"][k - 1])
            ga = float(d[f"{name}_mean_alpha"])
            ins.set_visible(True)
            ins.clear()
            ins.set_facecolor("white")
            ins.patch.set_alpha(0.88)
            ins.set_xticks([]); ins.set_yticks([])
            ins.plot(np.arange(1, K + 1), miou_full, color="#B25A00", lw=1.4)
            ins.set_xlim(1, K)
            ins.set_ylim(0, max(0.05, float(miou_full.max()) * 1.15))
            ins.plot([k], [miou_full[k - 1]], "o", color="#B25A00", ms=6)
            ins.text(0.03, 0.85, "mIoU (proxy)", transform=ins.transAxes,
                     fontsize=10.5, color="#774400")
            ax_f.text(0.03, 0.05,
                      "forecast (world model)\n"
                      f"mIoU {mm:.2f} · gate ᾱ {ga:.2f}",
                      transform=ax_f.transAxes, fontsize=15.5, color="#B25A00",
                      bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.88))
            ax_r.text(0.03, 0.05,
                      "real latent (reference)\n" +
                      ("maps = decoded from the real latent"
                       if PURE_MAPS else "red boxes = actual traffic (GT)"),
                      transform=ax_r.transAxes, fontsize=15.5, color="#111111",
                      bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.88))
            title.set_text("BEV world model (nuScenes) — "
                           f"rollout step {k}/{K}  |  t = {0.5 * step:.1f} s")

        fig.savefig(frames / f"frame{step:03d}.png", dpi=fig.dpi)
        print("[hero] frame%03d" % step, flush=True)
    plt.close(fig)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="img")
    ap.add_argument("--name", default="dynamic")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--pure-maps", action="store_true",
                    help="segmentation-only example: no LiDAR/boxes anywhere "
                         "(both panels = decoded maps; nothing unpredicted)")
    args = ap.parse_args()
    global PURE_MAPS
    PURE_MAPS = args.pure_maps

    frames = render(args.npz, args.name, args.data_root, args.out)
    mp4 = Path(args.out) / f"readme_teaser{'' if args.name == 'dynamic' else '_intersection'}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", str(args.fps),
                    "-i", str(frames / "frame%03d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                    "-movflags", "+faststart", str(mp4)], check=True)
    print("[hero] %s" % mp4)


if __name__ == "__main__":
    main()
