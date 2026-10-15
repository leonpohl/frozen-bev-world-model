# SPDX-FileCopyrightText: 2026 Leon Pohl <leon.pohl@unibw.de>
#
# SPDX-License-Identifier: MIT
"""
render_forecast_hero_combo.py — detection-and-map forecast hero (README
style: rigged cameras + duet panels + sparkline) for the DET-selected
scene (front-arc traffic; scene selection documented by pick_det_scene.py).

Left panel (PURE world-model forecast): predicted BEV map segmentation
(frozen seg head) + predicted DETECTIONS (frozen TransFusion head; red),
both decoded from the model's own forecast latents; badge = proxy mIoU +
matched GT-class vehicles (<= 2 m; score threshold).
Right panel (real latent, decoded): forecast-free reference; cyan boxes =
detections decoded from the REAL future latent (the oracle); dashed gray =
actual traffic (GT maps context unchanged); badge unchanged (mIoU for the
reference; matched counts for the oracle).

Gray LiDAR = real measurement of the future frame — used only in the static
paper figures (filmstrip), never in these animated GIFs. Ego = sketched at
(0,0), front = up; masks np.rot90(k=1) on ±50 m (200 px seg grid at 0.5 m/px).

    python3 scripts/render_forecast_hero_combo.py \
        --seg-npz iv_rollout_detscene.npz \
        --pred-npz det_scene2_pred.npz --real-npz det_scene2_real.npz \
        --data-root path/to/nuscenes/complete --out img \
        [--agg-json agg_decay.json]   # aggregate decay curves in the sparkline
Output: img/forecast_hero_frames/ + img/detmaps_hero.mp4 (+ GIF).
"""
import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"]
EXT = 50.0   # seg-map mask scope: 200 px at 0.5 m/px (thesis convention)
CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT = CAM_ORDER[:3]
CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT = CAM_ORDER[3:]
DET_COLOR = "#FF3D3D"
ORACLE_COLOR = "#00B7EB"
GT_COLOR = "#666666"
VEH_CLASSES = {"car", "truck", "bus", "trailer", "construction_vehicle",
               "pedestrian", "motorcycle", "bicycle"}
CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer",
           "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
BG = "#ffffff"


def _cent(mask, ext=EXT):
    idx = np.argwhere(mask[0])
    if len(idx) == 0:
        return None
    h, w = mask.shape[1], mask.shape[2]
    # mask pixel frame -> the displayed (rot90) frame: rot90(k=1)
    idx_r = np.stack([idx[:, 1], mask.shape[2] - 1 - idx[:, 0]], axis=1)
    return (idx_r.mean(axis=0) - (np.array([w, h]) / 2)) * (2 * ext / w)


def mask_img(mask, alpha):
    rgba = np.zeros((mask.shape[1], mask.shape[2], 4))
    for c_ in range(mask.shape[0]):
        rgb = np.array(matplotlib.colors.to_rgb(OI[c_]))
        rgba[mask[c_]] = np.append(rgb, alpha)
    return np.rot90(rgba, k=1)


def bev_poly(b):
    c, s = np.cos(b[6]), np.sin(b[6])
    Rz = np.array([[c, -s], [s, c]])
    return (np.array([(-b[3] / 2, -b[4] / 2), (b[3] / 2, -b[4] / 2),
                      (b[3] / 2, b[4] / 2), (-b[3] / 2, b[4] / 2)]) @ Rz.T
            + np.asarray(b[:2]))


def corners8(b):
    c, s = np.cos(b[6]), np.sin(b[6])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    E = np.array([[dx, dy, dz] for dx in (-b[3] / 2, b[3] / 2)
                  for dy in (-b[4] / 2, b[4] / 2) for dz in (-b[5] / 2, b[5] / 2)])
    return E @ Rz.T + b[:3]


def draw_boxes_bev(ax, boxes, color, lw, dashed=False, score_thresh=None,
                   scores=None):
    for i, b in enumerate(boxes):
        if b[0] < -1e9:
            continue
        if score_thresh is not None and scores[i] < score_thresh:
            continue
        pl = bev_poly(b)
        ax.plot(np.append(pl[:, 0], pl[0, 0]), np.append(pl[:, 1], pl[0, 1]),
                color=color, lw=lw, ls="--" if dashed else "-", zorder=5)


def draw_boxes_cam(ax, boxes, R, t, K, color, lw, dashed=False,
                   score_thresh=None, scores=None):
    EDGE8 = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, b in enumerate(boxes):
        if b[0] < -1e9:
            continue
        if score_thresh is not None and scores[i] < score_thresh:
            continue
        Pc = (corners8(b) - t) @ R
        depth = Pc[:, 2]
        if depth.min() <= 0.3:
            continue
        uv = Pc @ K.T
        px = uv[:, 0] / depth
        py = uv[:, 1] / depth
        pts = [(px[j], py[j]) for j in range(8)]
        for (a_, b_) in EDGE8:
            ax.plot([pts[a_][0], pts[b_][0]], [pts[a_][1], pts[b_][1]],
                    color=color, lw=lw, ls="--" if dashed else "-",
                    alpha=0.92, zorder=5)


def match_gt(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_names,
             dist=2.0, score_thresh=0.2):
    found, dsum = 0, 0.0
    for j, g in enumerate(gt_boxes):
        if g[0] < -1e9 or gt_names[j] not in VEH_CLASSES:
            continue
        best = 1e9
        for i, b in enumerate(pred_boxes):
            if b[0] < -1e9 or pred_scores[i] < score_thresh:
                continue
            if CLASSES[int(pred_labels[i])] != gt_names[j]:
                continue
            d = float(np.hypot(b[0] - g[0], b[1] - g[1]))
            best = min(best, d)
        if best <= dist:
            found += 1
            dsum += best
    n = len([n for n in gt_names if n in VEH_CLASSES])
    return found, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-npz", required=True)
    ap.add_argument("--pred-npz", required=True)
    ap.add_argument("--real-npz", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="img")
    ap.add_argument("--score-thresh", type=float, default=0.2)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--k-shown", type=int, default=8,
                    help="render only the first k steps (the drift-tracked "
                         "range); the curve panel always shows the full decay")
    ap.add_argument("--arrows", action="store_true",
                    help="draw the next-step real world-shift arrow on both panels")
    ap.add_argument("--agg-json", default=None,
                    help="optional decay JSON {wm:[..],naiv:[..],ego:[..]} — "
                         "aggregate (validation-wide) curves drawn in the "
                         "corner sparkline for context")
    args = ap.parse_args()

    dg = np.load(args.seg_npz, allow_pickle=True)
    pr = np.load(args.pred_npz, allow_pickle=True)
    rr = np.load(args.real_npz, allow_pickle=True)
    tokens = [str(t) for t in dg["dynamic_tokens"]]
    n_input, K = int(dg["n_input"]), int(dg["k"])
    L = n_input + min(args.k_shown, K)
    k_full = K

    infos = pickle.load(open(Path(args.data_root) / "nuscenes_infos_val.pkl",
                             "rb"))["infos"]
    BY = {i["token"]: i for i in infos}

    frames = Path(args.out) / "forecast_hero_frames"
    frames.mkdir(exist_ok=True, parents=True)
    for f_ in frames.glob("frame*.png"):
        f_.unlink()

    fig = plt.figure(figsize=(11.0, 10.6), dpi=180)
    fig.patch.set_facecolor(BG)
    gsc = fig.add_gridspec(2, 1, height_ratios=(0.115, 0.62),
                           left=0.004, right=0.996, top=0.706, bottom=0.01,
                           wspace=0.06, hspace=0.10)
    ax_front = fig.add_axes((0.35, 0.775, 0.30, 0.175))
    ax_fl = fig.add_axes((0.012, 0.79, 0.248, 0.145))
    ax_fr = fig.add_axes((0.74, 0.79, 0.248, 0.145))
    ax_bl = fig.add_axes((0.012, 0.564, 0.248, 0.145))
    ax_b = fig.add_axes((0.35, 0.549, 0.30, 0.175))
    ax_br = fig.add_axes((0.74, 0.564, 0.248, 0.145))
    gsb = fig.add_gridspec(1, 2, left=0.004, right=0.996, top=0.528,
                           bottom=0.01, wspace=0.06)
    ax_f = fig.add_subplot(gsb[0])
    ax_r = fig.add_subplot(gsb[1])
    title = fig.suptitle("", fontsize=22, y=0.99)

    # corner sparkline: anchored to the upper-right of the forecast panel
    ins = fig.add_axes((0.298, 0.402, 0.18, 0.10))
    ins.set_facecolor("white"); ins.patch.set_alpha(0.9)
    ins.set_xticks([]); ins.set_yticks([])
    miou_full = dg["dynamic_mious"]
    agg = None
    if args.agg_json:
        agg = json.load(open(args.agg_json))
        for q, st in (("naiv", "dashed"), ("ego", "dotted")):
            c = agg[q]
            ins.plot(np.arange(1, len(c) + 1), c, color="#8a8a87", lw=1.4,
                     ls=st)
        ins.plot(np.arange(1, len(agg["wm"]) + 1), agg["wm"],
                 color="#555555", lw=1.8)
        ins.text(0.05, 0.06, "gray = aggregate (val) · orange = this scene",
                 transform=ins.transAxes, fontsize=6.2, color="#444444")

    rig_labels = {"CAM_FRONT": "front", "CAM_FRONT_LEFT": "front left",
                  "CAM_FRONT_RIGHT": "front right", "CAM_BACK": "back",
                  "CAM_BACK_LEFT": "back left", "CAM_BACK_RIGHT": "back right"}
    rig = [(ax_front, CAM_FRONT), (ax_fl, CAM_FRONT_LEFT),
           (ax_fr, CAM_FRONT_RIGHT), (ax_bl, CAM_BACK_LEFT),
           (ax_b, CAM_BACK), (ax_br, CAM_BACK_RIGHT)]

    for step in range(L):
        is_ctx = step < n_input
        tok = tokens[step]
        info = BY[tok]

        for ax, cam in rig:
            rel = info["cams"][cam]["data_path"]
            if rel.startswith("./data/nuscenes/"):
                rel = rel[len("./data/nuscenes/"):]
            ax.clear()
            ax.imshow(plt.imread(Path(args.data_root) / rel))
            ax.set_title(rig_labels[cam],
                         fontsize=15 if ax is ax_front else 12,
                         pad=3.5, color="#333333")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#222222"); sp.set_linewidth(1.0)

        gtb, gtn = info["gt_boxes"], info["gt_names"]
        for ax, kind in ((ax_f, "forecast"), (ax_r, "real")):
            ax.clear()
            # animated GIFs show decoded latents only (maps / detections /
            # GT dashes); LiDAR stays reserved for the static paper filmstrip
            mask = (dg["dynamic_ctx"][step] if is_ctx else
                    (dg["dynamic_pred"][step - n_input]
                     if kind == "forecast" else dg["dynamic_real"][step - n_input]))
            ax.imshow(mask_img(mask, 0.50), extent=(-EXT, EXT, -EXT, EXT),
                      zorder=3)
            if kind == "forecast":
                if is_ctx:
                    # context latent also decodes detections: show them and
                    # label honestly what was measured vs. predicted
                    draw_boxes_bev(ax, pr["boxes_%d" % step],
                                   DET_COLOR, 1.9,
                                   score_thresh=args.score_thresh,
                                   scores=pr["scores_%d" % step])
                    ax.text(0.03, 0.035,
                            "observed context\n"
                            "(detections decoded from\n the input latent)",
                            transform=ax.transAxes, fontsize=12.5,
                            color=DET_COLOR,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                      alpha=0.85))
                    ax.set_title("map context + detections (red)",
                                 fontsize=15, pad=5, color=DET_COLOR)
                else:
                    kk = step - n_input + 1
                    mm = float(dg["dynamic_mious"][kk - 1])
                    found, n_ = match_gt(pr["boxes_%d" % step],
                                         pr["scores_%d" % step],
                                         pr["labels_%d" % step], gtb, gtn,
                                         score_thresh=args.score_thresh)
                    draw_boxes_bev(ax, pr["boxes_%d" % step],
                                   DET_COLOR, 1.9,
                                   score_thresh=args.score_thresh,
                                   scores=pr["scores_%d" % step])
                    ax.text(0.03, 0.035,
                            "forecast (world model)\n"
                            f"mIoU {mm:.2f} · matched {found}/{n_}",
                            transform=ax.transAxes, fontsize=13.5,
                            color=DET_COLOR,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                      alpha=0.88))
                    ax.set_title("map forecast + detections (red)",
                                 fontsize=15, pad=5, color=DET_COLOR)
            else:
                draw_boxes_bev(ax, rr["boxes_%d" % step], ORACLE_COLOR, 1.5,
                               score_thresh=args.score_thresh,
                               scores=rr["scores_%d" % step])
                draw_boxes_bev(ax, gtb, GT_COLOR, 1.0, dashed=True)
                ax.text(0.03, 0.035,
                        "real latent (reference)\n"
                        "cyan = real decode · dashes = GT",
                        transform=ax.transAxes, fontsize=13.5, color="#0072B2",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  alpha=0.88))
                ax.set_title("real BEV, decoded", fontsize=15, pad=5,
                             color="#0072B2")
            # ego anchor
            ax.plot([0, 0], [0, 1.9], color="#222222", lw=1.6, zorder=6)
            ax.add_patch(plt.Rectangle((-0.95, -1.9), 1.9, 4.2,
                         fill=False, edgecolor="#222222", lw=1.4, zorder=6))
            ax.set_xlim(-EXT, EXT); ax.set_ylim(-EXT, EXT)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#222222"); sp.set_linewidth(1.2)

        if is_ctx:
            badge2 = ""
            ins.set_visible(False)
        else:
            kk = step - n_input + 1
            ins.set_visible(True)
            ins.clear()
            ins.set_facecolor("white"); ins.patch.set_alpha(0.9)
            ins.set_xticks([]); ins.set_yticks([])
            if agg:
                for q, st in (("naiv", "dashed"), ("ego", "dotted")):
                    c = agg[q]
                    ins.plot(np.arange(1, len(c) + 1), c,
                             color="#8a8a87", lw=1.4, ls=st)
                ins.plot(np.arange(1, len(agg["wm"]) + 1), agg["wm"],
                         color="#555555", lw=1.8)
                ins.text(0.05, 0.06, "gray = aggregate (val) · orange = scene",
                         transform=ins.transAxes, fontsize=6.2,
                         color="#444444")
            ins.plot(np.arange(1, K + 1), miou_full, color="#B25A00", lw=2.2)
            ins.set_xlim(1, K)
            ins.set_ylim(0, 1.0 if agg else
                         max(0.05, float(miou_full.max()) * 1.15))
            ins.plot([kk], [miou_full[kk - 1]], "o", color="#B25A00", ms=6)
            ins.text(0.03, 0.85, "mIoU (proxy)", transform=ins.transAxes,
                     fontsize=10.5, color="#774400")
            badge2 = ""

        title.set_text("BEV world model (nuScenes) — maps + detections  |  "
                       "t = %.1f s" % (0.5 * (step + 1)))

        if args.arrows and not is_ctx:
            try:
                cm_c = _cent(dg["dynamic_real"][step - n_input])
                cm_n = _cent(dg["dynamic_real"][step - n_input + 1]) \
                    if step + 1 < n_input + K else None
                if cm_c is not None and cm_n is not None:
                    d_ = (float(cm_n[0]) - float(cm_c[0]) * 0,
                          float(cm_n[1]) - float(cm_c[1]))
                    yx = (cm_c[0], EXT, cm_c[1], EXT)
                    for ax_ in (ax_f, ax_r):
                        ax_.annotate("", xy=(cm_c[0], cm_c[1]),
                                     xytext=(cm_c[0] + d_[0], cm_c[1] + d_[1]),
                                     arrowprops=dict(arrowstyle="->",
                                                     color="#1565C0", lw=2.4),
                                     zorder=8)
                        ax_.text(cm_c[0], cm_c[1] + 2.4, "next world shift",
                                 fontsize=9, color="#1565C0")
            except Exception as ex:
                pass
        fig.savefig(frames / ("frame%03d.png" % step), dpi=fig.dpi)
        print("[combo] frame%03d" % step, flush=True)
    plt.close(fig)

    mp4 = Path(args.out) / "detmaps_hero.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                    str(args.fps), "-i", str(frames / "frame%03d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                    "-movflags", "+faststart", str(mp4)], check=True)
    print("[combo] %s" % mp4)


if __name__ == "__main__":
    main()
