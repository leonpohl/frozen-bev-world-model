# SPDX-FileCopyrightText: 2026 Leon Pohl <leon.pohl@unibw.de>
#
# SPDX-License-Identifier: MIT
"""render_video.py — driving-demo video (MP4) of the rollout (render).

Reads the NPZ produced by scripts/rollout_compute.py and builds one MP4
per scene:
  row 1: CAM_FRONT (real observation)
  row 2: BEV forecast (world model)
  row 3: BEV real (reference)
The setup phase (frames 0..2) shows the observed context latents, then
rollout k = 1..K with an mIoU overlay per step (proxy style). 2 fps
(nuScenes keyframe distance = 0.5 s), H.264/yuv420p (PowerPoint-ready).
Orientation: ego front = up (rot90 k = 1, verified against the thesis
renders).

    python3 scripts/render_video.py --npz iv_rollout.npz --out img --fps 2
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

OI = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"]
ORIENT_K = 1


def mask_rgb(mask):
    h, w = mask.shape[1], mask.shape[2]
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    for c in range(mask.shape[0]):
        img[mask[c]] = np.array(matplotlib.colors.to_rgb(OI[c % len(OI)])) * 255
    return np.rot90(img, k=ORIENT_K).astype(np.uint8)


def render(d, name, out, fps=2):
    ctx, pred, real = d[f"{name}_ctx"], d[f"{name}_pred"], d[f"{name}_real"]
    mi, cams = d[f"{name}_mious"], list(d[f"{name}_cams"])
    n_input, k = int(d["n_input"]), int(d["k"])
    alpha = float(d[f"{name}_mean_alpha"])
    L = n_input + k

    fig = plt.figure(figsize=(8.0, 6.6), dpi=180)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 1, hspace=0.24, top=0.93, bottom=0.03,
                          left=0.02, right=0.98)
    ax_cam, ax_pred, ax_real = (fig.add_subplot(gs[i]) for i in range(3))
    title = fig.suptitle("", fontsize=12)

    frames = Path(out) / f"video_frames_{name}"
    frames.mkdir(parents=True, exist_ok=True)
    for step in range(L):
        ti = 0.5 * step
        if step < n_input:
            label = f"observation  $t$ = {ti:.1f} s   (frame {step + 1}/{L})"
            note = f"context mask | gate $\\bar\\alpha$ = {alpha:.2f}"
            m_pred, m_real = ctx[step], ctx[step]
        else:
            kk = step - n_input + 1
            label = (f"autoregressive forecast  $t$ = {ti:.1f} s   "
                     f"(horizon step k = {kk} of {k})")
            note = f"mIoU (forecast vs. real) = {mi[kk - 1]:.2f}"
            m_pred, m_real = pred[kk - 1], real[kk - 1]
        for ax, img in ((ax_cam, imread(cams[step])),
                        (ax_pred, mask_rgb(m_pred)), (ax_real, mask_rgb(m_real))):
            ax.clear()
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
        ax_cam.set_ylabel("CAM_FRONT (real)", fontsize=9)
        ax_pred.set_ylabel("BEV forecast\n(world model)", fontsize=9)
        ax_real.set_ylabel("BEV real\n(reference)", fontsize=9)
        title.set_text(f"BEV world model — {label}\n{note}")
        fig.savefig(frames / f"frame{step:03d}.png", dpi=fig.dpi)
        print(f"[video] frame{step:03d}", flush=True)

    mp4 = Path(out) / f"iv_rollout_{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", str(fps), "-i", str(frames / "frame%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-movflags", "+faststart", str(mp4)], check=True)
    print("[video] %s" % mp4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=2)
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    d = np.load(args.npz, allow_pickle=True)
    for name in ("dynamic", "static"):
        if f"{name}_pred" in d:
            render(d, name, args.out, args.fps)


if __name__ == "__main__":
    main()
