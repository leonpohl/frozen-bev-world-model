# SPDX-FileCopyrightText: 2026 Leon Pohl <leon.pohl@unibw.de>
#
# SPDX-License-Identifier: MIT
"""render_filmstrip.py — paper-style filmstrip of the rollout demo (render).

Reads the NPZ produced by scripts/rollout_compute.py and builds the compact
filmstrip figure: rows = camera (real) | BEV real (reference) |
BEV forecast (world model); columns = context + k = 1..k_show.
Output: iv_demo_<name>_3row.{pdf,png} and iv_demo_<name>_2row.{pdf,png}.

    python3 scripts/render_filmstrip.py --npz iv_rollout.npz --out img

Style: Okabe-Ito (color-vision safe); ego front = up (rot90 k = 1,
verified in the thesis renders). The mIoU annotations are proxy style
(forecast vs. decoded real latent).
"""
import argparse
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


def build(d, name, fname, rows, ncols_show, out):
    ctx, pred, real = d[f"{name}_ctx"], d[f"{name}_pred"], d[f"{name}_real"]
    mi, cams = d[f"{name}_mious"], list(d[f"{name}_cams"])
    n_input = int(d["n_input"])
    ncols = 1 + ncols_show

    fig, axes = plt.subplots(len(rows), ncols, figsize=(2.55 * ncols, 2.45 * len(rows)),
                             gridspec_kw=dict(wspace=0.06, hspace=0.18))
    axes = np.atleast_2d(axes)
    labels = {"cam": "camera (real)", "real": "BEV real\n(reference)",
              "pred": "BEV forecast\n(world model)"}
    for r, rowkey in enumerate(rows):
        for c in range(ncols):
            fi = n_input - 1 + c
            ax = axes[r, c]
            if rowkey == "cam":
                ax.imshow(imread(cams[fi]))
            elif rowkey == "real":
                ax.imshow(mask_rgb(real[fi - 1] if c else ctx[-1]))
            else:
                ax.imshow(mask_rgb(ctx[-1] if c == 0 else pred[c - 1]))
                if c:
                    ax.set_title("%.2f" % mi[c - 1], fontsize=8, pad=2)
            if r == 0:
                ax.set_title(f"$t$ = {0.5 * c:.1f} s" if c > 0 else "context",
                             fontsize=9.5)
            if c == 0:
                ax.set_ylabel(labels[rowkey], fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(Path(out) / f"{fname}.{ext}", dpi=200,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("[fig] %s/%s.{pdf,png}" % (out, fname))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k-show", type=int, default=4)
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    d = np.load(args.npz, allow_pickle=True)
    for name in ("dynamic", "static"):
        if f"{name}_pred" in d:
            build(d, name, f"iv_demo_{name}_3row", ("cam", "real", "pred"),
                  args.k_show, args.out)
            build(d, name, f"iv_demo_{name}_2row", ("cam", "pred"),
                  args.k_show, args.out)


if __name__ == "__main__":
    main()
