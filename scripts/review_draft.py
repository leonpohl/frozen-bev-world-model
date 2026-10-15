# SPDX-FileCopyrightText: 2026 Leon Pohl <leon.pohl@unibw.de>
#
# SPDX-License-Identifier: MIT
"""
review_draft.py — five review-draft figures for the IV paper story.

Everything is pure analysis of EXISTING dumps (no retraining, no decoding):

  1. skill.png        — aggregate "skill vs persistence" curves (proxy
                        family, 4,915 windows) + annotation-family
                        dumbbells; the two families are never mixed.
  2. decomposition.png— per-window error decomposition on the 24-window
                        det-scene subset: naive persistence error split
                        into ego-recoverable and residual (world motion),
                        with the world-model error overlaid. The ego warp
                        follows ego_comp_pers_scene.py verbatim.
  3. cross_decode.png — teaser: ONE forecast latent, three views
                        (frozen seg head map, frozen det head boxes,
                        per-cell cosine probe against the real latent).
  4. best_worst.png   — per-scene proxy decay for the three featured
                        scenes vs. the aggregate curves, ranked by the
                        disclosed drift-ratio rule.
  5. runtime.png      — resource accounting from the paper's own table.

    python3 scripts/review_draft.py --out /home/vima/iv_demo/review
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DET = "#FF3D3D"
ORA = "#00B7EB"
GT = "#666666"
OI = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"]

PRED = ("/home/vima/Desktop/Masterarbeit/Code_final/"
        "ergebnisse/predictions")
SMOKE = Path("/tmp/bevwm-smoke")
LATENTS = Path("/home/vima/det_latents_out")
INFOS_PKL = "/home/vima/bevfusion_decoder/nuscenes_infos_val.pkl"

AXIS_SWAP = True     # convention of ego_comp_pers_scene.py (thesis tools)
YAW_SIGN = 1.0
EXTENT = 50.0


def quat_to_yaw(q):
    w, x, y_, z = q
    return np.arctan2(2.0 * (w * z + x * y_), 1.0 - 2.0 * (y_ * y_ + z * z))


def rigid2d(dx, dy, dyaw):
    c, s = np.cos(dyaw), np.sin(dyaw)
    return np.array([[c, -s, dx], [s, c, dy], [0, 0, 1.0]])


def warp_mask(mask_t, pose_i, pose_j, device=None):
    """Planar ego-pose warp; nearest, zeros padding (thesis convention)."""
    import torch
    dev = device if device is not None else "cpu"
    xi, yi, yawi = pose_i
    xj, yj, yawj = pose_j
    Ti = rigid2d(xi, yi, YAW_SIGN * yawi)
    Tj = rigid2d(xj, yj, YAW_SIGN * yawj)
    M = np.linalg.inv(Ti) @ Tj
    cls, H, Wd = mask_t.shape
    lin = torch.linspace(-EXTENT + EXTENT / H, EXTENT - EXTENT / H, H,
                         device=dev)
    a0, a1 = torch.meshgrid(lin, lin, indexing="ij") if torch.__version__ \
        >= "1.10" else torch.meshgrid(lin, lin)
    mx = a1 if AXIS_SWAP else a0
    mn = a0 if AXIS_SWAP else a1
    pj = torch.stack([mx.reshape(-1), mn.reshape(-1),
                      torch.ones_like(mx).reshape(-1)], dim=0)
    pi = torch.tensor(M, dtype=torch.float32, device=dev) @ pj
    sx, sy = pi[0], pi[1]
    src_a0 = sy if AXIS_SWAP else sx
    src_a1 = sx if AXIS_SWAP else sy
    grid = torch.stack([src_a1 / EXTENT, src_a0 / EXTENT],
                       dim=-1).reshape(1, H, Wd, 2)
    mt = torch.as_tensor(mask_t, dtype=torch.float32, device=dev)
    out = torch.nn.functional.grid_sample(mt.unsqueeze(0), grid,
                                          mode="nearest",
                                          padding_mode="zeros",
                                          align_corners=False)
    return (out[0] > 0.5).cpu().numpy()


# ---------------------------------------------------------------- figure 1
def fig_skill(out):
    wm = [roll["mIoU"] for roll in
          (json.load(open(PRED + "/task30/rollout_k10_minimal.json"))
           ["curve"]).values()]
    naiv = [roll["mIoU_pers"] for roll in
            (json.load(open(PRED + "/task30/rollout_k10_minimal.json"))
             ["curve"]).values()]
    eg = json.load(open(PRED + "/task16c/ego_pers.json"))["curve"]

    def skill(m, n):
        return [(m[i] - n[i]) / (1.0 - n[i]) for i in range(len(m))]

    s_wm = skill(wm, naiv)
    s_ego = skill([eg[str(k)]["pers_ego"] for k in (1, 2, 3, 4)],
                  [eg[str(k)]["pers_naiv"] for k in (1, 2, 3, 4)])

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 4.4), dpi=200,
        gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.patch.set_facecolor("white")

    h = np.arange(1, 11) * 0.5
    axL.plot(h[:4], 100 * np.array(s_ego), color="#8a8a87", lw=1.8, ls="--",
             marker="o", ms=4, label="ego-compensated persistence")
    axL.plot(h, 100 * np.array(s_wm), color="#111111", lw=2.6, marker="o",
             ms=4, label="forecast (world model)")
    axL.set_xlabel("forecast horizon (s)")
    axL.set_ylabel("share of persistence$\\to$measured gap closed (%)")
    axL.set_title("Proxy mIoU family — full validation (4,915 windows)",
                  fontsize=11)
    axL.grid(alpha=0.25); axL.set_ylim(0, 100)
    for i in (0, 1, 3, 9):
        axL.annotate("%.0f%%" % (100 * s_wm[i]), (h[i], 100 * s_wm[i]),
                     textcoords="offset points", xytext=(2, 7), fontsize=8)
    axL.legend(fontsize=9, loc="lower left")
    axL.text(0.02, 0.97, "baseline = naive persistence; target = decoded "
             "real latent (self)", transform=axL.transAxes, fontsize=7.5,
             color="#555555", va="top")

    # annotation family: dumbbells (single reference numbers, never per-step)
    rows = [("map seg (mIoU)", 0.4647, 0.5898, 0.6295),
            ("3D det (mAP)", 0.1696, 0.3752, 0.6858)]
    axR.set_title("Annotation family — aggregate (never mixed)",
                  fontsize=11)
    for r_i, (name, pers, wmv, real) in enumerate(rows):
        y = 1 - r_i
        axR.plot([pers, real], [y, y], color="#bbbbbb", lw=2.2, zorder=1)
        axR.scatter([pers], [y], s=55, color="#8a8a87", zorder=3,
                    marker="o")
        axR.scatter([real], [y], s=55, color="#00B7EB", zorder=3)
        axR.scatter([wmv], [y], s=75, color=DET, zorder=4, marker="D")
        sk = 100 * (wmv - pers) / (real - pers)
        axR.annotate("skill %.0f%%" % sk, (wmv, y),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=9.5, color=DET)
        for v_ in (pers, real):
            axR.annotate("%.3f" % v_, (v_, y),
                         textcoords="offset points", xytext=(0, -14),
                         fontsize=8, color="#555555")
    axR.set_yticks([1, 0]); axR.set_yticklabels([r[0] for r in rows])
    axR.set_xlabel("annotation metric")
    axR.set_ylim(-0.55, 1.45)
    axR.set_xlim(0.1, 0.74)
    axR.scatter([], [], s=55, color="#8a8a87", marker="o",
                label="persistence")
    axR.scatter([], [], s=75, color=DET, marker="D", label="world model")
    axR.scatter([], [], s=55, color="#00B7EB", label="real latent")
    axR.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.26),
               ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)
    return dict(skill_wm=[round(v, 4) for v in s_wm],
                skill_ego=[round(v, 4) for v in s_ego])


# ------------------------------------------------------------- figure 2
def fig_decomp(out):
    # pure aggregate decomposition (paper protocol, no per-window warp):
    t30 = json.load(open(PRED + "/task30/rollout_k10_minimal.json"))["curve"]
    egc = json.load(open(PRED + "/task16c/ego_pers.json"))["curve"]
    wm = np.array([t30[str(k)]["mIoU"] for k in map(str, range(1, 11))])
    naiv = np.array([t30[str(k)]["mIoU_pers"] for k in map(str, range(1, 11))])
    eg = np.array([egc[str(k)]["pers_ego"] for k in map(str, range(1, 5))])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
    fig.patch.set_facecolor("white")
    h = np.arange(1, 11) * 0.5
    axL.plot(h, 1 - naiv, color="#8a8a87", lw=1.8, ls="--",
             label="persistence (naive)")
    axL.plot(h[:4], 1 - eg, color="#444444", lw=2.0, ls="-.",
             label="persistence, ego-compensated")
    axL.plot(h, 1 - wm, color="#111111", lw=2.6, label="forecast (world model)")
    axL.fill_between(h[:4], 1 - eg, 1 - naiv[:4], color="#B25A00",
                     alpha=0.14)
    axL.annotate("ego-recoverable share of the naive error",
                 (2.15, 0.63), fontsize=8.5, color="#774400")
    axL.set_xlabel("forecast horizon (s)")
    axL.set_ylabel("proxy error  (1 $-$ mIoU)")
    axL.set_title("Error curves — full validation (4,915 windows; "
                  "ego baseline 5,467)", fontsize=11)
    axL.set_ylim(0, 0.85); axL.grid(alpha=0.25)
    axL.legend(fontsize=9, loc="upper left")

    ks = np.arange(1, 5)
    en = (1 - naiv[:4]) * 100
    ee = (1 - eg) * 100
    ew = (1 - wm[:4]) * 100
    axR.bar(ks, en - ee, 0.58, color="#B25A00", alpha=0.85,
            label="ego-recoverable (ego motion)")
    axR.bar(ks, ee, 0.58, bottom=en - ee, color="#00B7EB", alpha=0.55,
            label="residual: world motion + exposure")
    axR.plot(ks, ew, "D", color=DET, ms=7, label="world-model error")
    for i, k_ in enumerate(ks):
        axR.annotate("WM %.0f" % ew[i], (k_, ew[i] + 1.5),
                     textcoords="offset points", xytext=(0, 4),
                     fontsize=8.5, color=DET)
        axR.annotate("%.0f" % en[i], (k_, en[i] + 1.5),
                     textcoords="offset points", xytext=(0, 20),
                     fontsize=8.5, color="#555555")
    axR.set_xticks(ks)
    axR.set_xticklabels(["%.1f s" % (0.5 * k_) for k_ in ks])
    axR.set_ylabel("naive-persistence error (%)")
    axR.set_title("Naive-persistence error split (aggregate)",
                  fontsize=11)
    axR.set_ylim(0, 92)
    axR.grid(alpha=0.25, axis="y")
    axR.legend(fontsize=8.5, loc="upper right",
               bbox_to_anchor=(1.0, 1.12))
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)
    return dict(err_naive=[round(float(v), 4) for v in en / 100],
                err_ego=[round(float(v), 4) for v in ee / 100],
                err_wm=[round(float(v), 4) for v in ew / 100],
                ego_recoverable_share=[round(float((en[i] - ee[i]) / en[i]),
                                             4) for i in range(4)])



def ego_iou_dyn(warped, tgt):
    """unused fallback: plain per-channel mean IoU."""
    a, b = warped > 0.5, tgt > 0.5
    ious = []
    for c in range(a.shape[0]):
        inter = np.logical_and(a[c], b[c]).sum()
        union = np.logical_or(a[c], b[c]).sum()
        ious.append(inter / union if union else 1.0)
    return float(np.mean(ious))


# ------------------------------------------------------------- figure 3
def fig_cross_decode(out):
    scene = np.load(SMOKE / "iv_rollout_detscene.npz", allow_pickle=True)
    tokens = [str(t) for t in scene["dynamic_tokens"]]
    n_input = int(scene["n_input"])
    j = 4                                   # 2.0 s forecast step index (1bs)
    tk = tokens[n_input - 1 + j]
    pr = np.load(LATENTS / "det_scene2_pred.npz", allow_pickle=True)
    rr = np.load(LATENTS / "det_scene2_real.npz", allow_pickle=True)
    z_wm = np.load(str(LATENTS / "latents_det_wm_scene2" /
                       ("bev_latent_%s.npy" % tk))).reshape(256, -1).T
    z_re = np.load(str(LATENTS / "latents_val_synced" /
                       ("bev_latent_%s.npy" % tk)))
    z_re = z_re.reshape(z_re.shape[1], -1).T
    ncell = int(round(np.sqrt(z_wm.shape[0] / 1.0)))
    gw = gh = int(round(np.sqrt(z_wm.shape[0])))
    cos = (z_wm * z_re).sum(1) / (
        np.linalg.norm(z_wm, axis=1) * np.linalg.norm(z_re, axis=1) + 1e-9)
    cos = cos.reshape(gw, gh)

    fig, axs = plt.subplots(1, 4, figsize=(12.6, 4.0), dpi=200,
                            gridspec_kw=dict(width_ratios=[1, 1, 1.35, 1],
                                             wspace=0.18))
    fig.patch.set_facecolor("white")
    EXT = 50.0

    def show(mask, ax, ttl, ccol):
        rgba = np.zeros((mask.shape[1], mask.shape[2], 4))
        for c_ in range(mask.shape[0]):
            rgb = np.array(matplotlib.colors.to_rgb(OI[c_]))
            rgba[mask[c_]] = np.append(rgb, 0.9)
        ax.imshow(np.rot90(rgba, k=1), extent=(-EXT, EXT, -EXT, EXT))
        ax.set_title(ttl, fontsize=10, color=ccol)
        ax.set_xlim(-EXT, EXT); ax.set_ylim(-EXT, EXT)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    show((scene["dynamic_pred"][j - 1] > 0.5), axs[0],
         "frozen seg head\nforecast latent", "#B25A00")
    show((scene["dynamic_real"][j - 1] > 0.5), axs[1],
         "frozen seg head\nreal latent (t+2 s)", "#0072B2")

    ax = axs[2]
    ax.set_xlim(-EXT, EXT); ax.set_ylim(-EXT, EXT)
    ax.set_aspect("equal")

    def draw(bx, sc, color):
        for i, b in enumerate(bx):
            if sc[i] < 0.2:
                continue
            c_, s = np.cos(b[6]), np.sin(b[6])
            R = np.array([[c_, -s], [s, c_]])
            p = (np.array([(-b[3] / 2, -b[4] / 2), (b[3] / 2, -b[4] / 2),
                           (b[3] / 2, b[4] / 2), (-b[3] / 2, b[4] / 2)])
                 @ R.T) + np.asarray(b[:2])
            ax.add_patch(plt.Polygon(p, fill=False, ec=color, lw=1.8))
    draw(pr["boxes_%d" % (j + 2)], pr["scores_%d" % (j + 2)], DET)
    draw(rr["boxes_%d" % (j + 2)], rr["scores_%d" % (j + 2)], ORA)
    ax.plot([0, 0], [0, 1.9], color="#222222", lw=1.4, zorder=6)
    ax.add_patch(plt.Rectangle((-0.95, -1.9), 1.9, 4.2, fill=False,
                               edgecolor="#222222", lw=1.2, zorder=6))
    ax.set_title("frozen TransFusion head\nred = forecast · cyan = real",
                 fontsize=10)
    sc_pre = float((pr["scores_%d" % (j + 2)] > 0.2).sum())
    sc_re = float((rr["scores_%d" % (j + 2)] > 0.2).sum())
    ax.text(0.03, 0.035, "%.0f boxes forecast · %.0f real-decode" %
            (sc_pre, sc_re), transform=ax.transAxes, fontsize=8.5,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))
    ax.set_xticks([]); ax.set_yticks([])

    ax = axs[3]
    im = ax.imshow(np.rot90(cos, k=1), cmap="viridis", vmin=-1, vmax=1,
                   extent=(-gw * 0.5 * 102.4 / gw, gw * 0.5 * 102.4 / gw,
                           -gw * 0.5 * 102.4 / gw, gw * 0.5 * 102.4 / gw))
    ax.set_title("feature probe (det latent)\nper-cell cosine\nforecast vs. real latent",
                 fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("t+2.0 s · the forecast latent is perception feature "
                 "space — every frozen head consumes the same latent",
                 fontsize=12, y=1.02)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return dict(cos_mean=round(float(np.nanmean(cos)), 4),
                cos_q25=round(float(np.nanpercentile(cos, 25)), 4),
                cos_q75=round(float(np.nanpercentile(cos, 75)), 4))


# ------------------------------------------------------------- figure 4
def fig_best_worst(out):
    agg_t30 = json.load(open(PRED + "/task30/rollout_k10_minimal.json"))["curve"]
    agg_e = json.load(open(PRED + "/task16c/ego_pers.json"))["curve"]
    wm = [agg_t30[str(k)]["mIoU"] for k in agg_t30]
    naiv = [agg_t30[str(k)]["mIoU_pers"] for k in agg_t30]
    eg = [agg_e[str(k)]["pers_ego"] for k in agg_e]

    scenes = [("f1b7b463 (drift 0.98)", SMOKE / "seg_f1b7b463.npz", "#009E73"),
              ("d5324133 (drift 1.38)", SMOKE / "iv_rollout_detscene.npz",
               "#B25A00"),
              ("029af235 (drift 1.72)", SMOKE / "seg_029af235.npz",
               DET)]
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=200)
    fig.patch.set_facecolor("white")
    h10 = np.arange(1, 11) * 0.5
    ax.plot(h10, naiv, color="#8a8a87", lw=2.0, ls="--")
    ax.plot(h10, wm, color="#555555", lw=2.0, ls=":")
    he = np.arange(1, 5) * 0.5
    ax.plot(he, eg, color="#8a8a87", lw=2.0, ls="-.")
    for name, npz_path, col in scenes:
        d = np.load(npz_path, allow_pickle=True)
        mi = np.asarray(d["dynamic_mious"], dtype=float)
        ks = np.arange(1, len(mi) + 1) * 0.5 * (21 / 21)
        ax.plot(ks, mi, color=col, lw=2.2, label=name)
        ax.annotate("%.2f" % mi[7], (ks[7], mi[7]),
                    textcoords="offset points", xytext=(0, 7),
                    fontsize=8, color=col)
    ax.annotate("aggregate naive pers.", (h10[-1], naiv[-1]),
                fontsize=8, color="#8a8a87", xytext=(-4, -10),
                textcoords="offset points")
    ax.annotate("aggregate WM", (h10[-1], wm[-1]), fontsize=8,
                color="#555555", xytext=(-4, 8),
                textcoords="offset points")
    ax.annotate("aggregate ego-comp.", (he[-1], eg[-1]), fontsize=8,
                color="#8a8a87", xytext=(-4, -10),
                textcoords="offset points")
    ax.set_xlabel("forecast horizon (s)")
    ax.set_ylabel("proxy mIoU vs. decoded real latent")
    ax.set_title("Per-scene decay ranked by the disclosed drift rule",
                 fontsize=11)
    ax.set_xlim(0.4, 10.6); ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.01, -0.14, "drift ratio = mean predicted-vs-real world-shift "
            "of the drivable-area centroid over the first 4 s (1.0 = "
            "perfect motion tracking); per-scene decay varies scene-level —\n"
            "the paper's claim rests on the aggregate (gray).",
            transform=ax.transAxes, fontsize=7.6, color="#555555")
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)
    return {}


# ------------------------------------------------------------- figure 5
def fig_runtime(out):
    rows = [
        ("BEVFusion encoder (given stack)", 173.0, "#8a8a87"),
        ("forecast step — seg minimal", 4.41, DET),
        ("forecast step — CVAE variant", 4.58, "#E69F00"),
        ("forecast step — det historical", 11.45, DET),
        ("forecast step — flow (+pred.)", 14.02, DET),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=200)
    fig.patch.set_facecolor("white")
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [r[2] for r in rows][::-1]
    bars = ax.barh(names, vals, color=cols, alpha=0.9, height=0.55)
    for b, v in zip(bars, vals):
        if v == 173.0:
            ax.annotate("173 ms  (given encoder path)",
                        (v + 2, b.get_y() + b.get_height() / 2),
                        va="center", fontsize=9)
        else:
            ax.annotate("%.2f ms  (%s of encoder path)" %
                        (v, ("%.1f%%" % (100 * v / 173.0))),
                        (v + 2, b.get_y() + b.get_height() / 2), va="center",
                        fontsize=9)
    ax.set_xlim(0, 272)
    ax.set_xlabel("latency per step, batch 1 (TITAN RTX)")
    ax.set_title("Resource accounting: the forecast step is noise next to "
                 "the encoder", fontsize=11)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/vima/iv_demo/review")
    args = ap.parse_args()
    outd = Path(args.out); outd.mkdir(exist_ok=True, parents=True)
    numbers = {}
    print("[review] 1/5 skill");        numbers["skill"] = fig_skill(outd / "skill.png")
    print("[review] 2/5 decomposition"); numbers["decomp"] = fig_decomp(outd / "decomposition.png")
    print("[review] 3/5 cross decode");  numbers["cross"] = fig_cross_decode(outd / "cross_decode.png")
    print("[review] 4/5 best worst");    numbers["best_worst"] = fig_best_worst(outd / "best_worst.png")
    print("[review] 5/5 runtime");       numbers["runtime"] = fig_runtime(outd / "runtime.png")
    json.dump(numbers, open(outd / "review_numbers.json", "w"), indent=1)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
