# SPDX-FileCopyrightText: 2026 Leon Pohl <leon.pohl@unibw.de>
#
# SPDX-License-Identifier: MIT
"""rollout_compute.py — autoregressive rollout + decoding for the demos.

Three real context latents, then `--k` autoregressive steps with the world
model; every prediction and every real frame is decoded by the FROZEN seg
decoder. Scene selection is disclosed (paper/filmstrip convention):
candidates = scene-start windows of full length; dynamics proxy = mean
gate-alpha of the first forward; dynamic = maximum, static = minimum.
NO cherry-picking on image quality. Honesty: per-step mIoU of the forecast
against the decoded real latent (proxy style), persistence baseline too.

Model, dataset, and decoder come with this repository (train_linux.py and
the root modules); run from anywhere — the repo root is auto-detected.

Typical call (paths are machine-specific):

    python3 scripts/rollout_compute.py \
        --repo-root ../thesis-repo \
        --data-root  /data/nuscenes/complete \
        --pkgs-root  /data/nuscenes/latents_packed_fp16 \
        --checkpoint checkpoints/00_BEST/seg/best_miou.pt \
        --seg-head   /data/bevfusion-seg.pth \
        --k 21 --out iv_rollout.npz

Note: latents are read from the memmap packs (pack_latents.py in the root,
thesis repo). If the raw .npy latent folder no longer exists, a stub folder
with 0-byte placeholder files per token satisfies the existence filter;
all data is read from packed.npy (same row assignment, same PKL).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

CAMS_KEY = "CAM_FRONT"


def resolve_cam(data_root, rel):
    """BEVFusion PKL paths are relative ("./data/nuscenes/...")."""
    if rel.startswith("./data/nuscenes/"):
        rel = rel[len("./data/nuscenes/"):]
    p = Path(data_root) / rel
    if not p.exists():
        raise FileNotFoundError(p)
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=Path(__file__).resolve().parent.parent,
                    help="this repo (contains train_linux.py and the modules); "
                         "default: the parent of scripts/")
    ap.add_argument("--data-root", required=True, help="nuScenes complete dir")
    ap.add_argument("--pkgs-root", required=True,
                    help="latents_packed_fp16 parent (split subfolders)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pkl-name", default="nuscenes_infos_%s.pkl")
    ap.add_argument("--checkpoint", required=True,
                    help="world-model checkpoint (train_linux format)")
    ap.add_argument("--seg-head", required=True, help="bevfusion-seg.pth")
    ap.add_argument("--stub-dir", default="/tmp/bev_latents_stub_val")
    ap.add_argument("--k", type=int, default=21, help="rollout steps")
    ap.add_argument("--tokens-file", default=None,
                    help="optional: fix the window tokens (one per line, "
                         "frame order); overrides gate scene selection")
    ap.add_argument("--out", default="iv_rollout.npz")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo))
    import train_linux as T                                  # noqa: E402
    from bev_dataset import BEVDatasetConfig, BEVLatentDataset  # noqa: E402

    data_root = Path(args.data_root)
    pkl = data_root / (args.pkl_name % args.split)
    with open(pkl, "rb") as f:
        infos = pickle.load(f)["infos"]

    stub = Path(args.stub_dir)
    stub.mkdir(parents=True, exist_ok=True)
    for i in infos:
        (stub / ("bev_latent_%s.npy" % i["token"])).touch()

    pkgs_root = Path(args.pkgs_root)
    pack_dir = pkgs_root / args.split
    if not (pack_dir / "packed.npy").exists():
        pack_dir = pkgs_root / "seg" / args.split          # local layout
    cfg = BEVDatasetConfig(
        sources=[(str(pkl), str(stub))],
        n_input_frames=3,
        packed_dir=str(pack_dir),
        latent_scale=1.0,
    )
    ds = BEVLatentDataset(cfg)
    manifest = json.load(open(pack_dir / "manifest.json"))
    assert manifest["num_frames"] == ds._n_packed_frames == len(infos), \
        "pack/dataset mismatch"
    print("[rollout] pack ok: %d frames, %d scenes"
          % (ds._n_packed_frames, len(ds._all_scenes)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_input = 3
    win_len = n_input + args.k

    model = T.BEVWorldModel(T.ModelConfig(
        d_model=256, n_frames=3, grid_size=32, n_heads=8,
        d_ff=1024, n_layers=4, dropout=0.1, phase="cell")).to(device)
    model.eval()
    T.load_checkpoint(args.checkpoint, model, device=str(device))
    seg = T.build_seg_decoder(args.seg_head, device, verbose=True)

    if args.tokens_file:
        toks = [l.strip() for l in open(args.tokens_file) if l.strip()]
        BY = {f["token"]: f for sc in ds._all_scenes for f in sc}
        try:
            win = [BY[t] for t in toks]
        except KeyError as ke:
            raise SystemExit("[rollout] token not in val set: %s" % ke)
        assert len(win) >= win_len, "window too short: %d < %d" % (len(win), win_len)
        cands = [win[:win_len]]
        a_of = {0: float("nan")}
        print("[rollout] fixed tokens override (%d frames)" % len(win))
    else:
        cands = [scene[:win_len] for scene in ds._all_scenes if len(scene) >= win_len]
    print("[rollout] %d candidate scenes (win_len=%d)" % (len(cands), win_len))
    _cap = {}
    h = model.gate.register_forward_hook(
        lambda m, i, o: _cap.__setitem__("alpha", o.detach()))
    a_of = {}
    for wi, win in enumerate(cands):
        lat = [torch.from_numpy(ds._load_latent(f)).float().to(device)
               for f in win[:n_input]]
        _ = model(torch.stack(lat, dim=0).unsqueeze(0))
        a_of[wi] = float(_cap["alpha"][0].mean().item())
    picks = {"dynamic": max(a_of, key=a_of.get), "static": min(a_of, key=a_of.get)}
    for n, wi in picks.items():
        print("[rollout] %-7s scene %d  mean-alpha %.3f" % (n, wi, a_of[wi]))

    def to_np(m):
        return (m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)).astype(bool)

    save = {"n_input": n_input, "k": args.k}
    for name, wi in picks.items():
        win = cands[wi]
        lat = [torch.from_numpy(ds._load_latent(f)).float().to(device) for f in win]
        ctx = [to_np(seg.latent_to_mask(lat[f], device=device)[0])
               for f in range(n_input)]
        real = [to_np(seg.latent_to_mask(lat[n_input - 1 + k], device=device)[0])
                for k in range(1, args.k + 1)]
        pers = [ctx[-1]] * args.k
        cur = torch.stack(lat[:n_input], dim=0).unsqueeze(0)
        pred, mious, pmi, gates = [], [], [], []
        for k in range(1, args.k + 1):
            pl = model(cur)[0].float()
            # gate alpha [B,256,128,128] -> per-cell mean over channels [128,128]
            gates.append(_cap["alpha"][0].mean(dim=0).half().cpu().numpy())
            pm = to_np(seg.latent_to_mask(pl, device=device)[0])
            pred.append(pm)
            mious.append(float(T.compute_iou(pm, real[k - 1], T.MAP_CLASSES)["mIoU"]))
            pmi.append(float(T.compute_iou(pers[k - 1], real[k - 1],
                                           T.MAP_CLASSES)["mIoU"]))
            cur = torch.cat([cur[:, 1:], pl.unsqueeze(0).unsqueeze(0)], dim=1)
        save[f"{name}_ctx"] = np.stack(ctx)
        save[f"{name}_pred"] = np.stack(pred)
        save[f"{name}_pers"] = np.stack(pers)
        save[f"{name}_real"] = np.stack(real)
        save[f"{name}_mious"] = np.array(mious, np.float32)
        save[f"{name}_pers_mious"] = np.array(pmi, np.float32)
        save[f"{name}_gate"] = np.stack(gates)          # [K,128,128] fp16
        save[f"{name}_mean_alpha"] = np.array(a_of[wi], np.float32)
        save[f"{name}_tokens"] = np.array([f["token"] for f in win])
        save[f"{name}_cams"] = np.array(
            [resolve_cam(data_root, f["cams"][CAMS_KEY]["data_path"]) for f in win])
        print("[rollout] %-7s mIoU k1=%.3f k5=%.3f k%d=%.3f (persistence %.3f/%.3f)"
              % (name, mious[0], mious[4], args.k, mious[-1], pmi[0], pmi[-1]))

    h.remove()
    np.savez_compressed(args.out, **save)
    print("[rollout] saved -> %s" % args.out)


if __name__ == "__main__":
    main()
