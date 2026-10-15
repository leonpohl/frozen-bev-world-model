# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from torchpack.utils.config import configs
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmdet3d.core.bbox import LiDARInstance3DBoxes

CFG_PATH = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
CKPT = "pretrained/bevfusion-det.pth"
PRED = Path("/output/dump_train_det_baseline")
REAL = Path("/output/latents_train_real_mix")
PKL = "data/nuscenes/nuscenes_infos_train.pkl"
OUT = Path("/output/adapted_det")
CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
HOLDOUT = 500


def build_gt_index():
    with open(PKL, "rb") as f:
        infos = pickle.load(f)["infos"]
    idx = {}
    for info in infos:
        mask = info["valid_flag"]
        boxes = info["gt_boxes"][mask]
        vel = info["gt_velocity"][mask]
        names = info["gt_names"][mask]
        labels = np.array([CLASSES.index(n) if n in CLASSES else -1 for n in names])
        keep = labels >= 0
        boxes = np.concatenate([boxes[keep], vel[keep]], axis=1)
        boxes[np.isnan(boxes)] = 0.0
        idx[info["token"]] = (boxes.astype(np.float32), labels[keep].astype(np.int64))
    return idx


def load_batch(tokens_batch, gt, real_set, device):
    xs, gbs, gls = [], [], []
    for tok in tokens_batch:
        src = REAL if tok in real_set else PRED
        a = np.load(src / f"bev_latent_{tok}.npy").astype(np.float32)
        xs.append(torch.from_numpy(a if a.ndim == 3 else a[0]))
        b, l = gt[tok]
        gbs.append(LiDARInstance3DBoxes(torch.from_numpy(b), box_dim=9))
        gls.append(torch.from_numpy(l).to(device))
    return torch.stack(xs).to(device), gbs, gls


def batch_loss(model, x, gbs, gls):
    feat = model.decoder["backbone"](x)
    feat = model.decoder["neck"](feat)
    pred = model.heads["object"](feat, [{} for _ in gls])
    losses = model.heads["object"].loss(gbs, gls, pred)
    return sum(v for v in losses.values() if v.requires_grad or not torch.is_grad_enabled())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed of the head training (init/shuffle/mix)")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    configs.load(CFG_PATH, recursive=True)
    cfg = Config(recursive_eval(configs), filename=CFG_PATH)
    model = build_model(cfg.model)
    load_checkpoint(model, CKPT, map_location="cpu")
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    trainable = (list(model.heads["object"].parameters())
                 + list(model.decoder["backbone"].parameters())
                 + list(model.decoder["neck"].parameters()))
    for p in trainable:
        p.requires_grad = True
    model.train()

    gt = build_gt_index()
    tokens = sorted(f.stem.replace("bev_latent_", "")
                    for f in PRED.glob("bev_latent_*.npy"))
    tokens = [t for t in tokens if t in gt and len(gt[t][1]) > 0]
    real_set = {f.stem.replace("bev_latent_", "") for f in REAL.glob("bev_latent_*.npy")}
    rng = np.random.RandomState(args.seed)
    rng.shuffle(tokens)
    hold = [t for t in tokens if t not in real_set][:HOLDOUT]   # holdout: preds only
    hold_set = set(hold)
    train_tokens = [t for t in tokens if t not in hold_set]
    n_real = sum(1 for t in train_tokens if t in real_set)
    print(f"[v2] {len(train_tokens)} train ({n_real} real = "
          f"{n_real/len(train_tokens):.1%}) | {len(hold)} holdout | 12 ep cosine",
          flush=True)
    OUT.mkdir(exist_ok=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    steps_ep = len(train_tokens) // args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_ep)

    @torch.no_grad()
    def val_loss():
        model.eval()
        tot, nb = 0.0, 0
        for i in range(0, len(hold), args.batch):
            x, gbs, gls = load_batch(hold[i:i+args.batch], gt, set(), device)
            tot += float(batch_loss(model, x, gbs, gls)); nb += 1
            if args.smoke and nb >= 2:
                break
        model.train()
        return tot / max(nb, 1)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(len(train_tokens))
        t0, run, nb = time.time(), 0.0, 0
        for bs in range(0, len(order), args.batch):
            bt = [train_tokens[j] for j in order[bs:bs+args.batch]]
            x, gbs, gls = load_batch(bt, gt, real_set, device)
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(model, x, gbs, gls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 35)
            opt.step(); sched.step()
            run += float(loss); nb += 1
            if nb % 200 == 0 or args.smoke:
                print(f"  ep{ep} [{nb:>5}/{steps_ep}] loss={run/nb:.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"{nb/(time.time()-t0):4.2f} it/s", flush=True)
            if args.smoke and nb >= 5:
                break
        vl = val_loss()
        print(f"[v2] ep{ep}: train {run/max(nb,1):.3f} | VAL {vl:.3f}"
              + (" <- best" if vl < best else ""), flush=True)
        torch.save({"state_dict": model.state_dict()}, OUT / ("det_adapted_v2_last.pth" if args.seed == 42 else f"det_adapted_v2_s{args.seed}_last.pth"))
        if vl < best:
            best = vl
            torch.save({"state_dict": model.state_dict()},
                       OUT / ("det_adapted_v2_best.pth" if args.seed == 42 else f"det_adapted_v2_s{args.seed}_best.pth"))
        if args.smoke:
            break
    print(f"[v2] DONE best-val {best:.3f}")


if __name__ == "__main__":
    main()
