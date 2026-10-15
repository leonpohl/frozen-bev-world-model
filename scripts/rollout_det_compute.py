# SPDX-FileCopyrightText: 2026 Leon Pohl <leon.pohl@unibw.de>
#
# SPDX-License-Identifier: MIT
"""
rollout_det_compute.py — DETECTION branch rollout on the README example scene.

Runs the DET world model (energy+cosine operating point, checkpoints/
00_BEST/det/best_val_loss.pt, mAP 0.3752) autoregressively over the SAME
scene tokens as the seg headline NPZ, and dumps the predicted DET latents
RAW-scaled (x / 0.3641) as bev_latent_<token>.npy — ready for the frozen
TransFusion head via the BEVFusion injection chain (bevfusion_patch/
README.md: LOAD_BEV_LATENTS / LATENT_LOAD_DIR inside the det_extract
container).
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

DET_SCALE = 0.3641


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pkgs-root", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="det world-model checkpoint (00_BEST/det/best_val_loss.pt)")
    ap.add_argument("--scene-npz", required=True,
                    help="seg headline NPZ that fixes the scene tokens")
    ap.add_argument("--stub-dir", default="/tmp/bev_latents_stub_det_val")
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--k", type=int, default=21)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo))
    import train_linux as T                                     # noqa: E402
    from bev_dataset import BEVDatasetConfig, BEVLatentDataset  # noqa: E402

    scene = np.load(args.scene_npz, allow_pickle=True)
    tokens = [str(t) for t in scene["dynamic_tokens"]]
    n_input = int(scene["n_input"])
    win_tokens = tokens[:n_input + args.k]
    print("[detr] scene window: %d tokens" % len(win_tokens))

    data_root = Path(args.data_root)
    pkl = data_root / "nuscenes_infos_val.pkl"
    stub = Path(args.stub_dir)
    stub.mkdir(parents=True, exist_ok=True)
    with open(pkl, "rb") as f:
        infos = pickle.load(f)["infos"]
    for i in infos:
        (stub / ("bev_latent_%s.npy" % i["token"])).touch()

    pkgs_root = Path(args.pkgs_root)
    pack_dir = pkgs_root / "det" / "val"
    if not (pack_dir / "packed.npy").exists():
        pack_dir = pkgs_root / "val"
    ds = BEVLatentDataset(
        BEVDatasetConfig(sources=[(str(pkl), str(stub))], n_input_frames=3,
                         packed_dir=str(pack_dir), latent_scale=DET_SCALE))
    manifest = json_load = __import__("json").load(open(pack_dir / "manifest.json"))
    assert manifest["num_frames"] == ds._n_packed_frames
    BY = {f["token"]: f for sc in ds._all_scenes for f in sc}
    win = [BY[t] for t in win_tokens]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T.BEVWorldModel(T.ModelConfig(
        d_model=256, n_frames=3, grid_size=45, n_heads=8,
        d_ff=1024, n_layers=4, dropout=0.1, phase="cell")).to(device)
    model.eval()
    T.load_checkpoint(args.checkpoint, model, device=str(device))
    print("[detr] det model loaded (%d params)"
          % (sum(p.numel() for p in model.parameters()) // 1000))

    lat = [torch.from_numpy(ds._load_latent(f)).float().to(device) for f in win]
    cur = torch.stack(lat[:n_input], dim=0).unsqueeze(0)

    dump = Path(args.dump_dir)
    dump.mkdir(parents=True, exist_ok=True)
    # context frames: dump the OBSERVED latents (raw scale) so the injected
    # pipeline reproduces reality there
    for f, l in zip(win[:n_input], lat[:n_input]):
        arr = (l.cpu().numpy() / DET_SCALE).astype(np.float32)
        np.save(dump / ("bev_latent_%s.npy" % f["token"]), arr)
        print("[detr] ctx dumped", f["token"], flush=True)
    with torch.no_grad():
        for k in range(1, args.k + 1):
            pl = model(cur)[0].float()
            arr = (pl.cpu().numpy() / DET_SCALE).astype(np.float32)
            np.save(dump / ("bev_latent_%s.npy" % win[n_input - 1 + k]["token"]), arr)
            print("[detr] step %02d -> %s  (std %.3f)"
                  % (k, win[n_input - 1 + k]["token"], float(pl.std())), flush=True)
            cur = torch.cat([cur[:, 1:], pl.unsqueeze(0).unsqueeze(0)], dim=1)
    print("[detr] raw-scale latents -> %s" % dump)


if __name__ == "__main__":
    main()
