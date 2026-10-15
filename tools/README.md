# tools/ — head adaptation

Continue-training the frozen perception heads on world-model predictions
(world model and encoder remain unchanged; thesis, chapter "Head
Adaptation"). All three scripts run **inside the BEVFusion Docker
container** (see bevfusion_patch/ and the root README).

| Script | Purpose |
|---|---|
| `mk_gt_masks_seg.py` | rasterizes the nuScenes map GT into the target grid (prerequisite for the seg adaptation) |
| `adapt_seg_head.py` | trains the seg decoder stack (SECOND + FPN + SegHead) on (prediction latent → map GT) |
| `adapt_det_head.py` | trains the TransFusion head on prediction latents; with real mix and val-loss selection |

Seg order: first `mk_gt_masks_seg.py`, then `adapt_seg_head.py`.
Thesis results: seg +0.011 mIoU, det +0.089 mAP (full validation,
multi-seed).
