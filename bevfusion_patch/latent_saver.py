# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
import os
import numpy as np
import torch
import torch.nn.functional as F

def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"

def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default

def _pick_tensor(x):
    
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise RuntimeError("latent_saver: x is empty list/tuple")
        return x[0]
    return x

def save_latent(x, metas):
    

    x = _pick_tensor(x)

    if not torch.is_tensor(x):
        raise RuntimeError(f"latent_saver: expected torch.Tensor, got {type(x)}")

    flatten = _env_flag("FLATTEN_BEV_LATENTS", "0")

    do_pool = _env_flag("LATENT_POOL", "0")
    pool_factor = _env_int("LATENT_POOL_FACTOR", 2)
    pool_mode = _env_str("LATENT_POOL_MODE", "avg").lower()

    dtype_str = _env_str("LATENT_DTYPE", "float16").lower()
    if dtype_str not in ("float16", "float32"):
        dtype_str = "float16"
    np_dtype = np.float16 if dtype_str == "float16" else np.float32
    torch_dtype = torch.float16 if np_dtype == np.float16 else torch.float32

    save_dir = "/output/latents"
    os.makedirs(save_dir, exist_ok=True)

    # Name / token
    if isinstance(metas, (list, tuple)) and len(metas) > 0:
        sample = metas[0].get("sample_idx", None) or metas[0].get("token", "unknown")
    else:
        sample = "unknown"

    bev_t = x.detach()

    # Pool only for 4D: [B,C,H,W]
    if do_pool:
        if bev_t.ndim != 4:
            print(f"[latent_saver] pooling skipped (ndim={bev_t.ndim}, shape={tuple(bev_t.shape)})", flush=True)
        else:
            if pool_factor < 1:
                pool_factor = 1
            if pool_factor > 1:
                if pool_mode == "max":
                    bev_t = F.max_pool2d(bev_t, kernel_size=pool_factor, stride=pool_factor)
                else:
                    bev_t = F.avg_pool2d(bev_t, kernel_size=pool_factor, stride=pool_factor)

    bev_np = bev_t.to(dtype=torch_dtype).cpu().numpy()

    # Flatten
    if flatten:
        if bev_np.ndim != 4:
            raise RuntimeError(f"Expected [B,C,H,W] for flatten, got {bev_np.shape}")
        B, C, H, W = bev_np.shape
        if B == 1:
            bev_np = bev_np.reshape(C, H * W).T  # [K, C]
        else:
            bev_np = bev_np.reshape(B, C, H * W).transpose(0, 2, 1).reshape(B * H * W, C)

    save_path = os.path.join(save_dir, f"bev_latent_{sample}.npy")
    np.save(save_path, bev_np.astype(np_dtype, copy=False))

    print(
        f"[💾 Saved BEV latent → {save_path}] "
        f"shape={bev_np.shape} dtype={bev_np.dtype} "
        f"(pool={'on' if do_pool else 'off'}, factor={pool_factor}, mode={pool_mode}, flatten={flatten})",
        flush=True
    )
