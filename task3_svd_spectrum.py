"""
Task 3 — Singular Value Spectrum.
Extracts N=2000 global features from each model, mean-centers (no normalization),
computes SVD, normalizes by sigma_max, and plots singular values in descending
order on log-Y scale.

Features are cached to .pt files for fast re-runs.

Usage:
    conda activate dinov3-mmlab-wj2
    python task3_svd_spectrum.py
"""

import sys
import types
import random
import glob
import os

import numpy as np
import pandas as pd
import torch
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_SAMPLES = 2000
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── dinov3 limx bootstrap ──────────────────────────────────────────────────
sys.modules["dinov3.models.RS_vision_transformer"] = types.ModuleType("x")
eu = types.ModuleType("dinov3.eval.utils")
eu.ModelWithIntermediateLayers = type("M", (), {})
sys.modules["dinov3.eval.utils"] = eu
sys.path.insert(0, "/mnt/ht2-nas2/00-model/00-limx/Codes/dinov3-main")
from dinov3.models.vision_transformer import vit_small, vit_large

# ── OlmoEarth bootstrap ────────────────────────────────────────────────────
_OLMO_ROOT = "/mnt/ht2-nas2/00-model/00-fb/olmo_test/olmoearth_inference_v2_1"
sys.path.insert(0, _OLMO_ROOT)

# ── Paths ──────────────────────────────────────────────────────────────────
VITS_CKPT   = "/mnt/ht2-nas2/00-model/00-fb/mmseg_data/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
FUSION_CKPT = "/mnt/ht2-nas2/00-model/00-common/weights/20260709/weights.pth"
OLMO_CKPT   = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base/weights.pth"
OLMO_H5_DIR = "/mnt/ht2-nas2/00-model/00-fb/olmo_test/inference_data"
POTSDAM_DIR = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/img_dir"
OUT_DIR     = "/mnt/qh2-nas3/00-model/00-fb/visualise"
CACHE_DIR   = os.path.join(OUT_DIR, "cache")

POTSDAM_MEAN = np.array([97.61828308705, 92.50345435337714, 85.8699012576488])
POTSDAM_STD  = np.array([36.295481104983764, 35.3808408869616, 36.78625007116312])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
os.makedirs(CACHE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — checkpoint
# ═══════════════════════════════════════════════════════════════════════════

def _unwrap_checkpoint(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for c in ("model", "state_dict", "teacher", "student"):
        if c in ckpt and isinstance(ckpt[c], dict):
            return ckpt[c]
    return ckpt


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — dinov3 / fusion dinov3
# ═══════════════════════════════════════════════════════════════════════════

def load_potsdam_image(filepath, img_size=512):
    img = cv2.cvtColor(cv2.imread(filepath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - POTSDAM_MEAN) / POTSDAM_STD
    return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)


def build_dinov3(img_size=512):
    vit = vit_small(patch_size=16, img_size=img_size,
                    n_storage_tokens=4, layerscale_init=1e-5)
    ckpt = torch.load(VITS_CKPT, map_location="cpu", weights_only=False)
    ckpt = _unwrap_checkpoint(ckpt)
    info = vit.load_state_dict(ckpt, strict=False)
    print(f"[dinov3] matched={len(ckpt)-len(info.unexpected_keys)} "
          f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()


def build_fusion_dino(img_size=512):
    vit = vit_large(patch_size=16, img_size=img_size,
                    n_storage_tokens=0, layerscale_init=1e-5)
    ckpt = torch.load(FUSION_CKPT, map_location="cpu", weights_only=False)
    ckpt = _unwrap_checkpoint(ckpt)
    sd = {}
    for k, v in ckpt.items():
        if isinstance(v, torch.Tensor) and k.startswith("backbone."):
            sd[k[len("backbone."):]] = v
    info = vit.load_state_dict(sd, strict=False)
    print(f"[fusion_dino] matched={len(sd)-len(info.unexpected_keys)} "
          f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()


def _extract_dino_global_batched(vit, img_size, filepaths, batch_size=64):
    """Extract cls tokens from vit_small/fusion, processing images in batches.

    Determines last-block index from vit.embed_dim (384→vit_small/12 blk,
    1024→vit_large/24 blk).
    """
    last_block = {384: 11, 1024: 23}[vit.embed_dim]
    feats = []
    n = len(filepaths)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xs = [load_potsdam_image(fp, img_size).to(DEVICE, non_blocking=True)
              for fp in filepaths[start:end]]
        x_batch = torch.cat(xs, dim=0)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            out = vit.get_intermediate_layers(
                x_batch, n=[last_block],
                return_class_token=True, reshape=False, norm=True
            )
        _, cls_tok = out[0]  # cls_tok: [B, D]
        for b in range(cls_tok.shape[0]):
            feats.append(cls_tok[b].cpu())
        if end % 400 == 0 or end >= n:
            print(f"  potsdam: {end}/{n}")
    return torch.stack(feats)


def extract_dinov3_global(vit, img_size, filepaths):
    return _extract_dino_global_batched(vit, img_size, filepaths)


def extract_fusion_dino_global(vit, img_size, filepaths):
    return _extract_dino_global_batched(vit, img_size, filepaths)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — OlmoEarth
# ═══════════════════════════════════════════════════════════════════════════

def build_olmoearth():
    from dataload.model import load_model_direct, load_model_with_weights
    model = load_model_direct()
    model = model.to(DEVICE)
    model.eval()
    model = load_model_with_weights(model, OLMO_CKPT)
    return model


def _build_olmo_metadata_csv(sample_indices):
    rows = [{"sample_index": idx, "sentinel2_l2a": 1,
             "sentinel1": 1, "landsat": 1} for idx in sample_indices]
    df = pd.DataFrame(rows)
    csv_path = f"{OUT_DIR}/olmo_metadata_tmp.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def extract_olmoearth_global(model, h5_indices, batch_size=2):
    from torch.utils.data import DataLoader
    from dataload.h5_loader import MultiModalEarthDataset, multimodal_collate_fn

    metadata_csv = _build_olmo_metadata_csv(h5_indices)
    dataset = MultiModalEarthDataset(
        metadata_csv, OLMO_H5_DIR, patch_size=4, normalize_strategy="predefined")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=multimodal_collate_fn,
                        shuffle=False, num_workers=0)

    feats = []
    processed = 0
    total = len(h5_indices)
    for sample in loader:
        sample = sample.to_device(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            output = model(sample, fast_pass=True, patch_size=4)
        tokens_and_masks = output["tokens_and_masks"]

        modality_pools = []
        for mod_name in tokens_and_masks.modalities:
            mod_features = getattr(tokens_and_masks, mod_name)
            pooled = mod_features.mean(dim=[3, 4])
            modality_pools.append(pooled)
        fused = torch.stack(modality_pools).mean(dim=0)
        global_feat = fused.mean(dim=[1, 2])
        feats.append(global_feat.cpu())

        processed += global_feat.shape[0]
        if processed % 200 == 0 or processed >= total:
            print(f"  olmoearth: {processed}/{total}")

    return torch.cat(feats, dim=0)[:total]


def _load_or_extract(cache_path, extract_fn, *extract_args, label=""):
    """Load cached .pt or run extraction and save cache."""
    if os.path.exists(cache_path):
        F = torch.load(cache_path, map_location="cpu")
        print(f"  {label}: loaded cached {tuple(F.shape)}")
        return F
    F = extract_fn(*extract_args)
    torch.save(F, cache_path)
    print(f"  {label}: cached to {cache_path}")
    return F


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cache_key = f"seed{SEED}_n{N_SAMPLES}"

    # ── collect file paths ──
    all_imgs = sorted(glob.glob(f"{POTSDAM_DIR}/train/*.png") +
                      glob.glob(f"{POTSDAM_DIR}/val/*.png"))
    all_h5 = sorted(glob.glob(f"{OLMO_H5_DIR}/sample_*.h5"))
    h5_indices = sorted([int(f.rsplit("_", 1)[-1].split(".")[0]) for f in all_h5])

    rng = random.Random(SEED)
    img_paths = rng.sample(all_imgs, N_SAMPLES)
    olmo_indices = rng.sample(h5_indices, min(N_SAMPLES, len(h5_indices)))
    N_olmo = len(olmo_indices)

    print(f"Potsdam images: {len(all_imgs)} → {N_SAMPLES}")
    print(f"OlmoEarth h5:   {len(h5_indices)} → {N_olmo}")

    # ── dinov3 ──
    print("\n=== dinov3 (vit_small) ===")
    F_dino = _load_or_extract(
        f"{CACHE_DIR}/F_dino_{cache_key}.pt",
        lambda: extract_dinov3_global(build_dinov3(img_size=512), 512, img_paths),
        label="dinov3")
    print(f"  F_dino: {tuple(F_dino.shape)}")
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── fusion dinov3 ──
    print("\n=== fusion dinov3 (vit_large) ===")
    F_fusion = _load_or_extract(
        f"{CACHE_DIR}/F_fusion_{cache_key}.pt",
        lambda: extract_fusion_dino_global(build_fusion_dino(img_size=512), 512, img_paths),
        label="fusion")
    print(f"  F_fusion: {tuple(F_fusion.shape)}")
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── OlmoEarth ──
    print("\n=== OlmoEarth ===")
    F_olmo = _load_or_extract(
        f"{CACHE_DIR}/F_olmo_{cache_key}.pt",
        lambda: extract_olmoearth_global(build_olmoearth(), olmo_indices),
        label="olmoearth")
    print(f"  F_olmo: {tuple(F_olmo.shape)}")
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── mean-center + SVD ──
    def svd_spectrum(F, name):
        Fc = F - F.mean(dim=0, keepdim=True)
        print(f"  {name}: shape={tuple(F.shape)}, centering done")
        U, S, Vh = torch.linalg.svd(Fc.float(), full_matrices=False)
        S = S / S[0]  # normalize by sigma_max
        print(f"  {name}: SVD done, σ/σmax ∈ [{S[-1].item():.6e}, 1], len={len(S)}")
        return S.numpy()

    S_dino   = svd_spectrum(F_dino,   "dinov3")
    S_fusion = svd_spectrum(F_fusion, "fusion dinov3")
    S_olmo   = svd_spectrum(F_olmo,   "OlmoEarth")

    # ── plot ──
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    x_dino   = np.arange(1, len(S_dino) + 1)
    x_fusion = np.arange(1, len(S_fusion) + 1)
    x_olmo   = np.arange(1, len(S_olmo) + 1)

    ax.plot(x_dino, S_dino,
            label=f"dinov3 (vit_s, D=384, N={N_SAMPLES})", lw=1.5)
    ax.plot(x_fusion, S_fusion,
            label=f"fusion dinov3 (vit_l, D=1024, N={N_SAMPLES})", lw=1.5)
    ax.plot(x_olmo, S_olmo,
            label=f"OlmoEarth (D=768, N={N_olmo})", lw=1.5)

    # y-axis: fit to data, no below-curve whitespace
    all_S = np.concatenate([S_dino, S_fusion, S_olmo])
    y_min = all_S.min()
    y_max = 1.0  # normalized by sigma_max, all start at 1
    ax.set_ylim(bottom=y_min * 0.8, top=y_max * 1.05)

    ax.set_yscale("log")
    ax.set_xlabel("Singular Value Index", fontsize=13)
    ax.set_ylabel("σ / σ_max (log scale)", fontsize=13)
    ax.set_title(f"Normalized Singular Value Spectrum (mean-centered, σ/σmax, N≈{N_SAMPLES})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/task3_svd_spectrum.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out_path}")

    print("\nDone. Cached features in:", CACHE_DIR)


if __name__ == "__main__":
    main()
