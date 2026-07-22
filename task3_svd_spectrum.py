"""
Task 3 — Singular Value Spectrum (σ/σmax, log-Y).
N=2000 per model: dinov3, fusion_23999 HR, fusion_23999 full, fusion_nocl HR, fusion_nocl full, OlmoEarth.
Mean-center, SVD, normalize by σ_max, plot. Features cached to .pt.
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

sys.modules["dinov3.models.RS_vision_transformer"] = types.ModuleType("x")
eu = types.ModuleType("dinov3.eval.utils")
eu.ModelWithIntermediateLayers = type("M", (), {})
sys.modules["dinov3.eval.utils"] = eu
sys.path.insert(0, "/mnt/ht2-nas2/00-model/00-limx/Codes/dinov3-main")
from dinov3.models.vision_transformer import vit_large
from dinov3.models.croma_vit_crosself_integration_opimize import MultiLayerCustomEncoder

_OLMO_ROOT = "/mnt/ht2-nas2/00-model/00-fb/olmo_test/olmoearth_inference_v2_1"
sys.path.insert(0, _OLMO_ROOT)

VITS_CKPT    = "/mnt/ht2-nas2/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
NEW_CKPT     = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/23999.pth"
NO_CL_CKPT   = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang_no_cl/9999.pth"
OLMO_CKPT    = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base/weights.pth"
OLMO_H5_DIR  = "/mnt/ht2-nas2/00-model/00-fb/olmo_test/inference_data"
POTSDAM_DIR  = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/img_dir"
OUT_DIR      = "/mnt/qh2-nas3/00-model/00-fb/visualise"
CACHE_DIR    = os.path.join(OUT_DIR, "cache")
FUS_SIZE     = 480
FUS_GRID     = FUS_SIZE // 16

POTSDAM_MEAN = np.array([97.61828308705, 92.50345435337714, 85.8699012576488])
POTSDAM_STD  = np.array([36.295481104983764, 35.3808408869616, 36.78625007116312])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
os.makedirs(CACHE_DIR, exist_ok=True)


def _unwrap_checkpoint(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for c in ("model", "state_dict", "teacher", "student"):
        if c in ckpt and isinstance(ckpt[c], dict):
            return ckpt[c]
    return ckpt


def load_potsdam_image(filepath, img_size=512):
    img = cv2.cvtColor(cv2.imread(filepath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - POTSDAM_MEAN) / POTSDAM_STD
    return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)


def build_dinov3(img_size=512):
    vit = vit_large(patch_size=16, img_size=img_size,
                    n_storage_tokens=4, layerscale_init=1e-5)
    ckpt = torch.load(VITS_CKPT, map_location="cpu", weights_only=False)
    ckpt = _unwrap_checkpoint(ckpt)
    info = vit.load_state_dict(ckpt, strict=False)
    print(f"[dinov3] matched={len(ckpt)-len(info.unexpected_keys)} "
          f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()


def _build_vit_backbone(ckpt_path, img_size):
    vit = vit_large(patch_size=16, img_size=img_size,
                    n_storage_tokens=4, layerscale_init=1e-5)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k[len("backbone."):]: v for k, v in ckpt.items()
          if isinstance(v, torch.Tensor) and k.startswith("backbone.")}
    info = vit.load_state_dict(sd, strict=False)
    print(f"  matched={len(sd)-len(info.unexpected_keys)} "
          f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()


def _build_vit_mce(ckpt_path):
    vit = vit_large(patch_size=16, img_size=FUS_SIZE,
                    n_storage_tokens=4, layerscale_init=1e-5)
    mce = MultiLayerCustomEncoder(dim=1024, depth=3, num_heads=8,
                                  num_patches_q=FUS_GRID ** 2,
                                  num_patches_kv=144, ff_mult=4)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_vit = {k[len("backbone."):]: v for k, v in ckpt.items()
              if isinstance(v, torch.Tensor) and k.startswith("backbone.")}
    sd_mce = {k[len("fusion_backbone."):]: v for k, v in ckpt.items()
              if isinstance(v, torch.Tensor) and k.startswith("fusion_backbone.")}
    vi = vit.load_state_dict(sd_vit, strict=False)
    mi = mce.load_state_dict(sd_mce, strict=False)
    print(f"  vit={len(sd_vit)-len(vi.unexpected_keys)} mce={len(sd_mce)-len(mi.unexpected_keys)}")
    return vit.to(DEVICE).eval(), mce.to(DEVICE).eval()


def _extract_vit_global(vit, img_size, filepaths, label="", batch_size=64):
    feats = []
    n = len(filepaths)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xs = [load_potsdam_image(fp, img_size).to(DEVICE, non_blocking=True)
              for fp in filepaths[start:end]]
        x_batch = torch.cat(xs, dim=0)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            _, cls_tok = vit.get_intermediate_layers(
                x_batch, n=[23], return_class_token=True, reshape=False, norm=True)[0]
        for b in range(cls_tok.shape[0]):
            feats.append(cls_tok[b].cpu())
        if end % 400 == 0 or end >= n:
            print(f"  {label}: {end}/{n}")
    return torch.stack(feats)


def _extract_mce_global(vit, mce, filepaths, label="", batch_size=64):
    N = len(filepaths)
    feats = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xs = [load_potsdam_image(fp, FUS_SIZE).to(DEVICE, non_blocking=True)
              for fp in filepaths[start:end]]
        x_batch = torch.cat(xs, dim=0)
        B = x_batch.shape[0]
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            g_patch, _ = vit.get_intermediate_layers(
                x_batch, n=[23], return_class_token=True, norm=True, reshape=False)[0]
            ctx = mce.fusion_mask_token.expand(B, mce.num_patches_kv, -1)
            ctx = ctx.to(dtype=g_patch.dtype, device=g_patch.device)
            bias = mce.attn_bias.to(dtype=g_patch.dtype, device=g_patch.device)
            cbias = mce.cross_attn_bias.to(dtype=g_patch.dtype, device=g_patch.device)
            cscale = mce.cross_scale.to(dtype=g_patch.dtype, device=g_patch.device)
            x = g_patch
            for block in mce.blocks:
                x = x + block["self_attn"](x, bias)
                x = x + cscale * block["cross_attn"](x, ctx, cbias)
                x = x + block["ffn"](x)
            x = mce.norm_out(x)
        feats.append(x.mean(dim=1).cpu())
        if end % 400 == 0 or end >= N:
            print(f"  {label}: {end}/{N}")
    return torch.cat(feats, dim=0)[:N]


def build_olmoearth():
    from dataload.model import load_model_direct, load_model_with_weights
    model = load_model_direct()
    model = model.to(DEVICE).eval()
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


def _load_or_extract(cache_path, extract_fn, label=""):
    if os.path.exists(cache_path):
        F = torch.load(cache_path, map_location="cpu")
        print(f"  {label}: loaded cached {tuple(F.shape)}")
        return F
    F = extract_fn()
    torch.save(F, cache_path)
    print(f"  {label}: cached to {cache_path}")
    return F


def main():
    ck = f"seed{SEED}_n{N_SAMPLES}"

    all_imgs = sorted(glob.glob(f"{POTSDAM_DIR}/train/*.png") +
                      glob.glob(f"{POTSDAM_DIR}/val/*.png"))
    all_h5 = sorted(glob.glob(f"{OLMO_H5_DIR}/sample_*.h5"))
    h5_indices = sorted([int(f.rsplit("_", 1)[-1].split(".")[0]) for f in all_h5])
    rng = random.Random(SEED)
    img_paths = rng.sample(all_imgs, N_SAMPLES)
    olmo_indices = rng.sample(h5_indices, min(N_SAMPLES, len(h5_indices)))
    N_olmo = len(olmo_indices)
    print(f"Potsdam: {len(all_imgs)} → {N_SAMPLES}   Olmo: {len(h5_indices)} → {N_olmo}")

    print("\n=== dinov3 ===")
    F_dino = _load_or_extract(
        f"{CACHE_DIR}/F_dino_{ck}.pt",
        lambda: _extract_vit_global(build_dinov3(512), 512, img_paths, "dinov3"),
        label="dinov3")
    print(f"  {tuple(F_dino.shape)}")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n=== fusion_23999 HR ===")
    F_2399_hr = _load_or_extract(
        f"{CACHE_DIR}/F_fusion_new_{ck}.pt",
        lambda: _extract_vit_global(_build_vit_backbone(NEW_CKPT, 512), 512, img_paths, "23999_hr"),
        label="23999 HR")
    print(f"  {tuple(F_2399_hr.shape)}")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n=== fusion_23999 full ===")
    F_2399_full = _load_or_extract(
        f"{CACHE_DIR}/F_fusion_23999_full_{ck}.pt",
        lambda: _extract_mce_global(*_build_vit_mce(NEW_CKPT), img_paths, "23999_full"),
        label="23999 full")
    print(f"  {tuple(F_2399_full.shape)}")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n=== fusion_nocl HR ===")
    F_nocl_hr = _load_or_extract(
        f"{CACHE_DIR}/F_fusion_nocl_{ck}.pt",
        lambda: _extract_vit_global(_build_vit_backbone(NO_CL_CKPT, 512), 512, img_paths, "nocl_hr"),
        label="nocl HR")
    print(f"  {tuple(F_nocl_hr.shape)}")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n=== fusion_nocl full ===")
    F_nocl_full = _load_or_extract(
        f"{CACHE_DIR}/F_fusion_nocl_full_{ck}.pt",
        lambda: _extract_mce_global(*_build_vit_mce(NO_CL_CKPT), img_paths, "nocl_full"),
        label="nocl full")
    print(f"  {tuple(F_nocl_full.shape)}")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n=== OlmoEarth ===")
    F_olmo = _load_or_extract(
        f"{CACHE_DIR}/F_olmo_{ck}.pt",
        lambda: extract_olmoearth_global(build_olmoearth(), olmo_indices),
        label="olmoearth")
    print(f"  {tuple(F_olmo.shape)}")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    def svd_spectrum(F, name):
        Fc = F - F.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(Fc.float(), full_matrices=False)
        S = S / S[0]
        print(f"  {name}: σ/σmax ∈ [{S[-1].item():.6e}, 1], len={len(S)}")
        return S.numpy()

    S_dino       = svd_spectrum(F_dino,       "dinov3")
    S_2399_hr    = svd_spectrum(F_2399_hr,    "23999 HR")
    S_2399_full  = svd_spectrum(F_2399_full,  "23999 full")
    S_nocl_hr    = svd_spectrum(F_nocl_hr,    "nocl HR")
    S_nocl_full  = svd_spectrum(F_nocl_full,  "nocl full")
    S_olmo       = svd_spectrum(F_olmo,       "OlmoEarth")

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    ax.plot(np.arange(1, len(S_dino)+1),       S_dino,
            label=f"dinov3 (vit_l, D=1024, N={N_SAMPLES})", lw=1.5)
    ax.plot(np.arange(1, len(S_2399_hr)+1),    S_2399_hr,
            label=f"fusion 23999 HR (D=1024, N={N_SAMPLES})", lw=1.5)
    ax.plot(np.arange(1, len(S_2399_full)+1),  S_2399_full,
            label=f"fusion 23999 full (vit+mce, D=1024, N={N_SAMPLES})", lw=1.5)
    ax.plot(np.arange(1, len(S_nocl_hr)+1),    S_nocl_hr,
            label=f"fusion 9999 no_cl HR (D=1024, N={N_SAMPLES})", lw=1.5)
    ax.plot(np.arange(1, len(S_nocl_full)+1),  S_nocl_full,
            label=f"fusion 9999 no_cl full (vit+mce, D=1024, N={N_SAMPLES})", lw=1.5)
    ax.plot(np.arange(1, len(S_olmo)+1),       S_olmo,
            label=f"OlmoEarth (D=768, N={N_olmo})", lw=1.5)

    all_S = np.concatenate([S_dino, S_2399_hr, S_2399_full, S_nocl_hr, S_nocl_full, S_olmo])
    ax.set_ylim(bottom=all_S.min() * 0.8, top=1.05)

    ax.set_yscale("log")
    ax.set_xlabel("Singular Value Index", fontsize=13)
    ax.set_ylabel("σ / σ_max (log scale)", fontsize=13)
    ax.set_title(f"Normalized Singular Value Spectrum (mean-centered, σ/σmax, N≈{N_SAMPLES})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/task3_svd_spectrum.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out_path}")
    print(f"Cached in: {CACHE_DIR}")


if __name__ == "__main__":
    main()
