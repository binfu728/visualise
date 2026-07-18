"""
Task 2 — Global Feature Self-Similarity Heatmaps.
N=100 per model: dinov3, fusion_old, fusion_9999, fusion_23999, fusion_full, OlmoEarth.
L2-normalize rows, S = F @ F^T, diag=0, heatmap ∈ [-1,1].
"""

import sys
import types
import random
import glob
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_SAMPLES = 100
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
FUSION_CKPT  = "/mnt/ht2-nas2/00-model/00-common/weights/20260709/weights.pth"
NEW_CKPT     = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/23999.pth"
NEW2_CKPT    = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/9999.pth"
OLMO_CKPT    = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base/weights.pth"
OLMO_H5_DIR  = "/mnt/ht2-nas2/00-model/00-fb/olmo_test/inference_data"
POTSDAM_DIR  = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/img_dir"
OUT_DIR      = "/mnt/qh2-nas3/00-model/00-fb/visualise"
FUS_SIZE     = 480
FUS_GRID     = FUS_SIZE // 16

POTSDAM_MEAN = np.array([97.61828308705, 92.50345435337714, 85.8699012576488])
POTSDAM_STD  = np.array([36.295481104983764, 35.3808408869616, 36.78625007116312])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


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


def build_fusion_old(img_size=512):
    vit = vit_large(patch_size=16, img_size=img_size,
                    n_storage_tokens=0, layerscale_init=1e-5)
    ckpt = torch.load(FUSION_CKPT, map_location="cpu", weights_only=False)
    ckpt = _unwrap_checkpoint(ckpt)
    sd = {k[len("backbone."):]: v for k, v in ckpt.items()
          if isinstance(v, torch.Tensor) and k.startswith("backbone.")}
    info = vit.load_state_dict(sd, strict=False)
    print(f"[fusion_old] matched={len(sd)-len(info.unexpected_keys)} "
          f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()


def build_fusion_23999(img_size=512):
    vit = vit_large(patch_size=16, img_size=img_size,
                    n_storage_tokens=4, layerscale_init=1e-5)
    ckpt = torch.load(NEW_CKPT, map_location="cpu", weights_only=False)
    sd = {k[len("backbone."):]: v for k, v in ckpt.items()
          if isinstance(v, torch.Tensor) and k.startswith("backbone.")}
    info = vit.load_state_dict(sd, strict=False)
    print(f"[fusion_23999] matched={len(sd)-len(info.unexpected_keys)} "
          f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()


def build_fusion_9999(img_size=512):
    vit = vit_large(patch_size=16, img_size=img_size,
                    n_storage_tokens=4, layerscale_init=1e-5)
    ckpt = torch.load(NEW2_CKPT, map_location="cpu", weights_only=False)
    sd = {k[len("backbone."):]: v for k, v in ckpt.items()
          if isinstance(v, torch.Tensor) and k.startswith("backbone.")}
    info = vit.load_state_dict(sd, strict=False)
    print(f"[fusion_9999] matched={len(sd)-len(info.unexpected_keys)} "
          f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()


def build_fusion_full():
    vit = vit_large(patch_size=16, img_size=FUS_SIZE,
                    n_storage_tokens=4, layerscale_init=1e-5)
    mce = MultiLayerCustomEncoder(dim=1024, depth=3, num_heads=8,
                                  num_patches_q=FUS_GRID ** 2,
                                  num_patches_kv=144, ff_mult=4)
    ckpt = torch.load(NEW_CKPT, map_location="cpu", weights_only=False)
    sd_vit = {k[len("backbone."):]: v for k, v in ckpt.items()
              if isinstance(v, torch.Tensor) and k.startswith("backbone.")}
    sd_mce = {k[len("fusion_backbone."):]: v for k, v in ckpt.items()
              if isinstance(v, torch.Tensor) and k.startswith("fusion_backbone.")}
    vi = vit.load_state_dict(sd_vit, strict=False)
    mi = mce.load_state_dict(sd_mce, strict=False)
    print(f"[fusion_full] vit={len(sd_vit)-len(vi.unexpected_keys)} "
          f"mce={len(sd_mce)-len(mi.unexpected_keys)}")
    return vit.to(DEVICE).eval(), mce.to(DEVICE).eval()


def _extract_vit_global_batched(vit, img_size, filepaths, batch_size=64):
    last_block = 23
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
                return_class_token=True, reshape=False, norm=True)
        _, cls_tok = out[0]
        for b in range(cls_tok.shape[0]):
            feats.append(cls_tok[b].cpu())
    return torch.stack(feats)


def extract_dinov3_global(vit, img_size, filepaths):
    return _extract_vit_global_batched(vit, img_size, filepaths)


def extract_fusion_old_global(vit, img_size, filepaths):
    return _extract_vit_global_batched(vit, img_size, filepaths)


def extract_fusion_23999_global(vit, img_size, filepaths):
    return _extract_vit_global_batched(vit, img_size, filepaths)


def extract_fusion_9999_global(vit, img_size, filepaths):
    return _extract_vit_global_batched(vit, img_size, filepaths)


def extract_fusion_full_global(vit, mce, filepaths, batch_size=64):
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
        global_feat = x.mean(dim=1).cpu()
        feats.append(global_feat)
        if end % 100 == 0 or end >= N:
            print(f"  fusion_full: {end}/{N}")
    return torch.cat(feats, dim=0)[:N]


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
        if processed % 20 == 0 or processed >= total:
            print(f"  olmoearth: {processed}/{total}")

    return torch.cat(feats, dim=0)[:total]


def main():
    all_imgs = sorted(glob.glob(f"{POTSDAM_DIR}/train/*.png") +
                      glob.glob(f"{POTSDAM_DIR}/val/*.png"))
    all_h5 = sorted(glob.glob(f"{OLMO_H5_DIR}/sample_*.h5"))
    h5_indices = sorted([int(f.rsplit("_", 1)[-1].split(".")[0]) for f in all_h5])

    rng = random.Random(SEED)
    img_paths = rng.sample(all_imgs, N_SAMPLES)
    olmo_indices = rng.sample(h5_indices, N_SAMPLES)

    print(f"Potsdam: {len(all_imgs)} → {N_SAMPLES}")
    print(f"Olmo:    {len(h5_indices)} → {N_SAMPLES}")

    print("\n=== dinov3 ===")
    dino = build_dinov3(512)
    F_dino = extract_dinov3_global(dino, 512, img_paths)
    print(f"  {tuple(F_dino.shape)}")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n=== fusion_old ===")
    fold = build_fusion_old(512)
    F_fold = extract_fusion_old_global(fold, 512, img_paths)
    print(f"  {tuple(F_fold.shape)}")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n=== fusion_9999 ===")
    f9999 = build_fusion_9999(512)
    F_f9999 = extract_fusion_9999_global(f9999, 512, img_paths)
    print(f"  {tuple(F_f9999.shape)}")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n=== fusion_23999 ===")
    f2399 = build_fusion_23999(512)
    F_f2399 = extract_fusion_23999_global(f2399, 512, img_paths)
    print(f"  {tuple(F_f2399.shape)}")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n=== fusion_full ===")
    vfull, mfull = build_fusion_full()
    F_fufu = extract_fusion_full_global(vfull, mfull, img_paths)
    print(f"  {tuple(F_fufu.shape)}")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n=== OlmoEarth ===")
    olmo = build_olmoearth()
    F_olmo = extract_olmoearth_global(olmo, olmo_indices)
    print(f"  {tuple(F_olmo.shape)}")

    def heatmap(F, name):
        Fn = F / (F.norm(dim=1, keepdim=True) + 1e-8)
        S = Fn @ Fn.t()
        S.fill_diagonal_(0)
        print(f"  {name}: S ∈ [{S.min():.4f}, {S.max():.4f}]")
        return S

    S_dino  = heatmap(F_dino, "dinov3")
    S_fold  = heatmap(F_fold, "fusion_old")
    S_f9999 = heatmap(F_f9999, "fusion_9999")
    S_f2399 = heatmap(F_f2399, "fusion_23999")
    S_fufu  = heatmap(F_fufu, "fusion_full")
    S_olmo  = heatmap(F_olmo, "OlmoEarth")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    data_list = [
        (S_dino,  f"dinov3 (vit_l, D=1024, N={N_SAMPLES})"),
        (S_fold,  f"fusion old (vit_l, stage1, D=1024, N={N_SAMPLES})"),
        (S_f9999, f"fusion 9999 (vit_l, D=1024, N={N_SAMPLES})"),
        (S_f2399, f"fusion 23999 (vit_l, D=1024, N={N_SAMPLES})"),
        (S_fufu,  f"fusion 23999 full (vit+mce, D=1024, N={N_SAMPLES})"),
        (S_olmo,  f"OlmoEarth (D=768, N={N_SAMPLES})"),
    ]
    for ax, (S, title) in zip(axes.flat, data_list):
        im = ax.imshow(S.numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Sample index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/task2_global_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
