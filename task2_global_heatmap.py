"""
Task 2 — Global Feature Self-Similarity Heatmaps.
Extracts N=100 global features from each model, L2-normalizes rows,
computes S = F @ F^T (diagonal=0), and visualises 3 heatmaps.

Models:
  - OlmoEarth: inference_data h5 -> global avg (D=768)
  - dinov3 (vit_small): potsdam -> cls token (D=384)
  - fusion dinov3 (vit_large): potsdam -> cls token (D=1024)

Usage:
    conda activate dinov3-mmlab-wj2
    python task2_global_heatmap.py
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

POTSDAM_MEAN = np.array([97.61828308705, 92.50345435337714, 85.8699012576488])
POTSDAM_STD  = np.array([36.295481104983764, 35.3808408869616, 36.78625007116312])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


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


def extract_dinov3_global(vit, img_size, filepaths):
    """Extract cls token as global feature from dinov3 vit_small."""
    feats = []
    for i, fp in enumerate(filepaths):
        if (i + 1) % 25 == 0:
            print(f"  dinov3: {i+1}/{len(filepaths)}")
        x = load_potsdam_image(fp, img_size).to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            _, cls_tok = vit.get_intermediate_layers(
                x, n=[11], return_class_token=True, reshape=False, norm=True
            )[0]
        feats.append(cls_tok[0].cpu())
    return torch.stack(feats)


def extract_fusion_dino_global(vit, img_size, filepaths):
    """Extract cls token as global feature from fusion vit_large."""
    feats = []
    for i, fp in enumerate(filepaths):
        if (i + 1) % 25 == 0:
            print(f"  fusion_dino: {i+1}/{len(filepaths)}")
        x = load_potsdam_image(fp, img_size).to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            _, cls_tok = vit.get_intermediate_layers(
                x, n=[23], return_class_token=True, reshape=False, norm=True
            )[0]
        feats.append(cls_tok[0].cpu())
    return torch.stack(feats)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — OlmoEarth
# ═══════════════════════════════════════════════════════════════════════════

def build_olmoearth():
    """Build OlmoEarth Encoder and load user-specified weights."""
    from dataload.model import load_model_direct, load_model_with_weights

    model = load_model_direct()
    model = model.to(DEVICE)
    model.eval()
    model = load_model_with_weights(model, OLMO_CKPT)
    return model


def _build_olmo_metadata_csv(sample_indices):
    """Build a minimal metadata CSV for MultiModalEarthDataset."""
    rows = []
    for idx in sample_indices:
        row = {"sample_index": idx,
               "sentinel2_l2a": 1, "sentinel1": 1, "landsat": 1}
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = f"{OUT_DIR}/olmo_metadata_tmp.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def extract_olmoearth_global(model, h5_indices, batch_size=2):
    """Extract global-avg feature from OlmoEarth on h5 samples."""
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
            pooled = mod_features.mean(dim=[3, 4])  # [B,H,W,D], pool over T,S
            modality_pools.append(pooled)
        fused = torch.stack(modality_pools).mean(dim=0)  # [B,H,W,D]
        global_feat = fused.mean(dim=[1, 2])  # [B,D], pool over H,W
        feats.append(global_feat.cpu())

        processed += global_feat.shape[0]
        if processed % 20 == 0 or processed >= total:
            print(f"  olmoearth: {processed}/{total}")

    return torch.cat(feats, dim=0)[:total]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── collect file paths ──
    all_imgs = sorted(glob.glob(f"{POTSDAM_DIR}/train/*.png") +
                      glob.glob(f"{POTSDAM_DIR}/val/*.png"))
    all_h5 = sorted(glob.glob(f"{OLMO_H5_DIR}/sample_*.h5"))
    h5_indices = sorted([int(f.rsplit("_", 1)[-1].split(".")[0]) for f in all_h5])

    rng = random.Random(SEED)
    img_paths = rng.sample(all_imgs, N_SAMPLES)
    olmo_indices = rng.sample(h5_indices, N_SAMPLES)

    print(f"Potsdam images: {len(all_imgs)} → {N_SAMPLES}")
    print(f"OlmoEarth h5:   {len(h5_indices)} → {N_SAMPLES}")

    # ── dinov3 ──
    print("\n=== dinov3 (vit_small) ===")
    dino = build_dinov3(img_size=512)
    F_dino = extract_dinov3_global(dino, 512, img_paths)
    print(f"  F_dino: {tuple(F_dino.shape)}")

    # ── fusion dinov3 ──
    print("\n=== fusion dinov3 (vit_large) ===")
    f_dino = build_fusion_dino(img_size=512)
    F_fusion = extract_fusion_dino_global(f_dino, 512, img_paths)
    print(f"  F_fusion: {tuple(F_fusion.shape)}")

    # ── OlmoEarth ──
    print("\n=== OlmoEarth ===")
    olmo = build_olmoearth()
    F_olmo = extract_olmoearth_global(olmo, olmo_indices)
    print(f"  F_olmo: {tuple(F_olmo.shape)}")

    # ── compute heatmaps ──
    def heatmap(F, name):
        Fn = F / (F.norm(dim=1, keepdim=True) + 1e-8)
        S = Fn @ Fn.t()
        S.fill_diagonal_(0)
        print(f"  {name}: S ∈ [{S.min():.4f}, {S.max():.4f}]")
        return S

    S_dino = heatmap(F_dino, "dinov3")
    S_fusion = heatmap(F_fusion, "fusion dinov3")
    S_olmo = heatmap(F_olmo, "OlmoEarth")

    # ── plot ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    data_list = [
        (S_dino,    f"dinov3 (vit_s, D=384, N={N_SAMPLES})"),
        (S_fusion,  f"fusion dinov3 (vit_l, D=1024, N={N_SAMPLES})"),
        (S_olmo,    f"OlmoEarth (D=768, N={N_SAMPLES})"),
    ]
    for ax, (S, title) in zip(axes, data_list):
        im = ax.imshow(S.numpy(), cmap="RdBu_r", vmin=-1, vmax=1,
                       aspect="auto")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Sample index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/task2_global_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
