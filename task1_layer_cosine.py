"""
Task 1 — Layer Cosine Similarity: dinov3 + 3 fusion timepoints.
Rows: dinov3, fusion_old (stage1), fusion_31999, fusion_23999.
Columns: first layer, middle layer, last layer cosine maps, PCA of last layer.
Single potsdam image, 512→3200 resize, fixed reference patch.
"""

import sys
import types
import random
import numpy as np
import torch
import cv2
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

sys.modules["dinov3.models.RS_vision_transformer"] = types.ModuleType("x")
eu = types.ModuleType("dinov3.eval.utils")
eu.ModelWithIntermediateLayers = type("M", (), {})
sys.modules["dinov3.eval.utils"] = eu
sys.path.insert(0, "/mnt/ht2-nas2/00-model/00-limx/Codes/dinov3-main")
from dinov3.models.vision_transformer import vit_small, vit_large

VITS_CKPT    = "/mnt/ht2-nas2/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
FUSION_CKPT  = "/mnt/ht2-nas2/00-model/00-common/weights/20260709/weights.pth"
NEW_CKPT     = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/23999.pth"
NEW2_CKPT    = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/31999.pth"
POTSDAM_DIR  = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/img_dir"
OUT_DIR      = "/mnt/qh2-nas3/00-model/00-fb/visualise"

IMG_SIZE   = 3200
PATCH_SIZE = 16
GRID       = IMG_SIZE // PATCH_SIZE

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


def load_potsdam_image(filepath, img_size):
    img = cv2.cvtColor(cv2.imread(filepath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - POTSDAM_MEAN) / POTSDAM_STD
    return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)


def build_dinov3():
    vit = vit_large(patch_size=PATCH_SIZE, img_size=IMG_SIZE,
                    n_storage_tokens=4, layerscale_init=1e-5)
    ckpt = torch.load(VITS_CKPT, map_location="cpu", weights_only=False)
    ckpt = _unwrap_checkpoint(ckpt)
    info = vit.load_state_dict(ckpt, strict=False)
    print(f"[dinov3] matched={len(ckpt)-len(info.unexpected_keys)} "
          f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()


def build_fusion_old():
    vit = vit_large(patch_size=PATCH_SIZE, img_size=IMG_SIZE,
                    n_storage_tokens=0, layerscale_init=1e-5)
    ckpt = torch.load(FUSION_CKPT, map_location="cpu", weights_only=False)
    ckpt = _unwrap_checkpoint(ckpt)
    sd = {k[len("backbone."):]: v for k, v in ckpt.items()
          if isinstance(v, torch.Tensor) and k.startswith("backbone.")}
    info = vit.load_state_dict(sd, strict=False)
    print(f"[fusion_old] matched={len(sd)-len(info.unexpected_keys)} "
          f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()


def build_fusion_new():
    vit = vit_large(patch_size=PATCH_SIZE, img_size=IMG_SIZE,
                    n_storage_tokens=4, layerscale_init=1e-5)
    ckpt = torch.load(NEW_CKPT, map_location="cpu", weights_only=False)
    sd = {k[len("backbone."):]: v for k, v in ckpt.items()
          if isinstance(v, torch.Tensor) and k.startswith("backbone.")}
    info = vit.load_state_dict(sd, strict=False)
    print(f"[fusion_23999] matched={len(sd)-len(info.unexpected_keys)} "
          f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()


def build_fusion_31999():
    vit = vit_large(patch_size=PATCH_SIZE, img_size=IMG_SIZE,
                    n_storage_tokens=4, layerscale_init=1e-5)
    ckpt = torch.load(NEW2_CKPT, map_location="cpu", weights_only=False)
    sd = {k[len("backbone."):]: v for k, v in ckpt.items()
          if isinstance(v, torch.Tensor) and k.startswith("backbone.")}
    info = vit.load_state_dict(sd, strict=False)
    print(f"[fusion_31999] matched={len(sd)-len(info.unexpected_keys)} "
          f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()


def cosine_map(feat, ry, rx):
    fn = feat / (feat.norm(dim=0, keepdim=True) + 1e-8)
    return (fn * fn[:, ry, rx][:, None, None]).sum(0).cpu().numpy()


def pca_components(feat_np, n_components=3):
    """feat_np [D, H, W] → top-k PCA projection normalized to [0,1]."""
    D, H, W = feat_np.shape
    X = feat_np.reshape(D, -1).T
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    scores = U[:, :n_components] * S[None, :n_components]
    comp = scores.reshape(H, W, n_components)
    for c in range(n_components):
        ch = comp[:, :, c]
        mn, mx = ch.min(), ch.max()
        comp[:, :, c] = (ch - mn) / (mx - mn + 1e-8)
    return comp


def main():
    all_images = sorted(glob.glob(f"{POTSDAM_DIR}/train/*.png") +
                        glob.glob(f"{POTSDAM_DIR}/val/*.png"))
    # img_path = rng.choice(all_images)
    # img_path = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/img_dir/train/2_10_2048_4096_2560_4608.png"
    # img_path = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/img_dir/train/2_10_1536_2560_2048_3072.png"
    # img_path = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/img_dir/train/2_10_4096_2048_4608_2560.png"
    img_path = "/mnt/ht2-nas2/00-model/guantp/dino/mm_dino/data/DIOR-R/JPEGImages-trainval/00050.jpg"
    print(f"Image: {img_path}")

    x_normed = load_potsdam_image(img_path, IMG_SIZE).to(DEVICE)

    img_raw = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img_display = cv2.resize(img_raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    dino  = build_dinov3()
    fold  = build_fusion_old()
    f2399 = build_fusion_new()
    f31999 = build_fusion_31999()

    dino_layers  = [0, 11, 23]
    vit_l_layers = [0, 11, 23]

    ref_x, ref_y = 30, 100
    px = int((ref_x + 0.5) / GRID * IMG_SIZE)
    py = int((ref_y + 0.5) / GRID * IMG_SIZE)
    print(f"Ref grid: ({ref_x}, {ref_y})  pixel: ({px}, {py})")

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        dino_feats   = dino.get_intermediate_layers(
            x_normed, n=dino_layers, reshape=True, norm=True)
        fold_feats   = fold.get_intermediate_layers(
            x_normed, n=vit_l_layers, reshape=True, norm=True)
        f2399_feats  = f2399.get_intermediate_layers(
            x_normed, n=vit_l_layers, reshape=True, norm=True)
        f31999_feats  = f31999.get_intermediate_layers(
            x_normed, n=vit_l_layers, reshape=True, norm=True)

    dino_sims  = [cosine_map(f[0], ref_y, ref_x) for f in dino_feats]
    fold_sims  = [cosine_map(f[0], ref_y, ref_x) for f in fold_feats]
    f2399_sims = [cosine_map(f[0], ref_y, ref_x) for f in f2399_feats]
    f31999_sims = [cosine_map(f[0], ref_y, ref_x) for f in f31999_feats]

    dino_dims  = [int(f.shape[1]) for f in dino_feats]
    fold_dims  = [int(f.shape[1]) for f in fold_feats]
    f2399_dims = [int(f.shape[1]) for f in f2399_feats]
    f31999_dims = [int(f.shape[1]) for f in f31999_feats]

    # ── PCA on last-layer features ──
    dino_last  = dino_feats[-1][0]
    fold_last  = fold_feats[-1][0]
    f2399_last = f2399_feats[-1][0]
    f31999_last = f31999_feats[-1][0]
    dino_pca  = pca_components(dino_last.cpu().numpy())
    fold_pca  = pca_components(fold_last.cpu().numpy())
    f2399_pca = pca_components(f2399_last.cpu().numpy())
    f31999_pca = pca_components(f31999_last.cpu().numpy())

    fig = plt.figure(figsize=(30, 32))
    gs = GridSpec(5, 4, figure=fig, hspace=0.3, wspace=0.25)

    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(img_display)
    ax_img.plot(px, py, "w+", markersize=20, markeredgewidth=4)
    ax_img.set_title(f"Original Image (seed={SEED}, ref=({ref_x},{ref_y}))",
                     fontsize=14, fontweight="bold")
    ax_img.axis("off")

    labels = ["First Layer", "Middle Layer", "Last Layer", "PCA of Last Layer"]
    # ── dinov3 row ──
    for col in range(3):
        ax = fig.add_subplot(gs[1, col])
        sim_up = cv2.resize(dino_sims[col], (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_CUBIC)
        im = ax.imshow(sim_up, cmap="viridis", vmin=0, vmax=1.0)
        ax.plot(px, py, "w+", markersize=20, markeredgewidth=4)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"dinov3 — {labels[col]} (blk {dino_layers[col]}, D={dino_dims[col]})",
                     fontsize=12, fontweight="bold")
        ax.axis("off")
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(cv2.resize(dino_pca, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC))
    ax.plot(px, py, "w+", markersize=20, markeredgewidth=4)
    ax.set_title(f"dinov3 — {labels[3]} (blk 23, D={dino_dims[2]})",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    # ── fusion_old row ──
    for col in range(3):
        ax = fig.add_subplot(gs[2, col])
        sim_up = cv2.resize(fold_sims[col], (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_CUBIC)
        im = ax.imshow(sim_up, cmap="viridis", vmin=0, vmax=1.0)
        ax.plot(px, py, "w+", markersize=20, markeredgewidth=4)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"fusion old — {labels[col]} (blk {vit_l_layers[col]}, D={fold_dims[col]})",
                     fontsize=12, fontweight="bold")
        ax.axis("off")
    ax = fig.add_subplot(gs[2, 3])
    ax.imshow(cv2.resize(fold_pca, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC))
    ax.plot(px, py, "w+", markersize=20, markeredgewidth=4)
    ax.set_title(f"fusion old — {labels[3]} (blk 23, D={fold_dims[2]})",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    # ── fusion_31999 row ──
    for col in range(3):
        ax = fig.add_subplot(gs[3, col])
        sim_up = cv2.resize(f31999_sims[col], (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_CUBIC)
        im = ax.imshow(sim_up, cmap="viridis", vmin=0, vmax=1.0)
        ax.plot(px, py, "w+", markersize=20, markeredgewidth=4)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"fusion 31999 — {labels[col]} (blk {vit_l_layers[col]}, D={f31999_dims[col]})",
                     fontsize=12, fontweight="bold")
        ax.axis("off")
    ax = fig.add_subplot(gs[3, 3])
    ax.imshow(cv2.resize(f31999_pca, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC))
    ax.plot(px, py, "w+", markersize=20, markeredgewidth=4)
    ax.set_title(f"fusion 31999 — {labels[3]} (blk 23, D={f31999_dims[2]})",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    # ── fusion_23999 row ──
    for col in range(3):
        ax = fig.add_subplot(gs[4, col])
        sim_up = cv2.resize(f2399_sims[col], (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_CUBIC)
        im = ax.imshow(sim_up, cmap="viridis", vmin=0, vmax=1.0)
        ax.plot(px, py, "w+", markersize=20, markeredgewidth=4)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"fusion 23999 — {labels[col]} (blk {vit_l_layers[col]}, D={f2399_dims[col]})",
                     fontsize=12, fontweight="bold")
        ax.axis("off")
    ax = fig.add_subplot(gs[4, 3])
    ax.imshow(cv2.resize(f2399_pca, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC))
    ax.plot(px, py, "w+", markersize=20, markeredgewidth=4)
    ax.set_title(f"fusion 23999 — {labels[3]} (blk 23, D={f2399_dims[2]})",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    out_path = f"{OUT_DIR}/task1_layer_cosine.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")

    for i, s in enumerate(dino_sims):
        print(f"  dinov3       layer {dino_layers[i]}: sim ∈ [{s.min():.3f}, {s.max():.3f}]")
    for i, s in enumerate(fold_sims):
        print(f"  fusion_old   layer {vit_l_layers[i]}: sim ∈ [{s.min():.3f}, {s.max():.3f}]")
    for i, s in enumerate(f31999_sims):
        print(f"  fusion_31999  layer {vit_l_layers[i]}: sim ∈ [{s.min():.3f}, {s.max():.3f}]")
    for i, s in enumerate(f2399_sims):
        print(f"  fusion_23999 layer {vit_l_layers[i]}: sim ∈ [{s.min():.3f}, {s.max():.3f}]")


if __name__ == "__main__":
    main()
