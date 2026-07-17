"""
Task 1 — Layer Cosine Similarity: dinov3 vs fusion dinov3 (HR ViT).
Compares first, middle, and last layer cosine-sim maps between:
  - Standalone dinov3 (vit_small, lvd1689m pretrained)
  - Fusion HR ViT (vit_large, backbone.* from fusion checkpoint)
Single potsdam image, 512→3072 resize, random reference patch.

Usage:
    conda activate dinov3-mmlab-wj2
    python task1_layer_cosine.py
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

# ── dinov3 limx bootstrap ──────────────────────────────────────────────────
sys.modules["dinov3.models.RS_vision_transformer"] = types.ModuleType("x")
eu = types.ModuleType("dinov3.eval.utils")
eu.ModelWithIntermediateLayers = type("M", (), {})
sys.modules["dinov3.eval.utils"] = eu
sys.path.insert(0, "/mnt/ht2-nas2/00-model/00-limx/Codes/dinov3-main")
from dinov3.models.vision_transformer import vit_small, vit_large

# ── Paths ──────────────────────────────────────────────────────────────────
VITS_CKPT   = "/mnt/ht2-nas2/00-model/00-fb/mmseg_data/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
FUSION_CKPT = "/mnt/ht2-nas2/00-model/00-common/weights/20260709/weights.pth"
POTSDAM_DIR = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/img_dir"
OUT_DIR     = "/mnt/qh2-nas3/00-model/00-fb/visualise"

IMG_SIZE    = 3072
PATCH_SIZE  = 16
GRID        = IMG_SIZE // PATCH_SIZE  # 192

# potsdam_norm.txt (0-255 scale, RGB order)
POTSDAM_MEAN = np.array([97.61828308705, 92.50345435337714, 85.8699012576488])
POTSDAM_STD  = np.array([36.295481104983764, 35.3808408869616, 36.78625007116312])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _unwrap_checkpoint(ckpt):
    """Unwrap container keys (model/state_dict/teacher/student)."""
    if not isinstance(ckpt, dict):
        return ckpt
    for container in ("model", "state_dict", "teacher", "student"):
        if container in ckpt and isinstance(ckpt[container], dict):
            return ckpt[container]
    return ckpt


def load_potsdam_image(filepath, img_size):
    """Load a potsdam PNG, resize, normalize with potsdam stats (0-255 scale)."""
    img = cv2.cvtColor(cv2.imread(filepath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - POTSDAM_MEAN) / POTSDAM_STD
    return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)


def build_dinov3():
    """Build standalone dinov3 vit_small, load lvd1689m checkpoint."""
    vit = vit_small(patch_size=PATCH_SIZE, img_size=IMG_SIZE,
                    n_storage_tokens=4, layerscale_init=1e-5)
    ckpt = torch.load(VITS_CKPT, map_location="cpu", weights_only=False)
    ckpt = _unwrap_checkpoint(ckpt)
    info = vit.load_state_dict(ckpt, strict=False)
    print(f"[dinov3] matched={len(ckpt)-len(info.unexpected_keys)} "
          f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()


def build_fusion_dino():
    """Build fusion HR ViT vit_large, load backbone.* from fusion checkpoint."""
    vit = vit_large(patch_size=PATCH_SIZE, img_size=IMG_SIZE,
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


def cosine_map(feat, ry, rx):
    """Cosine similarity of all positions vs reference (ry, rx)."""
    fn = feat / (feat.norm(dim=0, keepdim=True) + 1e-8)
    return (fn * fn[:, ry, rx][:, None, None]).sum(0).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── image ──
    all_images = sorted(glob.glob(f"{POTSDAM_DIR}/train/*.png") +
                        glob.glob(f"{POTSDAM_DIR}/val/*.png"))
    rng = random.Random(SEED)
    # img_path = rng.choice(all_images)
    img_path = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/img_dir/train/2_10_2048_4096_2560_4608.png"
    print(f"Image: {img_path}")

    x_normed = load_potsdam_image(img_path, IMG_SIZE)
    x_normed = x_normed.to(DEVICE)

    img_raw = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img_display = cv2.resize(img_raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # ── models ──
    dino = build_dinov3()
    f_dino = build_fusion_dino()

    # ── layers: first, middle, last ──
    dino_layers  = [0, 5, 11]   # vit_small: 12 blocks
    fdino_layers = [0, 11, 23]  # vit_large: 24 blocks

    # ── random reference patch ──
    # ref_x = rng.randint(0, GRID - 1)
    # ref_y = rng.randint(0, GRID - 1)
    ref_x = 30
    ref_y = 100
    px = int((ref_x + 0.5) / GRID * IMG_SIZE)
    py = int((ref_y + 0.5) / GRID * IMG_SIZE)
    print(f"Ref grid: ({ref_x}, {ref_y})  pixel: ({px}, {py})")

    # ── extract & compute cosine ──
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        dino_feats = dino.get_intermediate_layers(
            x_normed, n=dino_layers, reshape=True, norm=True)
        fdino_feats = f_dino.get_intermediate_layers(
            x_normed, n=fdino_layers, reshape=True, norm=True)

    dino_sims = [cosine_map(f[0], ref_y, ref_x) for f in dino_feats]
    fdino_sims = [cosine_map(f[0], ref_y, ref_x) for f in fdino_feats]

    dino_dims = [int(f.shape[1]) for f in dino_feats]
    fdino_dims = [int(f.shape[1]) for f in fdino_feats]

    # ── plot ──
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)

    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(img_display)
    ax_img.plot(px, py, "w+", markersize=20, markeredgewidth=4)
    ax_img.set_title(f"Original Image (seed={SEED}, ref=({ref_x},{ref_y}))",
                     fontsize=14, fontweight="bold")
    ax_img.axis("off")

    labels = ["First Layer", "Middle Layer", "Last Layer"]
    titles_dino = [f"dinov3 (vit_s) — {lbl} (blk {b}, D={d})"
                   for lbl, b, d in zip(labels, dino_layers, dino_dims)]
    titles_fdino = [f"fusion dinov3 (vit_l) — {lbl} (blk {b}, D={d})"
                    for lbl, b, d in zip(labels, fdino_layers, fdino_dims)]

    for col in range(3):
        ax = fig.add_subplot(gs[1, col])
        sim_up = cv2.resize(dino_sims[col], (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_CUBIC)
        im = ax.imshow(sim_up, cmap="viridis", vmin=0, vmax=1.0)
        ax.plot(px, py, "w+", markersize=20, markeredgewidth=4)
        ax.set_title(titles_dino[col], fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for col in range(3):
        ax = fig.add_subplot(gs[2, col])
        sim_up = cv2.resize(fdino_sims[col], (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_CUBIC)
        im = ax.imshow(sim_up, cmap="viridis", vmin=0, vmax=1.0)
        ax.plot(px, py, "w+", markersize=20, markeredgewidth=4)
        ax.set_title(titles_fdino[col], fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = f"{OUT_DIR}/task1_layer_cosine.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")

    for i, s in enumerate(dino_sims):
        print(f"  dinov3 layer {dino_layers[i]}: sim ∈ [{s.min():.3f}, {s.max():.3f}]")
    for i, s in enumerate(fdino_sims):
        print(f"  fusion  layer {fdino_layers[i]}: sim ∈ [{s.min():.3f}, {s.max():.3f}]")


if __name__ == "__main__":
    main()
