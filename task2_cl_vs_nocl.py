"""
Task 2b — CL vs No-CL Heatmaps.
2×4 layout:
  Row 1 (with CL):    hr 9999, fusion 9999, hr 23999, fusion 23999
  Row 2 (no CL):      no_cl hr 9999, no_cl fusion 9999, no_cl hr 23999, no_cl fusion 23999
"""

import sys, types, random, glob, numpy as np, torch, cv2
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42; N = 100
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

sys.modules["dinov3.models.RS_vision_transformer"] = types.ModuleType("x")
eu = types.ModuleType("dinov3.eval.utils"); eu.ModelWithIntermediateLayers = type("M",(),{})
sys.modules["dinov3.eval.utils"] = eu
sys.path.insert(0, "/mnt/ht2-nas2/00-model/00-limx/Codes/dinov3-main")
from dinov3.models.vision_transformer import vit_large
from dinov3.models.croma_vit_crosself_integration_opimize import MultiLayerCustomEncoder

CL_9999     = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/9999.pth"
CL_23999    = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/23999.pth"
NO_CL_9999  = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang_no_cl/9999.pth"
NO_CL_23999 = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang_no_cl/23999.pth"
POTSDAM_DIR = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/img_dir"
OUT_DIR     = "/mnt/qh2-nas3/00-model/00-fb/visualise"
FUS_SIZE, FUS_GRID = 480, 30

POTSDAM_MEAN = np.array([97.61828308705, 92.50345435337714, 85.8699012576488])
POTSDAM_STD  = np.array([36.295481104983764, 35.3808408869616, 36.78625007116312])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def load_img(fp, sz=512):
    img = cv2.cvtColor(cv2.imread(fp,1), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(sz,sz)).astype(np.float32)
    return torch.from_numpy(((img-POTSDAM_MEAN)/POTSDAM_STD)).float().permute(2,0,1).unsqueeze(0)

def _build_vit(ckpt_path, img_size):
    vit = vit_large(patch_size=16, img_size=img_size, n_storage_tokens=4, layerscale_init=1e-5)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k[9:]:v for k,v in ckpt.items() if isinstance(v,torch.Tensor) and k.startswith("backbone.")}
    vit.load_state_dict(sd, strict=False)
    return vit.to(DEVICE).eval()

def _build_mce(ckpt_path):
    vit = vit_large(patch_size=16, img_size=FUS_SIZE, n_storage_tokens=4, layerscale_init=1e-5)
    mce = MultiLayerCustomEncoder(dim=1024, depth=3, num_heads=8, num_patches_q=FUS_GRID**2, num_patches_kv=144, ff_mult=4)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sdv = {k[9:]:v for k,v in ckpt.items() if isinstance(v,torch.Tensor) and k.startswith("backbone.")}
    sdm = {k[15:]:v for k,v in ckpt.items() if isinstance(v,torch.Tensor) and k.startswith("fusion_backbone.")}
    vit.load_state_dict(sdv, strict=False); mce.load_state_dict(sdm, strict=False)
    return vit.to(DEVICE).eval(), mce.to(DEVICE).eval()

def _extract_hr(vit, fps, img_size=512, bs=64):
    feats = []
    for s in range(0,len(fps),bs):
        xb = torch.cat([load_img(fp,img_size).to(DEVICE) for fp in fps[s:s+bs]])
        with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
            _,cls = vit.get_intermediate_layers(xb,n=[23],return_class_token=True,reshape=False,norm=True)[0]
        feats.extend(cls.cpu().chunk(xb.shape[0]))
    return torch.stack([f.squeeze(0) for f in feats])

def _extract_fusion(vit, mce, fps, label="", bs=64):
    feats = []
    for s in range(0,len(fps),bs):
        e=min(s+bs,len(fps))
        xb = torch.cat([load_img(fp,FUS_SIZE).to(DEVICE) for fp in fps[s:e]])
        B=xb.shape[0]
        with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
            gp,_ = vit.get_intermediate_layers(xb,n=[23],return_class_token=True,reshape=False,norm=True)[0]
            ctx = mce.fusion_mask_token.expand(B,mce.num_patches_kv,-1).to(gp.dtype)
            bias = mce.attn_bias.to(gp.dtype); cbias = mce.cross_attn_bias.to(gp.dtype); cs = mce.cross_scale.to(gp.dtype)
            x=gp
            for blk in mce.blocks:
                x = x+blk["self_attn"](x,bias); x = x+cs*blk["cross_attn"](x,ctx,cbias); x = x+blk["ffn"](x)
            x = mce.norm_out(x)
        feats.append(x.mean(1).cpu())
        if e%100==0 or e>=len(fps): print(f"  {label}: {e}/{len(fps)}")
    return torch.cat(feats)[:len(fps)]

def main():
    all_imgs = sorted(glob.glob(f"{POTSDAM_DIR}/train/*.png")+glob.glob(f"{POTSDAM_DIR}/val/*.png"))
    fps = random.Random(SEED).sample(all_imgs, N)
    print(f"Potsdam: {len(all_imgs)} → {N}")

    configs = [
        ("hr 9999",        CL_9999,     "hr"),
        ("fusion 9999",    CL_9999,     "fusion"),
        ("hr 23999",       CL_23999,    "hr"),
        ("fusion 23999",   CL_23999,    "fusion"),
        ("no_cl hr 9999",  NO_CL_9999,  "hr"),
        ("no_cl fusion 9999", NO_CL_9999, "fusion"),
        ("no_cl hr 23999", NO_CL_23999, "hr"),
        ("no_cl fusion 23999", NO_CL_23999, "fusion"),
    ]

    results = []
    for name, ckpt, mode in configs:
        print(f"\n=== {name} ===")
        if mode == "hr":
            vit = _build_vit(ckpt, 512)
            F = _extract_hr(vit, fps)
        else:
            vit, mce = _build_mce(ckpt)
            F = _extract_fusion(vit, mce, fps, name)
        Fn = F / (F.norm(dim=1,keepdim=True)+1e-8)
        S = Fn @ Fn.t(); S.fill_diagonal_(0)
        print(f"  {name}: S ∈ [{S.min():.4f},{S.max():.4f}]")
        results.append((S.numpy(), name))
        if DEVICE.type=="cuda": torch.cuda.empty_cache()

    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    titles = [
        "hr 9999 (CL)", "fusion 9999 (CL)", "hr 23999 (CL)", "fusion 23999 (CL)",
        "no_cl hr 9999", "no_cl fusion 9999", "no_cl hr 23999", "no_cl fusion 23999",
    ]
    for ax, (S, name), title in zip(axes.flat, results, titles):
        im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"{title} (D=1024, N={N})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample idx"); ax.set_ylabel("Sample idx")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Row 1: With Contrastive Loss    |    Row 2: No Contrastive Loss", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.96])
    out = f"{OUT_DIR}/task2_cl_vs_nocl.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out}")

if __name__=="__main__": main()
