"""
Task 3b — CL vs No-CL SVD Spectrum (σ/σmax, log-Y).
10 curves: dinov3, hr/fusion × {9999,23999} × {CL, no_cl}, OlmoEarth.
Color scheme:
  HR (CL):      blue   |  Fusion (CL):     red
  HR (no_cl):   cyan   |  Fusion (no_cl):  orange
  dinov3:       green  |  OlmoEarth:       purple
  9999 = solid, 23999 = dashed.
"""

import sys, types, random, glob, os, numpy as np, torch, cv2
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42; N = 2000
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

sys.modules["dinov3.models.RS_vision_transformer"] = types.ModuleType("x")
eu = types.ModuleType("dinov3.eval.utils"); eu.ModelWithIntermediateLayers = type("M",(),{})
sys.modules["dinov3.eval.utils"] = eu
sys.path.insert(0, "/mnt/ht2-nas2/00-model/00-limx/Codes/dinov3-main")
from dinov3.models.vision_transformer import vit_large
from dinov3.models.croma_vit_crosself_integration_opimize import MultiLayerCustomEncoder

sys.path.insert(0, "/mnt/ht2-nas2/00-model/00-fb/olmo_test/olmoearth_inference_v2_1")

VITS_CKPT    = "/mnt/ht2-nas2/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
CL_9999      = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/9999.pth"
CL_23999     = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/23999.pth"
NO_CL_9999   = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang_no_cl/9999.pth"
NO_CL_23999  = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang_no_cl/23999.pth"
OLMO_CKPT    = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base/weights.pth"
OLMO_H5_DIR  = "/mnt/ht2-nas2/00-model/00-fb/olmo_test/inference_data"
POTSDAM_DIR  = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/img_dir"
OUT_DIR      = "/mnt/qh2-nas3/00-model/00-fb/visualise"
CACHE_DIR    = os.path.join(OUT_DIR, "cache")
FUS_SIZE, FUS_GRID = 480, 30

POTSDAM_MEAN = np.array([97.61828308705, 92.50345435337714, 85.8699012576488])
POTSDAM_STD  = np.array([36.295481104983764, 35.3808408869616, 36.78625007116312])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
os.makedirs(CACHE_DIR, exist_ok=True)

def _unwrap(ckpt):
    for c in ("model","state_dict","teacher","student"):
        if isinstance(ckpt,dict) and c in ckpt and isinstance(ckpt[c],dict): return ckpt[c]
    return ckpt

def load_img(fp, sz=512):
    img = cv2.cvtColor(cv2.imread(fp,1), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(sz,sz)).astype(np.float32)
    return torch.from_numpy(((img-POTSDAM_MEAN)/POTSDAM_STD)).float().permute(2,0,1).unsqueeze(0)

def build_dinov3(img_size=512):
    vit = vit_large(patch_size=16, img_size=img_size, n_storage_tokens=4, layerscale_init=1e-5)
    vit.load_state_dict(_unwrap(torch.load(VITS_CKPT,map_location="cpu",weights_only=False)), strict=False)
    return vit.to(DEVICE).eval()

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

def _extract_hr(vit, fps, img_size=512, bs=64, label=""):
    feats = []
    for s in range(0,len(fps),bs):
        e=min(s+bs,len(fps))
        xb = torch.cat([load_img(fp,img_size).to(DEVICE) for fp in fps[s:e]])
        with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
            _,cls = vit.get_intermediate_layers(xb,n=[23],return_class_token=True,reshape=False,norm=True)[0]
        feats.extend(cls.cpu().chunk(xb.shape[0]))
        if e%400==0 or e>=len(fps): print(f"  {label}: {e}/{len(fps)}")
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
        if e%400==0 or e>=len(fps): print(f"  {label}: {e}/{len(fps)}")
    return torch.cat(feats)[:len(fps)]

def build_olmoearth():
    from dataload.model import load_model_direct, load_model_with_weights
    m = load_model_direct().to(DEVICE).eval()
    m = load_model_with_weights(m, OLMO_CKPT)
    return m

def extract_olmoearth(model, h5_indices, bs=2):
    from torch.utils.data import DataLoader
    from dataload.h5_loader import MultiModalEarthDataset, multimodal_collate_fn
    df = pd.DataFrame([{"sample_index":i,"sentinel2_l2a":1,"sentinel1":1,"landsat":1} for i in h5_indices])
    csv = f"{OUT_DIR}/olmo_metadata_tmp.csv"; df.to_csv(csv, index=False)
    ds = MultiModalEarthDataset(csv, OLMO_H5_DIR, patch_size=4, normalize_strategy="predefined")
    loader = DataLoader(ds, batch_size=bs, collate_fn=multimodal_collate_fn, shuffle=False, num_workers=0)
    feats = []; proc=0; tot=len(h5_indices)
    for sample in loader:
        sample = sample.to_device(DEVICE)
        with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
            out = model(sample, fast_pass=True, patch_size=4)
        tam = out["tokens_and_masks"]
        pools = [getattr(tam,m).mean(dim=[3,4]) for m in tam.modalities]
        feats.append(torch.stack(pools).mean(0).mean(dim=[1,2]).cpu())
        proc += pools[0].shape[0]
        if proc%200==0 or proc>=tot: print(f"  olmo: {proc}/{tot}")
    return torch.cat(feats)[:tot]

def _load_or_extract(cache_path, fn, label=""):
    if os.path.exists(cache_path):
        F = torch.load(cache_path, map_location="cpu")
        print(f"  {label}: cached {tuple(F.shape)}"); return F
    F = fn(); torch.save(F, cache_path)
    print(f"  {label}: saved cache"); return F

def main():
    ck = f"seed{SEED}_n{N}"
    all_imgs = sorted(glob.glob(f"{POTSDAM_DIR}/train/*.png")+glob.glob(f"{POTSDAM_DIR}/val/*.png"))
    all_h5 = sorted(glob.glob(f"{OLMO_H5_DIR}/sample_*.h5"))
    h5_idx = sorted([int(f.rsplit("_",1)[-1].split(".")[0]) for f in all_h5])
    fps = random.Random(SEED).sample(all_imgs, N)
    olmo_idx = random.Random(SEED).sample(h5_idx, min(N, len(h5_idx)))
    print(f"Potsdam: {len(all_imgs)} → {N}   Olmo: {len(h5_idx)} → {len(olmo_idx)}")

    # (cache_key_suffix, extract_lambda, label)
    jobs = [
        ("dino",         lambda: _extract_hr(build_dinov3(), fps, label="dinov3"),                    "dinov3"),
        ("cl_hr_9999",   lambda: _extract_hr(_build_vit(CL_9999,512), fps, label="cl_hr_9999"),       "hr 9999 (CL)"),
        ("cl_fus_9999",  lambda: _extract_fusion(*_build_mce(CL_9999), fps, "cl_fus_9999"),           "fusion 9999 (CL)"),
        ("cl_hr_23999",  lambda: _extract_hr(_build_vit(CL_23999,512), fps, label="cl_hr_23999"),     "hr 23999 (CL)"),
        ("cl_fus_23999", lambda: _extract_fusion(*_build_mce(CL_23999), fps, "cl_fus_23999"),         "fusion 23999 (CL)"),
        ("nocl_hr_9999", lambda: _extract_hr(_build_vit(NO_CL_9999,512), fps, label="nocl_hr_9999"),  "no_cl hr 9999"),
        ("nocl_fus_9999",lambda: _extract_fusion(*_build_mce(NO_CL_9999), fps, "nocl_fus_9999"),      "no_cl fusion 9999"),
        ("nocl_hr_23999",lambda: _extract_hr(_build_vit(NO_CL_23999,512), fps, label="nocl_hr_23999"),"no_cl hr 23999"),
        ("nocl_fus_23999",lambda: _extract_fusion(*_build_mce(NO_CL_23999), fps, "nocl_fus_23999"),   "no_cl fusion 23999"),
    ]

    Fs = {}
    for suffix, fn, label in jobs:
        print(f"\n=== {label} ===")
        F = _load_or_extract(f"{CACHE_DIR}/F_{suffix}_{ck}.pt", fn, label)
        print(f"  {tuple(F.shape)}")
        if DEVICE.type=="cuda": torch.cuda.empty_cache()

    print("\n=== OlmoEarth ===")
    Fs["olmo"] = _load_or_extract(
        f"{CACHE_DIR}/F_olmo_{ck}.pt",
        lambda: extract_olmoearth(build_olmoearth(), olmo_idx), "olmo")
    print(f"  {tuple(Fs['olmo'].shape)}")

    # SVD
    def svd(F, name):
        Fc = F - F.mean(dim=0,keepdim=True)
        U,S,_ = torch.linalg.svd(Fc.float(), full_matrices=False)
        S = (S/S[0]).numpy()
        print(f"  {name}: σ/σmax [{S[-1]:.2e}, 1]")
        return S

    spectra = {}
    for suffix, _, label in jobs:
        F = _load_or_extract(f"{CACHE_DIR}/F_{suffix}_{ck}.pt", lambda s=suffix: torch.load(f"{CACHE_DIR}/F_{s}_{ck}.pt"), label)
        spectra[suffix] = svd(F, label)
    spectra["olmo"] = svd(torch.load(f"{CACHE_DIR}/F_olmo_{ck}.pt"), "OlmoEarth")

    # Plot with color scheme
    # HR(CL)=blue, Fusion(CL)=red, HR(no_cl)=cyan, Fusion(no_cl)=orange
    # 9999=solid, 23999=dashed; dinov3=green, olmo=purple
    C = {
        "dino":          ("forestgreen", "-"),
        "cl_hr_9999":    ("royalblue",   "-"),
        "cl_hr_23999":   ("royalblue",   "--"),
        "cl_fus_9999":   ("crimson",     "-"),
        "cl_fus_23999":  ("crimson",     "--"),
        "nocl_hr_9999":  ("darkturquoise","-"),
        "nocl_hr_23999": ("darkturquoise","--"),
        "nocl_fus_9999": ("darkorange",  "-"),
        "nocl_fus_23999":("darkorange",  "--"),
        "olmo":          ("purple",      "-"),
    }
    L = {
        "dino":          "dinov3 (D=1024)",
        "cl_hr_9999":    "hr 9999 (CL)",
        "cl_hr_23999":   "hr 23999 (CL)",
        "cl_fus_9999":   "fusion 9999 (CL)",
        "cl_fus_23999":  "fusion 23999 (CL)",
        "nocl_hr_9999":  "no_cl hr 9999",
        "nocl_hr_23999": "no_cl hr 23999",
        "nocl_fus_9999": "no_cl fusion 9999",
        "nocl_fus_23999":"no_cl fusion 23999",
        "olmo":          "OlmoEarth (D=768)",
    }

    fig, ax = plt.subplots(figsize=(14, 8))
    order = ["dino","cl_hr_9999","cl_fus_9999","cl_hr_23999","cl_fus_23999",
             "nocl_hr_9999","nocl_fus_9999","nocl_hr_23999","nocl_fus_23999","olmo"]
    for key in order:
        S = spectra[key]
        color, ls = C[key]
        ax.plot(np.arange(1,len(S)+1), S, color=color, ls=ls, lw=1.5, label=L[key])

    all_S = np.concatenate([spectra[k] for k in order])
    # ax.set_ylim(bottom=all_S.min()*0.8, top=1.05)
    ax.set_ylim(bottom=1e-3, top=1.05)
    ax.set_yscale("log"); ax.set_xlabel("Singular Value Index", fontsize=13)
    ax.set_ylabel("σ / σ_max (log)", fontsize=13)
    ax.set_title(f"SVD Spectrum — CL vs No-CL (σ/σmax, N={N})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{OUT_DIR}/task3_cl_vs_nocl.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out}")

if __name__=="__main__": main()
