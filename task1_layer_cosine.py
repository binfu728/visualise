"""
Task 1 — Layer Cosine Similarity: 5 model rows.
Rows: dinov3, hr 9999, no_cl hr 9999, hr 23999, no_cl hr 23999.
Columns: first layer, middle layer, last layer cosine maps, PCA of last layer.
"""

import sys, types, random, numpy as np, torch, cv2, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

sys.modules["dinov3.models.RS_vision_transformer"] = types.ModuleType("x")
eu = types.ModuleType("dinov3.eval.utils"); eu.ModelWithIntermediateLayers = type("M",(),{})
sys.modules["dinov3.eval.utils"] = eu
sys.path.insert(0, "/mnt/ht2-nas2/00-model/00-limx/Codes/dinov3-main")
from dinov3.models.vision_transformer import vit_large

VITS_CKPT   = "/mnt/ht2-nas2/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
CL_9999     = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/9999.pth"
CL_23999    = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/23999.pth"
NO_CL_9999  = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang_no_cl/9999.pth"
NO_CL_23999 = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang_no_cl/23999.pth"
OUT_DIR     = "/mnt/qh2-nas3/00-model/00-fb/visualise"

IMG_SIZE, PATCH_SIZE = 3200, 16
GRID = IMG_SIZE // PATCH_SIZE
POTSDAM_MEAN = np.array([97.61828308705, 92.50345435337714, 85.8699012576488])
POTSDAM_STD  = np.array([36.295481104983764, 35.3808408869616, 36.78625007116312])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def _unwrap(ckpt):
    for c in ("model","state_dict","teacher","student"):
        if isinstance(ckpt,dict) and c in ckpt and isinstance(ckpt[c],dict): return ckpt[c]
    return ckpt

def load_img(fp, sz):
    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(sz,sz)).astype(np.float32)
    return torch.from_numpy(((img-POTSDAM_MEAN)/POTSDAM_STD)).float().permute(2,0,1).unsqueeze(0)

def build_dinov3():
    vit = vit_large(patch_size=16, img_size=IMG_SIZE, n_storage_tokens=4, layerscale_init=1e-5)
    ckpt = _unwrap(torch.load(VITS_CKPT, map_location="cpu", weights_only=False))
    info = vit.load_state_dict(ckpt, strict=False)
    print(f"[dinov3] m={len(ckpt)-len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()

def build_hr(ckpt_path, name):
    vit = vit_large(patch_size=16, img_size=IMG_SIZE, n_storage_tokens=4, layerscale_init=1e-5)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k[9:]:v for k,v in ckpt.items() if isinstance(v,torch.Tensor) and k.startswith("backbone.")}
    info = vit.load_state_dict(sd, strict=False)
    print(f"[{name}] m={len(sd)-len(info.unexpected_keys)}")
    return vit.to(DEVICE).eval()

def cosine_map(feat, ry, rx):
    fn = feat / (feat.norm(dim=0, keepdim=True)+1e-8)
    return (fn * fn[:,ry,rx][:,None,None]).sum(0).cpu().numpy()

def pca(feat_np, n=3):
    D,H,W = feat_np.shape
    X = (feat_np.reshape(D,-1).T - feat_np.reshape(D,-1).T.mean(0))
    U,S,_ = np.linalg.svd(X, full_matrices=False)
    comp = (U[:,:n]*S[None,:n]).reshape(H,W,n)
    for c in range(n):
        ch=comp[:,:,c]; comp[:,:,c]=(ch-ch.min())/(ch.max()-ch.min()+1e-8)
    return comp

def main():
    img_path = "/mnt/ht2-nas2/00-model/guantp/dino/mm_dino/data/DIOR-R/JPEGImages-trainval/00050.jpg"
    print(f"Image: {img_path}")
    x = load_img(img_path, IMG_SIZE).to(DEVICE)
    img_disp = cv2.resize(cv2.cvtColor(cv2.imread(img_path,1), cv2.COLOR_BGR2RGB), (IMG_SIZE,IMG_SIZE))

    models = [
        ("dinov3",         build_dinov3(),                    "dinov3 (vit_l, pretrained)"),
        ("hr 9999",        build_hr(CL_9999,    "hr_9999"),   "hr 9999 (with CL)"),
        ("no_cl hr 9999",  build_hr(NO_CL_9999, "nocl_9999"), "no_cl hr 9999"),
        ("hr 23999",       build_hr(CL_23999,   "hr_23999"),  "hr 23999 (with CL)"),
        ("no_cl hr 23999", build_hr(NO_CL_23999,"nocl_23999"),"no_cl hr 23999"),
    ]
    L = [0,11,23]
    ref_x, ref_y = 30, 100
    px, py = int((ref_x+.5)/GRID*IMG_SIZE), int((ref_y+.5)/GRID*IMG_SIZE)
    print(f"Ref: ({ref_x},{ref_y}) px:({px},{py})")

    rows = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for name, vit, title in models:
            feats = vit.get_intermediate_layers(x, n=L, reshape=True, norm=True)
            sims = [cosine_map(f[0], ref_y, ref_x) for f in feats]
            dims = [int(f.shape[1]) for f in feats]
            pc = pca(feats[-1][0].cpu().numpy())
            rows.append((sims, dims, pc, title))
            print(f"  {name}: last sim [{sims[-1].min():.3f},{sims[-1].max():.3f}]")

    fig = plt.figure(figsize=(30, 38))
    gs = GridSpec(6, 4, figure=fig, hspace=0.3, wspace=0.25)
    ax0 = fig.add_subplot(gs[0,:]); ax0.imshow(img_disp)
    ax0.plot(px,py,"w+",ms=20,mew=4); ax0.set_title(f"Original (ref=({ref_x},{ref_y}))",fontsize=14,fontweight="bold"); ax0.axis("off")

    labels = ["First Layer","Middle Layer","Last Layer","PCA of Last Layer"]
    for ri, (sims, dims, pc, title) in enumerate(rows):
        r = ri+1
        for col in range(3):
            ax = fig.add_subplot(gs[r,col])
            up = cv2.resize(sims[col],(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_CUBIC)
            im = ax.imshow(up, cmap="viridis", vmin=0, vmax=1.0)
            ax.plot(px,py,"w+",ms=20,mew=4); plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
            ax.set_title(f"{title} — {labels[col]} (blk {L[col]}, D={dims[col]})",fontsize=12,fontweight="bold"); ax.axis("off")
        ax = fig.add_subplot(gs[r,3])
        ax.imshow(cv2.resize(pc,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_CUBIC))
        ax.plot(px,py,"w+",ms=20,mew=4)
        ax.set_title(f"{title} — {labels[3]} (D={dims[2]})",fontsize=12,fontweight="bold"); ax.axis("off")

    out = f"{OUT_DIR}/task1_layer_cosine.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")

if __name__=="__main__": main()
