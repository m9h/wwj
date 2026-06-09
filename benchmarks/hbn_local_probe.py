"""Local HBN-rest downstream probe — parallel leaderboard to join with RG diagnostics.

Since the CortexMAE eval_v2 task data is R2-gated, we build a fresh downstream leaderboard
on the LOCAL Brainmarks hbn-rest data (/data/datasets/brainmarks), which ships per-subject
ROI timeseries in three input spaces + HBN phenotype targets. For each input-space-matched
foundation-model encoder we mean-pool patch embeddings over the resting-state run, fit a
logistic probe (C selected on val), and report test balanced-accuracy.

Models are matched to the arrow space by ROI count / modality:
  schaefer400_tians3_buckner7 (457) -> cortex_mae_parcel_s400ts3, Brain-Semantoks (457)
  a424_mni (424)                     -> cortex_mae_parcel_a424, BrainLM (424)
  mni_cortex (volume)                -> cortex_mae_volume, NeuroSTORM/SwiFT (deferred)

This file does the CortexMAE side using cortex_mae.models_mae directly (its encoder forward
is self-contained; we lift only the torch-only pad_unfold). Peer encoders are added via
their own minimal forward adapters (see add_peer_* once their repos are vendored).

    PYTHONPATH=/home/mhough/dev/CortexMAE/src:/home/mhough/dev/CortexMAE/scripts \
    HF_HOME=/data/derivatives/cortexmae_ww/hf_cache \
      python benchmarks/hbn_local_probe.py --models cortex_mae_parcel_s400ts3 --targets sex age_bin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

BM = "/data/datasets/brainmarks"
SPACE_ARROW = {
    "parcel457": f"{BM}/hbn-rest.schaefer400_tians3_buckner7.arrow",
    "parcel424": f"{BM}/hbn-rest.a424_mni.arrow",
    "volume":    f"{BM}/hbn-rest.mni_cortex.arrow",
}
# model -> (input space key, builder tag)
CMAE_MODELS = {
    "cortex_mae_parcel_s400ts3": "parcel457",
    "cortex_mae_parcel_a424":    "parcel424",
}


def pad_unfold(bold, mask, num_frames):
    """Lifted verbatim from cortex_mae.inference (torch-only) to avoid importing the
    neuromaps-heavy inference module."""
    B, C, T, H, W = bold.shape
    if T < num_frames:
        pad = num_frames - T
        bold = F.pad(bold, (0, 0, 0, 0, 0, pad))
        mask = F.pad(mask, (0, 0, 0, 0, 0, pad))
        T = num_frames
    num_clips = T // num_frames
    bold = bold[:, :, : num_clips * num_frames]
    mask = mask[:, :, : num_clips * num_frames]
    if num_clips > 1:
        bold = rearrange(bold, "b c (n f) h w -> (b n) c f h w", n=num_clips)
        mask = rearrange(mask, "b c (n f) h w -> (b n) c f h w", n=num_clips)
    return bold, mask, num_clips


def build_cmae_encoder(name):
    sys.path.insert(0, "/home/mhough/dev/CortexMAE/scripts")
    import weightwatcher_cortexmae as wwc
    hf_prefix, registry = wwc.load_registry()
    enc, args = wwc.build_encoder(name, hf_prefix, registry,
                                  "/data/derivatives/cortexmae_ww/hf_cache")
    return enc.eval(), args


@torch.inference_mode()
def cmae_embed_subject(encoder, args, ex, device):
    """ex: arrow row. Returns mean-pooled embedding [D].

    The model's parcellation may use fewer ROIs than the arrow ships (e.g. the
    schaefer400_tians3 model expects 450 = Schaefer-400 + Tian-50, while the
    schaefer400_tians3_buckner7 arrow has 457 incl. 7 Buckner). Crop ROIs to the
    model's expected spatial size (img_size[1:] = H, W) in arrow order."""
    bold_np = np.asarray(ex["bold"], dtype=np.float32)
    mean = np.asarray(ex["mean"], dtype=np.float32)
    std = np.asarray(ex["std"], dtype=np.float32)
    nf, H, W = encoder.patchify.img_size
    R_keep = H * W
    # Match cortex_mae.transforms normalization (transforms.py:30-35): the arrow bold is
    # stored per-ROI normalized, so DENORMALIZE (bold*std+mean) then apply a GLOBAL scalar
    # z-score. (Doing per-ROI (bold-mean)/std double-normalizes -> destroys signal.)
    raw = bold_np[:, :R_keep] * std[:, :R_keep] + mean[:, :R_keep]
    g = (raw - raw.mean()) / (raw.std() + 1e-6)
    x = torch.from_numpy(g.astype(np.float32))                 # (T, R_keep)
    T = x.shape[0]
    x = x.reshape(1, 1, T, H, W)                               # [B,C,T,H,W]
    mask = torch.ones_like(x)
    xb, mb, nclip = pad_unfold(x, mask, nf)
    xb, mb = xb.to(device), mb.to(device)
    cls, reg, patch = encoder.forward_embedding(xb, mb)        # patch: [nclip, L, D]
    emb = patch.float().mean(dim=(0, 1))                       # pool clips + tokens
    if cls is not None:
        emb = torch.cat([emb, cls.float().mean(dim=(0, 1))])
    return emb.cpu().numpy()


def load_split_embeddings(embed_fn, arrow_path, split):
    from datasets import load_from_disk
    ds = load_from_disk(arrow_path)[split]
    subs, X = [], []
    for ex in ds:
        X.append(embed_fn(ex))
        subs.append(ex["sub"])
    return subs, np.vstack(X)


# ---------------------------------------------------------------------------
# Brain-Semantoks (457-ROI self-distilled CNN-TF) — exact match to the 457 arrow
# ---------------------------------------------------------------------------
SEMANTOKS_REPO = "/data/derivatives/peer_fm_ww/Brain-Semantoks"


def _resample_timeseries(series, tr, new_tr=2.0, kind="cubic", antialias=True):
    """Vendored from brainmarks.nisc.resample_timeseries (scipy-only)."""
    import scipy.signal, scipy.interpolate
    if tr == new_tr:
        return series
    fs, new_fs = 1.0 / tr, 1.0 / new_tr
    if antialias and new_fs < fs:
        q = fs / new_fs
        sos = scipy.signal.cheby1(8, 0.05, 0.8 / q, output="sos")
        series = scipy.signal.sosfiltfilt(sos, series, axis=0, padtype="even")
    x = tr * np.arange(len(series))
    new_len = int(tr * len(series) / new_tr)
    new_x = new_tr * np.arange(new_len)
    new_x = new_x[new_x <= x[-1]]
    interp = scipy.interpolate.interp1d(x, series, kind=kind, axis=0)
    return interp(new_x)


def build_semantoks(device):
    sys.path.insert(0, SEMANTOKS_REPO)
    from model.semantoks import CNN_TF
    from huggingface_hub import hf_hub_download
    nm = hf_hub_download("SamGijsen/Brain-Semantoks", "network_mapping.npz")
    tok_cfg = [{"type": "dense", "kernel_size": 3, "out_channels": 384, "depthwise": False},
               {"type": "sgconv", "kernel_size": 4, "out_channels": 384, "num_scales": 3,
                "decay_min": 2.0, "decay_max": 2.0}]
    enc = CNN_TF(target_time_length=100, patch_size=20, network_data_path=nm,
                 atlas_names=["schaefer400", "tian3", "buckner7"], atlas_network_counts=[7, 1, 1],
                 embedding_dim=768, depth=8, heads=12, mlp_dim=3072, drop_path_rate=0.0,
                 layer_scale_init_value=0.1, emb_dropout=0.0, do_masking=True,
                 tokenizer_config=tok_cfg, tokenizer_final_norm="layer",
                 tokenizer_pooling_type="mean", flash_attention="auto")
    ck = hf_hub_download("SamGijsen/Brain-Semantoks", "brainsemantoks_ckpt_epoch_100.pth")
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    sd = sd.get("model_state_dict", sd)
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    teacher = {k[len("teacher_encoder."):]: v for k, v in sd.items()
               if k.startswith("teacher_encoder.")}
    enc.load_state_dict(teacher)
    return enc.eval().to(device)


@torch.inference_mode()
def semantoks_embed_subject(encoder, ex, device):
    import math
    bold = np.asarray(ex["bold"], dtype=np.float32)        # [T, 457] (z-scored, used directly)
    tr = float(ex["tr"])
    if abs(tr - 2.0) > 0.01:
        bold = _resample_timeseries(bold, tr, 2.0, "cubic", antialias=(len(bold) >= 30))
    T = bold.shape[0]
    if T >= 100:
        bold = bold[:100]; npatch = 5
    else:
        npatch = math.ceil(T / 20)
        bold = np.pad(bold, ((0, 100 - T), (0, 0)))
    x = torch.from_numpy(bold.astype(np.float32).T)[None].to(device)   # [1,457,100]
    m = torch.ones(5, dtype=torch.bool); m[:npatch] = False
    mask = m[None, None].expand(1, 9, 5).to(device)
    with torch.autocast(device_type=device.split(":")[0] if isinstance(device, str) else device.type,
                        dtype=torch.bfloat16, enabled=(str(device).startswith("cuda"))):
        out = encoder(x, atlas=0, mask=mask)        # flash-attn needs fp16/bf16
    cls = out["global_cls"].float()                       # [1, D]
    tok = out["tokens"][:, 1:].float()                    # [1, N*P, D]
    return torch.cat([tok.mean(dim=(0, 1)), cls.mean(0)]).cpu().numpy()


def load_targets(name):
    d = json.load(open(f"{BM}/targets/hbn_target_map_{name}.json"))
    return d


def probe(X, y):
    """5-fold stratified CV logistic probe (C grid-searched on the same folds), pooled
    over all subjects. Returns CV balanced-accuracy mean/std. The fixed n=27 test split is
    too small/noisy; pooled CV over ~178 subjects is the powered estimate (matches the FC
    baseline calibration)."""
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))
    gs = GridSearchCV(pipe, {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]},
                      scoring="balanced_accuracy", cv=cv, n_jobs=-1)
    gs.fit(X, y)
    best = gs.best_index_
    return {"cv_bacc": float(gs.best_score_),
            "cv_bacc_std": float(gs.cv_results_["std_test_score"][best]),
            "C": gs.best_params_["logisticregression__C"],
            "n": len(y), "n_classes": len(set(y))}


def run_model(name, embed_fn, arrow_path, targets, out_dir):
    print(f"\n=== {name}  ({arrow_path.split('/')[-1]}) ===", flush=True)
    # pool all splits -> one (subs, X); we do our own CV
    subs, X = [], []
    for sp in ("train", "validation", "test"):
        s, x = load_split_embeddings(embed_fn, arrow_path, sp)
        subs += s; X.append(x)
    X = np.vstack(X); D = X.shape[1]
    sub2i = {s: i for i, s in enumerate(subs)}
    print(f"  embedded {len(subs)} subjects  dim={D}", flush=True)
    rows = {}
    for tname in targets:
        tmap = load_targets(tname)
        idx = [sub2i[s] for s in tmap if s in sub2i and str(tmap[s]) != "nan"]
        y = np.array([tmap[subs[i]] for i in idx])
        if len(set(y)) < 2 or len(y) < 20:
            print(f"  [skip {tname}] insufficient labels", flush=True); continue
        r = probe(X[idx], y)
        rows[tname] = r
        print(f"  {tname:18s} cv_bacc={r['cv_bacc']:.3f}±{r['cv_bacc_std']:.3f} "
              f"(C={r['C']}, n={r['n']}, k={r['n_classes']}, chance={1/r['n_classes']:.2f})", flush=True)
    out = {"model": name, "arrow": arrow_path, "emb_dim": int(D), "tasks": rows}
    (out_dir / f"hbn_probe_{name}.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(CMAE_MODELS))
    ap.add_argument("--targets", nargs="+", default=["sex", "age_bin"])
    ap.add_argument("--out-dir", default="/data/derivatives/peer_fm_ww/results/hbn_probe")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> device={device}", flush=True)

    results = []
    for name in args.models:
        if name in CMAE_MODELS:
            enc, margs = build_cmae_encoder(name)
            enc = enc.to(device)
            embed_fn = lambda ex, e=enc, a=margs: cmae_embed_subject(e, a, ex, device)
            results.append(run_model(name, embed_fn, SPACE_ARROW[CMAE_MODELS[name]],
                                     args.targets, out_dir))
        elif name == "brain_semantoks":
            enc = build_semantoks(device)
            embed_fn = lambda ex, e=enc: semantoks_embed_subject(e, ex, device)
            results.append(run_model(name, embed_fn, SPACE_ARROW["parcel457"],
                                     args.targets, out_dir))
        else:
            print(f"[skip] no embedder for {name} yet", flush=True)

    (out_dir / "hbn_leaderboard.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {out_dir}/hbn_leaderboard.json")


if __name__ == "__main__":
    main()
