"""Full-cohort (HBN n=1121) volume-fMRI-FM embedding + probe — harmonized with the
cross-modality benchmark (eeg-fm / smri-fm on the same HBN cohort, modality-restricted).

Only the VOLUME models can scale to the full local cohort: the parcel models read CIFTI
(fsLR) surface data (readers.py) which we don't have locally for HBN (CPAC + the 58-subj
fMRIPrep run are volume-only). So this runs the volume FMs that read MNI niis:
    NeuroSTORM (Swin4D+Mamba), SwiFT (contrastive Swin4D)  [brain mask, 228k voxels]
(CortexMAE-volume / mni_cortex can be added — different 132k cortex-mask extraction.)

Design (good-neighbor on the shared box): LOAD EACH NII ONCE, extract the 228k brain-mask
timeseries once, run every volume model on it -> halves NFS reads. Embeddings cached to
.npz so probing is separate/re-runnable. Probe (5-fold CV): age regression (Pearson r +
MAE yrs), sex (balanced acc), p_factor/attention/internalizing/externalizing regression.

Throttled for the shared DGX Spark: single process, one GPU, nice'd, sequential subjects.

    # phase 1 (embed, ~hours): writes embeddings cache
    python benchmarks/hbn_full_volume.py embed --limit 0
    # phase 2 (probe, fast):
    python benchmarks/hbn_full_volume.py probe
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

MANIFEST = "/data/derivatives/peer_fm_ww/hbn1228_manifest.csv"
EMB_DIR = Path("/data/derivatives/peer_fm_ww/hbn_full/emb"); EMB_DIR.mkdir(parents=True, exist_ok=True)
OUT = Path("/data/derivatives/peer_fm_ww/hbn_full"); OUT.mkdir(parents=True, exist_ok=True)
FACTORS = ["p_factor", "attention", "internalizing", "externalizing"]
sys.path.insert(0, "/home/mhough/dev/wwj/benchmarks")


def load_manifest():
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# embed: load each nii ONCE -> 228k brain-mask timeseries -> all volume models
# ---------------------------------------------------------------------------
def phase_embed(limit=0):
    import hbn_local_probe as H
    import nibabel as nib
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> device={dev}", flush=True)

    # build volume models (reuse validated builders; both use the 228k brain mask)
    ns_w, ns_t = H.build_neurostorm(dev)
    sw_w, sw_t = H.build_swift(dev)
    models = [("neurostorm", ns_w, ns_t), ("swift", sw_w, sw_t)]
    mask = ns_t.mask.cpu().numpy()  # (Z,Y,X) bool, 228k — shared by both transforms

    rows = load_manifest()
    if limit: rows = rows[:limit]
    done = 0
    for i, r in enumerate(rows):
        sub = r["sub"]; out = EMB_DIR / f"{sub}.npz"
        if out.exists():
            done += 1; continue
        path = r["nii"]
        try:
            img = nib.as_closest_canonical(nib.load(path))
            if img.shape[:3] != (91, 109, 91):
                print(f"  [skip {sub}] shape {img.shape}", flush=True); continue
            data = np.asarray(img.dataobj, dtype=np.float32)          # (X,Y,Z,T)
            data = np.ascontiguousarray(data.transpose(3, 2, 1, 0))   # (T,Z,Y,X)
            bold = data[:, mask]                                      # (T, 228k)
        except Exception as e:
            print(f"  [load fail {sub}] {type(e).__name__}: {e}", flush=True); continue

        embs = {}
        for name, w, t in models:
            try:
                sample = {"bold": torch.from_numpy(bold), "tr": float(r["nii_tr"]) if "nii_tr" in r else 0.8,
                          "mean": torch.zeros(1, bold.shape[1]), "std": torch.ones(1, bold.shape[1])}
                sample = t(sample)
                x = sample["bold"].unsqueeze(0).to(dev)
                with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16, enabled=dev.type=="cuda"):
                    o = w({"bold": x})
                embs[name] = o.patch_embeds.float().mean(dim=(0, 1)).cpu().numpy()
            except Exception as e:
                print(f"  [{name} fail {sub}] {type(e).__name__}: {e}", flush=True)
        if embs:
            np.savez(out, **embs); done += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  cached={done}", flush=True)
        del data, bold
    print(f">>> embed done: {done} subjects cached in {EMB_DIR}", flush=True)


# ---------------------------------------------------------------------------
# probe: age/factor regression + sex classification, 5-fold CV
# ---------------------------------------------------------------------------
def _reg_cv(X, y):
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, KFold
    p = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    yp = cross_val_predict(p, X, y, cv=KFold(5, shuffle=True, random_state=0))
    r = float(np.corrcoef(y, yp)[0, 1]); mae = float(np.mean(np.abs(y - yp)))
    return {"pearson_r": r, "mae": mae, "n": len(y)}


def _clf_cv(X, y):
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
                      {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]}, scoring="balanced_accuracy",
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=-1)
    gs.fit(X, y)
    return {"cv_bacc": float(gs.best_score_), "n": len(y), "C": gs.best_params_["logisticregression__C"]}


def phase_probe(model):
    rows = load_manifest()
    lab = {r["sub"]: r for r in rows}
    subs, E = [], []
    for r in rows:
        p = EMB_DIR / f"{r['sub']}.npz"
        if not p.exists(): continue
        d = np.load(p)
        if model not in d: continue
        subs.append(r["sub"]); E.append(d[model])
    if not subs:
        print(f"[{model}] no embeddings"); return None
    X = np.vstack(E)
    print(f"\n=== {model}  (n={len(subs)}, dim={X.shape[1]}) ===", flush=True)
    res = {"model": model, "n": len(subs), "dim": int(X.shape[1]), "tasks": {}}
    def col(name):
        keep = [(i, lab[s][name]) for i, s in enumerate(subs) if lab[s].get(name) not in ("", "n/a", "NaN", None)]
        idx = [i for i, _ in keep]; vals = [v for _, v in keep]
        return idx, vals
    # age regression
    idx, vals = col("age"); y = np.array([float(v) for v in vals])
    res["tasks"]["age"] = _reg_cv(X[idx], y)
    print(f"  age     r={res['tasks']['age']['pearson_r']:.3f} mae={res['tasks']['age']['mae']:.2f}yr n={len(y)}", flush=True)
    # sex classification
    idx, vals = col("sex"); y = np.array(vals)
    res["tasks"]["sex"] = _clf_cv(X[idx], y)
    print(f"  sex     bacc={res['tasks']['sex']['cv_bacc']:.3f} n={res['tasks']['sex']['n']}", flush=True)
    # factor regressions
    for fac in FACTORS:
        idx, vals = col(fac)
        if len(vals) < 50: continue
        y = np.array([float(v) for v in vals])
        res["tasks"][fac] = _reg_cv(X[idx], y)
        print(f"  {fac:14s} r={res['tasks'][fac]['pearson_r']:.3f} n={len(y)}", flush=True)
    (OUT / f"probe_{model}.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["embed", "probe"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["neurostorm", "swift"])
    args = ap.parse_args()
    if args.phase == "embed":
        phase_embed(args.limit)
    else:
        results = [phase_probe(m) for m in args.models]
        (OUT / "probe_all.json").write_text(json.dumps([r for r in results if r], indent=2))
        print(f"\n[done] {OUT}/probe_all.json")


if __name__ == "__main__":
    main()
