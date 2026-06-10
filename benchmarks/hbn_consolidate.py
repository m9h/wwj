"""Consolidate the local HBN mini-study: join each model's HBN downstream probe
(age_bin, sex) with its weights-only RG diagnostics, add the FC/Connectome baseline,
and emit a leaderboard table + figure.

    cd /home/mhough/dev/wwj && uv run python benchmarks/hbn_consolidate.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

HBN = "/data/derivatives/peer_fm_ww/results/hbn_probe"
CMAE = "/data/derivatives/cortexmae_ww/results"
PEER = "/data/derivatives/peer_fm_ww/results"
ARROW457 = "/data/datasets/brainmarks/hbn-rest.schaefer400_tians3_buckner7.arrow"
ARROW424 = "/data/datasets/brainmarks/hbn-rest.a424_mni.arrow"

# model -> (RG summary path, arrow space, family)
MODELS = {
    "cortex_mae_parcel_s400ts3": (f"{CMAE}/cortex_mae_parcel_s400ts3_summary.json", "457", "CortexMAE"),
    "cortex_mae_parcel_a424":    (f"{CMAE}/cortex_mae_parcel_a424_summary.json",    "424", "CortexMAE"),
    "brain_semantoks":           (f"{PEER}/brain_semantoks_summary.json",           "457", "Brain-Semantoks"),
}


def fc_baseline(arrow_path, targets):
    """Connectome baseline: Pearson FC upper-triangle -> logistic, 5-fold CV bacc."""
    from datasets import load_from_disk
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    d = load_from_disk(arrow_path)
    rows = [ex for sp in ("train", "validation", "test") for ex in d[sp]]
    def fc(b):
        b = np.asarray(b, np.float32); b = (b - b.mean(0)) / (b.std(0) + 1e-6)
        b = np.nan_to_num(b); c = np.nan_to_num(np.corrcoef(b.T))
        return c[np.triu_indices(c.shape[0], 1)]
    feat = {ex["sub"]: fc(ex["bold"]) for ex in rows}
    out = {}
    for t in targets:
        tmap = json.load(open(f"/data/datasets/brainmarks/targets/hbn_target_map_{t}.json"))
        X, y = [], []
        for s, f in feat.items():
            if s in tmap and str(tmap[s]) != "nan":
                X.append(f); y.append(tmap[s])
        X = np.array(X); y = np.array(y)
        gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
                          {"logisticregression__C": [1e-3, 1e-2, 1e-1]},
                          scoring="balanced_accuracy", cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=-1)
        gs.fit(X, y)
        out[t] = float(gs.best_score_)
    return out


def main():
    targets = ["age_bin", "sex"]
    rows = []
    for name, (rg_path, space, fam) in MODELS.items():
        probe = json.load(open(f"{HBN}/hbn_probe_{name}.json"))
        rg = json.load(open(rg_path)) if Path(rg_path).exists() else {}
        rows.append({
            "model": name, "family": fam, "space": space,
            "age_bin": probe["tasks"].get("age_bin", {}).get("cv_bacc"),
            "sex": probe["tasks"].get("sex", {}).get("cv_bacc"),
            "alpha_median": rg.get("alpha_median"), "frac_alpha_lt2": rg.get("frac_alpha_lt2"),
            "phi_1_median": rg.get("phi_1_median"), "M_tr_median": rg.get("M_tr_median"),
            "n_traps": rg.get("n_traps_isolated_total"),
        })
    # FC / Connectome baselines per space
    fc457 = fc_baseline(ARROW457, targets)
    fc424 = fc_baseline(ARROW424, targets)
    rows.append({"model": "Connectome-FC", "family": "baseline", "space": "457",
                 "age_bin": fc457["age_bin"], "sex": fc457["sex"]})
    rows.append({"model": "Connectome-FC", "family": "baseline", "space": "424",
                 "age_bin": fc424["age_bin"], "sex": fc424["sex"]})

    import pandas as pd
    df = pd.DataFrame(rows)
    out = Path(HBN) / "hbn_leaderboard_joined.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # markdown
    L = ["## Local HBN-rest downstream leaderboard (5-fold CV balanced-accuracy, n=178)", "",
         "fMRI-FM encoders probed on local HBN resting-state data, joined with weights-only",
         "RG diagnostics. age_bin chance=0.25, sex chance=0.50. Connectome-FC = Pearson",
         "functional-connectivity baseline (the bar an FM must clear).", "",
         "| model | family | space | age_bin | sex | α-med | %α<2 | φ₁ | M_tr | traps |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def f(v, n=3): return "—" if v is None or (isinstance(v,float) and not np.isfinite(v)) else (f"{v:.{n}f}" if isinstance(v,float) else str(v))
    for _, r in df.iterrows():
        L.append("| " + " | ".join([str(r['model']), str(r['family']), str(r['space']),
                  f(r['age_bin']), f(r['sex']), f(r.get('alpha_median')), f(r.get('frac_alpha_lt2')),
                  f(r.get('phi_1_median'),4), f(r.get('M_tr_median'),0), f(r.get('n_traps'),0)]) + " |")
    Path(f"{HBN}/hbn_leaderboard.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote {out}\nwrote {HBN}/hbn_leaderboard.md")


if __name__ == "__main__":
    main()
