"""Predictive-signal scrutiny: does each modality actually predict each target above a permutation null?

The step that must precede any alignment-β read (the pathology writeup's CV discipline): an exponent on a
target the features can't predict is the rank-decay of noise. Leakage-free 5-fold ridge CV-R^2 vs a
label-permutation null. Interpret β ONLY for (modality,target) pairs whose CV-R^2 clears the null.
Ridge solve is linear in y, so the per-fold operator is precomputed once and permutations are cheap.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/hbn_predictive_check.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbn_e1 import MANIFEST, HBN_TARGETS, MIN_VALID, load_hbn  # noqa: E402
from hbn_structural import _zscore, _connectome  # noqa: E402
from hbn_fm_existing import fm_embed, R as FMR  # noqa: E402

VC = "/data/derivatives/volume_conduction"


def _tg():
    rows = list(csv.DictReader(open(MANIFEST)))
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    return {r["sub"]: {t: f(r.get(t, "")) for t in HBN_TARGETS} for r in rows}


def _build_y(ids, tg):
    return {t: np.array([tg.get(s, {}).get(t, np.nan) for s in ids]) for t in HBN_TARGETS}


def cv_r2_null(X, y, k=5, n_perm=200, seed=0):
    m = np.isfinite(y)
    X, y = X[m], y[m]
    n = len(y)
    if n < MIN_VALID:
        return None
    lam = float(X.shape[1])                       # moderate ridge on standardized features
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    Ms, tes, trs = [], [], []
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(k) if j != f])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        Ms.append(Xte @ np.linalg.solve(A, Xtr.T))   # n_te x n_tr operator (y-independent)
        tes.append(te); trs.append(tr)

    def r2(yv):
        pred = np.empty(n)
        for f in range(k):
            tr, te = trs[f], tes[f]
            ym = yv[tr].mean()
            pred[te] = Ms[f] @ (yv[tr] - ym) + ym
        return 1.0 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)

    real = r2(y)
    null = np.array([r2(y[rng.permutation(n)]) for _ in range(n_perm)])
    p = (1 + int(np.sum(null >= real))) / (n_perm + 1)
    return real, float(np.percentile(null, 95)), p, n


def main():
    tg = _tg()
    mods = {}
    feats, tgts = load_hbn()                      # neurostorm, swift (X aligned, target maps)
    for mname in feats:
        mods[mname] = (feats[mname], {t: tgts[mname][t] for t in tgts[mname]})
    for name, sub in [("amaes_fm", "amaes_embed/fomo25_embeddings.npz"),
                      ("fomo60k_fm", "fomo60k_embed/fomo60k_embeddings.npz")]:
        X, ids = fm_embed(f"{FMR}/{sub}")
        mods[name] = (X, _build_y(ids, tg))
    for name, fn in [("morph_4s456", "morphometry_4s456.npz"),
                     ("blockpooled", "structural_emb.npz")]:
        d = np.load(f"{VC}/{fn}", allow_pickle=True)
        ids = [str(i).replace("sub-", "") for i in d["ids"]]
        mods[name] = (_zscore(d["X"].astype(float)), _build_y(ids, tg))
    Xc, idc = _connectome("456")
    mods["connectome"] = (Xc, _build_y(idc, tg))

    print(f"{'modality':14s} {'target':14s} {'n':>5s} {'CV-R2':>8s} {'null95':>8s} {'p':>7s}  signal", flush=True)
    print("-" * 72)
    for mname, (X, ys) in mods.items():
        for t in HBN_TARGETS:
            if t not in ys:
                continue
            r = cv_r2_null(X, ys[t])
            if r is None:
                continue
            real, n95, p, n = r
            sig = "REAL" if (p < 0.05 and real > 0) else "null"
            print(f"{mname:14s} {t:14s} {n:5d} {real:8.3f} {n95:8.3f} {p:7.3f}  {sig}", flush=True)


if __name__ == "__main__":
    main()
