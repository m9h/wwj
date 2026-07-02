"""Real-data confirmation: does ρ_q (position-preserving source) predict data sufficiency on HBN, where β fails?

For each HBN modality × {age (R²), sex (AUC)} we compute ρ_0.7 and β_sorted on the frozen representation,
then measure the actual ridge learning curve and extract N-to-reach (80% of the modality's own ceiling).
Report Spearman(predictor, log N_suff). Prediction from the synthetic result: ρ_0.7 tracks it, β does not.
Same X used for both the predictor and the curve (FM raw, classical z-scored); ridge scaled to the spectrum.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/zeta_realdata_rho.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wwj import alpha_posterior  # noqa: E402
from zeta_diagnostic import rank_decay_from_density_alpha, source_spectrum, source_rank  # noqa: E402
from hbn_e1 import MANIFEST  # noqa: E402
from hbn_structural import _zscore  # noqa: E402
from hbn_fm_existing import fm_embed, R as FMR  # noqa: E402

EMB = "/data/derivatives/peer_fm_ww/hbn_full/emb"
VC = "/data/derivatives/volume_conduction"


def manifest():
    rows = list(csv.DictReader(open(MANIFEST)))
    def fnum(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    def fsex(v):
        u = str(v).strip().upper()
        return 1.0 if u in ("F", "1", "FEMALE") else (0.0 if u in ("M", "0", "MALE") else np.nan)
    age = {r["sub"]: fnum(r.get("age", "")) for r in rows}
    sex = {r["sub"]: fsex(r.get("sex", "")) for r in rows}
    return rows, age, sex


def load_fm_cache(model, rows):
    subs, E = [], []
    for r in rows:
        p = Path(EMB) / f"{r['sub']}.npz"
        if not p.exists():
            continue
        try:
            d = np.load(p)
        except Exception:
            continue
        if model not in d:
            continue
        subs.append(r["sub"]); E.append(d[model])
    return (np.vstack(E) if E else np.zeros((0, 1))), subs


def beta_sorted(a):
    s = np.sort(a)[::-1]
    return rank_decay_from_density_alpha(alpha_posterior(jnp.asarray(s))["alpha_mean"])


def auc(yt, sc):
    pos, neg = sc[yt == 1], sc[yt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def curve_and_nsuff(X, y, binary, ridge, seed=1):
    rng = np.random.default_rng(seed)
    n = len(y); idx = rng.permutation(n)
    nte = max(40, n // 4); te, pool = idx[:nte], idx[nte:]
    Ns = sorted({N for N in [25, 50, 100, 200, 400, 800, 1600] if N <= len(pool)} | {len(pool)})
    perf = []
    p = X.shape[1]
    for N in Ns:
        tr = pool[:N]; Xtr = X[tr]; mu, ym = Xtr.mean(0), y[tr].mean()
        w = np.linalg.solve((Xtr - mu).T @ (Xtr - mu) + ridge * np.eye(p), (Xtr - mu).T @ (y[tr] - ym))
        pred = (X[te] - mu) @ w + ym
        perf.append(auc(y[te].astype(int), pred) if binary else
                    1 - np.sum((y[te] - pred) ** 2) / np.sum((y[te] - y[te].mean()) ** 2))
    perf = np.array(perf); chance = 0.5 if binary else 0.0
    ceil = perf[-1]
    if ceil <= chance + 0.03:
        return ceil, None
    thr = chance + 0.8 * (ceil - chance)
    for i, pv in enumerate(perf):
        if pv >= thr:
            if i == 0:
                return ceil, float(Ns[0])
            p0, p1, n0, n1 = perf[i - 1], perf[i], Ns[i - 1], Ns[i]
            return ceil, float(np.exp(np.log(n0) + (thr - p0) / (p1 - p0) * (np.log(n1) - np.log(n0))))
    return ceil, float(Ns[-1])


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    rows, age, sex = manifest()
    mods = []  # (name, X, ids)
    for m in ["neurostorm", "swift", "cortex_mae_volume"]:
        X, subs = load_fm_cache(m, rows)
        if len(subs) >= 250:
            mods.append((m, X, subs))
    for name, sub in [("amaes_fm", "amaes_embed/fomo25_embeddings.npz"),
                      ("fomo60k_fm", "fomo60k_embed/fomo60k_embeddings.npz")]:
        X, ids = fm_embed(f"{FMR}/{sub}"); mods.append((name, X, ids))
    for name, fn in [("morph_4s456", "morphometry_4s456.npz"), ("blockpooled", "structural_emb.npz")]:
        d = np.load(f"{VC}/{fn}", allow_pickle=True)
        mods.append((name, _zscore(d["X"].astype(float)), [str(i).replace("sub-", "") for i in d["ids"]]))

    print(f"{'modality':13s} {'target':4s} | {'n':>5s} {'ceil':>5s} {'N_suff':>7s} | {'ρ_0.7':>6s} {'β_srt':>6s}", flush=True)
    RQ, BS, NS = [], [], []
    for name, X, ids in mods:
        for tname, tmap, binary in [("age", age, False), ("sex", sex, True)]:
            y = np.array([tmap.get(s, np.nan) for s in ids])
            m = np.isfinite(y)
            if int(m.sum()) < 200:
                continue
            Xm, ym = X[m], y[m]
            a, lam = source_spectrum(Xm, ym)
            ridge = float(lam.sum())                    # = trace(C); the calibrated scale (matches the CV probe)
            ceil, nsuff = curve_and_nsuff(Xm, ym, binary, ridge)
            rq, bs = source_rank(a), beta_sorted(a)
            tag = f"{nsuff:7.0f}" if nsuff is not None else "  (null)"
            print(f"{name:13s} {tname:4s} | {int(m.sum()):5d} {ceil:5.2f} {tag} | {rq:6d} {bs:6.2f}", flush=True)
            if nsuff is not None:
                RQ.append(rq); BS.append(bs); NS.append(nsuff)
    ln = np.log(NS)
    print(f"\n{len(NS)} real (modality×target) points with signal.")
    print(f"Spearman with log(N_suff):   ρ_0.7 = {spearman(RQ, ln):+.3f}   β_sorted = {spearman(BS, ln):+.3f}")


if __name__ == "__main__":
    main()
