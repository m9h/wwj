"""Tier-B: does CBP predict REAL neural learning curves, and does eigenmode POSITION matter? (high-signal)

High-signal target battery: FM→FM cross-prediction (one FM's top PCs from another — strong shared signal,
steep curves, real range of difficulty) + age/sex. For each (rep, target) curve, predict R²(N) four ways
from the FROZEN representation's spectra and score by dynamic-range-normalised skill = 1 - MSE/MSE_null:
  CBP-correct : true (λρ, gρ²) pairing (position preserved)   -- 0 free params (ceiling anchored)
  CBP-sorted  : target power sorted onto eigenvalues (β-analog)-- 0 free params (ceiling anchored)
  power-law   : ceil·Nᶜ/(Nᶜ+N0ᶜ), fit c,N0                     -- flexible baseline
  (null skill = 0 by construction).  Confirm = CBP-correct ≈ power-law > CBP-sorted.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/zeta_tierB.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeta_cbp import solve_kappa  # noqa: E402
from hbn_e1 import MANIFEST  # noqa: E402
from hbn_fm_existing import fm_embed, R as FMR  # noqa: E402

EMB = "/data/derivatives/peer_fm_ww/hbn_full/emb"
NS = [50, 100, 200, 400, 800]


def maps():
    rows = list(csv.DictReader(open(MANIFEST)))
    def fn(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    def sx(v):
        u = str(v).strip().upper()
        return 1.0 if u in ("F", "1", "FEMALE") else (0.0 if u in ("M", "0", "MALE") else np.nan)
    return rows, {r["sub"]: fn(r.get("age", "")) for r in rows}, {r["sub"]: sx(r.get("sex", "")) for r in rows}


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
        if model in d:
            subs.append(r["sub"]); E.append(d[model])
    return (np.vstack(E) if E else np.zeros((0, 1))), subs


def curve(X, y, ridge, seed=1):
    rng = np.random.default_rng(seed)
    n = len(y); idx = rng.permutation(n); nte = max(60, n // 4)
    te, pool = idx[:nte], idx[nte:]
    Ns = [N for N in NS if N <= len(pool)]
    p = X.shape[1]; out = []
    for N in Ns:
        tr = pool[:N]; Xtr = X[tr]; mu, ym = Xtr.mean(0), y[tr].mean()
        w = np.linalg.solve((Xtr - mu).T @ (Xtr - mu) + ridge * np.eye(p), (Xtr - mu).T @ (y[tr] - ym))
        pred = (X[te] - mu) @ w + ym
        out.append(1 - np.sum((y[te] - pred) ** 2) / np.sum((y[te] - y[te].mean()) ** 2))
    return np.array(Ns, float), np.array(out)


def spectra(X, y):
    Xc, yc = X - X.mean(0), y - y.mean()
    C = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(C); w, V = np.clip(w[::-1], 0, None), V[:, ::-1]
    coeff = V.T @ (Xc.T @ yc / len(Xc))
    return w, coeff ** 2 / (w + 1e-3 * w.max())


def cbp_curve(lam, g2, R, Ns, ceil):
    g2 = g2 / g2.sum() * ceil
    sigma2 = 1.0 - ceil; out = []
    for N in Ns:
        k = solve_kappa(lam, N, R)
        L = N * lam / (N * lam + k)
        gamma = float(np.sum(L ** 2) / N)
        Eg = (float(np.sum((1 - L) ** 2 * g2)) + sigma2 * gamma) / max(1 - gamma, 1e-9)
        out.append(1.0 - (Eg + sigma2))
    return np.array(out)


def pl_mse(Ns, r, ceil):
    best = np.inf
    for c in np.linspace(0.3, 2.5, 12):
        for n0 in np.logspace(1, 3.5, 12):
            best = min(best, float(np.mean((ceil * Ns ** c / (Ns ** c + n0 ** c) - r) ** 2)))
    return best


def main():
    rows, age, sex = maps()
    reps = {}
    for m in ["neurostorm", "swift", "cortex_mae_volume"]:
        X, subs = load_fm_cache(m, rows)
        if len(subs) >= 300:
            reps[m] = {s: X[i] for i, s in enumerate(subs)}
    for name, sub in [("amaes_fm", "amaes_embed/fomo25_embeddings.npz"),
                      ("fomo60k_fm", "fomo60k_embed/fomo60k_embeddings.npz")]:
        X, ids = fm_embed(f"{FMR}/{sub}"); reps[name] = {s: X[i] for i, s in enumerate(ids)}

    sk = {"CBP-correct": [], "CBP-sorted": [], "power-law": []}
    skhi = {"CBP-correct": [], "CBP-sorted": [], "power-law": []}
    npts = nhi = 0

    def run(XA, y, tag, rname):
        nonlocal npts, nhi
        y = (y - y.mean()) / (y.std() + 1e-9)
        Xc = XA - XA.mean(0)
        ridge = float((Xc.T @ Xc / len(XA)).trace())
        Ns, r = curve(XA, y, ridge)
        ceil = float(r[-1])
        if ceil < 0.05 or len(Ns) < 4:
            return
        lam, g2 = spectra(XA, y)
        cc = cbp_curve(lam, g2, ridge, Ns, ceil)
        cs = cbp_curve(np.sort(lam)[::-1], np.sort(g2)[::-1], ridge, Ns, ceil)
        vnull = float(np.var(r)) + 1e-9
        s_cc = 1 - float(np.mean((cc - r) ** 2)) / vnull
        s_cs = 1 - float(np.mean((cs - r) ** 2)) / vnull
        s_pl = 1 - pl_mse(Ns, r, ceil) / vnull
        for k, v in [("CBP-correct", s_cc), ("CBP-sorted", s_cs), ("power-law", s_pl)]:
            sk[k].append(v)
            if ceil >= 0.30:
                skhi[k].append(v)
        npts += 1; nhi += ceil >= 0.30
        print(f"{rname:12s} {tag:14s} ceil={ceil:4.2f} | skill CBPcor={s_cc:+.2f} CBPsrt={s_cs:+.2f} plaw={s_pl:+.2f}", flush=True)

    for A, va in reps.items():
        idsA = list(va)
        for tname, tmap in [("age", age), ("sex", sex)]:
            common = [s for s in idsA if s in tmap and np.isfinite(tmap[s])]
            if len(common) >= 300:
                run(np.array([va[s] for s in common]), np.array([tmap[s] for s in common]), tname, A)
        for B, vb in reps.items():
            if B == A:
                continue
            common = [s for s in idsA if s in vb]
            if len(common) < 350:
                continue
            XB = np.array([vb[s] for s in common])
            U, S, _ = np.linalg.svd(XB - XB.mean(0), full_matrices=False)
            pcs = U[:, :4] * S[:4]
            XA = np.array([va[s] for s in common])
            for t in range(4):
                run(XA, pcs[:, t], f"{B[:6]}.PC{t}", A)

    print(f"\n{npts} curves ({nhi} high-ceiling ≥0.30).  mean skill = 1 - MSE/MSE_null (higher=better):")
    for k in ["CBP-correct", "CBP-sorted", "power-law"]:
        hi = f"{np.mean(skhi[k]):+.3f}" if skhi[k] else "  n/a"
        print(f"  {k:12s} all={np.mean(sk[k]):+.3f}   high-ceiling={hi}")
    print("\nConfirm = CBP-correct ≈ power-law > CBP-sorted (theory predicts real curves; position helps).")


if __name__ == "__main__":
    main()
