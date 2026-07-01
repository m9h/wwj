"""Does a POSITION-PRESERVING predictor forecast data sufficiency where β fails? (Canatar–Pehlevan source)

Same deconfounded alignment aρ = cρ²/λρ, two ways:
  * β_sorted  — sort aρ by magnitude, fit the rank-decay (wwj/Thompson). Loses eigenmode position.
  * ρ_0.7     — keep aρ in EIGENVALUE-RANK order; the rank at which cumulative aρ reaches 70% of its
                mass = "how many top-λ (learnable) modes hold the signal". The C–P source, position-kept.
Across a diverse target set (single modes at depth, bands, power-law decays) we measure the empirical
N-to-reach-R²=0.7 and report each predictor's Spearman correlation with log N. Prediction: ρ_0.7 tracks
sufficiency; β_sorted does not — because sorting is exactly what discards the position.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/zeta_thesis_v3.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wwj import alpha_posterior  # noqa: E402
from zeta_diagnostic import rank_decay_from_density_alpha  # noqa: E402

P, NBIG, COV_A, SNR, RIDGE, NTEST = 120, 9000, 1.0, 4.0, 0.5, 1500
NS = [40, 80, 160, 320, 640, 1280, 2560, 5000]
LAM = np.arange(1, P + 1.0) ** (-COV_A)


def make(g, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((NBIG, P))
    X = Z * np.sqrt(LAM)
    g = np.asarray(g, dtype=float); g = g / (np.linalg.norm(g) + 1e-12)
    sig = Z @ g; sig = sig / sig.std()
    y = sig + (1.0 / np.sqrt(SNR)) * rng.standard_normal(NBIG)
    return X, y


def ridge_curve(X, y, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); te, pool = idx[:NTEST], idx[NTEST:]
    Xte, yte = X[te], y[te]; out = []
    for N in NS:
        tr = pool[:N]; Xtr = X[tr]; mu, ym = Xtr.mean(0), y[tr].mean()
        w = np.linalg.solve((Xtr - mu).T @ (Xtr - mu) + RIDGE * np.eye(P), (Xtr - mu).T @ (y[tr] - ym))
        pred = (Xte - mu) @ w + ym
        out.append(float(1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)))
    return np.array(out)


def n_to_reach(lc, thr=0.7):
    for i, r in enumerate(lc):
        if r >= thr:
            if i == 0:
                return float(NS[0])
            r0, r1, n0, n1 = lc[i - 1], lc[i], NS[i - 1], NS[i]
            return float(np.exp(np.log(n0) + (thr - r0) / (r1 - r0) * (np.log(n1) - np.log(n0))))
    return float(NS[-1] * 2)                                   # censored: never reached


def unsorted_alignment(X, y, ridge_frac=1e-3):
    Xc, yc = X - X.mean(0), y - y.mean()
    C = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(C); w, V = w[::-1], V[:, ::-1]      # descending eigenvalue order
    coeff = V.T @ (Xc.T @ yc / len(Xc))
    lam = np.clip(w, 0.0, None)
    return coeff ** 2 / (lam + ridge_frac * lam.max())         # aρ in EIGENVALUE-RANK order (unsorted)


def rho_q(a, q=0.7):
    c = np.cumsum(a) / a.sum()
    return int(np.searchsorted(c, q) + 1)


def beta_sorted(a):
    s = np.sort(a)[::-1]
    return rank_decay_from_density_alpha(alpha_posterior(jnp.asarray(s))["alpha_mean"])


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    def onehot(k):
        v = np.zeros(P); v[k] = 1.0; return v
    def band(a, b):
        v = np.zeros(P); v[a:b] = 1.0; return v
    idx = np.arange(1, P + 1.0)
    targets = {
        "mode1": onehot(0), "mode5": onehot(4), "mode20": onehot(19), "mode50": onehot(49),
        "mode90": onehot(89), "band0-15": band(0, 15), "band40-60": band(40, 60),
        "band70-100": band(70, 100), "decay^-1.0": idx ** -1.0, "decay^-0.3": idx ** -0.3,
    }
    print(f"C–P source (ρ_0.7, position-kept) vs β_sorted  —  predicting N-to-reach R²=0.7")
    print(f"{'target':12s} | {'N_suff':>7s} | {'β_sorted':>8s} | {'ρ_0.7':>6s}", flush=True)
    ns, bs, rq = [], [], []
    for name, g in targets.items():
        X, y = make(g)
        n = n_to_reach(ridge_curve(X, y))
        a = unsorted_alignment(X, y)
        b, r = beta_sorted(a), rho_q(a)
        ns.append(n); bs.append(b); rq.append(r)
        print(f"{name:12s} | {n:7.0f} | {b:8.2f} | {r:6d}", flush=True)
    ln = np.log(ns)
    print(f"\nSpearman corr with log(N_suff):  β_sorted = {spearman(bs, ln):+.3f}   "
          f"ρ_0.7 = {spearman(rq, ln):+.3f}")
    print("If ρ_0.7 ≫ β_sorted in |corr|, the position-preserving source forecasts sufficiency and β does not.")


if __name__ == "__main__":
    main()
