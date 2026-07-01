"""What actually predicts data sufficiency? β sorts the alignment by magnitude and loses eigenmode POSITION.

Stark test: single-mode targets y = Z_k + noise, placed at increasing eigenvalue depth k (k=1 is the
top/high-λ mode, k=80 a deep low-λ mode). The FROZEN-representation alignment spectrum is a single spike
for every one of them, so β (and the covariance-α) are IDENTICAL across cases — yet learnability, and
hence the learning curve, should differ enormously (top mode saturates instantly; deep mode never does).
If so, neither β nor the (α,β) pair can forecast sufficiency; the position-preserving quantity (how
target power sits over the eigenvalue rank — the Canatar–Pehlevan source) is what's needed.

NOTE: features are NOT standardized here — standardizing would erase the λ scaling that IS the signal.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/zeta_thesis_v2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wwj import alpha_posterior  # noqa: E402
from zeta_diagnostic import covariance_spectrum, alignment_spectrum, rank_decay_from_density_alpha  # noqa: E402

P, NBIG, COV_A, SNR, RIDGE, NTEST = 120, 9000, 1.0, 4.0, 0.5, 1500
NS = [40, 80, 160, 320, 640, 1280, 2560, 5000]
LAM = np.arange(1, P + 1.0) ** (-COV_A)


def make(mode_k, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((NBIG, P))
    X = Z * np.sqrt(LAM)
    y = Z[:, mode_k] + (1.0 / np.sqrt(SNR)) * rng.standard_normal(NBIG)   # signal = k-th unit-var score
    return X, y


def ridge_curve(X, y, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    te, pool = idx[:NTEST], idx[NTEST:]
    Xte, yte = X[te], y[te]
    out = []
    for N in NS:
        tr = pool[:N]
        Xtr = X[tr]
        mu, ym = Xtr.mean(0), y[tr].mean()
        Xn, Xt = Xtr - mu, Xte - mu                       # center only — do NOT rescale (keeps λ structure)
        w = np.linalg.solve(Xn.T @ Xn + RIDGE * np.eye(P), Xn.T @ (y[tr] - ym))
        pred = Xt @ w + ym
        out.append(float(1.0 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)))
    return np.array(out)


def read_beta(X, y):
    ap = alpha_posterior(jnp.asarray(np.asarray(alignment_spectrum(X, y), dtype=float)))
    return rank_decay_from_density_alpha(ap["alpha_mean"]), float(ap["p_alpha_lt_2"])


def n_to_reach(lc, thr=0.7):
    """Smallest training-N to reach R²=thr (log-interpolated) — the data-SUFFICIENCY number."""
    for i, r in enumerate(lc):
        if r >= thr:
            if i == 0:
                return NS[0]
            r0, r1, n0, n1 = lc[i - 1], lc[i], NS[i - 1], NS[i]
            frac = (thr - r0) / (r1 - r0)
            return int(np.exp(np.log(n0) + frac * (np.log(n1) - np.log(n0))))
    return None


def main():
    print(f"single-mode targets at increasing eigenvalue depth  (p={P}, snr={SNR}, ridge λ={RIDGE})")
    print(f"{'signal mode k':>13s} {'λ_k (learnab.)':>14s} | {'β_sorted':>8s} {'α_cov':>6s} | "
          f"{'R²@40':>6s} {'R²@5000':>8s} | {'N to reach R²=0.7':>17s}", flush=True)
    for k in [0, 2, 9, 29, 79]:
        X, y = make(k)
        beta, _ = read_beta(X, y)
        acov = alpha_posterior(jnp.asarray(np.asarray(covariance_spectrum(X), dtype=float)))["alpha_mean"]
        lc = ridge_curve(X, y)
        nsuff = n_to_reach(lc)
        print(f"{k+1:13d} {LAM[k]:14.4f} | {beta:8.2f} {acov:6.2f} | "
              f"{lc[0]:6.2f} {lc[-1]:8.2f} | {str(nsuff):>17s}", flush=True)
    print("\nSufficiency (N-to-reach) should track λ_k (eigenmode position) across ~2 orders of magnitude")
    print("while β_sorted and α_cov barely move — i.e. β (and the α,β pair) do NOT forecast sufficiency;")
    print("the position-preserving covariance–alignment overlap (Canatar–Pehlevan source) is what does.")


if __name__ == "__main__":
    main()
