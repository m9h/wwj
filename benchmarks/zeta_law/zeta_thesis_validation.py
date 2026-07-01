"""Does β (read off a FROZEN representation) predict data sufficiency? — the direct test of Thompson's thesis.

Plant synthetic (representation, target) pairs spanning a range of alignment structures. For each:
  (1) READ β off the frozen full-data alignment spectrum with our Bayesian estimator;
  (2) MEASURE the actual ridge learning curve R²(N) as training N grows toward the ceiling.
Then report the EMPIRICAL relationship — does "more data still helps" go with high β or low β? — which
settles the internal contradiction between the nanopath writeup (β≳1 → saturated) and zeta_diagnostic
(β>1 → variance-limited, collect more data). No assumption about the direction is baked in.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/zeta_thesis_validation.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wwj import alpha_posterior  # noqa: E402
from zeta_diagnostic import alignment_spectrum, rank_decay_from_density_alpha  # noqa: E402


def make(p, n, cov_a, tgt_t, snr, seed):
    """Frozen rep X (planted covariance λ_i=i^-cov_a) + target y with per-mode weight c_i=i^-tgt_t + noise.
    The alignment spectrum works out to a_i = c_i^2 = i^-(2·tgt_t), so planted β ≈ 2·tgt_t."""
    rng = np.random.default_rng(seed)
    lam = np.arange(1, p + 1.0) ** (-cov_a)
    Z = rng.standard_normal((n, p))
    X = Z * np.sqrt(lam)
    c = np.arange(1, p + 1.0) ** (-tgt_t)
    sig = Z @ c
    sig = sig / sig.std()
    y = sig + (1.0 / np.sqrt(snr)) * rng.standard_normal(n)   # snr sets the R² ceiling
    return X, y


def ridge_curve(X, y, Ns, lam, n_test, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    te, pool = idx[:n_test], idx[n_test:]
    Xte, yte = X[te], y[te]
    out = []
    for N in Ns:
        tr = pool[:N]
        Xtr = X[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xn, Xt = (Xtr - mu) / sd, (Xte - mu) / sd
        ym = y[tr].mean()
        w = np.linalg.solve(Xn.T @ Xn + lam * np.eye(Xn.shape[1]), Xn.T @ (y[tr] - ym))
        pred = Xt @ w + ym
        out.append(1.0 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2))
    return np.array(out)


def read_beta(X, y):
    a = alignment_spectrum(X, y)
    ap = alpha_posterior(jnp.asarray(np.asarray(a, dtype=float)))
    return rank_decay_from_density_alpha(ap["alpha_mean"]), float(ap["p_alpha_lt_2"])


def main():
    p, nbig, snr, lam, n_test = 150, 8000, 1.0, 5.0, 1500
    Ns = [80, 160, 320, 640, 1280, 2560, 5000]
    print(f"synthetic learning-curve validation of Thompson's β  (p={p}, snr={snr}, ridge λ={lam})")
    print(f"{'tgt_t':>5s} {'planted_β':>9s} {'read_β':>7s} {'P(β>1)':>7s} | "
          f"{'R²@'+str(Ns[0]):>7s} {'R²@'+str(Ns[-1]):>8s} {'late_gain(4×)':>13s} | "
          f"{'observed':>10s} | {'zeta_diag says':>16s}", flush=True)
    for tgt_t in [0.25, 0.5, 0.75, 1.0, 1.5]:
        X, y = make(p, nbig, cov_a=1.0, tgt_t=tgt_t, snr=snr, seed=0)
        b, pb = read_beta(X, y)
        lc = ridge_curve(X, y, Ns, lam, n_test, seed=1)
        late = float(lc[-1] - lc[-3])                       # gain over the last ~4× more data
        observed = "STILL-HELPS" if late > 0.02 else "saturated"
        diag = "variance-limited" if b > 1 else "resolution-lim"   # what the code's label claims
        print(f"{tgt_t:5.2f} {2*tgt_t:9.2f} {b:7.2f} {pb:7.2f} | "
              f"{lc[0]:7.2f} {lc[-1]:8.2f} {late:13.3f} | {observed:>10s} | {diag:>16s}", flush=True)
    print("\nRead: if STILL-HELPS lines up with LOW β (β<1), the writeup is right and the code's "
          "'β>1 = collect more data' label is BACKWARDS.")


if __name__ == "__main__":
    main()
