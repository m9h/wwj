"""Does ENIGMA-scale data (10-100x HBN, not 10000x) close the CBP real-data estimation gap?

Isolates the labeled-sample-size (M) effect on estimating the per-mode target power gρ², which is what
broke the naive/robust CBP estimator on real HBN data (zeta_tierB.py, zeta_robust.py). For each M drawn
from the KNOWN synthetic generative model, estimate (λ, gρ²) from M (X,y) pairs (reusing the exact
spectra_raw/robust_g2 estimators used on real data), predict the CBP curve, and score against the
THEORETICAL curve computed from the TRUE spectrum (no curve-measurement noise -- isolates estimation
quality alone). M sweeps HBN-like (1e3) through realistic single-study-ENIGMA (1e4-1e5) to a deliberate
overshoot (3e5) to see if/where the naive and robust estimators converge to the true-spectrum prediction.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/zeta_scale.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeta_robust import robust_g2, cbp_curve, skill  # noqa: E402
from zeta_estimator import spectra_coeff, naive_g2, madaptive_g2, noise_floor  # noqa: E402

NS = [50, 100, 200, 400, 800]
MS = [300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
SEEDS = (0, 1, 2)


def run_one(a, b, snr, M, seed, P):
    rng = np.random.default_rng(seed)
    rho = np.arange(1, P + 1.0)
    lam_t = rho ** (-a)
    g2_t = rho ** (-b); g2_t = g2_t / g2_t.sum()
    ceil = snr / (1.0 + snr)                              # exact asymptotic R^2 ceiling for this SNR
    ridge = float(lam_t.sum())

    Z = rng.standard_normal((M, P)); X = Z * np.sqrt(lam_t)
    g = np.sqrt(g2_t) * np.sqrt(ceil)                      # scale true signal so Var(signal)=ceil
    y = Z @ g + np.sqrt(1 - ceil) * rng.standard_normal(M)

    w_est, coeff = spectra_coeff(X, y)
    g2_naive = naive_g2(w_est, coeff)
    g2_robust = robust_g2(w_est, g2_naive.copy())
    g2_mad = madaptive_g2(w_est, coeff, (1.0 - ceil) / M)   # known σ²/M (synthetic): the estimator on its own terms

    cbp_true = cbp_curve(lam_t, g2_t.copy() * ceil, ridge, NS, ceil)   # theory from TRUE spectrum
    cbp_naive = cbp_curve(w_est, g2_naive, ridge, NS, ceil)
    cbp_robust = cbp_curve(w_est, g2_robust, ridge, NS, ceil)
    cbp_mad = cbp_curve(w_est, g2_mad, ridge, NS, ceil)
    return skill(cbp_naive, cbp_true), skill(cbp_robust, cbp_true), skill(cbp_mad, cbp_true)


def main():
    scenarios = [(0.8, 0.5, 1.0), (1.5, 1.5, 4.0), (1.5, 0.5, 4.0)]
    P = 500                                            # realistic embedding dim (fomo60k=768, amaes=320, morph=456)
    print("Does more LABELED data (M) close the naive/robust CBP estimation gap vs the TRUE-spectrum curve?")
    print(f"P={P} (realistic embedding dim). HBN full-cohort M ~ 3e2-2.5e3 (p/M ~0.2-1.7 at these P);")
    print(f"single-study ENIGMA structural-MRI M ~ 1e4-1e5;  3e5 = deliberate overshoot.\n")
    for a, b, snr in scenarios:
        print(f"-- a={a} b={b} snr={snr} (ceil={snr/(1+snr):.2f}) --")
        print(f"{'M':>9s} {'p/M':>6s} | {'naive':>7s} {'robust':>7s} {'M-adapt':>8s}")
        for M in MS:
            sn, sr, sm = zip(*(run_one(a, b, snr, M, s, P) for s in SEEDS))
            print(f"{M:9d} {P/M:6.2f} | {np.mean(sn):+7.2f} {np.mean(sr):+7.2f} {np.mean(sm):+8.2f}", flush=True)
        print()
    print("If skill climbs toward +1.0 as M grows, the real-data CBP failure is a finite-sample estimation")
    print("artifact that ENIGMA-scale labeled cohorts would plausibly resolve -- NOT a flaw in the theory.")
    print("(Separately: the single-sorted-beta falsification used M=9000 already and is NOT rescued by this.)")


if __name__ == "__main__":
    main()
