"""M-adaptive robust estimator for per-mode target power gρ² (Track A1 of the roadmap).

The naive estimate gρ² = coeffρ²/λρ carries a per-mode noise floor E[gρ²] = true + σ²/M and blows up for
tiny λρ. It is biased-but-consistent: at large labeled M it converges to the truth (verified in
zeta_scale.py), but at HBN-scale M it fails. The fixed-threshold robust heuristic in zeta_robust.py
(subtract tail-median, hard-truncate λ<tol·λmax) does NOT scale — its correction is M-independent, so it
keeps discarding real signal as M grows.

This estimator is M-adaptive on both axes, each term ∝ σ²/M so it vanishes as M→∞ (→ naive, consistent)
and regularizes at small M:
  * ridge the division at the noise scale:   gρ² = coeffρ² / (λρ + σ²/M)   — controls the tiny-λ blowup
  * soft-threshold per-mode power:           gρ² = max(0, gρ² − k·σ²/M)    — removes the noise floor
`k` (≈2) sets how many noise-floor units a mode must clear to be kept (false-positive control).
"""
from __future__ import annotations

import numpy as np


def spectra_coeff(X, y):
    """Return (eigenvalues desc, per-eigenmode cross-covariance coeffρ) from full-data X, y."""
    Xc, yc = X - X.mean(0), y - y.mean()
    C = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(C)
    w, V = np.clip(w[::-1], 0.0, None), V[:, ::-1]
    coeff = V.T @ (Xc.T @ yc / len(Xc))
    return w, coeff


def naive_g2(w, coeff):
    w = np.clip(np.asarray(w, float), 0.0, None)
    return coeff ** 2 / (w + 1e-12 * (w.max() + 1e-30))


def noise_floor(w, coeff, tail_frac=0.5):
    """Empirical per-mode noise floor σ²/M (in gρ² units), from the noise/tail modes where true power≈0.

    For a noise mode E[coeffρ²] = (σ²/M)·λρ, so σ²/M is the origin-slope of coeffρ² vs λρ over the tail —
    a data-driven estimate that does NOT depend on the (biased, unsaturated) learning-curve ceiling."""
    w = np.clip(np.asarray(w, float), 0.0, None)
    k0 = int(len(w) * (1.0 - tail_frac))
    lam, c2 = w[k0:], np.asarray(coeff, float)[k0:] ** 2
    return float(np.sum(lam * c2) / (np.sum(lam ** 2) + 1e-30))


def madaptive_g2(w, coeff, floor, k=2.0):
    """M-adaptive per-mode target power. `floor` = empirical σ²/M noise floor (see noise_floor)."""
    w = np.clip(np.asarray(w, float), 0.0, None)
    floor = float(floor)
    g2 = coeff ** 2 / (w + floor)                          # noise-scale ridge → no tiny-λ blowup
    g2 = np.clip(g2 - k * floor, 0.0, None)               # soft-threshold at k noise floors
    return g2 * (w > 1e-8 * (w.max() + 1e-30))            # drop numerical null space (p>M artifacts)


def estimate_sigma2_from_ceiling(var_y, ceil):
    """Irreducible noise variance from the measured learning-curve ceiling: σ² = Var(y)·(1−R²_∞)."""
    return float(var_y) * max(0.0, 1.0 - float(ceil))
