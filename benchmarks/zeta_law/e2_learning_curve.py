"""E2 — the Bayesian zeta-law learning curve + calibration, wired to HBN.

From the covariance decay γ and the alignment decay β (E1), the captured signal at sample size N is a
partial-ζ sum over the modes resolvable at N:

    K(N) ∝ N^{1/(2(γ+1))}            (resolvable modes; Thompson)
    S(N) = Σ_{n≤K(N)} n^{-β}         (captured aligned signal; → ζ(β) iff β>1)

so performance saturates iff β>1 and grows without bound iff β≤1 (the ζ-pole, data-insufficient
regime). `learning_curve_posterior` propagates the wwjd Gamma posteriors over (γ, β) to a credible band
over S(N) and to P(no saturation)=P(β≤1). `observed_curve` measures the real curve (ridge-CV) by
subsampling, and `calibration_coverage` checks band coverage.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/e2_learning_curve.py   # HBN if present, else demo
    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/test_e2.py
"""
from __future__ import annotations

import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from wwj import alpha_posterior

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeta_diagnostic import covariance_spectrum, alignment_spectrum   # noqa: E402

_KCAP = 1_000_000


def _zeta(beta: float) -> float:
    try:
        from scipy.special import zeta
        return float(zeta(beta))
    except Exception:
        return float(np.sum(np.arange(1, 1_000_001.0) ** (-beta)))


def _partial_harmonic(beta: float, K: np.ndarray) -> np.ndarray:
    """Σ_{n=1}^{round(K)} n^{−β} for each K (vectorised via one cumulative sum)."""
    K = np.asarray(K, dtype=float)
    Kmax = int(min(_KCAP, max(1, np.ceil(K.max()))))
    csum = np.cumsum(np.arange(1, Kmax + 1.0) ** (-beta))
    idx = np.clip(np.round(K).astype(int), 1, Kmax) - 1
    return csum[idx]


def predicted_learning_curve(gamma: float, beta: float, N_grid: np.ndarray,
                             normalize: bool = True) -> np.ndarray:
    """Captured aligned signal S(N). If normalize and β>1, divide by ζ(β) → captured fraction ∈ (0,1]."""
    res = 1.0 / (2.0 * (gamma + 1.0))
    K = np.clip(np.asarray(N_grid, float) ** res, 1.0, _KCAP)
    S = _partial_harmonic(beta, K)
    if normalize and beta > 1.0:
        S = S / _zeta(beta)
    return S


def n_star(gamma: float, beta: float, target: float = 0.9, n_max: float = 1e12) -> float:
    """Smallest N reaching `target` of the asymptotic performance. ∞ if β≤1 (never saturates)."""
    if beta <= 1.0:
        return float("inf")
    Ns = np.logspace(2, np.log10(n_max), 400)
    frac = predicted_learning_curve(gamma, beta, Ns, normalize=True)
    hit = np.where(frac >= target)[0]
    return float(Ns[hit[0]]) if len(hit) else float("inf")


def _exponent_draws(eigs, n_draws: int, rng) -> np.ndarray:
    """Posterior draws of the Thompson exponent q = 1/(α−1) from the wwjd Gamma posterior on β=α−1."""
    ap = alpha_posterior(jnp.asarray(np.asarray(eigs, float)))
    beta_wwjd = rng.gamma(shape=ap["post_a"], scale=1.0 / ap["post_b"], size=n_draws)
    return 1.0 / np.clip(beta_wwjd, 1e-9, None)


def learning_curve_posterior(cov_eigs, align_eigs, N_grid, n_draws: int = 400,
                             ci: float = 0.9, seed: int = 0) -> dict:
    """Credible band over S(N) by propagating the (γ, β) posteriors; plus P(β≤1) = P(no saturation)."""
    rng = np.random.default_rng(seed)
    gammas = _exponent_draws(cov_eigs, n_draws, rng)
    betas = _exponent_draws(align_eigs, n_draws, rng)
    curves = np.array([predicted_learning_curve(g, b, N_grid, normalize=False)
                       for g, b in zip(gammas, betas)])
    lo, hi = (1 - ci) / 2, (1 + ci) / 2
    return {
        "N_grid": np.asarray(N_grid, float),
        "median": np.median(curves, axis=0),
        "lo": np.quantile(curves, lo, axis=0),
        "hi": np.quantile(curves, hi, axis=0),
        "p_no_saturation": float(np.mean(betas <= 1.0)),     # P(β≤1) = resolution-limited
        "beta_median": float(np.median(betas)),
        "gamma_median": float(np.median(gammas)),
    }


def calibration_coverage(band: dict, observed: np.ndarray) -> float:
    """Fraction of finite observations falling inside the band (same-scale; see `_rescale` for real data)."""
    o = np.asarray(observed, float)
    m = np.isfinite(o)
    if not m.any():
        return float("nan")
    inside = (o[m] >= band["lo"][m]) & (o[m] <= band["hi"][m])
    return float(np.mean(inside))


# ---------------------------------------------------------------------------- observed curve (ridge CV)
def _ridge_cv_r(X: np.ndarray, y: np.ndarray, alpha: float = 10.0, k: int = 5, seed: int = 0) -> float:
    """k-fold ridge-CV Pearson r of out-of-fold predictions — pure numpy (no sklearn), standardised
    per fold, ridge solved in closed form (matches phase_probe's StandardScaler+Ridge(alpha=10))."""
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    yp = np.empty(n)
    for fold in np.array_split(idx, k):
        tr = np.setdiff1d(idx, fold, assume_unique=False)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[fold] - mu) / sd
        yb = y[tr].mean()
        A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ (y[tr] - yb))
        yp[fold] = Xte @ w + yb
    return float(np.corrcoef(y, yp)[0, 1])


def observed_curve(X: np.ndarray, y: np.ndarray, N_grid, seed: int = 0) -> np.ndarray:
    """Empirical performance (5-fold ridge-CV Pearson r) at each subsample size; NaN where N > n."""
    m = np.isfinite(np.asarray(y, float))
    X, y = np.asarray(X)[m], np.asarray(y, float)[m]
    rng = np.random.default_rng(seed)
    out = []
    for n in np.asarray(N_grid, int):
        if n > len(y):
            out.append(np.nan); continue
        sub = rng.choice(len(y), size=int(n), replace=False)
        out.append(_ridge_cv_r(X[sub], y[sub], seed=seed))
    return np.array(out)


def _rescale(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float)
    lo, hi = np.nanmin(c), np.nanmax(c)
    return (c - lo) / (hi - lo) if hi > lo else np.zeros_like(c)


# ---------------------------------------------------------------------------- main
def main() -> None:
    from hbn_e1 import load_hbn, EMB_DIR
    if not (EMB_DIR.exists() and any(EMB_DIR.glob("*.npz"))):
        print("HBN cache not found — synthetic E2 demo:\n")
        cov, al = np.arange(1, 401.0) ** -1.0, np.arange(1, 401.0) ** -1.3
        N = np.array([250.0, 500.0, 1000.0, 2000.0, 4000.0])
        band = learning_curve_posterior(cov, al, N)
        print(f"  β_med={band['beta_median']:.2f}  γ_med={band['gamma_median']:.2f}  "
              f"P(no-sat)={band['p_no_saturation']:.2f}  n*(0.9)={n_star(band['gamma_median'], band['beta_median']):.0f}")
        return
    feats, tgts = load_hbn()
    model = next(iter(feats))                      # one modality for a focused run
    X = feats[model]
    n = X.shape[0]
    N = np.unique(np.clip(np.array([250, 500, 1000, n]), 1, n)).astype(float)
    print(f"E2 on HBN modality '{model}' (n={n})  N grid={N.astype(int).tolist()}\n")
    print(f"  {'target':14s} {'β_med':>6s} {'P(no-sat)':>9s} {'n*(0.9)':>9s} {'coverage':>9s} {'r@max':>6s}")
    cov = covariance_spectrum(X)
    for t, y in tgts[model].items():
        al = alignment_spectrum(X[np.isfinite(y)], y[np.isfinite(y)])
        band = learning_curve_posterior(cov, al, N)
        ns = n_star(band["gamma_median"], band["beta_median"])
        ns_s = "inf" if not np.isfinite(ns) else f"{ns:.0f}"
        obs = observed_curve(X, y, N)                    # pure-numpy ridge CV (no sklearn)
        rel = {"lo": _rescale(band["lo"]), "hi": _rescale(band["hi"]), "median": _rescale(band["median"])}
        cov_frac = calibration_coverage(rel, _rescale(obs))
        print(f"  {t:14s} {band['beta_median']:6.2f} {band['p_no_saturation']:9.2f} {ns_s:>9s} "
              f"{cov_frac:9.2f} {np.nanmax(obs):6.2f}")


if __name__ == "__main__":
    main()
