"""Data-side spectral diagnostics for the (Bayesian) Zeta Law — extending Thompson (arXiv:2604.17581).

Thompson predicts data sufficiency from the data covariance spectrum (rank decay `s`/`γ`) and the
signal-alignment spectrum (rank decay `β`), with the `ζ`-pole at `β = 1` marking the regime where
more data stops helping. This module reuses wwj's weight-spectrum machinery on the *data* side:

  - wwj fits the eigenvalue DENSITY tail  p(λ) ∝ λ^{−α}.
  - Thompson uses the RANK decay        λ_n ∝ n^{−s}.
  - they convert by                      s = 1 / (α − 1).

So `γ = 1/(α_cov − 1)` from the covariance ESD and `β = 1/(α_align − 1)` from the alignment spectrum,
and the data-sufficient regime `β > 1` is exactly `α_align < 2` — i.e. **`P(β>1) = p_alpha_lt_2`**
(wwjd's existing criticality probability), evaluated on the alignment spectrum. The same `α=2` HT-SR
critical point, now as Thompson's sufficiency threshold.

Run the synthetic validation:
    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/zeta_diagnostic.py
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)        # CSN/MLE precision; matches wwj's exact pipeline

import numpy as np
import jax.numpy as jnp

from wwj import _eigvals, fit_distributions, alpha_posterior, model_posterior, ppc_pvalue


# ---------------------------------------------------------------------------- conversions
def rank_decay_from_density_alpha(alpha: float) -> float:
    """Thompson rank-decay exponent s from wwj's density-tail α:  s = 1/(α−1)."""
    return float("inf") if alpha <= 1.0 else 1.0 / (alpha - 1.0)


# ---------------------------------------------------------------------------- spectra
def covariance_spectrum(X: np.ndarray) -> np.ndarray:
    """Eigenvalues (descending) of the feature covariance of X (n_samples × n_features)."""
    return np.asarray(_eigvals(jnp.asarray(X, dtype=jnp.float64)))


def source_spectrum(X: np.ndarray, y: np.ndarray, ridge_frac: float = 1e-3):
    """Position-preserving Canatar–Bordelon–Pehlevan **source** spectrum: the covariance-deconfounded
    per-mode target power `aᵢ = coeffᵢ² / λᵢ`, kept in **eigenvalue-rank order** (NOT magnitude-sorted).

    This is the raw material for both data-side predictors, and the sort is the fork between them:
      * keep it in eigenrank order  → the source spectrum, fed to `source_rank` (ρ_q).  Keeps *which*
        eigenmode carries the signal.
      * sort it by magnitude        → Thompson's β pipeline (`alignment_spectrum`).  Discards position.
    Eigenmode position is exactly what sets data sufficiency (high-λ modes are learnable with little
    data, low-λ modes are data-hungry), so the sort is where the single-β sufficiency claim loses its
    information (falsified: synthetic Spearman ≈0.25, real ≈−0.02; ρ_q recovers ≈0.96 / ≈0.53).

    Deconfounding note: using the raw cross-covariance `coeffᵢ²` instead conflates the target with the
    covariance decay (every target then looks concentrated → spurious β>1, as the first HBN run showed);
    dividing by the eigenvalue isolates the target's spectral content. The division is ridge-regularized
    (`+ ridge_frac·λ_max`) so near-zero (noise) eigenvalues do not blow up.

    Returns `(a, lam)`: the source power `a` in descending-eigenvalue order, and the eigenvalues `lam`
    (also descending). `lam.sum() = trace(C)` is the calibrated ridge scale for the matched CV probe."""
    Xc = X - X.mean(0, keepdims=True)
    yc = np.asarray(y, dtype=float) - float(np.mean(y))
    C = (Xc.T @ Xc) / Xc.shape[0]
    w, V = np.linalg.eigh(C)                     # ascending
    w, V = w[::-1], V[:, ::-1]                   # descending eigenvalue order
    cross = (Xc.T @ yc) / Xc.shape[0]            # cross-covariance vector
    coeff = V.T @ cross                          # target coefficient per eigenmode
    lam = np.clip(w, 0.0, None)
    a = coeff ** 2 / (lam + ridge_frac * float(lam.max()))   # per-mode source power, eigenrank order
    return a, lam


def source_rank(a: np.ndarray, q: float = 0.7) -> int:
    """ρ_q — the position-preserving CBP **source predictor** of data sufficiency: the number of top
    eigenmodes (in eigenvalue-rank order; pass the `a` from `source_spectrum`, NOT a sorted spectrum)
    whose cumulative source mass first reaches fraction `q` of the total.

    Small ρ_q = signal concentrated in a few learnable high-λ modes = data-cheap; large ρ_q = signal
    spread into low-λ (data-hungry) modes. Forecasts N-to-reach-target at Spearman ≈0.96 (synthetic) /
    ≈0.53 (real HBN), where Thompson's single magnitude-sorted β does not (≈0.25 / ≈−0.02) — the named
    replacement for the sorted exponent as a sufficiency predictor.

    GOODHART CAVEAT: like α (weights) and β (data), ρ_q must be reported jointly with held-out
    separability — compressing a representation into fewer modes shrinks ρ_q without discovering any
    new learnable signal, so the exponent must never be read alone."""
    a = np.asarray(a, dtype=float)
    return int(np.searchsorted(np.cumsum(a) / a.sum(), q) + 1)


def alignment_spectrum(X: np.ndarray, y: np.ndarray, ridge_frac: float = 1e-3) -> np.ndarray:
    """Thompson's β material: the `source_spectrum` per-mode power sorted **descending by magnitude**
    (position discarded). Feed to the density-α → β pipeline (`diagnose_spectrum(..., 'alignment')`).
    For the position-preserving predictor use `source_spectrum` + `source_rank` instead."""
    a, _ = source_spectrum(X, y, ridge_frac)
    return np.sort(a)[::-1]                      # aligned energy, descending


# ---------------------------------------------------------------------------- diagnose one spectrum
def diagnose_spectrum(eigs: np.ndarray, kind: str) -> dict:
    """Run wwj's power-law fit + Bayesian posterior + model comparison on a spectrum and report the
    Thompson exponent. `kind` ∈ {'covariance','alignment'} sets the label (γ vs β)."""
    e = jnp.asarray(np.asarray(eigs, dtype=float))
    ap = alpha_posterior(e)
    mp = model_posterior(e)
    ppc = ppc_pvalue(e)
    alpha = ap["alpha_mean"]
    exponent = rank_decay_from_density_alpha(alpha)
    # Validity gate: power-law must beat BOTH alternatives AND the posterior-predictive KS check must
    # not reject it. (pl-vs-lognormal alone is unreliable — the textbook hard case, Clauset et al.;
    # the pl-vs-exponential factor + PPC are what catch Marchenko–Pastur / light-tailed spectra.)
    is_power_law = (mp["logbf_pl_vs_ln"] > 0) and (mp["logbf_pl_vs_exp"] > 0) and (ppc["p_value"] >= 0.05)
    out = {
        "kind": kind,
        "exponent_name": "gamma" if kind == "covariance" else "beta",
        "alpha_density": alpha,
        "alpha_ci": (ap["ci_low"], ap["ci_high"]),
        "rank_decay": exponent,                  # γ or β = 1/(α−1)
        "is_power_law": bool(is_power_law),
        "logbf_pl_vs_lognormal": mp["logbf_pl_vs_ln"],
        "logbf_pl_vs_exponential": mp["logbf_pl_vs_exp"],
        "ppc_pvalue": ppc["p_value"],
        "tail_size": ap["tail_size"],
    }
    if kind == "alignment":
        # β > 1  ⇔  α_align < 2  ⇒  P(β>1) = P(α<2) = p_alpha_lt_2
        out["p_beta_gt_1"] = ap["p_alpha_lt_2"]
        out["regime"] = ("variance-limited (collect more data)" if exponent > 1
                         else "resolution-limited (improve representation)")
    return out


# ---------------------------------------------------------------------------- synthetic validation
def _planted_powerlaw_eigs(s: float, M: int = 600) -> np.ndarray:
    """Eigenvalues with exact rank decay λ_n = n^{−s} (so density-α should be 1 + 1/s)."""
    n = np.arange(1, M + 1, dtype=float)
    return n ** (-s)


def _planted_mp_eigs(n_rows: int = 1200, n_cols: int = 600, seed: int = 0) -> np.ndarray:
    """Marchenko–Pastur: eigenvalues of a random Gaussian matrix — the HT-SR 'random / no structure'
    null. Bounded support + sharp edge, so NOT power-law; the validity gate must reject it."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_rows, n_cols))
    return np.asarray(_eigvals(jnp.asarray(W)))


def main() -> None:
    print("Zeta-Law data-side diagnostic — synthetic validation\n" + "-" * 56)
    ok = True

    # (1) recover a planted covariance rank-decay s
    for s_true in (0.7, 1.5):
        d = diagnose_spectrum(_planted_powerlaw_eigs(s_true), "covariance")
        rec = d["rank_decay"]
        good = abs(rec - s_true) <= 0.2 and d["is_power_law"]
        ok &= good
        print(f"  covariance  s_true={s_true:.2f}  recovered γ={rec:.2f}  "
              f"power-law={d['is_power_law']}  -> {'OK' if good else 'CHECK'}")

    # (2) recover a planted alignment β and the regime / P(β>1)
    for b_true, want in ((1.6, "variance-limited"), (0.6, "resolution-limited")):
        d = diagnose_spectrum(_planted_powerlaw_eigs(b_true), "alignment")
        rec, p = d["rank_decay"], d["p_beta_gt_1"]
        good = abs(rec - b_true) <= 0.2 and d["regime"].startswith(want) \
            and ((p > 0.5) == (b_true > 1))
        ok &= good
        print(f"  alignment   β_true={b_true:.2f}  recovered β={rec:.2f}  "
              f"P(β>1)={p:.2f}  regime={d['regime'].split(' (')[0]}  -> {'OK' if good else 'CHECK'}")

    # (3) validity gate must REJECT a Marchenko–Pastur (random / no-structure) spectrum
    dmp = diagnose_spectrum(_planted_mp_eigs(), "covariance")
    good = not dmp["is_power_law"]
    ok &= good
    print(f"  validity    Marchenko–Pastur (random)  power-law={dmp['is_power_law']}  "
          f"(logBF pl/exp={dmp['logbf_pl_vs_exponential']:+.1f}, ppc p={dmp['ppc_pvalue']:.2f})  "
          f"-> {'OK' if good else 'CHECK'}")

    # (4) source_rank ρ_q is monotone in eigenmode position (deeper signal ⇒ larger ρ_q)
    rng = np.random.default_rng(0)
    P, NB = 120, 4000
    lam = np.arange(1, P + 1.0) ** -1.0
    Z = rng.standard_normal((NB, P)); Xs = Z * np.sqrt(lam)
    rqs = []
    for k in (0, 19, 89):                        # signal planted in mode 1, 20, 90
        ys = Z[:, k] + 0.5 * rng.standard_normal(NB)
        a, _ = source_spectrum(Xs, ys)
        rqs.append(source_rank(a))
    mono = rqs[0] <= rqs[1] <= rqs[2]
    ok &= mono
    print(f"  source_rank ρ_0.7 vs planted depth (mode 1/20/90): {rqs}  monotone -> {'OK' if mono else 'CHECK'}")

    print("-" * 56)
    print(f"VERDICT: {'data-side zeta diagnostic recovers planted s/β, ρ_q tracks position, rejects MP null — OK'
                      if ok else 'CHECK — a synthetic case did not pass'}")


if __name__ == "__main__":
    main()
