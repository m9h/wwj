"""E4 — the cross-modal spectrum (EEG↔structural shared subspace), for the zeta-law / structure–function
analysis.

The relationship between two modalities is the singular spectrum of their *whitened* cross-covariance —
the canonical correlations ρ₁≥ρ₂≥… ∈ [0,1]. Strong ρ = a shared mode; the count/decay of strong ρ is
the shared-subspace dimension (structure–function coupling). For the EEG↔sMRI case the top modes are
dominated by age-related anatomy → volume conduction; pass `covariate=age` to residualize it out and
see coupling *beyond* conduction (the cognitively interesting residual).

Pure numpy. Consumes any two subject-aligned feature matrices (e.g. REVE EEG embeddings ⊕ the tier-2
structural embedding).
"""
from __future__ import annotations

import numpy as np


def _residualize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Regress out covariate(s) C (with intercept) from each column of X."""
    C = np.asarray(C, float)
    if C.ndim == 1:
        C = C[:, None]
    Cb = np.column_stack([np.ones(len(C)), C])
    return X - Cb @ (np.linalg.pinv(Cb) @ X)


def _whiten(X: np.ndarray, reg: float) -> np.ndarray:
    """Center then map to identity covariance via a ridge-regularised inverse square root."""
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / len(Xc)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(w + reg * float(w.max()))) @ V.T
    return Xc @ inv_sqrt


def cross_modal_spectrum(Xa: np.ndarray, Xb: np.ndarray, reg: float = 1e-3,
                         covariate: np.ndarray | None = None) -> np.ndarray:
    """Canonical correlations between modalities Xa, Xb (descending). If `covariate` is given (e.g.
    age), residualize it out of both first — removing the conduction/trivial shared term."""
    Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float)
    if covariate is not None:
        Xa, Xb = _residualize(Xa, covariate), _residualize(Xb, covariate)
    M = (_whiten(Xa, reg).T @ _whiten(Xb, reg)) / len(Xa)        # whitened cross-covariance
    s = np.linalg.svd(M, compute_uv=False)
    return np.clip(np.sort(s)[::-1], 0.0, 1.0)


def shared_subspace_summary(rho: np.ndarray, thresh: float = 0.5) -> dict:
    """Summarise the cross-modal spectrum: # strong modes, effective dimension (participation ratio),
    top and mean canonical correlation."""
    rho = np.asarray(rho, float)
    pr = (rho.sum() ** 2) / (np.sum(rho ** 2) + 1e-30) if rho.size else 0.0
    return {
        "n_strong": int(np.sum(rho > thresh)),
        "participation_ratio": float(pr),
        "top": float(rho[0]) if rho.size else float("nan"),
        "mean": float(rho.mean()) if rho.size else float("nan"),
    }


def main() -> None:
    """Synthetic demo: a 3-mode shared subspace, with mode 0 age-driven → residualizing age drops it."""
    rng = np.random.default_rng(0)
    n = 800
    age = rng.standard_normal(n)
    Z = np.column_stack([age, rng.standard_normal(n), rng.standard_normal(n)])
    Wa, Wb = rng.standard_normal((3, 30)), rng.standard_normal((3, 25))
    s = np.array([5.0, 4.0, 4.0])[:, None]
    Xa = (Z * s.T) @ Wa + rng.standard_normal((n, 30))
    Xb = (Z * s.T) @ Wb + rng.standard_normal((n, 25))
    full = cross_modal_spectrum(Xa, Xb)
    resid = cross_modal_spectrum(Xa, Xb, covariate=age)
    print("cross-modal spectrum (top 6):")
    print(f"  full           ρ = {np.round(full[:6], 2)}  -> {shared_subspace_summary(full)}")
    print(f"  age-residualized ρ = {np.round(resid[:6], 2)}  -> {shared_subspace_summary(resid)}")
    print("  (age-driven shared mode drops out under residualization — coupling beyond conduction)")


if __name__ == "__main__":
    main()
