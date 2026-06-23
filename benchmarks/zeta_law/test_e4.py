"""Red-green TDD for E4 — the cross-modal spectrum (EEG↔structural shared subspace).

The "relationship" between two modalities is the singular spectrum of their whitened cross-covariance,
i.e. the canonical correlations ρ₁≥ρ₂≥… ∈ [0,1] — how strongly the modalities co-vary, mode by mode.
The number/decay of strong ρ is the shared-subspace dimension (structure–function coupling). Per the
volume-conduction insight, the top shared modes are dominated by age-related anatomy/conduction; passing
`covariate=age` residualizes it out, revealing coupling *beyond* conduction.

    <numpy-python> benchmarks/zeta_law/test_e4.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e4_cross_modal import cross_modal_spectrum, shared_subspace_summary   # noqa: E402 (red until module)


def _shared(n, da, db, strengths, seed, extra_latent=None):
    """Two views sharing latent factors Z (one per `strengths` entry); strong strength → ρ≈1."""
    rng = np.random.default_rng(seed)
    k = len(strengths)
    Z = rng.standard_normal((n, k)) if extra_latent is None else extra_latent
    Wa, Wb = rng.standard_normal((k, da)), rng.standard_normal((k, db))
    s = np.asarray(strengths)[:, None]
    Xa = (Z * s.T) @ Wa + rng.standard_normal((n, da))
    Xb = (Z * s.T) @ Wb + rng.standard_normal((n, db))
    return Xa, Xb, Z


def test_recovers_shared_dimension():
    Xa, Xb, _ = _shared(800, 30, 25, strengths=[4, 4, 4, 0, 0, 0], seed=1)   # 3 strong shared dims
    rho = cross_modal_spectrum(Xa, Xb)
    assert rho[0] >= rho[1] >= rho[2] and np.all((rho >= -1e-6) & (rho <= 1 + 1e-6))
    assert (rho[:3] > 0.7).all() and rho[3] < 0.5                            # exactly 3 strong modes
    assert shared_subspace_summary(rho, thresh=0.5)["n_strong"] == 3


def test_no_shared_subspace_is_low():
    rng = np.random.default_rng(2)
    Xa, Xb = rng.standard_normal((1000, 25)), rng.standard_normal((1000, 25))  # independent
    rho = cross_modal_spectrum(Xa, Xb)
    assert rho[0] < 0.45                                                       # no genuine coupling


def test_residualizing_covariate_removes_the_shared_age_mode():
    rng = np.random.default_rng(3)
    age = rng.standard_normal(800)
    # mode 0 is driven by age (the conduction/trivial term); modes 1-2 are age-independent coupling
    Z = np.column_stack([age, rng.standard_normal(800), rng.standard_normal(800)])
    Xa, Xb, _ = _shared(800, 30, 25, strengths=[5, 4, 4], seed=4, extra_latent=Z)
    rho_full = cross_modal_spectrum(Xa, Xb)
    rho_resid = cross_modal_spectrum(Xa, Xb, covariate=age)
    assert shared_subspace_summary(rho_full, 0.5)["n_strong"] == 3
    assert shared_subspace_summary(rho_resid, 0.5)["n_strong"] == 2          # age-mode removed
    assert rho_resid[0] < rho_full[0]


if __name__ == "__main__":
    for _fn in (test_recovers_shared_dimension, test_no_shared_subspace_is_low,
                test_residualizing_covariate_removes_the_shared_age_mode):
        _fn(); print(f"PASS  {_fn.__name__}")
    print("all E4 tests passed")
