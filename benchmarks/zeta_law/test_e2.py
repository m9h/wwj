"""Red-green TDD for E2 — the Bayesian zeta-law learning curve + calibration.

E2 turns the spectral exponents into a *prediction*: from the covariance decay γ and the alignment
decay β, the captured signal accumulates as a partial-ζ sum over the modes resolvable at sample size N,
so performance(N) saturates iff β>1 (ζ converges) and grows without bound iff β≤1 (the ζ-pole / data-
insufficient regime). The Bayesian layer propagates the wwjd posteriors over (γ, β) to a credible band
over the curve and to P(no saturation)=P(β≤1).

Run:  /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/test_e2.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2_learning_curve import (   # noqa: E402  (red until e2_learning_curve.py exists)
    predicted_learning_curve, n_star, learning_curve_posterior, calibration_coverage,
)

N = np.logspace(2.4, 5.5, 6)            # ~log-uniform grid (≈250 … 3e5) so increments are comparable


def _planted(rank_decay: float, M: int = 400) -> np.ndarray:
    """Spectrum with rank decay `q` (λ_n = n^{-q}); fed to alpha_posterior it recovers Thompson's
    exponent ≈ q (density α = 1 + 1/q → 1/(α−1) = q)."""
    return np.arange(1, M + 1.0) ** (-rank_decay)


def test_curve_monotone_and_saturation():
    c_sat = predicted_learning_curve(gamma=1.0, beta=1.6, N_grid=N, normalize=True)
    c_uns = predicted_learning_curve(gamma=1.0, beta=0.6, N_grid=N, normalize=False)
    assert np.all(np.diff(c_sat) >= -1e-9) and np.all(np.diff(c_uns) >= -1e-9)     # monotone in N
    assert c_sat[-1] <= 1.0 + 1e-6                                                  # captured fraction ≤ 1
    # on a log-uniform grid: β>1 is concave (increments DECREASE → saturation)…
    assert (c_sat[-1] - c_sat[-2]) < (c_sat[1] - c_sat[0])
    # …while β≤1 is convex (increments INCREASE → no saturation)
    assert (c_uns[-1] - c_uns[-2]) > (c_uns[1] - c_uns[0])


def test_n_star_finite_vs_infinite():
    assert np.isfinite(n_star(gamma=1.0, beta=1.6, target=0.9))
    assert n_star(gamma=1.0, beta=0.6, target=0.9) == float("inf")    # resolution-limited: never reaches


def test_posterior_band_and_p_no_saturation():
    cov = _planted(1.0)
    b_sat = learning_curve_posterior(cov, _planted(1.6), N, seed=0)   # β≈1.6 → saturates
    b_uns = learning_curve_posterior(cov, _planted(0.6), N, seed=0)   # β≈0.6 → no saturation
    assert b_sat["p_no_saturation"] < 0.2
    assert b_uns["p_no_saturation"] > 0.8
    assert np.all(b_sat["lo"] <= b_sat["median"] + 1e-9)
    assert np.all(b_sat["median"] <= b_sat["hi"] + 1e-9)


def test_calibration_band_covers_synthetic_observations():
    band = learning_curve_posterior(_planted(1.0), _planted(1.4), N, ci=0.9, seed=1)
    rng = np.random.default_rng(0)
    obs = band["median"] * (1.0 + 0.02 * rng.standard_normal(len(N)))   # observations from the model
    assert calibration_coverage(band, obs) >= 0.7


if __name__ == "__main__":
    for _fn in (test_curve_monotone_and_saturation, test_n_star_finite_vs_infinite,
                test_posterior_band_and_p_no_saturation,
                test_calibration_band_covers_synthetic_observations):
        _fn(); print(f"PASS  {_fn.__name__}")
    print("all E2 tests passed")
