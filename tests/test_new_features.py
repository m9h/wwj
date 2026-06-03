# Tests for JAX-exact features that WeightWatcher approximates for compute reasons:
# bootstrap CI, and multi-distribution fit + LRT.
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import wwj
from wwj.core import _eigvals


def _trained_like_matrix(shape, alpha_target=3.0, n=2000, seed=0):
    """Generate a weight matrix whose W^T W spectrum is approximately power-law
    with exponent alpha_target. Used to test fits on data that ACTUALLY has a
    power-law tail (random Gaussian doesn't)."""
    rng = np.random.default_rng(seed)
    # Sample eigenvalues from a Pareto(alpha-1) distribution + bulk
    n_tail = max(1, n // 4)
    n_bulk = n - n_tail
    tail = (rng.pareto(alpha_target - 1, size=n_tail) + 1).astype(np.float32) * 0.1
    bulk = np.abs(rng.standard_normal(n_bulk).astype(np.float32)) * 0.01
    eigs = np.concatenate([tail, bulk])
    # Construct matrix with these singular values
    svals = np.sqrt(np.sort(eigs)[::-1][:min(shape)])
    U = np.linalg.qr(rng.standard_normal((shape[0], shape[0])).astype(np.float32))[0]
    V = np.linalg.qr(rng.standard_normal((shape[1], shape[1])).astype(np.float32))[0]
    return U[:, :len(svals)] @ np.diag(svals) @ V[:len(svals), :]


def test_bootstrap_alpha_ci_runs_and_is_finite():
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((512, 256)).astype(np.float32) / np.sqrt(256))
    eigs = _eigvals(W)
    result = wwj.bootstrap_alpha_ci(eigs, n_bootstrap=200, ci=0.95)
    assert "alpha" in result and "ci_low" in result and "ci_high" in result
    assert result["ci_low"] <= result["alpha"] <= result["ci_high"]
    assert np.isfinite(result["alpha_std"])
    assert result["alpha_std"] > 0  # there must be bootstrap variation


def test_bootstrap_ci_tightens_with_more_samples():
    """More bootstrap iterations should not dramatically widen the CI."""
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((1024, 256)).astype(np.float32) / np.sqrt(256))
    eigs = _eigvals(W)
    r200 = wwj.bootstrap_alpha_ci(eigs, n_bootstrap=200, ci=0.95)
    r2000 = wwj.bootstrap_alpha_ci(eigs, n_bootstrap=2000, ci=0.95)
    # CI width shouldn't blow up (it should stabilize)
    w200 = r200["ci_high"] - r200["ci_low"]
    w2000 = r2000["ci_high"] - r2000["ci_low"]
    assert w2000 < 2.0 * w200, f"CI inflated: {w200:.3f} -> {w2000:.3f}"


def test_fit_distributions_returns_all_three_loglikelihoods():
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((1024, 256)).astype(np.float32) / np.sqrt(256))
    eigs = _eigvals(W)
    fit = wwj.fit_distributions(eigs)
    for k in ("alpha", "xmin", "pl_loglik", "exp_loglik", "ln_loglik",
              "pl_vs_exp_lrt", "pl_vs_ln_lrt"):
        assert k in fit
        assert np.isfinite(fit[k]), f"non-finite {k}"


def test_lrt_favors_power_law_on_pareto_data():
    """A genuine power-law-ish spectrum should give positive pl_vs_exp_lrt."""
    W = jnp.asarray(_trained_like_matrix((1024, 256), alpha_target=3.0, n=1024))
    eigs = _eigvals(W)
    fit = wwj.fit_distributions(eigs)
    # On power-law-like data, the LRT should favor PL over exponential
    assert fit["pl_vs_exp_lrt"] > 0, \
        f"PL not favored on power-law data: lrt={fit['pl_vs_exp_lrt']:.3f}"
