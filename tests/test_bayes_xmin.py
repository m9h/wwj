# Phase 2: posterior over the scaling window xmin + Bayesian model averaging of
# alpha across it. The frequentist pipeline picks one xmin by argmin-KS and
# conditions on it; here every candidate xmin gets a comparable marginal-likelihood
# weight (Pareto-tail evidence vs a fixed exponential base measure), and the alpha
# posterior becomes a Gamma mixture weighted by p(xmin | data).
import numpy as np
import jax.numpy as jnp
import pytest

import wwj
from wwj.core import _eigvals, _ks_select_xmin


def _pareto_matrix(shape, alpha_target=3.0, n=1024, seed=0):
    rng = np.random.default_rng(seed)
    n_tail = max(1, n // 4)
    n_bulk = n - n_tail
    tail = (rng.pareto(alpha_target - 1, size=n_tail) + 1).astype(np.float32) * 0.1
    bulk = np.abs(rng.standard_normal(n_bulk).astype(np.float32)) * 0.01
    eigs = np.concatenate([tail, bulk])
    svals = np.sqrt(np.sort(eigs)[::-1][:min(shape)])
    U = np.linalg.qr(rng.standard_normal((shape[0], shape[0])).astype(np.float32))[0]
    V = np.linalg.qr(rng.standard_normal((shape[1], shape[1])).astype(np.float32))[0]
    return U[:, :len(svals)] @ np.diag(svals) @ V[:len(svals), :]


def test_xmin_posterior_is_valid_simplex():
    eigs = _eigvals(jnp.asarray(_pareto_matrix((512, 256), alpha_target=3.0)))
    out = wwj.alpha_posterior_bma(eigs, min_tail_size=20)
    w = np.asarray(out["xmin_weights"])
    assert np.all(w >= 0) and abs(w.sum() - 1.0) < 1e-5, f"weights not a simplex: sum={w.sum()}"
    for k in ("alpha_mean", "ci_low", "ci_high", "p_alpha_lt_2", "xmin_map", "n_eff"):
        assert k in out and np.isfinite(out[k]), f"bad key {k}"
    assert out["ci_low"] <= out["alpha_mean"] <= out["ci_high"]
    # MAP xmin is the argmax-weight candidate
    cand = np.asarray(out["xmin_candidates"])
    assert out["xmin_map"] == pytest.approx(float(cand[np.argmax(w)]), rel=1e-5)
    # effective number of windows is in [1, #candidates]
    assert 1.0 <= out["n_eff"] <= (w > 0).sum() + 1e-6


def test_bma_collapses_to_fixed_xmin_when_evidence_concentrates():
    """When one window dominates, BMA alpha should track the fixed-xmin posterior
    evaluated at the MAP window."""
    eigs = _eigvals(jnp.asarray(_pareto_matrix((1024, 256), alpha_target=3.0, n=1024)))
    bma = wwj.alpha_posterior_bma(eigs, min_tail_size=20)
    fixed = wwj.alpha_posterior(eigs, xmin=jnp.asarray(bma["xmin_map"]))
    # if the MAP window carries most of the mass, the means should be close
    if np.max(bma["xmin_weights"]) > 0.5:
        assert abs(bma["alpha_mean"] - fixed["alpha_mean"]) < 0.3, \
            f"BMA {bma['alpha_mean']:.3f} vs fixed-at-MAP {fixed['alpha_mean']:.3f}"


def test_bma_variance_decomposition_obeys_total_variance_law():
    """The mixture variance must equal within-window + across-window variance, and
    the across-window term (the cost of not knowing xmin) must be non-negative.
    This is the law of total variance -- the honest statement of what BMA adds,
    rather than the (false) claim that the mixture is wider than any single window."""
    eigs = _eigvals(jnp.asarray(_pareto_matrix((512, 256), alpha_target=3.0)))
    bma = wwj.alpha_posterior_bma(eigs, min_tail_size=20)
    total = bma["alpha_std"] ** 2
    within = bma["within_window_var"]
    across = bma["across_window_var"]
    assert across >= -1e-9, f"across-window variance negative: {across}"
    assert total == pytest.approx(within + across, rel=1e-5, abs=1e-9)
    # total uncertainty is at least the within-window part (across >= 0)
    assert total >= within - 1e-9
