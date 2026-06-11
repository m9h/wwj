# Degeneracy guard for the Bayesian (wwjd) path.
#
# Bug: bayes_analyze_matrix lacked the `_n_meaningful(eigs) < min_tail_size` guard
# that the frequentist bootstrap_alpha_ci / fit_distributions already apply, so it
# fit a SPURIOUS finite alpha to a near-dead / low-rank matrix (no fittable heavy
# tail), and bayes_summary then averaged that garbage in. Real trained models DO
# contain such matrices (e.g. POYO zeros out ~96/106 projections while decoding),
# so this matters in practice. A degenerate matrix must be flagged + excluded from
# the aggregate, mirroring core's nan/degenerate convention.
import math

import numpy as np
import jax.numpy as jnp
import pytest

import wwj
from wwj.core import _eigvals, _n_meaningful
from wwj.bayes import bayes_analyze_matrix, bayes_summary


def _pareto_matrix(shape, alpha_target=3.0, n=2000, seed=0):
    """Heavy-tailed matrix with a fittable power-law tail (mirrors test_bayes.py)."""
    rng = np.random.default_rng(seed)
    n_tail = max(1, n // 4)
    tail = (rng.pareto(alpha_target - 1, size=n_tail) + 1).astype(np.float32) * 0.1
    bulk = np.abs(rng.standard_normal(n - n_tail).astype(np.float32)) * 0.01
    eigs = np.concatenate([tail, bulk])
    svals = np.sqrt(np.sort(eigs)[::-1][:min(shape)])
    U = np.linalg.qr(rng.standard_normal((shape[0], shape[0])).astype(np.float32))[0]
    V = np.linalg.qr(rng.standard_normal((shape[1], shape[1])).astype(np.float32))[0]
    return jnp.asarray(U[:, :len(svals)] @ np.diag(svals) @ V[:len(svals), :])


def _degenerate_matrix(n=256, block=10, seed=1):
    """Genuinely degenerate: zero matrix with a small nonzero block => STRUCTURAL
    zero eigenvalues, so _n_meaningful < min_tail_size. (A merely low-rank float32
    product is NOT degenerate by the relative 1e-9 floor; only structural zeros are.)"""
    W = np.zeros((n, n), dtype="float32")
    W[:block, :block] = np.random.default_rng(seed).standard_normal((block, block))
    return jnp.asarray(W)


def test_degenerate_matrix_is_flagged_not_spuriously_fit():
    W = _degenerate_matrix()
    assert int(_n_meaningful(_eigvals(W))) < 50          # fixture really is degenerate
    bs = bayes_analyze_matrix(W, name="dead", min_tail_size=50)
    assert bs.degenerate is True
    assert math.isnan(bs.alpha_mean)                     # no spurious finite alpha


def test_healthy_matrix_not_flagged():
    bs = bayes_analyze_matrix(_pareto_matrix((1024, 256), n=1024), name="alive", min_tail_size=50)
    assert bs.degenerate is False
    assert math.isfinite(bs.alpha_mean) and bs.alpha_mean > 1.0


def test_bayes_summary_excludes_degenerate_and_reports_accounting():
    alive = bayes_analyze_matrix(_pareto_matrix((1024, 256), n=1024), name="alive", min_tail_size=50)
    dead = bayes_analyze_matrix(_degenerate_matrix(), name="dead", min_tail_size=50)
    s = bayes_summary([alive, dead])
    assert s["n_layers"] == 2
    assert s["n_degenerate"] == 1
    assert s["n_fit"] == 1
    assert math.isfinite(s["alpha_mean"])                # not nan-poisoned by the dead layer
    assert s["alpha_mean"] == pytest.approx(float(alive.alpha_mean), rel=1e-5)
