# Tests for wwjd (Bayesian WeightWatcher). Phase 1: conjugate Gamma posterior
# over the power-law exponent alpha.
#
# Key identity under test: the tail above xmin is Pareto, so beta = (alpha - 1)
# is the rate of an Exponential over t = log(lambda/xmin). With a Gamma(a0, b0)
# prior on beta, the posterior is Gamma(a0 + n, b0 + sum t) -- closed form. In
# the flat-prior (a0, b0 -> 0) limit the posterior mean of alpha must collapse
# to the CSN MLE 1 + n / sum t.
import numpy as np
import jax.numpy as jnp
import pytest

import wwj
from wwj.core import _eigvals, _ks_select_xmin, _csn_mle_alpha


def _pareto_matrix(shape, alpha_target=3.0, n=2000, seed=0):
    """Weight matrix whose W^T W spectrum has an approximately power-law tail
    with exponent alpha_target. Mirrors tests/test_new_features.py."""
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


def test_alpha_posterior_keys_and_ordering():
    W = jnp.asarray(_pareto_matrix((1024, 256), alpha_target=3.0, n=1024))
    eigs = _eigvals(W)
    post = wwj.alpha_posterior(eigs, ci=0.95)
    for k in ("alpha_mean", "alpha_map", "xmin", "ci_low", "ci_high",
              "post_a", "post_b", "p_alpha_lt_2", "tail_size"):
        assert k in post, f"missing key {k}"
        assert np.isfinite(post[k]), f"non-finite {k}={post[k]}"
    assert post["ci_low"] <= post["alpha_mean"] <= post["ci_high"]
    assert post["post_a"] > 0 and post["post_b"] > 0
    assert 0.0 <= post["p_alpha_lt_2"] <= 1.0
    assert post["alpha_mean"] > 1.0  # alpha = 1 + positive rate


def test_alpha_posterior_matches_mle_in_flat_prior_limit():
    """Posterior mean of alpha with a vanishing prior must equal the CSN MLE
    on the same xmin/tail. This is an exact algebraic identity, not a fit."""
    W = jnp.asarray(_pareto_matrix((2048, 512), alpha_target=3.5, n=2048))
    eigs = _eigvals(W)
    xmin = _ks_select_xmin(eigs)
    mle = float(_csn_mle_alpha(eigs, xmin))
    post = wwj.alpha_posterior(eigs, xmin=xmin, a0=1e-9, b0=1e-9)
    assert post["alpha_mean"] == pytest.approx(mle, rel=1e-4), \
        f"flat-prior posterior mean {post['alpha_mean']:.6f} != MLE {mle:.6f}"


def test_alpha_posterior_ci_agrees_with_bootstrap():
    """The closed-form credible interval should land in the same ballpark as the
    frequentist bootstrap CI (same xmin, same data)."""
    W = jnp.asarray(_pareto_matrix((2048, 512), alpha_target=3.0, n=2048))
    eigs = _eigvals(W)
    xmin = _ks_select_xmin(eigs)
    boot = wwj.bootstrap_alpha_ci(eigs, n_bootstrap=2000, ci=0.95)
    post = wwj.alpha_posterior(eigs, xmin=xmin, ci=0.95)
    # intervals overlap
    assert post["ci_low"] <= boot["ci_high"] and boot["ci_low"] <= post["ci_high"], \
        f"no overlap: post[{post['ci_low']:.3f},{post['ci_high']:.3f}] " \
        f"boot[{boot['ci_low']:.3f},{boot['ci_high']:.3f}]"
    # comparable width (not off by more than 3x either way)
    wp = post["ci_high"] - post["ci_low"]
    wb = boot["ci_high"] - boot["ci_low"]
    assert 1 / 3 < wp / wb < 3, f"CI widths disagree: post {wp:.3f} vs boot {wb:.3f}"


def test_alpha_posterior_ci_tightens_with_more_data():
    """Larger tail -> tighter credible interval (posterior concentrates)."""
    small = _eigvals(jnp.asarray(_pareto_matrix((512, 256), alpha_target=3.0, n=512, seed=1)))
    large = _eigvals(jnp.asarray(_pareto_matrix((4096, 512), alpha_target=3.0, n=4096, seed=1)))
    ws = wwj.alpha_posterior(small)["ci_high"] - wwj.alpha_posterior(small)["ci_low"]
    wl = wwj.alpha_posterior(large)["ci_high"] - wwj.alpha_posterior(large)["ci_low"]
    assert wl < ws, f"CI did not tighten with more data: {ws:.3f} -> {wl:.3f}"


def test_alpha_posterior_p_alpha_lt_2_low_for_steep_tail():
    """Data generated with alpha well above 2 should put little posterior mass
    below 2."""
    W = jnp.asarray(_pareto_matrix((2048, 512), alpha_target=4.0, n=2048))
    eigs = _eigvals(W)
    post = wwj.alpha_posterior(eigs)
    assert post["p_alpha_lt_2"] < 0.5, \
        f"unexpected mass below alpha=2: {post['p_alpha_lt_2']:.3f} (mean {post['alpha_mean']:.2f})"
