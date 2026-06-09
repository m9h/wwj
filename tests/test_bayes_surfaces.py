# Phase 6: wwjd surfaces -- bayes_analyze / bayes_summary / bayes_alpha_loss,
# the Bayesian mirrors of core.analyze / summary / alpha_loss.
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import wwj


def _small_model(key=jax.random.PRNGKey(0)):
    k1, k2 = jax.random.split(key)
    # two qualifying (>= min_dim) weight matrices
    return eqx.nn.MLP(in_size=64, out_size=64, width_size=64, depth=1, key=k1)


def test_bayes_analyze_returns_per_layer_records():
    model = _small_model()
    stats = wwj.bayes_analyze(model)
    assert len(stats) >= 2, f"expected >=2 weight matrices, got {len(stats)}"
    for s in stats:
        assert s.ci_low <= s.alpha_mean <= s.ci_high
        assert 0.0 <= s.p_alpha_lt_2 <= 1.0
        assert 0.0 <= s.prob_powerlaw <= 1.0
        assert s.best_model in ("powerlaw", "exponential", "lognormal")
        assert s.tail_size >= 1
        assert s.ppc_pvalue is None          # off by default
        assert np.isfinite(s.alpha_std) and s.alpha_std >= 0


def test_bayes_analyze_ppc_flag_populates_pvalue():
    model = _small_model()
    stats = wwj.bayes_analyze(model, ppc=True)
    assert all(s.ppc_pvalue is not None and 0.0 <= s.ppc_pvalue <= 1.0 for s in stats)


def test_bayes_summary_aggregates():
    model = _small_model()
    summ = wwj.bayes_summary(wwj.bayes_analyze(model))
    for k in ("n_layers", "alpha_mean", "alpha_dist_mean", "mean_posterior_std",
              "mean_p_alpha_lt_2", "frac_powerlaw_best", "mean_prob_powerlaw"):
        assert k in summ and np.isfinite(summ[k]), f"bad summary key {k}"
    assert summ["n_layers"] >= 2
    assert 0.0 <= summ["frac_powerlaw_best"] <= 1.0
    assert summ["mean_posterior_std"] >= 0.0


def test_bayes_summary_empty():
    assert wwj.bayes_summary([]) == {"n_layers": 0}


def test_bayes_alpha_loss_is_differentiable_and_finite():
    model = _small_model()
    loss = wwj.bayes_alpha_loss(model, target=2.0)
    assert jnp.ndim(loss) == 0 and np.isfinite(float(loss))
    grads = eqx.filter_grad(lambda m: wwj.bayes_alpha_loss(m, target=2.0))(model)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if eqx.is_inexact_array(g)]
    assert any(np.any(np.asarray(g) != 0) for g in leaves), "no gradient flowed"
    assert all(np.all(np.isfinite(np.asarray(g))) for g in leaves), "non-finite grad"


def test_bayes_alpha_loss_minimized_at_target_alpha():
    """Loss should be smaller when the layer posterior-mean alphas sit at the
    target than when the target is far away."""
    model = _small_model()
    # measure the realised mean alpha, then compare loss at that target vs a far one
    summ = wwj.bayes_summary(wwj.bayes_analyze(model))
    a_here = summ["alpha_mean"]
    near = float(wwj.bayes_alpha_loss(model, target=a_here, var_weight=0.0))
    far = float(wwj.bayes_alpha_loss(model, target=a_here + 5.0, var_weight=0.0))
    assert near < far, f"loss not minimized near realised alpha: near={near:.3f} far={far:.3f}"
