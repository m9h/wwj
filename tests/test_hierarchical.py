# Phase 5: hierarchical cross-layer model (NumPyro). Partial-pools per-layer alpha
# toward a population mean -> a credible interval on the model's typical alpha and
# its distance from the RG fixed point 2, plus shrinkage of noisy per-layer fits.
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import wwj
from wwj.core import _eigvals

numpyro = pytest.importorskip("numpyro")  # optional 'bayes' dependency


def _powerlaw_eigs(n_tail=200, alpha=3.0, xmin0=1.0, seed=0):
    rng = np.random.default_rng(seed)
    tail = xmin0 * (1.0 + rng.pareto(alpha - 1.0, size=n_tail))
    bulk = rng.uniform(0.01 * xmin0, 0.1 * xmin0, size=80)
    eigs = np.sort(np.concatenate([tail, bulk]))[::-1].astype(np.float32)
    return jnp.asarray(eigs.copy())


def test_hierarchical_recovers_population_mean():
    """Six layers all generated near alpha=3 -> population posterior mean should
    land near 3 with a finite credible interval bracketing it."""
    layers = [_powerlaw_eigs(alpha=3.0, seed=s) for s in range(6)]
    out = wwj.hierarchical_alpha(layers, num_steps=1500, key=jax.random.PRNGKey(0))
    for k in ("mu_mean", "mu_ci_low", "mu_ci_high", "tau_mean", "layer_alphas",
              "prob_mu_lt_2"):
        assert k in out, f"missing {k}"
    assert len(out["layer_alphas"]) == 6
    assert out["mu_ci_low"] <= out["mu_mean"] <= out["mu_ci_high"]
    assert 2.4 < out["mu_mean"] < 3.6, f"population mean off: {out['mu_mean']:.3f}"
    assert 0.0 <= out["prob_mu_lt_2"] <= 1.0


def test_hierarchical_shrinks_toward_population():
    """Partial pooling must shrink the spread of per-layer alphas relative to the
    independent (unpooled) per-layer posterior means."""
    # deliberately heterogeneous/noisy layers (varying alpha and small tails)
    layers = [_powerlaw_eigs(n_tail=120, alpha=a, seed=s)
              for s, a in enumerate([2.2, 2.8, 3.0, 3.4, 4.0, 2.5])]
    out = wwj.hierarchical_alpha(layers, num_steps=1500, key=jax.random.PRNGKey(1))
    unpooled = np.array([wwj.alpha_posterior(e)["alpha_mean"] for e in layers])
    pooled = np.array(out["layer_alphas"])
    assert pooled.std() <= unpooled.std() + 1e-6, \
        f"no shrinkage: pooled std {pooled.std():.3f} > unpooled std {unpooled.std():.3f}"


def test_hierarchical_nuts_reports_convergence_diagnostics():
    """The NUTS path should expose R-hat / ESS so convergence is checkable."""
    layers = [_powerlaw_eigs(alpha=3.0, seed=s) for s in range(4)]
    out = wwj.hierarchical_alpha(layers, method="nuts", num_warmup=300,
                                 num_samples=300, key=jax.random.PRNGKey(2))
    assert "mu_rhat" in out and np.isfinite(out["mu_rhat"])
    assert out["mu_rhat"] < 1.2, f"NUTS not converged: r_hat={out['mu_rhat']:.3f}"
    assert "mu_ess" in out and out["mu_ess"] > 0
