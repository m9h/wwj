# Phase 4: posterior-predictive check. Replaces the bare KS distance with a
# calibrated Bayesian p-value: draw replicate tails from the Pareto posterior
# predictive (Lomax), compare a parameter-dependent KS discrepancy of replicate
# vs observed. p ~ 0.5 => data look like draws from the fitted power law; p small
# => the power-law tail is a poor fit.
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import wwj


def _powerlaw_eigs(n_tail=300, alpha=3.0, xmin0=1.0, seed=0):
    """A spectrum whose tail is GENUINELY Pareto(alpha) above xmin0, plus a small
    light bulk so the KS window selector locks onto the power-law tail."""
    rng = np.random.default_rng(seed)
    tail = xmin0 * (1.0 + rng.pareto(alpha - 1.0, size=n_tail))
    bulk = rng.uniform(0.01 * xmin0, 0.1 * xmin0, size=120)
    eigs = np.sort(np.concatenate([tail, bulk]))[::-1].astype(np.float32)
    return jnp.asarray(eigs.copy())


def _flat_eigs(n_tail=300, xmin0=1.0, seed=0):
    """A clearly NON-power-law tail: linearly spaced (uniform) eigenvalues."""
    rng = np.random.default_rng(seed)
    tail = rng.uniform(xmin0, 5.0 * xmin0, size=n_tail)
    bulk = rng.uniform(0.01 * xmin0, 0.1 * xmin0, size=120)
    eigs = np.sort(np.concatenate([tail, bulk]))[::-1].astype(np.float32)
    return jnp.asarray(eigs.copy())


def test_ppc_keys_and_range():
    eigs = _powerlaw_eigs(seed=0)
    out = wwj.ppc_pvalue(eigs, n_rep=300, key=jax.random.PRNGKey(0))
    for k in ("p_value", "ks_obs_mean", "xmin", "tail_size"):
        assert k in out and np.isfinite(out[k]), f"bad key {k}"
    assert 0.0 <= out["p_value"] <= 1.0
    assert out["tail_size"] >= 50


def test_ppc_not_rejected_on_genuine_power_law():
    """Data actually drawn from a power law should not be rejected -- the Bayesian
    p-value should be comfortably above a 0.05 rejection threshold."""
    eigs = _powerlaw_eigs(n_tail=400, alpha=3.0, seed=1)
    out = wwj.ppc_pvalue(eigs, n_rep=500, key=jax.random.PRNGKey(1))
    assert out["p_value"] > 0.05, f"genuine power law rejected: p={out['p_value']:.3f}"


def test_ppc_lower_for_nonpowerlaw_than_powerlaw():
    """A flat/uniform tail should fit the power law worse than a genuine power-law
    tail -> strictly smaller posterior-predictive p-value."""
    p_pl = wwj.ppc_pvalue(_powerlaw_eigs(seed=2), n_rep=400, key=jax.random.PRNGKey(2))["p_value"]
    p_flat = wwj.ppc_pvalue(_flat_eigs(seed=2), n_rep=400, key=jax.random.PRNGKey(2))["p_value"]
    assert p_flat < p_pl, f"non-power-law not penalised: flat={p_flat:.3f} pl={p_pl:.3f}"
