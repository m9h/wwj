# Bayesian-workflow diagnostics added on top of the core posteriors:
#   #6 prob_in_rope        -- ROPE around alpha=2 (SETOL 'Ideal layer' test)
#   #5 prior_sensitivity   -- power-scaling prior sensitivity (Kallioinen 2023)
#   #2 model_posterior(extended=True) -- + truncated-PL and generalized-Pareto tails
#   #1 accuracy_alpha_regression -- errors-in-variables accuracy ~ alpha_hat
import numpy as np
import jax.numpy as jnp
import pytest

import wwj
from wwj.core import _eigvals


def _synth_powerlaw_eigs(alpha_true, n=512, m=512, xmin=1.0, seed=0):
    rng = np.random.default_rng(seed)
    k = min(n, m)
    u = rng.uniform(size=k)
    eig = xmin * u ** (-1.0 / (alpha_true - 1.0))
    N = max(n, m)
    sv = np.sqrt(np.sort(eig)[::-1] * N)
    U, _ = np.linalg.qr(rng.standard_normal((n, k)))
    V, _ = np.linalg.qr(rng.standard_normal((m, k)))
    W = ((U * sv) @ V.T).astype(np.float32)
    return _eigvals(jnp.asarray(W))


# ---- #6 ROPE -------------------------------------------------------------------
def test_rope_concentrates_for_alpha_near_2():
    near2 = wwj.prob_in_rope(_synth_powerlaw_eigs(2.0), center=2.0, delta=0.25)
    far = wwj.prob_in_rope(_synth_powerlaw_eigs(4.0), center=2.0, delta=0.25)
    assert near2["prob_in_rope"] > 0.5, near2
    assert far["prob_in_rope"] < near2["prob_in_rope"]
    assert 0.0 <= near2["prob_in_rope"] <= 1.0


# ---- #5 prior sensitivity ------------------------------------------------------
def test_prior_sensitivity_low_on_large_tail():
    """A well-populated tail with a weak prior should be nearly prior-insensitive:
    the posterior mean barely moves across prior power-scalings."""
    s = wwj.prior_sensitivity(_synth_powerlaw_eigs(3.0, n=1024, m=1024))
    assert s["max_abs_shift"] < 0.05, s
    assert s["sensitivity"] >= 0.0
    assert set(s["alpha_mean_by_scale"]).issuperset({0.25, 1.0, 4.0})


# ---- #2 extended model comparison (TPL + GPD) ----------------------------------
def test_extended_model_posterior_adds_tpl_gpd():
    eigs = _synth_powerlaw_eigs(3.0)
    out = wwj.model_posterior(eigs, extended=True)
    for k in ("logev_truncated_powerlaw", "logev_generalized_pareto",
              "logbf_pl_vs_tpl", "logbf_pl_vs_gpd",
              "prob_truncated_powerlaw", "prob_generalized_pareto"):
        assert k in out, k
    probs = [out[f"prob_{m}"] for m in ("powerlaw", "exponential", "lognormal",
                                        "truncated_powerlaw", "generalized_pareto")]
    assert all(np.isfinite(p) and p >= 0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-5, probs
    assert out["best_model"] in ("powerlaw", "exponential", "lognormal",
                                 "truncated_powerlaw", "generalized_pareto")
    for k in ("logev_truncated_powerlaw", "logev_generalized_pareto"):
        assert np.isfinite(out[k]), k


def test_default_model_posterior_unchanged():
    """extended=False keeps the original 3-way behaviour (back-compat)."""
    out = wwj.model_posterior(_synth_powerlaw_eigs(3.0))
    assert "logev_truncated_powerlaw" not in out
    probs = [out["prob_powerlaw"], out["prob_exponential"], out["prob_lognormal"]]
    assert abs(sum(probs) - 1.0) < 1e-5


# ---- #1 errors-in-variables regression -----------------------------------------
def test_eiv_regression_recovers_negative_slope():
    """Synthetic: accuracy = 80 - 5*alpha_hat + noise. The EIV posterior slope should
    be negative with high probability (the HT-SR 'smaller alpha_hat -> higher acc')."""
    rng = np.random.default_rng(0)
    ah = np.linspace(1.0, 3.0, 12).astype(np.float32)
    se = np.full(12, 0.1, dtype=np.float32)
    acc = (80.0 - 5.0 * ah + rng.normal(0, 0.5, size=12)).astype(np.float32)
    out = wwj.accuracy_alpha_regression(ah, se, acc, method="nuts",
                                        num_warmup=500, num_samples=1000)
    assert out["slope_mean"] < 0, out
    assert out["prob_slope_neg"] > 0.9, out
    assert out["slope_ci_high"] < 0  # whole CI below zero
    # recovers roughly the true slope -5 (EIV widens it a bit vs naive)
    assert -8.0 < out["slope_mean"] < -2.0, out
