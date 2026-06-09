# Phase 3: Bayesian model comparison. Replaces fit_distributions' Vuong LRT
# (a sign on a log-likelihood ratio) with analytic marginal likelihoods for
# power-law / exponential / lognormal -> Bayes factors + posterior model
# probabilities. All three evidences are closed form under conjugate priors
# (Gamma for PL/exp rate, Normal-Inverse-Gamma for lognormal), computed on the
# SAME tail data so the ratios are valid.
import numpy as np
import jax.numpy as jnp
import pytest

import wwj
from wwj.core import _eigvals


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


def test_model_posterior_valid_and_finite():
    eigs = _eigvals(jnp.asarray(_pareto_matrix((1024, 256), alpha_target=3.0)))
    out = wwj.model_posterior(eigs)
    probs = np.array([out["prob_powerlaw"], out["prob_exponential"], out["prob_lognormal"]])
    assert np.all(probs >= 0) and abs(probs.sum() - 1.0) < 1e-5, f"probs={probs}"
    for k in ("logev_powerlaw", "logev_exponential", "logev_lognormal",
              "logbf_pl_vs_exp", "logbf_pl_vs_ln", "best_model", "xmin", "tail_size"):
        assert k in out
    assert out["best_model"] in ("powerlaw", "exponential", "lognormal")
    for k in ("logev_powerlaw", "logev_exponential", "logev_lognormal"):
        assert np.isfinite(out[k]), f"non-finite {k}"
    # best_model is the argmax of the posterior probabilities
    names = ["powerlaw", "exponential", "lognormal"]
    assert out["best_model"] == names[int(np.argmax(probs))]


@pytest.mark.parametrize("seed,alpha_target,shape", [
    (0, 3.0, (1024, 256)),   # power-law-ish tail
    (1, 5.0, (1024, 256)),   # steeper
    (2, 2.5, (768, 256)),    # shallower
])
def test_logbf_pl_vs_exp_sign_agrees_with_frequentist_lrt(seed, alpha_target, shape):
    """The Bayesian log Bayes factor (PL vs exp) and the frequentist Vuong LRT
    (pl_vs_exp_lrt) are computed on the same tail; their SIGNS should agree. This
    ties the new Bayesian comparison to the existing oracle in fit_distributions."""
    eigs = _eigvals(jnp.asarray(_pareto_matrix(shape, alpha_target=alpha_target, seed=seed)))
    bayes = wwj.model_posterior(eigs)
    freq = wwj.fit_distributions(eigs)
    assert np.sign(bayes["logbf_pl_vs_exp"]) == np.sign(freq["pl_vs_exp_lrt"]), \
        f"sign disagreement: logBF={bayes['logbf_pl_vs_exp']:.3f} " \
        f"LRT={freq['pl_vs_exp_lrt']:.3f}"


def test_logbf_favors_power_law_on_pareto_data():
    """Mirror of test_lrt_favors_power_law_on_pareto_data, Bayesian side."""
    eigs = _eigvals(jnp.asarray(_pareto_matrix((1024, 256), alpha_target=3.0)))
    out = wwj.model_posterior(eigs)
    assert out["logbf_pl_vs_exp"] > 0, \
        f"PL not favored over exp on power-law data: logBF={out['logbf_pl_vs_exp']:.3f}"
