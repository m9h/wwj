# wwjd Phase 5: hierarchical cross-layer model. The per-layer conjugate posteriors
# (bayes.py) treat layers independently; this partial-pools them. Each layer's tail
# above its KS window is Exponential(beta_layer), beta_layer = alpha_layer - 1, and
# the layer alphas are drawn from a population Normal(mu, tau). Fitting mu gives a
# credible interval on the model's *typical* alpha and its distance from the RG
# fixed point 2 -- the population-level statement the frequentist summary() can't
# make -- while the per-layer estimates shrink toward mu (noisy small-tail layers
# borrow strength).
#
# Likelihood uses per-layer sufficient statistics (n_l, S_l) only: for an
# Exponential, sum log p = n log(beta) - beta S, so the full data likelihood is
# captured exactly by (n, S). Requires the optional 'bayes' extra (numpyro).

from __future__ import annotations

import jax
import jax.numpy as jnp

from wwj.core import _eigvals, _ks_select_xmin
from wwj.bayes import _tail_stats


def _layer_suffstats(layers):
    """(n_arr, S_arr) per layer at each layer's KS-selected xmin. `layers` is a list
    of descending-sorted eigenvalue arrays (wwj.core._eigvals output)."""
    ns, Ss = [], []
    for eigs in layers:
        xmin = _ks_select_xmin(eigs)
        n, S = _tail_stats(eigs, xmin)
        ns.append(n)
        Ss.append(S)
    return jnp.stack(ns).astype(jnp.float32), jnp.stack(Ss).astype(jnp.float32)


def _build_model(numpyro, dist, centered: bool):
    """Population model over per-layer alpha. `centered` toggles the
    parameterisation: centered converges better under SVI with strong per-layer
    data; non-centered avoids the funnel and is the right choice for NUTS."""
    def model(n_arr, S_arr):
        n_layers = n_arr.shape[0]
        mu = numpyro.sample("mu", dist.Normal(2.0, 1.0))      # population mean alpha
        tau = numpyro.sample("tau", dist.HalfNormal(0.5))     # between-layer spread
        with numpyro.plate("layers", n_layers):
            if centered:
                alpha = numpyro.sample("alpha", dist.Normal(mu, tau))
            else:
                z = numpyro.sample("z", dist.Normal(0.0, 1.0))
                alpha = numpyro.deterministic("alpha", mu + tau * z)
        beta = alpha - 1.0
        # Exponential sufficient-stat log-likelihood; guard the invalid beta<=0 region.
        ll = jnp.where(beta > 0.0,
                       n_arr * jnp.log(jnp.clip(beta, 1e-12)) - beta * S_arr,
                       -1e8)
        numpyro.factor("lik", jnp.sum(ll))
    return model


def _summarise(mu_s, tau_s, alpha_s, ci):
    lo, hi = (1 - ci) / 2, (1 + ci) / 2
    return {
        "mu_mean": float(jnp.mean(mu_s)),
        "mu_ci_low": float(jnp.quantile(mu_s, lo)),
        "mu_ci_high": float(jnp.quantile(mu_s, hi)),
        "tau_mean": float(jnp.mean(tau_s)),
        "prob_mu_lt_2": float(jnp.mean(mu_s < 2.0)),
        "mu_dist_from_2": float(jnp.abs(jnp.mean(mu_s) - 2.0)),
        "layer_alphas": [float(a) for a in jnp.mean(alpha_s, axis=0)],
    }


def hierarchical_alpha(layers, method: str = "svi", ci: float = 0.95,
                       num_steps: int = 2000, num_warmup: int = 500,
                       num_samples: int = 1000, num_draws: int = 2000, key=None):
    """Hierarchical population posterior over per-layer alpha.

    Args:
        layers: list of descending-sorted eigenvalue arrays (one per weight matrix;
            use wwj.core._eigvals). Or pass an Equinox model to hierarchical_analyze.
        method: "svi" (AutoNormal + Adam, fast, default) or "nuts" (adds R-hat/ESS).
        num_steps: SVI optimisation steps. num_warmup/num_samples: NUTS budget.

    Returns mu_mean, mu_ci_low/high, tau_mean, prob_mu_lt_2, mu_dist_from_2,
    layer_alphas (shrunk per-layer estimates); the NUTS path adds mu_rhat, mu_ess.
    """
    try:
        import numpyro
        import numpyro.distributions as dist
    except ImportError as e:  # pragma: no cover
        raise ImportError("hierarchical_alpha needs the optional 'bayes' extra: "
                          "pip install 'wwj[bayes]'") from e

    import jax.random as jr
    key = jr.PRNGKey(0) if key is None else key
    n_arr, S_arr = _layer_suffstats(layers)

    if method == "svi":
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoMultivariateNormal
        from numpyro.optim import Adam
        model = _build_model(numpyro, dist, centered=True)
        k_fit, k_draw = jr.split(key)
        guide = AutoMultivariateNormal(model)
        svi = SVI(model, guide, Adam(0.02), Trace_ELBO())
        res = svi.run(k_fit, num_steps, n_arr, S_arr, progress_bar=False)
        post = guide.sample_posterior(k_draw, res.params, sample_shape=(num_draws,))
        return _summarise(post["mu"], post["tau"], post["alpha"], ci)

    elif method == "nuts":
        from numpyro.infer import MCMC, NUTS
        from numpyro.diagnostics import summary as mcmc_summary
        model = _build_model(numpyro, dist, centered=False)
        mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=1, progress_bar=False)
        mcmc.run(key, n_arr, S_arr)
        s = mcmc.get_samples()
        out = _summarise(s["mu"], s["tau"], s["alpha"], ci)
        diag = mcmc_summary(mcmc.get_samples(group_by_chain=True))
        out["mu_rhat"] = float(diag["mu"]["r_hat"])
        out["mu_ess"] = float(diag["mu"]["n_eff"])
        return out

    raise ValueError(f"unknown method {method!r}; use 'svi' or 'nuts'")


def hierarchical_analyze(model, min_dim: int = 50, **kwargs):
    """Convenience: walk an Equinox model's 2D weight matrices and fit the
    hierarchical population posterior over their alphas."""
    from wwj.core import _walk_matrices
    layers = [_eigvals(W) for _, W in _walk_matrices(model, min_dim=min_dim)]
    return hierarchical_alpha(layers, **kwargs)


def accuracy_alpha_regression(alpha_hat, alpha_hat_se, accuracy,
                              method: str = "nuts", ci: float = 0.95,
                              num_warmup: int = 1000, num_samples: int = 2000,
                              num_steps: int = 4000, key=None):
    """Errors-in-variables Bayesian regression of a downstream metric (e.g. test
    accuracy) on the weights-only weighted-alpha alpha_hat, PROPAGATING the alpha_hat
    posterior uncertainty. This closes the HT-SR loop: WeightWatcher reports a point
    correlation between alpha_hat and accuracy, but alpha_hat is itself a noisy
    posterior summary. A measurement-error model gives a calibrated posterior on the
    predictive slope and honest prediction intervals on 'what accuracy does this
    checkpoint's spectrum imply?'.

    Model (latent true alpha_hat per model i):
        x_true[i] ~ Normal(mu_x, tau_x)
        alpha_hat[i] ~ Normal(x_true[i], alpha_hat_se[i])     # measurement
        accuracy[i] ~ Normal(intercept + slope * x_true[i], sigma)   # structural

    Args:
        alpha_hat: per-model weighted-alpha point/posterior-mean (array-like).
        alpha_hat_se: per-model standard error on alpha_hat (e.g. posterior SD of the
            per-model weighted-alpha; pass small constants if unknown).
        accuracy: per-model downstream metric (array-like, same length).

    Returns: slope_mean, slope_ci_low/high, prob_slope_neg (the HT-SR claim
    'smaller alpha_hat -> higher accuracy'), intercept_mean, sigma_mean, the naive
    OLS slope that ignores measurement error (for contrast), and (NUTS) slope_rhat.
    """
    try:
        import numpyro
        import numpyro.distributions as dist
    except ImportError as e:  # pragma: no cover
        raise ImportError("accuracy_alpha_regression needs the 'bayes' extra: "
                          "pip install 'wwj[bayes]'") from e
    import jax.random as jr
    import numpy as np

    key = jr.PRNGKey(0) if key is None else key
    x = jnp.asarray(alpha_hat, dtype=jnp.float32)
    se = jnp.asarray(alpha_hat_se, dtype=jnp.float32)
    y = jnp.asarray(accuracy, dtype=jnp.float32)
    # standardise x and y for sampling stability; map the slope back to raw units after.
    x_mu, x_sd = jnp.mean(x), jnp.std(x) + 1e-6
    y_mu, y_sd = jnp.mean(y), jnp.std(y) + 1e-6
    xz, sez, yz = (x - x_mu) / x_sd, se / x_sd, (y - y_mu) / y_sd

    def model(xz, sez, yz):
        n = xz.shape[0]
        intercept = numpyro.sample("intercept", dist.Normal(0.0, 2.0))
        slope = numpyro.sample("slope", dist.Normal(0.0, 2.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        mu_x = numpyro.sample("mu_x", dist.Normal(0.0, 2.0))
        tau_x = numpyro.sample("tau_x", dist.HalfNormal(2.0))
        with numpyro.plate("models", n):
            x_true = numpyro.sample("x_true", dist.Normal(mu_x, tau_x))
            numpyro.sample("x_obs", dist.Normal(x_true, jnp.clip(sez, 1e-3)), obs=xz)
            numpyro.sample("y_obs", dist.Normal(intercept + slope * x_true, sigma), obs=yz)

    def _finish(slope_s, intercept_s, sigma_s, extra=None):
        lo, hi = (1 - ci) / 2, (1 + ci) / 2
        slope_raw = slope_s * (y_sd / x_sd)               # de-standardise slope to raw units
        out = {
            "slope_mean": float(jnp.mean(slope_raw)),
            "slope_ci_low": float(jnp.quantile(slope_raw, lo)),
            "slope_ci_high": float(jnp.quantile(slope_raw, hi)),
            "prob_slope_neg": float(jnp.mean(slope_raw < 0)),   # the HT-SR direction
            "intercept_mean": float(jnp.mean(intercept_s) * y_sd + y_mu),
            "sigma_mean": float(jnp.mean(sigma_s) * y_sd),
            # naive OLS slope (ignores measurement error) for contrast
            "naive_ols_slope": float(np.polyfit(np.asarray(x), np.asarray(y), 1)[0]),
            "n_models": int(x.shape[0]),
        }
        if extra:
            out.update(extra)
        return out

    if method == "svi":
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoMultivariateNormal
        from numpyro.optim import Adam
        k_fit, k_draw = jr.split(key)
        guide = AutoMultivariateNormal(model)
        svi = SVI(model, guide, Adam(0.01), Trace_ELBO())
        res = svi.run(k_fit, num_steps, xz, sez, yz, progress_bar=False)
        post = guide.sample_posterior(k_draw, res.params, sample_shape=(2000,))
        return _finish(post["slope"], post["intercept"], post["sigma"])

    elif method == "nuts":
        from numpyro.infer import MCMC, NUTS
        from numpyro.diagnostics import summary as mcmc_summary
        mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=1, progress_bar=False)
        mcmc.run(key, xz, sez, yz)
        s = mcmc.get_samples()
        diag = mcmc_summary(mcmc.get_samples(group_by_chain=True))
        return _finish(s["slope"], s["intercept"], s["sigma"],
                       {"slope_rhat": float(diag["slope"]["r_hat"]),
                        "slope_ess": float(diag["slope"]["n_eff"])})

    raise ValueError(f"unknown method {method!r}; use 'svi' or 'nuts'")
