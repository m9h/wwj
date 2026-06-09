# wwjd -- Bayesian WeightWatcher. Phase 1: conjugate Gamma posterior over the
# HT-SR power-law exponent alpha.
#
# The tail above xmin is Pareto, so beta = (alpha - 1) is the rate of an
# Exponential over t = log(lambda / xmin). With a conjugate Gamma(a0, b0) prior
# on beta, the posterior is closed form:
#
#     beta | data ~ Gamma(shape = a0 + n, rate = b0 + sum t),   alpha = 1 + beta
#
# Posterior mean of alpha = 1 + (a0 + n) / (b0 + sum t); in the flat-prior limit
# (a0, b0 -> 0) this is exactly the CSN MLE 1 + n / sum t. Credible intervals are
# exact Gamma quantiles; P(alpha < 2) = P(beta < 1) is an exact Gamma CDF. No
# sampling, fully differentiable -- pure JAX (no tfp/numpyro).

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln, ndtri
from jaxtyping import Array, Float

from wwj.core import _ks_select_xmin


def _gammaincinv(a: Float[Array, ""], q: Float[Array, ""],
                 n_iter: int = 12) -> Float[Array, ""]:
    """Inverse of the regularized lower incomplete gamma: return x with
    gammainc(a, x) = q (i.e. the q-quantile of a unit-rate Gamma(a)).

    JAX ships gammainc but not its inverse, so we seed with the Wilson-Hilferty
    normal approximation and polish with Newton steps. f(x) = gammainc(a, x) - q,
    f'(x) = pdf = exp((a-1) log x - x - gammaln(a)). Halley would be marginally
    faster but Newton on this convex-ish CDF converges in ~6-12 steps to 1e-10."""
    # Wilson-Hilferty: Gamma(a) quantile ~ a * (1 - 1/(9a) + z/sqrt(9a))^3.
    z = ndtri(q)
    g = 1.0 / (9.0 * a)
    x = a * jnp.clip(1.0 - g + z * jnp.sqrt(g), 1e-6) ** 3
    x = jnp.maximum(x, 1e-8)

    def step(x, _):
        f = gammainc(a, x) - q
        log_pdf = (a - 1.0) * jnp.log(x) - x - gammaln(a)
        pdf = jnp.exp(log_pdf)
        x_new = x - f / jnp.maximum(pdf, 1e-30)
        return jnp.maximum(x_new, 1e-12), None

    x, _ = jax.lax.scan(step, x, None, length=n_iter)
    return x


def _tail_stats(eigs: Float[Array, "k"], xmin: Float[Array, ""]):
    """(n, S) for the tail lambda >= xmin: n = tail count, S = sum log(lambda/xmin)."""
    mask = eigs >= xmin
    n = jnp.sum(mask)
    S = jnp.sum(jnp.where(mask, jnp.log(eigs / xmin), 0.0))
    return n, S


def alpha_posterior(eigs: Float[Array, "k"], xmin: Float[Array, ""] | None = None,
                    a0: float = 1e-3, b0: float = 1e-3, ci: float = 0.95) -> dict:
    """Conjugate Gamma posterior over the power-law exponent alpha.

    Args:
        eigs: descending-sorted eigenvalues (output of wwj.core._eigvals).
        xmin: scaling-window lower bound. If None, picked by KS (_ks_select_xmin),
            matching the frequentist pipeline.
        a0, b0: Gamma(shape, rate) prior on beta = alpha - 1. Defaults are weakly
            informative so the flat-prior limit reproduces the CSN MLE.
        ci: central credible level (0.95 -> 2.5%/97.5% quantiles).

    Returns dict with posterior summaries:
        alpha_mean : 1 + (a0 + n) / (b0 + S)               [posterior mean]
        alpha_map  : 1 + (a0 + n - 1) / (b0 + S)           [posterior mode, shape>1]
        xmin, tail_size, post_a, post_b                     [Gamma params on beta]
        ci_low, ci_high                                     [credible interval on alpha]
        p_alpha_lt_2 : P(alpha < 2) = P(beta < 1)          [exact Gamma CDF]
    """
    if xmin is None:
        xmin = _ks_select_xmin(eigs)
    n, S = _tail_stats(eigs, xmin)

    post_a = a0 + n                 # posterior Gamma shape on beta
    post_b = b0 + S                 # posterior Gamma rate on beta
    beta_mean = post_a / post_b
    beta_mode = jnp.maximum(post_a - 1.0, 0.0) / post_b

    lo_q, hi_q = (1.0 - ci) / 2.0, (1.0 + ci) / 2.0
    beta_lo = _gammaincinv(post_a, jnp.asarray(lo_q)) / post_b
    beta_hi = _gammaincinv(post_a, jnp.asarray(hi_q)) / post_b

    # P(alpha < 2) = P(beta < 1) = CDF_Gamma(shape=post_a, rate=post_b)(1)
    p_alpha_lt_2 = gammainc(post_a, post_b)

    return {
        "alpha_mean": float(1.0 + beta_mean),
        "alpha_map": float(1.0 + beta_mode),
        "xmin": float(xmin),
        "tail_size": int(n),
        "post_a": float(post_a),
        "post_b": float(post_b),
        "ci_low": float(1.0 + beta_lo),
        "ci_high": float(1.0 + beta_hi),
        "p_alpha_lt_2": float(p_alpha_lt_2),
    }


# --- Phase 2: posterior over the scaling window xmin + Bayesian model averaging --

def _candidate_log_scores(eigs: Float[Array, "k"], a0: float, b0: float,
                          min_tail_size: int):
    """For every eigenvalue as a candidate xmin, return (n, S, log_score) where
    log_score is a *comparable-across-xmin* marginal-likelihood: the log Bayes
    factor of "the top-n exceedances are Pareto (alpha integrated out under the
    Gamma(a0,b0) prior)" vs "they come from a fixed exponential base measure".

    Comparability is the crux. Raw per-window Pareto evidence is not comparable
    across xmin because each window conditions on different data; subtracting the
    same fixed base density f0(lambda)=r0 exp(-r0 lambda) (r0 = 1/mean(eigs), fit
    once on the whole spectrum) for the same points cancels that, leaving a proper
    log-ratio. This is the Bayesian analogue of the KS pick: it rewards tails that
    are genuinely more Pareto than the bulk, penalising both too-small tails (little
    evidence) and too-large tails (bulk points that fit the base better)."""
    r0 = 1.0 / jnp.maximum(jnp.mean(eigs), 1e-12)        # fixed exponential base rate
    log_r0 = jnp.log(r0)

    def score_for(xmin):
        mask = eigs >= xmin
        n = jnp.sum(mask)
        S = jnp.sum(jnp.where(mask, jnp.log(eigs / xmin), 0.0))
        sum_log_lam = jnp.sum(jnp.where(mask, jnp.log(eigs), 0.0))
        sum_lam = jnp.sum(jnp.where(mask, eigs, 0.0))
        post_a = a0 + n
        post_b = b0 + S
        # Pareto-tail log evidence on the lambda values (prior const a0*log b0 -
        # gammaln(a0) is window-independent -> dropped, cancels in the softmax):
        log_ev_pl = -sum_log_lam + gammaln(post_a) - post_a * jnp.log(post_b)
        # exponential base log density for the same n points:
        log_ev_base = n * log_r0 - r0 * sum_lam
        valid = n >= min_tail_size
        return n, S, jnp.where(valid, log_ev_pl - log_ev_base, -jnp.inf)

    return jax.vmap(score_for)(eigs)


def _mixture_alpha_quantile(q, weights, post_a, post_b, lo=1.0 + 1e-4, hi=60.0,
                            n_iter: int = 50):
    """q-quantile of the Gamma mixture over alpha: F(alpha) = sum_j w_j *
    gammainc(a_j, b_j (alpha-1)). Monotone in alpha, so bisection is exact."""
    def cdf(alpha):
        beta = alpha - 1.0
        return jnp.sum(weights * gammainc(post_a, post_b * beta))

    def body(state, _):
        a, b = state
        mid = 0.5 * (a + b)
        go_right = cdf(mid) < q
        return (jnp.where(go_right, mid, a), jnp.where(go_right, b, mid)), None

    (a, b), _ = jax.lax.scan(body, (jnp.asarray(lo), jnp.asarray(hi)), None, length=n_iter)
    return 0.5 * (a + b)


def alpha_posterior_bma(eigs: Float[Array, "k"], a0: float = 1e-3, b0: float = 1e-3,
                        ci: float = 0.95, min_tail_size: int = 50) -> dict:
    """Bayesian-model-averaged posterior over alpha, marginalising over the scaling
    window xmin instead of conditioning on the single KS-selected window.

    Each candidate xmin gets weight proportional to exp(comparable log evidence)
    under a uniform prior over valid windows (tail >= min_tail_size). The alpha
    posterior is the Gamma mixture sum_j w_j Gamma(a0 + n_j, b0 + S_j); summaries
    below are exact mixture quantities (mean and P(alpha<2) are linear in the
    weights; the credible interval comes from bisecting the mixture CDF).

    Returns the alpha_posterior keys plus: xmin_candidates, xmin_weights,
    xmin_map, n_eff (exp of the weight entropy = effective number of windows).
    """
    n_c, S_c, log_score = _candidate_log_scores(eigs, a0, b0, min_tail_size)
    weights = jax.nn.softmax(log_score)                  # over candidates
    post_a = a0 + n_c
    post_b = b0 + S_c

    # mixture mean of alpha and exact P(alpha < 2) = P(beta < 1), linear in weights
    comp_mean = 1.0 + post_a / post_b                    # per-window posterior mean
    comp_var = post_a / post_b ** 2                      # per-window posterior var
    alpha_mean = jnp.sum(weights * comp_mean)
    p_lt_2 = jnp.sum(weights * gammainc(post_a, post_b))   # b_j * 1

    # Law of total variance: total = within-window + across-window. This decomposition
    # is itself a diagnostic -- how much of the alpha uncertainty is the window pick.
    within_var = jnp.sum(weights * comp_var)
    across_var = jnp.sum(weights * comp_mean ** 2) - alpha_mean ** 2
    total_var = within_var + across_var

    lo_q, hi_q = (1.0 - ci) / 2.0, (1.0 + ci) / 2.0
    ci_low = _mixture_alpha_quantile(jnp.asarray(lo_q), weights, post_a, post_b)
    ci_high = _mixture_alpha_quantile(jnp.asarray(hi_q), weights, post_a, post_b)

    map_idx = jnp.argmax(weights)
    entropy = -jnp.sum(jnp.where(weights > 0, weights * jnp.log(weights + 1e-30), 0.0))
    n_eff = jnp.exp(entropy)

    return {
        "alpha_mean": float(alpha_mean),
        "xmin_map": float(eigs[map_idx]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_alpha_lt_2": float(p_lt_2),
        "n_eff": float(n_eff),
        "alpha_std": float(jnp.sqrt(jnp.maximum(total_var, 0.0))),
        "within_window_var": float(within_var),
        "across_window_var": float(jnp.maximum(across_var, 0.0)),
        "xmin_candidates": eigs,
        "xmin_weights": weights,
    }


# --- Phase 3: Bayesian model comparison (power-law vs exponential vs lognormal) --

def _logev_powerlaw(eigs, xmin, a0: float, b0: float):
    """Log marginal likelihood of the tail lambda >= xmin under a Pareto model
    with beta = alpha - 1 ~ Gamma(a0, b0), integrated analytically.
    Includes the full prior normaliser (needed for cross-model comparison)."""
    n, S = _tail_stats(eigs, xmin)
    mask = eigs >= xmin
    sum_log_lam = jnp.sum(jnp.where(mask, jnp.log(eigs), 0.0))
    return (-sum_log_lam
            + a0 * jnp.log(b0) - gammaln(a0)
            + gammaln(a0 + n) - (a0 + n) * jnp.log(b0 + S))


def _logev_exponential(eigs, xmin, c0: float, d0: float):
    """Log marginal likelihood of the excesses (lambda - xmin) >= 0 under an
    Exponential(rate) model with rate ~ Gamma(c0, d0)."""
    mask = eigs >= xmin
    n = jnp.sum(mask)
    E = jnp.sum(jnp.where(mask, eigs - xmin, 0.0))
    return (c0 * jnp.log(d0) - gammaln(c0)
            + gammaln(c0 + n) - (c0 + n) * jnp.log(d0 + E))


def _logev_lognormal(eigs, xmin, kappa0: float, a_ig: float, b_ig: float):
    """Log marginal likelihood of the tail under a lognormal model: log(lambda) ~
    Normal(mu, sigma^2) with a Normal-Inverse-Gamma prior (m0 = empirical mean of
    the log-tail, weak kappa0). The -sum log(lambda) Jacobian puts this on the same
    lambda-density footing as the Pareto/exponential evidences so the ratios are valid."""
    mask = eigs >= xmin
    n = jnp.sum(mask)
    y = jnp.where(mask, jnp.log(eigs), 0.0)
    ybar = jnp.sum(y) / jnp.maximum(n, 1)
    m0 = ybar                                            # weakly-informative empirical centre
    sse = jnp.sum(jnp.where(mask, (y - ybar) ** 2, 0.0))
    kappa_n = kappa0 + n
    a_n = a_ig + n / 2.0
    b_n = b_ig + 0.5 * sse + 0.5 * kappa0 * n * (ybar - m0) ** 2 / kappa_n
    sum_log_lam = jnp.sum(y)
    nig = (gammaln(a_n) - gammaln(a_ig)
           + a_ig * jnp.log(b_ig) - a_n * jnp.log(b_n)
           + 0.5 * (jnp.log(kappa0) - jnp.log(kappa_n))
           - (n / 2.0) * jnp.log(2.0 * jnp.pi))
    return -sum_log_lam + nig


def model_posterior(eigs: Float[Array, "k"], xmin: Float[Array, ""] | None = None,
                    a0: float = 1.0, b0: float = 1.0, c0: float = 1.0, d0: float = 1.0,
                    kappa0: float = 1.0, a_ig: float = 1.0, b_ig: float = 1.0) -> dict:
    """Bayesian replacement for fit_distributions' Vuong LRT: analytic marginal
    likelihoods for power-law / exponential / lognormal on the same tail, turned
    into Bayes factors and (equal-prior) posterior model probabilities.

    Defaults are unit-information-scale conjugate priors. NOTE: Bayes-factor
    *magnitudes* are prior-sensitive (Lindley-Bartlett); the SIGN of logbf_pl_vs_exp
    is robust and is what the test suite pins against the frequentist LRT.

    Returns log evidences, posterior probabilities, log Bayes factors (natural log;
    >0 favours power-law), best_model, xmin, tail_size.
    """
    if xmin is None:
        xmin = _ks_select_xmin(eigs)
    n, _ = _tail_stats(eigs, xmin)

    log_pl = _logev_powerlaw(eigs, xmin, a0, b0)
    log_exp = _logev_exponential(eigs, xmin, c0, d0)
    log_ln = _logev_lognormal(eigs, xmin, kappa0, a_ig, b_ig)

    logevs = jnp.stack([log_pl, log_exp, log_ln])
    probs = jax.nn.softmax(logevs)
    names = ["powerlaw", "exponential", "lognormal"]
    best = names[int(jnp.argmax(logevs))]

    return {
        "logev_powerlaw": float(log_pl),
        "logev_exponential": float(log_exp),
        "logev_lognormal": float(log_ln),
        "prob_powerlaw": float(probs[0]),
        "prob_exponential": float(probs[1]),
        "prob_lognormal": float(probs[2]),
        "logbf_pl_vs_exp": float(log_pl - log_exp),   # >0 favours power-law
        "logbf_pl_vs_ln": float(log_pl - log_ln),     # >0 favours power-law
        "best_model": best,
        "xmin": float(xmin),
        "tail_size": int(n),
    }


# --- Phase 4: posterior-predictive check -> Bayesian p-value ------------------

def _ks_exp(t_sorted: Float[Array, "n"], beta: Float[Array, ""]):
    """One-sample KS distance between the empirical CDF of t_sorted (ascending,
    length n) and the model CDF F(t) = 1 - exp(-beta t) of an Exponential(beta)."""
    n = t_sorted.shape[0]
    F = 1.0 - jnp.exp(-beta * t_sorted)
    i = jnp.arange(1, n + 1)
    d_plus = jnp.max(i / n - F)
    d_minus = jnp.max(F - (i - 1) / n)
    return jnp.maximum(d_plus, d_minus)


def ppc_pvalue(eigs: Float[Array, "k"], xmin: Float[Array, ""] | None = None,
               a0: float = 1e-3, b0: float = 1e-3, n_rep: int = 500, key=None) -> dict:
    """Posterior-predictive check for the power-law tail. In t = log(lambda/xmin)
    space the tail is Exponential(beta), beta = alpha - 1, with posterior
    Gamma(a0+n, b0+S). For each posterior draw beta_s we form a realised KS
    discrepancy of the OBSERVED tail and of a REPLICATE tail simulated from the
    same beta_s (the posterior predictive is Lomax), then report the Bayesian
    p-value P(D_rep >= D_obs). p ~ 0.5 => indistinguishable from a power law;
    small p => the power-law tail is a poor description.

    Host-side tail extraction (like bootstrap_alpha_ci) -- this is a one-shot
    diagnostic, not a jitted training path.
    """
    import jax.random as jr
    key = jr.PRNGKey(0) if key is None else key
    if xmin is None:
        xmin = _ks_select_xmin(eigs)
    xmin = jnp.asarray(xmin)

    tail = eigs[eigs >= xmin]
    n = int(tail.shape[0])
    t = jnp.sort(jnp.log(tail / xmin))                   # ascending exceedances
    S = jnp.sum(t)
    post_a = a0 + n
    post_b = b0 + S

    k_beta, k_rep = jr.split(key)
    betas = jr.gamma(k_beta, post_a, shape=(n_rep,)) / post_b   # Gamma(post_a, rate post_b)
    rep_keys = jr.split(k_rep, n_rep)

    def one(beta, rk):
        d_obs = _ks_exp(t, beta)
        t_rep = jnp.sort(jr.exponential(rk, shape=(n,)) / beta)
        d_rep = _ks_exp(t_rep, beta)
        return d_obs, d_rep

    d_obs, d_rep = jax.vmap(one)(betas, rep_keys)
    p_value = jnp.mean(d_rep >= d_obs)

    return {
        "p_value": float(p_value),
        "ks_obs_mean": float(jnp.mean(d_obs)),
        "xmin": float(xmin),
        "tail_size": n,
        "n_rep": n_rep,
    }
