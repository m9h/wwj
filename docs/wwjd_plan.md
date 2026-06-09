# `wwjd` — Bayesian WeightWatcher

Bayesian extension of `wwj`. Where `wwj` reports frequentist point estimates of the
HT-SR power-law exponent α (CSN MLE + KS window + bootstrap CI + Vuong LRT), `wwjd`
reports **calibrated posteriors**: credible intervals on α, a posterior over the
scaling window `xmin`, Bayesian model comparison via Bayes factors, posterior-predictive
goodness-of-fit, and a cross-layer hierarchical posterior on the population mean α.

## Core idea (why most of this is closed-form)

The tail above `xmin` is Pareto:

    p(λ | α, xmin) = (α−1)/xmin · (λ/xmin)^(−α),   λ ≥ xmin

Substituting `t = log(λ/xmin)` makes the tail **Exponential with rate β = (α−1)**.
CSN's estimator `_csn_mle_alpha = 1 + n / Σ t` is exactly the MLE of that rate.
With a conjugate **Gamma(a₀, b₀) prior on β**, the posterior is **closed-form**:

    β | data ~ Gamma(a₀ + n,  b₀ + Σ t),     α = 1 + β

So credible intervals are exact Gamma quantiles, P(α<2) is an exact Gamma CDF, and
the whole thing is differentiable — no MCMC. Sampling (NumPyro) is needed only for the
cross-layer hierarchical model (Phase 5).

## Frequentist → Bayesian map

| Current (`core.py`)                  | `wwjd`                                   | Method                              |
|--------------------------------------|------------------------------------------|-------------------------------------|
| `_csn_mle_alpha` point estimate      | posterior over α                         | conjugate Gamma, closed form        |
| `bootstrap_alpha_ci` (resampling)    | credible interval                        | exact Gamma quantiles               |
| `_ks_select_xmin` (argmin KS, fixed) | posterior over `xmin` + BMA over α       | evidence × prior over candidates    |
| `fit_distributions` → Vuong LRT sign | Bayes factors / posterior model probs    | analytic evidence (3 dists)         |
| bare KS distance                     | posterior-predictive Bayesian p-value    | Lomax PPC sampling                  |
| `summary` over point αs              | hierarchical population posterior        | NumPyro SVI/NUTS                    |
| `alpha_loss` (Hill point)            | posterior-aware α→2 loss (mean + var)    | diff. through Gamma posterior       |

## Phases (TDD — failing tests first)

- **Phase 1 — `alpha_posterior`** (keystone, no sampling). Conjugate Gamma posterior
  over α given a fixed `xmin`. Returns `alpha_mean`, `alpha_map`, `ci_low/high`,
  `post_a/post_b`, `P(alpha<2)`. Tests: flat-prior large-n → matches `_csn_mle_alpha`;
  CI ≈ `bootstrap_alpha_ci` band on synthetic Pareto.
- **Phase 2 — `xmin_posterior` + BMA.** Marginal likelihood per candidate `xmin`
  (α integrated out → Lomax evidence), prior over candidates → `p(xmin | data)`.
  α posterior becomes a Gamma mixture weighted by it.
- **Phase 3 — `model_posterior`.** Analytic evidence for power-law / exponential /
  lognormal (Gamma & Normal-Inverse-Gamma conjugacy) → Bayes factors + posterior
  model probabilities. Oracle: `log BF(pl vs exp)` sign agrees with `fit_distributions`.
- **Phase 4 — `ppc_pvalue`.** Lomax posterior-predictive sampling; KS/max-gap
  discrepancy vs observed → calibrated Bayesian p-value.
- **Phase 5 — hierarchical** (`hierarchical.py`, NumPyro). `α_layer ~ Normal(μ_pop, τ)`;
  SVI (default) + optional NUTS. Population credible interval on mean α and |mean−2|;
  R-hat/ESS.
- **Phase 6 — surfaces.** `bayes_analyze`, `bayes_summary`, `bayes_alpha_loss`;
  `__init__` exports; `pyproject` optional dep `bayes = ["numpyro>=0.15"]`; README.

## Files

- `src/wwj/bayes.py` — Phases 1–4 + 6 conjugate surfaces (pure JAX).
- `src/wwj/hierarchical.py` — Phase 5 (NumPyro).
- `tests/test_bayes.py`, `tests/test_hierarchical.py`.

## Decisions / risks

- **Priors**: default weakly-informative `a₀ = b₀ = 1e-3` so flat-prior ≈ MLE; expose them.
- **Tail convention**: reuse `core.py`'s masked-tail/`_ks_select_xmin` so results stay
  comparable. The conjugate path uses the exact Pareto likelihood, so `core.py` rough-edge
  #1 (KS empirical-CDF surrogate) does not affect it.
- **Cost**: only Phase 5 samples; SVI keeps it JIT-friendly and cheap.
