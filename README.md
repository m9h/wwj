# wwj

JAX/Equinox port of [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher), targeting the Kidger SciML stack.

Per-layer spectral diagnostics (HTSR α via Clauset-Shalizi-Newman MLE + KS window; trap counts; trace-log alignment) plus a **differentiable α→2 regularizer** for RG-guided optimization — the explicit alternative to Muon's indirect spectral shaping per Martin's *Renormalization Group Theory of Learning* (arXiv 2026).

On top of the frequentist core, `wwj` adds a **Bayesian layer** (`wwjd`): calibrated posteriors over α (closed-form Gamma–Pareto conjugacy, no MCMC), Bayesian model comparison (power-law vs exponential vs log-normal), posterior-predictive checks, and a NumPyro cross-layer hierarchical posterior on the population mean α. See [Bayesian extension](#bayesian-extension-wwjd).

## What this is

- A small JAX module that operates on weight matrices extracted from any model (Equinox-native, or via numpy-bridged from PyTorch / TF / Flax).
- Two analysis modes:
  - `csn` — Clauset-Shalizi-Newman MLE + KS scaling-window selection. Matches the reference Python WeightWatcher. Non-differentiable.
  - `hill` — Closed-form Hill estimator on a fixed window. Fully differentiable. Use this in `alpha_loss` as a training-time regularizer.
- A `MultiOptimizer`-friendly `alpha_loss(model, target=2.0)` that drops into any optax pipeline.

## Status

MVP. Validated against the CalculatedContent WeightWatcher fork as numerical oracle on representative weight matrices from four projects (smri-fm, nanopath, eeg-fm-spectral, hippy-feat). See `benchmarks/compare_projects.py`.

## Workshop paper

[`paper/wwj_rg_robustness.pdf`](paper/wwj_rg_robustness.pdf) — ICBINB-style workshop draft of the RG-robustness experiments. Tests Martin's RG-theory prediction that α ≈ 2 yields better robustness using `wwj` to estimate α and `wwj.alpha_loss` as the interventional handle. See [`paper/README.md`](paper/README.md) for the rebuild instructions.

Known rough edges are listed at the bottom of `src/wwj/core.py` — fix before any published comparison.

## JAX-exact features beyond the Python WW

Wherever the reference WeightWatcher takes a shortcut for compute reasons, wwj does the exact version because JAX's JIT+vmap makes it free:

- **Adaptive xmin search** (`_ks_select_xmin`) — every eigenvalue as a candidate, vmap-parallel KS distance; matches Clauset-Shalizi-Newman 2009 / Alstott-Bullmore-Plenz 2014 algorithmically rather than via the log-spaced grid the Python WW falls back to.
- **`bootstrap_alpha_ci(eigs, n_bootstrap=1000)`** — vmap-parallel bootstrap confidence interval for α. Reference WW supports this but it's slow (serial numpy resampling); here it's one `jax.vmap` call.
- **`fit_distributions(eigs)`** — fits power-law + exponential + lognormal MLEs in a single jit pass; returns Vuong's LRT (positive = power-law preferred). The reference WW loops through distributions one-at-a-time via the `powerlaw` package; we do them in parallel. This is the standard "is this layer actually a power-law?" validation that HTSR α claims require.
- **`alpha_loss(model, target=2.0)`** — fully-differentiable α→2 regularizer via the Hill estimator; plug into any optax loss. This is the explicit alternative to Muon's indirect spectral shaping per Martin's RG theory.

## Bayesian extension (`wwjd` — *what would Jaynes do*)

`wwjd` is the Bayesian layer: *what would Jaynes do* with a heavy-tailed weight spectrum? Headed to [MaxEnt 2027 — the 45th International Workshop on Bayesian Inference and Maximum Entropy Methods in Science and Engineering](https://indico.dzastro.de/event/12/) (Görlitz, Sep 20–24 2027), the workshop series built on E. T. Jaynes' maximum-entropy / Bayesian-probability lineage.

**Reading — Jaynes, *Probability Theory: The Logic of Science*** (Cambridge UP, 2003; ed. G. L. Bretthorst):
- **Full manuscript PDF** (the source) from the official WUSTL Bayes repository: [bayes.wustl.edu/etj/prob/book.pdf](https://bayes.wustl.edu/etj/prob/book.pdf). Per-chapter PostScript/PDF + historical notes at the [repository root](https://bayes.wustl.edu/etj/prob.html).
- **Companion index + errata** (Arnold Baise) — searchable name/subject index and a running corrections file for the 2005 reprint, plus reviews: [etjaynesinfo.com](http://www.etjaynesinfo.com/). (The published edition has a famously sparse index and a number of typos.)
- **Open access**: the published 2003/2004 Cambridge edition is borrowable/streamable on the [Internet Archive](https://archive.org/search?query=Jaynes+Probability+Theory+The+Logic+of+Science).

Where the core reports frequentist *point* estimates of α, the Bayesian layer reports *calibrated posteriors*. Most of it is closed-form: the tail above `xmin` is Pareto, the substitution `t = log(λ/xmin)` makes it Exponential with rate `β = α − 1`, and a conjugate `Gamma(a₀, b₀)` prior on β gives an exact `Gamma(a₀+n, b₀+Σt)` posterior — so credible intervals are Gamma quantiles, `P(α<2)` is a Gamma CDF, and it's all differentiable. Only the cross-layer hierarchical model samples (NumPyro). Design notes: [`docs/wwjd_plan.md`](docs/wwjd_plan.md).

| Frequentist (`core`) | Bayesian (`bayes` / `hierarchical`) | Method |
|---|---|---|
| `_csn_mle_alpha` point estimate | `alpha_posterior(eigs)` → mean / MAP / CI / `P(α<2)` | conjugate Gamma, closed form |
| `bootstrap_alpha_ci` (resampling) | `alpha_posterior(...)` credible interval | exact Gamma quantiles |
| `_ks_select_xmin` (single window) | `alpha_posterior_bma(eigs)` | evidence-weighted BMA over `xmin` |
| `fit_distributions` → Vuong LRT sign | `model_posterior(eigs)` → Bayes factors + posterior model probs | analytic evidence (3 dists) |
| bare KS distance | `ppc_pvalue(eigs)` | Lomax posterior-predictive p-value |
| `summary` over point αs | `hierarchical_alpha(layers)` | NumPyro SVI/NUTS population posterior |
| `alpha_loss` (Hill point) | `bayes_alpha_loss(model, var_weight=...)` | differentiable through the Gamma posterior |

```python
import wwj

# Per-matrix Bayesian record (BMA α posterior + Bayesian model comparison):
stats = wwj.bayes_analyze(model)          # list[BayesLayerStats]
print(wwj.bayes_summary(stats))           # population posterior diagnostics
for s in stats:
    print(f"{s.name}: α={s.alpha_mean:.2f} [{s.ci_low:.2f},{s.ci_high:.2f}] "
          f"P(α<2)={s.p_alpha_lt_2:.2f}  best={s.best_model} ({s.prob_powerlaw:.2f})")

# Posterior-aware regularizer (penalizes posterior mean toward 2, optionally its variance):
loss = task_loss + wwj.bayes_alpha_loss(model, target=2.0, var_weight=0.1, weight=0.01)

# Cross-layer hierarchical population posterior on mean α (needs `uv sync --extra bayes`):
hier = wwj.hierarchical_analyze(model)    # μ_pop credible interval, |μ−2|, R-hat/ESS
```

Single-matrix posteriors (PyTorch interop):
```python
from wwj.core import _eigvals
eigs = _eigvals(jnp.asarray(W.numpy()))
post = wwj.alpha_posterior(eigs)          # {alpha_mean, alpha_map, ci_low, ci_high, p_alpha_lt_2, ...}
mp   = wwj.model_posterior(eigs)          # {best_model, prob_powerlaw, logbf_pl_vs_exp, ...}
```

## Install

```bash
uv sync                       # core (JAX/Equinox) only
uv sync --extra bayes         # + numpyro, for the hierarchical population model
uv sync --extra benchmarks    # + torch/pandas/HF/timm, for the cross-FM surveys
uv pip install -e /home/mhough/dev/weightwatcher    # the WW fork as oracle (test extra)
uv run --extra test --extra bayes pytest -q          # full suite
```

The conjugate per-layer Bayesian surfaces (`alpha_posterior`, `model_posterior`, `bayes_analyze`, `bayes_alpha_loss`) are pure JAX and need no extra. Only `hierarchical_alpha` / `hierarchical_analyze` require the `bayes` extra (NumPyro).

## Usage

```python
import equinox as eqx
import wwj

model = MyEquinoxModule(...)
stats = wwj.analyze(model, mode="csn")
print(wwj.summary(stats))

# As a regularizer in an optax training loop:
def loss_fn(model, batch):
    return task_loss(model, batch) + wwj.alpha_loss(model, target=2.0, weight=0.01)
```

For PyTorch interop:
```python
import torch, jax.numpy as jnp, wwj
from wwj.core import _eigvals

ckpt = torch.load("path/to/latest.pt", map_location="cpu", weights_only=False)
mats = [(k, jnp.asarray(v.numpy())) for k, v in ckpt["model"].items()
        if v.ndim == 2 and min(v.shape) >= 50]

# Per-layer multi-observable: alpha + bootstrap CI + distribution-selection LRT
for name, W in mats:
    eigs = _eigvals(W)
    boot = wwj.bootstrap_alpha_ci(eigs, n_bootstrap=1000, ci=0.95)
    fit = wwj.fit_distributions(eigs)
    pl_valid = fit["pl_vs_exp_lrt"] > 0 and fit["pl_vs_ln_lrt"] > 0
    print(f"{name}: alpha={boot['alpha']:.2f} CI=[{boot['ci_low']:.2f}, {boot['ci_high']:.2f}] PL valid: {pl_valid}")
```

## License

MIT.
