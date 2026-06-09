# Replicating WeightWatcher's canonical test cases with wwjd

A deep-research pass (25/25 claims verified against primary sources: the
arXiv/Nature/JMLR papers, the CalculatedContent WeightWatcher + ww-trends-2020
notebooks, torchvision docs, the PGDL winning repo) catalogued the specific,
publicly-reproducible model families Martin & Mahoney ran WeightWatcher on, and
mapped each to the wwjd Bayesian diagnostic that would most visibly differ from
their published point estimates.

This is the **methods-validation-on-the-originator's-own-examples** section of
the paper: confirm the flagship VGG trend *with error bars*, then show where the
Bayesian treatment nuances or overturns. Every checkpoint is a free download — no
retraining.

## Ranked shortlist

### 1. GPT vs GPT2 — overturn/nuance (HIGHEST payoff)
- **Charles's claim** (Nature Comms 2021): GPT2 is better-trained; GPT has *"numerous
  unusually large α — meaning they are not well-described by a PL fit"*, GPT2-small
  has all α ≤ 6 with smaller mean/median α.
- **wwjd diagnostic**: `model_posterior` PL-vs-lognormal/exponential **Bayes factor**
  per layer + credible intervals on the scale-invariant mean α (α-bar). The "better-
  trained" gap rests on layers the author *admits aren't power laws* — exactly where
  a Bayes factor + posterior could narrow or reverse it.
- **Reproducibility**: HF `openai-community/openai-gpt` (OpenAIGPTModel) and
  `gpt2` / `gpt2-medium` / `gpt2-large` / `gpt2-xl` (GPT2Model). The ww-trends-2020
  notebook only ever prints α *histograms*, never per-model α numerics — so wwjd
  posteriors add genuinely new information.
- **Payoff**: could formalize, narrow, or reverse the canonical NLP claim.

### 2. DenseNet α~8 layers — overturn
- **Charles's claim** (Nature Comms "Correlation Flow"): DenseNet α rises almost
  immediately and ranges to **~8.0**; he concedes *"even a PL model is probably a
  poor fit ... DenseNet has too many connections."*
- **wwjd diagnostic**: `ppc_pvalue` posterior-predictive goodness-of-fit + `model_posterior`
  — formally reject the power law where he rejected it by eye.
- **Reproducibility**: torchvision `densenet121/161/169/201`, `*_Weights.IMAGENET1K`.
- **Payoff**: clearest case to formally show the PL assumption fails on a famous model.

### 3. PGDL contest Simpson's paradox — robustness test
- **Charles's claim** (Post-mortem 2021): on the NeurIPS 2020 PGDL corpora (Task1 =
  96 VGG-like/CIFAR10 in 4 depth-subgroups of 24; Task2 = 54 stacked-Dense/SVHN in
  3 subgroups of 18), α anti-correlates with quality *within* fixed-depth subgroups
  but the trend *reverses* when depths are aggregated.
- **wwjd diagnostic**: `hierarchical_alpha` NumPyro cross-model **population posterior**
  — does the within-subgroup anti-correlation survive credible intervals? The paper
  itself admits some subgroups are only "modestly to weakly" anti-correlated.
- **Reproducibility**: `parthnatekar/pgdl` (winning Team Interpex solution) + CodaLab
  competition models. Moderate effort (download the corpus).
- **Payoff**: could show the headline reversal is partly a point-estimate artifact.

### 4. VGG α̂ depth/quality trend — confirm (ANCHOR / positive control)
- **Charles's claim** (HT-SR 2019): α̂ = Σₗ αₗ·log λ_max,ₗ has a *near-perfect linear*
  relation to ImageNet test accuracy across VGG11/13/16/19 (±BN); VGG13/VGG13_BN are
  the only log-norm outliers, removed by α̂.
- **wwjd diagnostic**: propagate per-layer **credible intervals** into α̂ → put error
  bars on the metric the whole literature ranks models by; `hierarchical` mean α.
- **Reproducibility**: torchvision `vgg11/13/16/19` (+`_bn`), `*_Weights.IMAGENET1K`.
- **Payoff**: strongest published claim, most likely to hold — "wwjd reproduces the
  flagship result *with* error bars" is the validation baseline.

### 5. SETOL α=2 "Ideal" layers — nuance (ties to our RG-robustness paper)
- **Charles's claim** (SETOL 2025): a layer is "Ideal" when its ESD fits a PL with
  α=2 (claimed universal), equivalent to an ERG trace-log condition; validated on a
  controlled 3-layer MLP + SOTA nets.
- **wwjd diagnostic**: do "Ideal" layers' α **posteriors actually contain 2**, or are
  they merely point-close? The 3-layer MLP is small-sample — the regime where a
  posterior matters most.
- **Reproducibility**: MLP constructable; SOTA nets via torchvision. Recent (Jul 2025),
  not yet independently replicated — timely to engage.
- **Payoff**: directly tests the α=2 falsifiable target with uncertainty; connects to
  our own `paper/wwj_rg_robustness`.

## Methodology hooks (why wwjd, in the authors' own words)
WeightWatcher fits α by MLE over `[xmin, λ_max]` with a single KS-optimal `xmin`.
The papers themselves flag every wwjd hook: the MLE "works very well for α ∈ (2,4);
adequate, although imprecise, for smaller and especially larger α"; some KS-distance-
vs-xmin plots "have less of a well-defined minimum" (ambiguous xmin → BMA); and the
tool ships PL/TPL/E_TPL options by hand because "PL fits over-estimate α" (→ Bayesian
model comparison). wwjd replaces all three with posteriors.

## Implementation
`benchmarks/ww_replication.py` — loads these checkpoints, runs both the frequentist
`wwj.analyze_matrix(mode="csn")` and the Bayesian `wwj.bayes_analyze_matrix`, and emits
per-layer + per-model comparison tables. Start with #1 (GPT/GPT2) + #4 (VGG): the
confirm-then-nuance arc, both trivial loads.

## Open questions (answered by running the survey)
- Per-layer α point estimates + eigenvalue sample sizes for GPT early layers and
  DenseNet α~8 layers → expected posterior width + Bayes-factor power.
- Does PGDL within-subgroup anti-correlation survive a hierarchical posterior?
- Do SETOL "Ideal" layers' α posteriors contain 2?
