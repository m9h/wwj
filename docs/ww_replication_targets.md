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

## RESULTS (2026-06-09, benchmarks/ww_replication.py, torchvision VGG + HF GPT/GPT2)

Three findings, all on Martin & Mahoney's own published checkpoints:

**1. GPT/GPT2 dissolves under xmin marginalization (overturn — the headline).**
The "GPT poorly-trained, GPT2 well-trained" claim rests on GPT's frequentist
mean α = 4.12 ("numerous unusually large α"). Under the BMA posterior over xmin,
**GPT sits at 2.90 — the same healthy band as GPT2 (2.73) and GPT2-medium (2.75).**
The frequentist gap (GPT−GPT2 = 0.50) shrinks ~3× to 0.16. Per layer, GPT's
768×768 attention projections drop from freq α 5.2–6.9 to bayes α 2.8–3.5, and
the Bayes factor calls every one `powerlaw` (P=1.0) — contra "not well-described
by a PL fit." Validated stable across the zero-eigenvalue bug fix (2.897 before
and after), so it is not a numerical artifact. Mechanism = Charles's own conceded
"plain PL fits over-estimate α" (the reason WW ships TPL): the single KS-xmin
inflates α on hard-to-fit layers; marginalizing the window corrects it.

**2. VGG α̂ confirm — and the Bayesian version is SHARPER (confirm + bonus).**
Weighted-alpha α̂ = mean_l(α_l · log10 λmax,l) vs ImageNet top-1, Pearson r:
frequentist −0.772 (reproduces Charles's "smaller α̂ → higher accuracy"),
**Bayesian −0.982 (near-perfect — stronger than frequentist).** The BMA correction
that pulls down inflated large-α layers makes the metric MORE predictive.
NB the metric must be the MEAN over layers; a SUM gives +0.78 (depth confound),
and the unweighted mean α gives +0.89 (the Simpson's paradox the post-mortem
paper warns about) — wwjd reproduces both the right answer and the trap.

**3. BatchNorm breaks the power law (new observation).**
BN-VGG variants have 8–17% of layers Bayes-factor-rejected as non-power-law,
rising with depth (vgg19_bn 17%); plain VGG and all GPT models are 0%. The
Bayesian model comparison (model_posterior) sees structure the point-α is blind to.

Data: /data/mhough/wwj_ww_replication/ (per-layer CSVs + summary). A zero-eigenvalue
BMA NaN bug was caught + fixed on VGG16 during this run (commit 76380dd).

## Bayesian-workflow diagnostics applied (2026-06-09, benchmarks/ww_bayes_diagnostics.py)

**#1 EIV regression — VGG α̂ is a CALIBRATED predictor of accuracy.** Errors-in-variables
regression of ImageNet top-1 on α̂ (Bayesian α̂ + propagated per-model SE): posterior
slope **−6.21 pp per unit α̂, 95% CI [−7.76, −4.84], P(slope<0)=1.000**, residual σ=0.31pp,
R-hat 1.0. The measurement-error slope (−6.21) is slightly steeper than naive OLS (−6.04) —
attenuation-corrected as expected. So the weights-only spectrum predicts ImageNet top-1 to
within ~0.3 points, with the HT-SR direction certain. Turns Charles's point correlation into
a predictive instrument with honest uncertainty.

**#2 TPL/GPD — the GPT "large α" is power-law-WITH-truncation mis-specification.** Adding the
truncated power law (WW's own TPL) and the generalized Pareto (EVT) to the comparison: across
all 50 openai-gpt layers, **the plain power law loses to the truncated power law 50/50**
(mean logBF PL-vs-TPL = −1.54), and the 5-way best model is GPD (35) or TPL (15) — never plain
PL. So the layers ARE heavy-tailed (PL beats exp/lognormal in the 3-way), but a *truncated*
power law fits better — which is exactly why plain-PL α is over-estimated (the finite-size
cutoff WW's TPL option exists for). This is the mechanism behind the freq-4.1 → bayes-2.9
correction, made explicit.

**#6 ROPE — but GPT is still ABOVE the RG optimum.** Only 1/50 GPT layers have posterior mass
P(α ∈ [1.75,2.25]) > 0.5 (mean 0.023). So the Bayesian α≈2.9 is much lower than the freq 4.1
but still not at the SETOL-ideal α=2 — honest nuance, not an over-correction.

**#5 Prior sensitivity — conclusions are not prior-driven.** Mean power-scaling sensitivity
across GPT layers = 0.001 posterior-SD units (≈0). The findings are robust to the prior.

## Open questions (next runs)
- DenseNet α~8 layers (#2 shortlist) — the clearest formal PL-rejection target, untested.
- PGDL within-subgroup anti-correlation under a hierarchical posterior (#3 shortlist).
- SETOL "Ideal" layers' α posteriors: do they contain 2? (#5 shortlist).
- ppc_pvalue + per-layer Bayes-factor detail on the GPT large-α layers (in progress).
