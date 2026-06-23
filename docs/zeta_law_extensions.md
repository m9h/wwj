# A Bayesian, falsifiable Zeta Law — extending Thompson with wwj/wwjd

*Compiled 2026-06-22. Companion to [`data_vs_weight_spectra.md`](data_vs_weight_spectra.md).*

Thompson, *"The Zeta Law of Discoverability in Biomedical Data"* (arXiv:2604.17581), predicts data
sufficiency from the spectral structure of the data covariance. It is **theory-only** — no datasets, no
data/code-availability statement, empirical validation explicitly deferred to future work. So the
experiments below are *first-of-kind tests*, and wwj/wwjd supplies both the rigor and the data.

His law: AUC accumulates cumulative SNR over spectral modes; under power-law **covariance decay** and
power-law **signal alignment**, the partial sums become a `ζ`-function and yield a scaling law. The
pole of `ζ` at **alignment exponent β = 1** marks the regime where data is fundamentally insufficient
(β ≤ 1: diffuse signal across many modes → diminishing returns; the paper's own advice is to *improve
representations rather than collect more data*).

## The headline contribution: what wwjd's Bayesian layer adds

Thompson's law is **deterministic and assumes power-law spectra**. `wwjd` turns it into a *validated,
uncertainty-quantified, decision-ready* tool:

1. **Validity gate (model comparison) — load-bearing.** The `ζ` structure exists only if the spectra
   are power-law. `model_posterior` (Bayes factors: power-law vs lognormal vs exponential) + `ppc`
   *tests* that per modality. If a spectrum is lognormal, the zeta law does not apply — wwjd says so,
   with an evidence ratio. It is the precondition Thompson never checks.
2. **Posterior over the answer, not a point.** Calibrated posteriors over the exponents propagate to a
   **credible band over AUC(N) and N\***. At HBN scale the exponents are uncertain, so the point `N*`
   is fragile; the band is the honest "how much data is enough."
3. **BMA over nuisances.** Marginalize the scaling-window `xmin` (`alpha_posterior_bma`) and the tail
   model — removing researcher degrees of freedom point estimates bury.
4. **Hierarchical pooling.** `hierarchical_alpha` partially pools exponents across modalities / sites /
   disorders — the fix for HBN's noisy small-N per-(modality, target) estimates.
5. **Decision-relevant probabilities.** `P(β < 1)` is a Gamma CDF — "are we resolution-limited (fix
   the encoder) or variance-limited (collect more)?" with calibrated confidence — plus value-of-
   information for ΔN.
6. **MaxEnt grounding.** A power-law tail is max-entropy given a log-scale constraint; the zeta law's
   emergence is a Jaynes statement (`wwjd` = *what would Jaynes do*) — on-theme for MaxEnt 2027.

They compose: (1) licenses (2); (3) cleans the estimate; (4) rescues small-N; (5) consumes the
posterior; (6) grounds it. The framing: *a Bayesian, falsifiable zeta law for data sufficiency.*

## The technical bridge: wwj's α **is** Thompson's exponents (one conversion)

wwj fits the eigenvalue **density** tail `p(λ) ∝ λ^{−α}`; Thompson uses the **rank** decay
`λ_n ∝ n^{−s}`. They are the same content in two coordinates:

> **s = 1 / (α − 1)**     (density-tail α ⇔ rank-decay s)

- On the **covariance** ESD → Thompson's `γ ≡ s_cov = 1/(α_cov − 1)`.
- On the **alignment** spectrum (target projected onto data eigenmodes, energy per mode) →
  Thompson's `β = 1/(α_align − 1)`.

Elegant consequence — the criticality point transfers exactly:

> **β = 1  ⇔  α_align = 2.**

So the data-sufficient regime `β > 1` is exactly `α_align < 2`, and **wwjd's `p_alpha_lt_2` evaluated on
the alignment spectrum is `P(β > 1) = P(data-sufficient regime)`.** The same `α = 2` HT-SR critical
point we track for *weights* is, on the *data alignment*, Thompson's `ζ`-pole. The whole wwjd machinery
(conjugate posterior, BMA, model comparison, hierarchical) applies verbatim — just point it at the data
covariance and the alignment spectrum instead of `WᵀW`.

## HBN experiment protocol

Multimodal HBN (sMRI, dMRI, rs/naturalistic fMRI, high-density EEG + dimensional phenotyping) is the
testbed: open, multimodal, already in `benchmarks/hbn_full_volume.py`, with matched encoders
(`smri-fm`, `eeg-fm-spectral`).

- **E1 — validity + γ.** Per modality: feature covariance ESD → `fit_distributions` + `model_posterior`
  (power-law? vs lognormal/exp) and `γ = 1/(α_cov − 1)`. *The precondition test.*
- **E2 — β, the answer + calibration.** Alignment spectrum → `alpha_posterior` → posterior on β and
  `P(β>1)`; propagate (γ, β) through the zeta/cumulative-SNR formula → credible band over AUC(N), N\*;
  **validate by subsampling HBN** at `N ∈ {250, 500, 1k, 2k, full}` and checking band calibration.
- **E4 — cross-over.** Cross-modal operator (cross-covariance) spectrum → predict the N where fused
  beats best-single; test vs measured multimodal AUC(N).
- **E6 — binding constraint.** Per (modality, target): `β > 1` (variance-limited → collect data) vs
  `β ≤ 1` (resolution-limited → improve encoder), reported as `P(β<1)`.
- **Targets:** age (positive control — concentrated signal, expect β>1, fast saturation) + a
  psychopathology dimension (stress case, expect β≤1).
- **Caveat:** HBN complete-multimodal N is modest (~1–3k); some (modality, target) pairs will be
  under-powered to call the regime — `hierarchical_alpha` (pooling) + the credible band report that
  honestly. State which pairs are conclusive.

## Where it starts (this commit)

`benchmarks/zeta_law/zeta_diagnostic.py` — the data-side diagnostic:
`covariance_spectrum` / `alignment_spectrum` (from a feature matrix `X`, target `y`) → wwj
`fit_distributions` / `alpha_posterior` / `model_posterior`, with the density-α → rank-s/β conversion
and `P(β>1)` from `p_alpha_lt_2`. Ships with a **synthetic validation** (plant `s`, `β` → recover them;
confirm the validity gate distinguishes power-law from lognormal). Point it at HBN features next (E1).
The alignment energy uses the covariance-deconfounded **explained-variance** normalization
`coeffᵢ²/λᵢ` (ridge-regularized); the full Canatar–Pehlevan kernel-target normalization is the
further refinement.

## First HBN E1 run

Run on the cached HBN volume-FM embeddings (`neurostorm`, `swift`) via `benchmarks/zeta_law/hbn_e1.py`:

- **Covariance side (the E1 precondition):** *both* modalities have genuinely **power-law** covariance
  spectra — validity gate passes (logBF(pl/exp) ≫ 0, PPC p > 0.5) — so Thompson's premise *holds* here,
  with `γ ≈ 1.55` (neurostorm) and `2.26` (swift). This is the first empirical check that the zeta law
  even applies to real biomedical feature spectra.
- **Alignment side:** the first pass used raw `coeffᵢ²`, which is **confounded by the covariance** and
  reported *every* target as β>1; the deconfounded `coeffᵢ²/λᵢ` fixes it. Result: **age** (positive
  control) is firmly variance-limited (β≈1.7 / 2.8), while several **psychopathology factors** drop
  toward the ζ-pole — neurostorm attention β≈1.11, internalizing ≈1.18, p_factor ≈1.24
  (`P(β>1)` 0.82–0.98) — i.e. near the **resolution-limited boundary**, consistent with their being
  harder, more diffuse targets (age/externalizing stay clearly variance-limited). None drop
  conclusively below 1, so E2's subsampling learning-curve + credible band is needed to call the regime.

*The real-data value of the gate:* the raw-`coeffᵢ²` proxy passed on synthetic data but was exposed as
confounded only on real HBN — which is exactly why E1 measures real spectra instead of assuming the
law's form.

## First HBN E2 run

E2 (`benchmarks/zeta_law/e2_learning_curve.py`) turns the exponents into a predicted learning curve and
`n*` (samples for 90% of asymptotic performance), per target, on `neurostorm` (n=645):

| target | β (med) | P(no-sat)=P(β≤1) | n*(0.9) | read |
|---|---|---|---|---|
| externalizing | 1.93 | 0.00 | ~15k | variance-limited → ~15k subjects reaches 90% (HBN ≪ this ⇒ more data helps) |
| age | 1.65 | 0.00 | ~5M | saturates, but data-hungry for this embedding |
| p_factor | 1.24 | 0.01 | ∞ (>1e12) | near pole → effectively resolution-limited |
| internalizing | 1.18 | 0.07 | ∞ | near pole → resolution-limited |
| attention | 1.11 | 0.19 | ∞ | at the pole → 19% posterior prob of no saturation; improve the encoder, not N |

Actionable per-target verdict: collect more subjects for **externalizing** (~15k); for the harder
factors (**attention / internalizing / p_factor**, β≈1.1–1.2) scaling past HBN won't realistically help
— invest in the representation. *Caveats:* n=645 here is small ⇒ wide posteriors on β near 1; `n*` is a
zeta-law **extrapolation** pending the observed-curve calibration below.

### Calibration (first run — `e2_calibrate.py`)

Over the cache (`neurostorm`, `swift`; n=645, ~0.6 decade of N — pure-numpy ridge CV, no sklearn). The
run is honest about its own limits:

- **Only `age` carries predictable signal** (r@max 0.68 / 0.56). The **psychopathology factors are
  ~unpredictable from these fMRI-volume embeddings** (r@max ≈ 0), so their β/n* are **vacuous** — you
  cannot measure signal-alignment decay where there is no signal. *β is only interpretable when r@max
  is non-trivial.*
- For **age**, the observed-curve shape **weakly tracks the prediction**: `swift` (steep β=2.76 → fast
  saturation) decelerates (obs_inc=0.35), `neurostorm` (shallow β=1.64) still accelerates (obs_inc=3.0)
  — a tentative calibration hit on the one signal-bearing target.
- **Under-powered**: ~0.6 decade of N; n* extrapolations span 1e2–1e8 and are not yet trustworthy.

Verdict: the framework runs end-to-end on real HBN, but conclusive calibration needs (a) the
full/larger cohort for N-range, and (b) embeddings/targets that actually carry signal. The latter
motivates the **cross-modal (E4) EEG↔sMRI** direction on the harmonized HBN cohort, where a
shared structure–function subspace may carry more than the fMRI-volume embeddings do for these factors.

### E4 cross-modal spectrum (core built)

`benchmarks/zeta_law/e4_cross_modal.py` (+ `test_e4.py`, green) — the EEG↔structural "relationship" as
the canonical-correlation spectrum of the whitened cross-covariance (ρ₁≥ρ₂≥…∈[0,1]); strong ρ = shared
modes (structure–function coupling). Crucially it takes `covariate=age` to **residualize out the
age-driven / volume-conduction term** and reveal coupling beyond conduction. Synthetic demo: a 3-mode
shared subspace with mode-0 age-driven → full spectrum ρ≈[1,1,0.99] (3 strong), age-residualized
ρ≈[1,1,0.22] (2 strong) — the conduction mode drops out. Consumes any two subject-aligned feature
matrices (REVE EEG embeddings ⊕ the emeg-fm tier-2 structural embedding). Completes the spectral trio:
within-modality covariance γ (E1) · target alignment β (E2) · cross-modal coupling (E4).

## References
- Thompson (2026), arXiv:2604.17581 — the zeta law (data-spectrum, theory-only).
- Martin & Mahoney — HT-SR / WeightWatcher (weight-spectrum α; the `α=2` critical point).
- Bahri, Dyer, Kaplan, Lee, Sharma (PNAS 2024) — *Explaining neural scaling laws* (variance- vs
  resolution-limited regimes from data + target spectra).
- Canatar, Bordelon, Pehlevan (Nat. Commun. 2021) — spectral bias & task–model alignment (the rigorous
  β / kernel-target-alignment formalism).
- Saxe, McClelland, Ganguli (2014) — deep-linear bridge (weight SVD tracks data SVD).
