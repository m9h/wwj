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
The alignment-energy definition is a first-pass proxy; the rigorous Canatar–Pehlevan kernel-target
normalization is the refinement.

## References
- Thompson (2026), arXiv:2604.17581 — the zeta law (data-spectrum, theory-only).
- Martin & Mahoney — HT-SR / WeightWatcher (weight-spectrum α; the `α=2` critical point).
- Bahri, Dyer, Kaplan, Lee, Sharma (PNAS 2024) — *Explaining neural scaling laws* (variance- vs
  resolution-limited regimes from data + target spectra).
- Canatar, Bordelon, Pehlevan (Nat. Commun. 2021) — spectral bias & task–model alignment (the rigorous
  β / kernel-target-alignment formalism).
- Saxe, McClelland, Ganguli (2014) — deep-linear bridge (weight SVD tracks data SVD).
