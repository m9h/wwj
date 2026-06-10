# Paper restructure plan (2026-06-09)

## From → To

**Current** (`wwj_rg_robustness.tex`): a 4-page ICBINB workshop paper, "When Does α=2
Matter? Empirical Validation of the RG-Robustness Prediction for Neuroimaging FMs."
Scope = frequentist wwj + 3 neuroimaging experiments. Does NOT contain the Bayesian
extension, the WeightWatcher-replication results, or the cross-modality / pathology work.

**Target**: a longer paper whose backbone is the **Bayesian WeightWatcher (wwj/wwjd)**
and whose thesis is broader than α=2:

> Point-estimate heavy-tailed spectral diagnostics (WeightWatcher's α) can mislead —
> the single KS-x_min estimate inflates α on hard-to-fit layers and never tests whether
> a layer is a power law at all. A calibrated Bayesian treatment (posteriors over α, BMA
> over x_min, model comparison incl. TPL/GPD, posterior-predictive checks, hierarchical
> pooling) changes published conclusions. We demonstrate this on WeightWatcher's own
> canonical examples and then apply it across modalities — where the RG α=2 prediction
> is confirmed in some settings and falsified in others.

## Candidate titles
1. "A Bayesian WeightWatcher: Calibrated Heavy-Tailed Spectral Diagnostics Across Modalities"
2. "When Does α=2 Matter? A Bayesian Reanalysis of Heavy-Tailed Self-Regularization"
3. "Putting Error Bars on WeightWatcher: Bayesian HTSR Diagnostics and What They Overturn"

## Section map

1. **Introduction** — HTSR/α, the RG α=2 prediction, the two gaps (point-estimate fragility
   + untested power-law assumption), and the cross-modality question. State the three things
   the Bayesian treatment changes.
2. **Method: wwj + wwjd** — frequentist core (CSN x_min, differentiable Hill α, Vuong LRT)
   AND the Bayesian layer (conjugate Gamma-Pareto α posterior, BMA over x_min, model
   comparison PL/exp/lognormal/TPL/GPD via Laplace, posterior-predictive p-value,
   hierarchical population posterior, the EIV accuracy~α̂ regression, ROPE, prior-sensitivity).
3. **Validation on WeightWatcher's canonical cases** ← NEW, strongest new section. The
   "reanalyze the originator's own examples" results: VGG α̂ confirm (sharper under Bayes),
   GPT/GPT2 dissolution, BatchNorm breaks the PL, the TPL/GPD mechanism, the EIV calibrated
   predictor, ROPE + prior-sensitivity robustness. All on public checkpoints, reproducible.
4. **Cross-modality application** — the RG α=2 / robustness thread, positive AND negative:
   - neuroimaging (CURRENT paper's Exp 1-3): survey, WAND cohort, MLP α_loss disentangle.
   - pathology (nanopath): the FM-scale α_loss NULL (α_loss neutral, robustness flat at
     ViT-S scale) — the honest "when α=2 does NOT transfer" counterpoint.
   - fMRI (peer_fmri survey), + EEG / sMRI / invasive AS DATA ALLOWS (see inventory).
5. **Discussion** — what the Bayesian treatment overturns; where α=2 holds vs not; limits.
6. **Supplementary** — implementation, hyperparameters, per-layer distributions, full tables.

## Evidence inventory (solid vs aspirational — be honest in the paper)

SOLID (run, in hand):
- wwj + wwjd methods + full test suite (43 passing).
- WW-replication: VGG (r −0.77 freq / −0.98 bayes), GPT/GPT2 (4.12→2.90, gap 3× shrink),
  BN PL-rejection 8–17%, TPL beats PL 50/50, EIV slope −6.21 [−7.76,−4.84] P=1.0, ROPE 1/50,
  prior-sens 0.001. Data: /data/mhough/wwj_ww_replication.
- Neuroimaging Exp 1-3 (current paper).
- Pathology FM-scale α_loss null (nanopath Arm C).

PARTIAL / NEEDS A RUN:
- peer_path_fm_survey (15 H&E FMs) — scaffolded, smoke only.
- peer_fmri_fm_survey (CortexMAE + 6 fMRI FMs) — exists, re-run/tabulate.
- GPT-2 fine-tuning axis (DialoGPT drop-in; or train GPT-2-124M-neuro) — see BrainGPT note.

ASPIRATIONAL (need checkpoints pointed at the diagnostics):
- EEG, sMRI, invasive modality FM surveys — user has the FMs; point wwjd at them.

## Open framing decisions for the user
- Title / venue / length (workshop 4pp vs full paper)?
- Is the WW-replication the lead result (methods paper) or a validation section before the
  cross-modality application (application paper)? The data is strongest for the former.
- Which modalities make the cut for v1 vs a follow-up?

## Drafted so far
- `sec_ww_replication.tex` — the WeightWatcher-replication results section (Section 3), ready
  to drop in once the structure is approved.
