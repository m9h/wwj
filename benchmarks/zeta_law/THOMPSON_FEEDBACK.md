# Feedback on the Zeta Law of Discoverability (arXiv:2604.17581)

Empirical + theoretical stress-test of the single-exponent discoverability claim, on synthetic data
and a real neuroimaging cohort (Child Mind Institute HBN, n=645–2500 depending on modality). All code
is in this directory (`wwj` repo, branch `data-side-spectra`); every number below cites the script and
git commit that produced it, so it can be reproduced or checked. Summary first, evidence after.

## Summary verdict

**The single alignment exponent β, as currently defined (rank-decay of the *sorted* alignment
spectrum), does not forecast data sufficiency — falsified directly, in both synthetic and real data.**
The mechanism is identifiable: sorting the alignment spectrum by magnitude discards *which eigenmode*
the target signal occupies, and that position (high-eigenvalue = learnable-with-little-data vs
low-eigenvalue = data-hungry) is exactly what determines sufficiency. A **position-preserving**
statistic computed from the *same* underlying quantity recovers a real, useful signal. The full
parameter-free theory this points to (Canatar–Bordelon–Pehlevan) is validated to high precision in
idealized synthetic settings, but its practical real-data estimator is the weak link: per-mode target
power is not reliably estimable from finite samples, and even after correcting for that, the theory is
only a rough qualitative guide on real neural data — a simple flexible curve fit still predicts
measured learning curves more accurately.

## 1. β does not predict sufficiency — direct falsification

**Method:** plant a frozen representation with known covariance spectrum and a target with known
per-mode signal; measure the *actual* ridge-regression learning curve as N grows; ask whether β (fit
on the full-data alignment spectrum) predicts how much data was needed.

- Single-mode targets at increasing eigenvalue depth (`zeta_thesis_v2.py`, commit `3272e21`): data-to-
  reach-R²=0.7 climbs monotonically 40→344 as the signal moves to deeper eigenmodes, while β is
  *non-monotonic* (0.92→2.05→1.74) and at the easy end is lowest — backwards from what "high β = more
  data needed" would require.
- Across 10 diverse synthetic targets (`zeta_thesis_v3.py`): **Spearman(β, log N-to-reach) = +0.25**
  (no real relationship).
- On real HBN data, 14 (modality × {age, sex}) points (`zeta_realdata_rho.py`): **Spearman(β,
  log N-to-reach) = −0.02** (flat).

**Mechanism:** the deconfounded alignment value per mode, `a_ρ = coeff_ρ² / λ_ρ`, is the right
quantity — but β is extracted by *sorting* `{a_ρ}` by magnitude and fitting the rank-decay. Sorting
erases the eigenvalue index ρ each value came from. Two targets with identical sorted-magnitude
distributions but signal sitting at different eigenvalue depths get the *identical* β, despite needing
very different amounts of data.

## 2. A position-preserving statistic works

Keep the *same* `a_ρ`, but in eigenvalue-rank order (don't sort), and take `ρ_q` = the rank at which
cumulative `a_ρ` reaches a target fraction q (e.g. 0.7) of its mass — "how many top-eigenvalue modes
hold most of the signal."

- Synthetic (`zeta_thesis_v3.py`): **Spearman(ρ_0.7, log N-to-reach) = +0.96**.
- Real HBN (`zeta_realdata_rho.py`): **Spearman(ρ_0.7, log N-to-reach) = +0.53** (n=14, borderline
  significant — suggestive, not airtight, but the ρ_q≫β contrast is unambiguous).

This is the practical recommendation: **a position-preserving cumulative-mass statistic on the
eigenvalue-rank-ordered alignment spectrum, not the sorted rank-decay exponent.**

## 3. The full theory is right in principle; the real-data estimator is the weak link

The natural generalization beyond a single scalar exponent is the Canatar–Bordelon–Pehlevan (CBP)
deterministic-equivalent learning curve, which uses the *full* (eigenvalue, aligned-target-power) pair,
position intact, self-consistently.

- **Validated** against simulated ridge learning curves across an (capacity-exponent, source-exponent,
  SNR, N) grid: parameter-free prediction (zero free parameters — computed directly from the true
  spectra) matches measured curves at **MAE = 0.012, Pearson = 0.998** over 56 test points
  (`zeta_cbp.py`, commit `6447361`).
- **Fails catastrophically on real data** when the per-mode target power is estimated naively as
  `coeff_ρ² / λ_ρ` from finite samples: mean skill (1 − MSE/Var, so 0 = no better than predicting the
  mean) is **−1.89**, i.e. *worse than a flat baseline*, with individual curves as bad as skill −38
  (`zeta_tierB.py`, commit `c32eb44`). Cause: every mode's estimated `coeff_ρ²` carries an additive
  noise floor ≈σ²/n; dividing by small λ_ρ in the tail amplifies this into huge spurious estimated
  power exactly where the eigenvalues are smallest — corrupting the very quantity the theory needs
  most precisely.
- **Robust estimation** (debias the noise floor, truncate the eigenvalue tail below a resolvability
  threshold) recovers a sensible qualitative result: mean skill on high-signal curves rises to **+0.14**,
  and — importantly — **the position-preserving estimate now correctly beats the sorted (β-style)
  version** (+0.14 vs −0.26), confirming the position-matters mechanism *does* show up on real data once
  the estimator is fixed (`zeta_robust.py`, commit `bf62d67`).
- **Even so, it underperforms a simple empirical fit.** Scored fairly — leave-one-N-out cross-validation
  for the 2-parameter flexible power-law, so it isn't allowed to fit the same points it's evaluated on
  (an earlier in-sample comparison inflated its apparent margin, +0.82 vs the honest +0.60/+0.69) — the
  flexible curve still clearly wins: **+0.69 (power-law, fair) vs +0.14 (robust-CBP, zero-parameter)**
  on high-signal real curves (`zeta_fair.py`, commit `67cbcb7`).
- A **multi-seed synthetic check** shows part of the real-data gap is inherent measurement noise in a
  single-realization learning curve (even the *true*-spectrum CBP prediction only reaches skill
  0.39–0.72 against a single noisy realization, vs its near-perfect match to a many-seed-averaged
  curve) — but not all of it; the ceiling-anchoring protocol used here (calibrating the predicted
  asymptote to the largest measured N) has its own real limitations independent of estimation noise.

## 4. Would ENIGMA-scale data change this? — split answer

An obvious objection: our real-cohort tests use HBN (~600–2500 subjects per modality), far below what
ENIGMA-scale consortium access provides. We tested this directly rather than speculate (`zeta_scale.py`,
commit `a198768`) — the answer **splits cleanly by which finding it is**.

**The single-sorted-β falsification is NOT rescued by more data, at any scale.** The synthetic
falsification in Section 1 already estimated the alignment spectrum from 9,000 samples — large by any
standard. It fails for a *structural* reason: sorting the alignment spectrum by magnitude is a
many-to-one map that discards eigenmode position, so two targets with identical sorted-magnitude
distributions but signal at different depths get identical β regardless of how precisely β is
estimated. More data sharpens the estimate of a quantity that doesn't carry the needed information —
it cannot restore what the sorting operation already discarded.

**The full-theory real-data estimation failure (Section 3) plausibly IS a finite-sample artifact, and
the scale needed is far more modest than "10,000×."** Isolating the per-mode target-power estimation
problem at a realistic embedding size (P=500) and sweeping labeled sample size M: the *naive* estimator
is biased-but-consistent (bias ~σ²/(M·λρ), shrinking with M). At HBN-like M=300 it reproduces the
real-data failure pattern (skill as low as −2.86 against the true theoretical curve, in the hardest
tested regime); by **M≈10,000–30,000 — squarely within a single ENIGMA structural-MRI study's realistic
scale — naive-estimator skill converges to +1.00 in every regime tested.** So roughly a **10–30×**
increase in labeled sample size over HBN, not 10,000×, is plausibly sufficient to resolve this
particular failure mode.

**A genuine surprise, and a correction to our own Section 3/Recommendation-3 fix:** the "robust"
debias-and-truncate estimator we built and used to rescue real-data performance (skill −1.89 → +0.14)
does **not** share this convergence property — swept across the same M range, its skill stays flat or
gets *worse* (pinned around −1 to −24 in two of three regimes), because it applies a fixed heuristic
correction rather than one that backs off as M grows. **The practical recommendation for someone at
ENIGMA scale is therefore not our robust heuristic as built** — it is more likely: use the simple
(naive) estimator once M is large enough, or better, replace the fixed-threshold heuristic with a
properly M-adaptive shrinkage estimator that interpolates between aggressive correction at low M and
none at high M. We flag this as an open item rather than a solved one.

## Recommendation

1. **Retire or heavily caveat the single sorted-β regime call** for practical use — it does not
   forecast sufficiency and can point in the wrong direction (Sections 1 above).
2. **Report a position-preserving statistic instead** (ρ_q or equivalent) as the practical
   discoverability diagnostic — it recovers a real, if noisy, real-data signal that β does not have.
3. **If the full learning-curve theory is used, flag per-mode target-power estimation as a first-class
   problem**, not an implementation detail — our results suggest it is the dominant source of real-data
   failure, more than the theory itself (which is exact in the idealized case). This failure looks like
   a genuinely resolvable finite-sample artifact at ENIGMA-relevant labeled-cohort sizes (~10–30× HBN,
   Section 4) — but our own fixed-heuristic robust estimator does *not* scale correctly and would need
   to be replaced with a properly M-adaptive shrinkage estimator before it can be trusted at that scale.
   A worked example spanning cohort sizes, with an M-adaptive estimator, would strengthen the paper
   considerably and is the most actionable open item here.
4. **Report real-data validation against an honest, out-of-sample baseline** (e.g. leave-one-N-out on a
   simple flexible curve) — an in-sample comparison can make a theory look better than it is.

## Caveats on this feedback itself

- Synthetic tests use ridge regression on i.i.d. Gaussian features with power-law spectra; untested
  outside that generative family.
- Real-data n is modest by the standard of the claim: 10–14 points for the direct ρ_q/β test, 81
  learning curves (44 high-signal) for the CBP/power-law comparison, one cohort (HBN), a handful of
  representations (5–8 modalities, mostly vision/structural-MRI foundation models plus hand-built
  features).
- We did not test the paper's own worked examples (the pathology-probe results in the companion
  write-up at `/data/mhough/zeta_beta/ZETA_BETA_writeup.md`) — only our independent HBN construction.
- All code, exact numbers, and commit hashes above are reproducible from this repo; happy to share
  data/scripts directly.
