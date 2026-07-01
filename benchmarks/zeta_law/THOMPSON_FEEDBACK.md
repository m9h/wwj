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

## Recommendation

1. **Retire or heavily caveat the single sorted-β regime call** for practical use — it does not
   forecast sufficiency and can point in the wrong direction (Sections 1 above).
2. **Report a position-preserving statistic instead** (ρ_q or equivalent) as the practical
   discoverability diagnostic — it recovers a real, if noisy, real-data signal that β does not have.
3. **If the full learning-curve theory is used, flag per-mode target-power estimation as a first-class
   problem**, not an implementation detail — our results suggest it is the dominant source of real-data
   failure, more than the theory itself (which is exact in the idealized case). A worked real-data
   example with an explicit robust/shrinkage estimator (and its failure mode without one) would
   strengthen the paper considerably.
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
