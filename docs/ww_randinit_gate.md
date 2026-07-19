# Positive control: does the wwjd validity gate reject random matrices?

`benchmarks/ww_randinit_gate.py`. Run 2026-07-07, results in
`/data/mhough/wwj_randinit_gate/`.

## Why

Several weight-side claims rest entirely on the Bayesian model comparison saying
*"this tail is not a power law"* — most directly the BatchNorm finding in
[`ww_replication_targets.md`](ww_replication_targets.md) (8–17% of BN-VGG layers
rejected; plain VGG and GPT at 0%). A rejection test is only citable if it rejects
things that ought to be rejected, and nothing had shown ours does.

The specific failure mode to hunt is documented by the tool we improve on:
WeightWatcher's own docs concede that at aspect ratio Q = m/n ≈ 1 a *random* matrix
ESD can look heavy-tailed, and Clauset/Shalizi/Newman (SIAM Rev 2009) show
power-law-vs-log-normal is undecidable without very large samples.

Two arms: **synthetic** (iid Gaussian at controlled Q — ground truth Marchenko–Pastur,
provably not a power law) and **arch** (real torchvision architectures at
`weights=None` — real init, real shapes, real depth, no training; matched to the
trained runs on everything except training).

All statistics below are **signs or pass/fail rates**. No Bayes factor magnitudes are
interpreted, per the Lindley–Bartlett warning in `model_posterior`'s own docstring.

## Pre-registered predictions, and what happened

| | prediction | outcome |
|---|---|---|
| **P1** | untrained matrices rejected at a high rate | **mixed** — 100% synthetic, but only 5.6–13.3% of untrained VGG layers |
| **P2** | rejection hardest at Q=1, easiest at large Q | **wrong** — 100% rejection at *every* Q, no gradient |
| **P3** | trained plain VGG rejects far less than untrained | **confirmed in direction**, but both are ~0% on the matched criterion; the real contrast is BN (below) |
| **P4** | the PPC leg does more work than the Bayes-factor legs | **arm-dependent** — false on synthetic (exp leg, 100%), true on architectures (PPC, 67% of rejections) |

## Result 1 — the gate discriminates

Synthetic arm, 25 matrices, Q ∈ {1, 2, 4, 8, 16}:

```
Q=1  rejected 100.0%     Q=8   rejected 100.0%
Q=2  rejected 100.0%     Q=16  rejected 100.0%
Q=4  rejected 100.0%
--> FALSE POSITIVE RATE: 0.0%
```

**Zero false positives on matrices with known non-power-law ground truth, including at
Q = 1** — the case WeightWatcher's docs flag as the trap. Rejection-based claims are
structurally sound.

Rejection came from the **exponential** leg, 100% of the time, with
`best_model = exponential`. P2 and P4 were both wrong here for the same reason: the MP
bulk has a *hard spectral edge*, so its tail decays faster than any power law and the
exponential wins outright at every aspect ratio. There is no Q-gradient because the
edge exists at all Q. Rejection happens for a clean, interpretable reason rather than
generic misfit — a better outcome than predicted.

## Result 2 — the BatchNorm finding is not an initialisation artifact

**A gate-definition mismatch had to be fixed first.** The trained CSVs in
`/data/mhough/wwj_ww_replication/` have no `ppc` column, so the original 8–17% figure
was computed as `best_model != "powerlaw"` — *not* the 3-leg gate (ln + exp + PPC) used
here. Comparing the two directly would have been invalid. Both sides recomputed on the
matched `best_model` criterion:

| model | trained | untrained |
|---|---|---|
| vgg11 (plain) | 0.0% (0/10) | 0.0% (0/10) |
| vgg16 (plain) | 0.0% (0/15) | 0.0% (0/15) |
| **vgg16_bn** | **13.3% (2/15)** | **0.0% (0/15)** |
| **vgg19_bn** | **16.7% (3/18)** | **0.0% (0/18)** |

Untrained BN-VGG rejects at **exactly zero**. So the power-law breakdown in BN networks
is **caused by training**, not by architecture, initialisation, or matrix shape — and
it is specific to BN, since plain VGG sits at 0% in both conditions. That is a stronger
claim than the original finding supported, and it is the claim the control was built to
license.

**But the counts are small, and the effect is not statistically established:**

```
vgg16_bn   trained 2/15  untrained 0/15   Fisher p = 0.483
vgg19_bn   trained 3/18  untrained 0/18   Fisher p = 0.229
BN pooled  trained 5/33  untrained 0/33   Fisher p = 0.053
```

Pooled p = 0.053 does not clear 0.05. The direction is clean and consistent across both
BN models, and the zero-vs-nonzero contrast is qualitatively striking, but **n = 33
layers cannot establish it**. The paper should state the BN result as *suggestive and
mechanistically interpretable*, not as demonstrated. Adding vgg11_bn/vgg13_bn and
non-VGG BN architectures is the cheap fix and should be done before the claim is
leaned on.

## Result 3 — the gate definition changes the answer by 10×

Untrained `densenet121`, same 121 layers, two criteria:

- `best_model != "powerlaw"` → **5.8%** rejected
- 3-leg gate including PPC → **56.2%** rejected

The 3-way Bayes factor picks the *least bad* of three candidate models; the PPC asks
whether the winner *actually fits*. On DenseNet they disagree wildly. Two consequences:

1. **Always report which criterion was used.** They are not interchangeable, and the
   gap is not a detail.
2. **The planned DenseNet PL-rejection experiment needs this baseline.** Martin &
   Mahoney's DenseNet claim (α ≈ 8, "even a PL model is probably a poor fit") is a
   target in `ww_replication_targets.md` §2. An *untrained* DenseNet is already 56%
   PPC-rejected. Any trained-DenseNet rejection rate must be read against that, or
   architecture and shape effects will be misattributed to training. This control
   pre-empts a confound that would otherwise have gone into the paper.

## What this licenses, and what it does not

**Licensed:** the gate rejects known-random matrices with a 0% false-positive rate at
all aspect ratios, so rejection-based evidence is admissible. The BN breakdown is
training-induced, not an init artifact.

**Not licensed:** the BN effect size (p = 0.053, n = 33). Any cross-criterion
comparison of rejection rates. Any claim about DenseNet before the trained-vs-untrained
pair is run.

## Reproduce

```bash
uv run --extra benchmarks python benchmarks/ww_randinit_gate.py \
    --reps 5 --out-dir /data/mhough/wwj_randinit_gate
# synthetic arm only (fast):
uv run --extra benchmarks python benchmarks/ww_randinit_gate.py --synthetic-only --reps 2
```

## Next

- Extend the BN arm (vgg11_bn, vgg13_bn, + a non-VGG BN architecture) to get the
  pooled test off p = 0.053.
- Trained-vs-untrained DenseNet pair, using the baseline established here
  (`ww_replication_targets.md` §2).
- Consider adding a `ppc_p` column to `ww_replication.py`'s output so trained and
  untrained runs are gate-comparable by construction.
