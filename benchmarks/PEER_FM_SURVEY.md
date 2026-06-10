# Peer fMRI-FM spectral survey — where CortexMAE sits among the field

HT-SR α + RG diagnostics (φ_k, M_tr, isolated Correlation Traps) run on the other fMRI
foundation models in the [Brainmarks](https://github.com/MedARC-AI/Brainmarks) registry,
on the same WeightWatcher fork used for the 70 CortexMAE encoders, so they land in one
comparable table. Architecture-agnostic: each published checkpoint's *encoder* sub-state-dict
(extracted with the Brainmarks wrapper's exact key recipe) is wrapped as synthetic
`nn.Linear` matrices and analyzed. Harness: `benchmarks/peer_fmri_fm_survey.py`. Run
2026-06-07 on gx10-dgx-spark in `pytorch_26.04.sif`.

## Cross-FM table (sorted by median α)

| model | family | input space | params | α-med | %α<2 | %[2,6] | %α>6 | φ₁ | M_tr | traps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Brain-JEPA | JEPA | parcel-450 | 87M | **1.53** | **96%** | 4% | 0% | 0.003 | 576 | 1 |
| BrainLM-13M | MAE | parcel-424 | 11M | 2.09 | 38% | 58% | 4% | 0.008 | 256 | 1 |
| **CortexMAE-volume** | MAE | MNI | 87M | 2.78 | 11% | 82% | 7% | 0.005 | 384 | 0 |
| **CortexMAE-flat** | MAE | flat | 86M | 2.93 | 21% | 75% | 4% | 0.005 | 384 | 0 |
| **CortexMAE-parcel** | MAE | Schaefer-400 | 85M | 2.98 | 1% | 96% | 3% | 0.005 | 384 | 0 |
| Brain-Semantoks | self-distill | parcel-457 | 63M | 3.05 | 31% | 47% | 22% | 0.010 | 192 | 3 |
| NeuroSTORM | MAE 4D | volume-4D | 7.7M | 3.46 | 17% | 67% | 16% | 0.031 | 57 | **22** |
| BrainLM-111M | MAE | parcel-424 | 86M | 4.02 | 3% | 86% | 11% | 0.005 | 384 | 3 |
| Brain-Harmony-F | TR-aware | parcel | 88M | 5.14 | 4% | 61% | 35% | 0.003 | 576 | 11 |
| SwiFT | contrastive 4D | volume-4D | 4.4M | **7.32** | 0% | 42% | **58%** | 0.017 | 108 | 0 |

## Findings

1. **CortexMAE occupies a uniquely tight, healthy band; the peers scatter widely — and the
   training *objective*, not the input space, sets where a model lands.** Median α spans
   1.5 → 7.3 across peers, vs CortexMAE's narrow 2.0–3.0. The MAE-with-heavy-masking recipe
   is an outlier in how self-averaging it is.

2. **Objective signature in the spectrum:**
   - **JEPA (Brain-JEPA): extreme heavy tails** — α-med 1.53, 96% of layers α<2. A bare HT-SR
     read ("α<2 ⇒ undertrained") would brand this strong model as completely untrained; it is
     simply a different spectral regime than MAE. Concrete caution: **α thresholds are
     objective-specific** — the same caveat the CortexMAE data/input axes raised, now across
     objectives.
   - **MAE (CortexMAE, BrainLM, NeuroSTORM): the healthy middle** — α-med ~2–4.
   - **Contrastive / small (SwiFT, 4.4M): light tails / over-parameterized** — α-med 7.3, 58%
     of layers α>6, 0 traps (no concentrated modes because the spectrum never developed them).
   - **Brain-Harmony-F: light-tailed (5.1) with many traps (11)** — over-parameterized *and*
     non-self-averaging.

3. **BrainLM independently reproduces the CortexMAE scaling laws.** 13M → 111M: α-med
   2.09 → 4.02, %α<2 38% → 3%, **φ₁ 0.008 → 0.005, M_tr 256 → 384** — the same width law
   (φ₁ ↓, M_tr ↑ with width) and the same capacity→higher-α trend found on CortexMAE's own
   model-scaling axis, in a completely independent training pipeline. Cross-FM validation
   that the *gated predictor's* premise — capacity (M_tr/φ₁) is an architecture coordinate,
   α/%α<2 a training coordinate — is not a CortexMAE artifact.

4. **Correlation Traps separate CortexMAE from the field.** CortexMAE: ~0 traps across all 70
   encoders. Peers: NeuroSTORM 22, Brain-Harmony 11, Semantoks/BrainLM-111M 3 — markedly more
   incipient-memorization signal. CortexMAE is genuinely the least memorization-prone family.

![cross-FM spectral landscape](../../../data/derivatives/peer_fm_ww/results/cross_fm_landscape.png)

## Caveats

- **Encoder extraction is heuristic for the framework-coupled models.** Brain-JEPA
  (`target_encoder`), Brain-Semantoks (`teacher_encoder.`), Brain-Harmony (`encoder_ema.`) give
  clean encoder sub-dicts; NeuroSTORM and SwiFT use a drop-decoder filter, so a few
  projection/embedding matrices may be included that shift absolute α. The synthetic-Linear
  wrapper analyzes every 2D encoder matrix (not only transformer-block weights). Gross patterns
  (JEPA heavy, SwiFT light, BrainLM scaling, trap counts) are robust; exact per-FM α to ±0.3 is not.
- **Architectures/objectives differ — this is a survey, not a controlled comparison.** The one
  controlled axis here is BrainLM 13M-vs-111M (same pipeline).
- **No downstream join yet.** The CortexMAE gated-predictor validation used MedARC's published
  `eval_v2` Brainmarks leaderboard; the peer FMs' Brainmarks probe numbers aren't in hand. The
  open extension is to run/obtain those and test whether the gated predictor transfers *across
  objectives* — the spectral side already says capacity vs training coordinates separate cleanly.

## Local HBN-rest downstream mini-study

Since the CortexMAE `eval_v2` task data is R2-gated and peer downstream results aren't
published anywhere, we built a fresh **local** downstream leaderboard on the Brainmarks
`hbn-rest` data (`/data/datasets/brainmarks`, n=178 subjects, 5-fold CV balanced-accuracy)
for the input-space-matched FMs. Harness: `benchmarks/hbn_local_probe.py` (+ `hbn_consolidate.py`).
This is a *separate* task from `eval_v2`, so it does not join the CortexMAE gated-predictor table.

| model | space | age_bin | sex | α-med | %α<2 | φ₁ | M_tr | traps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CortexMAE-parcel-s400ts3 | 457 | 0.354 | 0.521 | 3.01 | 1% | 0.005 | 384 | 0 |
| CortexMAE-parcel-a424 | 424 | 0.375 | 0.508 | 3.01 | 1% | 0.005 | 384 | 0 |
| **Brain-Semantoks** | 457 | 0.379 | **0.709** | 3.05 | 31% | 0.010 | 192 | 3 |
| **NeuroSTORM** | volume-4D | **0.455** | 0.536 | 3.46 | 17% | 0.031 | 57 | 22 |
| Connectome-FC (baseline) | 457 | 0.435 | 0.580 | — | | | | |
| Connectome-FC (baseline) | 424 | 0.408 | 0.548 | — | | | | |

*age_bin chance 0.25, sex chance 0.50. Connectome-FC = Pearson functional-connectivity + logistic — the bar an FM must clear. Three input spaces covered: parcel-457, parcel-424, volume-4D.*

![local HBN leaderboard](../../../data/derivatives/peer_fm_ww/results/hbn_probe/hbn_leaderboard.png)

**Findings:**
1. **Different architectures win different targets — and each beats FC on its strength.**
   **NeuroSTORM** (whole-brain volume-4D Swin+Mamba) tops **age_bin (0.455)**, the only FM above
   the FC baseline (0.44) on age. **Brain-Semantoks** (parcel self-distill) tops **sex (0.71)**,
   the only FM above FC (0.58) on sex. CortexMAE-parcel beats FC on neither (≈chance on sex).
   No single FM dominates both — input representation + objective matter per task.
2. **The within-CortexMAE RG coordinates do NOT transfer across architectures — they invert.**
   The two best-downstream models are the *spectrally "worst"* by the within-CortexMAE
   self-averaging logic: NeuroSTORM has the **most traps (22)**, highest φ₁ (0.031), lowest M_tr
   (57); Brain-Semantoks has the highest %α<2 (31%). CortexMAE-parcel is the cleanest spectrum
   (1% α<2, 0 traps) yet the weakest downstream. So the gated predictor's capacity/training
   coordinates are **within-family** signals, not a universal cross-architecture quality ranking
   — exactly the spectral survey's headline (objective/architecture dominates the spectrum→quality map).

**Caveats.** n=178, single site (HBN); only age_bin/sex decodable (p_factor ~chance); 4 FMs is
too few for a real RG-vs-downstream correlation, so finding (2) is qualitative (but the inversion
is striking). Embeddings validated faithful by clearing the FC-calibrated floor — two transform
bugs were caught this way (CortexMAE: the stored bold is per-ROI z-scored, so the real transform
*denormalizes* then global-z-scores — per-ROI re-normalization gave near-chance; and the parcel
model expects fewer ROIs than the arrow ships → crop to `img_size`).

**Engineering note — NeuroSTORM on DGX Spark (GB10 / aarch64 / CUDA 13).** NeuroSTORM pins a
2023 stack (py<3.12, pytorch-lightning==1.9.4, torch 2.6, **mamba-ssm**), but per "containers make
any env work" we ran it *inside the up-to-date NGC 26.04 container* (torch 2.12 / CUDA 13.2, which
supports GB10 sm_121) rather than a CPU-only legacy container. Three fixes made it work: (i) the
"fundamental incompatibility" was actually monai's `torch_tensorrt` optional-import hanging —
stub `torch_tensorrt` in `sys.modules`; (ii) PL 2.6 turned out compatible with NeuroSTORM's
LightningModule for construction; (iii) the Swin4D blocks genuinely embed `Mamba` layers, so we
**built `causal-conv1d` for aarch64/CUDA-13.2/sm_121** (mamba-ssm 2.3.2 itself is JIT via
tilelang/cutlass, arch-agnostic). The local `mni_cortex` arrow is cortex-only (132k voxels) — wrong
for NeuroSTORM's whole-brain input — so embeddings were computed from the **raw CPAC MNI BOLD niis**
(`/data/raw/hbn-cpac`, 91×109×91, brain-mask → 228k voxels). The aarch64 `causal-conv1d` wheel +
build notes are staged at `/data/derivatives/peer_fm_ww/wheels/` for public release. **BrainLM (a424)
remains blocked** — its 2024 custom-HF ViTMAE is incompatible with transformers 5.8 (multi-point port).

## Reproduction

```bash
# checkpoints already fetched to /data/derivatives/peer_fm_ww/ckpts (+ hf_cache)
cd /home/mhough/dev/wwj
apptainer exec --no-init -B /data:/data -B /home/mhough:/home/mhough \
  --env APPTAINERENV_PYTHONPATH=/home/mhough/dev/weightwatcher \
  /data/derivatives/containers/pytorch_26.04.sif \
  python benchmarks/peer_fmri_fm_survey.py
```

```bash
# local HBN downstream probe (parcel-space FMs) + consolidation:
apptainer exec --no-init --nv -B /data:/data -B /home/mhough:/home/mhough \
  --env APPTAINERENV_PYTHONPATH=/home/mhough/dev/weightwatcher:/home/mhough/dev/CortexMAE/src:/home/mhough/dev/CortexMAE/scripts:/data/derivatives/peer_fm_ww/Brain-Semantoks \
  /data/derivatives/containers/pytorch_26.04.sif \
  python benchmarks/hbn_local_probe.py \
    --models cortex_mae_parcel_s400ts3 cortex_mae_parcel_a424 brain_semantoks --targets age_bin sex
python benchmarks/hbn_consolidate.py
```

Artifacts: `/data/derivatives/peer_fm_ww/results/` — `{model}_details.csv`,
`{model}_summary.json`, `peer_fms_summary.{csv,json}`, `cross_fm_table.csv`,
`cross_fm_landscape.png`, and `hbn_probe/` (`hbn_probe_{model}.json`,
`hbn_leaderboard_joined.csv`, `hbn_leaderboard.md`, `hbn_leaderboard.png`).

## fMRI FMs surveyed (Brainmarks registry)

BrainLM (Ortega Caro et al., ICLR 2024) · Brain-JEPA (Dong et al., NeurIPS 2024) ·
SwiFT (Kim et al., NeurIPS 2023) · NeuroSTORM · Brain-Semantoks (Gijsen et al.) ·
Brain-Harmony-F (hzlab). Non-FM baselines in the registry but not spectrally analyzable
(no learned weight matrices to fit): Connectome (Pearson FC), Identity.
