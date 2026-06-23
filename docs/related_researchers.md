# Researcher map: the HT-SR / weight-spectrum landscape around Charles Martin

Who is working on lines adjacent to Martin & Mahoney's Heavy-Tailed Self-Regularization /
WeightWatcher program — for the paper's Related Work and for outreach. Tiered from the
literal continuation of the WeightWatcher-α line outward. Compiled 2026-06-09.

## 1. Closest continuation — HT-SR α turned into practice
- **Michael W. Mahoney** (UC Berkeley / ICSI / LBNL) — Martin's longtime co-author; senior
  anchor of the whole program.
- **Yaoqing Yang** (Dartmouth) — the most active continuation; turns the per-layer α
  *diagnostic* into *interventions*:
  - **TempBalance** (NeurIPS 2023) — per-layer HT-SR α sets layer-wise learning rates.
  - **AlphaPruning** (NeurIPS 2024, arXiv:2410.10912; Haiquan Lu, Yefan Zhou, +) — α sets
    layer-wise *pruning* ratios for LLMs. Code: github.com/haiquanlu/AlphaPruning.
- **Shiwei Liu, Zhangyang "Atlas" Wang** (UT Austin) — sparsity / LLM-efficiency co-authors
  on the above.
- **"Eigenspectrum Analysis of Neural Networks without Aspect-Ratio Bias"** (arXiv:2506.06280,
  2025) — methodological refinement of ESD fitting; direct sibling to wwj/wwjd.

## 2. Heavy-tailed dynamics — why the tails form (SGD side)
- **Umut Şimşekli** (INRIA / ENS) — heavy-tailed SGD, tail-index estimation,
  Hausdorff-dimension generalization bounds.
- **Liam Hodgkinson** (Melbourne; ex-Berkeley/Mahoney) — heavy-tailed RMT + multiplicative-
  noise generalization (arXiv:2006.06293); bridges static-weight and dynamics views.
- **Mert Gürbüzbalaban, Lingjiong Zhu** — heavy tails in stochastic optimization.

## 3. RMT-of-neural-networks neighbors (spectral theory)
- **Jeffrey Pennington** (Google) — nonlinear RMT, Jacobian/Hessian spectra (original RMT-of-NNs).
- **Romain Couillet & Zhenyu Liao** — *Random Matrix Methods for ML* (book).
- **Levent Sagun, Behrooz Ghorbani, Stanislav Fort, Vardan Papyan** — Hessian / loss-landscape
  eigenspectra (Hessian rather than weight matrices, but adjacent).

## 4. "Predict quality without training" cousins
- **Behnam Neyshabur & Yiding Jiang** — *Fantastic Generalization Measures* + the PGDL
  competition (the contest Martin's "post-mortem" paper dissected).
- Zero-cost NAS proxy community (SynFlow et al.).

## 5. Physics / RG-of-learning (Martin's SETOL / RG framing)
- **Dan Roberts & Sho Yaida** — *Principles of Deep Learning Theory* (effective field theory /
  RG-flavored).
- **James Halverson, Anindita Maiti** — neural-network field theory / RG.
- **Lenka Zdeborová & Florent Krzakala** (EPFL) — statistical physics of learning (phase
  transitions, replica method). Also the MaxEnt-2027 statistical-physics neighborhood.

## 6. Data-side spectra / scaling laws (the complement, not a competitor)
The mirror image of HT-SR: same "power-law exponent of an ESD predicts an outcome" paradigm, but on
the **data covariance** spectrum rather than the **weight** spectrum. Full comparison in
[`docs/data_vs_weight_spectra.md`](data_vs_weight_spectra.md).
- **Paul M. Thompson** (USC / ENIGMA) — *The Zeta Law of Discoverability in Biomedical Data*
  (arXiv:2604.17581, 2026): data-sufficiency scaling from power-law decay of the data covariance
  spectrum, yielding a `ζ(s)` law. The data-side analogue of our weight-side α — predicts *how much
  data* (data-dependent) vs our *model quality* (data-free).
- **Carsen Stringer & Marius Pachitariu** — power-law *representation* spectrum of population activity
  (`~ n^{−(1+2/d)}`, Nature 2019); the data/representation-spectrum counterpart.
- **Neural-scaling-law line** — Sorscher, Bahri, Ganguli (data pruning / "beyond power-law scaling");
  scaling exponents from data statistics — same family, derived from stat-mech / teacher–student
  rather than analytic number theory.

*Bridge to us:* `data ESD (s) → [learning] → weight ESD (α)` — rigorous in deep-linear nets
(Saxe–McClelland–Ganguli 2014), empirical via HT-SR. Candidate wwj experiment: point our ESD estimator
at the activation/data covariance and test whether `s` and `α` co-flow to criticality during training.

## How wwj/wwjd is positioned against this
The Yang/Mahoney line (TempBalance, AlphaPruning) makes **decisions** — learning rates,
pruning ratios — directly off the per-layer α **point estimate**. That is exactly where
`wwjd` adds what no one in this cluster has: a **calibrated posterior on the α that drives
those decisions**, plus the **power-law-validity test** (model_posterior) for whether α is
even meaningful per layer. "AlphaPruning / TempBalance, but with credible intervals on the
per-layer α" is a natural, citable follow-on — and **Yaoqing Yang's group is the obvious
audience/collaborator**. The MaxEnt 2027 venue also puts the work adjacent to the
Zdeborová/Krzakala statistical-physics community.

## Key papers to pull for Related Work
- AlphaPruning — arXiv:2410.10912 (NeurIPS 2024)
- TempBalance — Zhou/Yang/Mahoney (NeurIPS 2023)
- Eigenspectrum without aspect-ratio bias — arXiv:2506.06280 (2025)
- Hodgkinson & Mahoney, multiplicative noise / heavy tails — arXiv:2006.06293
- Şimşekli et al., Hausdorff-dimension generalization bound
- Jiang/Neyshabur, Fantastic Generalization Measures (PGDL)
- Thompson, *Zeta Law of Discoverability in Biomedical Data* — arXiv:2604.17581 (2026) [data-side]
