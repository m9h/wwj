# Data-matrix vs weight-matrix spectra: where the Zeta Law sits relative to HT-SR

*Compiled 2026-06-22. For the paper's Related Work / positioning, and as a candidate extension.*

A recurring question for the HT-SR / WeightWatcher program: how does our **weight-matrix** spectral
diagnostic relate to the **data-matrix** (covariance) spectral diagnostics other groups use? The
cleanest current foil is **Thompson, *"How Much Data is Enough? The Zeta Law of Discoverability in
Biomedical Data"* (arXiv:2604.17581)**, which builds a data-sufficiency scaling law from the power-law
spectrum of the *data covariance operator*. Short version: **same paradigm, opposite matrix,
complementary question — not a competitor.**

## The strong overlap (identical paradigm)

Both wwj/wwjd and the Zeta Law do the same three moves:
1. take a matrix's **empirical spectral density** (ESD),
2. fit a **power law to the tail**, read off an **exponent**,
3. use that exponent to **predict a learning outcome**.

Both are RMT-grounded the same way — Marchenko–Pastur as the random null, power-law / heavy tail as
the signal of structure — and both are explicitly **scaling-/criticality-flavoured** (HT-SR: RG flow
to the `α≈2` critical fixed point, per Martin's *RG Theory of Learning*; Zeta Law: a `ζ(s)` scaling
law from power-law spectral decay). The "power-law exponent of an ESD governs the behaviour" template
is shared almost verbatim.

## The orthogonal axes (what makes them complementary)

| | **wwj / wwjd (HT-SR)** | **Zeta Law (Thompson 2604.17581)** |
|---|---|---|
| Matrix | **weight** ESD (`WᵀW`, per layer) | **data** covariance ESD (`XᵀX`) |
| Exponent | `α` (weight-tail), critical ≈ 2 | `s` (data-spectrum decay) → `ζ(s)` |
| Predicts | **model quality / generalization** | **data sufficiency / scaling, cross-over regimes** |
| Data needed | **none** — data-free diagnostic (our signature) | **inherently data-dependent** |
| Direction | diagnoses the *trained model*, post hoc | predicts *data needs*, a priori |
| Home field | DNN model selection | biomedical ML scaling (fMRI, imaging genetics) |

The defining contrast is the **data-free** axis: wwj reads quality off the weights with no data or
labels; the Zeta Law is a statement *about* the data spectrum and needs it. They sit at opposite poles
of "what a spectrum can tell you."

## Where they actually meet (the pipeline)

They are the two ends of one process. Weight spectra are *shaped by* data spectra through learning —
this is the HT-SR thesis itself ("the spectrum leads the circuit": the weight tail reflects learned
data correlations). The Zeta Law characterizes precisely the **input** to that process whose **output**
we measure:

> **data covariance spectrum (Zeta Law) → [ learning ] → weight ESD α (HT-SR / wwj).**

- **Rigorous bridge — deep *linear* nets (Saxe–McClelland–Ganguli 2014):** training aligns the weight
  SVD to the input–output correlation SVD, so the weight spectrum *provably* tracks the data spectrum;
  here `s` and `α` are directly linked.
- **Empirical coupling — nonlinear nets (Martin–Mahoney HT-SR):** the heavy weight tail forms as the
  net fits data correlations; looser, training-/task-dependent.

So the two exponents are **not interchangeable** (`α≈2` critical, weight-side; `s` a data-decay rate,
data-side; cf. also Stringer–Pachitariu 2019's representation-spectrum exponent `~1+2/d`), but they are
causally connected by the learning map.

## What this means for wwj / wwjd

1. **Related Work positioning.** The Zeta Law is the **data-side analogue** of HT-SR, not a rival
   "predict-quality-without-training" method. It belongs in the map next to the RMT-of-NN neighbours
   (Pennington; Couillet–Liao) and the scaling-law line — a distinct *data-spectrum* tier. Add a one-
   line entry to `related_researchers.md` under a new "data-side spectra / scaling laws" heading
   (Thompson; Stringer–Pachitariu; the neural-scaling-law line, Sorscher/Bahri).
2. **A candidate extension — a data-side diagnostic.** wwj already fits a power-law exponent to an ESD
   with adaptive `xmin`, bootstrap CI, Vuong/Bayes-factor model comparison. *Point it at the data /
   representation covariance ESD* (`XᵀX` of a layer's activations) and you get a Thompson-style `s`
   with the same machinery — then test the HT-SR-native question: **do the data-side `s` and the
   weight-side `α` flow to criticality together during training?** That is the directly-measurable form
   of "the spectrum leads the circuit," and it's a clean experiment wwj is uniquely set up to run
   (same estimator, two matrices).
3. **The MaxEnt / `wwjd` angle.** Both methods reduce to "a power-law exponent of a spectrum," so
   `wwjd`'s closed-form **Gamma–Pareto posterior over the exponent** applies verbatim to the data ESD —
   a *Bayesian Zeta Law* (calibrated posterior over `s`, hence over the predicted data-sufficiency
   curve) is a natural, low-cost `wwjd` result, and squarely on-theme for MaxEnt 2027 (a power-law
   exponent is a maximum-entropy tail given a scale).

## References
- Thompson, P. M. (2026). *How Much Data is Enough? The Zeta Law of Discoverability in Biomedical
  Data.* arXiv:2604.17581. — data covariance spectrum → `ζ(s)` data-sufficiency scaling.
- Martin, C. H. & Mahoney, M. W. — Heavy-Tailed Self-Regularization; *Implicit Self-Regularization …*
  (JMLR 2021); *Renormalization Group Theory of Learning* (2026). — weight ESD `α`, criticality.
- Saxe, A., McClelland, J., Ganguli, S. (2014). *Exact solutions to the nonlinear dynamics of learning
  in deep linear networks.* — the rigorous data-spectrum ↔ weight-spectrum bridge.
- Stringer, C., Pachitariu, M., et al. (2019). *High-dimensional geometry of population activity.*
  Nature. — power-law *representation* (data-side) spectrum `~ n^{-(1+2/d)}`.
- Marchenko–Pastur (1967); Baik–Ben Arous–Péché (2005). — the shared RMT null + spike transition.
