# Reservoir computing as the untrained-criticality limit of HT-SR

A theory sidebar for the paper's Related Work and a cheap experiment idea. Thesis:
reservoir computing (RC) and Heavy-Tailed Self-Regularization (HT-SR) / RG-of-learning are
**two spectral-criticality theories sharing a random-matrix-theory substrate**, on opposite
sides of the trained/untrained line. RC tunes a *fixed random* matrix to criticality by
initialization; HT-SR describes how *training* flows a matrix's spectrum toward criticality.

> Caveat up front: the *correspondence* below (RC's edge-of-chaos ↔ HT-SR's α=2 fixed point)
> is a conceptual synthesis, not an established formal equivalence. No paper we know of
> unifies them; that gap is itself the interesting part.

## The shared RMT substrate
Both fields read computation off the eigenvalue spectrum of a weight matrix.
- **RC** cares about the spectrum of the *fixed random recurrent* matrix; computation is
  maximized near the **edge of chaos** (spectral radius ≈ 1).
- **HT-SR** cares about how the spectrum of *trained* matrices departs from the random
  baseline: Marchenko–Pastur bulk at init ("phase 1") → heavy power-law tail with α → 2.

The random baseline that RC operates *at* is the same baseline HT-SR measures the departure
*from*.

## The bridge papers
- **Sompolinsky, Crisanti & Sommers (1988), "Chaos in Random Neural Networks"** (Phys. Rev.
  Lett. 61, 259–262) — the RMT foundation of RC's edge-of-chaos. A mean-field theory (exact
  as N→∞) gives a transition from a stationary to a chaotic phase at a **critical gain g = 1**
  — i.e. when the random recurrent matrix's **spectral radius crosses 1** (circular-law
  eigenvalue support), with the maximal Lyapunov exponent crossing zero there. This is
  "disorder → chaos": frozen randomness in the connectivity produces dynamical criticality.
  https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.61.259
- **Jaeger (2001), "The 'echo state' approach…"** (GMD Report 148) — the RC/ESN formalization.
  The **Echo State Property** (effect of initial conditions vanishes) is commonly identified
  with **spectral radius < 1** (with the well-known caveat that this is necessary-ish, not
  strictly sufficient, and input-amplitude-dependent). Only the linear readout is trained;
  the reservoir stays a tuned random matrix. https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf
- **Rajan & Abbott (2006), "Eigenvalue Spectra of Random Matrices for Neural Networks"**
  (Phys. Rev. Lett. 97, 188104) — extends the random-matrix spectrum to *structured*
  connectivity (excitatory/inhibitory columns, different means/variances) and shows the
  eigenvalue spectrum is **independent of the element means**. The bridge from textbook
  circular/MP law to the structured random matrices real recurrent nets (and, by analogy,
  real weight matrices) actually have. https://link.aps.org/doi/10.1103/PhysRevLett.97.188104

## The criticality correspondence (the synthesis)
| | Reservoir computing | HT-SR / RG-of-learning |
|---|---|---|
| Object | fixed random recurrent matrix | trained weight matrix |
| Control parameter | spectral radius (λ_max) ≈ 1 | power-law exponent α → 2 (λ_max-weighted α̂) |
| Critical regime | edge of chaos (max Lyapunov ≈ 0) | α = 2 RG fixed point |
| How criticality is reached | **tuned** by initialization | **learned** by training |
| RMT result invoked | circular law / SCS g=1 transition | Marchenko–Pastur → heavy tail |

Both say *computation is maximized at a spectral phase transition*, and both speak the
language of self-organization in disordered systems. The open question: **are heavy tails
the learned analog of the edge-of-chaos tuning RC does by hand?**

## Hooks for wwj/wwjd
1. **Reservoir = a clean non-power-law negative control.** A well-tuned reservoir's recurrent
   matrix is genuinely *not* heavy-tailed (circular/MP spectrum), so `wwjd.model_posterior`
   should **reject the power law** (Bayes factor favoring exponential/lognormal) and α should
   be large/ill-defined. This is a theoretically-guaranteed "is this even a power law?" case
   to validate the Bayesian model comparison — and it needs **no training**, just analysis.
2. **Learned-vs-tuned criticality, an α-steering experiment.** Use `wwj.alpha_loss` to *train*
   a reservoir's recurrent weights toward α = 2 and ask whether learned-criticality beats
   hand-tuned edge-of-chaos (spectral-radius-1 init) on a memory/prediction task. The
   RC-flavored version of the paper's interventional α-steering, and a direct test of the
   "learned analog" question above.

## References
- Sompolinsky, Crisanti, Sommers (1988). Chaos in Random Neural Networks. PRL 61, 259.
- Jaeger (2001). The "echo state" approach to analysing and training RNNs. GMD Report 148.
- Rajan, Abbott (2006). Eigenvalue Spectra of Random Matrices for Neural Networks. PRL 97, 188104.
- (HT-SR side) Martin & Mahoney, Implicit Self-Regularization (JMLR 2021); Martin RG-of-learning.
