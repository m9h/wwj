# wwj

JAX/Equinox port of [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher), targeting the Kidger SciML stack.

Per-layer spectral diagnostics (HTSR α via Clauset-Shalizi-Newman MLE + KS window; trap counts; trace-log alignment) plus a **differentiable α→2 regularizer** for RG-guided optimization — the explicit alternative to Muon's indirect spectral shaping per Martin's *Renormalization Group Theory of Learning* (arXiv 2026).

## What this is

- A small JAX module that operates on weight matrices extracted from any model (Equinox-native, or via numpy-bridged from PyTorch / TF / Flax).
- Two analysis modes:
  - `csn` — Clauset-Shalizi-Newman MLE + KS scaling-window selection. Matches the reference Python WeightWatcher. Non-differentiable.
  - `hill` — Closed-form Hill estimator on a fixed window. Fully differentiable. Use this in `alpha_loss` as a training-time regularizer.
- A `MultiOptimizer`-friendly `alpha_loss(model, target=2.0)` that drops into any optax pipeline.

## Status

MVP. Validated against the CalculatedContent WeightWatcher fork as numerical oracle on representative weight matrices from four projects (smri-fm, nanopath, eeg-fm-spectral, hippy-feat). See `benchmarks/compare_projects.py`.

## Workshop paper

[`paper/wwj_rg_robustness.pdf`](paper/wwj_rg_robustness.pdf) — ICBINB-style workshop draft of the RG-robustness experiments. Tests Martin's RG-theory prediction that α ≈ 2 yields better robustness using `wwj` to estimate α and `wwj.alpha_loss` as the interventional handle. See [`paper/README.md`](paper/README.md) for the rebuild instructions.

Known rough edges are listed at the bottom of `src/wwj/core.py` — fix before any published comparison.

## JAX-exact features beyond the Python WW

Wherever the reference WeightWatcher takes a shortcut for compute reasons, wwj does the exact version because JAX's JIT+vmap makes it free:

- **Adaptive xmin search** (`_ks_select_xmin`) — every eigenvalue as a candidate, vmap-parallel KS distance; matches Clauset-Shalizi-Newman 2009 / Alstott-Bullmore-Plenz 2014 algorithmically rather than via the log-spaced grid the Python WW falls back to.
- **`bootstrap_alpha_ci(eigs, n_bootstrap=1000)`** — vmap-parallel bootstrap confidence interval for α. Reference WW supports this but it's slow (serial numpy resampling); here it's one `jax.vmap` call.
- **`fit_distributions(eigs)`** — fits power-law + exponential + lognormal MLEs in a single jit pass; returns Vuong's LRT (positive = power-law preferred). The reference WW loops through distributions one-at-a-time via the `powerlaw` package; we do them in parallel. This is the standard "is this layer actually a power-law?" validation that HTSR α claims require.
- **`alpha_loss(model, target=2.0)`** — fully-differentiable α→2 regularizer via the Hill estimator; plug into any optax loss. This is the explicit alternative to Muon's indirect spectral shaping per Martin's RG theory.

## Install

```bash
uv sync
uv pip install -e /home/mhough/dev/weightwatcher    # the WW fork as oracle
uv run pytest tests/                                 # smoke + oracle tests
```

## Usage

```python
import equinox as eqx
import wwj

model = MyEquinoxModule(...)
stats = wwj.analyze(model, mode="csn")
print(wwj.summary(stats))

# As a regularizer in an optax training loop:
def loss_fn(model, batch):
    return task_loss(model, batch) + wwj.alpha_loss(model, target=2.0, weight=0.01)
```

For PyTorch interop:
```python
import torch, jax.numpy as jnp, wwj
from wwj.core import _eigvals

ckpt = torch.load("path/to/latest.pt", map_location="cpu", weights_only=False)
mats = [(k, jnp.asarray(v.numpy())) for k, v in ckpt["model"].items()
        if v.ndim == 2 and min(v.shape) >= 50]

# Per-layer multi-observable: alpha + bootstrap CI + distribution-selection LRT
for name, W in mats:
    eigs = _eigvals(W)
    boot = wwj.bootstrap_alpha_ci(eigs, n_bootstrap=1000, ci=0.95)
    fit = wwj.fit_distributions(eigs)
    pl_valid = fit["pl_vs_exp_lrt"] > 0 and fit["pl_vs_ln_lrt"] > 0
    print(f"{name}: alpha={boot['alpha']:.2f} CI=[{boot['ci_low']:.2f}, {boot['ci_high']:.2f}] PL valid: {pl_valid}")
```

## License

MIT.
