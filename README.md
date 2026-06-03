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

Known rough edges are listed at the bottom of `src/wwj/core.py` — fix before any published comparison.

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
ckpt = torch.load("path/to/latest.pt", map_location="cpu", weights_only=False)
mats = [(k, jnp.asarray(v.numpy())) for k, v in ckpt["model"].items()
        if v.ndim == 2 and min(v.shape) >= 50]
stats = [wwj.analyze_matrix(W, name) for name, W in mats]
```

## License

MIT.
