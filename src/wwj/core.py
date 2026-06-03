# weightwatcher_jax.py: MVP JAX/Equinox port of WeightWatcher (Martin & Mahoney
# HT-SR; Martin "RG Theory of Learning" arXiv 2026). Per-layer spectral
# diagnostics for any Equinox module: extract 2D weight matrices, eigendecompose
# W^T W / N, fit the power-law tail, report RG-aligned observables.
#
# Two analysis modes:
#   - "csn": Clauset-Shalizi-Newman MLE + KS scaling-window selection. Matches
#     reference Python WeightWatcher. Non-differentiable (discrete xmin pick).
#   - "hill": Closed-form Hill estimator on a fixed window. Fully differentiable;
#     usable as a regularizer in training to drive layers toward alpha=2 directly.
#
# Sketch / MVP only. Known rough edges flagged in trailing comments.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


@dataclass(frozen=True)
class LayerStats:
    name: str
    n_evals: int
    alpha: Float[Array, ""]
    log_norm: Float[Array, ""]
    stable_rank: Float[Array, ""]
    entropy: Float[Array, ""]
    num_pl_spikes: int           # Correlation Trap count (csn mode only)
    alpha_dist_from_2: Float[Array, ""]
    lambda_max: Float[Array, ""]


def _eigvals(W: Float[Array, "n m"]) -> Float[Array, "k"]:
    """Eigenvalues of W^T W / N (smaller side), sorted descending."""
    n, m = W.shape
    N = max(n, m)
    X = (W.T @ W) / N if n >= m else (W @ W.T) / N
    return jnp.linalg.eigvalsh(X)[::-1]


def _csn_mle_alpha(eigs: Float[Array, "k"], xmin: Float[Array, ""]) -> Float[Array, ""]:
    """alpha = 1 + n / sum(log(lambda / xmin)) over the tail lambda >= xmin.
    Uses masked sum so shape stays static (jit-friendly)."""
    mask = eigs >= xmin
    n = jnp.sum(mask)
    s = jnp.sum(jnp.where(mask, jnp.log(eigs / xmin), 0.0))
    return 1.0 + n / s


def _ks_select_xmin(eigs: Float[Array, "k"], n_candidates: int = 32) -> Float[Array, ""]:
    """Pick xmin by KS distance over a log-spaced grid (CSN 2009).
    Non-differentiable through argmin; vmap'd inner loop is fast."""
    lo = jnp.log(jnp.maximum(eigs[-1], 1e-12))
    hi = jnp.log(eigs[0]) - 0.5            # leave headroom for the tail
    cands = jnp.exp(jnp.linspace(lo, hi, n_candidates))

    def ks_for(xmin):
        # eigs is sorted descending; the tail is at positions [0, m-1]. For each
        # tail point at index i, cumsum(mask) gives the count of values >= eigs[i]
        # (its rank from the top, 1-indexed). The empirical CDF P(X <= eigs[i])
        # is rank-from-bottom / m = (m - rank_from_top + 1) / m.
        mask = eigs >= xmin
        a = _csn_mle_alpha(eigs, xmin)
        log_z = jnp.where(mask, jnp.log(eigs / xmin), 0.0)
        m = jnp.maximum(jnp.sum(mask), 1)
        emp = jnp.where(mask, (m + 1 - jnp.cumsum(mask)) / m, 0.0)
        fit = jnp.where(mask, 1.0 - jnp.exp(-(a - 1.0) * log_z), 0.0)
        return jnp.max(jnp.abs(emp - fit))

    return cands[jnp.argmin(jax.vmap(ks_for)(cands))]


def _hill_alpha(eigs: Float[Array, "k"], frac: float = 0.5) -> Float[Array, ""]:
    """Differentiable Hill estimator on the top `frac` of the spectrum.
    Loses CSN's optimal window selection but is end-to-end differentiable."""
    k = max(1, int(eigs.shape[0] * frac))
    top = eigs[:k]
    return 1.0 + k / jnp.sum(jnp.log(top / top[-1]))


def _common_stats(W, eigs, alpha) -> dict:
    fro = jnp.linalg.norm(W)
    spec = jnp.sqrt(eigs[0] * max(W.shape))    # ||W||_2 from W^T W's lambda_max
    p = eigs / jnp.sum(eigs)
    return dict(
        log_norm=jnp.log(fro),
        stable_rank=(fro / spec) ** 2,
        entropy=-jnp.sum(p * jnp.log(p + 1e-12)),
        alpha_dist_from_2=jnp.abs(alpha - 2.0),
        lambda_max=eigs[0],
        n_evals=eigs.shape[0],
    )


def _layer_stats_csn(W, name: str) -> LayerStats:
    eigs = _eigvals(W)
    xmin = _ks_select_xmin(eigs)
    alpha = _csn_mle_alpha(eigs, xmin)
    # MP/Tracy-Widom edge: lambda_+ = sigma^2 (1 + sqrt(M/N))^2; spikes beyond it
    # are Correlation Trap candidates per Martin RG paper.
    n, m = W.shape
    N, M = max(n, m), min(n, m)
    sigma2 = jnp.var(W)
    mp_edge = sigma2 * (1.0 + jnp.sqrt(M / N)) ** 2
    traps = int(jnp.sum(eigs > mp_edge))
    return LayerStats(name=name, alpha=alpha, num_pl_spikes=traps,
                      **_common_stats(W, eigs, alpha))


def _layer_stats_hill(W, name: str, frac: float = 0.5) -> LayerStats:
    eigs = _eigvals(W)
    alpha = _hill_alpha(eigs, frac=frac)
    return LayerStats(name=name, alpha=alpha, num_pl_spikes=0,
                      **_common_stats(W, eigs, alpha))


def _walk_matrices(model: PyTree, min_dim: int = 50) -> list[tuple[str, Float[Array, "n m"]]]:
    """Find 2D float arrays in an Equinox model. Returns [(jax-key-path, W), ...]."""
    out = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(model):
        if not eqx.is_inexact_array(leaf): continue
        if leaf.ndim != 2: continue
        if min(leaf.shape) < min_dim: continue
        out.append((jax.tree_util.keystr(path), leaf))
    return out


def analyze(model: PyTree, mode: Literal["csn", "hill"] = "csn",
            min_dim: int = 50, hill_frac: float = 0.5) -> list[LayerStats]:
    """Run WeightWatcher on every 2D weight matrix in `model`."""
    fn = _layer_stats_csn if mode == "csn" else (
        lambda W, name: _layer_stats_hill(W, name, frac=hill_frac))
    return [fn(W, name) for name, W in _walk_matrices(model, min_dim=min_dim)]


def analyze_matrix(W: Float[Array, "n m"], name: str = "",
                   mode: Literal["csn", "hill"] = "csn",
                   hill_frac: float = 0.5) -> LayerStats:
    """Matrix-level entry. Use for PyTorch/TF/Flax interop -- extract weight tensors
    as numpy from the source framework, convert via jnp.asarray, pass here."""
    if mode == "csn":
        return _layer_stats_csn(W, name)
    return _layer_stats_hill(W, name, frac=hill_frac)


def alpha_loss(model: PyTree, target: float = 2.0, min_dim: int = 50,
               hill_frac: float = 0.5, weight: float = 1.0) -> Float[Array, ""]:
    """Differentiable regularizer pulling every layer's alpha toward `target`.
    Plug into any optax loss: `loss + alpha_loss(model)`. Drives the model
    toward Martin's RG-optimal alpha=2 fixed point directly -- the explicit
    alternative to Muon's indirect orthogonalization."""
    mats = _walk_matrices(model, min_dim=min_dim)
    if not mats:
        return jnp.zeros(())
    alphas = jnp.stack([_hill_alpha(_eigvals(W), frac=hill_frac) for _, W in mats])
    return weight * jnp.mean((alphas - target) ** 2)


def summary(stats: list[LayerStats]) -> dict[str, float]:
    """Aggregate per-layer stats into the RG diagnostic line analyze_ww.py prints."""
    alphas = jnp.array([s.alpha for s in stats])
    return {
        "n_layers": len(stats),
        "alpha_mean": float(alphas.mean()),
        "alpha_median": float(jnp.median(alphas)),
        "alpha_dist_mean": float(jnp.abs(alphas - 2.0).mean()),
        "near2(1.5-2.5)": int(jnp.sum((alphas >= 1.5) & (alphas <= 2.5))),
        "alpha_lt_2": int(jnp.sum(alphas < 2.0)),
        "alpha_gt_6": int(jnp.sum(alphas > 6.0)),
        "traps_total": int(sum(s.num_pl_spikes for s in stats)),
        "entropy_mean": float(jnp.mean(jnp.array([s.entropy for s in stats]))),
    }


# --- Known rough edges (not blocking for MVP, fix before publishing) ---
#
# 1. _ks_select_xmin's `ks_for` uses jnp.cumsum(mask) but the empirical CDF
#    should be over the *sorted tail*, not raw indices. The MVP gives a
#    monotonically-increasing surrogate; numerical agreement with the Python WW
#    needs a proper sort/rank inside the masked region.
#
# 2. _layer_stats_csn does `int(jnp.sum(eigs > mp_edge))` -- host-side trace.
#    Acceptable for one-shot analysis; for inline-during-training analysis,
#    return as jax scalar and only convert at the print/log boundary.
#
# 3. The MP/Tracy-Widom edge approximation `sigma^2 * (1 + sqrt(M/N))^2` uses
#    jnp.var(W) as sigma^2, which mixes the bulk and tail. The reference WW
#    fits a separate MP baseline on a randomized weight matrix. MVP skips
#    that; v0.1 should add it for accurate trap counting.
#
# 4. Hill estimator with fixed `frac` is crude; a differentiable surrogate
#    for the KS window selection (e.g., softmin over KS-distances) would
#    keep the optimal-window property while staying gradient-friendly.
#
# 5. _walk_matrices excludes Conv2d kernels (4D); for vision models add a
#    reshape that treats (out, in, k, k) as (out, in*k*k) per WW convention.
#
# 6. No vmap across layers because layer shapes differ. For uniform-stack
#    transformers (e.g. all attention QKV the same shape), an inner vmap
#    over a stacked tensor would JIT cleanly and run 10-100x faster.
