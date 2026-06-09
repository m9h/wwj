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


# Eigenvalues below this (relative to the matrix's largest) are numerical zero,
# not part of the spectrum. Real trained models DO contain near-dead / low-rank
# weight matrices (e.g. POYO learns to zero out some attention/proc projections
# while still decoding); their ESD has no meaningful heavy tail, so alpha is
# undefined there. We must report that as degenerate rather than nan-poison the
# aggregate (eigvals -> 0 -> log(0) -> nan was rough-edge that broke real models).
_EIG_RTOL = 1e-9


def _eigvals(W: Float[Array, "n m"]) -> Float[Array, "k"]:
    """Eigenvalues of W^T W / N (smaller side), sorted descending.
    W^T W is PSD; tiny negatives from finite precision are clipped to 0."""
    n, m = W.shape
    N = max(n, m)
    X = (W.T @ W) / N if n >= m else (W @ W.T) / N
    return jnp.maximum(jnp.linalg.eigvalsh(X)[::-1], 0.0)


def _n_meaningful(eigs: Float[Array, "k"]) -> int:
    """Count eigenvalues above the relative numerical-zero floor."""
    floor = float(eigs.max()) * _EIG_RTOL
    return int(jnp.sum(eigs > floor))


def _csn_mle_alpha(eigs: Float[Array, "k"], xmin: Float[Array, ""]) -> Float[Array, ""]:
    """alpha = 1 + n / sum(log(lambda / xmin)) over the tail lambda >= xmin.
    Uses masked sum so shape stays static (jit-friendly). Guards a non-positive
    xmin (log blow-up) and an empty/degenerate tail (s<=0) by returning nan,
    so a degenerate resample/layer signals cleanly instead of poisoning stats."""
    xmin = jnp.maximum(xmin, jnp.finfo(eigs.dtype).tiny)
    mask = eigs >= xmin
    n = jnp.sum(mask)
    s = jnp.sum(jnp.where(mask, jnp.log(eigs / xmin), 0.0))
    return jnp.where(s > 0.0, 1.0 + n / s, jnp.nan)


def _ks_select_xmin(eigs: Float[Array, "k"], min_tail_size: int = 50) -> Float[Array, ""]:
    """Pick xmin by KS distance, searching EVERY eigenvalue as a candidate.
    Matches the reference powerlaw package (Alstott, Bullmore, Plenz 2014) and the
    Clauset-Shalizi-Newman (2009) algorithm exactly -- no log-spaced grid
    approximation. JAX makes this cheap via vmap+jit (an N x N KS computation
    that the Python reference avoids for compute reasons).

    eigs is sorted descending. tail of candidate xmin = positions [0, m-1] where
    eigs >= xmin. Empirical CDF P(X <= eigs[i]) = (m + 1 - cumsum_from_top) / m.
    Candidates with tail < min_tail_size get an unreachable penalty."""

    def ks_for(xmin):
        mask = eigs >= xmin
        m = jnp.sum(mask)
        valid = (m >= min_tail_size) & (xmin > 0.0)  # positive xmin only
        a = _csn_mle_alpha(eigs, xmin)
        log_z = jnp.where(mask, jnp.log(eigs / xmin), 0.0)
        m_safe = jnp.maximum(m, 1)
        emp = jnp.where(mask, (m_safe + 1 - jnp.cumsum(mask)) / m_safe, 0.0)
        fit = jnp.where(mask, 1.0 - jnp.exp(-(a - 1.0) * log_z), 0.0)
        ks = jnp.max(jnp.abs(emp - fit))
        return jnp.where(valid, ks, 1e6)

    ks_values = jax.vmap(ks_for)(eigs)
    return eigs[jnp.argmin(ks_values)]


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


def bootstrap_alpha_ci(eigs: Float[Array, "k"], n_bootstrap: int = 1000,
                       ci: float = 0.95, key=None) -> dict:
    """Bootstrap confidence interval for the power-law exponent alpha.
    Resamples the eigenvalues with replacement n_bootstrap times, recomputes
    alpha for each resample using the original xmin, returns the quantile-CI
    of the bootstrap distribution. JAX vmap makes 1000 iterations near-free --
    the Python WW supports this but it's slow in numpy because the resampling
    is serial; here it's one vmap call.

    Args:
        eigs: descending-sorted eigenvalues (output of _eigvals).
        n_bootstrap: number of bootstrap iterations.
        ci: two-sided confidence level (0.95 -> 2.5% and 97.5% quantiles).
        key: optional PRNGKey; defaults to PRNGKey(0).

    Returns:
        {"alpha": point-estimate, "xmin": ..., "ci_low": ..., "ci_high": ...,
         "alpha_std": bootstrap std of alpha, "n_bootstrap": ...}
    """
    import jax.random as jr
    key = jr.PRNGKey(0) if key is None else key

    # Degenerate / near-dead layer: too few meaningful eigenvalues to fit a tail.
    # Flag it instead of returning a garbage alpha (its ESD has no power-law).
    if _n_meaningful(eigs) < 50:
        return {"alpha": float("nan"), "xmin": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "alpha_std": float("nan"),
                "n_bootstrap": 0, "degenerate": True}

    xmin = _ks_select_xmin(eigs)
    alpha = _csn_mle_alpha(eigs, xmin)

    def single_boot(k):
        idx = jr.choice(k, eigs.shape[0], (eigs.shape[0],), replace=True)
        return _csn_mle_alpha(eigs[idx], xmin)

    keys = jr.split(key, n_bootstrap)
    alphas = jax.vmap(single_boot)(keys)
    alphas = alphas[jnp.isfinite(alphas)]  # drop degenerate resamples before quantiles
    lo_q, hi_q = (1 - ci) / 2, (1 + ci) / 2
    return {
        "alpha": float(alpha),
        "xmin": float(xmin),
        "ci_low": float(jnp.quantile(alphas, lo_q)) if alphas.size else float("nan"),
        "ci_high": float(jnp.quantile(alphas, hi_q)) if alphas.size else float("nan"),
        "alpha_std": float(jnp.std(alphas)) if alphas.size else float("nan"),
        "n_bootstrap": int(alphas.size),
        "degenerate": False,
    }


def fit_distributions(eigs: Float[Array, "k"]) -> dict:
    """Fit power-law, exponential, and lognormal MLEs to the tail and return
    log-likelihood-ratio statistics for distribution selection. The Python WW
    fits these one-at-a-time via the powerlaw package's loop; we compute all
    three in a single jit-able pass.

    A positive `pl_vs_exp_lrt` favors power-law over exponential; same for
    `pl_vs_ln_lrt` against lognormal. The standard claim "this layer has
    HTSR alpha=X" needs both LRTs positive to be defensible.
    """
    if _n_meaningful(eigs) < 50:  # degenerate/near-dead layer: no fittable tail
        return {"alpha": float("nan"), "xmin": float("nan"), "tail_size": 0,
                "pl_loglik": float("nan"), "exp_loglik": float("nan"),
                "ln_loglik": float("nan"), "pl_vs_exp_lrt": float("nan"),
                "pl_vs_ln_lrt": float("nan"), "exp_lambda": float("nan"),
                "ln_mu": float("nan"), "ln_sigma": float("nan"), "degenerate": True}
    xmin = _ks_select_xmin(eigs)
    mask = eigs >= xmin
    m = jnp.maximum(jnp.sum(mask), 1).astype(eigs.dtype)
    log_eigs = jnp.where(mask, jnp.log(eigs), 0.0)
    log_z = jnp.where(mask, jnp.log(eigs / xmin), 0.0)
    excess = jnp.where(mask, eigs - xmin, 0.0)

    # Power-law MLE: alpha = 1 + n / sum log(x/xmin)
    alpha = 1.0 + m / jnp.sum(log_z)
    pl_ll = jnp.sum(jnp.where(mask,
        jnp.log(alpha - 1) + (alpha - 1) * jnp.log(xmin) - alpha * log_eigs, 0.0))

    # Exponential MLE: lambda = 1 / mean(x - xmin); shifted to start at xmin
    mean_excess = jnp.sum(excess) / m
    lam = 1.0 / jnp.maximum(mean_excess, 1e-12)
    exp_ll = jnp.sum(jnp.where(mask, jnp.log(lam) - lam * excess, 0.0))

    # Lognormal MLE on log(tail)
    mu = jnp.sum(log_eigs) / m
    sig2 = jnp.sum(jnp.where(mask, (log_eigs - mu) ** 2, 0.0)) / m
    sig = jnp.sqrt(jnp.maximum(sig2, 1e-12))
    ln_const = -jnp.log(sig * jnp.sqrt(2 * jnp.pi))
    ln_ll = jnp.sum(jnp.where(mask, -log_eigs + ln_const - (log_eigs - mu) ** 2 / (2 * sig2), 0.0))

    return {
        "alpha": float(alpha),
        "xmin": float(xmin),
        "tail_size": int(m),
        "pl_loglik": float(pl_ll),
        "exp_loglik": float(exp_ll),
        "ln_loglik": float(ln_ll),
        "pl_vs_exp_lrt": float(pl_ll - exp_ll),   # >0 favors power-law
        "pl_vs_ln_lrt": float(pl_ll - ln_ll),     # >0 favors power-law
        "exp_lambda": float(lam),
        "ln_mu": float(mu),
        "ln_sigma": float(sig),
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
