# Numerical agreement vs the CalculatedContent WeightWatcher fork as oracle.
# Run with:  uv run pytest tests/test_oracle_match.py -v
import numpy as np
import jax.numpy as jnp
import pytest

import wwj


def _ww_oracle_alpha(W: np.ndarray) -> float:
    """Use the Python WeightWatcher fork's tail-fit on a raw matrix.
    This bypasses the model-walker and exercises the same MLE+KS path
    we're trying to match in wwj.
    """
    import weightwatcher as ww
    import torch.nn as nn
    import torch
    # Wrap as a torch.nn.Linear so ww's analyzer can ingest it.
    n, m = W.shape
    layer = nn.Linear(m, n, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(W.astype(np.float32)))
    watcher = ww.WeightWatcher(model=layer)
    details = watcher.analyze(min_evals=50)
    return float(details["alpha"].iloc[0])


@pytest.mark.parametrize("shape", [(384, 384), (1152, 384), (1536, 384)])
def test_csn_alpha_matches_oracle_within_tolerance(shape):
    """Random matrices have MP-like spectra, not power laws -- so both
    implementations should agree on whatever spurious alpha they fit. The
    bar is *numerical agreement between implementations*, not "correct alpha"."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal(shape).astype(np.float32) / np.sqrt(shape[1])

    oracle_alpha = _ww_oracle_alpha(W)
    wwj_stats = wwj.analyze_matrix(jnp.asarray(W), name="test", mode="csn")
    wwj_alpha = float(wwj_stats.alpha)

    # MVP tolerance: 20% relative; tighten once rough-edge #1 (KS sort/rank)
    # is fixed. Random matrices push the MLE around so even reference WW is
    # noisy here -- the actually-stringent test is on a trained ckpt.
    assert abs(wwj_alpha - oracle_alpha) / abs(oracle_alpha) < 0.20, \
        f"shape={shape}: wwj={wwj_alpha:.3f} vs oracle={oracle_alpha:.3f}"


def test_hill_alpha_is_differentiable():
    """The whole point of mode='hill' is gradient flow for alpha_loss."""
    import jax
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((384, 384)).astype(np.float32))

    def loss(W):
        stats = wwj.analyze_matrix(W, name="t", mode="hill")
        return stats.alpha_dist_from_2
    g = jax.grad(loss)(W)
    assert g.shape == W.shape
    assert jnp.isfinite(g).all()
    assert not jnp.allclose(g, 0.0)


def test_log_norm_and_stable_rank_exact_agreement():
    """These are closed-form; should agree to numerical precision."""
    rng = np.random.default_rng(0)
    W_np = rng.standard_normal((384, 384)).astype(np.float32)
    W = jnp.asarray(W_np)

    stats = wwj.analyze_matrix(W, name="t", mode="csn")
    fro_oracle = np.linalg.norm(W_np)
    spec_oracle = np.linalg.svd(W_np, compute_uv=False)[0]
    log_norm_oracle = np.log(fro_oracle)
    stable_rank_oracle = (fro_oracle / spec_oracle) ** 2

    assert abs(float(stats.log_norm) - log_norm_oracle) < 1e-3
    assert abs(float(stats.stable_rank) - stable_rank_oracle) / stable_rank_oracle < 0.01
