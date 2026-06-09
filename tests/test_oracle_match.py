# Numerical agreement vs the CalculatedContent WeightWatcher fork as oracle.
# Run with:  uv run --extra test pytest tests/test_oracle_match.py -v
#
# The meaningful correctness bar is RECOVERY of a known power-law exponent on a
# synthetic spectrum where alpha is well-defined -- not agreement on the spurious
# alpha two implementations happen to fit to a random (Marchenko-Pastur) matrix,
# where alpha is undefined and the CSN window choice legitimately diverges.
import numpy as np
import jax.numpy as jnp
import pytest

import wwj


def _synth_powerlaw_W(alpha_true, n=512, m=512, xmin=1.0, seed=0):
    """Construct W whose eigenvalues of W^T W / N follow a power law with the
    given exponent. Eigenvalues are Pareto(alpha_true-1) >= xmin; singular values
    are sqrt(lambda * N) placed in a random orthonormal basis."""
    rng = np.random.default_rng(seed)
    k = min(n, m)
    u = rng.uniform(size=k)
    eig = xmin * u ** (-1.0 / (alpha_true - 1.0))      # tail ~ lambda^-alpha_true
    N = max(n, m)
    sv = np.sqrt(np.sort(eig)[::-1] * N)
    U, _ = np.linalg.qr(rng.standard_normal((n, k)))
    V, _ = np.linalg.qr(rng.standard_normal((m, k)))
    return ((U * sv) @ V.T).astype(np.float32)


def _ww_oracle_alpha(W: np.ndarray) -> float:
    """Reference Python WeightWatcher fork's tail-fit on a raw matrix."""
    import weightwatcher as ww
    import torch.nn as nn
    import torch
    n, m = W.shape
    layer = nn.Linear(m, n, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(W.astype(np.float32)))
    details = ww.WeightWatcher(model=layer).analyze(min_evals=50)
    return float(details["alpha"].iloc[0])


@pytest.mark.parametrize("alpha_true", [2.0, 2.5, 3.0])
def test_recovers_known_powerlaw_alpha(alpha_true):
    """wwj's CSN fit recovers a synthetic power-law exponent. This is the
    actual correctness property -- empirically ~3% off truth, we allow 10%."""
    W = _synth_powerlaw_W(alpha_true, seed=1)
    wwj_alpha = float(wwj.analyze_matrix(jnp.asarray(W), name="t", mode="csn").alpha)
    assert abs(wwj_alpha - alpha_true) / alpha_true < 0.10, \
        f"alpha_true={alpha_true}: wwj recovered {wwj_alpha:.3f}"


@pytest.mark.parametrize("alpha_true", [2.0, 2.5, 3.0])
def test_oracle_agreement_on_powerlaw(alpha_true):
    """On a real power law (alpha well-defined) wwj and the reference WW agree.
    Empirically <1% apart; we allow 5%."""
    W = _synth_powerlaw_W(alpha_true, seed=1)
    wwj_alpha = float(wwj.analyze_matrix(jnp.asarray(W), name="t", mode="csn").alpha)
    oracle_alpha = _ww_oracle_alpha(W)
    assert abs(wwj_alpha - oracle_alpha) / abs(oracle_alpha) < 0.05, \
        f"alpha_true={alpha_true}: wwj={wwj_alpha:.3f} vs oracle={oracle_alpha:.3f}"


def test_random_matrix_flagged_not_powerlaw():
    """A random (MP-spectrum) matrix is NOT a power law -- alpha is undefined.
    Both implementations correctly fit a large alpha (>4), even though their
    exact spurious values diverge (the point estimate is meaningless here, and
    that divergence is itself the xmin-sensitivity wwj's BMA addresses). This
    documents why the old 'agree on the random-matrix alpha' bar was wrong."""
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((384, 384)).astype(np.float32) / np.sqrt(384))
    wwj_alpha = float(wwj.analyze_matrix(jnp.asarray(W), name="t", mode="csn").alpha)
    oracle_alpha = _ww_oracle_alpha(W)
    assert wwj_alpha > 4.0, f"expected non-power-law alpha>4, wwj={wwj_alpha:.3f}"
    assert oracle_alpha > 4.0, f"expected non-power-law alpha>4, oracle={oracle_alpha:.3f}"


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
