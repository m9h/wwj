"""Differentiable alpha->2 regularizer in PyTorch -- a faithful port of wwj.core.alpha_loss
/ _hill_alpha (which are JAX) so it can be added to the HF/torch from-scratch training loss
for the CAUSAL intervention: does pushing the weight spectrum toward the RG-optimal alpha=2
make circuits form earlier (heavy tail = enabling cause), or manufacture the spectral
signature WITHOUT the circuit (Goodhart -- alpha is only a readout)?

wwj's _hill_alpha is the differentiable surrogate (fixed top-`frac` tail, no KS xmin
selection). We compute the ESD via the smaller-side Gram matrix eigvalsh(W^T W / max(n,m)) --
exactly as wwj.core._eigvals does -- NOT svdvals(W): for the token-embedding matrix
(50257x768) a full SVD hits cuSOLVER's slow tall-matrix path (~20 s/call on an A100, an 11x
training slowdown), whereas the 768x768 Gram eigendecomposition is milliseconds.

Lazy torch import so the module can be mounted into a Modal image whose launching env has no
torch.
"""


def hill_alpha(W, frac: float = 0.5, eps: float = 1e-12):
    """Differentiable Hill exponent on the top-`frac` of W's eigenvalue spectrum (ESD of
    W^T W / max(n,m)), matching wwj.core._hill_alpha / _eigvals. Returns a scalar torch tensor."""
    import torch
    n, m = W.shape[0], W.shape[1]
    N = float(max(n, m))
    X = (W.t() @ W) / N if n >= m else (W @ W.t()) / N   # smaller-side Gram, PSD, <= 768x768
    lam = torch.linalg.eigvalsh(X).flip(0).clamp_min(0.0)  # eigenvalues, descending, >= 0
    k = max(1, int(lam.shape[0] * frac))
    top = lam[:k]
    xmin = top[-1].clamp_min(eps)
    denom = torch.log(top.clamp_min(eps) / xmin).sum().clamp_min(eps)
    return 1.0 + k / denom


def alpha_penalty(model, target: float = 2.0, frac: float = 0.5, min_dim: int = 50):
    """mean_l (alpha_l - target)^2 over the same 2D weight matrices wwjd analyses
    (min dim >= min_dim), on the transformer body (so tied lm_head is not double-counted)."""
    import torch
    base = model.transformer if hasattr(model, "transformer") else model
    terms = []
    for nm, p in base.named_parameters():
        if nm.endswith("weight") and p.ndim >= 2:
            W = p.reshape(p.shape[0], -1)
            if min(W.shape) >= min_dim:
                terms.append((hill_alpha(W, frac) - target) ** 2)
    return torch.stack(terms).mean()
