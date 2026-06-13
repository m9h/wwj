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


def _gram(W, eps=1e-12):
    """Smaller-side Gram W^T W / max(n,m) (or W W^T), the PSD matrix whose eigenvalues are
    wwj.core._eigvals' ESD. For GPT-2 the smaller side is always 768, so every Gram is
    768x768 -- which lets alpha_penalty batch them into a single eigvalsh (see below)."""
    import torch
    n, m = W.shape[0], W.shape[1]
    N = float(max(n, m))
    return (W.t() @ W) / N if n >= m else (W @ W.t()) / N


def _hill_from_eigs(lam, frac: float = 0.5, eps: float = 1e-12):
    """Hill exponent from descending eigenvalues lam (..., d); batched over leading dims."""
    import torch
    k = max(1, int(lam.shape[-1] * frac))
    top = lam[..., :k].clamp_min(eps)
    xmin = top[..., -1:].clamp_min(eps)
    denom = torch.log(top / xmin).sum(-1).clamp_min(eps)
    return 1.0 + k / denom


def hill_alpha(W, frac: float = 0.5, eps: float = 1e-12):
    """Differentiable Hill exponent on the top-`frac` of W's ESD (W^T W / max(n,m)),
    matching wwj.core._hill_alpha / _eigvals. Returns a scalar torch tensor."""
    import torch
    lam = torch.linalg.eigvalsh(_gram(W)).flip(-1).clamp_min(0.0)
    return _hill_from_eigs(lam, frac, eps)


def alpha_penalty(model, target: float = 2.0, frac: float = 0.5, min_dim: int = 50):
    """mean_l (alpha_l - target)^2 over the same 2D weight matrices wwjd analyses (min dim
    >= min_dim), on the transformer body (so tied lm_head is not double-counted). All Gram
    matrices are 768x768, so we stack them and run ONE batched eigvalsh -- ~10x faster than
    50 sequential cuSOLVER calls on GPU (the per-step training cost), identical math."""
    import torch
    base = model.transformer if hasattr(model, "transformer") else model
    grams = [_gram(p.reshape(p.shape[0], -1))
             for nm, p in base.named_parameters()
             if nm.endswith("weight") and p.ndim >= 2
             and min(p.shape[0], p.numel() // p.shape[0]) >= min_dim]
    lam = torch.linalg.eigvalsh(torch.stack(grams)).flip(-1).clamp_min(0.0)   # (L, d)
    alphas = _hill_from_eigs(lam, frac)                                       # (L,)
    return ((alphas - target) ** 2).mean()
