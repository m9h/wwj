"""Mechanistic-interpretability metrics to pair with the wwjd alpha-trajectory.

The bridge idea: a weight matrix growing a heavy tail (alpha -> 2) is the same event,
in eigenvalue-space, as a circuit forming in computation-space. These metrics make that
testable over a training trajectory:

  ov_copying_scores   -- per-head OV copying score, an EIGENVALUE statistic (Elhage et al.
                         2021, "A Mathematical Framework for Transformer Circuits"); speaks
                         wwjd's own language, so it overlays naturally on alpha(step).
  induction_scores    -- per-head induction score on a repeated-random sequence (Olsson
                         et al. 2022); detects the induction circuit forming.
  stable_rank         -- ||W||_F^2 / ||W||_2^2, a cheap heavy-tail proxy (drops as the
                         spectral spike grows).
  top_subspace / principal_angle_drift -- "subspace freezing": how much a matrix's top
                         singular subspace rotates between consecutive checkpoints (-> 0
                         when the circuit snaps into place).

Copying score and the weight metrics are computed on the RAW HF weight matrices (the same
ones wwjd analyzes), NOT LayerNorm-folded -- a standard approximation that keeps every
quantity in one consistent basis with alpha(step). The induction score is behavioural
(forward pass) and basis-free.
"""
from __future__ import annotations
import numpy as np
import torch


def _dims(model):
    cfg = model.config
    return cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.n_embd // cfg.n_head


def _base(model):
    return model.transformer if hasattr(model, "transformer") else model


@torch.no_grad()
def ov_copying_scores(model) -> np.ndarray:
    """Per-head OV copying score = fraction of the OV circuit's (nonzero) eigenvalues with
    positive real part. The full OV circuit W_E W_OV W_U is (V x V) but rank <= d_head; with
    tied embeddings (W_U = W_E^T) its nonzero eigenvalues equal those of the d_head x d_head
    matrix  W_O^h (W_E^T W_E) W_V^h, which is what we diagonalise. ~0.5 = no copying (random),
    -> 1.0 = a clean copying/induction head (attends-to-X raises logit of X)."""
    base = _base(model)
    nL, nH, d, dh = _dims(model)
    WE = base.wte.weight.detach().float()                 # (V, d)
    G = (WE.T @ WE).numpy()                                # (d, d) = W_U W_E
    out = np.full((nL, nH), np.nan)
    for l in range(nL):
        attn = base.h[l].attn
        Wqkv = attn.c_attn.weight.detach().float().numpy()  # Conv1D (d, 3d): cols [q|k|v]
        Wproj = attn.c_proj.weight.detach().float().numpy() # Conv1D (d, d): rows = concat heads
        Wv = Wqkv[:, 2 * d:3 * d]                          # (d, d) input -> value
        Wo = Wproj                                         # (d, d) head-concat -> output
        for h in range(nH):
            sl = slice(h * dh, (h + 1) * dh)
            Mh = Wo[sl, :] @ G @ Wv[:, sl]                 # (dh, dh): nonzero OV-circuit eig
            ev = np.linalg.eigvals(Mh)
            out[l, h] = float((ev.real > 0).mean())
    return out                                             # (nL, nH)


@torch.no_grad()
def induction_scores(model, seq_len: int = 64, batch: int = 16, seed: int = 0,
                     device: str = "cpu") -> np.ndarray:
    """Per-head induction score on a [BOS, rand, rand] sequence: the mean attention a head
    places on the 'induction stripe' -- from a query in the 2nd copy to the token that
    FOLLOWED the same token's previous occurrence. ~1/T = chance (untrained); a real
    induction head -> 0.3-0.8."""
    base = _base(model)
    nL, nH, d, dh = _dims(model)
    V = base.wte.weight.shape[0]
    bos = getattr(model.config, "eos_token_id", None) or 0
    g = torch.Generator().manual_seed(seed)
    rand = torch.randint(0, V, (batch, seq_len), generator=g)
    pre = torch.full((batch, 1), int(bos), dtype=torch.long)
    ids = torch.cat([pre, rand, rand], dim=1).to(device)   # (B, 2L+1)
    model.eval()
    att = base(ids, output_attentions=True).attentions     # tuple nL of (B, nH, T, T)
    qs = torch.arange(seq_len + 1, 2 * seq_len + 1)         # queries in the 2nd copy
    ks = qs - seq_len + 1                                   # the induction-stripe keys
    scores = np.zeros((nL, nH))
    for l in range(nL):
        vals = att[l][:, :, qs, ks]                        # (B, nH, L)
        scores[l] = vals.float().mean(dim=(0, 2)).cpu().numpy()
    return scores                                          # (nL, nH)


def stable_rank(W: np.ndarray) -> float:
    s = np.linalg.svd(W, compute_uv=False)
    return float((s ** 2).sum() / (s[0] ** 2 + 1e-12))


def top_subspace(W: np.ndarray, k: int = 8) -> np.ndarray:
    """Top-k right singular subspace (input/feature directions), orthonormal cols (n, k)."""
    _, _, Vh = np.linalg.svd(W, full_matrices=False)
    return Vh[:k].T


def principal_angle_drift(Vprev: np.ndarray, Vcur: np.ndarray) -> float:
    """1 - mean cos(principal angle) between two orthonormal subspaces. 0 = frozen
    (identical subspace), 1 = orthogonal (fully rotated)."""
    sv = np.linalg.svd(Vprev.T @ Vcur, compute_uv=False)
    return float(1.0 - np.clip(sv, -1.0, 1.0).mean())


def weight_matrices(model, min_dim: int = 50):
    """The same 2D weight matrices wwjd analyses (>= min_dim on both axes)."""
    out = {}
    for nm, p in model.named_parameters():
        if nm.endswith("weight") and p.ndim >= 2:
            W = p.detach().float().numpy().reshape(p.shape[0], -1)
            if min(W.shape) >= min_dim:
                out[nm] = W
    return out
