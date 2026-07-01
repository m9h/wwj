"""Robust spectrum estimation for CBP: does the theory predict real curves once gρ² is estimated properly?

Stage 1 — robust estimator: naive gρ²=coeffρ²/λρ carries a per-mode noise floor σ²/n; we (a) debias by
   subtracting the tail-median floor and (b) truncate the un-resolvable deep tail (λρ < tol·λmax).
Stage 2 — synthetic validation: on finite noisy synthetic data, CBP with TRUE / NAIVE-estimated /
   ROBUST-estimated spectra. Expect true≈good, naive≈broken, robust≈recovered. Earns the estimator.
Stage 3 — real Tier-B re-run: robust-CBP vs CBP-sorted vs free power-law vs null on FM→FM + age/sex.
   Verdict: robust-CBP ≈ power-law → theory confirmed, estimation was the problem. Still fails → theory
   incomplete on real data and coarse summaries (ρ_q) are the ceiling.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/zeta_robust.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeta_cbp import solve_kappa  # noqa: E402
from zeta_tierB import maps, load_fm_cache, curve, pl_mse  # noqa: E402
from hbn_fm_existing import fm_embed, R as FMR  # noqa: E402


def spectra_raw(X, y):
    Xc, yc = X - X.mean(0), y - y.mean()
    C = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(C); w, V = np.clip(w[::-1], 0, None), V[:, ::-1]
    coeff = V.T @ (Xc.T @ yc / len(Xc))
    return w, coeff ** 2 / (w + 1e-9 * w.max())          # (λ, naive gρ²)


def robust_g2(w, g2, tol=1e-2):
    floor = float(np.median(g2[len(g2) // 2:]))          # noise floor ≈ σ²/n from the (near-zero-power) tail
    g2 = np.clip(g2 - floor, 0.0, None)
    g2 = g2 * (w >= tol * w.max())                        # drop un-resolvable deep tail
    return g2


def cbp_curve(lam, g2, R, Ns, ceil):
    if g2.sum() <= 0:
        return np.full(len(Ns), np.nan)
    g2 = g2 / g2.sum() * ceil
    sigma2 = 1.0 - ceil; out = []
    for N in Ns:
        k = solve_kappa(lam, N, R)
        L = N * lam / (N * lam + k)
        gamma = min(float(np.sum(L ** 2) / N), 0.98)     # γ<1 safeguard
        Eg = (float(np.sum((1 - L) ** 2 * g2)) + sigma2 * gamma) / (1 - gamma)
        out.append(1.0 - (Eg + sigma2))
    return np.array(out)


def skill(pred, r):
    return 1.0 - float(np.mean((pred - r) ** 2)) / (float(np.var(r)) + 1e-9)


# ----------------------------------------------------------------- stage 2: synthetic validation
def stage_synth():
    print("== Stage 2: synthetic — CBP with TRUE vs NAIVE-est vs ROBUST-est spectra ==")
    print(f"{'a':>4s} {'b':>4s} {'snr':>4s} {'ceil':>5s} | skill: {'true':>6s} {'naive':>7s} {'robust':>7s}", flush=True)
    P, NB = 150, 1500
    Ns = np.array([50, 100, 200, 400, 800], float)
    rho = np.arange(1, P + 1.0)
    for a in (0.8, 1.5):
        for b in (0.5, 1.5):
            for snr in (1.0, 4.0):
                lam_t = rho ** (-a); g2_t = rho ** (-b); g2_t = g2_t / g2_t.sum()
                g = np.sqrt(g2_t); sigma = np.sqrt(1.0 / snr)
                rng = np.random.default_rng(0)
                Z = rng.standard_normal((NB, P)); X = Z * np.sqrt(lam_t); yv = Z @ g + sigma * rng.standard_normal(NB)
                ridge = float((X - X.mean(0)).var(0).sum() * P / P + lam_t.sum())  # ~ trace scale
                ridge = float(lam_t.sum())
                _, r = curve(X, yv, ridge)
                if len(r) < 4:
                    continue
                ceil = float(r[-1]); Ncur = np.array([50, 100, 200, 400, 800][:len(r)], float)
                w_hat, g2_naive = spectra_raw(X, yv)
                g2_rob = robust_g2(w_hat, g2_naive.copy())
                s_true = skill(cbp_curve(lam_t, g2_t.copy(), ridge, Ncur, ceil), r)
                s_naive = skill(cbp_curve(w_hat, g2_naive, ridge, Ncur, ceil), r)
                s_rob = skill(cbp_curve(w_hat, g2_rob, ridge, Ncur, ceil), r)
                print(f"{a:4.1f} {b:4.1f} {snr:4.0f} {ceil:5.2f} |        {s_true:+.2f} {s_naive:+7.2f} {s_rob:+7.2f}", flush=True)


# ----------------------------------------------------------------- stage 3: real Tier-B re-run
def stage_real():
    print("\n== Stage 3: real Tier-B — robust-CBP vs CBP-sorted vs power-law ==")
    rows, age, sex = maps()
    reps = {}
    for m in ["neurostorm", "swift", "cortex_mae_volume"]:
        X, subs = load_fm_cache(m, rows)
        if len(subs) >= 300:
            reps[m] = {s: X[i] for i, s in enumerate(subs)}
    for name, sub in [("amaes_fm", "amaes_embed/fomo25_embeddings.npz"),
                      ("fomo60k_fm", "fomo60k_embed/fomo60k_embeddings.npz")]:
        X, ids = fm_embed(f"{FMR}/{sub}"); reps[name] = {s: X[i] for i, s in enumerate(ids)}

    sk = {"robust-CBP": [], "CBP-sorted": [], "power-law": []}
    skhi = {"robust-CBP": [], "CBP-sorted": [], "power-law": []}
    npts = nhi = 0

    def run(XA, y):
        nonlocal npts, nhi
        y = (y - y.mean()) / (y.std() + 1e-9)
        Xc = XA - XA.mean(0); ridge = float((Xc.T @ Xc / len(XA)).trace())
        Ns, r = curve(XA, y, ridge)
        ceil = float(r[-1])
        if ceil < 0.05 or len(Ns) < 4:
            return
        w, g2n = spectra_raw(XA, y)
        g2r = robust_g2(w, g2n.copy())
        cc = cbp_curve(w, g2r, ridge, Ns, ceil)
        cs = cbp_curve(np.sort(w)[::-1], np.sort(g2r)[::-1], ridge, Ns, ceil)
        s_cc, s_cs = skill(cc, r), skill(cs, r)
        s_pl = 1 - pl_mse(Ns, r, ceil) / (float(np.var(r)) + 1e-9)
        for k, v in [("robust-CBP", s_cc), ("CBP-sorted", s_cs), ("power-law", s_pl)]:
            sk[k].append(v)
            if ceil >= 0.30:
                skhi[k].append(v)
        npts += 1; nhi += ceil >= 0.30

    for A, va in reps.items():
        idsA = list(va)
        for tmap in (age, sex):
            common = [s for s in idsA if s in tmap and np.isfinite(tmap[s])]
            if len(common) >= 300:
                run(np.array([va[s] for s in common]), np.array([tmap[s] for s in common]))
        for B, vb in reps.items():
            if B == A:
                continue
            common = [s for s in idsA if s in vb]
            if len(common) < 350:
                continue
            XB = np.array([vb[s] for s in common])
            U, S, _ = np.linalg.svd(XB - XB.mean(0), full_matrices=False)
            pcs = U[:, :4] * S[:4]; XA = np.array([va[s] for s in common])
            for t in range(4):
                run(XA, pcs[:, t])

    print(f"{npts} curves ({nhi} high-ceiling ≥0.30).  mean skill (higher=better):")
    for k in ["robust-CBP", "CBP-sorted", "power-law"]:
        hi = f"{np.mean(skhi[k]):+.3f}" if skhi[k] else "n/a"
        print(f"  {k:11s} all={np.mean(sk[k]):+.3f}   high-ceiling={hi}")


if __name__ == "__main__":
    stage_synth()
    stage_real()
