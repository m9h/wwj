"""Fair model comparison: power-law must be scored OUT-OF-SAMPLE (leave-one-N-out), like CBP already is.

zeta_tierB/zeta_robust fit the power-law's (c, N0) by minimizing MSE directly against the SAME curve
points used to score it — in-sample. Robust-CBP and CBP-sorted use ZERO parameters fit to the curve (only
the frozen representation's spectra + a single shared ceiling anchor). That's not a fair contest; it
inflates the power-law's apparent win. Here the power-law is refit via leave-one-N-out CV: for each held-
out N, fit (c,N0) on the OTHER points, predict the held-out one, aggregate. Also multi-seed the synthetic
stage-2 check (single-seed curves are noisy, capping even CBP-true's apparent skill) to separate real
model gap from curve-measurement noise.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/zeta_fair.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeta_cbp import solve_kappa  # noqa: E402
from zeta_tierB import maps, load_fm_cache, curve  # noqa: E402
from zeta_robust import spectra_raw, robust_g2, cbp_curve, skill  # noqa: E402
from zeta_estimator import spectra_coeff, madaptive_g2, noise_floor  # noqa: E402
from hbn_fm_existing import fm_embed, R as FMR  # noqa: E402


def pl_pred_loo(Ns, r, ceil):
    """Leave-one-N-out power-law prediction at every point (each held out when fit)."""
    preds = np.empty_like(r)
    for i in range(len(Ns)):
        mask = np.arange(len(Ns)) != i
        best = (np.inf, None)
        for c in np.linspace(0.3, 2.5, 12):
            for n0 in np.logspace(1, 3.5, 12):
                pred_tr = ceil * Ns[mask] ** c / (Ns[mask] ** c + n0 ** c)
                mse = float(np.mean((pred_tr - r[mask]) ** 2))
                if mse < best[0]:
                    best = (mse, (c, n0))
        c, n0 = best[1]
        preds[i] = ceil * Ns[i] ** c / (Ns[i] ** c + n0 ** c)
    return preds


def pl_pred_insample(Ns, r, ceil):
    best = (np.inf, None)
    for c in np.linspace(0.3, 2.5, 12):
        for n0 in np.logspace(1, 3.5, 12):
            pred = ceil * Ns ** c / (Ns ** c + n0 ** c)
            mse = float(np.mean((pred - r) ** 2))
            if mse < best[0]:
                best = (mse, pred)
    return best[1]


# ------------------------------------------------------------------- stage 2b: multi-seed synthetic ceiling
def stage2b_multiseed():
    print("== Stage 2b: multi-seed synthetic — does curve-measurement noise cap CBP-true's own skill? ==")
    P, NB = 150, 1500
    NS = [50, 100, 200, 400, 800]
    rho = np.arange(1, P + 1.0)
    for a, b, snr in [(0.8, 0.5, 1.0), (1.5, 1.5, 4.0), (1.5, 0.5, 4.0)]:
        lam_t = rho ** (-a); g2_t = rho ** (-b); g2_t = g2_t / g2_t.sum()
        g = np.sqrt(g2_t); sigma = np.sqrt(1.0 / snr); ridge = float(lam_t.sum())
        curves = []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            Z = rng.standard_normal((NB, P)); X = Z * np.sqrt(lam_t); yv = Z @ g + sigma * rng.standard_normal(NB)
            _, r = curve(X, yv, ridge, seed=seed)
            if len(r) == len(NS):
                curves.append(r)
        curves = np.array(curves)
        r_mean = curves.mean(0)                      # multi-seed averaged curve = low-noise ground truth
        ceil = float(r_mean[-1])
        Ncur = np.array(NS, float)
        cc_true = cbp_curve(lam_t, g2_t.copy(), ridge, Ncur, ceil)
        s_true_vs_mean = skill(cc_true, r_mean)
        s_true_vs_single = np.mean([skill(cc_true, curves[i]) for i in range(len(curves))])
        print(f"  a={a} b={b} snr={snr}: skill(CBP-true, MULTI-seed mean)={s_true_vs_mean:+.2f}   "
              f"mean skill(CBP-true, SINGLE-seed)={s_true_vs_single:+.2f}   (gap = curve-measurement noise)")


# ------------------------------------------------------------------- stage 3b: fair (LOO) real comparison
def stage3b_fair_real():
    print("\n== Stage 3b: real Tier-B, power-law now LEAVE-ONE-N-OUT (fair vs 0-param CBP) ==")
    rows, age, sex = maps()
    reps = {}
    for m in ["neurostorm", "swift", "cortex_mae_volume"]:
        X, subs = load_fm_cache(m, rows)
        if len(subs) >= 300:
            reps[m] = {s: X[i] for i, s in enumerate(subs)}
    for name, sub in [("amaes_fm", "amaes_embed/fomo25_embeddings.npz"),
                      ("fomo60k_fm", "fomo60k_embed/fomo60k_embeddings.npz")]:
        X, ids = fm_embed(f"{FMR}/{sub}"); reps[name] = {s: X[i] for i, s in enumerate(ids)}

    sk = {"madaptive-CBP": [], "robust-CBP": [], "CBP-sorted": [], "power-law-LOO": [], "power-law-insample": []}
    skhi = {k: [] for k in sk}
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
        w2, coeff = spectra_coeff(XA, y)                             # same eigenvalues, plus per-mode coeff
        g2m = madaptive_g2(w2, coeff, float(np.var(y)) * (1.0 - ceil) / len(y))   # σ²/M from measured ceiling
        cm = cbp_curve(w2, g2m, ridge, Ns, ceil)
        cc = cbp_curve(w, g2r, ridge, Ns, ceil)
        cs = cbp_curve(np.sort(w)[::-1], np.sort(g2r)[::-1], ridge, Ns, ceil)
        pl_loo = pl_pred_loo(Ns, r, ceil)
        pl_in = pl_pred_insample(Ns, r, ceil)
        for k, pred in [("madaptive-CBP", cm), ("robust-CBP", cc), ("CBP-sorted", cs),
                        ("power-law-LOO", pl_loo), ("power-law-insample", pl_in)]:
            s = skill(pred, r)
            sk[k].append(s)
            if ceil >= 0.30:
                skhi[k].append(s)
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

    print(f"{npts} curves ({nhi} high-ceiling >=0.30).  mean skill (higher=better):")
    for k in ["madaptive-CBP", "robust-CBP", "CBP-sorted", "power-law-LOO", "power-law-insample"]:
        hi = f"{np.mean(skhi[k]):+.3f}" if skhi[k] else "n/a"
        print(f"  {k:19s} all={np.mean(sk[k]):+.3f}   high-ceiling={hi}")
    print("\n(power-law-insample repeated for reference -- the previously-reported, unfairly-favourable number)")


if __name__ == "__main__":
    stage2b_multiseed()
    stage3b_fair_real()
