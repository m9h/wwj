"""Canatar–Bordelon–Pehlevan deterministic-equivalent learning curve — implement + VALIDATE on synthetic.

The proper model: from a representation's covariance eigenvalues {λρ} and the target's per-mode power
{gρ²} plus noise σ² and ridge, the CBP theory predicts the FULL learning curve E_gen(N) — parameter-free.
This uses BOTH capacity (eigenvalues) and source (position-preserved target power), which ρ_q only
summarises. Step 1 validates the implementation: does the parameter-free CBP R²(N) match simulated ridge
learning curves across a (capacity a, source b, SNR) grid? If yes, the tool is trustworthy for real data.

Deterministic equivalent (ridge λr on the (1/N)-normalised objective):
  κ solves  κ = λr + κ Σρ λρ/(N λρ + κ);   Lρ = N λρ/(N λρ + κ);   γ = (1/N) Σρ Lρ²
  E_g(N)   = [ Σρ (1-Lρ)² gρ²  +  σ² γ ] / (1-γ)          (signal excess risk)
  R²(N)    = 1 - (E_g + σ²) / (Σρ gρ² + σ²)

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/zeta_cbp.py
"""
from __future__ import annotations

import numpy as np


def solve_kappa(lam, N, R):
    """κ solves  κ = R + κ Σρ λρ/(N λρ + κ), with R the ABSOLUTE ridge penalty on X^T X."""
    def f(k):
        return R + k * np.sum(lam / (N * lam + k)) - k
    lo, hi = 1e-14, 1e8
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cbp_r2(lam, g2, sigma2, N, R):
    if N <= 0:
        return 0.0
    k = solve_kappa(lam, N, R)
    L = N * lam / (N * lam + k)
    gamma = float(np.sum(L ** 2) / N)
    Eg = (float(np.sum((1 - L) ** 2 * g2)) + sigma2 * gamma) / max(1 - gamma, 1e-9)
    return 1.0 - (Eg + sigma2) / (float(g2.sum()) + sigma2)


def empirical_r2(lam, g, sigma, N, ridge_emp, ntest=2000, seeds=(0, 1, 2)):
    p = len(lam); r2 = []
    for s in seeds:
        rng = np.random.default_rng(s)
        Ztr = rng.standard_normal((N, p)); Xtr = Ztr * np.sqrt(lam)
        ytr = Ztr @ g + sigma * rng.standard_normal(N)
        Zte = rng.standard_normal((ntest, p)); Xte = Zte * np.sqrt(lam)
        yte = Zte @ g + sigma * rng.standard_normal(ntest)
        mu, ym = Xtr.mean(0), ytr.mean()
        w = np.linalg.solve((Xtr - mu).T @ (Xtr - mu) + ridge_emp * np.eye(p), (Xtr - mu).T @ (ytr - ym))
        pred = (Xte - mu) @ w + ym
        r2.append(1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2))
    return float(np.mean(r2))


def main():
    P = 200
    NS = [50, 100, 200, 400, 800, 1600, 3200]
    RIDGE = 3.0                                       # FIXED absolute penalty on X^T X; same in theory + sim
    rho = np.arange(1, P + 1.0)
    print(f"CBP theory vs simulated ridge learning curves  (p={P}, fixed absolute ridge={RIDGE})")
    print(f"{'a':>4s} {'b':>4s} {'snr':>4s} | " + " ".join(f"N={n:<5d}" for n in NS) + "   |  MAE", flush=True)
    all_emp, all_cbp = [], []
    for a in (0.8, 1.5):
        for b in (0.5, 1.5):
            for snr in (1.0, 4.0):
                lam = rho ** (-a)
                g2 = rho ** (-b); g2 = g2 / g2.sum()          # target power per mode, Σ=1
                g = np.sqrt(g2)
                sigma2 = 1.0 / snr; sigma = np.sqrt(sigma2)
                emp = [empirical_r2(lam, g, sigma, N, RIDGE) for N in NS]
                cbp = [cbp_r2(lam, g2, sigma2, N, RIDGE) for N in NS]
                all_emp += emp; all_cbp += cbp
                mae = float(np.mean(np.abs(np.array(emp) - np.array(cbp))))
                cells = " ".join(f"{e:.2f}/{c:.2f}" for e, c in zip(emp, cbp))
                print(f"{a:4.1f} {b:4.1f} {snr:4.0f} | {cells}   | {mae:.3f}", flush=True)
    e, c = np.array(all_emp), np.array(all_cbp)
    print(f"\n(each cell = empirical/CBP R²)   overall MAE = {np.mean(np.abs(e-c)):.3f}   "
          f"Pearson(emp,CBP) = {np.corrcoef(e, c)[0,1]:.4f}   over {len(e)} points")
    print("If MAE is small and Pearson≈1, the parameter-free CBP curve is validated — ready for real data.")


if __name__ == "__main__":
    main()
