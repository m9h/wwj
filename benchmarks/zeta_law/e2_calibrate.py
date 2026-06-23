"""E2 calibration over the full HBN cache — the real test of the zeta-law `n*` extrapolations.

For every cached modality × target: predict the learning curve + credible band from the spectra
(E1/E2), measure the observed ridge-CV learning curve by within-cohort subsampling, and report the band
coverage, the observed increment ratio (an empirical saturation check), and `n*`. Pure-numpy (no
sklearn); runs in the wwj `.venv`.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/e2_calibrate.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbn_e1 import load_hbn, EMB_DIR                                              # noqa: E402
from zeta_diagnostic import covariance_spectrum, alignment_spectrum, diagnose_spectrum  # noqa: E402
from e2_learning_curve import (                                                   # noqa: E402
    learning_curve_posterior, observed_curve, n_star, calibration_coverage, _rescale,
)


def main() -> None:
    if not (EMB_DIR.exists() and any(EMB_DIR.glob("*.npz"))):
        print(f"HBN cache not found at {EMB_DIR}"); return
    feats, tgts = load_hbn()
    rows = []
    for model, X in feats.items():
        n = X.shape[0]
        cov = covariance_spectrum(X)
        cd = diagnose_spectrum(cov, "covariance")
        N = np.unique(np.clip(np.array([150, 300, n // 2, n]), 50, n)).astype(float)
        for t, y in tgts[model].items():
            y = np.asarray(y, float); mask = np.isfinite(y)
            al = alignment_spectrum(X[mask], y[mask])
            band = learning_curve_posterior(cov, al, N, n_draws=600)
            obs = observed_curve(X, y, N)
            rel = {k: _rescale(band[k]) for k in ("lo", "median", "hi")}
            cov_frac = calibration_coverage(rel, _rescale(obs))
            inc = np.diff(obs)                                  # observed increments
            obs_inc_ratio = float(inc[-1] / inc[0]) if np.isfinite(inc).all() and abs(inc[0]) > 1e-9 else float("nan")
            ns = n_star(band["gamma_median"], band["beta_median"])
            rows.append({
                "modality": model, "n": int(n), "target": t,
                "gamma": round(cd["rank_decay"], 2), "gamma_powerlaw": bool(cd["is_power_law"]),
                "beta_med": round(band["beta_median"], 2), "p_no_sat": round(band["p_no_saturation"], 2),
                "n_star": None if not np.isfinite(ns) else int(ns),
                "coverage": round(cov_frac, 2), "r_at_max": round(float(np.nanmax(obs)), 3),
                "obs_inc_ratio": round(obs_inc_ratio, 2), "N_grid": [int(v) for v in N],
            })

    hdr = ["modality", "n", "target", "γ", "β", "P(no-sat)", "n*", "cover", "r@max", "obs_inc"]
    print(f"# E2 calibration on HBN — {len(feats)} modalities (within-cohort subsampling)\n")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        print(f"| {r['modality']} | {r['n']} | {r['target']} | {r['gamma']} | {r['beta_med']} | "
              f"{r['p_no_sat']} | {r['n_star'] if r['n_star'] is not None else 'inf'} | "
              f"{r['coverage']} | {r['r_at_max']} | {r['obs_inc_ratio']} |")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_calibration_results.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    decade = np.log10(max(r["N_grid"][-1] for r in rows) / min(r["N_grid"][0] for r in rows))
    print(f"\nwrote {out}")
    print(f"Note: within-cohort N spans ~{decade:.1f} decade(s) — shape calibration (coverage/obs_inc) is "
          f"UNDER-POWERED at this scale; the n* extrapolations need the full/larger cohort to confirm.\n"
          f"obs_inc > 1 ⇒ observed still accelerating (no saturation, β≤1-like); < 1 ⇒ decelerating "
          f"(saturating, β>1-like).")


if __name__ == "__main__":
    main()
