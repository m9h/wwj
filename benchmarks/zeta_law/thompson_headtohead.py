"""Head-to-head: Thompson's ORIGINAL point-slope method vs our Bayesian+gated method, same spectra.

Thompson (and the nanopath writeup): the exponent is the raw log-log slope of the sorted spectrum, a
point estimate, no validity gate, read regardless of whether the target is predictable.
Ours: Bayesian posterior on α/β (mean, P), the Bayes-factor + PPC validity gate, and — for alignment —
a predictive gate (only interpret β if the target clears a CV null).

Three tables show where they agree (headline) and where they diverge (borderline classification; a
false psychopathology "finding" that only the predictive gate catches).

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/thompson_headtohead.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wwj import alpha_posterior, model_posterior, ppc_pvalue  # noqa: E402
from zeta_diagnostic import covariance_spectrum, alignment_spectrum, rank_decay_from_density_alpha  # noqa: E402
from hbn_e1 import load_hbn, MANIFEST, MIN_VALID  # noqa: E402
from hbn_structural import _zscore, _connectome  # noqa: E402
from hbn_fm_existing import fm_embed, R as FMR  # noqa: E402
from hbn_predictive_check import cv_r2_null  # noqa: E402

VC = "/data/derivatives/volume_conduction"
# hand-built / classical modalities; everything else (all FM-cache models + structural FMs) = learned
CLASSICAL = {"morph_4s456", "morph_4s1056", "blockpooled", "connectome"}
TGTS = ["age", "p_factor"]


def thompson_slope(spectrum):
    """Thompson's estimator: rank-decay exponent s = −OLS slope of log(value) vs log(rank), descending."""
    v = np.sort(np.asarray(spectrum, dtype=float))[::-1]
    v = v[v > v.max() * 1e-6]
    if len(v) < 5:
        return float("nan")
    r = np.arange(1, len(v) + 1)
    return float(-np.polyfit(np.log(r), np.log(v), 1)[0])


def our_post(eigs):
    e = jnp.asarray(np.asarray(eigs, dtype=float))
    ap, mp, ppc = alpha_posterior(e), model_posterior(e), ppc_pvalue(e)
    gate = (mp["logbf_pl_vs_ln"] > 0) and (mp["logbf_pl_vs_exp"] > 0) and (ppc["p_value"] >= 0.05)
    return ap, bool(gate)


def _tmap():
    rows = list(csv.DictReader(open(MANIFEST)))
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    return {r["sub"]: {t: f(r.get(t, "")) for t in TGTS} for r in rows}


def main():
    tg = _tmap()
    mods = []  # (name, X, {target: y})
    feats, tgts = load_hbn()
    for m in feats:
        ys = {t: np.asarray(tgts[m].get(t, np.full(len(feats[m]), np.nan)), float) for t in TGTS}
        mods.append((m, np.asarray(feats[m], float), ys))
    for name, sub in [("amaes_fm", "amaes_embed/fomo25_embeddings.npz"),
                      ("fomo60k_fm", "fomo60k_embed/fomo60k_embeddings.npz")]:
        X, ids = fm_embed(f"{FMR}/{sub}")
        ys = {t: np.array([tg.get(s, {}).get(t, np.nan) for s in ids]) for t in TGTS}
        mods.append((name, X, ys))
    for name, fn in [("morph_4s456", "morphometry_4s456.npz"), ("blockpooled", "structural_emb.npz")]:
        d = np.load(f"{VC}/{fn}", allow_pickle=True)
        ids = [str(i).replace("sub-", "") for i in d["ids"]]
        ys = {t: np.array([tg.get(s, {}).get(t, np.nan) for s in ids]) for t in TGTS}
        mods.append((name, _zscore(d["X"].astype(float)), ys))
    Xc, idc = _connectome("456")
    ys = {t: np.array([tg.get(s, {}).get(t, np.nan) for s in idc]) for t in TGTS}
    mods.append(("connectome", Xc, ys))

    print("=== TABLE A: COVARIANCE-α — Thompson point-slope vs Bayesian+gate ===")
    print(f"{'modality':13s} {'group':9s} | {'Thompson α_T':>12s} {'call':>10s} | "
          f"{'ours α':>7s} {'P(α<2)':>7s} {'gate':>5s} | agree?", flush=True)
    for name, X, _ in mods:
        eigs = covariance_spectrum(X)
        s = thompson_slope(eigs)
        aT = 1 + 1 / s if s > 0 else float("inf")
        ap, gate = our_post(eigs)
        tcall = "super-crit" if aT < 2 else "not"
        ocrit = ap["p_alpha_lt_2"] >= 0.5
        agree = "yes" if (aT < 2) == ocrit else "**NO**"
        g = "classical" if name in CLASSICAL else "learned"
        print(f"{name:13s} {g:9s} | {aT:12.2f} {tcall:>10s} | {ap['alpha_mean']:7.2f} "
              f"{ap['p_alpha_lt_2']:7.2f} {str(gate):>5s} | {agree}", flush=True)

    for tgt in TGTS:
        kind = "REAL target" if tgt == "age" else "NULL target"
        print(f"\n=== TABLE {'B' if tgt=='age' else 'C'}: ALIGNMENT-β on {tgt} ({kind}) — "
              f"Thompson vs ours + predictive gate ===")
        print(f"{'modality':13s} | {'Thompson β':>10s} {'→ regime':>20s} | {'ours β':>7s} "
              f"{'P(β>1)':>7s} | {'CV-R²':>7s} {'predictive verdict':>20s}", flush=True)
        for name, X, ys in mods:
            y = ys[tgt]
            m = np.isfinite(y)
            if int(m.sum()) < MIN_VALID:
                continue
            al = alignment_spectrum(X[m], y[m])
            bT = thompson_slope(al)
            treg = "variance-limited" if bT > 1 else "resolution-limited"
            ap, _ = our_post(al)
            ourb = rank_decay_from_density_alpha(ap["alpha_mean"])
            cv = cv_r2_null(X, y, n_perm=100)
            if cv is None:
                r2, verdict = float("nan"), "n/a"
            else:
                r2, _, p, _ = cv
                verdict = "REAL → β valid" if (p < 0.05 and r2 > 0) else "NULL → β is NOISE"
            print(f"{name:13s} | {bT:10.2f} {treg:>20s} | {ourb:7.2f} {ap['p_alpha_lt_2']:7.2f} | "
                  f"{r2:7.3f} {verdict:>20s}", flush=True)


if __name__ == "__main__":
    main()
