"""Fully-Bayesian covariance-α + age-β on HBN modalities — the improved-Thompson analysis (no bootstrap).

Uses wwj's posterior machinery directly:
  * per modality: posterior over the density-tail α (mean, 95% credible interval, P(α<2)=posterior prob
    super-critical), the Bayes-factor validity gate (pl-vs-lognormal AND pl-vs-exponential) and the PPC.
  * learned-FM vs classical: propagate the per-modality P(α<2) into a group-level posterior separation
    (P(all learned super-critical), P(no classical super-critical), group-mean P(α<2)).
  * alignment-β on AGE (our one predictive-null-surviving target): P(β>1) and regime *probability*.

This is Thompson's exponent estimated Bayesianly, gated for power-law validity, and — unlike the
vanilla point-slope method — reported with full posterior uncertainty.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/hbn_bayesian_alpha.py
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

VC = "/data/derivatives/volume_conduction"
# classical/hand-built modalities; everything else (all FM-cache models + structural FMs) is learned
CLASSICAL = {"morph_4s456", "morph_4s1056", "blockpooled", "connectome"}


def _age_map():
    rows = list(csv.DictReader(open(MANIFEST)))
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    return {r["sub"]: f(r.get("age", "")) for r in rows}


def _post(eigs):
    e = jnp.asarray(np.asarray(eigs, dtype=float))
    ap, mp, ppc = alpha_posterior(e), model_posterior(e), ppc_pvalue(e)
    gate = (mp["logbf_pl_vs_ln"] > 0) and (mp["logbf_pl_vs_exp"] > 0) and (ppc["p_value"] >= 0.05)
    return ap, mp, ppc, bool(gate)


def main():
    agem = _age_map()
    mods = []  # (name, X, age_aligned)
    feats, tgts = load_hbn()
    for m in feats:
        age = np.asarray(tgts[m].get("age", np.full(len(feats[m]), np.nan)), float)
        mods.append((m, np.asarray(feats[m], float), age))
    for name, sub in [("amaes_fm", "amaes_embed/fomo25_embeddings.npz"),
                      ("fomo60k_fm", "fomo60k_embed/fomo60k_embeddings.npz")]:
        X, ids = fm_embed(f"{FMR}/{sub}")
        mods.append((name, X, np.array([agem.get(s, np.nan) for s in ids])))
    for name, fn in [("morph_4s456", "morphometry_4s456.npz"), ("blockpooled", "structural_emb.npz")]:
        d = np.load(f"{VC}/{fn}", allow_pickle=True)
        ids = [str(i).replace("sub-", "") for i in d["ids"]]
        mods.append((name, _zscore(d["X"].astype(float)), np.array([agem.get(s, np.nan) for s in ids])))
    Xc, idc = _connectome("456")
    mods.append(("connectome", Xc, np.array([agem.get(s, np.nan) for s in idc])))

    print("== COVARIANCE-α (posterior; validity-gated) ==")
    print(f"{'modality':13s} {'group':9s} {'α_mean':>7s} {'95% CI':>14s} {'P(α<2)':>7s} "
          f"{'BF_ln':>7s} {'BF_exp':>7s} {'ppc':>5s} {'gate':>5s}", flush=True)
    pbygroup = {"learned": [], "classical": []}
    alphas_o, groups_o = [], []
    for name, X, _ in mods:
        ap, mp, ppc, gate = _post(covariance_spectrum(X))
        g = "classical" if name in CLASSICAL else "learned"
        pbygroup[g].append(ap["p_alpha_lt_2"])
        alphas_o.append(float(ap["alpha_mean"])); groups_o.append(g)
        print(f"{name:13s} {g:9s} {ap['alpha_mean']:7.2f} "
              f"[{ap['ci_low']:.2f},{ap['ci_high']:.2f}]".rjust(14) +
              f" {ap['p_alpha_lt_2']:7.2f} {mp['logbf_pl_vs_ln']:7.1f} {mp['logbf_pl_vs_exp']:7.1f} "
              f"{ppc['p_value']:5.2f} {str(gate):>5s}", flush=True)

    lp, cp = np.array(pbygroup["learned"]), np.array(pbygroup["classical"])
    print("\n== learned-vs-classical group posterior (from per-modality P(α<2)) ==")
    print(f"  learned   P(α<2): mean {lp.mean():.3f}   P(ALL learned super-critical) = {np.prod(lp):.3f}")
    print(f"  classical P(α<2): mean {cp.mean():.3f}   P(NO classical super-critical) = {np.prod(1-cp):.3f}")

    # exact permutation test of the learned-vs-classical α separation (enumerable → exact p)
    from itertools import combinations
    alphas = np.array(alphas_o)
    is_learned = np.array([g == "learned" for g in groups_o])
    def stat(mask):
        return float(alphas[~mask].mean() - alphas[mask].mean())   # classical − learned; larger = more separated
    obs = stat(is_learned)
    n, k = len(alphas), int(is_learned.sum())
    combos = list(combinations(range(n), k))
    null = np.array([stat(np.isin(np.arange(n), c)) for c in combos])
    p_exact = float(np.mean(null >= obs))
    print("\n== EXACT permutation test: learned α < classical α (label-exchangeability null) ==")
    print(f"  observed Δ(classical−learned) = {obs:.3f};  exact p = {p_exact:.4f}  "
          f"({len(combos)} assignments; min achievable p = {1/len(combos):.4f})")

    print("\n== ALIGNMENT-β on AGE (predictive-null-surviving target; posterior) ==")
    print(f"{'modality':13s} {'β_mean':>7s} {'P(β>1)':>7s} {'gate':>5s}  regime(prob)", flush=True)
    for name, X, age in mods:
        m = np.isfinite(age)
        if int(m.sum()) < MIN_VALID:
            continue
        ap, mp, ppc, gate = _post(alignment_spectrum(X[m], age[m]))
        beta = rank_decay_from_density_alpha(ap["alpha_mean"])
        p = ap["p_alpha_lt_2"]  # P(β>1) = P(α_align<2)
        reg = f"variance-limited (P={p:.2f})" if p >= 0.5 else f"resolution-limited (P={1-p:.2f})"
        print(f"{name:13s} {beta:7.2f} {p:7.2f} {str(gate):>5s}  {reg}", flush=True)


if __name__ == "__main__":
    main()
