"""GPT/GPT2 large-alpha layers: Bayesian goodness-of-fit detail (shortlist case #1).

Charles's Nature-Comms claim is that GPT's "numerous unusually large alpha" layers
are "not well-described by a PL fit". This script tests that hand-wave on the exact
layers with two independent Bayesian tools, per layer:
  - the log Bayes factors PL-vs-exponential and PL-vs-lognormal (is a power law the
    PREFERRED tail model?), and
  - the posterior-predictive p-value (is the best PL ADEQUATE in absolute terms?
    high p = the observed KS discrepancy is typical of data the fitted PL generates),
alongside the frequentist alpha, the BMA posterior alpha + 95% CI, and the
xmin-window-pick share of the alpha variance (how much of the point estimate's
'large alpha' is just the single-KS-window choice).

Usage: uv run --extra benchmarks --extra bayes python benchmarks/ww_gpt_detail.py
"""
from __future__ import annotations
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pandas as pd
import jax.random as jr

import wwj
from wwj.core import _eigvals


def layers_of(hf_name: str, min_dim: int = 50):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(hf_name).eval()
    out = []
    for nm, p in m.named_parameters():
        if nm.endswith("weight") and p.ndim >= 2:
            W = p.detach().float().cpu().numpy().reshape(p.shape[0], -1)
            if min(W.shape) >= min_dim:
                out.append((nm, W.astype(np.float32)))
    return out


def detail(hf_name: str, key0: int = 0) -> pd.DataFrame:
    rows = []
    for i, (nm, W) in enumerate(layers_of(hf_name)):
        eigs = _eigvals(jnp.asarray(W))
        freq = wwj.analyze_matrix(jnp.asarray(W), name=nm, mode="csn")
        bma = wwj.alpha_posterior_bma(eigs)
        mp = wwj.model_posterior(eigs)
        ppc = wwj.ppc_pvalue(eigs, key=jr.PRNGKey(key0 + i))
        rows.append({
            "layer": nm, "shape": f"{W.shape[0]}x{W.shape[1]}",
            "alpha_freq": float(freq.alpha),
            "alpha_bayes": bma["alpha_mean"], "ci_low": bma["ci_low"], "ci_high": bma["ci_high"],
            # fraction of the alpha variance that is the xmin-window choice (vs within-window)
            "xmin_var_share": float(bma["across_window_var"] /
                                    max(bma["across_window_var"] + bma["within_window_var"], 1e-12)),
            "best_model": mp["best_model"], "prob_powerlaw": mp["prob_powerlaw"],
            "logbf_pl_vs_exp": mp["logbf_pl_vs_exp"], "logbf_pl_vs_ln": mp["logbf_pl_vs_ln"],
            "ppc_pvalue": ppc["p_value"],
        })
    return pd.DataFrame(rows)


def main():
    out_dir = Path("/data/mhough/wwj_ww_replication"); out_dir.mkdir(parents=True, exist_ok=True)
    for hf, tag in [("openai-community/openai-gpt", "openai-gpt"), ("openai-community/gpt2", "gpt2")]:
        print(f"\n=== {tag} ===", flush=True)
        df = detail(hf)
        df.to_csv(out_dir / f"{tag}_gptdetail.csv", index=False)
        # the layers that drive Charles's "large alpha = poorly trained" claim
        big = df[df["alpha_freq"] > 5.0].sort_values("alpha_freq", ascending=False)
        print(f"  {len(df)} layers; {len(big)} with frequentist alpha>5 (Charles's 'unusually large alpha'):")
        if len(big):
            show = big[["shape", "alpha_freq", "alpha_bayes", "ci_low", "ci_high",
                        "xmin_var_share", "logbf_pl_vs_exp", "logbf_pl_vs_ln", "ppc_pvalue", "best_model"]]
            with pd.option_context("display.width", 200, "display.max_columns", None,
                                   "display.float_format", "{:.3f}".format):
                print(show.head(10).to_string(index=False))
        # verdicts
        pl_pref = ((df["logbf_pl_vs_exp"] > 0) & (df["logbf_pl_vs_ln"] > 0)).mean()
        ppc_ok = (df["ppc_pvalue"] > 0.05).mean()
        print(f"  PL preferred (both BF>0): {pl_pref:.0%}   |   PL adequate (ppc p>0.05): {ppc_ok:.0%}")
        print(f"  mean alpha: freq={df['alpha_freq'].mean():.3f}  bayes={df['alpha_bayes'].mean():.3f}  "
              f"mean xmin-var-share={df['xmin_var_share'].mean():.0%}")
    print("\n[done] /data/mhough/wwj_ww_replication/{openai-gpt,gpt2}_gptdetail.csv")


if __name__ == "__main__":
    main()
