"""Apply the four added Bayesian diagnostics to the WeightWatcher-replication data.

  #1 EIV regression  : VGG accuracy ~ alpha_hat with alpha_hat uncertainty propagated
                       -> calibrated posterior slope + the HT-SR direction prob.
  #2 TPL / GPD       : on GPT's large-alpha layers, does the truncated power law beat
                       plain PL? (the explicit form of the over-estimation the BMA found)
  #6 ROPE            : how many layers are practically alpha=2 (posterior mass in band)?
  #5 prior-sensitivity: are the GPT conclusions prior-driven? (should be ~no)

Reads the per-layer CSVs written by ww_replication.py for #1 (alpha_hat + SE) and
reloads openai-gpt for the layer-level #2/#5/#6 (which need eigenvalues).

Usage: uv run --extra benchmarks --extra bayes python benchmarks/ww_bayes_diagnostics.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import jax.numpy as jnp

import wwj
from wwj.core import _eigvals

DATA = Path("/data/mhough/wwj_ww_replication")
VGG_TOP1 = {"vgg11":69.020,"vgg13":69.928,"vgg16":71.592,"vgg19":72.376,
            "vgg11_bn":70.370,"vgg13_bn":71.586,"vgg16_bn":73.360,"vgg19_bn":74.218}


def vgg_alpha_hat_with_se():
    """Per-model weighted-alpha alpha_hat (posterior mean) + its standard error,
    propagated from per-layer alpha posterior SDs through the weighted mean."""
    rows = []
    for v, acc in VGG_TOP1.items():
        df = pd.read_csv(DATA / f"{v}_layers.csv")
        w = df["log10_snorm2"].to_numpy()
        a = df["alpha_bayes"].to_numpy()
        sd = (df["ci_high"].to_numpy() - df["ci_low"].to_numpy()) / (2 * 1.959964)  # ~Gaussian CI -> SD
        ok = np.isfinite(a) & np.isfinite(sd)
        wa = (a * w)[ok]; sd_wa = (np.abs(w) * sd)[ok]; N = ok.sum()
        ahat = wa.mean()
        se = np.sqrt(np.sum(sd_wa ** 2)) / N            # SE of a mean of independent terms
        rows.append({"model": v, "acc": acc, "alpha_hat": ahat, "alpha_hat_se": se})
    return pd.DataFrame(rows)


def gpt_layers(min_dim=50):
    from transformers import AutoModel
    m = AutoModel.from_pretrained("openai-community/openai-gpt").eval()
    out = []
    for nm, p in m.named_parameters():
        if nm.endswith("weight") and p.ndim >= 2:
            W = p.detach().float().cpu().numpy().reshape(p.shape[0], -1)
            if min(W.shape) >= min_dim:
                out.append((nm, _eigvals(jnp.asarray(W.astype(np.float32)))))
    return out


def main():
    print("=" * 90 + "\n#1  Errors-in-variables: VGG accuracy ~ alpha_hat (uncertainty propagated)\n" + "=" * 90)
    d = vgg_alpha_hat_with_se()
    print(d.to_string(index=False))
    reg = wwj.accuracy_alpha_regression(d["alpha_hat"].to_numpy(), d["alpha_hat_se"].to_numpy(),
                                        d["acc"].to_numpy(), method="nuts")
    print(f"\n  posterior slope (acc per unit alpha_hat): {reg['slope_mean']:+.2f} "
          f"[{reg['slope_ci_low']:+.2f}, {reg['slope_ci_high']:+.2f}]   "
          f"P(slope<0) = {reg['prob_slope_neg']:.3f}   (HT-SR: smaller alpha_hat -> higher acc)")
    print(f"  naive OLS slope (ignores alpha_hat error): {reg['naive_ols_slope']:+.2f}   "
          f"residual sigma = {reg['sigma_mean']:.2f}pp   rhat={reg.get('slope_rhat', float('nan')):.3f}")

    print("\n" + "=" * 90 + "\n#2  TPL / GPD on GPT's large-alpha layers (does truncation beat plain PL?)\n" + "=" * 90)
    layers = gpt_layers()
    rows = []
    for nm, e in layers:
        mp = wwj.model_posterior(e, extended=True)
        rope = wwj.prob_in_rope(e, center=2.0, delta=0.25)
        sens = wwj.prior_sensitivity(e)
        rows.append({"layer": nm, "best_model": mp["best_model"],
                     "prob_pl": mp["prob_powerlaw"], "prob_tpl": mp["prob_truncated_powerlaw"],
                     "prob_gpd": mp["prob_generalized_pareto"],
                     "logbf_pl_vs_tpl": mp["logbf_pl_vs_tpl"], "logbf_pl_vs_gpd": mp["logbf_pl_vs_gpd"],
                     "prob_in_rope2": rope["prob_in_rope"], "prior_sens": sens["sensitivity"]})
    r = pd.DataFrame(rows)
    r.to_csv(DATA / "openai-gpt_bayes_diagnostics.csv", index=False)
    nbig = (r["best_model"] == "truncated_powerlaw").sum()
    print(f"  {len(r)} layers. best_model counts: " +
          ", ".join(f"{k}={v}" for k, v in r['best_model'].value_counts().items()))
    print(f"  layers where TPL beats plain PL (logbf_pl_vs_tpl<0): {(r['logbf_pl_vs_tpl']<0).sum()}/{len(r)} "
          f"(TPL preferred = the over-estimation correction made explicit)")
    print(f"  mean logbf PL-vs-TPL = {r['logbf_pl_vs_tpl'].mean():+.2f}  PL-vs-GPD = {r['logbf_pl_vs_gpd'].mean():+.2f}")

    print("\n" + "=" * 90 + "\n#6  ROPE: how many GPT layers are practically alpha=2 (P in [1.75,2.25])?\n" + "=" * 90)
    print(f"  layers with P(alpha in ROPE) > 0.5: {(r['prob_in_rope2']>0.5).sum()}/{len(r)};  "
          f"mean P_in_ROPE = {r['prob_in_rope2'].mean():.3f}  (low = GPT sits ABOVE 2, not at the RG optimum)")

    print("\n" + "=" * 90 + "\n#5  Prior sensitivity (are the GPT conclusions prior-driven?)\n" + "=" * 90)
    print(f"  mean prior-sensitivity across GPT layers = {r['prior_sens'].mean():.3f} "
          f"(in posterior-SD units; <<1 = robust, conclusions not prior-driven)")
    print(f"\n[done] {DATA/'openai-gpt_bayes_diagnostics.csv'}")


if __name__ == "__main__":
    main()
