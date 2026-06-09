"""Replicate WeightWatcher's canonical published test cases with wwjd.

Runs both the frequentist fit (wwj.analyze_matrix mode='csn', the WeightWatcher
point estimate) and the Bayesian treatment (BMA alpha posterior over xmin +
Bayesian power-law-vs-alternative model comparison) on the exact checkpoints
Martin & Mahoney published on, so we can see where the posterior diverges from
the point estimate. See docs/ww_replication_targets.md for the ranked rationale.

Two flagship cases (the confirm-then-nuance arc):
  vgg  -- VGG11/13/16/19 (+BN) ImageNet [torchvision]. Charles's flagship claim:
          weighted-alpha  alpha_hat = sum_l alpha_l * log(lambda_max_l)  has a
          near-perfect linear relation to test accuracy. CONFIRM target -- does it
          survive with posterior error bars on alpha_hat?
  gpt  -- openai-gpt vs gpt2/medium/large/xl [HuggingFace]. Charles's claim:
          GPT2 better-trained; GPT has "numerous unusually large alpha ... not
          well-described by a PL fit". OVERTURN candidate -- do those large-alpha
          layers get formally rejected as power-law by a Bayes factor, and does
          the mean-alpha gap survive credible intervals?

Per-layer output: frequentist alpha + lambda_max, Bayesian alpha posterior
(mean / 95% CI / SD), the xmin-window-pick variance (how much alpha uncertainty
is just the scaling-window choice -- the thing WeightWatcher fixes by KS argmin),
P(alpha<2), the Bayesian best-fit distribution + P(power-law), and the PL-vs-exp
/ PL-vs-lognormal log Bayes factors.

Usage:
  uv run --extra benchmarks python benchmarks/ww_replication.py \
      --models vgg gpt --out-dir /data/mhough/wwj_ww_replication
  # subsets: --models vgg   |   --models gpt   |   --gpt-variants openai-gpt gpt2 gpt2-medium
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch

import wwj
from wwj.core import _eigvals


# torchvision IMAGENET1K_V1 published top-1 accuracy, for the alpha_hat-vs-accuracy
# trend (the metric whose linearity is Charles's flagship VGG result).
VGG_TOP1 = {
    "vgg11": 69.020, "vgg13": 69.928, "vgg16": 71.592, "vgg19": 72.376,
    "vgg11_bn": 70.370, "vgg13_bn": 71.586, "vgg16_bn": 73.360, "vgg19_bn": 74.218,
}
GPT_VARIANTS = ["openai-gpt", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


# ---------------------------------------------------------------------------
# checkpoint loaders -> list[(layer_name, 2D weight as np.float32)]
# ---------------------------------------------------------------------------
def _matrices_from_torch_module(module, min_dim: int) -> list[tuple[str, np.ndarray]]:
    """Every weight param with ndim>=2, reshaped to 2D as (shape[0], prod(rest)).
    Covers Linear (2D), Conv2d (4D), and HF Conv1D (2D) the same way WeightWatcher
    reshapes conv stacks to a matrix. Eigenvalues of W^T W are orientation-agnostic."""
    out = []
    for name, p in module.named_parameters():
        if not name.endswith("weight") or p.ndim < 2:
            continue
        W = p.detach().float().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1)
        if min(W2.shape) < min_dim:
            continue
        out.append((name, W2.astype(np.float32)))
    return out


def load_vgg(variant: str, min_dim: int):
    import torchvision.models as tvm
    ctor = getattr(tvm, variant)
    try:
        model = ctor(weights="IMAGENET1K_V1")
    except TypeError:  # very old torchvision
        model = ctor(pretrained=True)
    model.eval()
    return _matrices_from_torch_module(model, min_dim)


def load_gpt(variant: str, min_dim: int):
    from transformers import AutoModel
    name = "openai-community/openai-gpt" if variant == "openai-gpt" else f"openai-community/{variant}"
    try:
        model = AutoModel.from_pretrained(name)
    except Exception:
        model = AutoModel.from_pretrained(variant)  # bare-name fallback
    model.eval()
    return _matrices_from_torch_module(model, min_dim)


# ---------------------------------------------------------------------------
# per-matrix frequentist + Bayesian fit on shared eigenvalues
# ---------------------------------------------------------------------------
def analyse_matrix(name: str, W: np.ndarray) -> dict:
    Wj = jnp.asarray(W)
    eigs = _eigvals(Wj)
    freq = wwj.analyze_matrix(Wj, name=name, mode="csn")   # WeightWatcher point estimate
    bma = wwj.alpha_posterior_bma(eigs)                    # posterior over alpha, marginalising xmin
    mp = wwj.model_posterior(eigs)                         # is it actually a power law?
    # WeightWatcher's weighted-alpha weights by log10 of the spectral norm SQUARED
    # (the un-normalised top eigenvalue of W^T W = ||W||_2^2). wwj._eigvals divides
    # by N, so undo that here to match Charles's alpha_hat convention.
    spectral_norm_sq = float(freq.lambda_max) * max(W.shape)
    return {
        "layer": name,
        "shape": f"{W.shape[0]}x{W.shape[1]}",
        "tail_size": int(mp["tail_size"]),
        # frequentist (WeightWatcher)
        "alpha_freq": float(freq.alpha),
        "lambda_max": float(freq.lambda_max),
        "log10_snorm2": float(np.log10(max(spectral_norm_sq, 1e-12))),
        # Bayesian alpha posterior (BMA over xmin)
        "alpha_bayes": bma["alpha_mean"],
        "ci_low": bma["ci_low"],
        "ci_high": bma["ci_high"],
        "ci_width": bma["ci_high"] - bma["ci_low"],
        "xmin_pick_var": bma.get("across_var", float("nan")),   # alpha-uncertainty from window choice
        "n_eff_windows": bma.get("n_eff", float("nan")),
        "p_alpha_lt_2": bma["p_alpha_lt_2"],
        # Bayesian model comparison (the "is it a power law?" test)
        "best_model": mp["best_model"],
        "prob_powerlaw": mp["prob_powerlaw"],
        "logbf_pl_vs_exp": mp["logbf_pl_vs_exp"],
        "logbf_pl_vs_ln": mp["logbf_pl_vs_ln"],
    }


def model_summary(name: str, rows: list[dict], extra: dict | None = None) -> dict:
    df = pd.DataFrame(rows)
    w = df["log10_snorm2"].to_numpy()   # WeightWatcher's log10(||W||_2^2) weighting
    s = {
        "model": name,
        "n_layers": len(df),
        # Charles's weighted-alpha alpha_hat = sum_l alpha_l * log10(||W_l||_2^2) (point)
        # vs its posterior-mean counterpart
        "alpha_hat_freq": float(np.sum(df["alpha_freq"].to_numpy() * w)),
        "alpha_hat_bayes": float(np.sum(df["alpha_bayes"].to_numpy() * w)),
        "mean_alpha_freq": float(df["alpha_freq"].mean()),
        "mean_alpha_bayes": float(df["alpha_bayes"].mean()),
        "median_alpha_bayes": float(df["alpha_bayes"].median()),
        "mean_ci_width": float(df["ci_width"].mean()),
        # the "large alpha = not a PL fit" story (GPT vs GPT2)
        "n_alpha_gt6": int((df["alpha_bayes"] > 6.0).sum()),
        # Bayesian verdict: fraction of layers the Bayes factor says are NOT power-law
        "frac_pl_best": float((df["best_model"] == "powerlaw").mean()),
        "frac_pl_rejected": float((df["best_model"] != "powerlaw").mean()),
        "mean_prob_powerlaw": float(df["prob_powerlaw"].mean()),
    }
    if extra:
        s.update(extra)
    return s


def run_family(family: str, variants: list[str], min_dim: int, out_dir: Path) -> list[dict]:
    loader = {"vgg": load_vgg, "gpt": load_gpt}[family]
    summaries = []
    for v in variants:
        print(f"\n=== {v} ===", flush=True)
        try:
            mats = loader(v, min_dim)
        except Exception as e:
            print(f"  [load failed] {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        print(f"  {len(mats)} weight matrices (min_dim>={min_dim})", flush=True)
        rows = [analyse_matrix(n, W) for n, W in mats]
        pd.DataFrame(rows).to_csv(out_dir / f"{v}_layers.csv", index=False)
        extra = {"published_top1": VGG_TOP1[v]} if family == "vgg" and v in VGG_TOP1 else None
        s = model_summary(v, rows, extra)
        summaries.append(s)
        print(f"  alpha_hat: freq={s['alpha_hat_freq']:.2f} bayes={s['alpha_hat_bayes']:.2f}  "
              f"mean_alpha: freq={s['mean_alpha_freq']:.2f} bayes={s['mean_alpha_bayes']:.2f}  "
              f"n(alpha>6)={s['n_alpha_gt6']}  PL-rejected={s['frac_pl_rejected']:.0%}  "
              f"mean_CI_width={s['mean_ci_width']:.2f}", flush=True)
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["vgg", "gpt"], choices=["vgg", "gpt"])
    ap.add_argument("--vgg-variants", nargs="+", default=list(VGG_TOP1))
    ap.add_argument("--gpt-variants", nargs="+", default=["openai-gpt", "gpt2", "gpt2-medium"])
    ap.add_argument("--min-dim", type=int, default=50)
    ap.add_argument("--out-dir", default="/data/mhough/wwj_ww_replication")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    if "vgg" in args.models:
        summaries += run_family("vgg", args.vgg_variants, args.min_dim, out_dir)
    if "gpt" in args.models:
        summaries += run_family("gpt", args.gpt_variants, args.min_dim, out_dir)

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(out_dir / "ww_replication_summary.csv", index=False)
    (out_dir / "ww_replication_summary.json").write_text(json.dumps(summaries, indent=2, default=float))

    print("\n" + "=" * 100 + "\nwwjd replication of WeightWatcher canonical cases\n" + "=" * 100)
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        cols = ["model", "n_layers", "alpha_hat_freq", "alpha_hat_bayes", "mean_alpha_freq",
                "mean_alpha_bayes", "mean_ci_width", "n_alpha_gt6", "frac_pl_rejected"]
        if "published_top1" in sdf.columns:
            cols.insert(2, "published_top1")
        print(sdf[[c for c in cols if c in sdf.columns]].to_string(index=False))

    # Headline diagnostics
    vgg = sdf[sdf["model"].isin(VGG_TOP1)] if "published_top1" in sdf.columns else sdf.iloc[0:0]
    if len(vgg) >= 4:
        rf = np.corrcoef(vgg["alpha_hat_freq"], vgg["published_top1"])[0, 1]
        rb = np.corrcoef(vgg["alpha_hat_bayes"], vgg["published_top1"])[0, 1]
        print(f"\n[VGG confirm] alpha_hat vs top1 accuracy: Pearson r  freq={rf:+.3f}  bayes={rb:+.3f}  "
              f"(Charles: near-perfect; negative = smaller alpha_hat -> higher acc)")
    gpts = sdf[sdf["model"].isin(GPT_VARIANTS)]
    if {"openai-gpt"}.issubset(set(gpts["model"])) and len(gpts) >= 2:
        g = gpts.set_index("model")
        base = g.loc["openai-gpt"]
        print(f"\n[GPT overturn] openai-gpt: mean_alpha={base['mean_alpha_bayes']:.2f} "
              f"n(alpha>6)={int(base['n_alpha_gt6'])} PL-rejected={base['frac_pl_rejected']:.0%}")
        for m in [x for x in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl") if x in g.index]:
            r = g.loc[m]
            print(f"             {m}: mean_alpha={r['mean_alpha_bayes']:.2f} "
                  f"n(alpha>6)={int(r['n_alpha_gt6'])} PL-rejected={r['frac_pl_rejected']:.0%}")
        print("  -> Does GPT's mean-alpha exceed GPT2's, and are GPT's large-alpha layers the ones the\n"
              "     Bayes factor flags as NOT power-law? That tests Charles's 'better-trained' claim.")
    print(f"\n[done] {out_dir/'ww_replication_summary.csv'}")


if __name__ == "__main__":
    main()
