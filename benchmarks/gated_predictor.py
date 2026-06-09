"""Gated weights-only quality predictor — prototype.

Open thread from the CortexMAE RG study (docs WEIGHTWATCHER_RG_ANALYSIS.md): the
single weights-only diagnostic that best predicts downstream is *axis-dependent* —
alpha-mean on the data-scaling axis, phi_1/M_tr on the model-scaling axis, %alpha<2 on
the input-space axis — and a single fixed-coefficient combo (alpha+phi_1+M_tr) does NOT
beat the best single diagnostic when the axes are pooled (it often hurts). That negative
result motivates a *gated* predictor that routes each diagnostic to the regime where it
is informative, instead of one global formula.

The gate is structural and observable from the weights alone:

  * CAPACITY signal (M_tr ~ width; phi_1 ~ 1/width) varies ACROSS architecture clusters
    (different embed_dim/depth) and is what tracks downstream on the model-scaling axis.
  * TRAINING-QUALITY signal (alpha-mean, %alpha<2) varies WITHIN a fixed architecture and
    is what tracks downstream on the data / input / objective axes (where width is fixed,
    so the capacity signal is constant and carries no information).

So we use a hierarchical (mixture-of-experts) predictor whose gate is the architecture
cluster, derived from each model's own spectrum (n_layers -> depth bucket; M_tr -> width):

    pred(i) = capacity_term(M_tr_i)                         # across-cluster, global OLS
            + within_cluster_residual(alpha_i, %a<2_i)      # within-cluster OLS, big clusters only

The capacity term ranks the architecture clusters; the within-cluster term ranks models
that share an architecture (where capacity is constant and only training quality moves).
Small clusters (e.g. the depth-ablation points, 2 models each) get capacity-only, which is
exactly correct — on the model-scaling axis capacity *is* the signal.

Everything is scored by leave-one-out CV Spearman (per-fold standardization + cluster
re-estimation), so the comparison against the single-diagnostic and naive-combo baselines
is honest at this n. Reads the joined RG+downstream table written by
CortexMAE/scripts/build_rg_benchmark_table.py.

    cd /home/mhough/dev/wwj && uv run python benchmarks/gated_predictor.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TABLE = "/data/derivatives/cortexmae_ww/results/rg_benchmark_table.csv"
TARGETS = ["downstream_meanrank", "hcpya_task21", "nsd_cococlip"]
CAPACITY = "M_tr_median"                      # ~ embed_dim/2; varies across architectures
TRAINING = ["alpha_mean", "frac_alpha_lt2"]   # vary within a fixed architecture
MIN_CLUSTER_FOR_TRAINING = 6                   # need enough within-cluster points for OLS


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def spearman(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    rx = pd.Series(x[m]).rank().to_numpy() - 0.0
    ry = pd.Series(y[m]).rank().to_numpy() - 0.0
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d else np.nan


def _ols_fit(X, y):
    """standardize columns on the training rows, return (mu, sd, beta) for an OLS with
    intercept. X: (n, p)."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = (X - mu) / sd
    A = np.column_stack([np.ones(len(Xs)), Xs])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return mu, sd, beta


def _ols_pred(x_row, mu, sd, beta):
    xs = (x_row - mu) / sd
    return float(beta[0] + xs @ beta[1:])


def loo_ols(df, feats, y) -> np.ndarray:
    """Leave-one-out OLS predictions over the whole df for the given feature columns."""
    X = df[feats].to_numpy(float)
    n = len(df)
    pred = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        keep = tr & np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        if keep.sum() < len(feats) + 2:
            continue
        mu, sd, beta = _ols_fit(X[keep], y[keep])
        if np.all(np.isfinite(X[i])):
            pred[i] = _ols_pred(X[i], mu, sd, beta)
    return pred


def loo_gated(df, y, clusters) -> np.ndarray:
    """The gated/hierarchical predictor, leave-one-out.

    base    = global LOO-OLS on CAPACITY (M_tr) -> ranks architecture clusters
    within  = within-(held-out's cluster) LOO-OLS of (y - base) on TRAINING feats,
              fit on that cluster's *training-fold* members; only when the cluster has
              >= MIN_CLUSTER_FOR_TRAINING members (else 0 -> capacity-only).
    pred = base + within
    """
    Xc = df[[CAPACITY]].to_numpy(float)
    Xt = df[TRAINING].to_numpy(float)
    n = len(df)
    pred = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        keepc = tr & np.isfinite(y) & np.all(np.isfinite(Xc), axis=1)
        if keepc.sum() < 3:
            continue
        mu, sd, beta = _ols_fit(Xc[keepc], y[keepc])
        base_all = np.array([_ols_pred(Xc[j], mu, sd, beta) for j in range(n)])
        base_i = base_all[i]

        # within-cluster residual expert for the held-out model's cluster
        ci = clusters[i]
        mates = tr & (clusters == ci) & np.isfinite(y) & np.all(np.isfinite(Xt), axis=1)
        within_i = 0.0
        if mates.sum() >= MIN_CLUSTER_FOR_TRAINING:
            resid = y[mates] - base_all[mates]
            mu_t, sd_t, beta_t = _ols_fit(Xt[mates], resid)
            if np.all(np.isfinite(Xt[i])):
                within_i = _ols_pred(Xt[i], mu_t, sd_t, beta_t)
        pred[i] = base_i + within_i
    return pred


def _seeded(n):
    return bool(re.fullmatch(r"cortex_mae_(flat|volume|parcel)(_r\d+)?", n))


def axis_members(models):
    return {
        "data_scaling": [n for n in models if "_n" in n and n.split("_n")[-1].split("_")[0].isdigit()],
        "model_scaling": [n for n in models if "_d" in n and n.split("_d")[-1].split("_")[0].isdigit()],
        "parcellation": [n for n in models if n.startswith("cortex_mae_parcel")],
        "input_space": [n for n in models if _seeded(n)],
    }


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--out", default="/data/derivatives/cortexmae_ww/results/gated_predictor.md")
    args = ap.parse_args()

    df = pd.read_csv(args.table).reset_index(drop=True)
    # architecture cluster = (n_layers) bucket; n_layers tracks the joint depth+width
    # nanochat scaling, so it separates d3/d6/d9/d12+base/d15 and lumps every ViT-B
    # (data/input/tpatch/rest/parcellation) variant into one big cluster.
    clusters = df["n_layers"].to_numpy()
    nclust = df.groupby("n_layers").size().to_dict()
    print(f">>> {len(df)} models; architecture clusters (n_layers -> count): {nclust}")
    print(f">>> training-expert active on clusters with >= {MIN_CLUSTER_FOR_TRAINING} members\n")

    axes = axis_members(list(df.model))
    rows = []
    for target in TARGETS:
        y = df[target].to_numpy(float)
        ysign = -y if target == "downstream_meanrank" else y  # higher = better everywhere

        preds = {
            "single_alpha": loo_ols(df, ["alpha_mean"], ysign),
            "single_Mtr": loo_ols(df, [CAPACITY], ysign),
            "single_fa2": loo_ols(df, ["frac_alpha_lt2"], ysign),
            "naive_combo": loo_ols(df, ["alpha_mean", "phi_1_median", CAPACITY], ysign),
            "GATED": loo_gated(df, ysign, clusters),
        }
        rec = {"target": target}
        for name, p in preds.items():
            rec[name] = spearman(p, ysign)
        # per-axis Spearman of the GATED predictor (does it stay strong everywhere?)
        gp = preds["GATED"]
        for ax, names in axes.items():
            idx = df.model.isin(names).to_numpy()
            rec[f"GATED@{ax}"] = spearman(gp[idx], ysign[idx])
        rows.append(rec)
    res = pd.DataFrame(rows)

    def fmt(v):
        return "n/a" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else f"{v:.3f}"

    base_cols = ["target", "single_alpha", "single_Mtr", "single_fa2", "naive_combo", "GATED"]
    axis_cols = ["target"] + [f"GATED@{a}" for a in axes]

    L = ["## CortexMAE: gated weights-only quality predictor (leave-one-out Spearman)", "",
         "Pooled over all models. `single_*` and `naive_combo` are global LOO-OLS baselines;",
         "`GATED` routes capacity (M_tr, across architecture clusters) vs training-quality",
         "(alpha, %a<2, within cluster). +rho = predicts better downstream.", "",
         "| " + " | ".join(c.replace("single_", "") for c in base_cols) + " |",
         "|" + "|".join(["---"] * len(base_cols)) + "|"]
    for _, r in res.iterrows():
        L.append("| " + " | ".join(fmt(r[c]) if c != "target" else r[c] for c in base_cols) + " |")
    L += ["", "### Per-axis Spearman of the GATED predictor (stays strong on every axis?)", "",
          "| " + " | ".join(c.replace("GATED@", "") for c in axis_cols) + " |",
          "|" + "|".join(["---"] * len(axis_cols)) + "|"]
    for _, r in res.iterrows():
        L.append("| " + " | ".join(fmt(r[c]) if c != "target" else r[c] for c in axis_cols) + " |")
    Path(args.out).write_text("\n".join(L) + "\n")

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
    print("=== pooled LOO Spearman: gated vs baselines ===")
    print(res[base_cols].to_string(index=False))
    print("\n=== per-axis LOO Spearman of the GATED predictor ===")
    print(res[axis_cols].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
