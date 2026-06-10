"""Three-regime training-maturity spectral comparison for the paper: base GPT-2 vs a
1-epoch neuroscience fine-tune vs a 1-epoch from-scratch GPT-2-Neuro, all GPT-2-124M.
Tests whether the HT-SR alpha tracks training MATURITY (it does) vs data DOMAIN (it
does not). Writes per-layer + summary CSVs to /data/mhough/wwj_braingpt for the paper's
generate.py / make_paper_figures.

Models:
  base     openai-community/gpt2 (web-pretrained, mature)
  finetune /data/mhough/gpt2_neuro_ft/final     (base + 1 epoch neuroscience, lr 2e-5)
  scratch  /data/mhough/gpt2_neuro_scratch/final (random init + 1 epoch neuroscience)

Usage: uv run --extra benchmarks --extra bayes python benchmarks/braingpt_spectral.py
"""
from pathlib import Path
import numpy as np, pandas as pd, jax.numpy as jnp
import wwj
from wwj.core import _eigvals
from transformers import AutoModel

OUT = Path("/data/mhough/wwj_braingpt"); OUT.mkdir(parents=True, exist_ok=True)
MODELS = {
    "base": "openai-community/gpt2",
    "finetune": "/data/mhough/gpt2_neuro_ft/final",
    "scratch": "/data/mhough/gpt2_neuro_scratch/final",
}


def mats(path, min_dim=50):
    m = AutoModel.from_pretrained(path).eval()
    out = {}
    for nm, p in m.named_parameters():
        if nm.endswith("weight") and p.ndim >= 2:
            W = p.detach().float().numpy().reshape(p.shape[0], -1)
            if min(W.shape) >= min_dim:
                out[nm] = W.astype(np.float32)
    return out


def per_layer(W):
    e = _eigvals(jnp.asarray(W))
    af = float(wwj.analyze_matrix(jnp.asarray(W), mode="csn").alpha)
    ab = wwj.alpha_posterior_bma(e)["alpha_mean"]
    plr = wwj.model_posterior(e)["best_model"] != "powerlaw"
    return af, ab, plr


def main():
    mm = {k: mats(v) for k, v in MODELS.items()}
    common = sorted(set(mm["base"]) & set(mm["finetune"]) & set(mm["scratch"]),
                    key=lambda k: list(mm["base"]).index(k))
    common = [k for k in common if mm["base"][k].shape == mm["scratch"][k].shape]
    rows, summ = [], []
    cache = {m: {k: per_layer(mm[m][k]) for k in common} for m in MODELS}
    for k in common:
        r = {"layer": k}
        for m in MODELS:
            af, ab, plr = cache[m][k]
            r[f"{m}_freq"] = af; r[f"{m}_bayes"] = ab; r[f"{m}_plrej"] = plr
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "braingpt_layers.csv", index=False)
    for m in MODELS:
        ab = df[f"{m}_bayes"]; af = df[f"{m}_freq"]
        summ.append({"model": m, "n_layers": len(df),
                     "mean_alpha_freq": float(af.mean()),
                     "mean_alpha_bayes": float(ab.mean()),
                     "mean_absdist2_bayes": float((ab - 2).abs().mean()),
                     "frac_pl_rejected": float(df[f"{m}_plrej"].mean())})
    sdf = pd.DataFrame(summ); sdf.to_csv(OUT / "braingpt_summary.csv", index=False)
    print(sdf.to_string(index=False))
    print(f"\n[done] {OUT}/braingpt_summary.csv")


if __name__ == "__main__":
    main()
