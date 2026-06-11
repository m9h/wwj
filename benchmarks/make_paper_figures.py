"""Generate the WeightWatcher-replication figures for the paper (seaborn) from the
per-layer CSVs in /data/mhough/wwj_ww_replication. Writes PNGs into paper/figures/.

Also importable: the figure functions take an output dir, so the PythonTeX literate
build (paper/wwjd_paper_lit.tex) calls them at document-compile time -- the Sweave
analogue, figures generated from data as the paper builds.

Usage: uv run --extra benchmarks python benchmarks/make_paper_figures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA = Path("/data/mhough/wwj_ww_replication")
VGG = {"vgg11":69.020,"vgg13":69.928,"vgg16":71.592,"vgg19":72.376,
       "vgg11_bn":70.370,"vgg13_bn":71.586,"vgg16_bn":73.360,"vgg19_bn":74.218}
GPT = ["openai-gpt", "gpt2", "gpt2-medium"]
sns.set_theme(context="paper", style="ticks", font_scale=1.0)


def _vgg_alpha_hat():
    rows = []
    for v, acc in VGG.items():
        df = pd.read_csv(DATA / f"{v}_layers.csv"); w = df["log10_snorm2"].to_numpy()
        a = df["alpha_bayes"].to_numpy(); sd = (df["ci_high"]-df["ci_low"]).to_numpy()/(2*1.959964)
        ok = np.isfinite(a) & np.isfinite(sd); N = ok.sum()
        rows.append({"model": v, "acc": acc, "ahat": (a*w)[ok].mean(),
                     "se": np.sqrt(np.sum(((np.abs(w)*sd)[ok])**2))/N,
                     "ahat_f": np.nanmean(df["alpha_freq"].to_numpy()*w),
                     "bn": "bn" in v})
    return pd.DataFrame(rows)


def fig_gpt_dissolution(out: Path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    per = pd.concat([pd.read_csv(DATA / f"{t}_layers.csv").assign(model=t) for t in ["openai-gpt", "gpt2"]], ignore_index=True)
    sns.scatterplot(per, x="alpha_freq", y="alpha_bayes", hue="model", s=22, alpha=0.75,
                    palette={"openai-gpt": "#d62728", "gpt2": "#1f77b4"}, ax=axL, edgecolor="none")
    axL.plot([1.5, 7.2], [1.5, 7.2], "k--", lw=0.8)
    axL.axhline(2.0, color="green", lw=0.8, ls=":")
    axL.set(xlim=(1.5, 7.2), ylim=(1.8, 4.2),
            xlabel=r"frequentist $\alpha$ (single KS window)", ylabel=r"Bayesian $\alpha$ (BMA over $x_{\min}$)",
            title="Per-layer: large-$\\alpha$ layers collapse")
    axL.legend(fontsize=8, title=None)
    s = pd.read_csv(DATA / "ww_replication_summary.csv").set_index("model")
    bars = pd.DataFrame({"model": GPT*2,
                         "alpha": [s.loc[m, "mean_alpha_freq"] for m in GPT] + [s.loc[m, "mean_alpha_bayes"] for m in GPT],
                         "estimator": ["frequentist"]*3 + ["Bayesian"]*3})
    sns.barplot(bars, x="model", y="alpha", hue="estimator",
                palette={"frequentist": "#bbbbbb", "Bayesian": "#1f77b4"}, ax=axR)
    axR.axhline(2.0, color="green", lw=0.8, ls=":")
    axR.set(xlabel=None, ylabel=r"mean $\alpha$", title='"GPT poorly trained" gap shrinks $3\\times$')
    axR.tick_params(axis="x", labelrotation=12); axR.legend(fontsize=8, title=None)
    sns.despine(fig); fig.tight_layout(); fig.savefig(out / "fig_gpt_dissolution.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_vgg_eiv(out: Path):
    d = _vgg_alpha_hat()
    rb = np.corrcoef(d["ahat"], d["acc"])[0, 1]; rf = np.corrcoef(d["ahat_f"], d["acc"])[0, 1]
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    sns.regplot(d, x="ahat", y="acc", ax=ax, scatter=False, color="#1f77b4",
                line_kws={"lw": 1.5}, ci=95)   # regression line + 95% band (Sweave-friendly)
    for bn, mk, c in [(False, "o", "#1f77b4"), (True, "s", "#d62728")]:
        sub = d[d["bn"] == bn]
        ax.errorbar(sub["ahat"], sub["acc"], xerr=sub["se"], fmt=mk, color=c, ms=6, capsize=3,
                    label="VGG-BN" if bn else "VGG", ls="none")
    for _, r in d.iterrows():
        ax.annotate(r["model"].replace("vgg", "").replace("_bn", "bn"), (r["ahat"], r["acc"]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel=r"Bayesian weighted-$\hat\alpha$", ylabel="ImageNet top-1 (%)",
           title=f"Calibrated predictor (EIV slope $-6.2$ pp/unit)\nPearson $r$: Bayes ${rb:.2f}$ vs freq ${rf:.2f}$")
    ax.legend(fontsize=8); sns.despine(fig); fig.tight_layout()
    fig.savefig(out / "fig_vgg_eiv.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_mechanism(out: Path):
    per = pd.concat([pd.read_csv(DATA / f"{t}_layers.csv").assign(model=t)
                     for t in ["openai-gpt", "vgg19_bn", "vgg16"]], ignore_index=True)
    per["correction"] = per["alpha_freq"] - per["alpha_bayes"]
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    sns.scatterplot(per, x="alpha_freq", y="correction", hue="model", style="model", s=26, alpha=0.75,
                    palette={"openai-gpt": "#d62728", "vgg19_bn": "#1f77b4", "vgg16": "#2ca02c"}, ax=ax, edgecolor="none")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel=r"frequentist $\alpha$", ylabel=r"posterior pull-down  $\alpha_{\rm freq}-\alpha_{\rm bayes}$",
           title="Mechanism: larger frequentist $\\alpha$, larger correction")
    ax.legend(fontsize=8, title=None); sns.despine(fig); fig.tight_layout()
    fig.savefig(out / "fig_mechanism.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_training_maturity(out: Path):
    """Three-regime: base GPT-2 vs 1-epoch neuroscience fine-tune vs 1-epoch from-scratch.
    alpha tracks training maturity (scratch far from 2) not data domain (fine-tune inert)."""
    bg = Path("/data/mhough/wwj_braingpt/braingpt_summary.csv")
    if not bg.exists():
        return
    s = pd.read_csv(bg).set_index("model")
    order = ["base", "finetune", "scratch"]
    labels = ["base GPT-2\n(mature)", "+ neuro\nfine-tune", "neuro\nfrom-scratch"]
    bars = pd.DataFrame({"x": labels * 2,
                         "alpha": [s.loc[m, "mean_alpha_freq"] for m in order] +
                                  [s.loc[m, "mean_alpha_bayes"] for m in order],
                         "estimator": ["frequentist"] * 3 + ["Bayesian"] * 3})
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    sns.barplot(bars, x="x", y="alpha", hue="estimator",
                palette={"frequentist": "#bbbbbb", "Bayesian": "#1f77b4"}, ax=ax)
    ax.axhline(2.0, color="green", lw=0.9, ls=":")
    ax.text(2.4, 2.05, r"RG optimum $\alpha=2$", color="green", fontsize=8, va="bottom", ha="right")
    ax.set(xlabel=None, ylabel=r"mean $\alpha$",
           title=r"$\alpha$ tracks training maturity, not domain")
    ax.legend(fontsize=8, title=None, loc="upper left")
    sns.despine(fig); fig.tight_layout()
    fig.savefig(out / "fig_training_maturity.png", dpi=170, bbox_inches="tight"); plt.close(fig)


_LAYER_TYPE = [
    ("wte.weight", "token embed", "#9467bd"),
    ("wpe.weight", "pos embed", "#8c564b"),
    ("attn.c_attn", "attn QKV", "#1f77b4"),
    ("attn.c_proj", "attn out-proj", "#17becf"),
    ("mlp.c_fc", "MLP in", "#ff7f0e"),
    ("mlp.c_proj", "MLP out-proj", "#2ca02c"),
]


def _traj_type(name: str) -> str:
    for key, lab, _ in _LAYER_TYPE:
        if key in name:
            return lab
    return "other"


def fig_alpha_trajectory(out: Path):
    """alpha(step) as GPT-2-124M trains from scratch (lr 6e-4 cosine). Left: the Bayesian
    alpha of every layer type descends from the random-init (Marchenko-Pastur) light tail
    toward the RG optimum 2 -- token embeddings start lightest (alpha=6.1) and travel
    farthest, output projections form their heavy tail first. The thick line is the mean.
    Right: the frequentist single-KS-window inflation (freq-bayes pull-down) collapses from
    7.5 at random init to ~0.7 as the scaling window stabilises -- the paper's mechanism
    traced through training time."""
    summ = Path("/data/mhough/wwj_traj/traj_summary.csv")
    lay = Path("/data/mhough/wwj_traj/traj_layers.csv")
    if not (summ.exists() and lay.exists()):
        return
    s = pd.read_csv(summ).sort_values("step").reset_index(drop=True)
    L = pd.read_csv(lay)
    L["t"] = L["layer"].map(_traj_type)
    piv = L.pivot_table(index="step", columns="t", values="alpha_bayes", aggfunc="mean").sort_index()
    xs = lambda v: np.where(np.asarray(v) == 0, 0.7, v)   # log-axis: step 0 (init) -> 0.7
    ticks = [0.7, 10, 100, 1000, 8837]
    tlabs = ["init", "10", "100", "1k", "8.8k"]

    def _xfmt(ax):
        ax.set_xscale("log"); ax.set_xlim(0.55, 1.3e4)
        ax.set_xticks(ticks); ax.set_xticklabels(tlabs)
        ax.set_xlabel("training step (log; 0 = random init)")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for key, lab, c in _LAYER_TYPE:
        if lab in piv.columns:
            axA.plot(xs(piv.index), piv[lab], "-", color=c, lw=1.3, alpha=0.9, label=lab)
    axA.plot(xs(s["step"]), s["mean_alpha_bayes"], "k-", lw=2.4, label="mean", zorder=5)
    axA.axhline(2.0, color="green", lw=0.9, ls=":")
    axA.text(8837, 2.02, r"RG optimum $\alpha=2$", color="green",
             fontsize=7.5, va="bottom", ha="right")
    _xfmt(axA)
    axA.set(ylabel=r"Bayesian $\alpha$", title=r"Heavy tail forms layer-by-layer ($\alpha\!\to\!2$)")
    axA.legend(fontsize=7, ncol=2, title=None, loc="upper right")

    pull = s["mean_alpha_freq"] - s["mean_alpha_bayes"]
    axB.plot(xs(s["step"]), pull, "o-", color="#d62728", ms=4, lw=1.6)
    axB.axhline(0, color="k", lw=0.7)
    axB.annotate(f"init: {pull.iloc[0]:.1f}", (0.7, pull.iloc[0]),
                 fontsize=7.5, xytext=(6, -3), textcoords="offset points")
    _xfmt(axB)
    axB.set(ylabel=r"single-window inflation  $\alpha_{\rm freq}-\alpha_{\rm bayes}$",
            title="Frequentist inflation collapses as spectrum matures")
    sns.despine(fig); fig.tight_layout()
    fig.savefig(out / "fig_alpha_trajectory.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_circuit_trajectory(out: Path):
    """Mechanistic interpretability paired with the heavy-tail formation. Over the same
    from-scratch GPT-2 checkpoints: (A) the induction circuit ignites just as the Bayesian
    alpha crosses into the heavy-tailed band; (B) the weight stable rank collapses first
    (spectral spike forms) and then the top singular subspaces freeze (drift -> 0); (C) one
    head (L7H1) comes online with both signatures at once -- behavioural induction and its
    OV copying eigenvalues. Causal ordering: heavy tail forms, then the circuit lights up."""
    d = Path("/data/mhough/wwj_traj")
    need = [d / f for f in ("circuit_summary.csv", "circuit_heads.csv", "traj_summary.csv")]
    if not all(p.exists() for p in need):
        return
    s = pd.read_csv(d / "circuit_summary.csv").sort_values("step").reset_index(drop=True)
    H = pd.read_csv(d / "circuit_heads.csv")
    a = pd.read_csv(d / "traj_summary.csv").sort_values("step").reset_index(drop=True)
    xs = lambda v: np.where(np.asarray(v) == 0, 0.7, v)
    ticks, tlabs = [0.7, 10, 100, 1000, 8837], ["init", "10", "100", "1k", "8.8k"]

    def _xfmt(ax):
        ax.set_xscale("log"); ax.set_xlim(0.55, 1.3e4)
        ax.set_xticks(ticks); ax.set_xticklabels(tlabs)
        ax.set_xlabel("training step (log; 0 = random init)")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.8, 3.7))

    # A: alpha (left) + max induction (right) -- the temporal lock
    axA.plot(xs(a["step"]), a["mean_alpha_bayes"], "o-", color="#1f77b4", ms=3.5, lw=1.8)
    axA.axhline(2.0, color="green", lw=0.9, ls=":")
    axA.set_ylabel(r"Bayesian $\alpha$", color="#1f77b4"); axA.tick_params(axis="y", labelcolor="#1f77b4")
    axA.axvspan(300, 1000, color="orange", alpha=0.10, lw=0)
    aR = axA.twinx()
    aR.plot(xs(s["step"]), s["max_induction"], "s-", color="#d62728", ms=3.5, lw=1.8)
    aR.set_ylabel("max induction score", color="#d62728"); aR.tick_params(axis="y", labelcolor="#d62728")
    aR.set_ylim(-0.03, 0.85)
    _xfmt(axA); axA.set_title(r"Induction ignites as $\alpha\!\to\!$ heavy-tail band")

    # B: stable rank collapse (left) + subspace freezing (right)
    axB.plot(xs(s["step"]), s["mean_stable_rank"], "o-", color="#9467bd", ms=3.5, lw=1.8)
    axB.set_ylabel("mean stable rank", color="#9467bd"); axB.tick_params(axis="y", labelcolor="#9467bd")
    bR = axB.twinx()
    bR.plot(xs(s["step"]), s["mean_subspace_drift"], "^-", color="#ff7f0e", ms=3.5, lw=1.8)
    bR.set_ylabel("top-subspace drift (0 = frozen)", color="#ff7f0e"); bR.tick_params(axis="y", labelcolor="#ff7f0e")
    bR.set_ylim(-0.02, None)
    _xfmt(axB); axB.set_title("Spike forms first, then subspaces freeze")

    # C: one head, two signatures (induction + OV copying)
    h = H[(H["layer"] == 7) & (H["head"] == 1)].sort_values("step")
    axC.plot(xs(h["step"]), h["induction"], "s-", color="#d62728", ms=3.5, lw=1.8, label="induction (behavioural)")
    axC.plot(xs(h["step"]), h["copying"], "o-", color="#17becf", ms=3.5, lw=1.8, label="OV copying (eigenvalue)")
    axC.axhline(0.5, color="grey", lw=0.7, ls=":")
    axC.set_ylabel("score"); axC.set_ylim(0, 1.0)
    _xfmt(axC); axC.set_title("One circuit (L7H1), two signatures")
    axC.legend(fontsize=7.5, loc="upper left")

    sns.despine(fig, right=False); fig.tight_layout()
    fig.savefig(out / "fig_circuit_trajectory.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def make_all(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    fig_gpt_dissolution(out); fig_vgg_eiv(out); fig_mechanism(out)
    fig_training_maturity(out); fig_alpha_trajectory(out); fig_circuit_trajectory(out)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "paper" / "figures"
    make_all(out)
    print(f"wrote figures to {out}:")
    for f in sorted(out.glob("fig_*.png")):
        print(f"  {f.name}  {f.stat().st_size//1024} KB")
