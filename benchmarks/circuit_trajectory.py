"""Run the mechanistic-interpretability metrics (benchmarks/circuit_metrics.py) over the
from-scratch GPT-2-124M alpha-trajectory checkpoints, to test whether circuit formation is
locked to the wwjd heavy-tail formation we measured in alpha(step).

For each of the 17 log-spaced checkpoints (step 0 = random init -> step 8837 = one epoch):
  - per-head INDUCTION score (Olsson et al.)            -> circuit_heads.csv
  - per-head OV COPYING score (eigenvalue stat)          -> circuit_heads.csv
  - per-matrix STABLE RANK (heavy-tail proxy)            -> circuit_layers.csv
  - per-matrix top-subspace DRIFT vs previous checkpoint -> circuit_layers.csv ("freezing")
  - step-level aggregates                                -> circuit_summary.csv

Pairs with traj_summary.csv / traj_layers.csv (the alpha curves) for fig_circuit_trajectory.

Usage: uv run --extra benchmarks python benchmarks/circuit_trajectory.py [CKPT_DIR] [OUT_DIR]
       (defaults to the baseline trajectory; pass the areg_base / areg_a2 dirs for the
        causal-intervention arms, writing <tag>_circuit_*.csv into OUT_DIR)
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel
import circuit_metrics as cm

CKPTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/mhough/wwj_traj/ckpts/scratch_traj_ep1")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/data/mhough/wwj_traj")
PREFIX = sys.argv[3] + "_" if len(sys.argv) > 3 else ""   # e.g. "areg_a2_" to tag the CSVs
K_SUB = 8          # top-k singular subspace tracked for "freezing"
IND_THRESH = 0.3   # an "induction head" for the head-count summary


def main():
    steps = sorted([d for d in CKPTS.iterdir() if re.fullmatch(r"step\d+", d.name)],
                   key=lambda d: int(d.name[4:]))
    assert steps, f"no step* checkpoints under {CKPTS}"
    summ, heads, layers = [], [], []
    prev_sub = {}
    for d in steps:
        step = int(d.name[4:])
        model = GPT2LMHeadModel.from_pretrained(d, attn_implementation="eager").eval()
        base = model.transformer
        ind = cm.induction_scores(model, seq_len=64, batch=16)   # (nL, nH)
        cop = cm.ov_copying_scores(model)                        # (nL, nH)
        nL, nH = ind.shape
        for l in range(nL):
            for h in range(nH):
                heads.append({"step": step, "layer": l, "head": h,
                              "induction": float(ind[l, h]), "copying": float(cop[l, h])})

        W = cm.weight_matrices(base)                             # same matrices as wwjd
        drifts = []
        for nm, mat in W.items():
            sr = cm.stable_rank(mat)
            sub = cm.top_subspace(mat, K_SUB)
            drift = np.nan
            if nm in prev_sub:
                drift = cm.principal_angle_drift(prev_sub[nm], sub)
                drifts.append(drift)
            prev_sub[nm] = sub
            layers.append({"step": step, "matrix": nm, "stable_rank": sr,
                           "subspace_drift": drift})

        amax = np.unravel_index(np.argmax(ind), ind.shape)
        summ.append({"step": step,
                     "max_induction": float(ind.max()),
                     "argmax_layer": int(amax[0]), "argmax_head": int(amax[1]),
                     "n_induction_heads": int((ind > IND_THRESH).sum()),
                     "mean_copying": float(np.nanmean(cop)),
                     "max_copying": float(np.nanmax(cop)),
                     "mean_stable_rank": float(np.mean([cm.stable_rank(m) for m in W.values()])),
                     "mean_subspace_drift": float(np.mean(drifts)) if drifts else np.nan})
        print(f"[circ] step {step:>6}: max_ind={summ[-1]['max_induction']:.3f} "
              f"(L{amax[0]}H{amax[1]}) n_ind={summ[-1]['n_induction_heads']:>2} "
              f"mean_copy={summ[-1]['mean_copying']:.3f} "
              f"mean_sr={summ[-1]['mean_stable_rank']:.1f} "
              f"drift={summ[-1]['mean_subspace_drift']:.3f}", flush=True)
        del model

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summ).to_csv(OUT / f"{PREFIX}circuit_summary.csv", index=False)
    pd.DataFrame(heads).to_csv(OUT / f"{PREFIX}circuit_heads.csv", index=False)
    pd.DataFrame(layers).to_csv(OUT / f"{PREFIX}circuit_layers.csv", index=False)
    print(f"\n[circ] wrote {OUT}/{PREFIX}circuit_summary.csv (+heads,+layers)")


if __name__ == "__main__":
    main()
