"""Positive control: does the wwjd validity gate actually REJECT random matrices?

WHY THIS IS LOAD-BEARING. Several of our published-facing weight-side claims rest
entirely on the Bayesian model comparison saying "this tail is not a power law" --
most directly the BatchNorm finding (8-17% of BN-VGG layers Bayes-factor-rejected,
plain VGG and GPT at 0%; docs/ww_replication_targets.md). A rejection test is only
worth citing if it rejects things that ought to be rejected. Nothing has yet shown
that ours does.

The trap is documented by the tool we are improving on: WeightWatcher's own docs
concede that at aspect ratio Q = m/n ~ 1 the ESD of a *random* matrix can look
heavy-tailed, and Clauset/Shalizi/Newman (SIAM Rev 2009) show power-law vs
log-normal is undecidable without very large samples. So "random matrices look
power-law" is the specific failure mode to hunt, and Q=1 is where to hunt it.

DESIGN. Two arms, because they fail differently:

  synthetic -- iid Gaussian W at controlled aspect ratios Q in {1,2,4,8,16}. The ESD
      of W^T W is Marchenko-Pastur: bounded support, hard edge, provably NOT a power
      law. Any gate pass here is a false positive with no ambiguity about ground truth.

  arch -- real torchvision architectures instantiated with weights=None. Real init
      schemes (Kaiming/Xavier), real layer shapes, real depth -- but no training. This
      is the control that matters for interpreting the trained results, because it is
      matched to them on everything except training. Paired by layer name with the
      trained model, so the comparison is same-shape/same-position.

PRE-REGISTERED PREDICTIONS (written before running; see RESULTS at the bottom of
docs/ww_randinit_gate.md for the outcome):

  P1. Untrained matrices are rejected at a HIGH rate -- most do not pass the gate.
  P2. Rejection is HARDEST at Q=1 (the documented trap) and easiest at large Q.
  P3. Trained VGG (already measured: 0% rejection for plain VGG) rejects far less
      than its own random-init counterpart. If random-init also comes out ~0%, the
      gate is not discriminating and the BatchNorm finding is an artifact -- we
      report that.
  P4. Where random matrices ARE rejected, the PPC leg does more of the work than the
      Bayes-factor legs, because MP is neither power-law nor exponential nor
      log-normal: a 3-way comparison among wrong models can still return "power law"
      as least-bad. That is precisely why the gate has a goodness-of-fit leg, and
      this experiment is the demonstration.

HONESTY CONSTRAINT. model_posterior's docstring warns that Bayes factor MAGNITUDES
are prior-sensitive (Lindley-Bartlett) while the SIGN is reliable. Every statistic
here is a sign or a pass/fail rate. No logBF magnitudes are interpreted.

Usage:
  uv run --extra benchmarks python benchmarks/ww_randinit_gate.py \
      --out-dir /data/mhough/wwj_randinit_gate
  # cheap smoke test:
  uv run --extra benchmarks python benchmarks/ww_randinit_gate.py --synthetic-only --reps 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

import wwj
from wwj.core import _eigvals


# Aspect ratios to sweep. Q=1 is the documented failure mode (WeightWatcher's own
# docs: a random matrix ESD "looks" heavy-tailed there); large Q is the easy case.
Q_SWEEP = (1, 2, 4, 8, 16)
# Fixed number of columns; rows = Q * N_COLS. Chosen so the smallest tail is still
# fittable and the largest matrix stays cheap on a shared box.
N_COLS = 256

# Architectures to instantiate untrained. Paired against the trained runs in
# ww_replication.py, which used these same torchvision models.
ARCHES = ("vgg11", "vgg16", "vgg16_bn", "vgg19_bn", "densenet121")


def gate(mp: dict, ppc: dict) -> tuple[bool, str]:
    """The wwjd validity gate, and WHICH leg rejected.

    Identical in form to the gate used on the data side (benchmarks/zeta_law/
    hbn_bayesian_alpha.py) so weight-side and data-side rejections mean the same
    thing. Signs only -- never BF magnitudes.
    """
    ok_ln = mp["logbf_pl_vs_ln"] > 0        # power law beats log-normal
    ok_exp = mp["logbf_pl_vs_exp"] > 0      # power law beats exponential
    ok_ppc = ppc["p_value"] >= 0.05         # and actually fits
    passed = bool(ok_ln and ok_exp and ok_ppc)
    if passed:
        return True, "pass"
    failed = [n for n, ok in (("ln", ok_ln), ("exp", ok_exp), ("ppc", ok_ppc)) if not ok]
    return False, "+".join(failed)


def analyse(name: str, W: np.ndarray, **meta) -> dict:
    eigs = _eigvals(jnp.asarray(W))
    mp = wwj.model_posterior(eigs)
    ppc = wwj.ppc_pvalue(eigs)
    bma = wwj.alpha_posterior_bma(eigs)
    passed, why = gate(mp, ppc)
    return {
        "name": name,
        "shape": f"{W.shape[0]}x{W.shape[1]}",
        "Q": round(max(W.shape) / min(W.shape), 3),
        "tail_size": int(mp["tail_size"]),
        "alpha_bayes": bma["alpha_mean"],
        "ci_low": bma["ci_low"],
        "ci_high": bma["ci_high"],
        "best_model": mp["best_model"],
        "prob_powerlaw": mp["prob_powerlaw"],
        "logbf_pl_vs_ln": mp["logbf_pl_vs_ln"],
        "logbf_pl_vs_exp": mp["logbf_pl_vs_exp"],
        "ppc_p": ppc["p_value"],
        "gate_pass": passed,
        "reject_leg": why,
        **meta,
    }


def synthetic_arm(reps: int, seed: int) -> list[dict]:
    """iid Gaussian at controlled Q. Ground truth: Marchenko-Pastur, NOT a power law.
    Any gate pass is an unambiguous false positive."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in Q_SWEEP:
        for r in range(reps):
            W = rng.standard_normal((q * N_COLS, N_COLS)).astype(np.float32)
            rows.append(analyse(f"gauss_Q{q}_r{r}", W, arm="synthetic", arch=f"gauss_Q{q}", rep=r))
            print(f"  gauss Q={q:2d} rep {r}: gate={rows[-1]['gate_pass']} "
                  f"({rows[-1]['reject_leg']}) best={rows[-1]['best_model']}", flush=True)
    return rows


def arch_arm(arches: tuple[str, ...], seed: int) -> list[dict]:
    """Real architectures, real init, NO training. Matched to the trained runs on
    shape and position -- the only difference is that training never happened."""
    import torch
    import torchvision.models as tvm

    torch.manual_seed(seed)
    rows = []
    for arch in arches:
        model = getattr(tvm, arch)(weights=None)          # untrained
        for lname, p in model.named_parameters():
            if p.ndim < 2:
                continue
            W = p.detach().numpy()
            W = W.reshape(W.shape[0], -1)                 # conv -> (out, in*kh*kw)
            if min(W.shape) < 32:                         # too small to fit a tail
                continue
            rows.append(analyse(f"{arch}.{lname}", W, arm="arch", arch=arch, rep=0))
        n = sum(1 for r in rows if r["arch"] == arch)
        rej = sum(1 for r in rows if r["arch"] == arch and not r["gate_pass"])
        print(f"  {arch:12s}: {rej}/{n} rejected ({100*rej/max(n,1):.1f}%)", flush=True)
        del model
    return rows


def report(df: pd.DataFrame) -> dict:
    """Rejection rates. The headline number is the synthetic false-positive rate:
    matrices with KNOWN non-power-law ground truth that the gate let through."""
    out = {}
    print("\n== SYNTHETIC (ground truth: NOT a power law; gate SHOULD reject all) ==")
    syn = df[df.arm == "synthetic"]
    if len(syn):
        for q, g in syn.groupby("Q"):
            rej = (~g.gate_pass).mean()
            legs = g[~g.gate_pass].reject_leg.value_counts().to_dict()
            print(f"  Q={q:<5g} rejected {100*rej:5.1f}%  ({len(g)} mats)  legs={legs}")
            out[f"synthetic_reject_Q{q:g}"] = float(rej)
        fp = float((syn.gate_pass).mean())
        out["synthetic_false_positive_rate"] = fp
        print(f"  --> FALSE POSITIVE RATE (gate passed a random matrix): {100*fp:.1f}%")

    print("\n== UNTRAINED ARCHITECTURES (real init, real shapes, no training) ==")
    arc = df[df.arm == "arch"]
    if len(arc):
        for a, g in arc.groupby("arch"):
            rej = (~g.gate_pass).mean()
            print(f"  {a:12s} rejected {100*rej:5.1f}%  ({len(g)} layers)")
            out[f"arch_reject_{a}"] = float(rej)

    print("\n== WHICH LEG DOES THE REJECTING (P4) ==")
    bad = df[~df.gate_pass]
    if len(bad):
        for leg, c in bad.reject_leg.value_counts().items():
            print(f"  {leg:16s} {c:4d}  ({100*c/len(bad):.1f}% of rejections)")
        out["reject_leg_counts"] = {str(k): int(v) for k, v in
                                    bad.reject_leg.value_counts().items()}
    print("\n== VERDICT ==")
    if len(syn):
        fp = out["synthetic_false_positive_rate"]
        if fp <= 0.05:
            print(f"  GATE IS DISCRIMINATING: passes only {100*fp:.1f}% of known-random "
                  f"matrices. Rejection-based claims (e.g. the BatchNorm finding) stand.")
        elif fp >= 0.5:
            print(f"  GATE IS BROKEN: passes {100*fp:.1f}% of known-random matrices. "
                  f"Rejection-based claims must be withdrawn or re-derived.")
        else:
            print(f"  GATE IS PARTIAL: passes {100*fp:.1f}% of known-random matrices. "
                  f"Usable as evidence but must be reported WITH this false-positive rate.")
        out["verdict_fp_rate"] = fp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/data/mhough/wwj_randinit_gate"))
    ap.add_argument("--reps", type=int, default=5, help="random matrices per aspect ratio")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--arches", nargs="+", default=list(ARCHES))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("== synthetic arm (iid Gaussian, controlled aspect ratio) ==", flush=True)
    rows = synthetic_arm(args.reps, args.seed)
    if not args.synthetic_only:
        print("\n== untrained-architecture arm ==", flush=True)
        rows += arch_arm(tuple(args.arches), args.seed)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "randinit_gate.csv", index=False)
    summary = report(df)
    (args.out_dir / "randinit_gate_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out_dir}/randinit_gate.csv  (+ summary json)")


if __name__ == "__main__":
    main()
