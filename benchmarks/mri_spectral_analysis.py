"""Spectral analysis pass over MRI foundation-model checkpoints + a demo of
the TorchIO-based perturbation API.

Targets:
  - BrainIAC: /home/mhough/dev/BrainIAC/src/checkpoints/{BrainIAC,brainage}.ckpt
    (PyTorch Lightning format)
  - realtime-mindeye: /data/derivatives/rtmindeye_paper/rt3t/data/model/*.pth
    (plain torch state_dicts)

Outputs per ckpt:
  - per-layer alpha (with bootstrap 95% CI)
  - per-layer power-law-vs-exp / vs-lognormal LRT (validates the HTSR claim)
  - aggregate spectral health vs RG optimum (mean |alpha - 2|, near2 count)
  - distribution-validity rate (fraction of layers where PL is preferred over
    both alternatives)

The full perturbation-vs-robustness sweep (forward pass through encoder, drift
under TorchIO transforms, correlation with per-layer alpha) requires the
project-specific encoder builder and a held-out volume; documented as future
work below.

Usage:
    uv run python benchmarks/mri_spectral_analysis.py --ckpt /path/to/x.ckpt
    uv run python benchmarks/mri_spectral_analysis.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


CKPT_CANDIDATES = {
    "brainiac": "/home/mhough/dev/BrainIAC/src/checkpoints/BrainIAC.ckpt",
    "brainiac_brainage": "/home/mhough/dev/BrainIAC/src/checkpoints/brainage.ckpt",
    "rtmindeye_multisubject": "/data/derivatives/rtmindeye_paper/rt3t/data/model/multisubject_sdxlturbo_excludingsubj01_40sess.pth",
    "rtmindeye_sub005_finetune": "/data/derivatives/rtmindeye_paper/rt3t/data/model/sub-005_ses-01-03_task-C_bs24_MST_rishab_MSTsplit_unionmask_ses-01-03_finetune.pth",
    "rtmindeye_sub005_avg": "/data/derivatives/rtmindeye_paper/rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_MSTsplit_0_avgrepeats_finalmask.pth",
}


def _extract_state_dict(ckpt):
    """Lightning ckpts wrap state_dict in 'state_dict'; raw .pth is the dict itself."""
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict", "encoder"):
            if key in ckpt:
                inner = ckpt[key]
                if isinstance(inner, dict) and len(inner) > 0 and any(hasattr(v, "ndim") for v in inner.values()):
                    return inner
        return ckpt
    return {"weight": ckpt}


def _extract_matrices(sd, min_dim=50, max_layers=64):
    """Walk a state_dict for 2D weight tensors above min_dim."""
    import torch
    out = []
    for name, t in sd.items():
        if not isinstance(t, torch.Tensor): continue
        if t.ndim != 2: continue
        if min(t.shape) < min_dim: continue
        out.append((name, t.detach().cpu().numpy().astype(np.float32)))
        if len(out) >= max_layers: break
    return out


def analyze_ckpt(path: Path, max_layers: int = 16):
    import torch
    import jax.numpy as jnp
    import wwj
    from wwj.core import _eigvals

    print(f"\n=== {path.name}  ({path.stat().st_size / 1e6:.0f} MB) ===")
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except RuntimeError as e:
        print(f"  CORRUPT / unreadable: {e.__class__.__name__}")
        return None
    sd = _extract_state_dict(ckpt)
    print(f"  state_dict: {len(sd)} entries")
    mats = _extract_matrices(sd, min_dim=50, max_layers=max_layers)
    print(f"  2D matrices >= 50 min-dim: {len(mats)} (showing up to {max_layers})\n")

    if not mats:
        print("  (no 2D matrices found at min_dim=50; skipping)")
        return None

    print(f"{'layer':38s}  {'shape':>14s}  {'alpha':>6s}  {'CI95':>14s}  {'PL>EXP':>8s}  {'PL>LN':>8s}  {'PL valid'}")
    print("-" * 110)
    results = []
    for name, W in mats:
        eigs = _eigvals(jnp.asarray(W))
        boot = wwj.bootstrap_alpha_ci(eigs, n_bootstrap=200, ci=0.95)
        fit = wwj.fit_distributions(eigs)
        a = boot["alpha"]
        pl_valid = (fit["pl_vs_exp_lrt"] > 0) and (fit["pl_vs_ln_lrt"] > 0) and np.isfinite(a)
        results.append({"name": name, "alpha": a, "ci_low": boot["ci_low"],
                        "ci_high": boot["ci_high"], "pl_valid": pl_valid,
                        "shape": W.shape})
        ci_str = f"[{boot['ci_low']:.2f},{boot['ci_high']:.2f}]"
        print(f"{name[-38:]:38s}  {str(W.shape):>14s}  {a:6.2f}  {ci_str:>14s}  "
              f"{fit['pl_vs_exp_lrt']:+8.1f}  {fit['pl_vs_ln_lrt']:+8.1f}  "
              f"{'yes' if pl_valid else 'NO'}")

    alphas = np.array([r["alpha"] for r in results])
    finite = alphas[np.isfinite(alphas)]
    n_valid = sum(r["pl_valid"] for r in results)
    print(f"\n  Aggregate (finite α only, n={len(finite)}/{len(results)}): "
          f"mean α={finite.mean():.2f}  mean |α-2|={np.abs(finite - 2).mean():.2f}  "
          f"near2(1.5-2.5)={int(((finite >= 1.5) & (finite <= 2.5)).sum())}  "
          f"PL-valid layers: {n_valid}/{len(results)}")
    return results


def demo_perturbation_sweep():
    """Sketch of the full validation pipeline; runs the perturbation API on a
    synthetic MRI-like volume to confirm the TorchIO transforms produce sensible
    drift magnitudes. The actual robustness-vs-alpha correlation requires the
    project-specific encoder + a held-out validation set."""
    import wwj.perturbations as wp
    print("\n=== TorchIO MRI perturbation API smoke test (synthetic volume) ===")
    rng = np.random.default_rng(0)
    vol = rng.standard_normal((48, 64, 64)).astype(np.float32) * 100 + 500
    print(f"{'perturbation':16s}  {'severity':>8s}  {'mean |drift|':>14s}  {'max |drift|':>14s}")
    for P in wp.DEFAULT_MRI_SUITE:
        for sev in (0.25, 0.5, 1.0):
            out = P.apply(vol, severity=sev)
            d = np.abs(out - vol)
            print(f"{P.name:16s}  {sev:8.2f}  {d.mean():14.3f}  {d.max():14.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None, help="single ckpt path")
    ap.add_argument("--all", action="store_true", help="run all known candidates")
    ap.add_argument("--max-layers", type=int, default=16,
                    help="cap the per-ckpt layer count for readable output")
    ap.add_argument("--no-perturbation-demo", action="store_true")
    args = ap.parse_args()

    if args.ckpt is not None:
        if not args.ckpt.is_file():
            print(f"missing: {args.ckpt}"); sys.exit(1)
        analyze_ckpt(args.ckpt, max_layers=args.max_layers)
    elif args.all:
        for label, path in CKPT_CANDIDATES.items():
            p = Path(path)
            if p.is_file():
                analyze_ckpt(p, max_layers=args.max_layers)
            else:
                print(f"\n=== {label}  MISSING at {path}")
    else:
        # Default: just BrainIAC
        analyze_ckpt(Path(CKPT_CANDIDATES["brainiac"]), max_layers=args.max_layers)

    if not args.no_perturbation_demo:
        demo_perturbation_sweep()


if __name__ == "__main__":
    main()
