"""Cohort-scale RG-robustness validation: scale the single-subject sweep at
benchmarks/brainage_perturbation_sweep.py up to n=20 WAND fmriprep T1ws and
both BrainIAC ckpts, with paired statistics.

Tests the Martin RG prediction at proper N:
  H0: drift slopes are the same for brainage and brainiac ckpts.
  H1: brainage (mean |alpha-2|=0.12) has smaller drift slopes than brainiac
      (mean |alpha-2|=0.40) -- the RG-spectral robustness prediction.

For each subject we compute the same per-perturbation drift slope as the
single-subject script, then run paired tests (Wilcoxon signed-rank for
robustness; sign-test for the directional check) per perturbation type.

Usage:
    uv run python benchmarks/cohort_sweep.py
    uv run python benchmarks/cohort_sweep.py --n 10 --cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from brainage_perturbation_sweep import (
    CKPT_META, load_brainage_model, encode, load_t1,
)


WAND_ANAT = Path("/data/datasets/fmriprep/wand")


def find_t1s(n: int = 20) -> list[Path]:
    """Find n native-space preprocessed T1ws in WAND fmriprep output."""
    cand = sorted(WAND_ANAT.glob("sub-*/anat/sub-*_desc-preproc_T1w.nii.gz"))
    # Exclude MNI-space files
    cand = [p for p in cand if "space-MNI" not in p.name]
    return cand[:n]


def per_subject_slopes(backbone, head, vol_np, perturbations, severities, rng, device):
    """Return {perturbation_name: (slope, r2)} for one subject on one ckpt."""
    import torch
    cls_clean, _ = encode_dev(backbone, head, vol_np, device)
    cls_clean_norm = float(np.linalg.norm(cls_clean))
    out = {}
    for P in perturbations:
        rel_drifts = []
        for sev in severities:
            if sev == 0:
                rel_drifts.append(0.0)
                continue
            vol_p = P.apply(vol_np, severity=sev, rng=rng)
            cls_p, _ = encode_dev(backbone, head, vol_p.astype(np.float32), device)
            d = float(np.linalg.norm(cls_p - cls_clean) / (cls_clean_norm + 1e-12))
            rel_drifts.append(d)
        # least-squares slope; report r^2 as a noisiness check
        coef = np.polyfit(severities, rel_drifts, 1)
        slope, intercept = float(coef[0]), float(coef[1])
        yhat = np.polyval(coef, severities)
        ys = np.array(rel_drifts)
        ss_res = float(np.sum((ys - yhat) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        out[P.name] = (slope, r2)
    return out


def encode_dev(backbone, head, vol_np, device):
    import torch
    x = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        features = backbone(x)
        cls = features[0][:, 0]
        age = head(cls).squeeze().item()
    return cls.squeeze(0).cpu().numpy(), age


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of subjects")
    ap.add_argument("--severities", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--cuda", action="store_true", help="run on CUDA if available")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import wwj.perturbations as wp

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    subjects = find_t1s(args.n)
    print(f"Found {len(subjects)} subjects in {WAND_ANAT}")
    if len(subjects) < args.n:
        print(f"  WARN: requested {args.n} but only {len(subjects)} available")

    perturbations = wp.DEFAULT_MRI_SUITE
    rng = np.random.default_rng(args.seed)

    # Load volumes once
    print(f"Loading {len(subjects)} T1 volumes (resize + normalize to 96^3)...")
    t0 = time.time()
    vols = []
    for p in subjects:
        try:
            vols.append((p.stem.split("_")[0], load_t1(p)))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    print(f"  loaded {len(vols)} volumes in {time.time() - t0:.1f}s")

    # Pre-generate perturbations once per subject so brainage and brainiac
    # see exactly the same perturbed inputs (paired comparison).
    print("Pre-generating perturbed volumes (paired across ckpts)...")
    t0 = time.time()
    perturbed = {}  # (subj_id, pert_name, sev) -> np.ndarray
    for subj_id, vol in vols:
        for P in perturbations:
            for sev in args.severities:
                if sev == 0:
                    continue
                perturbed[(subj_id, P.name, sev)] = P.apply(vol, severity=sev, rng=rng)
    print(f"  generated {len(perturbed)} perturbed volumes in {time.time() - t0:.1f}s")

    results = {}  # ckpt -> {subj_id: {pert_name: (slope, r2)}}
    for ckpt_label in ("brainage", "brainiac"):
        print(f"\n=== {ckpt_label} ckpt  (mean |α-2| = {CKPT_META[ckpt_label]['alpha_dist']}) ===")
        t0 = time.time()
        backbone, head = load_brainage_model(ckpt_label)
        backbone.to(device); head.to(device)
        print(f"  model loaded + moved to {device} in {time.time() - t0:.1f}s")

        per_subj = {}
        t0 = time.time()
        for subj_id, vol in vols:
            cls_clean, _ = encode_dev(backbone, head, vol, device)
            cls_clean_norm = float(np.linalg.norm(cls_clean))
            subj_out = {}
            for P in perturbations:
                drifts = [0.0]  # severity=0
                sevs = [0.0]
                for sev in args.severities:
                    if sev == 0:
                        continue
                    vol_p = perturbed[(subj_id, P.name, sev)]
                    cls_p, _ = encode_dev(backbone, head, vol_p.astype(np.float32), device)
                    drifts.append(float(np.linalg.norm(cls_p - cls_clean) / (cls_clean_norm + 1e-12)))
                    sevs.append(sev)
                coef = np.polyfit(sevs, drifts, 1)
                slope = float(coef[0])
                yhat = np.polyval(coef, sevs)
                ys = np.array(drifts)
                ss_res = float(np.sum((ys - yhat) ** 2))
                ss_tot = float(np.sum((ys - ys.mean()) ** 2)) + 1e-12
                subj_out[P.name] = (slope, 1.0 - ss_res / ss_tot)
            per_subj[subj_id] = subj_out
        print(f"  cohort forward pass complete in {time.time() - t0:.1f}s")
        results[ckpt_label] = per_subj

    # === Statistics ===
    print(f"\n{'='*72}")
    print(f"COHORT RESULTS  (n={len(vols)} subjects, severities={args.severities})")
    print(f"{'='*72}")

    from scipy.stats import wilcoxon

    print(f"\n{'perturbation':14s}  {'brainage mean':>14s}  {'brainiac mean':>14s}  "
          f"{'Δ (b<i favored)':>17s}  {'sign (n+/-)':>13s}  {'Wilcoxon p':>11s}  {'RG win'}")
    print("-" * 110)

    overall_rg_wins = 0
    overall_perts = 0
    for P in perturbations:
        sl_brainage = np.array([results["brainage"][sid][P.name][0] for sid, _ in vols])
        sl_brainiac = np.array([results["brainiac"][sid][P.name][0] for sid, _ in vols])
        delta = sl_brainiac - sl_brainage  # positive if brainage is more robust (RG prediction)
        n_pos = int((delta > 0).sum())
        n_neg = int((delta < 0).sum())
        try:
            stat, p = wilcoxon(sl_brainiac, sl_brainage, alternative="greater")
        except Exception:
            p = float("nan")
        rg_win = (n_pos > n_neg) and (p < 0.1)
        overall_rg_wins += int(rg_win)
        overall_perts += 1
        print(f"{P.name:14s}  {sl_brainage.mean():14.4f}  {sl_brainiac.mean():14.4f}  "
              f"{delta.mean():+17.4f}  {n_pos}+/{n_neg}-".rjust(13).ljust(13) +
              f"  {p:11.4f}  {'✓' if rg_win else '·'}")

    print(f"\nOverall: RG prediction holds in {overall_rg_wins}/{overall_perts} perturbations "
          f"(directional + p<0.1)")
    # Also report combined across all (perturbation, subject) pairs
    all_brainage = np.concatenate([
        [results["brainage"][sid][P.name][0] for sid, _ in vols] for P in perturbations
    ])
    all_brainiac = np.concatenate([
        [results["brainiac"][sid][P.name][0] for sid, _ in vols] for P in perturbations
    ])
    all_delta = all_brainiac - all_brainage
    n_pos = int((all_delta > 0).sum())
    n_neg = int((all_delta < 0).sum())
    print(f"\nAll (perturbation, subject) pairs combined (n={len(all_delta)}):")
    print(f"  brainage mean slope:  {all_brainage.mean():.4f}")
    print(f"  brainiac mean slope:  {all_brainiac.mean():.4f}")
    print(f"  mean Δ (b<i):         {all_delta.mean():+.4f}")
    print(f"  sign test:            {n_pos} support RG, {n_neg} against ({n_pos / (n_pos + n_neg + 1e-12):.1%})")
    try:
        s, p = wilcoxon(all_brainiac, all_brainage, alternative="greater")
        print(f"  Wilcoxon (one-sided): p = {p:.4g}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
