"""Multi-seed cohort sweep -- characterize the seed sensitivity of the
RG-robustness finding from benchmarks/cohort_sweep.py (n=20 WAND, p=0.0002
at seed=0).

For each of N seeds, we regenerate the per-subject perturbation set with that
seed and run the same brainage-vs-brainiac paired comparison. Models are
loaded ONCE (not per seed) since the per-seed work is in the perturbation
generation and the per-volume forward passes, both of which depend on seed
but the model weights don't. This makes N seeds roughly N times slower than
a single seed run, dominated by perturbation generation + forward passes.

Output per seed:
  - per-perturbation Wilcoxon p (brainiac > brainage, one-sided)
  - combined-all-pairs Wilcoxon p
  - mean delta per perturbation

Aggregated across seeds:
  - distribution of per-perturbation effect sizes (mean ± std)
  - fraction of seeds where the combined-all-pairs Wilcoxon is significant
  - paired (brainage, brainiac) slopes averaged across seeds per subject

Usage:
    uv run python benchmarks/multi_seed_sweep.py --n 20 --n_seeds 10 --cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainage_perturbation_sweep import (
    CKPT_META, load_brainage_model, load_t1,
)
from cohort_sweep import find_t1s, encode_dev, WAND_ANAT


def run_one_seed(seed, vols, perturbations, severities, models_devs):
    """Generate perturbations with `seed` and run paired forward pass on all
    ckpts in models_devs (dict ckpt_label -> (backbone, head, device)).
    Returns: results[ckpt_label][subj_id][pert_name] = slope (float)."""
    rng = np.random.default_rng(seed)
    # Generate perturbations once per seed; shared across ckpts (paired).
    perturbed = {}
    for subj_id, vol in vols:
        for P in perturbations:
            for sev in severities:
                if sev == 0:
                    continue
                perturbed[(subj_id, P.name, sev)] = P.apply(vol, severity=sev, rng=rng)

    results = {}
    for ckpt_label, (backbone, head, device) in models_devs.items():
        per_subj = {}
        for subj_id, vol in vols:
            cls_clean, _ = encode_dev(backbone, head, vol, device)
            cls_clean_norm = float(np.linalg.norm(cls_clean))
            subj_out = {}
            for P in perturbations:
                drifts = [0.0]
                sevs = [0.0]
                for sev in severities:
                    if sev == 0: continue
                    vol_p = perturbed[(subj_id, P.name, sev)]
                    cls_p, _ = encode_dev(backbone, head, vol_p.astype(np.float32), device)
                    drifts.append(float(np.linalg.norm(cls_p - cls_clean) / (cls_clean_norm + 1e-12)))
                    sevs.append(sev)
                slope = float(np.polyfit(sevs, drifts, 1)[0])
                subj_out[P.name] = slope
            per_subj[subj_id] = subj_out
        results[ckpt_label] = per_subj
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of subjects")
    ap.add_argument("--n_seeds", type=int, default=10, help="number of perturbation seeds")
    ap.add_argument("--severities", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed_start", type=int, default=0)
    args = ap.parse_args()

    import torch
    import wwj.perturbations as wp
    from scipy.stats import wilcoxon

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    subjects = find_t1s(args.n)
    print(f"Found {len(subjects)} subjects")
    perturbations = wp.DEFAULT_MRI_SUITE

    # Load volumes once
    t0 = time.time()
    vols = []
    for p in subjects:
        try:
            vols.append((p.stem.split("_")[0], load_t1(p)))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    print(f"Loaded {len(vols)} volumes in {time.time() - t0:.1f}s")

    # Load both models once
    models_devs = {}
    for ckpt_label in ("brainage", "brainiac"):
        bb, hd = load_brainage_model(ckpt_label)
        bb.to(device); hd.to(device)
        models_devs[ckpt_label] = (bb, hd, device)
    print(f"Models loaded; running {args.n_seeds} seeds...")

    # Per-seed runs
    all_seeds = {}  # seed -> results dict
    per_seed_p = []  # combined p per seed
    per_seed_pert_p = {P.name: [] for P in perturbations}
    per_seed_pert_delta = {P.name: [] for P in perturbations}

    for s in range(args.seed_start, args.seed_start + args.n_seeds):
        t0 = time.time()
        results = run_one_seed(s, vols, perturbations, args.severities, models_devs)
        # Compute stats
        ba = []  # brainage slopes flat
        bi = []  # brainiac slopes flat
        for P in perturbations:
            sl_ba = np.array([results["brainage"][sid][P.name] for sid, _ in vols])
            sl_bi = np.array([results["brainiac"][sid][P.name] for sid, _ in vols])
            ba.extend(sl_ba.tolist())
            bi.extend(sl_bi.tolist())
            try:
                _, p_pert = wilcoxon(sl_bi, sl_ba, alternative="greater")
            except Exception:
                p_pert = float("nan")
            per_seed_pert_p[P.name].append(float(p_pert))
            per_seed_pert_delta[P.name].append(float((sl_bi - sl_ba).mean()))
        ba = np.array(ba); bi = np.array(bi)
        try:
            _, p_comb = wilcoxon(bi, ba, alternative="greater")
        except Exception:
            p_comb = float("nan")
        per_seed_p.append(float(p_comb))
        all_seeds[s] = results
        print(f"  seed={s:3d}  combined Wilcoxon p={p_comb:.4g}  brainage mean={ba.mean():.4f}  "
              f"brainiac mean={bi.mean():.4f}  in {time.time()-t0:.1f}s")

    # === Aggregate across seeds ===
    print(f"\n{'='*78}")
    print(f"MULTI-SEED STABILITY  (N_seeds={args.n_seeds}, n_subjects={len(vols)})")
    print(f"{'='*78}")

    print(f"\nCombined-all-pairs Wilcoxon p across {args.n_seeds} seeds:")
    p_arr = np.array(per_seed_p)
    print(f"  mean p:    {p_arr.mean():.4g}")
    print(f"  median p:  {np.median(p_arr):.4g}")
    print(f"  max p:     {p_arr.max():.4g}")
    print(f"  fraction p < 0.01:  {(p_arr < 0.01).mean()*100:.0f}%  ({(p_arr<0.01).sum()}/{len(p_arr)})")
    print(f"  fraction p < 0.05:  {(p_arr < 0.05).mean()*100:.0f}%  ({(p_arr<0.05).sum()}/{len(p_arr)})")

    print(f"\nPer-perturbation effect size across seeds (delta = brainiac - brainage slope):")
    print(f"{'perturbation':14s}  {'mean Δ':>10s}  {'std Δ':>10s}  {'min Δ':>10s}  {'max Δ':>10s}  "
          f"{'p<0.05':>8s}  {'sign+':>6s}")
    print("-" * 86)
    for P in perturbations:
        deltas = np.array(per_seed_pert_delta[P.name])
        ps = np.array(per_seed_pert_p[P.name])
        sign_pos = (deltas > 0).sum()
        print(f"{P.name:14s}  {deltas.mean():+10.4f}  {deltas.std():10.4f}  "
              f"{deltas.min():+10.4f}  {deltas.max():+10.4f}  "
              f"{(ps < 0.05).sum()}/{len(ps)}".rjust(8) +
              f"  {sign_pos}/{len(deltas)}".rjust(8))

    # Per-subject mean across seeds (does the RG effect hold per subject?)
    print(f"\nPer-subject summary (mean delta across {args.n_seeds} seeds, averaged over 5 perturbations):")
    subj_deltas = {}
    for sid, _ in vols:
        ds = []
        for s in range(args.seed_start, args.seed_start + args.n_seeds):
            for P in perturbations:
                d = all_seeds[s]["brainiac"][sid][P.name] - all_seeds[s]["brainage"][sid][P.name]
                ds.append(d)
        subj_deltas[sid] = np.mean(ds)
    n_pos = sum(1 for d in subj_deltas.values() if d > 0)
    print(f"  Subjects supporting RG (mean Δ > 0 across all seeds and perts): {n_pos}/{len(subj_deltas)}")
    print(f"  Mean across all subjects: {np.mean(list(subj_deltas.values())):+.4f}")


if __name__ == "__main__":
    main()
