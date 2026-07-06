"""Dual-regression stage-1 + FSLnets-style netmats for the MELODIC FC backbone (HBN rest).

Given group-MELODIC spatial ICs, for each subject: DR stage-1 spatial regression (group maps ->
per-subject IC timeseries), then node AMPLITUDES (ts std) + full-correlation and ridge
PARTIAL-correlation netmats (FSLnets default). Node amplitudes (esp. a global/vigilance IC) are
the plausibly-most-EEG-coupled feature; partial-corr netmats are the Oxford-default FC edges.

    python benchmarks/dr_netmats.py --gica /mnt/t9/melodic_fc/pilot40.gica \
        --list /mnt/t9/melodic_fc/pilot40.txt --out /mnt/t9/melodic_fc/netmats_pilot40.npz
"""
from __future__ import annotations
import argparse
import numpy as np
import nibabel as nib


def ridge_partial(ts, rho=0.1):
    """FSLnets-style ridge partial correlation from IC timeseries (n_ic, T)."""
    z = (ts - ts.mean(1, keepdims=True)) / (ts.std(1, keepdims=True) + 1e-9)
    cov = np.cov(z)
    P = np.linalg.inv(cov + rho * np.eye(cov.shape[0]))     # ridge-regularized precision
    d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 0.0)
    return pc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gica", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rho", type=float, default=0.1)
    a = ap.parse_args()

    gica = a.gica.rstrip("/")
    ic = nib.load(f"{gica}/melodic_IC.nii.gz")
    mask = nib.load(f"{gica}/mask.nii.gz").get_fdata() > 0
    maps = np.asarray(ic.dataobj, dtype=np.float32).reshape(-1, ic.shape[-1])[mask.reshape(-1)]  # (V, n_ic)
    n_ic = maps.shape[1]
    pinv = np.linalg.pinv(maps)                                 # (n_ic, V) — DR stage-1 operator
    print(f">>> group ICs={n_ic}, mask voxels={maps.shape[0]}", flush=True)

    files = [l.strip() for l in open(a.list) if l.strip()]
    amps, full, part, subs = [], [], [], []
    for i, f in enumerate(files):
        try:
            bold = np.asarray(nib.load(f).dataobj, dtype=np.float32)
            bold = bold.reshape(-1, bold.shape[-1])[mask.reshape(-1)]   # (V, T)
            ts = pinv @ bold                                            # (n_ic, T) DR stage-1
            amps.append(ts.std(1))
            full.append(np.corrcoef(ts))
            part.append(ridge_partial(ts, a.rho))
            subs.append(f.split("/sub-")[1].split("_")[0] if "/sub-" in f else str(i))
        except Exception as e:
            print(f"  [fail {i}] {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    amps = np.vstack(amps); full = np.stack(full); part = np.stack(part)
    np.savez(a.out, amplitudes=amps, netmat_full=full, netmat_partial=part,
             subjects=np.array(subs), n_ic=n_ic)
    # quick sanity: mean partial-corr edge strength + amplitude spread
    iu = np.triu_indices(n_ic, 1)
    print(f">>> saved {a.out}: {amps.shape[0]} subj, {n_ic} ICs", flush=True)
    print(f"    mean|partial edge|={np.abs(part[:, iu[0], iu[1]]).mean():.3f}  "
          f"amp range={amps.mean(0).min():.2f}-{amps.mean(0).max():.2f}", flush=True)


if __name__ == "__main__":
    main()
