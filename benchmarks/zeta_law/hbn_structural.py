"""E1-structural — fold the HBN *structural* modalities into the data-side zeta diagnostic.

The 4S morphometry (subjects × regional-GM-volume) and the 4S structural connectome (subjects ×
vectorised edges) are just two more data modalities: `run_e1` gives each its covariance-spectrum γ
(with the 3-factor power-law validity gate) and, per target (age + the four psychopathology
factors), the alignment-spectrum β / regime — directly comparable to the functional FM embeddings
(neurostorm/swift) in `hbn_e1.py` and to the weight spectra in the wwjd paper.

Notes:
  - Structural feature columns are standardised (z-scored) first: unlike FM embeddings, raw regional
    volumes / streamline counts span orders of magnitude, so the uncentred second moment would be
    dominated by scale rather than spectral structure.
  - The connectome edge vector is high-dim (p≈104k ≫ n); we PC-reduce to its nonzero-variance scores
    (lossless — rank ≤ n), so both the covariance and the p×p alignment spectra are memory-safe.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/hbn_structural.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbn_e1 import run_e1, format_report, MANIFEST, HBN_TARGETS, MIN_VALID  # noqa: E402

VC = "/data/derivatives/volume_conduction"


def _zscore(X: np.ndarray) -> np.ndarray:
    sd = X.std(0)
    keep = sd > 0
    return (X[:, keep] - X[:, keep].mean(0)) / sd[keep]


def _pc_reduce(X: np.ndarray) -> np.ndarray:
    """Lossless reduction to nonzero-variance PC scores when p > n (rank ≤ n)."""
    Xc = X - X.mean(0, keepdims=True)
    if Xc.shape[1] <= Xc.shape[0]:
        return Xc
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U * S


def _targets() -> dict:
    rows = list(csv.DictReader(open(MANIFEST)))
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    return {r["sub"]: {t: f(r.get(t, "")) for t in HBN_TARGETS} for r in rows}


def _morphometry(res: str = "456"):
    d = np.load(f"{VC}/morphometry_4s{res}.npz", allow_pickle=True)
    ids = [str(i).replace("sub-", "") for i in d["ids"]]
    return _zscore(d["X"].astype(float)), ids


def _connectome(res: str = "456"):
    # tck2connectome sizes each matrix by the max parcel label PRESENT in that subject's warped atlas,
    # so sizes vary (a few high parcels drop out per subject) -> pad/truncate every subject to P×P.
    P = int(res)
    iu = np.triu_indices(P, 1)
    files = sorted(glob.glob(f"{VC}/connectomes/sub-*/connectome_{res}.csv"))
    rows, ids = [], []
    for fp in files:
        C = np.loadtxt(fp, delimiter=",")
        if C.ndim != 2 or C.shape[0] != C.shape[1] or C.sum() <= 0:
            continue
        M = np.zeros((P, P)); n = min(C.shape[0], P); M[:n, :n] = C[:n, :n]
        rows.append(M[iu]); ids.append(os.path.basename(os.path.dirname(fp)).replace("sub-", ""))
    return _pc_reduce(_zscore(np.array(rows, float))), ids


def main():
    tg = _targets()
    feats, tgts = {}, {}
    for name, loader in [("struct_morph_4S456", _morphometry), ("struct_conn_4S456", _connectome)]:
        X, ids = loader()
        feats[name] = X
        tmap = {}
        for t in HBN_TARGETS:
            y = np.array([tg.get(s, {}).get(t, float("nan")) for s in ids])
            if int(np.isfinite(y).sum()) >= MIN_VALID:
                tmap[t] = y
        tgts[name] = tmap
        print(f"{name}: X{X.shape}  {len(ids)} subjects (manifest-matched: "
              f"{sum(s in tg for s in ids)})  targets {list(tmap)}")
    print()
    print(format_report(run_e1(feats, tgts)))


if __name__ == "__main__":
    main()
