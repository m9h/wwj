"""Fold ALREADY-EXTRACTED HBN structural-FM embeddings into the data-side zeta diagnostic.

No re-extraction / no brainmarks run needed: smri-fm-fomo26's hbn_brainage_probe already cached
AMAES (320-d), FOMO60k (768-d), and MMUNetVAE (512-d) structural-FM embeddings on HBN, keyed by
`sub-NDAR…_ses-waveN`. We take one session per subject (baseline wave) and run them through `run_e1`
exactly like the functional FMs (neurostorm/swift) — raw embeddings, no z-scoring (homogeneous scale).

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/hbn_fm_existing.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbn_e1 import run_e1, format_report, MANIFEST, HBN_TARGETS, MIN_VALID  # noqa: E402

R = "/home/mhough/dev/smri-fm-fomo26/experiments/hbn_brainage_probe/results"
FM = {
    "amaes_struct":     f"{R}/amaes_embed/fomo25_embeddings.npz",
    "fomo60k_struct":   f"{R}/fomo60k_embed/fomo60k_embeddings.npz",
    "mmunetvae_struct": f"{R}/mmunetvae_embed_smoke/fomo25_mmunetvae_embeddings.npz",
}


def fm_embed(path: str):
    """{sub-NDAR…_ses-waveN: vec} -> (X, ids) taking the baseline (wave1) session per subject."""
    d = np.load(path, allow_pickle=True)
    by = {}
    for k in d.files:
        sid = k.split("_ses-")[0].replace("sub-", "")
        wave = k.split("_ses-")[1] if "_ses-" in k else ""
        if sid not in by or wave.startswith("wave1"):
            by[sid] = np.asarray(d[k], dtype=float)
    ids = list(by)
    return np.array([by[s] for s in ids]), ids


def targets():
    rows = list(csv.DictReader(open(MANIFEST)))
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    return {r["sub"]: {t: f(r.get(t, "")) for t in HBN_TARGETS} for r in rows}


def main():
    tg = targets()
    feats, tgts = {}, {}
    for name, path in FM.items():
        if not os.path.exists(path):
            print(f"[skip {name}] missing {path}"); continue
        X, ids = fm_embed(path)
        feats[name] = X
        tmap = {}
        for t in HBN_TARGETS:
            y = np.array([tg.get(s, {}).get(t, float("nan")) for s in ids])
            if int(np.isfinite(y).sum()) >= MIN_VALID:
                tmap[t] = y
        tgts[name] = tmap
        print(f"{name}: X{X.shape}  {len(ids)} subj (manifest-matched {sum(s in tg for s in ids)})  "
              f"targets {list(tmap)}")
    print()
    print(format_report(run_e1(feats, tgts)))


if __name__ == "__main__":
    main()
