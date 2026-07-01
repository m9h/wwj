"""Unified data-side zeta analysis over ALL ready HBN modalities — max N, FM vs classical examples.

Folds every modality currently in subjects×features form through one `run_e1`, so the covariance-α
(learned-FM super-critical α<2 vs classical) and alignment-β/regime (data- vs resolution-limited) are
read off consistently. FM embeddings are fed raw (homogeneous scale); hand-built/classical features are
z-scored (heterogeneous scale); the high-dim connectome is z-scored then PC-reduced (lossless, rank ≤ n).

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/hbn_dataside_all.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbn_e1 import run_e1, format_report, MANIFEST, HBN_TARGETS, MIN_VALID, load_hbn  # noqa: E402
from hbn_structural import _zscore, _connectome  # noqa: E402
from hbn_fm_existing import fm_embed, R as FMR  # noqa: E402

VC = "/data/derivatives/volume_conduction"


def _targets():
    rows = list(csv.DictReader(open(MANIFEST)))
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    return {r["sub"]: {t: f(r.get(t, "")) for t in HBN_TARGETS} for r in rows}


def _add(feats, tgts, tg, name, X, ids):
    feats[name] = X
    tmap = {}
    for t in HBN_TARGETS:
        y = np.array([tg.get(s, {}).get(t, float("nan")) for s in ids])
        if int(np.isfinite(y).sum()) >= MIN_VALID:
            tmap[t] = y
    tgts[name] = tmap
    print(f"  {name:18s} X{str(X.shape):12s} {len(ids)} subj (matched {sum(s in tg for s in ids)})", flush=True)


def main():
    tg = _targets()
    feats, tgts = load_hbn()                       # functional FMs: neurostorm, swift (with targets)
    for m in list(feats):
        print(f"  {m:18s} X{str(feats[m].shape):12s} (functional FM)", flush=True)

    # structural FM embeddings (raw)
    for name, sub in [("amaes_fm", "amaes_embed/fomo25_embeddings.npz"),
                      ("fomo60k_fm", "fomo60k_embed/fomo60k_embeddings.npz")]:
        X, ids = fm_embed(f"{FMR}/{sub}")
        _add(feats, tgts, tg, name, X, ids)

    # classical / hand-built (z-scored)
    for name, fn in [("morph_4s456", "morphometry_4s456.npz"),
                     ("morph_4s1056", "morphometry_4s1056.npz"),
                     ("blockpooled", "structural_emb.npz")]:
        d = np.load(f"{VC}/{fn}", allow_pickle=True)
        ids = [str(i).replace("sub-", "") for i in d["ids"]]
        _add(feats, tgts, tg, name, _zscore(d["X"].astype(float)), ids)

    # connectome (current cohort; z-scored + PC-reduced inside _connectome)
    Xc, idc = _connectome("456")
    if len(idc) >= MIN_VALID:
        _add(feats, tgts, tg, "connectome", Xc, idc)

    print("\n" + format_report(run_e1(feats, tgts)))


if __name__ == "__main__":
    main()
