"""Enrich the HBN n=1121 volume-FM cache with CortexMAE-volume (mni_cortex) embeddings.

The 3rd volume FM for the full-cohort leaderboard. CortexMAE-volume's MNICortexReader reads a
plain MNI152-2mm nii and masks to the schaefer-400 cortex voxels -> takes the HBN CPAC niis
directly (no surface projection). Adds a "cortex_mae_volume" key to each existing per-subject
.npz (alongside neurostorm/swift), re-reading the nii once per subject.

Runs in ghcr.io/m9h/neurostorm-arm:26.04-arm.2 (torch 2.12 satisfies CortexMAE's torch<2.13 pin;
26.05/26.06 would NOT). Env: PYTHONPATH=CortexMAE/src:~/.local site-packages, HF_HOME=hf_cache,
HF_TOKEN mounted.

    python benchmarks/cmae_volume_enrich.py --limit 1      # 1-subject smoke test
    python benchmarks/cmae_volume_enrich.py                # full cohort (idempotent skip)
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import torch

MANIFEST = "/data/derivatives/peer_fm_ww/hbn1228_manifest.csv"
EMB_DIR = Path("/data/derivatives/peer_fm_ww/hbn_full/emb")
HBN_TR = 0.8  # HBN rest TR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="cortex_mae_volume")
    args = ap.parse_args()
    key = args.model

    # Stub the surface-only deps (pycortex `cortex`, `neuromaps`) that transforms.py/nisc.py import
    # at module level but the VOLUME (mni_cortex) path never calls — avoids the pycortex/neuromaps
    # aarch64 install (same bypass the WW analysis used). If the volume path ever touches them this
    # raises AttributeError loudly rather than silently mis-embedding.
    import sys, types
    _nm = types.ModuleType("neuromaps"); _nmt = types.ModuleType("neuromaps.transforms")
    _nm.transforms = _nmt
    sys.modules.setdefault("neuromaps", _nm)
    sys.modules.setdefault("neuromaps.transforms", _nmt)
    sys.modules.setdefault("cortex", types.ModuleType("cortex"))

    from cortex_mae.inference import CortexMAE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> building {args.model} on {dev}", flush=True)
    m = CortexMAE.from_pretrained(args.model, device=dev)
    print(">>> model ready", flush=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    if args.limit:
        rows = rows[: args.limit]
    done = skip = miss = 0
    for i, r in enumerate(rows):
        sub = r["sub"]; npz = EMB_DIR / f"{sub}.npz"
        if not npz.exists():
            miss += 1; continue
        d = dict(np.load(npz))
        if key in d and not args.limit:
            skip += 1; continue
        try:
            out = m.run_embedding(r["nii"], tr=HBN_TR)
            emb = out.patch_embeds.float().mean(dim=(0, 1)).cpu().numpy()  # (D,)
            d[key] = emb
            np.savez(npz, **d)
            done += 1
            if args.limit:
                print(f"  [{sub}] {key} emb dim={emb.shape} mean|v|={np.abs(emb).mean():.4f}", flush=True)
        except Exception as e:
            print(f"  [fail {sub}] {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  done={done} skip={skip} miss={miss}", flush=True)
    print(f">>> {key}: {done} embedded, {skip} already had it, {miss} no base npz", flush=True)


if __name__ == "__main__":
    main()
