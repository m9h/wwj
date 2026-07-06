"""Per-WINDOW CortexMAE-volume latents for HBN rest — the FMScope identity-trap substrate.

Unlike cmae_volume_enrich.py (one mean-pooled vector per subject), this keeps the per-temporal-
window latents (N_win, D) so there are MULTIPLE samples per subject — the within-subject repeat
structure the identity-trap audit needs (subject-variance ratio + subject re-ID / fingerprinting).

CortexMAE's run_embedding pads/unfolds the run into non-overlapping temporal windows internally;
patch_embeds is [N_win, L, D] -> we mean over patches L only, keeping windows -> [N_win, D].

Per-subject checkpoint to .npz (resumable). Runs in neurostorm-arm:26.04 via docker -d (the path
that completed the full B embed). Lazy transforms.py patch + cortex/neuromaps stubs as before.

    python benchmarks/cmae_window_embed.py --limit 400
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import torch

MANIFEST = "/data/derivatives/peer_fm_ww/hbn1228_manifest.csv"
HBN_TR = 0.8


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="cortex_mae_volume")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--tag", default="")   # "" = rest -> hbn_window/cmae; "_movieDM" -> hbn_window_movieDM/cmae
    args = ap.parse_args()
    WIN_DIR = Path(f"/data/derivatives/peer_fm_ww/hbn_window{args.tag}/cmae"); WIN_DIR.mkdir(parents=True, exist_ok=True)

    import sys, types
    _nm = types.ModuleType("neuromaps"); _nmt = types.ModuleType("neuromaps.transforms")
    _nm.transforms = _nmt
    sys.modules.setdefault("neuromaps", _nm); sys.modules.setdefault("neuromaps.transforms", _nmt)
    sys.modules.setdefault("cortex", types.ModuleType("cortex"))

    from cortex_mae.inference import CortexMAE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> building {args.model} on {dev}", flush=True)
    m = CortexMAE.from_pretrained(args.model, device=dev)
    print(">>> model ready", flush=True)

    rows = list(csv.DictReader(open(args.manifest)))
    if args.limit:
        rows = rows[: args.limit]
    done = skip = miss = 0
    for i, r in enumerate(rows):
        sub = r["sub"]; out = WIN_DIR / f"{sub}.npz"
        if out.exists():
            skip += 1; continue
        try:
            emb = m.run_embedding(r["nii"], tr=HBN_TR)
            w = emb.patch_embeds.float().mean(dim=1).cpu().numpy()   # (N_win, D) — keep windows
            np.savez(out, win=w.astype(np.float32))                  # CHECKPOINT immediately
            done += 1
            if done <= 2:
                print(f"  [{sub}] windows={w.shape}", flush=True)
        except Exception as e:
            print(f"  [fail {sub}] {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  done={done} skip={skip} miss={miss}", flush=True)
    print(f">>> per-window: {done} embedded, {skip} cached", flush=True)


if __name__ == "__main__":
    main()
