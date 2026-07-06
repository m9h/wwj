"""Cross-run (rest_run-1 vs rest_run-2) mean-pooled latents for HBN — fMRI FM-latent fingerprinting.

The fMRI analog of the EEG cross-session permanence test: enroll on run-1, re-identify on run-2.
For each subject, one mean-pooled latent per (run, model) for the 3 volume FMs -> the cross-run
fingerprint + differential-identifiability (I_diff, Amico-Goñi) can be computed on frozen FM
latents and compared to raw-FC connectome fingerprinting (Finn 2015, ~92-94%).

Per-subject checkpoint (resumable). neurostorm-arm:26.04 via docker -d. CortexMAE via run_embedding
(mni_cortex), NeuroSTORM/SwiFT via the whole-run brain-mask forward (reuse hbn_local_probe builders).

    python benchmarks/cross_run_embed.py --limit 400
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/mhough/dev/wwj/benchmarks")
MANIFEST = "/data/derivatives/peer_fm_ww/hbn_2run_manifest.csv"
EMB = Path("/data/derivatives/peer_fm_ww/hbn_2run/emb"); EMB.mkdir(parents=True, exist_ok=True)
HBN_TR = 0.8


@torch.inference_mode()
def nsw_wholerun(models_nsw, mask, nii_path, dev):
    """{name: (D,)} mean-pooled whole-run latent for NeuroSTORM/SwiFT from one nii."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(nii_path))
    if img.shape[:3] != (91, 109, 91):
        return {}
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = np.ascontiguousarray(data.transpose(3, 2, 1, 0))     # (T,Z,Y,X)
    bold = np.nan_to_num(data[:, mask])                          # (T, V)
    out = {}
    for name, wrapper, transform in models_nsw:
        sample = {"bold": torch.from_numpy(bold), "tr": HBN_TR,
                  "mean": torch.zeros(1, bold.shape[1]), "std": torch.ones(1, bold.shape[1])}
        sample = transform(sample)
        x = sample["bold"].unsqueeze(0).to(dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=str(dev).startswith("cuda")):
            o = wrapper({"bold": x})
        out[name] = o.patch_embeds.float().mean(dim=(0, 1)).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import types
    for m in ("neuromaps", "neuromaps.transforms", "cortex"):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["neuromaps"].transforms = sys.modules["neuromaps.transforms"]

    import hbn_local_probe as H
    from cortex_mae.inference import CortexMAE
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> device={dev}; building CortexMAE + NeuroSTORM + SwiFT", flush=True)
    cmae = CortexMAE.from_pretrained("cortex_mae_volume", device=str(dev))
    ns_w, ns_t = H.build_neurostorm(dev); sw_w, sw_t = H.build_swift(dev)
    models_nsw = [("neurostorm", ns_w, ns_t), ("swift", sw_w, sw_t)]
    mask = ns_t.mask.cpu().numpy()
    print(">>> models ready", flush=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    if args.limit:
        rows = rows[: args.limit]
    done = skip = 0
    for i, r in enumerate(rows):
        sub = r["sub"]; out = EMB / f"{sub}.npz"
        if out.exists():
            skip += 1; continue
        try:
            emb = {}
            for run, key in ((r["nii1"], "r1"), (r["nii2"], "r2")):
                emb[f"cmae_{key}"] = cmae.run_embedding(run, tr=HBN_TR).patch_embeds.float().mean(dim=(0, 1)).cpu().numpy()
                for name, v in nsw_wholerun(models_nsw, mask, run, dev).items():
                    emb[f"{name}_{key}"] = v
            if len(emb) == 6:
                np.savez(out, **{k: v.astype(np.float32) for k, v in emb.items()})
                done += 1
                if done <= 2:
                    print(f"  [{sub}] keys={sorted(emb)} dims={ {k:v.shape for k,v in emb.items()} }", flush=True)
            else:
                print(f"  [skip {sub}] only {len(emb)}/6 embeds", flush=True)
        except Exception as e:
            print(f"  [fail {sub}] {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  done={done} skip={skip}", flush=True)
    print(f">>> cross-run: {done} embedded, {skip} cached", flush=True)


if __name__ == "__main__":
    main()
