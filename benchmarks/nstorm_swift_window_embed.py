"""Per-WINDOW NeuroSTORM + SwiFT latents for HBN rest — companions to cmae_window_embed.py.

These two volume FMs take the whole run in one forward (-> one pooled latent). To get the
within-subject repeat structure the identity-trap audit needs, we window the raw (T, V) brain-mask
timeseries into non-overlapping nf=20-frame clips (matching CortexMAE's ~18-21 windows so #windows
is comparable across the 3 FMs), forward each clip -> per-window latent (N_win, D).

Reads each nii ONCE (both models share the 228k brain mask). Per-subject checkpoint (resumable).
Runs in neurostorm-arm:26.04 via docker -d (mamba/causal-conv1d baked). Reuses the validated
build_neurostorm / build_swift from hbn_local_probe.

    python benchmarks/nstorm_swift_window_embed.py --limit 400
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/mhough/dev/wwj/benchmarks")
MANIFEST = "/data/derivatives/peer_fm_ww/hbn1228_manifest.csv"
NF = 20  # temporal window (frames); matches the models' img_size temporal dim


@torch.inference_mode()
def windowed_latents(models, mask, nii_path, tr, device, nf=NF):
    """Return {name: (N_win, D)} for one subject. One nii read; window -> forward each clip."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(nii_path))
    if img.shape[:3] != (91, 109, 91):
        return None
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = np.ascontiguousarray(data.transpose(3, 2, 1, 0))   # (T,Z,Y,X)
    bold = data[:, mask]                                       # (T, V) mask C-order
    T = bold.shape[0]; nclip = T // nf
    if nclip < 1:
        return None
    out = {name: [] for name, _, _ in models}
    for c in range(nclip):
        chunk = np.nan_to_num(bold[c * nf:(c + 1) * nf])       # (nf, V)
        for name, wrapper, transform in models:
            sample = {"bold": torch.from_numpy(chunk), "tr": float(tr),
                      "mean": torch.zeros(1, chunk.shape[1]), "std": torch.ones(1, chunk.shape[1])}
            sample = transform(sample)
            x = sample["bold"].unsqueeze(0).to(device)
            with torch.autocast("cuda", torch.bfloat16, enabled=str(device).startswith("cuda")):
                o = wrapper({"bold": x})
            out[name].append(o.patch_embeds.float().mean(dim=(0, 1)).cpu().numpy())
    return {name: np.stack(v) for name, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--tag", default="")   # "" = rest; "_movieDM" -> hbn_window_movieDM/{model}
    args = ap.parse_args()
    WDIR = {n: Path(f"/data/derivatives/peer_fm_ww/hbn_window{args.tag}/{n}") for n in ("neurostorm", "swift")}
    for d in WDIR.values():
        d.mkdir(parents=True, exist_ok=True)

    import hbn_local_probe as H
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> device={dev}; building NeuroSTORM + SwiFT", flush=True)
    ns_w, ns_t = H.build_neurostorm(dev)
    sw_w, sw_t = H.build_swift(dev)
    models = [("neurostorm", ns_w, ns_t), ("swift", sw_w, sw_t)]
    mask = ns_t.mask.cpu().numpy()                             # shared 228k brain mask
    print(">>> models ready", flush=True)

    rows = list(csv.DictReader(open(args.manifest)))
    if args.limit:
        rows = rows[: args.limit]
    done = skip = 0
    for i, r in enumerate(rows):
        sub = r["sub"]
        outs = {n: WDIR[n] / f"{sub}.npz" for n, _, _ in models}
        if all(p.exists() for p in outs.values()):
            skip += 1; continue
        try:
            tr = float(r.get("nii_tr", 0.8) or 0.8)
            res = windowed_latents(models, mask, r["nii"], tr, dev)
            if res:
                for name, w in res.items():
                    np.savez(outs[name], win=w.astype(np.float32))   # CHECKPOINT
                done += 1
                if done <= 2:
                    print(f"  [{sub}] " + " ".join(f"{n}={w.shape}" for n, w in res.items()), flush=True)
        except Exception as e:
            print(f"  [fail {sub}] {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  done={done} skip={skip}", flush=True)
    print(f">>> per-window NeuroSTORM/SwiFT: {done} embedded, {skip} cached", flush=True)


if __name__ == "__main__":
    main()
