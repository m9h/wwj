"""RG-robustness forward-pass experiment on BrainIAC's brain-age model.

Loads brainage.ckpt (which our spectral survey found is RG-optimal: mean
|alpha-2|=0.12, 15/16 layers PL-valid), pushes a real T1 volume through with
TorchIO-perturbed copies at multiple severities, measures:

  1) brain-age prediction drift (in years) vs severity
  2) CLS-token embedding L2 drift relative to clean vs severity
  3) drift slope (= dL2/dseverity) per perturbation type

The RG prediction (Martin 2026): models with layer alpha distributions
concentrated near 2 should have flatter robustness curves than models with
alpha far from 2. brainage.ckpt is the cleanest spectral case in our survey;
this is the first concrete empirical test of that prediction on a real
clinical MRI foundation model.

Architecture (from BrainIAC/src/model.py):
    ViT(in_channels=1, img_size=(96,96,96), patch_size=(16,16,16),
        hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12)
    + Linear(768, 1) head

Input preprocessing follows BrainIAC's validation transform:
    Resized to 96x96x96 trilinear, ScaleIntensity to [0, 1].

Usage:
    uv run python benchmarks/brainage_perturbation_sweep.py
    uv run python benchmarks/brainage_perturbation_sweep.py --volume <T1.nii.gz>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


BRAINAGE_CKPT = "/home/mhough/dev/BrainIAC/src/checkpoints/brainage.ckpt"
BRAINIAC_CKPT = "/home/mhough/dev/BrainIAC/src/checkpoints/BrainIAC.ckpt"
DEFAULT_T1 = "/home/mhough/dev/BrainIAC/src/data/sample/processed/subpixar009_T1w.nii.gz"

CKPT_META = {
    "brainage": {"path": BRAINAGE_CKPT, "alpha_dist": 0.12, "has_head": True},
    "brainiac": {"path": BRAINIAC_CKPT, "alpha_dist": 0.40, "has_head": False},
}


def load_brainage_model(which: str = "brainage"):
    """Build the bare ViT + Linear head and load the requested ckpt's state_dict.
    which='brainage' has the trained regression head; which='brainiac' is the
    SimCLR pretrained backbone (head stays at random init -- only CLS-drift
    comparison is meaningful)."""
    import torch
    import torch.nn as nn
    from monai.networks.nets import ViT

    meta = CKPT_META[which]
    backbone = ViT(in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
                   hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
                   save_attn=True)
    head = nn.Linear(768, 1)

    ckpt = torch.load(meta["path"], map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    bb_prefix = "model.backbone.backbone." if which == "brainage" else "backbone."
    head_prefix = "model.classifier.fc." if which == "brainage" else None
    bb_sd, head_sd = {}, {}
    for k, v in sd.items():
        if k.startswith(bb_prefix):
            bb_sd[k[len(bb_prefix):]] = v
        elif head_prefix and k.startswith(head_prefix):
            head_sd[k[len(head_prefix):]] = v
    # strict=False because MONAI 1.5.2's ViT added cross-attention blocks that the
    # BrainIAC training-era ViT didn't have. The cross_attn.* and norm_cross_attn.*
    # parameters stay at random init -- safe because they're only used when context
    # tokens are passed to forward(), which we don't do (single-volume self-attn only).
    missing, unexpected = backbone.load_state_dict(bb_sd, strict=False)
    expected_missing = sum(1 for k in missing if "cross_attn" in k or "norm_cross_attn" in k)
    actual_missing = [k for k in missing if "cross_attn" not in k and "norm_cross_attn" not in k]
    if actual_missing or unexpected:
        raise RuntimeError(f"unexpected key drift -- missing={actual_missing[:3]} unexpected={unexpected[:3]}")
    print(f"  backbone: skipped {expected_missing} cross-attention keys (MONAI 1.5.2 addition; unused at inference)")
    if meta["has_head"]:
        head.load_state_dict(head_sd, strict=True)
    else:
        print(f"  head: brainiac SimCLR ckpt has no regression head; using random init (CLS-drift only)")
    backbone.eval(); head.eval()
    return backbone, head


def load_t1(path: Path):
    """Load a T1 nifti, resize to 96^3, normalize to [0, 1]. Returns (D,H,W) float32."""
    import nibabel as nib
    img = nib.load(str(path)).get_fdata().astype(np.float32)
    if img.ndim == 4:
        img = img[..., 0]
    # MONAI's Resized uses trilinear; do it manually with scipy.ndimage.zoom
    from scipy.ndimage import zoom
    factors = tuple(96 / s for s in img.shape)
    vol = zoom(img, factors, order=1)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-12)
    return vol.astype(np.float32)


def encode(backbone, head, vol_np):
    """vol_np: (D, H, W) float32 in [0, 1]. Returns (cls_embedding 768-d, age_pred scalar)."""
    import torch
    x = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    with torch.no_grad():
        features = backbone(x)
        cls = features[0][:, 0]                   # (1, 768)
        age = head(cls).squeeze().item()
    return cls.squeeze(0).numpy(), age


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=Path, default=Path(DEFAULT_T1))
    ap.add_argument("--ckpt", choices=list(CKPT_META.keys()), default="brainage")
    ap.add_argument("--severities", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"Selected ckpt: {args.ckpt} (mean |alpha-2| = {CKPT_META[args.ckpt]['alpha_dist']})")

    if not args.volume.is_file():
        raise SystemExit(f"missing volume: {args.volume}")

    print(f"Loading model from {CKPT_META[args.ckpt]['path']}")
    backbone, head = load_brainage_model(args.ckpt)
    print(f"Loading T1 from {args.volume}")
    vol = load_t1(args.volume)
    print(f"  shape={vol.shape}  range=[{vol.min():.3f}, {vol.max():.3f}]  mean={vol.mean():.3f}")

    # Reference forward pass
    print("\nClean forward pass...")
    cls_clean, age_clean = encode(backbone, head, vol)
    cls_clean_norm = float(np.linalg.norm(cls_clean))
    print(f"  brain-age prediction: {age_clean:+.2f} years (relative)")
    print(f"  CLS embedding norm:   {cls_clean_norm:.3f}")

    # Perturbation sweep
    import wwj.perturbations as wp
    rng = np.random.default_rng(args.seed)
    suite = wp.DEFAULT_MRI_SUITE

    print(f"\nPerturbation sweep (severities = {args.severities}):\n")
    print(f"{'perturbation':14s}  {'sev':>5s}  {'age drift (y)':>14s}  {'CLS L2 drift':>13s}  {'CLS rel drift':>14s}")
    print("-" * 75)
    rows = []
    for P in suite:
        for sev in args.severities:
            vol_p = P.apply(vol, severity=sev, rng=rng)
            cls_p, age_p = encode(backbone, head, vol_p.astype(np.float32))
            age_drift = age_p - age_clean
            l2_drift = float(np.linalg.norm(cls_p - cls_clean))
            rel_drift = l2_drift / (cls_clean_norm + 1e-12)
            rows.append((P.name, sev, age_drift, l2_drift, rel_drift))
            print(f"{P.name:14s}  {sev:5.2f}  {age_drift:+14.3f}  {l2_drift:13.3f}  {rel_drift:14.4f}")
        print()

    # Per-perturbation drift slope (least-squares) -- the actual RG-prediction signal
    print("\n=== Drift slope per perturbation (rel-drift vs severity) ===")
    print(f"{'perturbation':14s}  {'slope':>9s}  {'intercept':>10s}  {'r^2':>6s}")
    slopes = {}
    for P in suite:
        xs = np.array([r[1] for r in rows if r[0] == P.name])
        ys = np.array([r[4] for r in rows if r[0] == P.name])
        # least-squares fit
        coef = np.polyfit(xs, ys, 1)
        slope, intercept = float(coef[0]), float(coef[1])
        y_pred = np.polyval(coef, xs)
        ss_res = np.sum((ys - y_pred) ** 2)
        ss_tot = np.sum((ys - ys.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        slopes[P.name] = slope
        print(f"{P.name:14s}  {slope:9.4f}  {intercept:10.4f}  {r2:6.3f}")

    print(f"\n=== RG-robustness reading ===")
    print(f"  brainage.ckpt spectral health (from prior wwj run): mean |alpha-2| = 0.12 (15/16 layers in [1.7, 2.3])")
    print(f"  Mean drift slope across perturbations: {np.mean(list(slopes.values())):.4f} / unit severity")
    print(f"  Interpretation: brainage is spectrally near the RG optimum (mean |a-2|=0.12);")
    print(f"  these drift slopes ARE the empirical robustness signal. The Martin RG prediction")
    print(f"  is that lower |alpha-2| -> lower drift slope. Validating that direction requires")
    print(f"  running the same sweep on a spectrally far-from-optimum ckpt (e.g.")
    print(f"  rtmindeye sub005 finetune, mean |alpha-2|=4.73) and comparing slopes.")


if __name__ == "__main__":
    main()
