"""Operationally-realistic H&E pathology perturbations -- the perturbation set
that is most directly relevant to nanopath's leaderboard suite. These mirror
the per-axis variation the Tellez 2019 / Macenko 2009 literature identifies
as the dominant cross-scanner / cross-laboratory nuisance for tile encoders.

Each perturbation's severity in [0, 1] maps to a physically-grounded magnitude:
- HEStainShift: severity 1.0 -> Macenko sigma_alpha=0.3, sigma_beta=0.1 (the
  upper end of cross-scanner variation per Tellez et al. 2019).
- ScannerBlur:  severity 1.0 -> Gaussian sigma=3.0 px (heavy focus drift,
  comparable to a worst-case z-stack misalignment).
- JPEGCompression: severity 1.0 -> quality=10 (extreme compression artifacts).
"""

from __future__ import annotations

from typing import Optional
import io

import numpy as np

from wwj.perturbations.base import BasePerturbation


# Macenko reference H&E stain matrix (rows = stain, cols = RGB OD vector).
# Same values as nanopath/dataloader.py's HE_STAIN_MATRIX.
HE_STAIN = np.array(
    [[0.5626, 0.7201, 0.4062],   # Hematoxylin (nuclei, purple-blue)
     [0.2159, 0.8012, 0.5581]],  # Eosin       (cytoplasm/stroma, pink)
    dtype=np.float32,
)


class HEStainShift(BasePerturbation):
    """OD-space concentration perturbation in the Macenko H&E stain matrix.
    Same math as nanopath/dataloader.py's BeerLambertJitter, but applied at
    test-time as a robustness probe rather than at train-time as augmentation.
    Severity maps to (sigma_alpha, sigma_beta) = (0.3, 0.1) * severity."""

    name = "stain_shift"

    def apply(self, image, severity=1.0, rng=None):
        rng = rng or np.random.default_rng()
        sigma_a = float(severity) * 0.3
        sigma_b = float(severity) * 0.1
        # Accept (H, W, 3) uint8 or float, or (N, H, W, 3) batch.
        was_batched = image.ndim == 4
        x = image[None] if not was_batched else image
        x = x.astype(np.float32)
        if x.max() > 1.5:
            x /= 255.0
        N, H, W, _ = x.shape
        rgb = np.clip(x, 1 / 255, 1.0)
        od = -np.log(rgb).reshape(N, H * W, 3)
        stain_pinv = np.linalg.pinv(HE_STAIN.T)
        conc = od @ stain_pinv.T          # (N, HW, 2)
        # Independent alpha/beta per batch item per stain.
        alpha = 1.0 + sigma_a * rng.uniform(-1, 1, size=(N, 1, 2)).astype(np.float32)
        beta = sigma_b * rng.uniform(-1, 1, size=(N, 1, 2)).astype(np.float32)
        conc_p = conc * alpha + beta
        od_p = conc_p @ HE_STAIN          # (N, HW, 3)
        rgb_p = np.exp(-od_p).reshape(N, H, W, 3)
        rgb_p = np.clip(rgb_p, 0.0, 1.0)
        out = (rgb_p * 255).astype(np.uint8) if image.dtype == np.uint8 else rgb_p
        return out if was_batched else out[0]


class ScannerBlur(BasePerturbation):
    """Gaussian blur emulating focus drift between WSI scanners.
    severity 1.0 -> sigma=3.0 px on a 224x224 tile."""

    name = "scanner_blur"

    def apply(self, image, severity=1.0, rng=None):
        from scipy.ndimage import gaussian_filter
        sigma = float(severity) * 3.0
        if sigma <= 0:
            return image
        # Per-channel blur; don't smooth across channels or batch axis.
        if image.ndim == 4:
            sig = (0, sigma, sigma, 0)
        else:
            sig = (sigma, sigma, 0)
        return gaussian_filter(image, sigma=sig)


class JPEGCompression(BasePerturbation):
    """Lossy JPEG re-encoding -- emulates the WSI storage pipeline used by
    most clinical scanners. severity 1.0 -> quality=10 (heavy artifacts)."""

    name = "jpeg_compression"

    def apply(self, image, severity=1.0, rng=None):
        from PIL import Image
        quality = max(1, int(round(100 - severity * 90)))  # severity 0 -> q100, 1 -> q10
        if image.ndim == 4:
            return np.stack([self._compress_one(x, quality) for x in image])
        return self._compress_one(image, quality)

    @staticmethod
    def _compress_one(arr, quality):
        from PIL import Image
        was_float = arr.dtype != np.uint8
        u = (arr * 255).clip(0, 255).astype(np.uint8) if was_float else arr
        buf = io.BytesIO()
        Image.fromarray(u).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out = np.array(Image.open(buf))
        return out.astype(np.float32) / 255 if was_float else out
