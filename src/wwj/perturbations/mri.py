"""Operationally-realistic MRI perturbations via TorchIO.

These are the realistic acquisition-side nuisances for smri-fm / fmri-fm /
realtime-mindeye / BrainIAC-style foundation models -- motion, susceptibility
distortion, RF inhomogeneity (bias field), k-space spikes, ghosting. Severity
in [0, 1] maps to physically-meaningful magnitudes calibrated against the
defaults TorchIO ships with (which themselves are calibrated to typical
clinical-acquisition realistic ranges).

Inputs are expected as 3D float arrays (D, H, W) or (C, D, H, W) -- TorchIO
operates on torchio.ScalarImage which we wrap transparently.

Why TorchIO and not a hand-rolled implementation: TorchIO's transforms are the
de-facto standard MRI augmentation library used across MONAI, fastMRI, and
most neuroimaging benchmarks; they implement physically-grounded perturbations
(e.g. RandomMotion uses Fourier-domain warping consistent with subject motion
during k-space acquisition, not just pixel-space translation).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from wwj.perturbations.base import BasePerturbation


def _as_torchio(volume):
    """Wrap a numpy array as torchio.ScalarImage. Lazy import."""
    import torch
    import torchio as tio
    arr = volume
    if arr.ndim == 3:
        arr = arr[None]  # add channel axis (C, D, H, W)
    t = torch.from_numpy(arr.astype(np.float32))
    return tio.ScalarImage(tensor=t)


def _from_torchio(img):
    out = img.tensor.numpy()
    return out[0] if out.shape[0] == 1 else out


class RandomMotion(BasePerturbation):
    """Subject-motion artifacts via k-space-domain warping (TorchIO RandomMotion).
    severity=1.0 -> degrees=15, translation=15mm, num_transforms=3 (heavy)."""

    name = "motion"

    def apply(self, volume, severity=1.0, rng=None):
        import torchio as tio
        s = float(severity)
        if s <= 0: return volume
        t = tio.RandomMotion(degrees=15 * s, translation=15 * s,
                             num_transforms=max(1, int(round(3 * s))))
        return _from_torchio(t(_as_torchio(volume)))


class RandomBiasField(BasePerturbation):
    """RF inhomogeneity (multiplicative low-frequency intensity drift).
    severity=1.0 -> coefficients amplitude=0.5 (TorchIO heavy default)."""

    name = "bias_field"

    def apply(self, volume, severity=1.0, rng=None):
        import torchio as tio
        s = float(severity)
        if s <= 0: return volume
        t = tio.RandomBiasField(coefficients=0.5 * s, order=3)
        return _from_torchio(t(_as_torchio(volume)))


class RandomGhosting(BasePerturbation):
    """Periodic ghosting artifacts -- typically caused by aliasing or pulsatile
    flow. severity=1.0 -> intensity=1.0 (max physically-realistic)."""

    name = "ghosting"

    def apply(self, volume, severity=1.0, rng=None):
        import torchio as tio
        s = float(severity)
        if s <= 0: return volume
        t = tio.RandomGhosting(intensity=1.0 * s,
                               num_ghosts=max(1, int(round(10 * s))))
        return _from_torchio(t(_as_torchio(volume)))


class RandomSpike(BasePerturbation):
    """k-space spike artifacts (electromagnetic interference during scan).
    severity=1.0 -> intensity=2.0, num_spikes=5."""

    name = "spike"

    def apply(self, volume, severity=1.0, rng=None):
        import torchio as tio
        s = float(severity)
        if s <= 0: return volume
        t = tio.RandomSpike(intensity=(0.5 * s, 2.0 * s),
                            num_spikes=max(1, int(round(5 * s))))
        return _from_torchio(t(_as_torchio(volume)))


class RandomNoise(BasePerturbation):
    """Gaussian noise -- standard MRI thermal noise. severity=1.0 -> std=0.25
    of dynamic range."""

    name = "noise"

    def apply(self, volume, severity=1.0, rng=None):
        import torchio as tio
        s = float(severity)
        if s <= 0: return volume
        t = tio.RandomNoise(std=0.25 * s)
        return _from_torchio(t(_as_torchio(volume)))


# Convenience: the standard sweep used for BrainIAC / smri-fm / realtime-mindeye
# style robustness validation. Mirrors the perturbation set TorchIO recommends
# for MRI augmentation pipelines.
DEFAULT_MRI_SUITE = [RandomMotion(), RandomBiasField(),
                     RandomGhosting(), RandomSpike(), RandomNoise()]
