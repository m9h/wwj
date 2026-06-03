# wwj.perturbations: operationally-realistic perturbation namespace, designed as
# the empirical-validation companion to wwj's spectral diagnostics.
#
# The motivation: Martin's RG theory (arXiv 2026) predicts that layers closer
# to alpha=2 should be more robust to natural input perturbations. To test
# that claim empirically you need (1) wwj's per-layer spectral observables and
# (2) modality-appropriate realistic perturbations applied at the model's
# input boundary. This namespace covers (2) via thin adapters around the
# canonical per-modality toolkits, with a shared BasePerturbation interface.
#
# Inspired by Kitware NRTK's framing (operationally-realistic, not adversarial)
# but generalized across modalities since three of the four projects this is
# built for are not CV (smri-fm, eeg-fm-spectral, hippy-feat).

from wwj.perturbations.base import BasePerturbation
from wwj.perturbations.pathology import HEStainShift, ScannerBlur, JPEGCompression

__all__ = ["BasePerturbation", "HEStainShift", "ScannerBlur", "JPEGCompression"]

# MRI imports are lazy because torchio is an optional dependency.
def _maybe_register_mri():
    try:
        from wwj.perturbations import mri as _mri
        for name in ("RandomMotion", "RandomBiasField", "RandomGhosting",
                     "RandomSpike", "RandomNoise", "DEFAULT_MRI_SUITE"):
            globals()[name] = getattr(_mri, name)
            __all__.append(name)
    except ImportError:
        pass

_maybe_register_mri()
