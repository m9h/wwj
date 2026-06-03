from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class BasePerturbation(ABC):
    """Apply an operationally-realistic perturbation at a given severity.

    Severity is a scalar in [0, 1] where 0 is identity (clean) and 1 is the
    perturbation's maximum physically-plausible magnitude. Each concrete
    subclass defines what that maximum means for its modality (e.g. for
    pathology stain shift, severity=1 corresponds to Macenko-σ_α=0.3, the
    upper bound of what cross-scanner variation produces in practice).

    Inputs are numpy arrays in the modality's native format:
      - CV images: HWC uint8
      - MRI volumes: 3D float (T1/T2/FLAIR intensity)
      - EEG epochs: (channels, time) float32 microvolts
      - Pathology tiles: HWC uint8 RGB

    The severity-vs-physically-meaningful-magnitude mapping is documented
    per-subclass so cross-modality robustness curves can be reported on a
    common 0-1 axis without losing the underlying physical interpretation."""

    name: str

    @abstractmethod
    def apply(self, inputs: np.ndarray, severity: float = 1.0,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Returns perturbed inputs of the same shape and dtype as input."""

    def sweep(self, inputs: np.ndarray, severities: list[float],
              rng: Optional[np.random.Generator] = None) -> list[np.ndarray]:
        """Apply at each severity, return a list of perturbed batches."""
        rng = rng or np.random.default_rng()
        return [self.apply(inputs, s, rng) for s in severities]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
