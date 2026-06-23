"""Red-green TDD for the E1 zeta-law runner (`run_e1`) and its HBN wiring.

E1 takes per-modality feature matrices + per-modality targets (mirroring HBN's
phase_probe: one embedding matrix per volume-FM model, targets = age + FACTORS) and returns, per
modality, the covariance-spectrum diagnostic (γ + validity) and, per (modality, target), the
alignment-spectrum diagnostic (β, P(β>1), regime).

Fixtures plant the spectra exactly via an orthonormal design X = Q·diag(d):
  cov(X) = diag(d²) so the covariance rank-decay is the planted s; and for y = Q·c the per-mode
  aligned energy is (dᵢcᵢ)² ∝ i^{-(s + (β-s))} = i^{-β}, so the alignment rank-decay is the planted β,
  set independently of s by cᵢ = i^{(s-β)/2}.

Run:  /home/mhough/dev/wwj/.venv/bin/python -m pytest benchmarks/zeta_law/test_e1.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbn_e1 import run_e1   # noqa: E402  (red until hbn_e1.py exists)


def _features_with_spectrum(n: int, p: int, s: float, seed: int) -> np.ndarray:
    """X = Q·diag(d), Q orthonormal columns, d_i = i^{-s/2} → cov(X)=diag(i^{-s})."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, p)))          # reduced QR: Q is n×p, orthonormal cols
    d = np.arange(1, p + 1.0) ** (-s / 2)
    return Q * d


def _target_with_alignment(X: np.ndarray, beta: float) -> np.ndarray:
    """y = Q·c with c_i = i^{-β/2}. With the explained-variance normalization a_i = coeffᵢ²/λᵢ ∝ c_i²,
    the alignment rank-decay is the planted β — independent of the covariance s (the deconfounding)."""
    p = X.shape[1]
    Q = X / np.linalg.norm(X, axis=0)                          # recover orthonormal Q (‖X_i‖ = d_i)
    c = np.arange(1, p + 1.0) ** (-beta / 2)
    return Q @ c


def _mp_features(n: int, p: int, seed: int) -> np.ndarray:
    """Random Gaussian matrix → Marchenko–Pastur covariance (the non-power-law null)."""
    return np.random.default_rng(seed).standard_normal((n, p))


def test_run_e1_structure_and_modality_validity():
    feats = {"pl": _features_with_spectrum(1200, 400, s=1.2, seed=1),
             "mp": _mp_features(1200, 400, seed=2)}
    rep = run_e1(feats, {})
    assert set(rep["modalities"]) == {"pl", "mp"}
    assert rep["modalities"]["pl"]["is_power_law"] is True           # genuine power-law modality
    assert rep["modalities"]["mp"]["is_power_law"] is False          # random/MP modality rejected
    assert abs(rep["modalities"]["pl"]["rank_decay"] - 1.2) < 0.3    # γ recovered


def test_run_e1_alignment_regimes():
    X = _features_with_spectrum(1400, 400, s=1.0, seed=3)
    targets = {"m": {"hi": _target_with_alignment(X, beta=1.6),        # concentrated → variance-limited
                     "lo": _target_with_alignment(X, beta=0.6)}}       # diffuse → resolution-limited
    rep = run_e1({"m": X}, targets)
    hi, lo = rep["alignments"][("m", "hi")], rep["alignments"][("m", "lo")]
    assert abs(hi["rank_decay"] - 1.6) < 0.4 and abs(lo["rank_decay"] - 0.6) < 0.4
    assert hi["p_beta_gt_1"] > 0.5 and lo["p_beta_gt_1"] < 0.5
    assert hi["regime"].startswith("variance") and lo["regime"].startswith("resolution")


def test_run_e1_skips_targets_with_too_few_valid():
    X = _features_with_spectrum(1200, 400, s=1.0, seed=6)
    y = np.full(1200, np.nan); y[:10] = 1.0                          # only 10 valid → below min tail
    rep = run_e1({"m": X}, {"m": {"short": y}})
    assert ("m", "short") not in rep["alignments"]


if __name__ == "__main__":   # runnable without pytest (the wwj .venv has wwj but not pytest)
    for _fn in (test_run_e1_structure_and_modality_validity,
                test_run_e1_alignment_regimes,
                test_run_e1_skips_targets_with_too_few_valid):
        _fn(); print(f"PASS  {_fn.__name__}")
    print("all E1 tests passed")
