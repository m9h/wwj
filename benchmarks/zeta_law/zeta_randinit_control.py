"""Random-init / untrained control for the covariance-α signature (the NeuroAtlas caveat).

NeuroAtlas (Kontras et al., arXiv:2605.14698) shows clinical EEG-FMs barely beat a *random-initialised*
backbone and do not beat generic time-series FMs, and that brain-age performance does not scale with
embedding dimension. That is a warning for our covariance-α result: every learned FM is super-critical
(α_cov<2) while classical features are at/above critical — but is α_cov<2 a signature of *learned* structure,
or merely of pushing a heavy-tailed brain-data covariance through a high-dim map (geometric, not quality)?

No untrained deep-FM embeddings exist on disk (a forward pass we have not run), so we use the faithful
runnable proxy of NeuroAtlas's random-init backbone: an UNTRAINED random Gaussian projection of a real brain
input (morphometry) into FM-like dimension. If that is *also* super-critical, α_cov<2 is inherited from the
input covariance geometry, not evidence the FM learned useful structure — sharpening "diagnostic, not
predictor". A pure-noise (no brain data) Gaussian matrix is included: the validity gate must reject it
(Marchenko–Pastur), confirming the gate is doing its job.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/zeta_randinit_control.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbn_bayesian_alpha import _post  # noqa: E402  (Bayesian α posterior + validity gate)
from zeta_diagnostic import covariance_spectrum  # noqa: E402
from hbn_structural import _zscore  # noqa: E402

VC = "/data/derivatives/volume_conduction"
FM_GLOB = "/data/derivatives/peer_fm_ww/hbn_full/emb/*.npz"
SEED = 0                                        # fixed; scripts forbid Math.random-style nondeterminism


def _load_fm(name):
    """Stack one learned-FM embedding across all cached subjects."""
    X = []
    for f in sorted(glob.glob(FM_GLOB)):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if name in d:
            X.append(np.asarray(d[name], float))
    return np.vstack(X)


def _rand_proj(X_in, d_out, rng):
    """Untrained random Gaussian projection of a real brain input to dimension d_out (NeuroAtlas
    random-init-backbone proxy). Column-normalised so the map neither inflates nor shrinks scale."""
    G = rng.standard_normal((X_in.shape[1], d_out)) / np.sqrt(X_in.shape[1])
    return X_in @ G


def _row(label, X):
    ap, mp, ppc, gate = _post(covariance_spectrum(X))
    print(f"{label:34s} n={X.shape[0]:5d} d={X.shape[1]:4d}  α={ap['alpha_mean']:5.2f} "
          f"[{ap['ci_low']:.2f},{ap['ci_high']:.2f}]  P(α<2)={ap['p_alpha_lt_2']:.2f}  "
          f"gate={str(gate):>5s}", flush=True)
    return ap["alpha_mean"], ap["p_alpha_lt_2"], gate


def main():
    rng = np.random.default_rng(SEED)
    morph = _zscore(np.load(f"{VC}/morphometry_4s456.npz", allow_pickle=True)["X"].astype(float))

    print("== Random-init / untrained control for covariance-α (NeuroAtlas caveat) ==")
    print("-- learned foundation models (super-critical claim under test) --")
    for name in ["neurostorm", "swift", "cortex_mae_volume"]:
        _row(f"learned:{name}", _load_fm(name))

    print("-- UNTRAINED random projection of real brain input (morphometry) --")
    for d_out in [288, 768]:
        _row(f"randproj(morph)->d{d_out}", _rand_proj(morph, d_out, rng))

    print("-- references --")
    _row("raw morphometry (classical input)", morph)
    _row("pure Gaussian noise (no brain data)", rng.standard_normal((morph.shape[0], 288)))

    print("\nReading: if the untrained random projection of morphometry is ALSO super-critical (α<2, gated),\n"
          "then α_cov<2 is inherited from the brain-input covariance geometry — a diagnostic of representational\n"
          "geometry, NOT evidence the FM learned useful structure. The pure-noise row should FAIL the gate.")


if __name__ == "__main__":
    main()
