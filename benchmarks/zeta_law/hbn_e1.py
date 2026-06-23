"""E1 — the data-side zeta-law diagnostic, wired to the HBN volume-FM cohort.

`run_e1` is the pure, tested core: given per-modality feature matrices and per-modality targets it
returns, per modality, the covariance-spectrum diagnostic (γ + power-law validity) and, per
(modality, target), the alignment-spectrum diagnostic (β, P(β>1), variance- vs resolution-limited
regime). See `zeta_law_extensions.md` and `data_vs_weight_spectra.md`.

The HBN adapter (`load_hbn`) mirrors `hbn_full_volume.phase_probe`'s loading WITHOUT importing or
modifying it: per-subject `.npz` embeddings keyed by volume-FM model (= modality), targets = age +
the four psychopathology FACTORS, with per-target missing-value handling. It is guarded by cache
existence, so on a machine without the HBN derivatives `main()` falls back to a synthetic demo.

    /home/mhough/dev/wwj/.venv/bin/python benchmarks/zeta_law/hbn_e1.py            # HBN if present, else demo
    /home/mhough/dev/wwj/.venv/bin/python -m pytest benchmarks/zeta_law/test_e1.py -q
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeta_diagnostic import covariance_spectrum, alignment_spectrum, diagnose_spectrum  # noqa: E402

MIN_VALID = 50                       # matches wwj's _n_meaningful gate / phase_probe's `< 50` skip

# --- HBN layout (mirrors hbn_full_volume.py constants; not imported to avoid its module-level mkdir)
EMB_DIR = Path("/data/derivatives/peer_fm_ww/hbn_full/emb")
MANIFEST = "/data/derivatives/peer_fm_ww/hbn1228_manifest.csv"
FACTORS = ["p_factor", "attention", "internalizing", "externalizing"]
HBN_TARGETS = ["age", *FACTORS]


# ---------------------------------------------------------------------------- core (tested)
def run_e1(features_by_modality: dict[str, np.ndarray],
           targets_by_modality: dict[str, dict[str, np.ndarray]],
           min_valid: int = MIN_VALID) -> dict:
    """E1 over a multimodal feature set.

    Args:
        features_by_modality: {modality: X (n_subjects × n_features)}.
        targets_by_modality:  {modality: {target: y (n_subjects,)}}, NaN marks missing subjects.
    Returns:
        {"modalities": {m: covariance diagnostic},
         "alignments": {(m, t): alignment diagnostic}}   (targets with < min_valid kept rows skipped)
    """
    modalities = {m: diagnose_spectrum(covariance_spectrum(X), "covariance")
                  for m, X in features_by_modality.items()}
    alignments = {}
    for m, X in features_by_modality.items():
        for t, y in targets_by_modality.get(m, {}).items():
            y = np.asarray(y, dtype=float)
            mask = np.isfinite(y)
            if int(mask.sum()) < min_valid:
                continue
            a = alignment_spectrum(X[mask], y[mask])
            alignments[(m, t)] = diagnose_spectrum(a, "alignment")
    return {"modalities": modalities, "alignments": alignments}


def format_report(rep: dict) -> str:
    lines = ["== E1: covariance spectra (per modality) =="]
    lines.append(f"  {'modality':22s} {'γ':>6s} {'power-law?':>10s} {'logBF(pl/exp)':>13s} {'ppc_p':>6s}")
    for m, d in rep["modalities"].items():
        lines.append(f"  {m:22s} {d['rank_decay']:6.2f} {str(d['is_power_law']):>10s} "
                     f"{d['logbf_pl_vs_exponential']:13.1f} {d['ppc_pvalue']:6.2f}")
    lines.append("\n== E1: alignment spectra (modality × target) ==")
    lines.append(f"  {'modality':18s} {'target':14s} {'β':>6s} {'P(β>1)':>7s}  regime")
    for (m, t), d in rep["alignments"].items():
        lines.append(f"  {m:18s} {t:14s} {d['rank_decay']:6.2f} {d['p_beta_gt_1']:7.2f}  "
                     f"{d['regime'].split(' (')[0]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- HBN adapter (wiring)
def _to_float(v) -> float:
    if v in ("", "n/a", "NaN", None):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _detect_models(emb_dir: Path) -> list[str]:
    for p in sorted(Path(emb_dir).glob("*.npz")):
        return list(np.load(p).files)          # keys of the first cached subject = the modalities
    return []


def load_hbn(models: list[str] | None = None, emb_dir: Path = EMB_DIR,
             manifest_path: str = MANIFEST, targets: list[str] = HBN_TARGETS,
             min_valid: int = MIN_VALID) -> tuple[dict, dict]:
    """Load HBN embeddings (per model = modality) + targets, mirroring `phase_probe`. Returns
    (features_by_modality, targets_by_modality) ready for `run_e1`. Requires the cache to exist."""
    emb_dir = Path(emb_dir)
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    lab = {r["sub"]: r for r in rows}
    if models is None:
        models = _detect_models(emb_dir)
    features, target_map = {}, {}
    for model in models:
        subs, E = [], []
        for r in rows:
            p = emb_dir / f"{r['sub']}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            if model not in d:
                continue
            subs.append(r["sub"]); E.append(d[model])
        if len(subs) < min_valid:
            continue
        features[model] = np.vstack(E)
        tmap = {}
        for t in targets:
            y = np.array([_to_float(lab[s].get(t)) for s in subs], dtype=float)   # NaN = missing
            if int(np.isfinite(y).sum()) >= min_valid:
                tmap[t] = y
        target_map[model] = tmap
    return features, target_map


# ---------------------------------------------------------------------------- synthetic fallback
def _synthetic_demo() -> None:
    rng = np.random.default_rng(0)

    def feats(n, p, s, seed):
        Q, _ = np.linalg.qr(rng.standard_normal((n, p)) if seed is None
                            else np.random.default_rng(seed).standard_normal((n, p)))
        return Q * (np.arange(1, p + 1.0) ** (-s / 2))

    X = feats(1400, 400, s=1.0, seed=3)
    Q = X / np.linalg.norm(X, axis=0)
    y_hi = Q @ (np.arange(1, 401.0) ** (-1.6 / 2))          # concentrated  (β≈1.6)
    y_lo = Q @ (np.arange(1, 401.0) ** (-0.6 / 2))          # diffuse       (β≈0.6)
    feats_by_mod = {"powerlaw_fm": X, "random_fm": rng.standard_normal((1400, 400))}
    tgts = {"powerlaw_fm": {"concentrated": y_hi, "diffuse": y_lo}}
    print("HBN cache not found — synthetic E1 demo (planted γ=1.0; β=1.6 / 0.6):\n")
    print(format_report(run_e1(feats_by_mod, tgts)))


def main() -> None:
    models = sys.argv[1:] or None
    if EMB_DIR.exists() and any(EMB_DIR.glob("*.npz")):
        feats, tgts = load_hbn(models)
        print(f"HBN E1 over {len(feats)} modalities from {EMB_DIR}\n")
        print(format_report(run_e1(feats, tgts)))
    else:
        _synthetic_demo()


if __name__ == "__main__":
    main()
