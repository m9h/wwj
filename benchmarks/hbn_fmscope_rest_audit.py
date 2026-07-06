"""fMRI-FMScope identity-trap audit of fMRI-FOUNDATION-MODEL latents (HBN resting-state).

CLAIM-2 core result: how much of a frozen fMRI-FM latent is SUBJECT IDENTITY? Uses per-window
latents (multiple windows per subject; cmae_window_embed.py) so subject identity is a decomposable
between-subject axis with within-subject repeats.

Two metrics (the rest-valid half of FMScope — the clinical/stimulus-survival half needs movies):
  1. null-subject-variance ratio: observed between-subject latent variance / a random-grouping null
     — the fMRI-FM analog of the EEG FMScope "13-89x null" identity domination.
  2. subject re-identification (fingerprinting): balanced accuracy of decoding WHICH subject a
     window came from (held-out windows), reusing fmscope.subject_probe. Chance = 1/n_subjects.
Also reports the LEACE subject-subspace rank (how many latent dims the identity axis occupies).

Pure numpy/sklearn/fmscope — no GPU. Positions vs Finn-2015 FC fingerprinting: here the fingerprint
is read from a FROZEN PRETRAINED FM LATENT, not raw FC.

    python benchmarks/hbn_fmscope_rest_audit.py --models cmae
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/mhough/dev/fmscope")
MANIFEST = "/data/derivatives/peer_fm_ww/hbn1228_manifest.csv"
WIN = {"cmae": Path("/data/derivatives/peer_fm_ww/hbn_window/cmae"),
       "neurostorm": Path("/data/derivatives/peer_fm_ww/hbn_window/neurostorm"),
       "swift": Path("/data/derivatives/peer_fm_ww/hbn_window/swift")}
OUT = Path("/home/mhough/dev/hippy-feat/results/fmscope"); OUT.mkdir(parents=True, exist_ok=True)


def load_windows(model):
    """X (total_windows, D), subj_idx (total_windows,) int, subs (list of subject ids)."""
    d = WIN[model]
    files = sorted(d.glob("*.npz"))
    X, sidx, subs = [], [], []
    for si, f in enumerate(files):
        w = np.load(f)["win"]                      # (N_win, D)
        if w.ndim != 2 or w.shape[0] < 4:          # need a few windows per subject
            continue
        X.append(w); sidx.append(np.full(w.shape[0], len(subs))); subs.append(f.stem)
    return np.vstack(X), np.concatenate(sidx), subs


def nsv_ratio(X, sidx, n_perm=100, seed=0):
    """Between-subject latent variance vs random-grouping null. Returns (obs, null_mean, ratio)."""
    gm = X.mean(0); N = len(X)
    def between(labels):
        b = 0.0
        for s in np.unique(labels):
            m = labels == s
            b += m.sum() * np.sum((X[m].mean(0) - gm) ** 2)
        return b / N
    obs = between(sidx)
    rng = np.random.RandomState(seed)
    nulls = [between(rng.permutation(sidx)) for _ in range(n_perm)]
    nm = float(np.mean(nulls))
    return float(obs), nm, float(obs / nm if nm > 0 else np.nan)


def audit_model(model):
    from fmscope.diagnostics.erasure import subject_probe, whiten, subject_eraser
    X, sidx, subs = load_windows(model)
    n_sub = len(subs); D = X.shape[1]
    print(f"\n=== {model}  ({len(X)} windows, {n_sub} subjects, dim={D}, "
          f"~{len(X)/max(n_sub,1):.1f} win/subj) ===", flush=True)
    # z-score features
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-6)
    obs, nm, ratio = nsv_ratio(Xz, sidx)
    print(f"  null-subject-variance ratio (between/null): {ratio:.1f}x", flush=True)
    ba, ns = subject_probe(Xz, sidx, kind="linear", cap=100, n_splits=5)
    chance = 1.0 / ns if ns else np.nan
    print(f"  subject re-ID (fingerprint) BA: {ba:.3f}  (chance 1/{ns} = {chance:.4f}, "
          f"= {ba/chance:.0f}x chance)", flush=True)
    mu, Xc, W, Wp, _ = whiten(Xz, shrinkage=True)
    _, _, rank = subject_eraser(Xc, W, Wp, sidx)
    print(f"  LEACE subject-subspace rank/dim: {rank}/{D}", flush=True)
    return {"model": model, "n_windows": len(X), "n_subjects": n_sub, "dim": D,
            "nsv_ratio": ratio, "nsv_obs": obs, "nsv_null": nm,
            "subject_reid_ba": ba, "reid_chance": chance, "reid_x_chance": ba / chance if chance else None,
            "subject_subspace_rank": int(rank)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["cmae"])
    a = ap.parse_args()
    res = []
    for m in a.models:
        if not WIN[m].exists() or not any(WIN[m].glob("*.npz")):
            print(f"[{m}] no per-window latents yet — skip"); continue
        res.append(audit_model(m))
    (OUT / "rest_fmscope_audit.json").write_text(json.dumps(res, indent=2))
    print(f"\n[saved] {OUT/'rest_fmscope_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
