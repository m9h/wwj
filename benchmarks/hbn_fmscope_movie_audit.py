"""fMRI-FMScope MOVIE audit — the full (subject × stimulus) identity trap on HBN movieDM.

Unlike rest, the movie is a time-locked shared stimulus: window k = the same movie-timepoint for
every subject. That gives a proper (subject × stimulus) factorial, so we can run the FMScope tests
rest could not:

  1. variance decomposition: latent variance -> SUBJECT vs STIMULUS(movie-timepoint) vs residual
     (+ subject/stimulus ratio and subject-vs-permutation-null).
  2. subject fingerprint BA (as rest, for comparison).
  3. STIMULUS-SURVIVAL: cross-subject movie-timepoint decoding (leave-subjects-out) BEFORE vs AFTER
     LEACE subject-axis erasure. Survival = how much of the shared-stimulus signal is separable
     from subject identity (the causal identity-trap test). Erasure should also drop subject BA
     to ~chance (sanity).

Pure numpy/sklearn/fmscope. Reads per-window latents from hbn_window_movieDM/{model}/*.npz.
Windows aligned across subjects by truncating to a common count W (drops subjects with too few).

    python benchmarks/hbn_fmscope_movie_audit.py --models cmae neurostorm swift
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/mhough/dev/fmscope")
WROOT = Path("/data/derivatives/peer_fm_ww/hbn_window_movieDM")
OUT = Path("/home/mhough/dev/hippy-feat/results/fmscope"); OUT.mkdir(parents=True, exist_ok=True)
MIN_WIN = 20   # drop subjects with fewer usable windows


def load_aligned(model):
    """Return T (n_subj, W, D) with window k time-aligned across subjects, + subs list.
    W = min window count among kept subjects (>= MIN_WIN); each subject truncated to first W."""
    d = WROOT / model
    per = {}
    for f in sorted(d.glob("*.npz")):
        w = np.load(f)["win"]
        if w.ndim == 2 and w.shape[0] >= MIN_WIN:
            per[f.stem] = w
    if not per:
        return None, []
    W = min(v.shape[0] for v in per.values())
    subs = sorted(per)
    T = np.stack([per[s][:W] for s in subs])         # (n_subj, W, D)
    return T, subs


def variance_decomp(T):
    """Two-way (subject × timepoint) SS decomposition of the latent. Returns fractions + ratio."""
    n, W, D = T.shape
    grand = T.mean((0, 1))
    subj_eff = T.mean(1) - grand                     # (n, D)
    time_eff = T.mean(0) - grand                     # (W, D)
    resid = T - grand - subj_eff[:, None, :] - time_eff[None, :, :]
    ss_subj = W * np.sum(subj_eff ** 2)
    ss_time = n * np.sum(time_eff ** 2)
    ss_res = np.sum(resid ** 2)
    tot = ss_subj + ss_time + ss_res
    # permutation null for subject SS: shuffle subject assignment of each (window) row
    flat = T.reshape(n * W, D); sid = np.repeat(np.arange(n), W)
    rng = np.random.RandomState(0); nulls = []
    gm = flat.mean(0)
    for _ in range(50):
        p = rng.permutation(sid)
        se = np.stack([flat[p == s].mean(0) for s in range(n)]) - gm
        nulls.append(W * np.sum(se ** 2))
    return {"frac_subject": ss_subj / tot, "frac_stimulus": ss_time / tot, "frac_residual": ss_res / tot,
            "subject_over_stimulus": ss_subj / ss_time if ss_time > 0 else float("nan"),
            "subject_over_null": ss_subj / float(np.mean(nulls))}


def stimulus_survival(T):
    """Cross-subject movie-timepoint decoding (leave-subjects-out) pre/post subject-axis erasure."""
    from fmscope.diagnostics.erasure import whiten, subject_eraser, apply_eraser, subject_probe
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    n, W, D = T.shape
    X = T.reshape(n * W, D)
    tp = np.tile(np.arange(W), n)                    # movie-timepoint label
    sid = np.repeat(np.arange(n), W)                 # subject label
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-6)
    # subject-axis eraser fit on all windows
    mu, Xc, Wh, Wp, _ = whiten(Xz, shrinkage=True)
    _, P_perp, rank = subject_eraser(Xc, Wh, Wp, sid)
    Xe = apply_eraser(Xz, mu, P_perp)

    def tp_decode(feats):
        gkf = GroupKFold(n_splits=5); bas = []
        for tr, te in gkf.split(feats, tp, groups=sid):
            sc = StandardScaler().fit(feats[tr])
            clf = LogisticRegression(max_iter=300, C=1.0)
            clf.fit(sc.transform(feats[tr]), tp[tr])
            bas.append(balanced_accuracy_score(tp[te], clf.predict(sc.transform(feats[te]))))
        return float(np.mean(bas))

    pre, post = tp_decode(Xz), tp_decode(Xe)
    subj_pre, _ = subject_probe(Xz, sid, kind="linear", cap=100, n_splits=5)
    subj_post, _ = subject_probe(Xe, sid, kind="linear", cap=100, n_splits=5)
    return {"tp_decode_pre": pre, "tp_decode_post": post, "tp_survival": post / pre if pre > 0 else float("nan"),
            "tp_chance": 1.0 / W, "subj_ba_pre": subj_pre, "subj_ba_post": subj_post,
            "subject_subspace_rank": int(rank), "n_timepoints": W}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["cmae", "neurostorm", "swift"])
    a = ap.parse_args()
    res = []
    for m in a.models:
        T, subs = load_aligned(m)
        if T is None:
            print(f"[{m}] no aligned windows — skip"); continue
        n, W, D = T.shape
        print(f"\n=== {m}  ({n} subjects × {W} movie-timepoints, dim={D}) ===", flush=True)
        vd = variance_decomp(T)
        ss = stimulus_survival(T)
        print(f"  variance: subject={vd['frac_subject']:.2f}  stimulus={vd['frac_stimulus']:.2f}  "
              f"residual={vd['frac_residual']:.2f}  (subj/stim={vd['subject_over_stimulus']:.1f}, "
              f"subj/null={vd['subject_over_null']:.1f}x)", flush=True)
        print(f"  movie-timepoint decode: pre={ss['tp_decode_pre']:.3f} -> post-erase={ss['tp_decode_post']:.3f} "
              f"(survival {ss['tp_survival']*100:.0f}%, chance {ss['tp_chance']:.3f})", flush=True)
        print(f"  subject BA: pre={ss['subj_ba_pre']:.3f} -> post-erase={ss['subj_ba_post']:.3f} "
              f"(rank {ss['subject_subspace_rank']}/{D})", flush=True)
        res.append({"model": m, "n_subjects": n, "n_timepoints": W, "dim": D, **vd, **ss})
    (OUT / "movie_fmscope_audit.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[saved] {OUT/'movie_fmscope_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
