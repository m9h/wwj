"""fMRI FM-latent cross-run FINGERPRINT on HBN (rest_run-1 vs rest_run-2).

The fMRI analog of EEG cross-session permanence, on FOUNDATION-MODEL latents (novel — connectome
fingerprinting, Finn 2015, is on raw FC; nobody's fingerprinted an fMRI FM latent cross-run).

Per model (CortexMAE/NeuroSTORM/SwiFT), using one mean-pooled latent per (subject, run):
  - RE-IDENTIFICATION accuracy (Finn 2015 method): for each subject's run-1 latent, the predicted
    identity is the subject whose run-2 latent it correlates most with; accuracy over all subjects,
    both directions (r1->r2, r2->r1). Chance = 1/n_subjects. Compare to Finn's ~92-94% on raw FC.
  - I_diff (Amico & Goñi 2018 differential identifiability): Iself = mean within-subject run1-run2
    correlation (identifiability-matrix diagonal); Iothers = mean between-subject; I_diff =
    (Iself - Iothers) x 100.

Pure numpy. Reads hbn_2run/emb/{sub}.npz (keys {model}_r1/{model}_r2).

    python benchmarks/cross_run_fingerprint.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

EMB = Path("/data/derivatives/peer_fm_ww/hbn_2run/emb")
OUT = Path("/home/mhough/dev/hippy-feat/results/fmscope"); OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["cmae", "neurostorm", "swift"]


def _norm(M):
    """row-wise zero-mean unit-norm so M @ M.T = Pearson correlation between vectors."""
    c = M - M.mean(1, keepdims=True)
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)


def audit(model, subs, d):
    R1 = _norm(np.vstack([d[s][f"{model}_r1"] for s in subs]))   # (n, D)
    R2 = _norm(np.vstack([d[s][f"{model}_r2"] for s in subs]))
    S = R1 @ R2.T                                                # (n, n) run1 x run2 correlation
    n = len(subs)
    acc_12 = float(np.mean(S.argmax(1) == np.arange(n)))         # enroll r1, identify in r2
    acc_21 = float(np.mean(S.argmax(0) == np.arange(n)))         # enroll r2, identify in r1
    iself = float(np.mean(np.diag(S)))
    iothers = float((S.sum() - np.trace(S)) / (n * n - n))
    idiff = (iself - iothers) * 100.0
    return {"model": model, "n_subjects": n, "dim": int(R1.shape[1]),
            "reid_acc_r1r2": acc_12, "reid_acc_r2r1": acc_21, "reid_acc_mean": (acc_12 + acc_21) / 2,
            "chance": 1.0 / n, "I_self": iself, "I_others": iothers, "I_diff": idiff}


def main():
    files = sorted(EMB.glob("*.npz"))
    d = {}
    for f in files:
        z = np.load(f)
        if all(f"{m}_r1" in z and f"{m}_r2" in z for m in MODELS):
            d[f.stem] = {k: z[k] for k in z.files}
    subs = sorted(d)
    print(f">>> cross-run fingerprint: {len(subs)} subjects with all 6 latents", flush=True)
    res = []
    for m in MODELS:
        r = audit(m, subs, d)
        res.append(r)
        print(f"  {m:11s} re-ID {r['reid_acc_mean']*100:5.1f}% (chance {r['chance']*100:.2f}%, "
              f"{r['reid_acc_mean']/r['chance']:.0f}x) | I_diff {r['I_diff']:.1f} "
              f"(I_self {r['I_self']:.3f} vs I_others {r['I_others']:.3f})", flush=True)
    (OUT / "cross_run_fingerprint.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[saved] {OUT/'cross_run_fingerprint.json'}", flush=True)


if __name__ == "__main__":
    main()
