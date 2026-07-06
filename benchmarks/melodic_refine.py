"""MELODIC refinements: (1) principled MIGP dim via Gavish-Donoho; (2) CC200 raw-FC cross-run
fingerprint (Finn baseline on HBN) as the robustness column vs the FM-latent fingerprint."""
import numpy as np, sys
sys.path.insert(0, "/home/mhough/Workspace/smni-cmi/src")
from smni_cmi.clean import gavish_donoho_rank

# (1) Gavish-Donoho principled dim from the MIGP eigenspectrum (relative SVs from cumulative var%)
cum = np.array([float(x) for x in open("/mnt/t9/melodic_fc/prod200.gica/eigenvalues_percent").read().split()])
per = np.diff(np.concatenate([[0.0], cum]))            # per-component variance %
sv = np.sqrt(np.clip(per, 0, None))                    # relative singular values
gd = gavish_donoho_rank(sv, m=len(sv), n=228483)       # m=MIGP eigenmaps, n=voxels
print(f"[GD] MIGP spectrum {len(sv)} comps -> Gavish-Donoho signal rank = {gd}  (pilot used dim=50)")
print(f"     var explained by top-50: {cum[49]:.1f}%  | by top-{gd}: {cum[min(gd,len(cum))-1]:.1f}%")

# (2) CC200 raw-FC cross-run fingerprint (Finn 2015 method) — robustness baseline
d = np.load("/mnt/t9/hbn_cc200_fc.npz")
r1, r2, nruns = d["fc_run1"], d["fc_run2"], d["nruns"]
valid = (nruns >= 2) & np.isfinite(r1).all(1) & np.isfinite(r2).all(1) & (r1.std(1) > 0) & (r2.std(1) > 0)
r1, r2 = r1[valid], r2[valid]; n = len(r1)
def _norm(M): c = M - M.mean(1, keepdims=True); return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
S = _norm(r1) @ _norm(r2).T
acc = (np.mean(S.argmax(1) == np.arange(n)) + np.mean(S.argmax(0) == np.arange(n))) / 2
idiff = (np.mean(np.diag(S)) - (S.sum() - np.trace(S)) / (n * n - n)) * 100
print(f"[CC200] raw-FC cross-run re-ID: {acc*100:.1f}%  (n={n}, chance {100/n:.2f}%, {acc*n:.0f}x)  I_diff {idiff:.1f}")
print(f"        vs FM-latent cross-run (CortexMAE 65.9% / NeuroSTORM 60.2% / SwiFT 5.2%) + Finn~92-94%")
np.savez("/mnt/t9/melodic_fc/refinements.npz", gd_rank=gd, migp_dim=len(sv),
         cc200_reid=acc, cc200_idiff=idiff, cc200_n=n)
print("[saved] /mnt/t9/melodic_fc/refinements.npz")
