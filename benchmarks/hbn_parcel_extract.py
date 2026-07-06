"""Faithful Schaefer-based parcel timeseries for HBN CPAC BOLD (replaces the wrong CC200).

The CortexMAE *parcel* models want a Schaefer-400+subcortical (~450-457 ROI) parcellation;
local HBN roi_timeseries was CC200 (wrong atlas + count). AtlasPack is fully local, so we
parcellate the CPAC MNI BOLD directly with the 4S456 atlas (Schaefer-400 cortical + 56
subcortical/cerebellar = 456 parcels).

Grid match (verified): CPAC `bandpassed_demeaned_filtered_antswarp.nii.gz` is the FSL
MNI152NLin6Asym 2mm grid (91x109x91), so we use the NLin6Asym 4S456 (1mm) resampled
nearest-neighbor to the BOLD grid. The resampled label volume is grid-invariant across
subjects (all share the standard grid) -> compute once, cache, reuse.

CAVEAT: this is a *volume* extraction of a surface-derived Schaefer parcellation; boundaries
differ slightly from the model's training (CIFTI/fsLR) parcellation. Faithful in label set +
count (the CC200 problem), approximate at cortical boundaries. The fsLR `den-91k` dlabel is
also local but needs HBN BOLD on the fsLR surface (CIFTI), which we don't have for HBN.

    from hbn_parcel_extract import parcel_timeseries
    ts = parcel_timeseries("/data/raw/hbn-cpac/.../bandpassed..._antswarp.nii.gz")  # (T, 456)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib

# NLin6 (exact match to the CPAC FSL grid) content isn't fetched in the local annex; the
# 2009c labels ARE present. resample_from_to aligns by world-affine, so 2009c labels on the
# NLin6 BOLD grid carry a sub-cm NLin6<->2009c boundary offset -> acceptable for a parcel
# robustness column. `git annex get` the NLin6 res-01 dseg later for the exact-space version.
ATLAS_NLIN6 = "/data/derivatives/atlases/AtlasPack/tpl-MNI152NLin6Asym_atlas-4S456Parcels_res-01_dseg.nii.gz"
ATLAS_2009C = "/data/derivatives/atlases/AtlasPack/tpl-MNI152NLin2009cAsym_atlas-4S456Parcels_res-01_dseg.nii.gz"
ATLAS = ATLAS_NLIN6 if Path(ATLAS_NLIN6).exists() and Path(ATLAS_NLIN6).stat().st_size > 1000 else ATLAS_2009C
ATLAS_TSV = "/data/derivatives/atlases/AtlasPack/atlas-4S456Parcels_dseg.tsv"
N_PARCELS = 456
_CACHE = Path("/data/derivatives/peer_fm_ww/hbn_full/atlas_4S456_2mm.npz")

# canonical FSL MNI152 2mm grid that CPAC antswarp targets
_BOLD_SHAPE = (91, 109, 91)
_BOLD_AFFINE = np.array([[-2., 0., 0., 90.], [0., 2., 0., -126.],
                         [0., 0., 2., -72.], [0., 0., 0., 1.]])


def _projector():
    """Sparse parcel-mean projector P (N_PARCELS x V_brain) + brain voxel index, on the 2mm
    BOLD grid. Resample 1mm 4S456 -> 2mm (NN) once, build the row-normalized one-hot. Cached
    so per-subject extraction is a single sparse matmul (fast), not a 456-way Python loop."""
    import scipy.sparse as sp
    if _CACHE.exists():
        d = np.load(_CACHE)
        P = sp.csr_matrix((d["P_data"], d["P_indices"], d["P_indptr"]), shape=tuple(d["P_shape"]))
        return P, d["brain_idx"]
    from nibabel.processing import resample_from_to
    atlas = nib.load(ATLAS)
    target = nib.Nifti1Image(np.zeros(_BOLD_SHAPE, np.int16), _BOLD_AFFINE)
    lab = np.asarray(resample_from_to(atlas, target, order=0).dataobj).astype(np.int32).reshape(-1)
    brain_idx = np.flatnonzero(lab > 0)                # (V_brain,) into the flattened (X,Y,Z)
    lv = lab[brain_idx] - 1                            # 0..N-1
    counts = np.bincount(lv, minlength=N_PARCELS).astype(np.float64)
    counts[counts == 0] = 1.0
    # P[p, v] = 1/counts[p] if brain-voxel v belongs to parcel p
    P = sp.csr_matrix((1.0 / counts[lv], (lv, np.arange(brain_idx.size))),
                      shape=(N_PARCELS, brain_idx.size))
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_CACHE, P_data=P.data, P_indices=P.indices, P_indptr=P.indptr,
                        P_shape=np.array(P.shape), brain_idx=brain_idx)
    return P, brain_idx


def parcel_timeseries(bold_path: str, proj=None) -> np.ndarray:
    """(T, 456) parcel-mean timeseries from a CPAC MNI BOLD nii. Single sparse matmul."""
    if proj is None:
        proj = _projector()
    P, brain_idx = proj
    img = nib.as_closest_canonical(nib.load(bold_path))
    if img.shape[:3] != _BOLD_SHAPE:
        raise ValueError(f"unexpected BOLD grid {img.shape[:3]} != {_BOLD_SHAPE}")
    data = np.asarray(img.dataobj, dtype=np.float32)            # (X,Y,Z,T)
    flat = data.reshape(-1, data.shape[-1])[brain_idx]          # (V_brain, T) — matches P cols
    return (P @ flat).astype(np.float32).T                      # (T, N_PARCELS)


if __name__ == "__main__":
    import csv, time
    t0 = time.time()
    proj = _projector()
    P, brain_idx = proj
    print(f"projector: {P.shape[0]} parcels x {brain_idx.size} brain voxels "
          f"({(P.getnnz(axis=1) > 0).sum()} non-empty parcels)  [{time.time()-t0:.1f}s]")
    row = next(csv.DictReader(open("/data/derivatives/peer_fm_ww/hbn1228_manifest.csv")))
    t1 = time.time()
    ts = parcel_timeseries(row["nii"], proj)
    nz = (ts.std(0) > 0).sum()
    print(f"sub {row['sub']}: parcel ts {ts.shape}, {nz}/{N_PARCELS} non-empty, "
          f"mean|val|={np.abs(ts).mean():.3f}  [extract {time.time()-t1:.1f}s]")
