# Cross-project comparison harness. Loads weight matrices from each project's
# model definition + (where available) trained checkpoints, runs both the
# WeightWatcher fork (Python oracle) and wwj on the same matrices, and prints
# side-by-side alpha + log_norm + stable_rank tables.
#
# Projects covered:
#   nanopath              -- real trained ckpts on disk from the leaderboard chase
#   smri-fm               -- model def only; random init for spectral comparison
#   eeg-fm-spectral       -- model def only; random init for spectral comparison
#   hippy-feat            -- model def only; random init for spectral comparison
#   realtime-mindeye      -- not on disk; skipped (see README)
#
# Usage:
#   cd /home/mhough/dev/wwj
#   uv run python benchmarks/compare_projects.py
#   uv run python benchmarks/compare_projects.py --project nanopath
#   uv run python benchmarks/compare_projects.py --ckpt /path/to/latest.pt

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import wwj


def _extract_matrices_from_torch_state_dict(sd: dict, min_dim: int = 50, max_layers: int = 32):
    """Pull 2D weight tensors out of a PyTorch state_dict. Returns [(name, np.ndarray), ...]."""
    out = []
    for name, t in sd.items():
        if t.ndim != 2: continue
        if min(t.shape) < min_dim: continue
        out.append((name, t.detach().cpu().numpy().astype(np.float32)))
        if len(out) >= max_layers: break
    return out


def _matrices_from_nanopath_ckpt(ckpt_path: Path):
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # Could be the training ckpt (has "model") or a raw state_dict
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return _extract_matrices_from_torch_state_dict(sd)


def _matrices_from_random_model(project: str):
    """Instantiate a project's model at random init and extract 2D matrices.
    Returns [] if the project's model isn't importable from this environment."""
    if project == "smri-fm":
        # smri-fm exports a foundation model module; try the most common pattern
        sys.path.insert(0, "/home/mhough/dev/smri-fm")
        try:
            import smri_fm  # noqa: F401
            # The repo likely has a builder; for MVP we just walk top-level torch.nn modules
            return []  # TODO: project author identifies the canonical builder
        finally:
            sys.path.pop(0)
    if project == "eeg-fm-spectral":
        sys.path.insert(0, "/home/mhough/dev/eeg-fm-spectral")
        try:
            from eeg_fm_spectral import eeg_fm  # noqa: F401
            return []  # TODO: project author identifies the canonical builder
        finally:
            sys.path.pop(0)
    if project == "hippy-feat":
        sys.path.insert(0, "/home/mhough/dev/hippy-feat")
        # TODO: import + instantiate; the FM module name is project-specific
        return []
    return []


def _oracle_alpha_for_matrix(W: np.ndarray) -> float:
    import weightwatcher as ww
    import torch
    import torch.nn as nn
    n, m = W.shape
    layer = nn.Linear(m, n, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(W))
    return float(ww.WeightWatcher(model=layer).analyze(min_evals=50)["alpha"].iloc[0])


def compare(matrices: list[tuple[str, np.ndarray]], label: str):
    print(f"\n=== {label} ({len(matrices)} matrices) ===")
    if not matrices:
        print("  (no matrices found / model builder not wired up)")
        return
    print(f"{'name':40s}  {'shape':>14s}  {'wwj α':>7s}  {'oracle α':>9s}  {'|Δα|':>6s}  {'wwj logN':>9s}")
    diffs = []
    for name, W in matrices:
        wwj_stats = wwj.analyze_matrix(jnp.asarray(W), name=name, mode="csn")
        wwj_alpha = float(wwj_stats.alpha)
        try:
            oracle_alpha = _oracle_alpha_for_matrix(W)
        except Exception as e:
            oracle_alpha = float("nan")
        d = abs(wwj_alpha - oracle_alpha)
        diffs.append(d)
        print(f"{name[:40]:40s}  {str(W.shape):>14s}  {wwj_alpha:7.3f}  {oracle_alpha:9.3f}  {d:6.3f}  {float(wwj_stats.log_norm):9.3f}")
    finite = [d for d in diffs if np.isfinite(d)]
    if finite:
        print(f"\n  agreement: mean |Δα| = {np.mean(finite):.3f}  median = {np.median(finite):.3f}  max = {np.max(finite):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", choices=["nanopath", "smri-fm", "eeg-fm-spectral", "hippy-feat", "all"], default="all")
    ap.add_argument("--ckpt", type=Path, default=None, help="override ckpt path for nanopath")
    args = ap.parse_args()

    if args.project in ("nanopath", "all"):
        ckpt = args.ckpt or Path("/data/datasets/nanopath/runs/main_run1/latest.pt")
        if ckpt.is_file():
            compare(_matrices_from_nanopath_ckpt(ckpt), f"nanopath (trained: {ckpt.name})")
        else:
            print(f"nanopath: ckpt not found at {ckpt}; skipping")

    for proj in ("smri-fm", "eeg-fm-spectral", "hippy-feat"):
        if args.project in (proj, "all"):
            compare(_matrices_from_random_model(proj), f"{proj} (random init; builder stub)")

    print("\nNote: realtime-mindeye (= fmri-fm) not on disk; not benchmarked here.")


if __name__ == "__main__":
    main()
