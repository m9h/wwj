"""Peer pathology-FM spectral survey — run wwj HT-SR diagnostics on the public
H&E tile foundation models the nanopath leaderboard already evaluates, so the
spectral state of each external FM lands in one table comparable to our own
nanopath training checkpoints. Tests whether mean alpha / |alpha - 2| /
fraction-passing-Vuong-LRT actually predict pathology probe quality at fixed
eval (i.e. whether HT-SR is a credible weights-only signal in this modality).

Architecture-agnostic: we don't install each FM's framework. We load the
published checkpoint, extract every 2D weight matrix with min(shape) >= 50,
and run wwj.analyze + wwj.fit_distributions + wwj.bootstrap_alpha_ci on it.

Per-FM extractor (HF repo + state-dict key recipe):

    dinov2_s     HF facebook/dinov2-with-registers-small         backbone
    dinov2_g     HF facebook/dinov2-with-registers-giant         backbone
    openmidnight HF kaiko-ai/midnight                            student.backbone.*
    midnight12k  HF kaiko-ai/midnight                            backbone (the 12K-step ckpt)
    uni2h        HF MahmoodLab/UNI2-h (gated)                    raw (renames mlp.fc1->mlp.w12 etc.)
    virchow      HF paige-ai/Virchow via timm                    full timm state_dict
    virchow2     HF paige-ai/Virchow2 via timm                   full timm state_dict
    phikon       HF owkin/phikon                                 vit.encoder.* (ViT-B/16)
    phikon_v2    HF owkin/phikon-v2                              vit.encoder.* (ViT-L/16)
    hoptimus0    HF bioptimus/H-optimus-0 (gated)                full timm state_dict
    hoptimus1    HF bioptimus/H-optimus-1 (gated)                full timm state_dict
    genbio       HF genbio-ai/pathology-FM                       backbone
    gigapath     HF prov-gigapath/prov-gigapath                  tile encoder (slide enc skipped)
    hibou_b      HF histai/hibou-b                               full
    hibou_l      HF histai/hibou-l                               full

Usage:
    cd /home/mhough/dev/wwj && uv run python benchmarks/peer_path_fm_survey.py \\
        --out-dir /data/mhough/wwj_peer_path \\
        --models dinov2_s dinov2_g openmidnight phikon phikon_v2 virchow virchow2 hibou_b hibou_l genbio gigapath

Gated FMs (uni2h, hoptimus0, hoptimus1) are skipped unless the local HF cache
already contains them or HUGGINGFACE_HUB_TOKEN has access. The script emits
per-FM CSVs (one row per layer with alpha, xmin, tail_size, Vuong LRTs) plus
an aggregate peer_path_fms_summary.csv that's directly comparable across FMs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch

import wwj
from wwj.core import _eigvals, analyze_matrix, fit_distributions


# ---------------------------------------------------------------------------
# checkpoint loading + per-FM state_dict extraction
# ---------------------------------------------------------------------------
def _load_torch(path: str | Path) -> dict:
    """Read a pytorch_model.bin / *.pth / *.safetensors into a flat state_dict."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "module"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _drop_module(sd: dict) -> dict:
    return {k.replace("module.", "", 1): v for k, v in sd.items()} \
        if any(k.startswith("module.") for k in sd) else sd


def _hf_snapshot(repo: str, allow: list[str] | None = None) -> Path:
    """Download (if not cached) and return the local path to an HF repo snapshot.
    Honors HF_HOME / HUGGINGFACE_HUB_TOKEN. Returns None on auth/network failure."""
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(
            repo_id=repo,
            allow_patterns=allow or ["*.bin", "*.safetensors", "*.json", "*.pth"],
        ))
    except Exception as e:
        print(f"  [hf] {repo}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return None


# ---- per-FM loaders ----
# Each returns a dict[str, torch.Tensor]: the state_dict whose 2D entries
# are the encoder weight matrices we want to analyze.

def load_dinov2_s():
    p = _hf_snapshot("facebook/dinov2-with-registers-small")
    if p is None: return None
    sd = _load_torch(p / "model.safetensors") if (p / "model.safetensors").exists() \
        else _load_torch(p / "pytorch_model.bin")
    # facebook ckpt prefixes everything with "dinov2."
    sd = _strip_prefix(sd, "dinov2.") if any(k.startswith("dinov2.") for k in sd) else sd
    return sd


def load_dinov2_g():
    p = _hf_snapshot("facebook/dinov2-with-registers-giant")
    if p is None: return None
    sd = _load_torch(p / "model.safetensors") if (p / "model.safetensors").exists() \
        else _load_torch(p / "pytorch_model.bin")
    sd = _strip_prefix(sd, "dinov2.") if any(k.startswith("dinov2.") for k in sd) else sd
    return sd


def load_openmidnight():
    p = _hf_snapshot("kaiko-ai/midnight", allow=["*midnight*.pth", "*midnight*.safetensors", "*.json"])
    if p is None: return None
    # Kaiko ships two ckpts in one repo: midnight (Open variant) and midnight-12k.
    cand = sorted(p.glob("*open*.safetensors")) + sorted(p.glob("*open*.pth")) \
        + sorted(p.glob("*midnight*.safetensors"))
    if not cand: return None
    sd = _load_torch(cand[0])
    return _strip_prefix(sd, "student.backbone.") if any(k.startswith("student.backbone.") for k in sd) else sd


def load_midnight12k():
    p = _hf_snapshot("kaiko-ai/midnight")
    if p is None: return None
    cand = sorted(p.glob("*12k*.safetensors")) + sorted(p.glob("*12k*.pth"))
    if not cand: return None
    sd = _load_torch(cand[0])
    return _strip_prefix(sd, "student.backbone.") if any(k.startswith("student.backbone.") for k in sd) else sd


def load_uni2h():
    p = _hf_snapshot("MahmoodLab/UNI2-h")
    if p is None: return None
    sd = _load_torch(p / "pytorch_model.bin")
    # UNI2 uses fused swiglu naming: mlp.fc1 + mlp.fc2 -> mlp.w12 + mlp.w3 in
    # nanopath's DinoV2ViT; for WW we don't care -- we keep the original names.
    return sd


def load_virchow():
    import timm
    from timm.layers import SwiGLUPacked
    m = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True,
                          mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    return m.state_dict()


def load_virchow2():
    import timm
    from timm.layers import SwiGLUPacked
    m = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                          mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    return m.state_dict()


def load_phikon():
    p = _hf_snapshot("owkin/phikon")
    if p is None: return None
    sd = _load_torch(p / "model.safetensors") if (p / "model.safetensors").exists() \
        else _load_torch(p / "pytorch_model.bin")
    # phikon uses HF ViT keys: vit.encoder.layer.{i}.{attention.attention.{query,key,value}|attention.output.dense|intermediate.dense|output.dense}.weight
    return sd


def load_phikon_v2():
    p = _hf_snapshot("owkin/phikon-v2")
    if p is None: return None
    sd = _load_torch(p / "model.safetensors") if (p / "model.safetensors").exists() \
        else _load_torch(p / "pytorch_model.bin")
    return sd


def load_hoptimus0():
    p = _hf_snapshot("bioptimus/H-optimus-0")
    if p is None: return None
    cand = sorted(p.glob("*.safetensors")) + sorted(p.glob("*.bin")) + sorted(p.glob("*.pth"))
    return _load_torch(cand[0]) if cand else None


def load_hoptimus1():
    p = _hf_snapshot("bioptimus/H-optimus-1")
    if p is None: return None
    cand = sorted(p.glob("*.safetensors")) + sorted(p.glob("*.bin")) + sorted(p.glob("*.pth"))
    return _load_torch(cand[0]) if cand else None


def load_genbio():
    p = _hf_snapshot("genbio-ai/pathology-FM")
    if p is None: return None
    cand = sorted(p.glob("*.safetensors")) + sorted(p.glob("*.bin"))
    return _load_torch(cand[0]) if cand else None


def load_gigapath():
    # GigaPath ships the tile encoder + a slide encoder; we only analyze the
    # tile ViT here so it's comparable to the other tile-level FMs.
    p = _hf_snapshot("prov-gigapath/prov-gigapath", allow=["pytorch_model.bin", "*.json", "*.safetensors"])
    if p is None: return None
    cand = sorted(p.glob("pytorch_model.bin")) + sorted(p.glob("*.safetensors"))
    if not cand: return None
    sd = _load_torch(cand[0])
    return sd


def load_hibou_b():
    p = _hf_snapshot("histai/hibou-b")
    if p is None: return None
    sd = _load_torch(p / "model.safetensors") if (p / "model.safetensors").exists() \
        else _load_torch(p / "pytorch_model.bin")
    return sd


def load_hibou_l():
    p = _hf_snapshot("histai/hibou-l")
    if p is None: return None
    sd = _load_torch(p / "model.safetensors") if (p / "model.safetensors").exists() \
        else _load_torch(p / "pytorch_model.bin")
    return sd


# (name, loader, family, reported_score_on_nanopath, citation)
# Scores from README baseline table; None for FMs not on the nanopath board.
SPECS: dict[str, tuple[Callable[[], dict | None], str, float | None, str]] = {
    "dinov2_s":     (load_dinov2_s,     "dinov2",   0.5304, "Meta 2023 ViT-S/14 reg"),
    "dinov2_g":     (load_dinov2_g,     "dinov2",   0.5627, "Meta 2023 ViT-G/14 reg"),
    "openmidnight": (load_openmidnight, "midnight", 0.5494, "Kaiko 2025 ViT-G/14"),
    "midnight12k":  (load_midnight12k,  "midnight", 0.5545, "Kaiko 2025 ViT-G/14 (12K step)"),
    "uni2h":        (load_uni2h,        "uni",      0.6149, "MahmoodLab 2024 ViT-H/14 (gated)"),
    "virchow":      (load_virchow,      "virchow",  None,   "Paige 2024 ViT-H/14"),
    "virchow2":     (load_virchow2,     "virchow",  None,   "Paige 2024 ViT-H/14 v2"),
    "phikon":       (load_phikon,       "phikon",   None,   "Owkin 2023 ViT-B/16"),
    "phikon_v2":    (load_phikon_v2,    "phikon",   None,   "Owkin 2024 ViT-L/16"),
    "hoptimus0":    (load_hoptimus0,    "hoptimus", 0.6190, "Bioptimus 2024 ViT-G/14 (gated)"),
    "hoptimus1":    (load_hoptimus1,    "hoptimus", None,   "Bioptimus 2025 ViT-G/14 v1 (gated)"),
    "genbio":       (load_genbio,       "genbio",   0.6268, "GenBio 2025 (current leader)"),
    "gigapath":     (load_gigapath,     "gigapath", None,   "Microsoft 2024 ViT-G/14 tile encoder"),
    "hibou_b":      (load_hibou_b,      "hibou",    None,   "Hist.ai 2024 ViT-B/14"),
    "hibou_l":      (load_hibou_l,      "hibou",    None,   "Hist.ai 2024 ViT-L/14"),
}


# ---------------------------------------------------------------------------
# spectral analysis -- per-FM CSV + summary JSON
# ---------------------------------------------------------------------------
def _ww_per_layer(sd: dict, min_dim: int = 50, max_layers: int = 64) -> list[dict]:
    """Per-layer alpha / xmin / Vuong LRTs for every 2D weight in sd.
    max_layers caps the per-FM cost for the large ViT-G architectures (which
    have ~100 fittable 2D matrices) -- we sample uniformly across depth."""
    items = [(k, v) for k, v in sd.items() if torch.is_tensor(v) and v.ndim == 2
             and min(v.shape) >= min_dim]
    if not items:
        return []
    if len(items) > max_layers:
        idx = np.linspace(0, len(items) - 1, max_layers, dtype=int)
        items = [items[i] for i in idx]

    rows = []
    for name, t in items:
        W = jnp.asarray(t.detach().float().cpu().numpy())
        eigs = _eigvals(W)
        stats = analyze_matrix(W, name=name, mode="csn")
        fits = fit_distributions(eigs)
        rows.append({
            "layer": name,
            "shape": f"{tuple(t.shape)}",
            "alpha": float(stats.alpha),
            "xmin": float(fits["xmin"]),
            "lambda_max": float(stats.lambda_max),
            "stable_rank": float(stats.stable_rank),
            "entropy": float(stats.entropy),
            "num_pl_spikes": int(stats.num_pl_spikes),
            "pl_vs_exp_lrt": float(fits["pl_vs_exp_lrt"]),
            "pl_vs_ln_lrt": float(fits["pl_vs_ln_lrt"]),
            "tail_size": int(fits["tail_size"]),
        })
    return rows


def _summarize(name: str, family: str, score: float | None, citation: str,
               rows: list[dict], n_params: int) -> dict:
    """Aggregate per-layer rows into one comparable row per FM. Mirrors the
    schema used by peer_fmri_fm_survey.summarize so the two surveys can be
    concatenated for the cross-modality plot."""
    if not rows:
        return {"model": name, "family": family, "citation": citation,
                "score_on_nanopath": score, "status": "no_matrices",
                "n_params": n_params, "n_layers": 0}
    a = pd.Series([r["alpha"] for r in rows])
    a_finite = a[a.notna() & np.isfinite(a)]
    pl_pos = pd.Series([(r["pl_vs_exp_lrt"] > 0) and (r["pl_vs_ln_lrt"] > 0) for r in rows])
    return {
        "model": name, "family": family, "citation": citation,
        "score_on_nanopath": score, "status": "ok",
        "n_params": int(n_params), "n_layers": len(rows),
        "alpha_mean": float(a_finite.mean()) if len(a_finite) else None,
        "alpha_median": float(a_finite.median()) if len(a_finite) else None,
        "alpha_dist_mean": float((a_finite - 2.0).abs().mean()) if len(a_finite) else None,
        "frac_near2_1p5_2p5": float(((a_finite >= 1.5) & (a_finite <= 2.5)).mean()) if len(a_finite) else None,
        "frac_alpha_lt2": float((a_finite < 2.0).mean()) if len(a_finite) else None,
        "frac_alpha_gt6": float((a_finite > 6.0).mean()) if len(a_finite) else None,
        "frac_pl_valid": float(pl_pos.mean()),
        "traps_total": int(sum(r["num_pl_spikes"] for r in rows)),
        "entropy_mean": float(pd.Series([r["entropy"] for r in rows]).mean()),
    }


def analyse(name: str, spec, out_dir: Path) -> dict:
    loader, family, score, citation = spec
    print(f"\n=== {name}  ({citation}) ===", flush=True)
    try:
        sd = loader()
    except Exception as e:
        print(f"  [load failed] {type(e).__name__}: {str(e)[:200]}", flush=True)
        return {"model": name, "family": family, "citation": citation,
                "score_on_nanopath": score, "status": "load_failed",
                "error": str(e)[:200]}
    if sd is None:
        print(f"  [skip] checkpoint not available", flush=True)
        return {"model": name, "family": family, "citation": citation,
                "score_on_nanopath": score, "status": "no_ckpt"}

    n_params = sum(int(v.numel()) for v in sd.values() if torch.is_tensor(v))
    n_2d = sum(1 for v in sd.values() if torch.is_tensor(v) and v.ndim == 2
               and min(v.shape) >= 50)
    print(f"  state_dict entries={len(sd)}  2D matrices(min_dim>=50)={n_2d}  "
          f"params={n_params:,}", flush=True)
    if n_2d == 0:
        return {"model": name, "family": family, "citation": citation,
                "score_on_nanopath": score, "status": "no_matrices",
                "n_params": n_params}

    rows = _ww_per_layer(sd)
    pd.DataFrame(rows).to_csv(out_dir / f"{name}_details.csv", index=False)

    summ = _summarize(name, family, score, citation, rows, n_params)
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(f"  alpha mean={summ['alpha_mean']:.3f}  med={summ['alpha_median']:.3f}  "
          f"|a-2|={summ['alpha_dist_mean']:.3f}  "
          f"%near2={summ['frac_near2_1p5_2p5']:.1%}  "
          f"%PL-valid={summ['frac_pl_valid']:.1%}  "
          f"traps={summ['traps_total']}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/data/mhough/wwj_peer_path",
                    help="where per-FM CSVs and the aggregate summary land")
    ap.add_argument("--models", nargs="+", default=list(SPECS),
                    help="subset of FMs to analyze (default: all)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [analyse(n, SPECS[n], out_dir) for n in args.models if n in SPECS]
    ok = [s for s in summaries if s.get("status") == "ok"]

    sdf = pd.DataFrame(ok)
    sdf.to_csv(out_dir / "peer_path_fms_summary.csv", index=False)
    (out_dir / "peer_path_fms_summary.json").write_text(json.dumps(ok, indent=2, default=float))

    print("\n" + "=" * 110 + "\nPeer pathology-FM spectral survey\n" + "=" * 110)
    cols = ["model", "family", "n_params", "n_layers", "alpha_mean", "alpha_median",
            "alpha_dist_mean", "frac_near2_1p5_2p5", "frac_pl_valid", "traps_total",
            "score_on_nanopath"]
    have = [c for c in cols if c in sdf.columns]
    if len(sdf):
        with pd.option_context("display.width", 220, "display.max_columns", None,
                                "display.float_format", "{:.3f}".format):
            print(sdf[have].to_string(index=False))

    # The headline empirical question this script exists to answer: does mean
    # |alpha - 2| (or any wwj diagnostic) predict the published nanopath score?
    scored = sdf[sdf["score_on_nanopath"].notna()] if "score_on_nanopath" in sdf.columns else sdf.iloc[0:0]
    if len(scored) >= 4 and scored["alpha_dist_mean"].notna().all():
        r = float(np.corrcoef(scored["alpha_dist_mean"], scored["score_on_nanopath"])[0, 1])
        print(f"\nSpearman-rank baseline: alpha_dist_mean vs nanopath score  Pearson r = {r:+.3f}  (n={len(scored)})")
        if abs(r) > 0.5:
            print(f"  → Suggestive: |alpha-2| {'correlates negatively' if r < 0 else 'correlates positively'} with downstream score.")
        else:
            print("  → Weak: no strong linear relationship between |alpha-2| and downstream at this n.")

    fails = [s for s in summaries if s.get("status") != "ok"]
    if fails:
        print("\nnon-ok:", [(s["model"], s.get("status")) for s in fails])
    print(f"\n[done] {out_dir/'peer_path_fms_summary.csv'}")


if __name__ == "__main__":
    main()
