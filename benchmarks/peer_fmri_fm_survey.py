"""Peer fMRI-FM spectral survey — run the CortexMAE HT-SR + RG diagnostics on the
other fMRI foundation models in the Brainmarks registry, so they land in one
cross-FM alpha/phi/M_tr/trap table comparable to the 70 CortexMAE encoders.

Architecture-agnostic: we don't install each FM's framework. We load the published
checkpoint, extract the *encoder* sub-state-dict using the exact key recipe from the
Brainmarks wrapper, wrap every 2D weight as a synthetic nn.Linear, and run the same
weightwatcher fork analyze(randomize=True, mp_fit=True) + RG extras (phi_k, M_tr,
isolated Correlation Traps) we used for CortexMAE. WW only needs the weight matrices.

Peer FMs (encoder extraction per Brainmarks wrapper):
  brain_semantoks  HF SamGijsen/Brain-Semantoks          teacher_encoder.*  (self-distilled, 457 ROI)
  neurostorm       HF zxcvb20001/NeuroSTORM              state_dict, drop decoder (4D MAE)
  brainlm_13m/111m HF vandijklab/brainlm                 ViT-MAE vit.encoder.* (parcel)
  brainjepa        GDrive jepa-ep300.pth.tar             ckpt['target_encoder'] (JEPA, 450 ROI)
  swift            GDrive contrastive_pretrained.ckpt    state_dict, drop decoder (Swin 4D)
  brainharmonix    GDrive harmonix-f/model.pth           encoder_ema.* (TR-aware)

    cd /home/mhough/dev/wwj && uv run python benchmarks/peer_fmri_fm_survey.py
(or run under the pytorch_26.04.sif container with the ww fork on PYTHONPATH.)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CK = "/data/derivatives/peer_fm_ww/ckpts"


# ---------------------------------------------------------------------------
# checkpoint loading + per-FM encoder extraction
# ---------------------------------------------------------------------------
def _load(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _strip(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _drop_module(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def enc_semantoks(path):
    c = _load(path)
    sd = c.get("model_state_dict", c)
    sd = _drop_module(sd)
    return _strip(sd, "teacher_encoder.")


def enc_brainjepa(path):
    c = _load(path)
    sd = c["target_encoder"] if "target_encoder" in c else c
    return _drop_module(sd)


def enc_harmonix(path):
    sd = _load(path)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd
    return _strip(sd, "encoder_ema.")


def _statedict_from_lightning(path):
    c = _load(path)
    for k in ("state_dict", "model_state_dict", "model"):
        if isinstance(c, dict) and k in c and isinstance(c[k], dict):
            return _drop_module(c[k])
    return _drop_module(c)


def enc_drop_decoder(path):
    """Generic: take the full state_dict, drop anything on the decoder/head side."""
    sd = _statedict_from_lightning(path)
    drop = ("decoder", "mask_token", "_dec", "head.", "reconstruct", "pred_head",
            "lm_head", "to_pixels", "unpatch")
    return {k: v for k, v in sd.items() if not any(d in k.lower() for d in drop)}


def enc_brainlm(path):
    sd = _statedict_from_lightning(path)
    # HF ViT-MAE: keep vit encoder side, drop decoder
    keep = {k: v for k, v in sd.items()
            if ("encoder" in k.lower() or k.lower().startswith("vit."))
            and "decoder" not in k.lower()}
    return keep or enc_drop_decoder(path)


# name -> (filename glob under CK, extractor, input_space label, citation tag)
SPECS = {
    "brain_semantoks": ("brainsemantoks_ckpt_epoch_100.pth", enc_semantoks, "parcel457", "Gijsen+ self-distill"),
    "neurostorm":      ("*pt_fmrifound_mae_ratio0.8.ckpt",   enc_drop_decoder, "volume4d", "NeuroSTORM MAE"),
    "brainlm_13m":     ("**/old_13M*/*.bin",                 enc_brainlm,    "parcel424", "BrainLM 13M (ICLR24)"),
    "brainlm_13m_st":  ("**/old_13M*/*.safetensors",         enc_brainlm,    "parcel424", "BrainLM 13M (ICLR24)"),
    "brainlm_111m":    ("**/vitmae_111M*/*.bin",             enc_brainlm,    "parcel424", "BrainLM 111M (ICLR24)"),
    "brainlm_111m_st": ("**/vitmae_111M*/*.safetensors",     enc_brainlm,    "parcel424", "BrainLM 111M (ICLR24)"),
    "brainjepa":       ("jepa-ep300.pth.tar",                enc_brainjepa,  "parcel450", "Brain-JEPA (NeurIPS24)"),
    "swift":           ("swift_contrastive_pretrained.ckpt", enc_drop_decoder, "volume4d", "SwiFT (NeurIPS23)"),
    "brainharmonix":   ("brainharmonix_f_model.pth",         enc_harmonix,   "parcel-tr", "Brain-Harmony-F"),
}


# ---------------------------------------------------------------------------
# synthetic nn.Linear wrapper so WW (and the fork's get_ESD/analyze_traps) can walk it
# ---------------------------------------------------------------------------
def statedict_wrapper(sd: dict, min_dim: int = 10):
    import torch.nn as nn

    class _W(nn.Module):
        def __init__(self, sd):
            super().__init__()
            self.kept = 0
            for name, t in sd.items():
                if not torch.is_tensor(t) or t.ndim < 2:
                    continue
                tf = t if t.ndim == 2 else t.reshape(t.shape[0], -1)
                if min(tf.shape) < min_dim:
                    continue
                lin = nn.Linear(tf.shape[1], tf.shape[0], bias=False)
                with torch.no_grad():
                    lin.weight.copy_(tf.float())
                setattr(self, name.replace(".", "__"), lin)
                self.kept += 1

        def forward(self, x):
            return x

    return _W(sd)


# ---------------------------------------------------------------------------
# RG/HT-SR diagnostics (same as CortexMAE/scripts/weightwatcher_cortexmae.py)
# ---------------------------------------------------------------------------
def _eig_metrics(evals, ks=(1, 5, 10)):
    e = np.asarray(evals, float)
    e = e[np.isfinite(e) & (e > 0)]
    out = {}
    if e.size == 0:
        for k in ks:
            out[f"phi_{k}"] = np.nan
        out.update(M_tr=np.nan, M_tr_frac=np.nan, num_evals_pos=0)
        return out
    s, s2 = float(e.sum()), float((e * e).sum())
    es = np.sort(e)[::-1]
    for k in ks:
        out[f"phi_{k}"] = float(es[:k].sum() / s)
    out["M_tr"] = float(s * s / s2)
    out["M_tr_frac"] = float((s * s / s2) / e.size)
    out["num_evals_pos"] = int(e.size)
    return out


def isolated_trap_counts(watcher, model=None):
    try:
        rm, ts = watcher.randomize_model(model=model, return_state=True)
        permuted = sorted(ts.get("permuted_ids", {}).keys())
        if not permuted:
            return {}
        traps = watcher.analyze_traps(randomized_model=rm, trap_state=ts, layers=permuted)
    except Exception as exc:
        print(f"    [warn] analyze_traps: {exc}", file=sys.stderr)
        return {}
    if isinstance(traps, tuple):
        traps = next((t for t in traps if isinstance(t, pd.DataFrame)), None)
    if traps is None or len(traps) == 0:
        return {}
    idc = next((c for c in ("layer_id", "id", "ww_layer_id") if c in traps.columns), None)
    return {int(k): int(v) for k, v in traps.groupby(idc).size().items()} if idc else {}


def compute_rg_extras(watcher, details, ks=(1, 5, 10), trap_counts=None):
    carry = [c for c in ("alpha", "alpha_weighted", "stable_rank", "log_norm",
                         "rand_num_spikes", "lambda_max", "Q") if c in details.columns]
    trap_counts = trap_counts or {}
    rows = []
    for _, r in details.iterrows():
        lid = int(r["layer_id"])
        rec = {"layer_id": lid, "name": r.get("name")}
        try:
            evals = watcher.get_ESD(layer=lid)
        except Exception:
            evals = []
        rec.update(_eig_metrics(evals, ks))
        for c in carry:
            rec[c] = r.get(c)
        rec["n_traps_isolated"] = int(trap_counts.get(lid, 0))
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(name, extras, family, space, citation, n_params, ks=(1, 5, 10)):
    a = extras["alpha"] if "alpha" in extras.columns else pd.Series([], dtype=float)
    s = {"model": name, "family": family, "input_space": space, "citation": citation,
         "status": "ok", "n_params": int(n_params), "n_layers": int(len(extras)),
         "alpha_mean": float(a.mean()) if len(a) else None,
         "alpha_median": float(a.median()) if len(a) else None,
         "alpha_min": float(a.min()) if len(a) else None,
         "alpha_max": float(a.max()) if len(a) else None,
         "frac_alpha_lt2": float((a < 2).mean()) if len(a) else None,
         "frac_healthy_2_6": float(((a >= 2) & (a <= 6)).mean()) if len(a) else None,
         "frac_alpha_gt6": float((a > 6).mean()) if len(a) else None}
    for c in [f"phi_{k}" for k in ks] + ["M_tr", "M_tr_frac"]:
        if c in extras.columns:
            s[f"{c}_mean"] = float(extras[c].mean())
            s[f"{c}_median"] = float(extras[c].median())
    if "n_traps_isolated" in extras.columns:
        s["n_traps_isolated_total"] = float(extras["n_traps_isolated"].fillna(0).sum())
        s["layers_with_n_traps_isolated"] = int((extras["n_traps_isolated"].fillna(0) > 0).sum())
    return s


def resolve(spec_glob):
    hits = sorted(glob.glob(str(Path(CK) / spec_glob), recursive=True)) \
        + sorted(glob.glob(str(Path("/data/derivatives/peer_fm_ww/hf_cache") / "**" / spec_glob), recursive=True))
    return hits[0] if hits else None


def analyse(name, spec, out_dir, ks):
    fn_glob, extractor, space, cite = spec
    path = resolve(fn_glob)
    print(f"\n=== {name} ({cite}) ===", flush=True)
    if not path:
        print(f"  [skip] no checkpoint matching {fn_glob}", flush=True)
        return {"model": name, "family": name.split("_")[0], "status": "no_ckpt"}
    print(f"  ckpt: {path}", flush=True)
    try:
        enc = extractor(path)
        n_params = sum(int(v.numel()) for v in enc.values() if torch.is_tensor(v))
        model = statedict_wrapper(enc)
        print(f"  encoder tensors={len(enc)}  WW-matrices={model.kept}  params={n_params:,}", flush=True)
        if model.kept == 0:
            return {"model": name, "family": name.split("_")[0], "status": "no_matrices"}
    except Exception as e:
        print(f"  [extract failed] {type(e).__name__}: {e}", flush=True)
        return {"model": name, "family": name.split("_")[0], "status": "extract_failed", "error": str(e)[:200]}

    import weightwatcher as ww
    try:
        w = ww.WeightWatcher(model=model)
        details = w.analyze(randomize=True, mp_fit=True)
        traps = isolated_trap_counts(w, model=model)
        extras = compute_rg_extras(w, details, ks=ks, trap_counts=traps)
    except Exception as e:
        print(f"  [WW failed] {type(e).__name__}: {e}", flush=True)
        return {"model": name, "family": name.split("_")[0], "status": "ww_failed", "error": str(e)[:200]}

    extras.to_csv(out_dir / f"{name}_details.csv", index=False)
    fam = "brainlm" if name.startswith("brainlm") else name
    summ = summarize(name, extras, fam, space, cite, n_params, ks=ks)
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(f"  alpha mean={summ['alpha_mean']:.3f} med={summ['alpha_median']:.3f}  "
          f"%a<2={summ['frac_alpha_lt2']:.1%}  %[2,6]={summ['frac_healthy_2_6']:.1%}  "
          f"phi1_med={summ.get('phi_1_median', float('nan')):.4f}  "
          f"M_tr_med={summ.get('M_tr_median', float('nan')):.1f}  "
          f"traps={summ.get('n_traps_isolated_total', 0):.0f}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/data/derivatives/peer_fm_ww/results")
    ap.add_argument("--models", nargs="+", default=list(SPECS))
    ap.add_argument("--ks", default="1,5,10")
    args = ap.parse_args()
    ks = tuple(int(x) for x in args.ks.split(","))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [analyse(n, SPECS[n], out_dir, ks) for n in args.models if n in SPECS]
    ok = [s for s in summaries if s.get("status") == "ok"]

    # de-dup brainlm bin/safetensors (keep whichever loaded)
    seen = {}
    for s in ok:
        seen[s["model"].replace("_st", "")] = s
    ok = list(seen.values())

    sdf = pd.DataFrame(ok)
    sdf.to_csv(out_dir / "peer_fms_summary.csv", index=False)
    (out_dir / "peer_fms_summary.json").write_text(json.dumps(ok, indent=2, default=float))

    print("\n" + "=" * 100 + "\nPeer fMRI-FM spectral survey\n" + "=" * 100)
    cols = ["model", "family", "input_space", "n_params", "n_layers", "alpha_mean",
            "alpha_median", "frac_alpha_lt2", "frac_healthy_2_6", "phi_1_median",
            "M_tr_median", "n_traps_isolated_total"]
    have = [c for c in cols if c in sdf.columns]
    if len(sdf):
        with pd.option_context("display.width", 220, "display.max_columns", None):
            print(sdf[have].to_string(index=False))
    fails = [s for s in summaries if s.get("status") != "ok"]
    if fails:
        print("\nnon-ok:", [(s["model"], s["status"]) for s in fails])
    print(f"\n[done] {out_dir/'peer_fms_summary.csv'}")


if __name__ == "__main__":
    main()
