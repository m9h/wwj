"""Backfill existing wwjd training runs into a trackio dashboard (project 'wwjd-htsr'), so
the HT-SR training diagnostics -- alpha, induction, copying, stable rank -- become live,
comparable, shareable curves rather than post-hoc CSVs. Replays HF Trainer log_history.json
(LM loss / lr / grad_norm / alpha_penalty) + traj_summary.csv (spectral) + circuit_summary.csv
(mechanistic-interp) as one trackio run per training arm.

Usage:
  uv run --extra benchmarks --with trackio python benchmarks/trackio_backfill.py
View:
  uv run --with trackio trackio show --project wwjd-htsr        # local dashboard (Gradio)
Share (opt-in, needs `huggingface-cli login`):
  pass space_id="m9h/wwjd-tracking" to trackio.init to host the dashboard on a HF Space.
"""
import json
import subprocess
from pathlib import Path
import pandas as pd
import trackio

PROJECT = "wwjd-htsr"
HERE = Path(__file__).resolve().parent


def _sha():
    try:
        return subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def backfill_run(name, group, config, log_history=None, traj_summary=None, circuit_summary=None,
                 space_id=None):
    events = {}   # step -> {metric: value}; merged so each step logs once (curves stay monotonic)

    def add(step, d):
        events.setdefault(int(step), {}).update(
            {k: v for k, v in d.items() if v is not None and v == v})   # drop None / NaN

    if log_history and Path(log_history).exists():
        for r in json.load(open(log_history)):
            if "loss" in r:
                add(r.get("step", 0), {"train/loss": r.get("loss"), "train/lr": r.get("learning_rate"),
                                       "train/grad_norm": r.get("grad_norm"),
                                       "train/alpha_penalty": r.get("alpha_penalty")})
    if traj_summary and Path(traj_summary).exists():
        for _, r in pd.read_csv(traj_summary).iterrows():
            add(r["step"], {"spectral/alpha_bayes": r.get("mean_alpha_bayes"),
                            "spectral/alpha_freq": r.get("mean_alpha_freq"),
                            "spectral/abs_dist_from_2": r.get("mean_absdist2_bayes"),
                            "spectral/frac_pl_rejected": r.get("frac_pl_rejected")})
    if circuit_summary and Path(circuit_summary).exists():
        for _, r in pd.read_csv(circuit_summary).iterrows():
            add(r["step"], {"circuit/max_induction": r.get("max_induction"),
                            "circuit/n_induction_heads": r.get("n_induction_heads"),
                            "circuit/mean_copying": r.get("mean_copying"),
                            "circuit/mean_stable_rank": r.get("mean_stable_rank"),
                            "circuit/subspace_drift": r.get("mean_subspace_drift")})

    trackio.init(project=PROJECT, name=name, group=group, space_id=space_id,
                 config={**config, "git_sha": _sha(), "backfilled": True})
    for step in sorted(events):
        trackio.log(events[step], step=step)
    trackio.finish()
    print(f"[trackio] backfilled '{name}': {len(events)} steps, "
          f"{sorted({k for d in events.values() for k in d})}")


def main(tag="scratch_traj", space_id=None):
    D = Path("/data/mhough/wwj_traj")
    if tag == "scratch_traj":   # the rich baseline alpha-trajectory (no regularizer)
        backfill_run(
            name="scratch_traj", group="trajectory",
            config={"recipe": "from-scratch GPT-2-124M", "lr": 6e-4, "beta2": 0.95,
                    "wd": 0.1, "epochs": 1, "alpha_reg": 0.0, "seed": "unseeded(original)"},
            log_history=D / "log_history.json", traj_summary=D / "traj_summary.csv",
            circuit_summary=D / "circuit_summary.csv", space_id=space_id)
    else:                       # a tagged causal-test arm (areg_base / areg_a2):
        backfill_run(            #   log_history + traj_summary under <tag>/, circuit CSV prefixed
            name=tag, group="areg",
            config={"recipe": "from-scratch GPT-2-124M (causal alpha-reg test)", "tag": tag,
                    "lr": 6e-4, "beta2": 0.95, "wd": 0.1, "epochs": 1, "seed": 42},
            log_history=D / tag / "log_history.json", traj_summary=D / tag / "traj_summary.csv",
            circuit_summary=D / f"{tag}_circuit_summary.csv", space_id=space_id)


if __name__ == "__main__":
    import sys
    main(tag=sys.argv[1] if len(sys.argv) > 1 else "scratch_traj",
         space_id=sys.argv[2] if len(sys.argv) > 2 else None)
