#!/usr/bin/env bash
# Reproducible build (Python/Sweave pattern): tangle the data into figures + macros +
# tables, then weave the LaTeX. One command regenerates the manuscript from the CSVs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TECTONIC=/home/mhough/miniforge3/envs/texlive/bin/tectonic
( cd "$ROOT" && uv run --extra benchmarks python paper/generate.py )   # tangle
( cd "$ROOT/paper" && "$TECTONIC" --keep-logs --reruns 3 wwjd_paper.tex )  # weave
echo "built paper/wwjd_paper.pdf"
