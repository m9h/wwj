#!/usr/bin/env bash
# Watch the HBN FM embedding cache for a random-init / untrained deep-FM checkpoint, and run the
# covariance-alpha TRUE control (zeta_randinit_control.py) the moment its embeddings land. Writes a
# persistent, timestamped result to $LOG so the outcome survives the launching session.
#
# Launch (background):
#   nohup bash /home/mhough/dev/wwj/benchmarks/zeta_law/watch_randinit_control.sh \
#     >/data/mhough/zeta_beta/randinit_watcher.out 2>&1 &
# Detection: any embedding key matching rand/init/untrain/scratch/shuffle that is not a known trained key
# (see zeta_randinit_control.RANDINIT_TOKENS / TRAINED).
set -u
WWJ=/home/mhough/dev/wwj
ZL="$WWJ/benchmarks/zeta_law"
PY=( taskset -c 18,19 nice -n 15 env -u CONDA_PREFIX OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu "$WWJ/.venv/bin/python" )
LOG=/data/mhough/zeta_beta/randinit_control_result.log
CADENCE=${CADENCE:-900}     # seconds between polls (15 min)
MAXITER=${MAXITER:-672}     # ~7 days

detect() {
  "${PY[@]}" - <<'PYEOF'
import sys
sys.path.insert(0, "/home/mhough/dev/wwj/benchmarks/zeta_law")
from zeta_randinit_control import _randinit_keys
ks = _randinit_keys()
print(",".join(ks))
sys.exit(0 if ks else 1)
PYEOF
}

echo "[$(date)] watcher started (cadence=${CADENCE}s, max=${MAXITER})" | tee -a "$LOG"
for i in $(seq 1 "$MAXITER"); do
  if keys=$(detect); then
    echo "[$(date)] random-init embeddings landed: ${keys}" | tee -a "$LOG"
    "${PY[@]}" "$ZL/zeta_randinit_control.py" 2>&1 | tee -a "$LOG"
    echo "[$(date)] TRUE untrained-deep-FM control complete." | tee -a "$LOG"
    exit 0
  fi
  sleep "$CADENCE"
done
echo "[$(date)] watcher timed out with no random-init embeddings." | tee -a "$LOG"
exit 2
