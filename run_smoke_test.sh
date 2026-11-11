#!/usr/bin/env bash
# Smoke test: tiny 4^4 and 6^4 lattices, a handful of sweeps/configs.
# Verifies the whole pipeline (update, flow, measure, analyse, write) runs.
# Not physics: the numbers it prints are meaningless at this size.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
OUT="${1:-smoke_output}"
rm -rf "$OUT"
start=$(date +%s)
"$PY" -m fingerprint \
  --sizes 4 6 --betas 2.3 \
  --n-therm 10 --n-configs 8 --n-sweeps 2 --n-or 1 \
  --flow-radius 1.0 --flow-eps 0.05 \
  --seeds 1 2 --dip-sims 20 --n-boot 40 \
  --outdir "$OUT" --quiet
for f in summary.txt run_config.json \
         L4x4x4x4_b2.3/summary.txt L4x4x4x4_b2.3/config_measurements.csv \
         L4x4x4x4_b2.3/measurements.npz L4x4x4x4_b2.3/hist_Q_over_V.csv \
         L4x4x4x4_b2.3/hist_qbar_ell2.csv L4x4x4x4_b2.3/block_scaling.csv \
         L4x4x4x4_b2.3/twopoint_radial.csv L4x4x4x4_b2.3/twopoint_axial.csv \
         L4x4x4x4_b2.3/flow_history.csv L4x4x4x4_b2.3/results.json \
         L6x6x6x6_b2.3/summary.txt; do
  [ -s "$OUT/$f" ] || { echo "SMOKE TEST FAILED: missing $OUT/$f"; exit 1; }
done
echo "smoke test passed in $(( $(date +%s) - start ))s; see $OUT/summary.txt"
