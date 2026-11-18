#!/usr/bin/env bash
# Reproduce the beta=2.3 multi-volume pilot (results/README_pilot.md) from a fresh clone.
#
# Runs 8 independent single-threaded Markov streams in parallel (6^4 x1, 8^4 x1, 10^4 x2, 12^4 x4),
# writes them to results/runs/L<L>_s<seed>/, then runs the cross-volume analysis.
# Wall time was about 27 min on an 8-core box (the 12^4 streams are the slowest).
# Stream RNG depends only on (seed, lattice size, beta), so the streams are deterministic;
# results should match the originals exactly on the same numpy/scipy versions (requirements.txt).
#
# usage:  bash results/reproduce_pilot.sh            # from the repo root
#         PYTHON=.venv/bin/python bash results/reproduce_pilot.sh
# The analysis step needs matplotlib (pip install -r requirements-plot.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
RUNS=results/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

COMMON="--betas 2.3 --n-therm 200 --n-sweeps 5 --n-or 2 --flow-radius 2.0 --flow-eps 0.025 \
--fit-min-ell 3 --dip-sims 100 --n-boot 200 --quiet"

run() {  # name size seed n_configs block_sizes...
  local name=$1 size=$2 seed=$3 n=$4; shift 4
  "$PY" -m fingerprint $COMMON --outdir "$RUNS/$name" --sizes "$size" --block-sizes "$@" \
        --seeds "$seed" --n-configs "$n" > "$RUNS/$name.log" 2>&1 &
}

start=$(date +%s)
run L6_s1   6  1 1200 2 3 6
run L8_s1   8  1 1000 2 4 8
run L10_s1 10  1  400 2 5 10
run L10_s2 10  2  400 2 5 10
run L12_s1 12  1  150 2 3 4 6 12
run L12_s2 12  2  150 2 3 4 6 12
run L12_s3 12  3  150 2 3 4 6 12
run L12_s4 12  4  150 2 3 4 6 12
wait
echo "streams done in $(( $(date +%s) - start ))s"
"$PY" results/cross_volume_analysis.py | tee results/cross_volume_analysis_output.txt > /dev/null
echo "analysis written: results/cross_volume_results.json, results/var_QV_vs_invV.{png,svg}"
