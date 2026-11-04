#!/usr/bin/env bash
# Runs the required scenario matrix (spec 18.2) and selected Gaussian radii.
#
# Usage: benchmarks/run_matrix.sh BUILD_DIR OUT_ROOT [BACKENDS] [THREADS]
#   BUILD_DIR  e.g. build/cpu or build/cuda
#   OUT_ROOT   new directory for all result runs (must not exist)
#   BACKENDS   default cpu-ref,cpu-omp,opencv-cpu,cuda
#   THREADS    cpu-omp threads (default: nproc)
# Env: WARMUP (20) ITERATIONS (200) REPETITIONS (5) WORKLOADS ("repeat ring")
# Inputs: data/bench from `python3 scripts/generate_fixtures.py benchmark --out data/bench`
set -euo pipefail

BUILD="${1:?build dir}"
OUT="${2:?output root}"
BACKENDS="${3:-cpu-ref,cpu-omp,opencv-cpu,cuda}"
THREADS="${4:-$(nproc)}"
WARMUP="${WARMUP:-20}"
ITER="${ITERATIONS:-200}"
REP="${REPETITIONS:-5}"
WORKLOADS="${WORKLOADS:-repeat ring}"
FP="$BUILD/bin/framepipe"
DATA="data/bench"

[[ -x "$FP" ]] || { echo "missing $FP" >&2; exit 2; }
[[ -d "$DATA" ]] || { echo "missing $DATA; run scripts/generate_fixtures.py benchmark --out $DATA" >&2; exit 2; }
[[ -e "$OUT" ]] && { echo "$OUT exists; choose a new directory" >&2; exit 3; }
mkdir -p "$OUT"
"$FP" doctor --json > "$OUT/doctor.json"

run() { # name manifest config extra...
  local name="$1" manifest="$2" config="$3"; shift 3
  echo "== $name" >&2
  "$FP" benchmark --manifest "$manifest" --config "$config" --backends "$BACKENDS" \
    --scopes host_processing,device_pipeline,application,startup,batch \
    --warmup "$WARMUP" --iterations "$ITER" --repetitions "$REP" --threads "$THREADS" \
    --output-dir "$OUT/$name" "$@" || echo "   (exit $? recorded in $OUT/$name/run.json)" >&2
}

for sc in tiny small hd uhd odd upscale same; do
  for wl in $WORKLOADS; do
    if [[ "$wl" == ring ]]; then
      run "${sc}_ring" "$DATA/${sc}_ring.jsonl" "$DATA/${sc}.config.json" --workload ring
    else
      run "${sc}" "$DATA/${sc}.jsonl" "$DATA/${sc}.config.json"
    fi
  done
done

# Gaussian radius sweep and output-stage comparison on representative sizes.
for r in 1 3 7 15; do
  run "hd_radius${r}" "$DATA/hd.jsonl" "$DATA/hd.config.json" --radius "$r"
done
for st in resize blur edges; do
  run "hd_stage_${st}" "$DATA/hd.jsonl" "$DATA/hd.config.json" --output-stage "$st"
done
echo "matrix complete: $OUT" >&2
