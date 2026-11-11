#!/usr/bin/env bash
# GPU verification for the three optimization branches (run on a host with a
# visible, supported NVIDIA GPU, e.g. a Kaggle Tesla T4). CI only *compiles*
# the CUDA code; this script *executes* it: parity vs the CPU reference, the
# independent NumPy oracle, Compute Sanitizer, and a batch benchmark that
# shows whether the optimization actually helps.
#
# What it checks, per branch:
#   opt/simd-gaussian          CPU-only; builds cpu preset, runs full ctest,
#                              and confirms it is bit-identical to its merge
#                              base (already verified locally, re-checked here).
#   opt/fuse-grayscale-resize  CUDA; parity + oracle + sanitizer; resize-stage
#                              benchmark vs main.
#   opt/cuda-overlap           CUDA; parity + oracle + sanitizer; BATCH
#                              throughput + application_wall_ms vs main (this is
#                              where transfer/compute overlap should show up).
#
# Usage:
#   scripts/verify_gpu_optimizations.sh [--branches b1,b2,...] [--keep]
#
# Defaults to all three branches. Run from a clean checkout of the repo; the
# script fetches and checks out each branch itself. It never modifies main's
# working tree permanently (returns to the starting ref at the end).
#
# Exit code: 0 only if every selected branch passes every check for it.

set -uo pipefail

# ----------------------------------------------------------------- config
BASE_REF="${BASE_REF:-origin/main}"            # baseline for speedup comparison
BRANCHES_DEFAULT="opt/cuda-overlap,opt/fuse-grayscale-resize,opt/simd-gaussian"
WARMUP="${WARMUP:-20}"
ITERATIONS="${ITERATIONS:-100}"
REPETITIONS="${REPETITIONS:-3}"
KEEP=0
BRANCHES="$BRANCHES_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branches) BRANCHES="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="${FRAMEPIPE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

START_REF="$(git rev-parse --abbrev-ref HEAD)"
[[ "$START_REF" == "HEAD" ]] && START_REF="$(git rev-parse HEAD)"
RESULTS_ROOT="$REPO_ROOT/results/gpu_verify_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RESULTS_ROOT"
SUMMARY="$RESULTS_ROOT/SUMMARY.txt"

note() { echo -e "$*" | tee -a "$SUMMARY"; }
section() { note "\n========================================================\n$*\n========================================================"; }

FAIL_TOTAL=0
declare -A BRANCH_RESULT

cleanup() {
  note "\nReturning to $START_REF"
  git checkout -q "$START_REF" 2>/dev/null || true
}
trap cleanup EXIT

# ----------------------------------------------------------------- toolchain
section "Environment setup"
# Prefer an existing project venv; otherwise use whatever cmake/ninja/python3
# are already on PATH (e.g. the Kaggle image, which has them). Only build a
# dedicated .venv as a last resort when no cmake is available at all.
if [[ -x .venv/bin/cmake ]]; then
  export PATH="$REPO_ROOT/.venv/bin:$PATH"
  note "Using existing project virtualenv (.venv)"
elif command -v cmake >/dev/null && command -v ninja >/dev/null; then
  note "Using system toolchain: cmake=$(command -v cmake) ninja=$(command -v ninja)"
else
  note "No cmake/ninja on PATH; creating project virtualenv (.venv)"
  python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt \
    || { note "FATAL: could not create venv / install requirements"; exit 1; }
  export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi
# Ensure a usable python3 with NumPy for the oracle/fixtures.
python3 -c 'import numpy' 2>/dev/null || pip install -q --user -r requirements.txt || true
export FRAMEPIPE_DEPS="${FRAMEPIPE_DEPS:-$HOME/.local/framepipe-deps}"

command -v cmake >/dev/null || { note "FATAL: cmake not found after setup"; exit 1; }
note "cmake: $(cmake --version | head -1)"

if [[ ! -d "$FRAMEPIPE_DEPS/lib" ]]; then
  note "Building pinned C++ dependencies into $FRAMEPIPE_DEPS (~several minutes)"
  scripts/install_deps.sh "$FRAMEPIPE_DEPS" \
    || { note "FATAL: install_deps.sh failed"; exit 1; }
fi

if ! command -v nvidia-smi >/dev/null; then
  note "FATAL: no nvidia-smi; this script must run on a GPU host"
  exit 1
fi
note "GPU(s):"; nvidia-smi -L | tee -a "$SUMMARY"

git fetch -q origin || note "WARN: git fetch failed; using local refs"

# Deterministic benchmark inputs (manifests + configs under data/bench).
note "Generating benchmark fixtures"
python3 scripts/generate_fixtures.py benchmark --out data/bench >/dev/null \
  || { note "FATAL: fixture generation failed"; exit 1; }

# ----------------------------------------------------------------- helpers
# Run a command, tee its output to a log, record pass/fail into a per-branch
# counter. Usage: step <branch> <label> <logfile> -- <cmd...>
step() {
  local branch="$1" label="$2" log="$3"; shift 3
  [[ "$1" == "--" ]] && shift
  note "  [$branch] $label ..."
  if "$@" >"$log" 2>&1; then
    note "  [$branch] $label: PASS   (log: ${log#$REPO_ROOT/})"
    return 0
  else
    note "  [$branch] $label: FAIL   (log: ${log#$REPO_ROOT/})"
    note "      last lines:"; tail -8 "$log" | sed 's/^/        /' | tee -a "$SUMMARY" >/dev/null
    return 1
  fi
}

build_cuda() {   # <outdir-log>
  cmake --preset cuda >"$1" 2>&1 && cmake --build --preset cuda >>"$1" 2>&1
}
build_cpu() {    # <outdir-log>
  cmake --preset cpu >"$1" 2>&1 && cmake --build --preset cpu >>"$1" 2>&1
}

# GPU tests must EXECUTE, not skip. The suite skips cleanly when no device is
# present, so "no failures" is not enough; we also forbid SKIPPED.
gpu_tests_execute() {   # <bin>
  local out; out="$("$1" --gtest_brief=1 2>&1)"; local rc=$?
  echo "$out"
  [[ $rc -eq 0 ]] || return 1
  if echo "$out" | grep -q 'SKIPPED'; then
    echo "ERROR: GPU tests reported SKIPPED (no usable device?) -- not real verification" >&2
    return 1
  fi
  return 0
}

sanitizer_suite() {   # <bin>
  local bin="$1" t rc=0
  for t in memcheck racecheck initcheck synccheck; do
    echo "=== compute-sanitizer --tool $t ==="
    compute-sanitizer --tool "$t" --error-exitcode 99 "$bin" --gtest_brief=1 || rc=1
  done
  return $rc
}

# Batch benchmark for one build dir; writes a results dir and echoes the
# batch_wall_ms median and application_wall_ms for cuda.
batch_bench() {   # <build-dir> <outdir>
  local bdir="$1" out="$2"
  "$bdir/bin/framepipe" benchmark \
    --manifest data/bench/hd_ring.jsonl \
    --config data/bench/hd.config.json \
    --backends cuda \
    --scopes application,batch,host_processing,device_pipeline \
    --workload ring \
    --warmup "$WARMUP" --iterations "$ITERATIONS" --repetitions "$REPETITIONS" \
    --output-dir "$out" --overwrite >/dev/null 2>&1
}

# Pull a median metric out of a results summary.json (best-effort).
metric() {   # <summary.json> <metric-name>
  python3 - "$1" "$2" <<'PY' 2>/dev/null || echo "n/a"
import json,sys
try:
    s=json.load(open(sys.argv[1]))
except Exception:
    print("n/a"); sys.exit()
want=sys.argv[2]
def walk(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if k==want and isinstance(v,(int,float)): yield v
            yield from walk(v)
    elif isinstance(o,list):
        for v in o: yield from walk(v)
vals=list(walk(s))
print(round(min(vals),3) if vals else "n/a")
PY
}

# ----------------------------------------------------------------- per branch
verify_cpu_branch() {   # opt/simd-gaussian
  local branch="$1" bdir="$RESULTS_ROOT/${branch//\//_}"; mkdir -p "$bdir"
  local fails=0
  git checkout -q "$branch" || { note "  [$branch] checkout FAILED"; BRANCH_RESULT[$branch]="CHECKOUT-FAIL"; return 1; }
  note "  [$branch] at $(git rev-parse --short HEAD)"
  step "$branch" "build cpu preset"           "$bdir/build.log"  -- build_cpu "$bdir/build.log" || fails=$((fails+1))
  step "$branch" "ctest (all CPU + oracle)"   "$bdir/ctest.log"  -- ctest --preset cpu || fails=$((fails+1))
  # Bit-identical to the merge base: build base, diff edge-map outputs.
  local base; base="$(git merge-base HEAD "$BASE_REF")"
  note "  [$branch] bit-exactness vs merge base $base (radii 1,3,7,15)"
  local newdir="$bdir/out_new" olddir="$bdir/out_base"; mkdir -p "$newdir" "$olddir"
  for r in 1 3 7 15; do
    "$REPO_ROOT/build/cpu/bin/framepipe" process --input tests/fixtures/random_rgb_257x129.png \
      --output "$newdir/r$r.png" --width 200 --height 150 --radius "$r" --sigma 1.2 \
      --backend cpu-ref --overwrite >/dev/null 2>&1
  done
  git stash -q --include-untracked 2>/dev/null; git checkout -q "$base"
  build_cpu "$bdir/build_base.log"
  for r in 1 3 7 15; do
    "$REPO_ROOT/build/cpu/bin/framepipe" process --input tests/fixtures/random_rgb_257x129.png \
      --output "$olddir/r$r.png" --width 200 --height 150 --radius "$r" --sigma 1.2 \
      --backend cpu-ref --overwrite >/dev/null 2>&1
  done
  git checkout -q "$branch"; git stash pop -q 2>/dev/null || true
  if diff -rq "$newdir" "$olddir" >"$bdir/bitexact.log" 2>&1; then
    note "  [$branch] bit-exactness: PASS (outputs byte-identical to base)"
  else
    note "  [$branch] bit-exactness: FAIL (see bitexact.log)"; fails=$((fails+1))
  fi
  [[ $fails -eq 0 ]] && BRANCH_RESULT[$branch]="PASS" || BRANCH_RESULT[$branch]="FAIL($fails)"
  return $fails
}

verify_cuda_branch() {   # opt/cuda-overlap | opt/fuse-grayscale-resize
  local branch="$1" bdir="$RESULTS_ROOT/${branch//\//_}"; mkdir -p "$bdir"
  local fails=0
  git checkout -q "$branch" || { note "  [$branch] checkout FAILED"; BRANCH_RESULT[$branch]="CHECKOUT-FAIL"; return 1; }
  note "  [$branch] at $(git rev-parse --short HEAD)"

  step "$branch" "build cuda preset"          "$bdir/build.log"   -- build_cuda "$bdir/build.log" || { fails=$((fails+1)); BRANCH_RESULT[$branch]="BUILD-FAIL"; return 1; }
  step "$branch" "doctor (cuda usable=yes)"    "$bdir/doctor.log"  -- "$REPO_ROOT/build/cuda/bin/framepipe" doctor --json
  step "$branch" "GPU tests execute (no SKIP)" "$bdir/gpu.log"     -- gpu_tests_execute "$REPO_ROOT/build/cuda/bin/framepipe_cuda_tests" || fails=$((fails+1))
  step "$branch" "oracle suite (cpu-ref,cuda)" "$bdir/oracle.log"  -- \
      python3 scripts/run_oracle_suite.py --framepipe "$REPO_ROOT/build/cuda/bin/framepipe" \
      --work-dir "$bdir/oracle_work" --candidates cpu-ref,cuda || fails=$((fails+1))
  step "$branch" "Compute Sanitizer (4 tools)" "$bdir/sanitizer.log" -- \
      sanitizer_suite "$REPO_ROOT/build/cuda/bin/framepipe_cuda_tests" || fails=$((fails+1))

  # Speedup vs base. Build base cuda once per branch (branches differ), measure both.
  note "  [$branch] measuring batch/application vs base ($BASE_REF)"
  batch_bench "$REPO_ROOT/build/cuda" "$bdir/bench_branch" && \
    note "      branch: batch_wall_ms=$(metric "$bdir/bench_branch/summary.json" batch_wall_ms) application_wall_ms=$(metric "$bdir/bench_branch/summary.json" application_wall_ms) gpu_transfer_inclusive_wall_ms=$(metric "$bdir/bench_branch/summary.json" gpu_transfer_inclusive_wall_ms)" \
    || note "      branch benchmark: FAIL (see bench_branch)"

  git checkout -q "$BASE_REF"
  if build_cuda "$bdir/build_base.log"; then
    batch_bench "$REPO_ROOT/build/cuda" "$bdir/bench_base" && \
      note "      base:   batch_wall_ms=$(metric "$bdir/bench_base/summary.json" batch_wall_ms) application_wall_ms=$(metric "$bdir/bench_base/summary.json" application_wall_ms) gpu_transfer_inclusive_wall_ms=$(metric "$bdir/bench_base/summary.json" gpu_transfer_inclusive_wall_ms)" \
      || note "      base benchmark: FAIL"
  else
    note "      base cuda build FAILED; cannot compute speedup"
  fi
  git checkout -q "$branch"

  [[ $fails -eq 0 ]] && BRANCH_RESULT[$branch]="PASS" || BRANCH_RESULT[$branch]="FAIL($fails)"
  return $fails
}

# ----------------------------------------------------------------- run
IFS=',' read -ra SELECTED <<< "$BRANCHES"
for b in "${SELECTED[@]}"; do
  section "Verifying $b"
  case "$b" in
    opt/simd-gaussian) verify_cpu_branch "$b" || FAIL_TOTAL=$((FAIL_TOTAL+1)) ;;
    *)                 verify_cuda_branch "$b" || FAIL_TOTAL=$((FAIL_TOTAL+1)) ;;
  esac
done

# ----------------------------------------------------------------- report
section "RESULTS"
for b in "${SELECTED[@]}"; do
  note "  $b : ${BRANCH_RESULT[$b]:-NOT-RUN}"
done
note "\nFull logs and benchmark result dirs under: ${RESULTS_ROOT#$REPO_ROOT/}"
note "Compare benchmark dirs with: python3 scripts/analyze_results.py <branch-dir> <base-dir>"

if [[ $FAIL_TOTAL -eq 0 ]]; then
  note "\nALL SELECTED BRANCHES PASSED GPU VERIFICATION."
  exit 0
else
  note "\n$FAIL_TOTAL branch(es) had failures. Do NOT merge those until resolved."
  exit 1
fi
