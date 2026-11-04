# Benchmark method

## Timing scopes

Each sample is labeled with one unambiguous metric name. `--scopes` selects
which are measured.

| `--scopes` value | Metric | Start | Stop | Includes | Excludes |
|---|---|---|---|---|---|
| `host_processing` | `cpu_core_wall_ms` (CPU backends, opencv-cpu) | before `process_into` with canonical input in host memory | `process_into` returned, FP32 output in caller view | all selected operators, call overhead | decode, prepare, quantize, encode |
| `host_processing` | `gpu_transfer_inclusive_wall_ms` (cuda) | same | same (`process_into` waits on the owned stream and copies into the caller view) | staging copy, H2D, kernels, D2H, stream sync, final host copy | decode, prepare, quantize, encode |
| `device_pipeline` | `gpu_device_pipeline_ms` (cuda only) | `cudaEventRecord(start)` before the first kernel | `cudaEventRecord(stop)` after the last kernel; `cudaEventSynchronize(stop)` before reading | the ordered kernel sequence on one stream | H2D/D2H (upload before, download after, both outside), host staging, I/O |
| `application` | `application_wall_ms` | before reading the file | after PNG write and close | read, decode, BGR→RGB, prepare if shape changed, process, quantize, encode, ordinary write | fsync (not performed; `fsync: false` recorded) |
| `startup` | `startup_wall_ms` | before constructing a new `Pipeline` | `prepare` returned | runtime/context setup, allocation, tables | per-frame work |
| `batch` | `batch_wall_ms` | before reading the first manifest frame | after the last output write | every frame's read/decode/process/encode, re-prepare | warm-up, startup |

`device_pipeline` is never synthesized for CPU backends. A missing GPU
measurement is absent (and its ratio is `null`), never zero.

Host intervals use `std::chrono::steady_clock`. Allocations and CUDA event
creation happen outside the measured loop. The CUDA backend synchronizes its
own stream (not the device) at completion; before each device-resident sample
the stream is drained so no earlier work is attributed to the sample.
Debug synchronization (`FRAMEPIPE_CUDA_DEBUG_SYNC`), stage downloads, verbose
logging, and sanitizers are off in benchmark builds and recorded in
`run.json.measurement_settings` / `environment.json.build`.

After every timed call the output is consumed (a sparse checksum, outside the
interval) so the work cannot be discarded.

## Protocol

1. Release build; correctness tests pass for the configuration.
2. The harness validates each candidate against cpu-ref on every manifest
   frame before measuring (`validation.json`). A backend that fails parity is
   marked `failed_correctness` and not measured. The OpenCV workload is
   measured either way but labeled `matched_semantics: true/false`.
3. Defaults: 20 warm-up iterations (unrecorded), 200 measured iterations, 5
   repetitions. Actual counts are stored in `summary.json.counts`.
4. Backend order rotates deterministically each repetition; the order is
   stored in `run.json.backend_run_order`.
5. `--workload repeat` measures each frame repeatedly (labeled as such);
   `--workload ring` cycles through same-shape frames.
6. No clocks, governors, or power limits are changed.

## Statistics

Computed from raw samples (`samples.jsonl`) and recomputable by
`scripts/validate_results.py` and `scripts/analyze_results.py`:

- median: middle value or mean of the two middle values
- p95: nearest rank, `sorted[ceil(0.95·N) − 1]`
- stddev: sample (N−1), `null` for N < 2
- `pooled_within_runs`: all repetitions pooled
- `median_of_run_medians_ms`: median of per-repetition medians (distinct from pooled)

Failed samples are kept in `samples.jsonl` with `success: false` and excluded
from aggregates.

## Comparable ratios (`summary.json.ratios`)

```
host_processing_speedup_vs_scalar_cpu_ref = median(cpu_core_wall_ms[cpu-ref]) / median(gpu_transfer_inclusive_wall_ms)
host_processing_speedup_vs_cpu_omp        = median(cpu_core_wall_ms[cpu-omp]) / median(gpu_transfer_inclusive_wall_ms)
resident_processing_ratio_vs_scalar_cpu_ref = median(cpu_core_wall_ms[cpu-ref]) / median(gpu_device_pipeline_ms)
cpu_omp_speedup_vs_scalar_cpu_ref         = median(cpu-ref) / median(cpu-omp)
```

The resident ratio describes a GPU-resident use case and is not an
application speedup. Application comparisons use `application_wall_ms`
directly for each backend.

## Result directory

| File | Contents |
|---|---|
| `run.json` | run ID, UTC times, exact command, git revision + dirty flag (null with note if unavailable), status, manifest hash, run order, failures, measurement settings |
| `environment.json` | CPU, logical CPUs, RAM, OS/kernel, compiler + flags, build type, OpenCV/CMake/CUDA versions, GPU list, env settings; unknowns are null with a reason in `unknown_reasons` |
| `resolved_config.json` | full effective configuration |
| `inputs.jsonl` | frame ID, shape, format, SHA-256, timestamp |
| `validation.json` | per-backend parity vs cpu-ref with tolerances and errors; `not_executed` for unavailable backends |
| `samples.jsonl` | one line per measured sample |
| `summary.json`, `summary.csv` | aggregates and ratios |
| `analysis.md` | generated human summary, limitations, reproduction command |

Schemas: `benchmarks/schema/results_v1.schema.json`. Validate a directory with
`python3 scripts/validate_results.py DIR`. Plot with
`python3 scripts/analyze_results.py DIR [DIR...]`; it refuses `not_run`
directories (exit 2).

## Reproducing the recorded matrix

```bash
python3 scripts/generate_fixtures.py benchmark --out data/bench   # deterministic synthetic inputs
WARMUP=20 ITERATIONS=200 REPETITIONS=5 \
  benchmarks/run_matrix.sh build/cpu results/local/<new-dir> cpu-ref,cpu-omp,opencv-cpu,cuda 12
```

On a CUDA host use `build/cuda`. Profiling (not a performance result):

```bash
nsys profile --trace=cuda,nvtx --output=results/framepipe_trace \
  build/cuda/bin/framepipe benchmark --manifest data/bench/hd.jsonl --config data/bench/hd.config.json \
  --backends cuda --scopes host_processing --warmup 20 --iterations 50 --repetitions 1 \
  --output-dir results/profile_run
```
