# 0004 Scope decisions and deviations

Status: accepted, 2026-11-03

1. **No shared-memory CUDA variant shipped.** Spec 12.6/20 allow optimization
   only after the basic kernels pass parity and profiling on hardware. No GPU
   was available, so `cuda.implementation` accepts only `basic`. An unvalidated
   variant would be a placeholder presented as a feature.

2. **OpenCV baseline is a benchmark-only workload** (`--backends opencv-cpu`),
   not a selectable processing backend. It reuses FramePipe's integer grayscale
   (shared input adaptation, included in its timing) and FramePipe's FP32
   Gaussian weights, with `INTER_LINEAR`, `BORDER_REPLICATE`, `Sobel ksize=3`,
   `CV_32F`. The harness validates it against cpu-ref per input and records
   `matched_semantics` true/false; its timings are labeled accordingly.

3. **Benchmark `--workload`**: `repeat` (each manifest frame measured
   repeatedly, labeled as such) and `ring` (cycle through same-shape frames).
   Both are required by spec 18.1.

4. **Timer-resolution guard.** Samples at or below timer resolution are counted
   in `summary.json.zero_duration_samples` instead of being dropped; none were
   observed in recorded runs.

5. **Batch scope in the benchmark** includes read, decode, prepare-if-needed,
   process, quantize, and PNG encode/write into a scratch file that is deleted
   after the run. Warm-up is excluded, as is `startup_wall_ms` (measured
   separately by constructing and preparing a fresh pipeline).

6. **cpu-omp chunking.** Rows are split into `4 × threads` chunks with dynamic
   scheduling (see `docs/benchmark_results.md`, optimization log). Each row is
   still computed by one thread with the scalar operator code, so results are
   bit-identical to cpu-ref (tested).

7. **License:** MIT, selected by the repository owner on 2026-11-11 (spec Section 9).
   All dependencies are compatible (see `THIRD_PARTY_NOTICES.md`).
