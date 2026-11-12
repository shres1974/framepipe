# Changelog

All notable changes to FramePipe. Versions follow [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-11

### Added
- `framepipe compare`: one command that runs every backend on an image and
  prints median time, speedup, and max error vs cpu-ref under the normative
  tolerance. `--output-dir` writes each output plus a tolerance-scaled
  error map; `--report` writes JSON.
- `batch --decode-threads N` and `--sync-write`. A bounded reorder window
  (`OrderedWindow`) keeps output order and memory bounded with several
  decoders.
- `examples/defect_inspection/`: synthetic labeled surfaces, an edge-based
  crack/scratch/pit detector calibrated on a train split, and a summary
  figure. 96.7% held-out accuracy, with both mistakes shown.
- Showcase: batch-throughput chart, `make_showcase.py scene`, and notebook
  cells for `compare` and the example on the GPU.

### Changed
- `batch` now decodes in parallel and writes on a dedicated thread by
  default: 3.6× (HD) and 3.4× (4K) more frames per second on the laptop,
  with byte-identical outputs. `--decode-threads 1 --sync-write` restores
  the serial path. The batch report records `decode_threads` and
  `async_write`.
- cpu-ref and cpu-omp Gaussian passes are vectorized (`opt/simd-gaussian`
  merged): 2× faster at radius 3, ~3.4× at radius 15, byte-identical output.

### Verification
- New tests: `OrderedWindow.*`, parallel batch byte-for-byte vs serial,
  async write-failure policies, `error_map`, and `compare` in the CLI suite.
  Batch tests also pass under ThreadSanitizer and ASan+UBSan.
- CI runs the defect-inspection example end to end.

## [0.1.0] - 2026-11-11

First public version, released under the MIT license.

### Pipeline
- Grayscale → bilinear half-pixel resize → separable Gaussian blur → Sobel
  edge magnitude, defined exactly in `docs/numerical_contract.md`.
- Backends: `cpu-ref` (scalar reference), `cpu-omp` (OpenMP, byte-identical to
  `cpu-ref`), and `cuda` (custom kernels, pinned staging, zero steady-state
  allocation). `opencv-cpu` is available as a benchmark-only baseline with
  matched semantics.
- `auto` backend selection records its reason. An explicit backend that is
  unavailable fails with exit code 4 and never falls back.

### CLI
- `doctor`, `process`, `batch` (bounded queue, manifest order,
  fail-fast or `--continue-on-error`), `validate` (against a backend or the
  NumPy oracle), and `benchmark` (separate timing scopes, schema-validated
  results).

### Verification
- Independent NumPy oracle, mutation-checked; 144 comparisons pass on a Tesla T4.
- 11 CUDA hardware tests and Compute Sanitizer (memcheck, racecheck,
  initcheck, synccheck) pass on a Tesla T4.
- CI: Debug, Release, ASan+UBSan, oracle, benchmark-schema, and CUDA-compile jobs.

### Evidence
- CPU matrix (Intel Core Ultra 5 135H, WSL2) and GPU matrix (Tesla T4,
  Kaggle) committed under `results/evidence/`.

### Showcase and tooling
- `scripts/make_showcase.py` generates the README hero image (by running the
  real CLI) and charts (from committed evidence).
- `Dockerfile` (CPU; the test suite gates the image build).
- `notebooks/reproduce_gpu.ipynb` reproduces the GPU results on a Kaggle or Colab T4.
- `docs/demo.tape` renders the terminal demo with VHS.
- `docs/writeup.md`: a write-up of what the measurements show.

### Known limitations
- Downscaling aliases (no prefilter, per the contract).
- `Pipeline` instances are not thread-safe.
- End-to-end throughput is bound by PNG decode and encode, not by processing
  (addressed in 0.2.0 by overlapping I/O in `batch`).
