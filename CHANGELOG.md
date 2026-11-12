# Changelog

All notable changes to FramePipe. Versions follow [Semantic Versioning](https://semver.org/).

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
- End-to-end throughput is bound by PNG decode and encode, not by processing.
