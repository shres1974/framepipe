# FramePipe

A reproducible C++17 image-preprocessing pipeline: grayscale → bilinear
resize → separable Gaussian blur → Sobel edge magnitude, with a scalar CPU
reference, an OpenMP CPU backend, custom CUDA kernels, an independent NumPy
oracle, and a benchmark harness that keeps timing scopes separate.

It turns frames such as a 1920×1080 camera sequence into 1280×720 grayscale,
blurred, or normalized edge-map PNGs, in manifest order, with machine-readable
correctness and timing reports. It does not make robotics decisions.

**Status (2026-11-05):** CPU-validated and GPU-executed. On an NVIDIA Tesla T4
(Kaggle, CUDA 13.0) all 11 CUDA hardware tests pass and the CUDA output matches
the CPU reference within tolerance on every benchmarked input. HD
(1920×1080 → 1280×720) processing including host↔device transfers takes
2.24 ms vs 28.5 ms for the scalar CPU reference (12.7×). See
[docs/benchmark_results.md](docs/benchmark_results.md). Compute Sanitizer and
the oracle suite with the CUDA candidate have not been run yet; see
[docs/implementation_status.md](docs/implementation_status.md).

## Quick start (Linux, no root, no GPU required)

```bash
# 1. Tools (pinned) in a project virtualenv
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PATH="$PWD/.venv/bin:$PATH"
#    If `python3 -m venv` fails (no ensurepip), use:
#    python3 -m venv --without-pip .venv && curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python

# 2. C++ dependencies (OpenCV 4.10, GoogleTest 1.15.2, nlohmann/json 3.11.3), ~4 min
scripts/install_deps.sh                  # installs to ~/.local/framepipe-deps
export FRAMEPIPE_DEPS=$HOME/.local/framepipe-deps

# 3. Build and test (CPU only; the default build never needs CUDA)
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu

# 4. First commands
build/cpu/bin/framepipe doctor
build/cpu/bin/framepipe process \
  --input tests/fixtures/checkerboard_rgb.png \
  --output outputs/checkerboard_edges.png \
  --config configs/edges_720p.json --backend cpu-ref \
  --report outputs/checkerboard_report.json
```

`configs/edges_720p.json` is the documented default configuration
(edges, 1280×720, radius 3, sigma 1.2). The equivalent raw CMake invocation
without presets is:

```bash
cmake -S . -B build/cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFRAMEPIPE_ENABLE_CUDA=OFF -DFRAMEPIPE_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=$FRAMEPIPE_DEPS
cmake --build build/cpu --parallel && ctest --test-dir build/cpu --output-on-failure
```

## Commands

Every command supports `--help`. Diagnostics go to stderr; stdout carries
only requested JSON. Exit codes: 0 ok, 2 bad arguments/config, 3 I/O or
unsupported format, 4 backend unavailable, 5 processing/resource failure,
6 correctness failure.

| Command | Example |
|---|---|
| `doctor` | `framepipe doctor --json` — build capabilities and runtime usability, reported separately |
| `process` | `framepipe process --input in.png --output out.png [--config F] [--backend auto\|cpu-ref\|cpu-omp\|cuda] [--output-stage resize\|blur\|edges] [--width W --height H --radius R --sigma S --threads N] [--report F\|-] [--overwrite]` |
| `batch` | `framepipe batch --manifest frames.jsonl --output-dir out/ [--continue-on-error] [--queue-capacity 2]` |
| `validate` | `framepipe validate --input in.png --candidate cuda --reference cpu-ref --report v.json` or `--reference oracle:DIR` |
| `benchmark` | `framepipe benchmark --manifest m.jsonl --backends cpu-ref,cpu-omp,opencv-cpu,cuda --scopes host_processing,device_pipeline,application,startup,batch --output-dir results/local/run1` |

Configuration precedence: built-in defaults < `--config` JSON < explicit
flags. Unknown JSON fields and unsupported values are errors. Schema:
[configs/pipeline_config.schema.json](configs/pipeline_config.schema.json).

Input policy: 8-bit gray or RGB PNG/JPEG. 16-bit, float, and alpha images are
rejected (not converted). EXIF orientation is ignored; pixels are processed
in stored order. Outputs are never overwritten without `--overwrite`, and an
output path may not resolve to the input.

Batch manifests are JSON Lines, one frame per line, processed in file order:

```json
{"frame_id": "cam0_000123", "path": "frames/000123.png", "source_timestamp_ns": 1730000000000000000}
```

## CUDA build (on a machine with a supported NVIDIA GPU and driver)

```bash
cmake --preset cuda                 # CMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset cuda
build/cuda/bin/framepipe doctor     # cuda must show usable=yes
ctest --test-dir build/cuda --output-on-failure
```

Without a visible GPU, compile-only checking uses `--preset cuda-compile`
(explicit architectures 75;86;89; change them for your device).
`FRAMEPIPE_ENABLE_CUDA=ON` fails clearly if no CUDA compiler is found or no
architecture is given. Check that your toolkit supports your host compiler.

## Documentation

| Document | Answers |
|---|---|
| [docs/numerical_contract.md](docs/numerical_contract.md) | What each operator computes, exactly |
| [docs/architecture.md](docs/architecture.md) | Who owns buffers, where work happens, one frame traced end to end |
| [docs/testing.md](docs/testing.md) | Fixtures, oracle, test matrix, GPU diagnostic commands |
| [docs/benchmark_method.md](docs/benchmark_method.md) | What each metric includes and how to reproduce a run |
| [docs/benchmark_results.md](docs/benchmark_results.md) | What was measured, on which hardware, and what is still unknown |
| [docs/implementation_status.md](docs/implementation_status.md) | Milestones, requirement traceability, exact commands run, open gaps |
| [docs/decisions/](docs/decisions/) | Dependency, error-model, CLI, and scope decisions |
| [docs/interview_notes.md](docs/interview_notes.md) | Answers to the review questions in spec Section 28 |

## License

Not yet selected; to be chosen by the repository owner. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependencies.
