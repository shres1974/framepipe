# Implementation status

Last updated: 2026-11-03. **Release state: a CPU-validated implementation,
with GPU validation pending.** This is not a complete P0 release. P0 requires
CUDA execution and parity on real hardware, and that has not happened.

## Environment facts (development host)

| Item | Found | Notes |
|---|---|---|
| OS | Ubuntu 24.04.3 LTS on WSL2, kernel 6.18.54.1-microsoft-standard-WSL2 | |
| CPU | Intel Core Ultra 5 135H, 18 logical CPUs | |
| Compiler | GCC 13.3.0 | clang not installed |
| CMake / Ninja | none on the system; installed 3.30.5 / 1.11.1.1 via pip into `.venv` | no root access |
| OpenCV | none on the system; built 4.10.0 from source (`scripts/install_deps.sh`) into `~/.local/framepipe-deps` | |
| GoogleTest, nlohmann/json | built 1.15.2 / 3.11.3 from SHA-256-pinned sources | |
| Python / NumPy / Matplotlib | 3.12.3 / 2.1.3 / 3.9.2 (venv) | system `python3-venv` lacked ensurepip; bootstrapped pip |
| OpenMP | libgomp, OpenMP 4.5 | |
| CUDA toolkit | none on the system; NVIDIA redistributable nvcc 12.6.85 + cudart/cccl 12.6.77 installed in `~/.local/cuda-12.6` (SHA-256 verified) | compile-only |
| Usable NVIDIA GPU | **No.** `/dev/dxg` exists but there is no `libcuda.so` in `/usr/lib/wsl/lib` and no `nvidia-smi`. The runtime reports `cudaErrorInsufficientDriver`. | Having a toolkit did not mean having a GPU |
| perf | 6.18 (linux-tools) | used for the CPU profile |

## Milestones

| Milestone | State | Evidence |
|---|---|---|
| M0 skeleton, CPU-only build, doctor | **passed** | CPU configure/build with no CUDA; `doctor` reports `cuda: built=no` on the CPU build and exits 0 |
| M1 CPU math + independent oracle | **passed** | 84 unit tests; oracle self-tests; 96 oracle comparisons (cpu-ref, cpu-omp × 16 cases × 3 stages); mutation check |
| M2 I/O + CLI | **passed** | `cli_integration`, `ImageIo.*`; README commands executed verbatim |
| M3 basic CUDA parity | **open: hardware gate not run** | Kernels, RAII owners, stream/events, pitched workspace, pinned staging, and 12 GPU tests are written. CUDA build compiles for sm_75/86/89 with zero warnings (`--Werror=all-warnings`). Of the 12 GPU tests, 11 hardware tests are **skipped**. Only the no-device check `CudaErrors.ExplicitCudaUnavailableNeverFallsBack` executes, and it passes. Compute Sanitizer **not run**. |
| M4 benchmarks + CPU baseline | **passed (CPU)**; GPU part not run | Full spec 18.2 matrix measured on CPU; cpu-omp and a validated OpenCV baseline; schema validation and recompute pass on 51 result dirs; GPU scopes present in code but unmeasured |
| M5 profiling + one optimization | **passed (CPU, inconclusive result documented)** | perf profile; OpenMP scheduling before/after; the static variant is retained; see benchmark_results.md. No GPU optimization (not permitted before M3) |
| M6 sequences + pilot | **sequence runner passed; pilot not started** | `batch` with bounded queue, ordering, fail-fast/continue, re-prepare, memory-bound test. **No pilot user**, and no usage is claimed. Async multi-slot CUDA execution and video: not implemented |
| M7 AMD HIP | **not started** | `hip` is not accepted or advertised anywhere |

## Requirements traceability (R01–R17)

| ID | Status | Evidence |
|---|---|---|
| R01 CPU build without CUDA | passed | `cmake --preset cpu`; no CUDA headers in `include/` or in CPU targets |
| R02 formats, layout | passed | `ImageView.*`, padded/cropped tests, NaN padding canaries, `ImageIo.RejectsAlpha16BitAndCorrupt` |
| R03 grayscale | passed | exhaustive 2^24 test, primary colors, BGR→RGB tests (C++ and CLI) |
| R04 half-pixel resize | passed | hand fixtures, table tests (floor, negative fractional edge coordinates), oracle |
| R05 Gaussian + Sobel | passed | constant, impulse, ramp, and border tests; signed components compared against the oracle |
| R06 custom CUDA | **implemented, execution unverified** | `src/cuda/*.cu`; parity tests exist but are skipped |
| R07 resource lifetime | passed on CPU; GPU tests pending | `Lifecycle.*` (fault injection, steady-state zero allocations, re-prepare); `CudaLifecycle.*` skipped |
| R08 backend selection | passed | auto→cpu-ref with reason; explicit cuda → exit 4, no output; wrong device index |
| R09 CLI workflow | passed | `cli_integration` (all commands, exit codes 0/2/3/4/5/6) |
| R10 float correctness on recorded hardware | passed for CPU backends; **GPU not run** | oracle suite; validation.json in every benchmark |
| R11 timing boundaries | passed (code review + tests) | docs/benchmark_method.md; harness tests ensure no CPU `device_pipeline` samples |
| R12 reproducibility | passed | run/environment/config/inputs/samples in every run. Git revision was **null** (no commits existed). Rerun after the first commit to record a revision |
| R13 fair CPU baseline | passed | cpu-omp (thread count recorded) and an OpenCV baseline with matched semantics |
| R14 measured GPU optimization | **not done** | needs M3 first |
| R15 bounded sequences | passed | `Batch.*`, `BoundedQueue.*` |
| R16 transparent status | passed | this file; ctest shows GPU skips; summaries record `unavailable` backends |
| R17 AMD backend | not requested / not started | — |

## Exact validation commands run (2026-11-03) and results

```bash
export PATH=$PWD/.venv/bin:$PATH FRAMEPIPE_DEPS=$HOME/.local/framepipe-deps
cmake --preset cpu && cmake --build --preset cpu && ctest --preset cpu            # 113/113 passed (1 skip: CpuParallel.UnavailableIsExplicit, expected because OpenMP is present)
cmake --preset cpu-debug && cmake --build --preset cpu-debug && ctest --preset cpu-debug        # 113/113 passed
cmake --preset cpu-sanitize && cmake --build --preset cpu-sanitize && ctest --preset cpu-sanitize  # 111/111 passed under ASan+UBSan (oracle label excluded)
PATH=$HOME/.local/cuda-12.6/bin:$PATH cmake --preset cuda-compile -DCUDAToolkit_ROOT=$HOME/.local/cuda-12.6
cmake --build --preset cuda-compile                                                # 0 warnings, 0 errors
ctest --test-dir build/cuda-compile                                                # 125/125 "passed": 11 GPU hardware tests SKIPPED (no device); 1 no-device test passed
WARMUP=20 ITERATIONS=200 REPETITIONS=5 benchmarks/run_matrix.sh build/cpu results/local/2026-11-03_cpu_matrix cpu-ref,cpu-omp,opencv-cpu,cuda 12
for d in results/evidence/*/*/; do python3 scripts/validate_results.py $d; done   # 51 valid
```

Mutation check (run manually, code restored afterward): Sobel-y sign flip →
10 oracle failures; floor→trunc → 9; green weight 150→151 → 36.

## Remaining gaps and next concrete steps

1. **Run on an NVIDIA GPU (closes M3, R06, R07-GPU, R10-GPU):**
   ```bash
   scripts/install_deps.sh && cmake --preset cuda && cmake --build --preset cuda
   build/cuda/bin/framepipe doctor                                    # must show cuda usable=yes
   build/cuda/bin/framepipe_cuda_tests                               # must show 0 SKIPPED
   python3 scripts/run_oracle_suite.py --framepipe build/cuda/bin/framepipe --work-dir build/cuda/ow --candidates cpu-ref,cpu-omp,cuda
   for t in memcheck racecheck initcheck synccheck; do compute-sanitizer --tool $t --error-exitcode 99 build/cuda/bin/framepipe_cuda_tests; done
   ```
   Then run `benchmarks/run_matrix.sh build/cuda results/local/<new>` and update benchmark_results.md.
   On WSL2 this requires the Windows NVIDIA driver with WSL support, so that `/usr/lib/wsl/lib/libcuda.so` exists.
2. Make a first git commit, then rerun one benchmark so `git_revision` is recorded.
3. Choose a LICENSE (owner decision).
4. Paste the full spec text into `docs/implementation_spec.md`.
5. After GPU parity passes: profile with Nsight Systems, then try one GPU optimization (a shared-memory Gaussian tile is the obvious candidate) with before/after evidence (R14).
6. CPU follow-up suggested by the data: SIMD in the Gaussian passes.
7. P2 items not started: async multi-slot CUDA runner, video adapter, Python/ROS binding, HIP, pilot user.

## Known limitations

- Downscaling aliases (no prefilter; the spec defines it this way).
- `Pipeline` is not thread-safe per instance.
- Timings come from a laptop VM without clock control. Variation between runs is up to ~30%.
- The CI workflow (`.github/workflows/ci.yml`) has not been run on GitHub. The GPU job needs a self-hosted runner.
