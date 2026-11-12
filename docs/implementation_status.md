# Implementation status

Last updated: 2026-11-05. **Release state: CPU-validated and GPU-validated
on a Tesla T4.** The CUDA backend passes its hardware tests, the independent
NumPy oracle suite, and all four Compute Sanitizer tools.

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
| Usable NVIDIA GPU | **No** on this host. `/dev/dxg` exists but there is no `libcuda.so` in `/usr/lib/wsl/lib` and no `nvidia-smi`. The runtime reports `cudaErrorInsufficientDriver`. | GPU validation ran on Kaggle instead (see below) |
| perf | 6.18 (linux-tools) | used for the CPU profile |

## Milestones

| Milestone | State | Evidence |
|---|---|---|
| M0 skeleton, CPU-only build, doctor | **passed** | CPU configure/build with no CUDA; `doctor` reports `cuda: built=no` on the CPU build and exits 0 |
| M1 CPU math + independent oracle | **passed** | 84 unit tests; oracle self-tests; 96 oracle comparisons (cpu-ref, cpu-omp × 16 cases × 3 stages); mutation check |
| M2 I/O + CLI | **passed** | `cli_integration`, `ImageIo.*`; README commands executed verbatim |
| M3 basic CUDA parity | **passed on Tesla T4, including Compute Sanitizer** | On Kaggle T4 (nvcc 13.0.88, `native` arch) all 11 hardware GPU tests pass (parity, launch boundaries, resize ratios, Gaussian radii, cropped views, pinned/pageable transfers, device-resident path, zero steady-state device allocation, re-prepare, budget and device-index errors). The one skip is the no-device test, as expected. Compute Sanitizer memcheck, racecheck, initcheck, and synccheck report 0 errors. Logs: `results/evidence/2026-11-05_kaggle-t4_gpu_{matrix,verify}/logs/` |
| M4 benchmarks + CPU baseline | **passed (CPU and GPU)** | Full spec 18.2 matrix on the laptop CPU (51 dirs). On the T4: 7 scenarios + radius and stage sweeps, repeat workload only (14 dirs), with every backend parity-checked per input. All 65 dirs validate. See benchmark_results.md |
| M5 profiling + one optimization | **passed (CPU, inconclusive result documented); GPU optimizations measured on branches** | perf profile; OpenMP scheduling before/after; the static variant is retained; see benchmark_results.md. Three optimization branches (`opt/simd-gaussian`, `opt/fuse-grayscale-resize`, `opt/cuda-overlap`) pass parity, oracle, and sanitizer on a T4; the CUDA ones are within noise end-to-end (benchmark_results.md, "Optimization work") |
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
| R06 custom CUDA | passed (T4) | `src/cuda/*.cu`; `CudaParity.*` pass on hardware |
| R07 resource lifetime | passed (CPU and T4) | `Lifecycle.*` (fault injection, steady-state zero allocations, re-prepare); `CudaLifecycle.*` pass on hardware |
| R08 backend selection | passed | auto→cpu-ref with reason; explicit cuda → exit 4, no output; wrong device index |
| R09 CLI workflow | passed | `cli_integration` (all commands, exit codes 0/2/3/4/5/6) |
| R10 float correctness on recorded hardware | passed (CPU and T4) | oracle suite (on T4: 144 comparisons covering cpu-ref, cpu-omp, and cuda, 0 failures); validation.json in every benchmark, including T4 runs |
| R11 timing boundaries | passed (code review + tests) | docs/benchmark_method.md; harness tests ensure no CPU `device_pipeline` samples |
| R12 reproducibility | passed | run/environment/config/inputs/samples in every run. The laptop CPU runs have a null git revision (no commits existed). The T4 runs record `fd080d4`, clean |
| R13 fair CPU baseline | passed | cpu-omp (thread count recorded) and an OpenCV baseline with matched semantics |
| R14 measured GPU optimization | **measured, null result** | two CUDA optimizations on branches, correct on T4, within noise end-to-end because the workload is I/O-bound; see benchmark_results.md |
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

## GPU validation commands run (2026-11-05, Kaggle Tesla T4) and results

```bash
scripts/install_deps.sh                                                   # 360 s
cmake --preset cuda && cmake --build --preset cuda -j $(nproc)            # nvcc 13.0.88, GCC 13.3.0
build/cuda/bin/framepipe doctor --json                                    # cuda usable, 2× Tesla T4 (sm_75)
ctest --test-dir build/cuda -L gpu -V                                     # 12/12: 11 passed, 1 skipped (no-device test)
ctest --test-dir build/cuda --output-on-failure                           # 124/125; Harness.CpuRunProducesAllArtifacts failed (test defect, since fixed)
python3 scripts/generate_fixtures.py benchmark --out data/bench
WORKLOADS=repeat WARMUP=20 ITERATIONS=100 REPETITIONS=3 \
  benchmarks/run_matrix.sh build/cuda <out> cpu-ref,cpu-omp,opencv-cpu,cuda 4
for d in <out>/*/; do python3 scripts/validate_results.py $d; done        # 14 valid
```

Verification run at `19ce81f` (`results/evidence/2026-11-05_kaggle-t4_gpu_verify/logs/`):

```bash
ctest --test-dir build/cuda --output-on-failure                           # 125/125 passed (harness fix confirmed on GPU)
build/cuda/bin/framepipe_cuda_tests                                       # 11 passed, 1 skipped (no-device test)
python3 scripts/run_oracle_suite.py --framepipe build/cuda/bin/framepipe --work-dir build/cuda/ow --candidates cpu-ref,cpu-omp,cuda
                                                                          # 144 comparisons, 0 failures
for t in memcheck racecheck initcheck synccheck; do compute-sanitizer --tool $t --error-exitcode 99 build/cuda/bin/framepipe_cuda_tests; done
                                                                          # all four: 11 tests passed, 0 errors / 0 hazards
```

## Remaining gaps and next concrete steps

1. Run the full ring workload matrix on GPU (only HD ring was measured, during the optimization work).
2. Rerun one CPU benchmark on the laptop so `git_revision` is recorded.
3. Paste the full spec text into `docs/implementation_spec.md`.
4. Profile with Nsight Systems. The measured GPU bottleneck is transfers plus host I/O, not kernels, so a shared-memory Gaussian tile is low priority.
5. Re-time `opt/simd-gaussian` on the laptop and merge it if the gain holds.
6. Overlap decode, process, and encode in `batch`: the measured end-to-end bottleneck.
7. P2 items not started: video adapter, Python/ROS binding, HIP, pilot user.

## Known limitations

- Downscaling aliases (no prefilter; the spec defines it this way).
- `Pipeline` is not thread-safe per instance.
- Timings come from a laptop VM without clock control. Variation between runs is up to ~30%.
- The CI workflow (`.github/workflows/ci.yml`) runs on GitHub and is green on
  `main` (CPU, oracle, benchmark-schema, and CUDA-compile jobs). The GPU
  execution job needs a self-hosted runner and is gated off.
