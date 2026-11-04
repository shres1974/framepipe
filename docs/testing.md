# Testing

Three independent kinds of evidence (spec 16.1), plus lifecycle, I/O, CLI,
and benchmark-honesty tests.

| Evidence | Where | What it can catch that others cannot |
|---|---|---|
| Hand-calculated fixtures | `tests/unit/test_{grayscale,tables,resize,gaussian,sobel,pipeline}.cpp` | A bug shared by the C++ code and the oracle |
| Independent NumPy oracle | `scripts/oracle.py`, driven by `scripts/run_oracle_suite.py` | General behavior across shapes/ratios/radii; shares no code with C++ |
| Backend differential | `test_parallel.cpp` (cpu-omp vs cpu-ref, bit-exact), `tests/gpu/test_cuda_parity.cpp` (CUDA vs cpu-ref, per stage) | Backend-specific indexing, stride, launch-boundary bugs |

## Running

```bash
ctest --preset cpu                    # everything available in the CPU build
ctest --preset cpu -L unit            # C++ unit tests
ctest --preset cpu -L oracle          # oracle self-tests + C++ vs oracle (needs NumPy; skip code 77 otherwise)
ctest --preset cpu -L integration     # I/O, manifest, batch, harness, CLI, result schema
ctest --preset cpu-sanitize           # ASan + UBSan build (oracle excluded for speed)
ctest --test-dir build/cuda -L gpu    # CUDA tests; each SKIPS with a reason if no device
```

## Hand fixtures (spec 16.4) and where they are checked

| Fixture | C++ test | Oracle self-test |
|---|---|---|
| Gray8 1×1 = 128 stays 128/255 through resize and blur; edges 0 | `Pipeline.HandFixtureGray1x1` | `test_oracle.py` |
| Red 77, green 149, blue 29, white 255, black 0 | `Grayscale.PrimaryColorsExact` (+ exhaustive 2^24) | yes |
| `[[0,1],[1,0]]` → 3×3 `[[0,.5,1],[.5,.5,.5],[1,.5,0]]`; → 1×1 = 0.5 | `Resize.HandFixture2x2To3x3/To1x1` | yes |
| Ramp `x/(W-1)`: gx = 8/(W-1), gy = 0, edge = √2/(W-1) | `Sobel.HorizontalRampHandFixture` | yes |
| Constant stays constant for r ∈ {0,1,3,15} incl. corners | `Gaussian.ConstantStaysConstantIncludingCorners` | yes |

Additional exact checks: negative fractional coordinates at an edge
(`ResizeTables.UpscaleNegativeFractionalCoordinateAtEdge`), floor vs
truncation, impulse response = outer product of weights, Sobel correlation
signs, replicated borders at a step edge, identity resize is bit-exact.

## Required test matrix (spec 16.3) coverage

| Category | Cases | Tests |
|---|---|---|
| Small geometry | 1×1, 1×7, 7×1, 2×2, 3×3 | RandomMatrix, oracle suite, CudaParity |
| Launch boundaries | 15×17, 16×16, 17×19, 31×33, 257×129 | same |
| Resize ratios | identity, up, down, non-integer, 1×1 output | ResizeTables, RandomMatrix, oracle, CudaParity |
| Intensities | 0, 255, 128 constant | RandomMatrix.IntensityExtremes, CudaParity |
| Patterns | ramps, impulse, checkerboard, stripes | Sobel/Gaussian units, oracle cases |
| Color | pure RGB, mixed, channel-order | Grayscale units, ImageIo.BgrIsReorderedToRgb, CLI RGB-order check |
| Layout | padded rows, cropped views, varied float strides | every operator test pads; `Pipeline.CroppedInputViewMatchesCopiedImage`; NaN canaries in padding |
| Gaussian | r 0,1,3,7,15; sigma 0.1..100 | GaussianWeights, RandomMatrix, oracle, CudaParity |
| Random | fixed seeds, logged on failure | RandomMatrix (seed printed in assertion), oracle suite (seed in case table) |
| Invalid requests | zero dims, bad stride, overflow, bad enum, NaN sigma, unknown fields | ImageView, Config, CLI |
| Runtime | no device, wrong device index, budget exceeded | Pipeline.ExplicitCudaNeverFallsBack, WrongDeviceIndexIsReported, CudaErrors.* |
| Lifecycle | re-prepare, repeated calls, independent instances, failure mid-allocation, destruction after failure | Lifecycle.*, CudaLifecycle.* |

## Oracle suite details

`run_oracle_suite.py` generates 16 deterministic inputs (geometry, ratio, and
radius matrix above), runs the oracle for each of the three output stages,
then runs `framepipe validate --reference oracle:DIR` for each usable
candidate (`cpu-ref`, `cpu-omp --threads 3`, add `cuda` with
`--candidates cpu-ref,cpu-omp,cuda`). Every stage, including signed Sobel
components, must pass the normative tolerances. 96 comparisons run on a CPU
build.

Mutation check (performed 2026-11-03, see implementation_status.md): flipping
the Sobel-y sign, replacing `floor` with `trunc` in the resize table, and
changing the green weight from 150 to 151 each made the oracle suite fail
(10, 9, and 36 failing comparisons respectively), and the unmodified code passes.

## GPU diagnostics (requires a real NVIDIA device)

Not executed on the development host (no NVIDIA driver). Commands to run:

```bash
cmake --preset cuda && cmake --build --preset cuda
ctest --test-dir build/cuda --output-on-failure             # gpu label must show 0 skipped
python3 scripts/run_oracle_suite.py --framepipe build/cuda/bin/framepipe \
    --work-dir build/cuda/oracle_work --candidates cpu-ref,cuda
for tool in memcheck racecheck initcheck synccheck; do
  compute-sanitizer --tool $tool --error-exitcode 99 build/cuda/bin/framepipe_cuda_tests || echo "$tool FAILED"
done
```

Do not collect performance data from sanitizer runs.
