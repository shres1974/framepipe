# 0001 Dependencies and toolchain

Status: accepted, 2026-11-03

## Decision

| Need | Choice | Pinned version | Why |
|---|---|---|---|
| Build | CMake (>= 3.24 required) + Ninja | 3.30.5, 1.11.1.1 (pip, project `.venv`) | Host had no CMake; pip wheels need no root |
| Image I/O + labeled baseline | OpenCV core, imgproc, imgcodecs | 4.10.0, built from source by `scripts/install_deps.sh` with bundled libpng/libjpeg/zlib | Host had no OpenCV; minimal build avoids GUI/video deps |
| Tests | GoogleTest | 1.15.2 (source, SHA-256 pinned) | Spec allows GoogleTest or Catch2; GoogleTest chosen and used everywhere |
| JSON | nlohmann/json | 3.11.3 (source, SHA-256 pinned) | Header-only, ordered output, strict parse errors |
| CLI parsing | Small in-repo parser (`src/app/cli_args.cpp`) | — | See 0003 |
| Oracle/analysis | Python 3.12, NumPy 2.1.3, Matplotlib 3.9.2 | `requirements.txt` | Independent float64 oracle and figures |
| CUDA (compile-only here) | NVIDIA redistributable nvcc/cudart/cccl | 12.6.85 / 12.6.77 | GCC 13.3 is a supported host compiler for CUDA 12.6 |
| Parallel CPU | OpenMP (GCC libgomp) | OpenMP 4.5 | Standard; serial backend still builds without it |

All source archives are verified by SHA-256 in `scripts/install_deps.sh`.
Builds never download anything; dependencies are found through
`CMAKE_PREFIX_PATH` (presets read `$FRAMEPIPE_DEPS`). System packages
(`libopencv-dev`, `libgtest-dev`, `nlohmann-json3-dev`) are an equivalent
alternative when versions satisfy the `find_package` minimums.

## Consequences

- A CPU-only developer needs no GPU packages.
- The CUDA toolkit used here was installed in user space only to prove the
  CUDA target compiles. This WSL2 host exposes no NVIDIA driver (`libcuda.so`
  absent from `/usr/lib/wsl/lib`), so GPU execution is untested.
