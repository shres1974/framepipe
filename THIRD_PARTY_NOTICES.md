# Third-party notices

FramePipe links or uses the following third-party software. None is
vendored in this repository; `scripts/install_deps.sh` downloads pinned
source archives and verifies their SHA-256.

| Component | Version | License | Use |
|---|---|---|---|
| OpenCV (core, imgproc, imgcodecs) | 4.10.0 | Apache-2.0 | Image decode/encode; separately labeled CPU baseline |
| libpng, libjpeg-turbo, zlib (bundled by the OpenCV build) | as bundled by OpenCV 4.10.0 | libpng / IJG + BSD-3 / zlib | PNG and JPEG codecs |
| nlohmann/json | 3.11.3 | MIT | JSON configuration and reports |
| GoogleTest | 1.15.2 | BSD-3-Clause | Tests only |
| NumPy | 2.1.3 | BSD-3-Clause | Oracle and analysis scripts only |
| Matplotlib | 3.9.2 | Matplotlib (PSF-based) | Figures only |
| CMake, Ninja (pip wheels) | 3.30.5, 1.11.1.1 | BSD-3-Clause, Apache-2.0 | Build tools only |
| NVIDIA CUDA Toolkit (nvcc, cudart, CCCL) | 12.6 | NVIDIA CUDA EULA | Optional CUDA backend; not redistributed |
| OpenMP runtime (GCC libgomp) | GCC 13 | GPL-3.0 with GCC Runtime Library Exception | Optional cpu-omp backend |

## Datasets

All test fixtures (`tests/fixtures/`) and benchmark inputs (`data/bench/`,
generated, not committed) are synthetic, produced by
`scripts/generate_fixtures.py` with recorded seeds (see each directory's
`metadata.json`). `tests/fixtures/exif_orientation6.jpg` was encoded with
OpenCV and tagged by `scripts/make_exif_fixture.py`. No third-party or
personal imagery is included.
