# Numerical contract

This is the normative definition of what every FramePipe backend computes
(spec Sections 6, 7, 16.2). Each operator lists the source that implements
it so a reviewer can audit the mapping directly.

## Input and canonicalization

| Rule | Implementation |
|---|---|
| Accept 8-bit 1-channel (Gray8) and 8-bit 3-channel (RGB8) only | `src/io/image_io.cpp` `decode_image` |
| Reject 16-bit, float, 2-channel, 4-channel with a specific `unsupported_format` message; never drop alpha or reduce depth | same |
| Decode flags `IMREAD_UNCHANGED \| IMREAD_IGNORE_ORIENTATION`: EXIF orientation is **ignored**, stored pixel order is preserved | same; fixture `tests/fixtures/exif_orientation6.jpg` |
| OpenCV decodes BGR; the adapter reorders to RGB (cost belongs to canonicalization) | same |
| Embedded color profiles are not applied; values are stored display-encoded channel values | same |
| Rows may be padded; `stride_bytes` is bytes between row starts; views carry `extent_bytes` | `include/framepipe/image_view.hpp` |
| Per-dimension hard maximum 16384; `resources.max_dimension` may lower it | `validate_dimensions` |
| All byte counts use checked `size_t` arithmetic before allocation | `checked_mul`, `checked_add`, `required_extent_bytes` |

## Border rule

`clamp_index(i, n) = min(max(i, 0), n - 1)`, applied independently to x and y,
for Gaussian, Sobel, and resize sample indices. `src/core/tables.hpp`,
`src/cpu/cpu_ops.cpp` (`clampi`), `src/cuda/kernels.cu` (`clampi`).

## Operators (pipeline order)

`output_stage` selects `resize`, `blur`, or `edges`; only the needed prefix runs.

### 1. Grayscale (spec 7.2) — `cpu::gray_u8`, `k_grayscale`

```
gray_u8  = (77*R + 150*G + 29*B + 128) >> 8      (uint32; RGB input)
gray_u8  = input                                  (Gray8 input)
gray_f32 = float(gray_u8) / 255.0f
```
Exact: black 0, white 255, red 77, green 149, blue 29. Verified exhaustively
over all 2^24 RGB triples (`Grayscale.ExhaustiveIntegerConvention`).

### 2. Bilinear half-pixel resize (spec 7.3) — `build_axis_table`, `cpu::resize_bilinear`, `k_resize`

```
u  = (double(x) + 0.5) * double(Wi) / double(Wo) - 0.5
x0_raw = floor(u)                     # floor, never truncation
ax = float(u - x0_raw)                # weight from the UNCLAMPED raw index
x0 = clamp_index(x0_raw, Wi); x1 = clamp_index(x0_raw + 1, Wi)
(same for y)
top    = p(y0,x0) + (p(y0,x1) - p(y0,x0)) * ax
bottom = p(y1,x0) + (p(y1,x1) - p(y1,x0)) * ax
out    = top + (bottom - top) * ay
```
Index/weight tables are built once on the host in `prepare()` and the same
int32/FP32 values are used by CPU and CUDA. Default aspect policy is
`stretch`; no other policy exists. Resizing is not an antialiasing prefilter:
large downscales can alias (documented limitation).

### 3. Separable Gaussian (spec 7.4) — `build_gaussian_weights`, `cpu::gaussian_*`, `k_gauss_*`

Ranges: `0 <= radius <= 15`, `0.1 <= sigma <= 100` (finite). Defaults r=3, sigma=1.2.
```
unnormalized[k] = exp(-(k*k) / (2*sigma*sigma))      (double)
weight[k]       = unnormalized[k] / sum(unnormalized[-r..r]),  then one conversion to FP32
horizontal(y,x) = sum_k weight[k] * resized(y, clamp(x+k))      (k increasing)
blurred(y,x)    = sum_k weight[k] * horizontal(clamp(y+k), x)
```
No renormalization at borders; no 8-bit intermediates; passes never alias.

### 4. Sobel and edge normalization (spec 7.5) — `cpu::sobel_*`, `k_sobel_*`

Correlation (no kernel flip):
```
Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]   Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
edge = clamp(sqrt(gx*gx + gy*gy) / sqrt(32.0f), 0, 1)
```
`sqrt(32)` is a fixed bound (max |component| = 4 for inputs in [0,1]); it is
not per-frame normalization. The validation path (`process_capture_stages`)
exposes signed gx/gy so sign errors are detectable.

### 5. Output quantization (spec 7.6) — `io::quantize`

```
output_u8 = uint8(floor(clamp(v, 0, 1) * 255.0f + 0.5f))
```
Host-side for every backend; outside core-processing comparisons, inside
application timing. Float outputs are validated before quantization.

## Floating point

Normal FP32, no fast-math. C++ is built with `CMAKE_CXX_EXTENSIONS=OFF`
(ISO mode), which on GCC disables FMA contraction in the CPU reference.
nvcc's default `--fmad=true` is left on; tolerances absorb the difference.
The oracle (`scripts/oracle.py`) computes in float64.

## Acceptance thresholds (spec 16.2)

Per pixel: `abs(c - r) <= atol + rtol * abs(r)`. Every logical pixel must
pass; NaN/Inf always fails; padding is excluded from numeric comparison and
checked by separate NaN-canary tests.

| Quantity | atol | rtol | Source |
|---|---|---|---|
| Integer grayscale | 0 | 0 | exhaustive unit test |
| Normalized grayscale | 1e-7 | 1e-6 | `tolerance_for(Stage::Gray)` |
| Bilinear resize | 2e-6 | 2e-6 | `Stage::Resized` |
| Each Gaussian pass | 1e-5 | 1e-5 | `Stage::BlurHorizontal/BlurVertical` |
| Signed Sobel x, y | 2e-5 | 1e-5 | `Stage::SobelX/SobelY` |
| Normalized edges (also in [0,1]) | 1e-5 | 1e-5 | `Stage::Edges` |
| Saved 8-bit output | 1 level | 0 | `validate` supplemental check |

Thresholds are fixed in code and never relaxed automatically. Reports include
max abs error, mean abs error, RMSE, failing-pixel count, location and values
of the largest error, and `psnr_db` (null with `exact_match: true` when MSE = 0).
