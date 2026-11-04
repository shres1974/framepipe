# Architecture

FramePipe is one static library set plus one executable. There are no
services or processes; the diagram in spec Section 5 shows ownership only.

```
CLI (src/app) ──> config resolve (src/core/config.cpp)
   │
   ├─ io: read bytes -> decode -> BGR->RGB canonical HostU8Image   (src/io/image_io.cpp)
   ├─ Pipeline (src/core/pipeline.cpp)  — validates views, owns one BackendImpl
   │     ├─ CpuBackend  "cpu-ref" (scalar) / "cpu-omp" (row-parallel)   (src/cpu)
   │     └─ CudaBackend "cuda" (custom kernels, one stream)              (src/cuda)
   ├─ io: quantize -> encode PNG
   └─ validation (src/validation) / benchmark (src/benchmark)
```

## Module boundaries

| Module | Owns | Does not |
|---|---|---|
| `include/framepipe` | Public API: views, config, errors, `Pipeline`. No CUDA headers. | — |
| `src/core` | Geometry checks, tables, weights, backend dispatch, SHA-256, tracked host allocation | call CUDA |
| `src/cpu` | `cpu_ops.cpp`: every operator as a function of an output-row range. `cpu_backend.cpp`: workspace. `cpu_parallel.cpp`: OpenMP row partitioning (only compiled with OpenMP). | use library kernels |
| `src/cuda` | RAII owners (`cuda_raii.hpp`), kernels (`kernels.cu`), backend (`cuda_backend.cu`), discovery (`cuda_info.cu`) | own codec policy |
| `src/io` | OpenCV decode/encode, quantization, manifests, JSON file helpers | hide processing in I/O timing |
| `src/app` | Argument parsing, commands, exit codes, ordered batch runner | re-implement math |
| `src/validation` | Per-stage comparison and diagnostics | relax thresholds |
| `src/benchmark` | Harness, statistics, environment, labeled OpenCV baseline | synthesize missing data |
| `scripts` | Oracle, fixture generator, result validator, analysis | be required at runtime |

The backend is chosen once in the `Pipeline` constructor. No pixel loop
branches on backend.

## Ownership of one frame (`framepipe process`, CUDA backend)

1. `io::read_file_bytes` → `std::string` (owned by the command).
2. `io::decode_image` → `cv::Mat` (temporary) → `HostU8Image` (packed RGB/Gray, 64-byte aligned, owned by the command). BGR→RGB happens here.
3. `Pipeline::prepare(spec)` → `CudaBackend::prepare`:
   - drains the owned stream, releases the previous workspace;
   - preflights the estimated workspace against `max_device_bytes` and `cudaMemGetInfo` (minus 64 MiB headroom);
   - allocates into locals, then commits: pitched `d_in` (u8), `d_gray` (f32, input size), two output-size f32 buffers `d_a`/`d_b`, one table buffer (x0,x1,wx,y0,y1,wy,gaussian weights), and pinned staging in/out if within `max_pinned_host_bytes`;
   - uploads tables synchronously (outside any steady-state timing).
   A throw at any step leaves the backend unprepared and frees what was allocated (locals' destructors).
4. `Pipeline::process_into(input_view, output_view)` validates both views (dimensions, stride, alignment, extent, no overlap) and calls the backend:
   - host copy of caller rows (honoring caller stride) into pinned staging, `cudaMemcpy2DAsync` H2D on the owned stream;
   - kernels in order on the same stream: grayscale → resize (→ `d_a`) → horizontal blur (`d_a`→`d_b`) → vertical blur (`d_b`→`d_a`) → Sobel (`d_a`→`d_b`). Each launch is checked immediately with `cudaGetLastError`;
   - `cudaMemcpy2DAsync` D2H into pinned staging, `cudaStreamSynchronize` (execution errors surface here, tagged with the stage), then host copy into the caller's strided view.
   On any error the stream is drained, the error is rethrown with stage/shape context, and the caller's output is never reported as valid.
5. `io::quantize_image` + `io::write_png_gray` on the host.

With the CPU backend steps 3–4 use `HostFloatImage` workspace (`gray`, `A`, `B`) and the same two-buffer schedule.

## Workspace and lifetime rules

- Persistent workspace is allocated only in `prepare`. A unit test asserts zero host allocations across 200 steady-state frames; a GPU test asserts stable device bytes across 300 frames.
- Re-preparing replaces the workspace (no per-shape cache).
- Tables and weights are per instance (no process-wide constant memory), so independent instances with different parameters cannot interfere (tested on CPU and GPU).
- `Pipeline` is move-only and **not thread-safe**; use one instance per caller.
- Destructors never throw. `CudaBackend` synchronizes its stream before releasing buffers.

Device memory estimate for the default two-buffer CUDA layout, ignoring pitch:
`3·Ni (RGB in) + 4·Ni (gray) + 8·No (two FP32 buffers) + 12·(Wo+Ho) + 4·(2r+1)`.
`Pipeline::workspace()` reports logical and actually allocated (pitched) bytes;
benchmark `summary.json` records both.

## Sequence runner (`framepipe batch`)

One decoder thread reads and decodes frames into a `BoundedQueue` (default
capacity 2). The calling thread processes, quantizes, and writes frames in
sequence order. `push` blocks when the queue is full, so no frames are
dropped and retained decoded frames are bounded by capacity + 2. Fail-fast by
default: on the first failure the queue is closed and cleared, the producer
is joined, remaining frames are recorded `not_processed`. `--continue-on-error`
records every failure and excludes it from throughput. A shape change calls
`prepare` (synchronous, so prior work is complete). Output names are
`<frame_id>.png`; frame IDs are restricted to `[A-Za-z0-9._-]` and cannot
start with `.`, so they cannot escape the output directory.

## Error model

Typed exceptions (`framepipe::Error`) carry a category, message, and context
(operation, backend, shape, underlying runtime detail). The CLI maps categories
to exit codes: 2 invalid arguments/config, 3 I/O or unsupported format,
4 backend unavailable, 5 processing/resource failure, 6 correctness failure.
See [decisions/0002-error-model.md](decisions/0002-error-model.md).
