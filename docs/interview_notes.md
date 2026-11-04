# Technical review notes (spec Section 28)

Answers based on this implementation. Performance statements cite
[benchmark_results.md](benchmark_results.md); no GPU performance is claimed
because no GPU run exists yet.

**Why a separable Gaussian, and which intermediates does it need?**
A (2r+1)² kernel that is an outer product of two 1-D kernels can be applied
as a horizontal pass then a vertical pass: 2(2r+1) multiply-adds per pixel
instead of (2r+1)² (31·2 = 62 vs 961 at r = 15). It needs one full-resolution
FP32 intermediate (the horizontal result). FramePipe keeps it in workspace
buffer B, then writes the vertical result back into A (two-buffer schedule).
With replicated borders the separable result equals the 2-D result exactly
in real arithmetic; FP32 rounding differs slightly, which the tolerance covers.

**Half-pixel vs align-corners resizing?**
Half-pixel maps pixel *centers*: `u = (x+0.5)·Wi/Wo − 0.5`. Align-corners maps
the corner pixels onto each other: `u = x·(Wi−1)/(Wo−1)`. They give different
values everywhere except special ratios, and align-corners is undefined for
Wo = 1. The hand fixture `[[0,1],[1,0]]` → 3×3 gives `[[0,.5,1],…]` under
half-pixel; FramePipe's `ResizeTables.UpscaleNegativeFractionalCoordinateAtEdge`
locks in floor (not trunc) and the weight computed from the unclamped index.

**Why do border rules and RGB/BGR order matter for parity?**
They change real pixel values, not just edge cases. Swapping R and B maps pure
red to 29 instead of 77. A different border mode changes every pixel within r
of an edge. A CPU/GPU pair that agrees on the wrong convention passes a
differential test, which is why the oracle and hand fixtures exist (spec 16.1).
The mutation check in testing.md shows the oracle catches a Sobel sign flip,
floor→trunc, and a 1-unit grayscale weight change.

**What does a CUDA event interval measure, and what does it omit?**
The device-timeline time between two events recorded in a stream: here, from
before the grayscale kernel to after the Sobel kernel. It omits H2D/D2H
transfers, host staging, launch overhead before the first event, and anything
on other streams. It can include scheduling gaps and interference.
`gpu_device_pipeline_ms` therefore describes GPU-resident processing only and
is never reported as an application speedup.

**Why can a GPU lose to a CPU on small images?**
Fixed costs dominate: launch latency for five kernels, H2D and D2H copies,
stream synchronization, and staging copies are roughly constant per frame,
while a 64×64 frame needs only a few microseconds of CPU work. On the CPU side the
tiny scenario already takes only 0.064 ms scalar (benchmark_results.md), a
budget in which a GPU must fit five launches, two copies, and a sync; whether
it does is unmeasured until a GPU run exists. The `tiny` scenario exists to show it on a GPU once measured.

**Frame latency vs batch throughput?**
Latency is how long one frame takes from input-ready to output-ready.
Throughput is completed frames divided by the wall time of a whole batch.
When stages overlap (decode thread + process thread in `batch`), throughput
can exceed 1000/latency; it must never be derived that way. FramePipe reports
`batch_wall_ms` and `throughput_completed_fps` from completed frames only.

**When can a host buffer be reused after an asynchronous copy?**
Only after the copy has completed: after a stream or event synchronization
that follows the copy. For pinned memory `cudaMemcpyAsync` returns before the
copy finishes, so overwriting the source early corrupts the transfer. For
pageable memory the runtime may stage it and return earlier or later
depending on size. FramePipe's synchronous facade copies into pinned staging,
enqueues the copy, and only refills staging on the next call, after the
previous call's `cudaStreamSynchronize`. A multi-slot runner would also have
to wait for the downstream encoder to release the host output (spec 15).

**What went wrong during an optimization, and which evidence identified it?**
The OpenMP backend: at 18 threads on a 9-core/18-thread hybrid laptop CPU in
WSL2, the HD pipeline took ~41 ms vs ~13.6 ms single-threaded, while 12–14
threads gave ~3 ms (thread sweep in
results/evidence/2026-11-03_ultra5-135h-wsl2_omp_thread_sweep). My hypothesis
was static-chunk load imbalance across P/E cores, so I switched to 4·threads
dynamic chunks. A same-session before/after over two sessions showed medians
within noise and only a slightly better p95 tail: the hypothesis was not
supported. The cliff tracks the VM's logical-CPU count, which suggests
oversubscription that cannot be isolated without pinning or clock control.
The profile showed the bigger opportunity is the Gaussian passes (58% of
scalar time; OpenCV's single-threaded separable filter beats 12-thread
cpu-omp at r=15), i.e. SIMD, not more threads. Documented in
benchmark_results.md.

**What would change if downstream processing already consumed device memory?**
The final D2H copy, host staging, and host quantization would disappear from
the critical path; `gpu_device_pipeline_ms` would become the relevant metric.
The API would need a device-output view type (CUDA pointer + pitch + stream
or event for synchronization) and an explicit ownership handoff, which the
current host-only public API deliberately does not expose.

**Which assumptions need re-evaluation for an AMD GPU or embedded system?**
Warp (32) vs wavefront (64) size affects the 16×16 block choice; pitched
allocation alignment; pinned memory cost and limits; whether host and device
share memory (Jetson-style unified memory makes H2D copies unnecessary);
toolkit/driver/compiler support; FMA contraction defaults. The basic kernels
avoid warp-size assumptions so a HIP port can start from them, but any port
must rerun the full parity suite on the real device.

## Résumé statement status

Supported today: "Built a C++17 image-preprocessing pipeline with custom
scalar and OpenMP CPU implementations and custom CUDA resize, Gaussian, and
Sobel kernels; validated the CPU implementations against an independent
float64 NumPy oracle across 16 geometry/ratio/radius cases and all intermediate
stages, with a benchmark harness that separates device, transfer-inclusive,
and application timing."

Not supported yet: any GPU correctness or speedup claim. Fill the spec's
template only after `framepipe_cuda_tests` and a CUDA benchmark run exist on
named hardware.
