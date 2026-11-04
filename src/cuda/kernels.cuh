#pragma once
// Private CUDA kernels (spec Section 12.1). One logical output pixel per
// thread, 16x16 blocks, every thread bounds-checks its output coordinate, all
// neighborhood reads use replicated borders, and all row addressing is done in
// bytes with the allocation pitch before casting to the element type.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace framepipe::cuda {

struct DevU8 {
    const std::uint8_t* data;
    int width;
    int height;
    std::size_t pitch;
    int channels;
};

struct DevF32 {
    float* data;
    int width;
    int height;
    std::size_t pitch;
};

struct DevF32C {
    const float* data;
    int width;
    int height;
    std::size_t pitch;
};

inline DevF32C as_const(const DevF32& v) { return {v.data, v.width, v.height, v.pitch}; }

struct DevTables {
    const std::int32_t* x0;
    const std::int32_t* x1;
    const float* wx;
    const std::int32_t* y0;
    const std::int32_t* y1;
    const float* wy;
};

// Each launcher returns the launch-configuration error (cudaGetLastError)
// immediately after the launch. Execution errors surface at the stream
// completion boundary and are checked by the caller.
cudaError_t launch_grayscale(DevU8 in, DevF32 out, cudaStream_t s);
cudaError_t launch_resize(DevF32C in, DevF32 out, DevTables t, cudaStream_t s);
cudaError_t launch_gaussian_horizontal(DevF32C in, DevF32 out, const float* weights, int radius, cudaStream_t s);
cudaError_t launch_gaussian_vertical(DevF32C in, DevF32 out, const float* weights, int radius, cudaStream_t s);
cudaError_t launch_sobel_edges(DevF32C in, DevF32 out, cudaStream_t s);
cudaError_t launch_sobel_components(DevF32C in, DevF32 gx, DevF32 gy, cudaStream_t s);

// Used by availability probing: fails with cudaErrorNoKernelImageForDevice if
// this binary has no compatible code for the current device.
cudaError_t probe_kernel_image();

}  // namespace framepipe::cuda
