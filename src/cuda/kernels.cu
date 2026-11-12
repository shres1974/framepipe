#include "cuda/kernels.cuh"

namespace framepipe::cuda {

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;
// sqrtf(32.0f), identical to the CPU constant kEdgeNormalization.
constexpr float kEdgeNorm = 5.6568542494923801952f;

__device__ __forceinline__ int clampi(int i, int extent) { return i < 0 ? 0 : (i > extent - 1 ? extent - 1 : i); }

__device__ __forceinline__ const float* row(const DevF32C& v, int y) {
    return reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(v.data) +
                                          static_cast<std::size_t>(y) * v.pitch);
}
__device__ __forceinline__ float* row(const DevF32& v, int y) {
    return reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(v.data) + static_cast<std::size_t>(y) * v.pitch);
}

dim3 grid_for(int w, int h) {
    return dim3(static_cast<unsigned>((w + kBlockX - 1) / kBlockX), static_cast<unsigned>((h + kBlockY - 1) / kBlockY));
}

__global__ void k_grayscale(DevU8 in, DevF32 out) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= in.width || y >= in.height) return;
    const std::uint8_t* src = in.data + static_cast<std::size_t>(y) * in.pitch;
    unsigned g;
    if (in.channels == 3) {
        const unsigned r = src[3 * x], gg = src[3 * x + 1], b = src[3 * x + 2];
        g = (77u * r + 150u * gg + 29u * b + 128u) >> 8;
    } else {
        g = src[x];
    }
    row(out, y)[x] = static_cast<float>(g) / 255.0f;
}

// Grayscale one input pixel exactly as k_grayscale does, returning the [0,1]
// float. Kept in one place so the fused kernel stays bit-identical.
__device__ __forceinline__ float gray_at(const DevU8& in, int x, int y) {
    const std::uint8_t* src = in.data + static_cast<std::size_t>(y) * in.pitch;
    unsigned g;
    if (in.channels == 3) {
        const unsigned r = src[3 * x], gg = src[3 * x + 1], b = src[3 * x + 2];
        g = (77u * r + 150u * gg + 29u * b + 128u) >> 8;
    } else {
        g = src[x];
    }
    return static_cast<float>(g) / 255.0f;
}

__global__ void k_grayscale_resize(DevU8 in, DevF32 out, DevTables t) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= out.width || y >= out.height) return;
    const int y0 = t.y0[y], y1 = t.y1[y];
    const int x0 = t.x0[x], x1 = t.x1[x];
    const float ax = t.wx[x], ay = t.wy[y];
    // Grayscale the four bilinear taps on the fly, then interpolate. Identical
    // to resizing a precomputed gray image because the same gray values feed
    // the same bilinear expression.
    const float g00 = gray_at(in, x0, y0), g01 = gray_at(in, x1, y0);
    const float g10 = gray_at(in, x0, y1), g11 = gray_at(in, x1, y1);
    const float top = g00 + (g01 - g00) * ax;
    const float bottom = g10 + (g11 - g10) * ax;
    row(out, y)[x] = top + (bottom - top) * ay;
}

__global__ void k_resize(DevF32C in, DevF32 out, DevTables t) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= out.width || y >= out.height) return;
    const float* r0 = row(in, t.y0[y]);
    const float* r1 = row(in, t.y1[y]);
    const int x0 = t.x0[x], x1 = t.x1[x];
    const float ax = t.wx[x], ay = t.wy[y];
    const float top = r0[x0] + (r0[x1] - r0[x0]) * ax;
    const float bottom = r1[x0] + (r1[x1] - r1[x0]) * ax;
    row(out, y)[x] = top + (bottom - top) * ay;
}

__global__ void k_gauss_h(DevF32C in, DevF32 out, const float* __restrict__ w, int radius) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= out.width || y >= out.height) return;
    const float* src = row(in, y);
    float acc = 0.0f;
    for (int k = -radius; k <= radius; ++k) acc += w[k + radius] * src[clampi(x + k, in.width)];
    row(out, y)[x] = acc;
}

__global__ void k_gauss_v(DevF32C in, DevF32 out, const float* __restrict__ w, int radius) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= out.width || y >= out.height) return;
    float acc = 0.0f;
    for (int k = -radius; k <= radius; ++k) acc += w[k + radius] * row(in, clampi(y + k, in.height))[x];
    row(out, y)[x] = acc;
}

__device__ __forceinline__ void sobel_at(const DevF32C& in, int x, int y, float& gx, float& gy) {
    const int xm = clampi(x - 1, in.width), xp = clampi(x + 1, in.width);
    const float* rm = row(in, clampi(y - 1, in.height));
    const float* r0 = row(in, y);
    const float* rp = row(in, clampi(y + 1, in.height));
    gx = (rm[xp] - rm[xm]) + 2.0f * (r0[xp] - r0[xm]) + (rp[xp] - rp[xm]);
    gy = (rp[xm] - rm[xm]) + 2.0f * (rp[x] - rm[x]) + (rp[xp] - rm[xp]);
}

__global__ void k_sobel_edges(DevF32C in, DevF32 out) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= out.width || y >= out.height) return;
    float gx, gy;
    sobel_at(in, x, y, gx, gy);
    const float e = sqrtf(gx * gx + gy * gy) / kEdgeNorm;
    row(out, y)[x] = fminf(fmaxf(e, 0.0f), 1.0f);
}

__global__ void k_sobel_components(DevF32C in, DevF32 gxo, DevF32 gyo) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= gxo.width || y >= gxo.height) return;
    float gx, gy;
    sobel_at(in, x, y, gx, gy);
    row(gxo, y)[x] = gx;
    row(gyo, y)[x] = gy;
}

__global__ void k_probe() {}

}  // namespace

cudaError_t launch_grayscale(DevU8 in, DevF32 out, cudaStream_t s) {
    k_grayscale<<<grid_for(in.width, in.height), dim3(kBlockX, kBlockY), 0, s>>>(in, out);
    return cudaGetLastError();
}

cudaError_t launch_resize(DevF32C in, DevF32 out, DevTables t, cudaStream_t s) {
    k_resize<<<grid_for(out.width, out.height), dim3(kBlockX, kBlockY), 0, s>>>(in, out, t);
    return cudaGetLastError();
}

cudaError_t launch_grayscale_resize(DevU8 in, DevF32 out, DevTables t, cudaStream_t s) {
    k_grayscale_resize<<<grid_for(out.width, out.height), dim3(kBlockX, kBlockY), 0, s>>>(in, out, t);
    return cudaGetLastError();
}

cudaError_t launch_gaussian_horizontal(DevF32C in, DevF32 out, const float* w, int radius, cudaStream_t s) {
    k_gauss_h<<<grid_for(out.width, out.height), dim3(kBlockX, kBlockY), 0, s>>>(in, out, w, radius);
    return cudaGetLastError();
}

cudaError_t launch_gaussian_vertical(DevF32C in, DevF32 out, const float* w, int radius, cudaStream_t s) {
    k_gauss_v<<<grid_for(out.width, out.height), dim3(kBlockX, kBlockY), 0, s>>>(in, out, w, radius);
    return cudaGetLastError();
}

cudaError_t launch_sobel_edges(DevF32C in, DevF32 out, cudaStream_t s) {
    k_sobel_edges<<<grid_for(out.width, out.height), dim3(kBlockX, kBlockY), 0, s>>>(in, out);
    return cudaGetLastError();
}

cudaError_t launch_sobel_components(DevF32C in, DevF32 gx, DevF32 gy, cudaStream_t s) {
    k_sobel_components<<<grid_for(gx.width, gx.height), dim3(kBlockX, kBlockY), 0, s>>>(in, gx, gy);
    return cudaGetLastError();
}

cudaError_t probe_kernel_image() {
    cudaFuncAttributes attr{};
    return cudaFuncGetAttributes(&attr, k_probe);
}

}  // namespace framepipe::cuda
