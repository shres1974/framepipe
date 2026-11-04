// CUDA runtime/device discovery. Never assumes a toolkit implies a usable GPU.

#include <cuda_runtime.h>

#include <string>

#include "core/backend.hpp"
#include "cuda/cuda_raii.hpp"
#include "cuda/kernels.cuh"
#include "framepipe/build_info.hpp"

namespace framepipe {

CudaRuntimeReport cuda_runtime_report() {
    CudaRuntimeReport r;
    r.built = true;
    r.toolkit_version = std::to_string(CUDART_VERSION / 1000) + "." + std::to_string((CUDART_VERSION % 1000) / 10);
    r.architectures = build_info::kCudaArchitectures;
    int v = 0;
    if (cudaRuntimeGetVersion(&v) == cudaSuccess) r.runtime_version = v;
    if (cudaDriverGetVersion(&v) == cudaSuccess && v > 0) r.driver_version = v;
    int n = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        r.error = cuda::cuda_error_text(e);
        (void)cudaGetLastError();
        return r;
    }
    r.device_count = n;
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
        r.devices.push_back({i, p.name, p.major, p.minor, p.totalGlobalMem, p.multiProcessorCount});
    }
    return r;
}

BackendAvailability cuda_availability(int device_index) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        (void)cudaGetLastError();
        return {true, false, "built, but no usable CUDA device: " + cuda::cuda_error_text(e)};
    }
    if (n == 0) return {true, false, "built, but no CUDA devices are visible"};
    if (device_index < 0 || device_index >= n) {
        return {true, false,
                "device_index " + std::to_string(device_index) + " is out of range (" + std::to_string(n) +
                    " device(s) visible)"};
    }
    e = cudaSetDevice(device_index);
    if (e == cudaSuccess) e = cudaFree(nullptr);  // initialize the context
    if (e == cudaSuccess) e = cuda::probe_kernel_image();
    if (e != cudaSuccess) {
        (void)cudaGetLastError();
        return {true, false,
                "device " + std::to_string(device_index) + " is not usable by this build (architectures " +
                    build_info::kCudaArchitectures + "): " + cuda::cuda_error_text(e)};
    }
    cudaDeviceProp p{};
    (void)cudaGetDeviceProperties(&p, device_index);
    return {true, true,
            std::string("device ") + std::to_string(device_index) + " " + p.name + " (compute " +
                std::to_string(p.major) + "." + std::to_string(p.minor) + ")"};
}

}  // namespace framepipe
