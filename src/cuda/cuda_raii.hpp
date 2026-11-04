#pragma once
// Private move-only CUDA resource owners (spec 12.2, 12.4). Destructors never
// throw; cleanup is best-effort so an original error is preserved.

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <utility>

#include "framepipe/errors.hpp"

namespace framepipe::cuda {

inline std::string cuda_error_text(cudaError_t e) {
    return std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e);
}

// Throws a typed Error. Out-of-memory becomes AllocationFailure (never an
// empty image); everything else is DeviceFailure. `stage` names the operation.
inline void check(cudaError_t e, const char* stage, const std::string& shape = "") {
    if (e == cudaSuccess) return;
    const ErrorCategory cat = (e == cudaErrorMemoryAllocation) ? ErrorCategory::AllocationFailure
                                                                : ErrorCategory::DeviceFailure;
    throw Error(cat, std::string("CUDA operation failed during ") + stage, {stage, "cuda", shape, cuda_error_text(e)});
}

class DeviceImage {
public:
    DeviceImage() noexcept = default;
    DeviceImage(int width, int height, std::size_t elem_bytes, const char* what) {
        void* p = nullptr;
        std::size_t pitch = 0;
        check(cudaMallocPitch(&p, &pitch, static_cast<std::size_t>(width) * elem_bytes, static_cast<std::size_t>(height)),
              what);
        ptr_ = p;
        width_ = width;
        height_ = height;
        elem_ = elem_bytes;
        pitch_ = pitch;
        bytes_ = pitch * static_cast<std::size_t>(height);
    }
    ~DeviceImage() { reset(); }
    DeviceImage(DeviceImage&& o) noexcept { *this = std::move(o); }
    DeviceImage& operator=(DeviceImage&& o) noexcept {
        if (this != &o) {
            reset();
            std::swap(ptr_, o.ptr_);
            width_ = o.width_;
            height_ = o.height_;
            elem_ = o.elem_;
            pitch_ = o.pitch_;
            bytes_ = o.bytes_;
            o.bytes_ = 0;
        }
        return *this;
    }
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;
    void reset() noexcept {
        if (ptr_) (void)cudaFree(ptr_);
        ptr_ = nullptr;
        bytes_ = 0;
    }
    void* ptr() const noexcept { return ptr_; }
    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    std::size_t pitch() const noexcept { return pitch_; }
    std::size_t row_bytes() const noexcept { return static_cast<std::size_t>(width_) * elem_; }
    std::size_t logical_bytes() const noexcept { return row_bytes() * static_cast<std::size_t>(height_); }
    std::size_t allocated_bytes() const noexcept { return bytes_; }

private:
    void* ptr_ = nullptr;
    int width_ = 0, height_ = 0;
    std::size_t elem_ = 0, pitch_ = 0, bytes_ = 0;
};

class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    DeviceBuffer(std::size_t bytes, const char* what) {
        void* p = nullptr;
        check(cudaMalloc(&p, bytes), what);
        ptr_ = p;
        bytes_ = bytes;
    }
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(DeviceBuffer&& o) noexcept { *this = std::move(o); }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            std::swap(ptr_, o.ptr_);
            bytes_ = o.bytes_;
            o.bytes_ = 0;
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void reset() noexcept {
        if (ptr_) (void)cudaFree(ptr_);
        ptr_ = nullptr;
        bytes_ = 0;
    }
    template <class T>
    T* as() const noexcept {
        return static_cast<T*>(ptr_);
    }
    std::size_t size() const noexcept { return bytes_; }

private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

class PinnedBuffer {
public:
    PinnedBuffer() noexcept = default;
    PinnedBuffer(std::size_t bytes, const char* what) {
        void* p = nullptr;
        check(cudaHostAlloc(&p, bytes, cudaHostAllocDefault), what);
        ptr_ = p;
        bytes_ = bytes;
    }
    ~PinnedBuffer() { reset(); }
    PinnedBuffer(PinnedBuffer&& o) noexcept { *this = std::move(o); }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            std::swap(ptr_, o.ptr_);
            bytes_ = o.bytes_;
            o.bytes_ = 0;
        }
        return *this;
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    void reset() noexcept {
        if (ptr_) (void)cudaFreeHost(ptr_);
        ptr_ = nullptr;
        bytes_ = 0;
    }
    unsigned char* data() const noexcept { return static_cast<unsigned char*>(ptr_); }
    std::size_t size() const noexcept { return bytes_; }

private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

class Stream {
public:
    Stream() noexcept = default;
    explicit Stream(const char* what) { check(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking), what); }
    ~Stream() { reset(); }
    Stream(Stream&& o) noexcept { std::swap(s_, o.s_); }
    Stream& operator=(Stream&& o) noexcept {
        if (this != &o) {
            reset();
            std::swap(s_, o.s_);
        }
        return *this;
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    void reset() noexcept {
        if (s_) {
            (void)cudaStreamSynchronize(s_);  // complete owned work before release
            (void)cudaStreamDestroy(s_);
        }
        s_ = nullptr;
    }
    cudaStream_t get() const noexcept { return s_; }

private:
    cudaStream_t s_ = nullptr;
};

class Event {
public:
    Event() noexcept = default;
    explicit Event(const char* what) { check(cudaEventCreate(&e_), what); }
    ~Event() { reset(); }
    Event(Event&& o) noexcept { std::swap(e_, o.e_); }
    Event& operator=(Event&& o) noexcept {
        if (this != &o) {
            reset();
            std::swap(e_, o.e_);
        }
        return *this;
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    void reset() noexcept {
        if (e_) (void)cudaEventDestroy(e_);
        e_ = nullptr;
    }
    cudaEvent_t get() const noexcept { return e_; }

private:
    cudaEvent_t e_ = nullptr;
};

}  // namespace framepipe::cuda
