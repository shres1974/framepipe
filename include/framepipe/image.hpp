#pragma once
// Move-only owners of host image memory. Allocation is tracked so lifecycle
// tests can observe retained application-owned bytes (spec Section 13).

#include <cstddef>
#include <cstdint>

#include "framepipe/image_view.hpp"

namespace framepipe {

// 64-byte aligned, move-only host allocation. Throws Error(AllocationFailure).
class HostBuffer {
public:
    HostBuffer() noexcept = default;
    explicit HostBuffer(std::size_t bytes);
    ~HostBuffer();
    HostBuffer(HostBuffer&& other) noexcept;
    HostBuffer& operator=(HostBuffer&& other) noexcept;
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    unsigned char* data() noexcept { return data_; }
    const unsigned char* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    void reset() noexcept;

private:
    unsigned char* data_ = nullptr;
    std::size_t size_ = 0;
};

// Owned single-channel FP32 image. pad_elements adds per-row padding. The whole
// allocation (including padding) is filled with `fill` (quiet NaN by default)
// so that any read of an unproduced pixel becomes a detectable non-finite value.
class HostFloatImage {
public:
    HostFloatImage() noexcept = default;
    HostFloatImage(int width, int height, std::size_t pad_elements = 0);
    HostFloatImage(int width, int height, std::size_t pad_elements, float fill);

    FloatImageView view() noexcept;
    ConstFloatImageView cview() const noexcept;
    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    std::size_t stride_bytes() const noexcept { return stride_bytes_; }
    std::size_t allocated_bytes() const noexcept { return buffer_.size(); }
    float at(int x, int y) const noexcept;
    float& at(int x, int y) noexcept;
    float* raw() noexcept { return reinterpret_cast<float*>(buffer_.data()); }
    bool empty() const noexcept { return buffer_.size() == 0; }

private:
    HostBuffer buffer_;
    int width_ = 0;
    int height_ = 0;
    std::size_t stride_bytes_ = 0;
};

// Owned 8-bit Gray8 or RGB8 image with optional row padding (in bytes).
class HostU8Image {
public:
    HostU8Image() noexcept = default;
    HostU8Image(int width, int height, PixelFormat format, std::size_t pad_bytes = 0,
                std::uint8_t fill = 0);

    InputImageView view() const noexcept;
    std::uint8_t* row(int y) noexcept { return buffer_.data() + static_cast<std::size_t>(y) * stride_bytes_; }
    const std::uint8_t* row(int y) const noexcept {
        return buffer_.data() + static_cast<std::size_t>(y) * stride_bytes_;
    }
    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    PixelFormat format() const noexcept { return format_; }
    std::size_t stride_bytes() const noexcept { return stride_bytes_; }
    std::size_t allocated_bytes() const noexcept { return buffer_.size(); }

private:
    HostBuffer buffer_;
    int width_ = 0;
    int height_ = 0;
    PixelFormat format_ = PixelFormat::Gray8;
    std::size_t stride_bytes_ = 0;
};

// Live bytes currently held by HostBuffer instances in this process.
std::size_t tracked_host_live_bytes() noexcept;
std::size_t tracked_host_allocation_count() noexcept;

namespace testing {
// Fault injection for lifecycle tests: the Nth subsequent HostBuffer
// allocation (1-based) throws AllocationFailure. Pass 0 to disable.
void fail_host_allocation_after(long n) noexcept;
}  // namespace testing

}  // namespace framepipe
