#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#include "framepipe/errors.hpp"
#include "framepipe/image.hpp"

namespace framepipe {

namespace {
std::atomic<std::size_t> g_live_bytes{0};
std::atomic<std::size_t> g_live_count{0};
std::atomic<long> g_fail_countdown{0};
constexpr std::size_t kAlign = 64;
}  // namespace

std::size_t tracked_host_live_bytes() noexcept { return g_live_bytes.load(); }
std::size_t tracked_host_allocation_count() noexcept { return g_live_count.load(); }

namespace testing {
void fail_host_allocation_after(long n) noexcept { g_fail_countdown.store(n); }
}  // namespace testing

HostBuffer::HostBuffer(std::size_t bytes) {
    if (bytes == 0) return;
    long c = g_fail_countdown.load();
    if (c > 0) {
        if (g_fail_countdown.fetch_sub(1) == 1) {
            throw Error(ErrorCategory::AllocationFailure, "injected host allocation failure (test hook)");
        }
    }
    const std::size_t rounded = checked_add(bytes, kAlign - 1, "host allocation") / kAlign * kAlign;
    void* p = std::aligned_alloc(kAlign, rounded);
    if (p == nullptr) {
        throw Error(ErrorCategory::AllocationFailure, "host allocation of " + std::to_string(bytes) + " bytes failed");
    }
    data_ = static_cast<unsigned char*>(p);
    size_ = bytes;
    g_live_bytes.fetch_add(size_);
    g_live_count.fetch_add(1);
}

HostBuffer::~HostBuffer() { reset(); }

void HostBuffer::reset() noexcept {
    if (data_ != nullptr) {
        std::free(data_);
        g_live_bytes.fetch_sub(size_);
        g_live_count.fetch_sub(1);
    }
    data_ = nullptr;
    size_ = 0;
}

HostBuffer::HostBuffer(HostBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

HostBuffer& HostBuffer::operator=(HostBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

HostFloatImage::HostFloatImage(int width, int height, std::size_t pad_elements)
    : HostFloatImage(width, height, pad_elements, std::numeric_limits<float>::quiet_NaN()) {}

HostFloatImage::HostFloatImage(int width, int height, std::size_t pad_elements, float fill) {
    validate_dimensions(width, height, kHardMaxDimension, "float image");
    const std::size_t elems_per_row = checked_add(static_cast<std::size_t>(width), pad_elements, "row elements");
    stride_bytes_ = checked_mul(elems_per_row, sizeof(float), "row stride");
    const std::size_t total = checked_mul(stride_bytes_, static_cast<std::size_t>(height), "image bytes");
    buffer_ = HostBuffer(total);
    width_ = width;
    height_ = height;
    float* p = reinterpret_cast<float*>(buffer_.data());
    const std::size_t n = total / sizeof(float);
    for (std::size_t i = 0; i < n; ++i) p[i] = fill;
}

FloatImageView HostFloatImage::view() noexcept {
    return {reinterpret_cast<float*>(buffer_.data()), width_, height_, stride_bytes_, buffer_.size()};
}

ConstFloatImageView HostFloatImage::cview() const noexcept {
    return {reinterpret_cast<const float*>(buffer_.data()), width_, height_, stride_bytes_, buffer_.size()};
}

float HostFloatImage::at(int x, int y) const noexcept { return row_ptr(cview(), y)[x]; }

float& HostFloatImage::at(int x, int y) noexcept {
    return reinterpret_cast<float*>(buffer_.data() + static_cast<std::size_t>(y) * stride_bytes_)[x];
}

HostU8Image::HostU8Image(int width, int height, PixelFormat format, std::size_t pad_bytes, std::uint8_t fill) {
    validate_dimensions(width, height, kHardMaxDimension, "u8 image");
    const std::size_t row =
        checked_mul(static_cast<std::size_t>(width), static_cast<std::size_t>(channel_count(format)), "row bytes");
    stride_bytes_ = checked_add(row, pad_bytes, "row stride");
    const std::size_t total = checked_mul(stride_bytes_, static_cast<std::size_t>(height), "image bytes");
    buffer_ = HostBuffer(total);
    std::memset(buffer_.data(), fill, total);
    width_ = width;
    height_ = height;
    format_ = format;
}

InputImageView HostU8Image::view() const noexcept {
    return {buffer_.data(), width_, height_, stride_bytes_, buffer_.size(), format_};
}

}  // namespace framepipe
