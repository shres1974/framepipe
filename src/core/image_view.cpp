#include "framepipe/image_view.hpp"

#include <cstdint>
#include <limits>

#include "framepipe/errors.hpp"

namespace framepipe {

const char* to_string(PixelFormat f) noexcept {
    switch (f) {
        case PixelFormat::Gray8: return "gray8";
        case PixelFormat::Rgb8: return "rgb8";
    }
    return "unknown";
}

int channel_count(PixelFormat f) noexcept { return f == PixelFormat::Rgb8 ? 3 : 1; }

bool operator==(const InputSpec& a, const InputSpec& b) noexcept {
    return a.width == b.width && a.height == b.height && a.format == b.format;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw Error(ErrorCategory::InvalidImage, std::string("size overflow computing ") + what);
    }
    return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* what) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw Error(ErrorCategory::InvalidImage, std::string("size overflow computing ") + what);
    }
    return a + b;
}

std::size_t required_extent_bytes(int width, int height, std::size_t bytes_per_pixel, std::size_t stride_bytes) {
    if (width <= 0 || height <= 0) {
        throw Error(ErrorCategory::InvalidImage, "dimensions must be positive");
    }
    const std::size_t row = checked_mul(static_cast<std::size_t>(width), bytes_per_pixel, "row bytes");
    const std::size_t before_last = checked_mul(static_cast<std::size_t>(height) - 1, stride_bytes, "image extent");
    return checked_add(before_last, row, "image extent");
}

std::string shape_string(int width, int height) { return std::to_string(width) + "x" + std::to_string(height); }

std::string shape_string(const InputSpec& s) { return shape_string(s.width, s.height) + " " + to_string(s.format); }

void validate_dimensions(int width, int height, int max_dimension, const char* what) {
    if (width <= 0 || height <= 0) {
        throw Error(ErrorCategory::InvalidImage,
                    std::string(what) + " dimensions must be positive, got " + shape_string(width, height));
    }
    const int limit = max_dimension < kHardMaxDimension ? max_dimension : kHardMaxDimension;
    if (width > limit || height > limit) {
        throw Error(ErrorCategory::InvalidImage, std::string(what) + " dimensions " + shape_string(width, height) +
                                                     " exceed the maximum of " + std::to_string(limit) +
                                                     " pixels per dimension");
    }
}

namespace {
void validate_common(const void* data, int width, int height, std::size_t stride_bytes, std::size_t extent_bytes,
                     std::size_t bytes_per_pixel, std::size_t alignment, int max_dimension, const char* what) {
    validate_dimensions(width, height, max_dimension, what);
    if (data == nullptr) {
        throw Error(ErrorCategory::InvalidImage, std::string(what) + " data pointer is null");
    }
    const std::size_t row = checked_mul(static_cast<std::size_t>(width), bytes_per_pixel, "row bytes");
    if (stride_bytes < row) {
        throw Error(ErrorCategory::InvalidImage, std::string(what) + " stride_bytes " + std::to_string(stride_bytes) +
                                                     " is smaller than the logical row size " + std::to_string(row));
    }
    if (alignment > 1) {
        if (stride_bytes % alignment != 0) {
            throw Error(ErrorCategory::InvalidImage, std::string(what) + " stride_bytes " +
                                                         std::to_string(stride_bytes) + " is not a multiple of " +
                                                         std::to_string(alignment));
        }
        if (reinterpret_cast<std::uintptr_t>(data) % alignment != 0) {
            throw Error(ErrorCategory::InvalidImage, std::string(what) + " data pointer is not suitably aligned");
        }
    }
    const std::size_t needed = required_extent_bytes(width, height, bytes_per_pixel, stride_bytes);
    if (extent_bytes < needed) {
        throw Error(ErrorCategory::InvalidImage, std::string(what) + " extent_bytes " + std::to_string(extent_bytes) +
                                                     " cannot contain the last logical pixel (needs " +
                                                     std::to_string(needed) + ")");
    }
}
}  // namespace

void validate_view(const InputImageView& v, int max_dimension) {
    if (v.format != PixelFormat::Gray8 && v.format != PixelFormat::Rgb8) {
        throw Error(ErrorCategory::UnsupportedFormat, "input pixel format must be gray8 or rgb8");
    }
    validate_common(v.data, v.width, v.height, v.stride_bytes, v.extent_bytes,
                    static_cast<std::size_t>(channel_count(v.format)), 1, max_dimension, "input image");
}

void validate_view(const FloatImageView& v, int max_dimension) {
    validate_common(v.data, v.width, v.height, v.stride_bytes, v.extent_bytes, sizeof(float), sizeof(float),
                    max_dimension, "float image");
}

void validate_view(const ConstFloatImageView& v, int max_dimension) {
    validate_common(v.data, v.width, v.height, v.stride_bytes, v.extent_bytes, sizeof(float), sizeof(float),
                    max_dimension, "float image");
}

bool byte_ranges_overlap(const void* a, std::size_t an, const void* b, std::size_t bn) noexcept {
    if (an == 0 || bn == 0) return false;
    const auto pa = reinterpret_cast<std::uintptr_t>(a);
    const auto pb = reinterpret_cast<std::uintptr_t>(b);
    return pa < pb + bn && pb < pa + an;
}

}  // namespace framepipe
