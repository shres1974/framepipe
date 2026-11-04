#pragma once
// Non-owning image views and checked geometry helpers (spec Sections 6 and 10).
//
// Views carry an explicit extent_bytes so validation can verify that the
// declared geometry fits inside the memory the owner actually provided. A raw
// pointer alone cannot establish allocation size.

#include <cstddef>
#include <cstdint>
#include <string>

namespace framepipe {

// Implementation hard ceiling per dimension. Configuration may lower it.
constexpr int kHardMaxDimension = 16384;

enum class PixelFormat { Gray8, Rgb8 };

const char* to_string(PixelFormat f) noexcept;
int channel_count(PixelFormat f) noexcept;

struct InputSpec {
    int width = 0;
    int height = 0;
    PixelFormat format = PixelFormat::Gray8;
};

bool operator==(const InputSpec& a, const InputSpec& b) noexcept;
inline bool operator!=(const InputSpec& a, const InputSpec& b) noexcept { return !(a == b); }

// Read-only 8-bit input. Rows may be padded: stride_bytes >= width * channels.
struct InputImageView {
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    std::size_t stride_bytes = 0;
    std::size_t extent_bytes = 0;
    PixelFormat format = PixelFormat::Gray8;
};

// Writable single-channel FP32 image. stride_bytes is in BYTES, must be a
// multiple of sizeof(float), and data must be float-aligned.
struct FloatImageView {
    float* data = nullptr;
    int width = 0;
    int height = 0;
    std::size_t stride_bytes = 0;
    std::size_t extent_bytes = 0;
};

struct ConstFloatImageView {
    const float* data = nullptr;
    int width = 0;
    int height = 0;
    std::size_t stride_bytes = 0;
    std::size_t extent_bytes = 0;
};

inline ConstFloatImageView as_const(const FloatImageView& v) noexcept {
    return {v.data, v.width, v.height, v.stride_bytes, v.extent_bytes};
}

inline InputSpec spec_of(const InputImageView& v) noexcept { return {v.width, v.height, v.format}; }

// Checked size_t arithmetic; throws Error(InvalidImage) naming `what` on overflow.
std::size_t checked_mul(std::size_t a, std::size_t b, const char* what);
std::size_t checked_add(std::size_t a, std::size_t b, const char* what);

// Minimum bytes needed for a view: stride*(h-1) + row_bytes. Throws on overflow.
std::size_t required_extent_bytes(int width, int height, std::size_t bytes_per_pixel,
                                  std::size_t stride_bytes);

// Validation throws Error(InvalidImage) with a specific message.
void validate_dimensions(int width, int height, int max_dimension, const char* what);
void validate_view(const InputImageView& v, int max_dimension = kHardMaxDimension);
void validate_view(const FloatImageView& v, int max_dimension = kHardMaxDimension);
void validate_view(const ConstFloatImageView& v, int max_dimension = kHardMaxDimension);

// True if the half-open byte ranges [a, a+an) and [b, b+bn) intersect.
bool byte_ranges_overlap(const void* a, std::size_t an, const void* b, std::size_t bn) noexcept;

std::string shape_string(int width, int height) ;
std::string shape_string(const InputSpec& s);

// Row addressing: offsets are computed in bytes before casting (spec 12.2).
inline const std::uint8_t* row_ptr(const InputImageView& v, int y) noexcept {
    return v.data + static_cast<std::size_t>(y) * v.stride_bytes;
}
inline float* row_ptr(const FloatImageView& v, int y) noexcept {
    return reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(v.data) +
                                    static_cast<std::size_t>(y) * v.stride_bytes);
}
inline const float* row_ptr(const ConstFloatImageView& v, int y) noexcept {
    return reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(v.data) +
                                          static_cast<std::size_t>(y) * v.stride_bytes);
}

}  // namespace framepipe
