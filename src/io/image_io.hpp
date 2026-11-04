#pragma once
// OpenCV I/O adapter (spec 6.1, 7.6).
//
// Decode policy: cv::imdecode with IMREAD_UNCHANGED | IMREAD_IGNORE_ORIENTATION
// so channel count and bit depth are preserved and EXIF orientation is NOT
// applied. Accepted: 8-bit 1-channel (Gray8) and 8-bit 3-channel (decoded as
// BGR by OpenCV, reordered here to RGB). Everything else is rejected with a
// specific UnsupportedFormat message. Embedded color profiles are ignored.

#include <string>

#include "framepipe/image.hpp"
#include "framepipe/image_view.hpp"

namespace framepipe::io {

struct DecodedImage {
    HostU8Image image;          // canonical Gray8 or RGB8, packed rows
    std::string source_format;  // e.g. "8-bit 3-channel (BGR from decoder)"
};

// Reads and decodes a file. Throws Error(IoFailure) for unreadable/corrupt
// files and Error(UnsupportedFormat) for disallowed formats.
DecodedImage read_image(const std::string& path, int max_dimension = kHardMaxDimension);

// Canonicalization only (from an already-read encoded byte buffer); exposed
// for testing and for separating read and decode in application timing.
std::string read_file_bytes(const std::string& path);
DecodedImage decode_image(const std::string& bytes, const std::string& label, int max_dimension = kHardMaxDimension);

// Host-side quantization: uint8(floor(clamp(v, 0, 1) * 255 + 0.5)).
std::uint8_t quantize(float v) noexcept;
HostU8Image quantize_image(const ConstFloatImageView& v);

// Writes a Gray8 image as PNG. Throws Error(IoFailure).
void write_png_gray(const std::string& path, const HostU8Image& img);

// Writes an RGB8 or Gray8 image to path using the extension's codec (testing helper).
void write_image(const std::string& path, const HostU8Image& img);

}  // namespace framepipe::io
