#include "io/image_io.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "framepipe/errors.hpp"

namespace framepipe::io {

std::string read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error(ErrorCategory::IoFailure, "cannot open input file: " + path);
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) throw Error(ErrorCategory::IoFailure, "read failed: " + path);
    if (data.empty()) throw Error(ErrorCategory::IoFailure, "input file is empty: " + path);
    return data;
}

namespace {
std::string depth_name(int depth) {
    switch (depth) {
        case CV_8U: return "8-bit unsigned";
        case CV_8S: return "8-bit signed";
        case CV_16U: return "16-bit unsigned";
        case CV_16S: return "16-bit signed";
        case CV_32S: return "32-bit integer";
        case CV_32F: return "32-bit float";
        case CV_64F: return "64-bit float";
        default: return "unknown depth";
    }
}
}  // namespace

DecodedImage decode_image(const std::string& bytes, const std::string& label, int max_dimension) {
    const cv::Mat buf(1, static_cast<int>(bytes.size()), CV_8U, const_cast<char*>(bytes.data()));
    cv::Mat m;
    try {
        m = cv::imdecode(buf, cv::IMREAD_UNCHANGED | cv::IMREAD_IGNORE_ORIENTATION);
    } catch (const cv::Exception& e) {
        throw Error(ErrorCategory::IoFailure, "failed to decode image: " + label, {"decode", "", "", e.what()});
    }
    if (m.empty()) {
        throw Error(ErrorCategory::IoFailure,
                    "failed to decode image (corrupt or unsupported container; PNG and JPEG are supported): " + label);
    }
    const int ch = m.channels();
    const int depth = m.depth();
    if (depth != CV_8U) {
        throw Error(ErrorCategory::UnsupportedFormat,
                    "unsupported bit depth (" + depth_name(depth) + ") in " + label +
                        "; P0 accepts only 8-bit images and does not reduce bit depth");
    }
    if (ch == 4) {
        throw Error(ErrorCategory::UnsupportedFormat,
                    "unsupported 4-channel (alpha) image " + label + "; P0 does not silently discard alpha");
    }
    if (ch == 2) {
        throw Error(ErrorCategory::UnsupportedFormat,
                    "unsupported 2-channel (gray+alpha) image " + label + "; P0 does not silently discard alpha");
    }
    if (ch != 1 && ch != 3) {
        throw Error(ErrorCategory::UnsupportedFormat, "unsupported channel count " + std::to_string(ch) + " in " + label);
    }
    validate_dimensions(m.cols, m.rows, max_dimension, "decoded input image");

    DecodedImage out;
    const PixelFormat fmt = ch == 3 ? PixelFormat::Rgb8 : PixelFormat::Gray8;
    out.image = HostU8Image(m.cols, m.rows, fmt);
    for (int y = 0; y < m.rows; ++y) {
        const std::uint8_t* src = m.ptr<std::uint8_t>(y);
        std::uint8_t* dst = out.image.row(y);
        if (ch == 3) {
            // OpenCV decodes color as BGR; the project contract is RGB.
            for (int x = 0; x < m.cols; ++x) {
                dst[3 * x + 0] = src[3 * x + 2];
                dst[3 * x + 1] = src[3 * x + 1];
                dst[3 * x + 2] = src[3 * x + 0];
            }
        } else {
            std::memcpy(dst, src, static_cast<std::size_t>(m.cols));
        }
    }
    out.source_format = ch == 3 ? "8-bit 3-channel (decoder BGR reordered to RGB)" : "8-bit 1-channel gray";
    return out;
}

DecodedImage read_image(const std::string& path, int max_dimension) {
    return decode_image(read_file_bytes(path), path, max_dimension);
}

std::uint8_t quantize(float v) noexcept {
    if (!(v > 0.0f)) v = 0.0f;  // also maps NaN to 0; NaN is rejected by validation before this point
    if (v > 1.0f) v = 1.0f;
    return static_cast<std::uint8_t>(std::floor(v * 255.0f + 0.5f));
}

HostU8Image quantize_image(const ConstFloatImageView& v) {
    HostU8Image out(v.width, v.height, PixelFormat::Gray8);
    for (int y = 0; y < v.height; ++y) {
        const float* src = row_ptr(v, y);
        std::uint8_t* dst = out.row(y);
        for (int x = 0; x < v.width; ++x) dst[x] = quantize(src[x]);
    }
    return out;
}

namespace {
cv::Mat to_mat(const HostU8Image& img) {
    if (img.format() == PixelFormat::Gray8) {
        return cv::Mat(img.height(), img.width(), CV_8UC1, const_cast<std::uint8_t*>(img.row(0)), img.stride_bytes());
    }
    cv::Mat bgr(img.height(), img.width(), CV_8UC3);
    for (int y = 0; y < img.height(); ++y) {
        const std::uint8_t* s = img.row(y);
        std::uint8_t* d = bgr.ptr<std::uint8_t>(y);
        for (int x = 0; x < img.width(); ++x) {
            d[3 * x] = s[3 * x + 2];
            d[3 * x + 1] = s[3 * x + 1];
            d[3 * x + 2] = s[3 * x];
        }
    }
    return bgr;
}

void write_mat(const std::string& path, const cv::Mat& m) {
    bool ok = false;
    try {
        ok = cv::imwrite(path, m);
    } catch (const cv::Exception& e) {
        throw Error(ErrorCategory::IoFailure, "failed to encode/write image: " + path, {"encode", "", "", e.what()});
    }
    if (!ok) throw Error(ErrorCategory::IoFailure, "failed to write image: " + path);
}
}  // namespace

void write_png_gray(const std::string& path, const HostU8Image& img) {
    if (img.format() != PixelFormat::Gray8) throw Error(ErrorCategory::Internal, "write_png_gray requires gray8");
    write_mat(path, to_mat(img));
}

void write_image(const std::string& path, const HostU8Image& img) { write_mat(path, to_mat(img)); }

}  // namespace framepipe::io
