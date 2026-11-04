#pragma once
// Custom CPU operators (spec Section 7, 11). Each operator processes a
// half-open range of OUTPUT rows [y_begin, y_end) so the scalar reference and
// the OpenMP backend share one audited implementation. The reference calls
// each with the full range.
//
// Contract for all operators: views are already validated, input and output
// do not alias, and dimensions agree as documented per function.

#include <cstdint>
#include <vector>

#include "core/tables.hpp"
#include "framepipe/image_view.hpp"

namespace framepipe::cpu {

// Exact integer grayscale convention (spec 7.2).
inline std::uint8_t gray_u8(std::uint8_t r, std::uint8_t g, std::uint8_t b) noexcept {
    const std::uint32_t s = 77u * r + 150u * g + 29u * b + 128u;
    return static_cast<std::uint8_t>(s >> 8);
}

// out has the same width/height as in.
void grayscale(const InputImageView& in, const FloatImageView& out, int y_begin, int y_end);

// out is (Wo, Ho); tables built for (in.width -> Wo, in.height -> Ho).
void resize_bilinear(const ConstFloatImageView& in, const FloatImageView& out, const ResizeTables& t, int y_begin,
                     int y_end);

// weights has length 2*radius+1. out has the same dimensions as in.
void gaussian_horizontal(const ConstFloatImageView& in, const FloatImageView& out, const std::vector<float>& weights,
                         int radius, int y_begin, int y_end);
void gaussian_vertical(const ConstFloatImageView& in, const FloatImageView& out, const std::vector<float>& weights,
                       int radius, int y_begin, int y_end);

// Normal path: writes only the normalized edge magnitude.
void sobel_edges(const ConstFloatImageView& in, const FloatImageView& out, int y_begin, int y_end);
// Validation path: signed components.
void sobel_components(const ConstFloatImageView& in, const FloatImageView& gx, const FloatImageView& gy, int y_begin,
                      int y_end);

// Edge normalization applied to a (gx, gy) pair; shared by both Sobel paths.
float edge_from_components(float gx, float gy) noexcept;

}  // namespace framepipe::cpu
