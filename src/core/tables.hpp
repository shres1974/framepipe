#pragma once
// Immutable per-prepared-pipeline parameter tables (spec Sections 7.3 and 7.4).
// Both production backends consume exactly these FP32/int32 values.

#include <cstdint>
#include <vector>

namespace framepipe {

constexpr float kEdgeNormalization = 5.6568542494923801952f;  // sqrtf(32.0f)

inline int clamp_index(long long i, int extent) noexcept {
    if (i < 0) return 0;
    if (i > extent - 1) return extent - 1;
    return static_cast<int>(i);
}

// Bilinear half-pixel tables for one axis: for each destination index d,
// sample indices i0[d], i1[d] (already clamped) and FP32 weight w[d].
struct AxisTable {
    std::vector<std::int32_t> i0;
    std::vector<std::int32_t> i1;
    std::vector<float> w;
};

AxisTable build_axis_table(int input_extent, int output_extent);

struct ResizeTables {
    AxisTable x;  // length Wo
    AxisTable y;  // length Ho
};

ResizeTables build_resize_tables(int in_w, int in_h, int out_w, int out_h);

// Normalized FP32 Gaussian weights for offsets -r..r (length 2r+1),
// computed in double precision and converted once.
std::vector<float> build_gaussian_weights(int radius, double sigma);

}  // namespace framepipe
