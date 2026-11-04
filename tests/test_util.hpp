#pragma once
// Shared test helpers. Deterministic generators only (seeded).

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "framepipe/image.hpp"
#include "framepipe/pipeline.hpp"

namespace fptest {

using namespace framepipe;

// Fills a float image's logical pixels from a function f(x, y).
template <class F>
inline HostFloatImage make_float(int w, int h, F f, std::size_t pad = 0) {
    HostFloatImage im(w, h, pad);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) im.at(x, y) = static_cast<float>(f(x, y));
    return im;
}

inline HostFloatImage make_float_from(const std::vector<std::vector<float>>& rows, std::size_t pad = 0) {
    const int h = static_cast<int>(rows.size());
    const int w = static_cast<int>(rows[0].size());
    return make_float(w, h, [&](int x, int y) { return rows[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)]; }, pad);
}

inline HostU8Image random_u8(int w, int h, PixelFormat f, std::uint32_t seed, std::size_t pad = 0) {
    HostU8Image im(w, h, f, pad, 0xAB);  // padding bytes are a canary value
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0, 255);
    const int ch = channel_count(f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w * ch; ++x) im.row(y)[x] = static_cast<std::uint8_t>(d(rng));
    return im;
}

inline void expect_close(const HostFloatImage& a, const HostFloatImage& b, double atol, double rtol,
                         const char* what = "") {
    ASSERT_EQ(a.width(), b.width()) << what;
    ASSERT_EQ(a.height(), b.height()) << what;
    for (int y = 0; y < a.height(); ++y)
        for (int x = 0; x < a.width(); ++x) {
            const double r = a.at(x, y), c = b.at(x, y);
            ASSERT_TRUE(std::isfinite(c)) << what << " non-finite at " << x << "," << y;
            ASSERT_LE(std::fabs(c - r), atol + rtol * std::fabs(r)) << what << " at (" << x << "," << y << ") ref=" << r
                                                                   << " cand=" << c;
        }
}

inline void expect_identical(const HostFloatImage& a, const HostFloatImage& b, const char* what = "") {
    ASSERT_EQ(a.width(), b.width());
    ASSERT_EQ(a.height(), b.height());
    for (int y = 0; y < a.height(); ++y)
        for (int x = 0; x < a.width(); ++x)
            ASSERT_EQ(a.at(x, y), b.at(x, y)) << what << " at (" << x << "," << y << ")";
}

inline PipelineConfig cfg(int wo, int ho, OutputStage s = OutputStage::Edges, int radius = 3, double sigma = 1.2) {
    PipelineConfig c;
    c.backend = Backend::CpuReference;
    c.resize.width = wo;
    c.resize.height = ho;
    c.output_stage = s;
    c.gaussian.radius = radius;
    c.gaussian.sigma = sigma;
    return c;
}

// Checks that per-row padding (beyond logical width) still holds the NaN
// canary written at allocation time, i.e. nothing wrote outside logical pixels.
inline void expect_padding_untouched(HostFloatImage& im) {
    const std::size_t stride_elems = im.stride_bytes() / sizeof(float);
    for (int y = 0; y < im.height(); ++y) {
        const float* row = im.raw() + static_cast<std::size_t>(y) * stride_elems;
        for (std::size_t x = static_cast<std::size_t>(im.width()); x < stride_elems; ++x)
            ASSERT_TRUE(std::isnan(row[x])) << "padding overwritten at row " << y << " col " << x;
    }
}

}  // namespace fptest
