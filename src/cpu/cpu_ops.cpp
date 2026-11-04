#include "cpu/cpu_ops.hpp"

#include <cmath>
#include <cstddef>

namespace framepipe::cpu {

namespace {
inline int clampi(int i, int extent) noexcept { return i < 0 ? 0 : (i > extent - 1 ? extent - 1 : i); }
}  // namespace

void grayscale(const InputImageView& in, const FloatImageView& out, int y_begin, int y_end) {
    const auto w = static_cast<std::size_t>(in.width);
    for (int y = y_begin; y < y_end; ++y) {
        const std::uint8_t* src = row_ptr(in, y);
        float* dst = row_ptr(out, y);
        if (in.format == PixelFormat::Rgb8) {
            for (std::size_t x = 0; x < w; ++x) {
                const std::uint8_t g = gray_u8(src[3 * x], src[3 * x + 1], src[3 * x + 2]);
                dst[x] = static_cast<float>(g) / 255.0f;
            }
        } else {
            for (std::size_t x = 0; x < w; ++x) dst[x] = static_cast<float>(src[x]) / 255.0f;
        }
    }
}

void resize_bilinear(const ConstFloatImageView& in, const FloatImageView& out, const ResizeTables& t, int y_begin,
                     int y_end) {
    const int wo = out.width;
    for (int y = y_begin; y < y_end; ++y) {
        const auto yi = static_cast<std::size_t>(y);
        const float* r0 = row_ptr(in, t.y.i0[yi]);
        const float* r1 = row_ptr(in, t.y.i1[yi]);
        const float ay = t.y.w[yi];
        float* dst = row_ptr(out, y);
        for (int x = 0; x < wo; ++x) {
            const auto xi = static_cast<std::size_t>(x);
            const std::int32_t x0 = t.x.i0[xi];
            const std::int32_t x1 = t.x.i1[xi];
            const float ax = t.x.w[xi];
            const float top = r0[x0] + (r0[x1] - r0[x0]) * ax;
            const float bottom = r1[x0] + (r1[x1] - r1[x0]) * ax;
            dst[x] = top + (bottom - top) * ay;
        }
    }
}

void gaussian_horizontal(const ConstFloatImageView& in, const FloatImageView& out, const std::vector<float>& weights,
                         int radius, int y_begin, int y_end) {
    const int w = in.width;
    const float* wt = weights.data();
    for (int y = y_begin; y < y_end; ++y) {
        const float* src = row_ptr(in, y);
        float* dst = row_ptr(out, y);
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {  // increasing k
                acc += wt[k + radius] * src[clampi(x + k, w)];
            }
            dst[x] = acc;
        }
    }
}

void gaussian_vertical(const ConstFloatImageView& in, const FloatImageView& out, const std::vector<float>& weights,
                       int radius, int y_begin, int y_end) {
    const int w = in.width;
    const int h = in.height;
    const float* wt = weights.data();
    for (int y = y_begin; y < y_end; ++y) {
        float* dst = row_ptr(out, y);
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {  // increasing k
                acc += wt[k + radius] * row_ptr(in, clampi(y + k, h))[x];
            }
            dst[x] = acc;
        }
    }
}

namespace {
// Correlation with Gx = [[-1,0,1],[-2,0,2],[-1,0,1]], Gy = [[-1,-2,-1],[0,0,0],[1,2,1]].
inline void sobel_at(const ConstFloatImageView& in, int x, int y, float& gx, float& gy) noexcept {
    const int w = in.width;
    const int h = in.height;
    const int xm = clampi(x - 1, w), xp = clampi(x + 1, w);
    const float* rm = row_ptr(in, clampi(y - 1, h));
    const float* r0 = row_ptr(in, y);
    const float* rp = row_ptr(in, clampi(y + 1, h));
    gx = (rm[xp] - rm[xm]) + 2.0f * (r0[xp] - r0[xm]) + (rp[xp] - rp[xm]);
    gy = (rp[xm] - rm[xm]) + 2.0f * (rp[x] - rm[x]) + (rp[xp] - rm[xp]);
}
}  // namespace

float edge_from_components(float gx, float gy) noexcept {
    const float mag = std::sqrt(gx * gx + gy * gy);
    float e = mag / kEdgeNormalization;
    if (e < 0.0f) e = 0.0f;
    if (e > 1.0f) e = 1.0f;
    return e;
}

void sobel_edges(const ConstFloatImageView& in, const FloatImageView& out, int y_begin, int y_end) {
    for (int y = y_begin; y < y_end; ++y) {
        float* dst = row_ptr(out, y);
        for (int x = 0; x < in.width; ++x) {
            float gx, gy;
            sobel_at(in, x, y, gx, gy);
            dst[x] = edge_from_components(gx, gy);
        }
    }
}

void sobel_components(const ConstFloatImageView& in, const FloatImageView& gx_out, const FloatImageView& gy_out,
                      int y_begin, int y_end) {
    for (int y = y_begin; y < y_end; ++y) {
        float* dx = row_ptr(gx_out, y);
        float* dy = row_ptr(gy_out, y);
        for (int x = 0; x < in.width; ++x) {
            float gx, gy;
            sobel_at(in, x, y, gx, gy);
            dx[x] = gx;
            dy[x] = gy;
        }
    }
}

}  // namespace framepipe::cpu
