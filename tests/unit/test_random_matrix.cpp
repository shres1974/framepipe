// Fixed-seed random matrix over the required geometry, ratio, and Gaussian
// cases. These compare the production pipeline with a small, separately
// written double-precision reference implemented directly in this test file
// (no production table builder or operator code). The full independent oracle
// lives in scripts/oracle.py and is exercised by ctest -L oracle.

#include <cmath>

#include "cpu/cpu_ops.hpp"
#include "test_util.hpp"

using namespace fptest;

namespace {

int cl(long long i, int n) { return static_cast<int>(i < 0 ? 0 : (i > n - 1 ? n - 1 : i)); }

std::vector<double> ref_pipeline(const HostU8Image& in, int wo, int ho, int r, double s, OutputStage st) {
    const int wi = in.width(), hi = in.height();
    std::vector<double> g(static_cast<std::size_t>(wi * hi));
    for (int y = 0; y < hi; ++y)
        for (int x = 0; x < wi; ++x) {
            const std::uint8_t* p = in.row(y);
            unsigned v = in.format() == PixelFormat::Rgb8 ? (77u * p[3 * x] + 150u * p[3 * x + 1] + 29u * p[3 * x + 2] + 128u) >> 8
                                                          : p[x];
            g[static_cast<std::size_t>(y * wi + x)] = v / 255.0;
        }
    auto G = [&](int x, int y) { return g[static_cast<std::size_t>(y * wi + x)]; };
    std::vector<double> rs(static_cast<std::size_t>(wo * ho));
    for (int y = 0; y < ho; ++y) {
        const double v = (y + 0.5) * hi / ho - 0.5;
        const double fy = std::floor(v);
        const double ay = v - fy;
        for (int x = 0; x < wo; ++x) {
            const double u = (x + 0.5) * wi / wo - 0.5;
            const double fx = std::floor(u);
            const double ax = u - fx;
            const int x0 = cl((long long)fx, wi), x1 = cl((long long)fx + 1, wi);
            const int y0 = cl((long long)fy, hi), y1 = cl((long long)fy + 1, hi);
            const double top = G(x0, y0) + (G(x1, y0) - G(x0, y0)) * ax;
            const double bot = G(x0, y1) + (G(x1, y1) - G(x0, y1)) * ax;
            rs[static_cast<std::size_t>(y * wo + x)] = top + (bot - top) * ay;
        }
    }
    if (st == OutputStage::Resize) return rs;
    std::vector<double> w(static_cast<std::size_t>(2 * r + 1));
    double sum = 0;
    for (int k = -r; k <= r; ++k) sum += (w[static_cast<std::size_t>(k + r)] = std::exp(-(double)k * k / (2 * s * s)));
    for (auto& x : w) x /= sum;
    std::vector<double> h(rs.size()), b(rs.size());
    for (int y = 0; y < ho; ++y)
        for (int x = 0; x < wo; ++x) {
            double a = 0;
            for (int k = -r; k <= r; ++k) a += w[static_cast<std::size_t>(k + r)] * rs[static_cast<std::size_t>(y * wo + cl(x + k, wo))];
            h[static_cast<std::size_t>(y * wo + x)] = a;
        }
    for (int y = 0; y < ho; ++y)
        for (int x = 0; x < wo; ++x) {
            double a = 0;
            for (int k = -r; k <= r; ++k) a += w[static_cast<std::size_t>(k + r)] * h[static_cast<std::size_t>(cl(y + k, ho) * wo + x)];
            b[static_cast<std::size_t>(y * wo + x)] = a;
        }
    if (st == OutputStage::Blur) return b;
    std::vector<double> e(rs.size());
    auto B = [&](int x, int y) { return b[static_cast<std::size_t>(cl(y, ho) * wo + cl(x, wo))]; };
    for (int y = 0; y < ho; ++y)
        for (int x = 0; x < wo; ++x) {
            const double gx = -B(x - 1, y - 1) + B(x + 1, y - 1) - 2 * B(x - 1, y) + 2 * B(x + 1, y) - B(x - 1, y + 1) + B(x + 1, y + 1);
            const double gy = -B(x - 1, y - 1) - 2 * B(x, y - 1) - B(x + 1, y - 1) + B(x - 1, y + 1) + 2 * B(x, y + 1) + B(x + 1, y + 1);
            e[static_cast<std::size_t>(y * wo + x)] = std::min(1.0, std::max(0.0, std::sqrt(gx * gx + gy * gy) / std::sqrt(32.0)));
        }
    return e;
}

struct Case {
    int wi, hi, wo, ho, r;
    double s;
};

}  // namespace

TEST(RandomMatrix, GeometryRatiosAndRadii) {
    const std::vector<Case> cases = {
        {1, 1, 1, 1, 3, 1.2},     {1, 7, 3, 5, 1, 0.5},     {7, 1, 5, 3, 3, 1.2},     {2, 2, 3, 3, 1, 1.0},
        {3, 3, 3, 3, 3, 1.2},     {15, 17, 15, 17, 3, 1.2}, {16, 16, 16, 16, 0, 1.2}, {17, 19, 31, 33, 7, 3.0},
        {31, 33, 17, 19, 15, 100}, {257, 129, 128, 64, 3, 1.2}, {64, 48, 1, 1, 3, 1.2},  {19, 23, 41, 7, 15, 0.1},
        {40, 30, 64, 48, 1, 1.2}};
    std::uint32_t seed = 1234;
    for (const auto& c : cases) {
        for (PixelFormat f : {PixelFormat::Gray8, PixelFormat::Rgb8}) {
            for (OutputStage st : {OutputStage::Resize, OutputStage::Blur, OutputStage::Edges}) {
                ++seed;
                const auto in = random_u8(c.wi, c.hi, f, seed, seed % 3);
                Pipeline p(cfg(c.wo, c.ho, st, c.r, c.s));
                p.prepare(spec_of(in.view()));
                HostFloatImage out(c.wo, c.ho, seed % 2);
                p.process_into(in.view(), out.view());
                const auto ref = ref_pipeline(in, c.wo, c.ho, c.r, c.s, st);
                const double atol = st == OutputStage::Resize ? 2e-6 : 1e-5;
                const double rtol = st == OutputStage::Resize ? 2e-6 : 1e-5;
                for (int y = 0; y < c.ho; ++y)
                    for (int x = 0; x < c.wo; ++x) {
                        const double rv = ref[static_cast<std::size_t>(y * c.wo + x)], cv = out.at(x, y);
                        ASSERT_LE(std::fabs(cv - rv), atol + rtol * std::fabs(rv))
                            << "seed=" << seed << " case " << c.wi << "x" << c.hi << "->" << c.wo << "x" << c.ho
                            << " r=" << c.r << " s=" << c.s << " fmt=" << to_string(f) << " stage=" << to_string(st)
                            << " at (" << x << "," << y << ")";
                    }
            }
        }
    }
}

TEST(RandomMatrix, IntensityExtremes) {
    for (int v : {0, 255, 128}) {
        HostU8Image in(33, 17, PixelFormat::Rgb8, 0, static_cast<std::uint8_t>(v));
        Pipeline p(cfg(19, 21, OutputStage::Blur, 15, 4.0));
        p.prepare(spec_of(in.view()));
        HostFloatImage out(19, 21);
        p.process_into(in.view(), out.view());
        const float expect = static_cast<float>(framepipe::cpu::gray_u8(static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(v))) / 255.0f;
        for (int y = 0; y < 21; ++y)
            for (int x = 0; x < 19; ++x) ASSERT_NEAR(out.at(x, y), expect, 1e-5);
    }
}
