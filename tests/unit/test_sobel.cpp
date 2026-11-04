#include <cmath>

#include "cpu/cpu_ops.hpp"
#include "test_util.hpp"

using namespace fptest;

namespace {
void components(const HostFloatImage& in, HostFloatImage& gx, HostFloatImage& gy) {
    gx = HostFloatImage(in.width(), in.height());
    gy = HostFloatImage(in.width(), in.height());
    framepipe::cpu::sobel_components(in.cview(), gx.view(), gy.view(), 0, in.height());
}
HostFloatImage edges(const HostFloatImage& in, std::size_t pad = 0) {
    HostFloatImage e(in.width(), in.height(), pad);
    framepipe::cpu::sobel_edges(in.cview(), e.view(), 0, in.height());
    return e;
}
}  // namespace

TEST(Sobel, HorizontalRampHandFixture) {
    for (int W : {3, 4, 9}) {
        const auto in = make_float(W, 5, [&](int x, int) { return x / static_cast<double>(W - 1); });
        HostFloatImage gx, gy;
        components(in, gx, gy);
        const auto e = edges(in);
        for (int y = 0; y < 5; ++y)
            for (int x = 1; x < W - 1; ++x) {
                EXPECT_NEAR(gx.at(x, y), 8.0 / (W - 1), 2e-5);
                EXPECT_NEAR(gy.at(x, y), 0.0, 2e-5);
                EXPECT_NEAR(e.at(x, y), std::sqrt(2.0) / (W - 1), 1e-5);
            }
    }
}

TEST(Sobel, SignsAreCorrelationNotConvolution) {
    // Increasing to the right => positive gx. Increasing downward => positive gy.
    const auto right = make_float(5, 5, [](int x, int) { return x / 4.0; });
    const auto down = make_float(5, 5, [](int, int y) { return y / 4.0; });
    HostFloatImage gx, gy;
    components(right, gx, gy);
    EXPECT_GT(gx.at(2, 2), 0.0f);
    EXPECT_NEAR(gy.at(2, 2), 0.0f, 1e-7);
    components(down, gx, gy);
    EXPECT_GT(gy.at(2, 2), 0.0f);
    EXPECT_NEAR(gx.at(2, 2), 0.0f, 1e-7);
    EXPECT_NEAR(gy.at(2, 2), 8.0 / 4.0, 2e-5);
}

TEST(Sobel, ConstantAndOnePixelAreZero) {
    for (auto [w, h] : {std::pair{1, 1}, {1, 7}, {7, 1}, {2, 2}, {3, 3}}) {
        const auto in = make_float(w, h, [](int, int) { return 128.0 / 255.0; });
        const auto e = edges(in);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) EXPECT_EQ(e.at(x, y), 0.0f);
    }
}

TEST(Sobel, ReplicatedBorderAtRampEdge) {
    // At x=0 on a ramp, replicated x-1 == x0: gx = 4 * (p(1) - p(0)).
    const int W = 5;
    const auto in = make_float(W, 3, [&](int x, int) { return x / static_cast<double>(W - 1); });
    HostFloatImage gx, gy;
    components(in, gx, gy);
    EXPECT_NEAR(gx.at(0, 1), 4.0 / (W - 1), 2e-5);
    EXPECT_NEAR(gx.at(W - 1, 1), 4.0 / (W - 1), 2e-5);
}

TEST(Sobel, MaxResponseClampsToOne) {
    // Checkerboard 0/1 produces large responses; edge must be within [0,1].
    const auto in = make_float(8, 8, [](int x, int y) { return (x + y) % 2; });
    const auto e = edges(in);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
            EXPECT_GE(e.at(x, y), 0.0f);
            EXPECT_LE(e.at(x, y), 1.0f);
        }
    // Step edge with maximal |gx|=4 and |gy|=4 => magnitude sqrt(32) => exactly 1.
    EXPECT_NEAR(framepipe::cpu::edge_from_components(4.0f, 4.0f), 1.0f, 1e-7);
    EXPECT_EQ(framepipe::cpu::edge_from_components(40.0f, 0.0f), 1.0f);
}

TEST(Sobel, PaddedOutput) {
    const auto in = make_float(15, 17, [](int x, int y) { return ((x * x + y) % 11) / 11.0; }, 4);
    const auto a = edges(in);
    auto b = edges(in, 6);
    expect_identical(a, b);
    expect_padding_untouched(b);
}
