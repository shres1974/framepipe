#include "cpu/cpu_ops.hpp"
#include "test_util.hpp"

using namespace fptest;

namespace {
HostFloatImage resize(const HostFloatImage& in, int wo, int ho, std::size_t pad = 0) {
    const auto t = build_resize_tables(in.width(), in.height(), wo, ho);
    HostFloatImage out(wo, ho, pad);
    framepipe::cpu::resize_bilinear(in.cview(), out.view(), t, 0, ho);
    return out;
}
}  // namespace

TEST(Resize, HandFixture2x2To3x3) {
    const auto in = make_float_from({{0, 1}, {1, 0}});
    const auto out = resize(in, 3, 3);
    const float expect[3][3] = {{0, 0.5f, 1}, {0.5f, 0.5f, 0.5f}, {1, 0.5f, 0}};
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x) EXPECT_NEAR(out.at(x, y), expect[y][x], 2e-6) << x << "," << y;
}

TEST(Resize, HandFixture2x2To1x1) {
    const auto in = make_float_from({{0, 1}, {1, 0}});
    const auto out = resize(in, 1, 1);
    EXPECT_NEAR(out.at(0, 0), 0.5f, 2e-6);
}

TEST(Resize, IdentityIsExact) {
    const auto in = make_float(17, 19, [](int x, int y) { return (x * 31 + y * 7) % 13 / 13.0; });
    const auto out = resize(in, 17, 19);
    expect_identical(in, out, "identity resize");
}

TEST(Resize, OnePixelToAnySize) {
    const auto in = make_float(1, 1, [](int, int) { return 128.0 / 255.0; });
    for (auto [w, h] : {std::pair{1, 1}, {7, 1}, {1, 7}, {13, 9}}) {
        const auto out = resize(in, w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) EXPECT_EQ(out.at(x, y), 128.0f / 255.0f);
    }
}

TEST(Resize, HorizontalRampDownscalePreservesLinearInterior) {
    // A linear ramp resampled with half-pixel bilinear stays linear in the
    // interior: out(x) = u(x) for u within [0, Wi-1].
    const int wi = 16, wo = 7;
    const auto in = make_float(wi, 3, [](int x, int) { return x; });
    const auto out = resize(in, wo, 3);
    for (int x = 0; x < wo; ++x) {
        const double u = (x + 0.5) * wi / wo - 0.5;
        const double expect = std::min(std::max(u, 0.0), static_cast<double>(wi - 1));
        EXPECT_NEAR(out.at(x, 1), expect, 1e-5);
    }
}

TEST(Resize, PaddedInputAndOutputViews) {
    const auto in_packed = make_float(9, 5, [](int x, int y) { return (x + 2 * y) / 20.0; });
    const auto in_pad = make_float(9, 5, [](int x, int y) { return (x + 2 * y) / 20.0; }, 7);
    const auto a = resize(in_packed, 4, 11);
    auto b = resize(in_pad, 4, 11, 5);
    expect_identical(a, b, "padded resize");
    expect_padding_untouched(b);
}
