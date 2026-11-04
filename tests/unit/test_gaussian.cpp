#include "cpu/cpu_ops.hpp"
#include "test_util.hpp"

using namespace fptest;

namespace {
HostFloatImage blur(const HostFloatImage& in, int r, double s, std::size_t pad = 0, HostFloatImage* hpass = nullptr) {
    const auto w = build_gaussian_weights(r, s);
    HostFloatImage h(in.width(), in.height(), pad), v(in.width(), in.height(), pad);
    framepipe::cpu::gaussian_horizontal(in.cview(), h.view(), w, r, 0, in.height());
    framepipe::cpu::gaussian_vertical(h.cview(), v.view(), w, r, 0, in.height());
    if (hpass) *hpass = std::move(h);
    return v;
}
}  // namespace

TEST(Gaussian, ConstantStaysConstantIncludingCorners) {
    for (int r : {0, 1, 3, 15}) {
        for (double s : {0.1, 1.2, 100.0}) {
            for (auto [w, h] : {std::pair{1, 1}, {1, 7}, {7, 1}, {3, 3}, {17, 19}}) {
                const auto in = make_float(w, h, [](int, int) { return 0.4; });
                const auto out = blur(in, r, s);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) ASSERT_NEAR(out.at(x, y), 0.4f, 1e-5) << r << " " << s;
            }
        }
    }
}

TEST(Gaussian, RadiusZeroIsIdentity) {
    const auto in = make_float(5, 4, [](int x, int y) { return (x * 3 + y) / 17.0; });
    expect_identical(in, blur(in, 0, 1.2), "radius 0");
}

TEST(Gaussian, ImpulseResponseIsOuterProductOfWeights) {
    const int r = 3, n = 15, c = 7;
    const auto w = build_gaussian_weights(r, 1.2);
    const auto in = make_float(n, n, [&](int x, int y) { return x == c && y == c ? 1.0 : 0.0; });
    HostFloatImage h;
    const auto out = blur(in, r, 1.2, 0, &h);
    for (int x = 0; x < n; ++x) {
        const int k = x - c;
        EXPECT_NEAR(h.at(x, c), std::abs(k) <= r ? w[k + r] : 0.0f, 1e-7) << "horizontal pass axis";
        EXPECT_EQ(h.at(x, c + 1), 0.0f) << "horizontal pass must not spread vertically";
    }
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            const int kx = x - c, ky = y - c;
            const float e = (std::abs(kx) <= r && std::abs(ky) <= r) ? w[ky + r] * w[kx + r] : 0.0f;
            ASSERT_NEAR(out.at(x, y), e, 1e-7);
        }
}

TEST(Gaussian, ReplicatedBorderAtEdge) {
    // Step at the left border: replicated samples mean out(0) uses in(0) for k<0.
    const int r = 2;
    const auto w = build_gaussian_weights(r, 1.0);
    const auto in = make_float(6, 1, [](int x, int) { return x == 0 ? 1.0 : 0.0; });
    HostFloatImage h(6, 1);
    framepipe::cpu::gaussian_horizontal(in.cview(), h.view(), w, r, 0, 1);
    EXPECT_NEAR(h.at(0, 0), w[0] + w[1] + w[2], 1e-7);  // k=-2,-1,0 all clamp to x=0
    EXPECT_NEAR(h.at(1, 0), w[0] + w[1], 1e-7);  // k=-2 clamps to x=0, k=-1 is x=0
    EXPECT_NEAR(h.at(2, 0), w[0], 1e-7);          // only k=-2 reaches x=0
    EXPECT_EQ(h.at(3, 0), 0.0f);
}

TEST(Gaussian, PaddedViewsMatchPacked) {
    const auto a_in = make_float(31, 33, [](int x, int y) { return ((x * 7 + y * 13) % 29) / 29.0; });
    const auto b_in = make_float(31, 33, [](int x, int y) { return ((x * 7 + y * 13) % 29) / 29.0; }, 9);
    const auto a = blur(a_in, 7, 2.5);
    auto b = blur(b_in, 7, 2.5, 3);
    expect_identical(a, b, "padded blur");
    expect_padding_untouched(b);
}
