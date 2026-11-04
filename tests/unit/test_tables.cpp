#include <cmath>

#include "core/tables.hpp"
#include "framepipe/errors.hpp"
#include "test_util.hpp"

using namespace framepipe;

TEST(ResizeTables, IdentityIsExact) {
    const auto t = build_axis_table(7, 7);
    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(t.i0[i], i);
        EXPECT_EQ(t.i1[i], std::min(i + 1, 6));
        EXPECT_EQ(t.w[i], 0.0f);
    }
}

TEST(ResizeTables, UpscaleNegativeFractionalCoordinateAtEdge) {
    // 2 -> 3: u = (x+0.5)*2/3 - 0.5 = -1/6, 0.5, 7/6
    const auto t = build_axis_table(2, 3);
    // x=0: raw = floor(-1/6) = -1 (floor, not truncation); weight from unclamped raw = 5/6
    EXPECT_EQ(t.i0[0], 0);
    EXPECT_EQ(t.i1[0], 0);
    EXPECT_NEAR(t.w[0], 5.0 / 6.0, 1e-7);
    EXPECT_EQ(t.i0[1], 0);
    EXPECT_EQ(t.i1[1], 1);
    EXPECT_NEAR(t.w[1], 0.5, 1e-7);
    // x=2: raw = 1, both indices clamp to 1, weight = 1/6
    EXPECT_EQ(t.i0[2], 1);
    EXPECT_EQ(t.i1[2], 1);
    EXPECT_NEAR(t.w[2], 1.0 / 6.0, 1e-7);
}

TEST(ResizeTables, TruncationWouldDiffer) {
    // Guard against replacing floor with truncation: for u in (-1, 0)
    // truncation gives 0 and weight u<0. floor gives -1 and weight in (0,1).
    const auto t = build_axis_table(4, 9);
    for (std::size_t i = 0; i < t.w.size(); ++i) {
        EXPECT_GE(t.w[i], 0.0f);
        EXPECT_LT(t.w[i], 1.0f);
    }
}

TEST(ResizeTables, DownscaleToOne) {
    const auto t = build_axis_table(2, 1);  // u = 0.5
    EXPECT_EQ(t.i0[0], 0);
    EXPECT_EQ(t.i1[0], 1);
    EXPECT_EQ(t.w[0], 0.5f);
}

TEST(ResizeTables, OnePixelInputClampsEverything) {
    const auto t = build_axis_table(1, 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(t.i0[i], 0);
        EXPECT_EQ(t.i1[i], 0);
    }
}

TEST(ResizeTables, MatchesDoubleFormulaManyRatios) {
    for (int wi : {1, 2, 3, 7, 16, 17, 1919, 1920}) {
        for (int wo : {1, 2, 5, 16, 33, 720, 1279}) {
            const auto t = build_axis_table(wi, wo);
            for (int x = 0; x < wo; ++x) {
                const double u = (x + 0.5) * wi / static_cast<double>(wo) - 0.5;
                const double f = std::floor(u);
                ASSERT_EQ(t.w[x], static_cast<float>(u - f));
                ASSERT_EQ(t.i0[x], clamp_index(static_cast<long long>(f), wi));
                ASSERT_EQ(t.i1[x], clamp_index(static_cast<long long>(f) + 1, wi));
            }
        }
    }
}

TEST(GaussianWeights, RadiusZeroIsOne) {
    const auto w = build_gaussian_weights(0, 1.2);
    ASSERT_EQ(w.size(), 1u);
    EXPECT_EQ(w[0], 1.0f);
}

TEST(GaussianWeights, SymmetricNormalized) {
    for (int r : {1, 3, 7, 15}) {
        for (double s : {0.1, 1.2, 5.0, 100.0}) {
            const auto w = build_gaussian_weights(r, s);
            ASSERT_EQ(w.size(), static_cast<std::size_t>(2 * r + 1));
            double sum = 0;
            for (std::size_t i = 0; i < w.size(); ++i) {
                sum += w[i];
                EXPECT_EQ(w[i], w[w.size() - 1 - i]);
                EXPECT_GE(w[i], 0.0f);
            }
            EXPECT_NEAR(sum, 1.0, 1e-6);
        }
    }
}

TEST(GaussianWeights, MatchesDoubleDefinition) {
    const int r = 3;
    const double s = 1.2;
    const auto w = build_gaussian_weights(r, s);
    double sum = 0;
    for (int k = -r; k <= r; ++k) sum += std::exp(-(k * k) / (2 * s * s));
    for (int k = -r; k <= r; ++k) EXPECT_EQ(w[k + r], static_cast<float>(std::exp(-(k * k) / (2 * s * s)) / sum));
}

TEST(GaussianWeights, RejectsOutOfRange) {
    EXPECT_THROW(build_gaussian_weights(-1, 1.0), Error);
    EXPECT_THROW(build_gaussian_weights(16, 1.0), Error);
    EXPECT_THROW(build_gaussian_weights(3, 0.05), Error);
    EXPECT_THROW(build_gaussian_weights(3, std::nan("")), Error);
    EXPECT_THROW(build_gaussian_weights(3, 1000.0), Error);
}
