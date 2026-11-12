#include "validation/compare.hpp"

#include <limits>

#include <nlohmann/json.hpp>

#include "framepipe/errors.hpp"

#include "test_util.hpp"

using namespace fptest;

TEST(Compare, ExactMatchHasNullPsnr) {
    const auto a = make_float(4, 3, [](int x, int y) { return x * 0.1 + y; });
    const auto r = compare_float_images(a.cview(), a.cview(), {0, 0});
    EXPECT_TRUE(r.passed);
    EXPECT_TRUE(r.exact_match);
    EXPECT_FALSE(r.psnr_db.has_value());
    const auto j = nlohmann::json::parse(compare_result_json(r));
    EXPECT_TRUE(j["psnr_db"].is_null());
    EXPECT_TRUE(j["exact_match"].get<bool>());
}

TEST(Compare, ToleranceRuleAndDiagnostics) {
    const auto a = make_float(3, 3, [](int, int) { return 1.0; });
    auto b = make_float(3, 3, [](int, int) { return 1.0; });
    b.at(2, 1) = 1.0f + 3e-5f;
    const auto pass = compare_float_images(a.cview(), b.cview(), {1e-5, 1e-5});  // 2e-5 allowed... 3e-5 fails
    EXPECT_FALSE(pass.passed);
    EXPECT_EQ(pass.failing_pixels, 1);
    EXPECT_EQ(pass.max_error_x, 2);
    EXPECT_EQ(pass.max_error_y, 1);
    EXPECT_NEAR(pass.candidate_at_max, 1.0 + 3e-5, 1e-6);
    const auto loose = compare_float_images(a.cview(), b.cview(), {2e-5, 2e-5});
    EXPECT_TRUE(loose.passed);
    EXPECT_TRUE(loose.psnr_db.has_value());
}

TEST(Compare, NonFiniteAlwaysFails) {
    const auto a = make_float(2, 2, [](int, int) { return 0.5; });
    auto b = make_float(2, 2, [](int, int) { return 0.5; });
    b.at(0, 0) = std::numeric_limits<float>::quiet_NaN();
    auto r = compare_float_images(a.cview(), b.cview(), {1, 1});
    EXPECT_FALSE(r.passed);
    EXPECT_EQ(r.nonfinite_candidate, 1);
    b.at(0, 0) = std::numeric_limits<float>::infinity();
    r = compare_float_images(a.cview(), b.cview(), {1, 1});
    EXPECT_FALSE(r.passed);
    // JSON stays finite
    const auto j = nlohmann::json::parse(compare_result_json(r));
    EXPECT_TRUE(j["max_abs_error"].is_number());
}

TEST(Compare, DimensionMismatchFails) {
    const auto a = make_float(2, 2, [](int, int) { return 0; });
    const auto b = make_float(3, 2, [](int, int) { return 0; });
    const auto r = compare_float_images(a.cview(), b.cview(), {1, 1});
    EXPECT_FALSE(r.passed);
    EXPECT_FALSE(r.dimensions_match);
}

TEST(Compare, RangeCheckForEdges) {
    const auto a = make_float(2, 1, [](int, int) { return 1.0; });
    const auto b = make_float(2, 1, [](int, int) { return 1.0 + 1e-7; });
    EXPECT_TRUE(compare_float_images(a.cview(), b.cview(), {1e-5, 1e-5}, false).passed);
    EXPECT_FALSE(compare_float_images(a.cview(), b.cview(), {1e-5, 1e-5}, true).passed);
}

TEST(Compare, SpecToleranceTable) {
    EXPECT_EQ(tolerance_for(Stage::Gray).atol, 1e-7);
    EXPECT_EQ(tolerance_for(Stage::Gray).rtol, 1e-6);
    EXPECT_EQ(tolerance_for(Stage::Resized).atol, 2e-6);
    EXPECT_EQ(tolerance_for(Stage::BlurHorizontal).atol, 1e-5);
    EXPECT_EQ(tolerance_for(Stage::BlurVertical).rtol, 1e-5);
    EXPECT_EQ(tolerance_for(Stage::SobelX).atol, 2e-5);
    EXPECT_EQ(tolerance_for(Stage::SobelY).rtol, 1e-5);
    EXPECT_EQ(tolerance_for(Stage::Edges).atol, 1e-5);
}

TEST(Compare, ErrorMapScalesByTolerance) {
    // Tolerance atol=0.1, rtol=0 so the limit is 0.1 everywhere.
    const auto ref = make_float_from({{0.5f, 0.5f, 0.5f, 0.5f, 0.5f}});
    const auto cand = make_float_from({{0.5f, 0.55f, 0.6f, 0.9f, std::numeric_limits<float>::quiet_NaN()}});
    const auto m = error_map(ref.cview(), cand.cview(), {0.1, 0.0});
    ASSERT_EQ(m.width(), 5);
    EXPECT_EQ(m.row(0)[0], 0) << "exact match is black";
    EXPECT_NEAR(m.row(0)[1], 128, 1) << "half the tolerance is mid-gray";
    EXPECT_EQ(m.row(0)[2], 255) << "at the tolerance is white";
    EXPECT_EQ(m.row(0)[3], 255) << "beyond the tolerance saturates";
    EXPECT_EQ(m.row(0)[4], 255) << "non-finite is white";
}

TEST(Compare, ErrorMapRejectsShapeMismatch) {
    const auto a = make_float_from({{0.f, 0.f}});
    const auto b = make_float_from({{0.f}});
    EXPECT_THROW(error_map(a.cview(), b.cview(), {1e-5, 1e-5}), framepipe::Error);
}
