#include <limits>

#include "framepipe/errors.hpp"
#include "test_util.hpp"

using namespace fptest;

namespace {
ErrorCategory category_of(const std::function<void()>& f) {
    try {
        f();
    } catch (const Error& e) {
        return e.category();
    }
    ADD_FAILURE() << "expected an Error";
    return ErrorCategory::Internal;
}
}  // namespace

TEST(ImageView, ValidInputViews) {
    HostU8Image a(5, 3, PixelFormat::Rgb8, 7);
    EXPECT_NO_THROW(validate_view(a.view()));
    HostU8Image b(1, 1, PixelFormat::Gray8);
    EXPECT_NO_THROW(validate_view(b.view()));
}

TEST(ImageView, ZeroAndNegativeDimensions) {
    std::uint8_t buf[64] = {};
    InputImageView v{buf, 0, 4, 4, 64, PixelFormat::Gray8};
    EXPECT_EQ(category_of([&] { validate_view(v); }), ErrorCategory::InvalidImage);
    v.width = 4;
    v.height = -1;
    EXPECT_EQ(category_of([&] { validate_view(v); }), ErrorCategory::InvalidImage);
}

TEST(ImageView, StrideTooSmall) {
    std::uint8_t buf[64] = {};
    InputImageView v{buf, 4, 2, 11, 64, PixelFormat::Rgb8};  // needs 12
    EXPECT_EQ(category_of([&] { validate_view(v); }), ErrorCategory::InvalidImage);
}

TEST(ImageView, ExtentCannotContainLastPixel) {
    std::uint8_t buf[64] = {};
    // 4x3 gray, stride 8: needs 8*2 + 4 = 20 bytes
    InputImageView v{buf, 4, 3, 8, 19, PixelFormat::Gray8};
    EXPECT_EQ(category_of([&] { validate_view(v); }), ErrorCategory::InvalidImage);
    v.extent_bytes = 20;  // exact minimum: last row need not include padding
    EXPECT_NO_THROW(validate_view(v));
}

TEST(ImageView, NullData) {
    InputImageView v{nullptr, 2, 2, 2, 4, PixelFormat::Gray8};
    EXPECT_EQ(category_of([&] { validate_view(v); }), ErrorCategory::InvalidImage);
}

TEST(ImageView, FloatStrideMustBeMultipleOfFloat) {
    alignas(16) float buf[64];
    FloatImageView v{buf, 3, 2, 14, sizeof buf};  // 14 bytes not divisible by 4
    EXPECT_EQ(category_of([&] { validate_view(v); }), ErrorCategory::InvalidImage);
    v.stride_bytes = 16;
    EXPECT_NO_THROW(validate_view(v));
}

TEST(ImageView, FloatMisalignedPointer) {
    alignas(16) unsigned char raw[256];
    FloatImageView v{reinterpret_cast<float*>(raw + 1), 2, 2, 8, 200};
    EXPECT_EQ(category_of([&] { validate_view(v); }), ErrorCategory::InvalidImage);
}

TEST(ImageView, MaxDimension) {
    std::uint8_t buf[4] = {};
    InputImageView v{buf, kHardMaxDimension + 1, 1, static_cast<std::size_t>(kHardMaxDimension + 1),
                     std::numeric_limits<std::size_t>::max(), PixelFormat::Gray8};
    EXPECT_EQ(category_of([&] { validate_view(v); }), ErrorCategory::InvalidImage);
    v.width = 100;
    v.stride_bytes = 100;
    EXPECT_EQ(category_of([&] { validate_view(v, 64); }), ErrorCategory::InvalidImage);  // configured lower limit
}

TEST(ImageView, CheckedArithmeticOverflow) {
    const std::size_t big = std::numeric_limits<std::size_t>::max() / 2 + 1;
    EXPECT_EQ(category_of([&] { checked_mul(big, 2, "t"); }), ErrorCategory::InvalidImage);
    EXPECT_EQ(category_of([&] { checked_add(big, big, "t"); }), ErrorCategory::InvalidImage);
    EXPECT_EQ(checked_mul(0, big, "t"), 0u);
    // Huge stride with large height overflows the extent computation.
    EXPECT_EQ(category_of([&] { required_extent_bytes(10, 16384, 1, std::numeric_limits<std::size_t>::max() / 1000); }),
              ErrorCategory::InvalidImage);
}

TEST(ImageView, OverlapDetection) {
    unsigned char buf[100] = {};
    EXPECT_TRUE(byte_ranges_overlap(buf, 50, buf + 49, 10));
    EXPECT_FALSE(byte_ranges_overlap(buf, 50, buf + 50, 10));
    EXPECT_FALSE(byte_ranges_overlap(buf, 0, buf, 10));
}

TEST(ImageView, HostFloatImageFillsWithNaN) {
    HostFloatImage im(3, 2, 2);
    EXPECT_EQ(im.stride_bytes(), 5 * sizeof(float));
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 3; ++x) EXPECT_TRUE(std::isnan(im.at(x, y)));
}
