#include "cpu/cpu_ops.hpp"
#include "test_util.hpp"

using namespace fptest;
using framepipe::cpu::gray_u8;

TEST(Grayscale, PrimaryColorsExact) {
    EXPECT_EQ(gray_u8(0, 0, 0), 0);
    EXPECT_EQ(gray_u8(255, 255, 255), 255);
    EXPECT_EQ(gray_u8(255, 0, 0), 77);
    EXPECT_EQ(gray_u8(0, 255, 0), 149);
    EXPECT_EQ(gray_u8(0, 0, 255), 29);
}

// Exhaustive check of the integer convention against an independent
// formulation using 64-bit arithmetic and explicit rounding.
TEST(Grayscale, ExhaustiveIntegerConvention) {
    for (int r = 0; r < 256; ++r)
        for (int g = 0; g < 256; ++g)
            for (int b = 0; b < 256; b += 1) {
                const long long s = 77LL * r + 150LL * g + 29LL * b;
                const long long expect = (s + 128) / 256;  // floor((s+128)/256) for non-negative s
                ASSERT_EQ(gray_u8(static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(g),
                                  static_cast<std::uint8_t>(b)),
                          expect)
                    << r << "," << g << "," << b;
            }
}

TEST(Grayscale, ChannelOrderMatters) {
    // A pure-red RGB pixel and the same bytes interpreted as BGR differ.
    EXPECT_NE(gray_u8(255, 0, 0), gray_u8(0, 0, 255));
}

TEST(Grayscale, NormalizationAndPaddedStride) {
    HostU8Image in(3, 2, PixelFormat::Rgb8, /*pad=*/5, 0xEE);
    const std::uint8_t px[2][9] = {{255, 0, 0, 0, 255, 0, 0, 0, 255}, {0, 0, 0, 255, 255, 255, 10, 20, 30}};
    for (int y = 0; y < 2; ++y)
        for (int i = 0; i < 9; ++i) in.row(y)[i] = px[y][i];
    HostFloatImage out(3, 2, /*pad=*/3);
    framepipe::cpu::grayscale(in.view(), out.view(), 0, 2);
    EXPECT_EQ(out.at(0, 0), 77.0f / 255.0f);
    EXPECT_EQ(out.at(1, 0), 149.0f / 255.0f);
    EXPECT_EQ(out.at(2, 0), 29.0f / 255.0f);
    EXPECT_EQ(out.at(0, 1), 0.0f);
    EXPECT_EQ(out.at(1, 1), 1.0f);
    EXPECT_EQ(out.at(2, 1), static_cast<float>(gray_u8(10, 20, 30)) / 255.0f);
    expect_padding_untouched(out);
}

TEST(Grayscale, Gray8PassThrough) {
    HostU8Image in(4, 1, PixelFormat::Gray8, 2);
    for (int x = 0; x < 4; ++x) in.row(0)[x] = static_cast<std::uint8_t>(x * 85);
    HostFloatImage out(4, 1);
    framepipe::cpu::grayscale(in.view(), out.view(), 0, 1);
    for (int x = 0; x < 4; ++x) EXPECT_EQ(out.at(x, 0), static_cast<float>(x * 85) / 255.0f);
}
