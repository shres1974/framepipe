#include <filesystem>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "framepipe/errors.hpp"
#include "io/image_io.hpp"
#include "test_util.hpp"

using namespace fptest;
namespace fs = std::filesystem;

namespace {
fs::path tmpdir() {
    const auto d = fs::temp_directory_path() / ("framepipe_io_" + std::to_string(::getpid()));
    fs::create_directories(d);
    return d;
}
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

TEST(ImageIo, BgrIsReorderedToRgb) {
    // Write a PNG whose first pixel is pure red (BGR = 0,0,255 in OpenCV).
    cv::Mat m(1, 3, CV_8UC3);
    m.at<cv::Vec3b>(0, 0) = {0, 0, 255};    // red
    m.at<cv::Vec3b>(0, 1) = {0, 255, 0};    // green
    m.at<cv::Vec3b>(0, 2) = {255, 0, 0};    // blue
    const auto p = (tmpdir() / "rgb.png").string();
    ASSERT_TRUE(cv::imwrite(p, m));
    const auto d = io::read_image(p);
    ASSERT_EQ(d.image.format(), PixelFormat::Rgb8);
    const std::uint8_t* r = d.image.row(0);
    EXPECT_EQ(r[0], 255); EXPECT_EQ(r[1], 0); EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 0); EXPECT_EQ(r[4], 255); EXPECT_EQ(r[5], 0);
    EXPECT_EQ(r[6], 0); EXPECT_EQ(r[7], 0); EXPECT_EQ(r[8], 255);
    // And the grayscale result uses the RGB convention (red -> 77).
    Pipeline pl(cfg(3, 1, OutputStage::Resize));
    pl.prepare(spec_of(d.image.view()));
    HostFloatImage out(3, 1);
    pl.process_into(d.image.view(), out.view());
    EXPECT_EQ(out.at(0, 0), 77.0f / 255.0f);
    EXPECT_EQ(out.at(1, 0), 149.0f / 255.0f);
    EXPECT_EQ(out.at(2, 0), 29.0f / 255.0f);
}

TEST(ImageIo, GrayPreserved) {
    cv::Mat m(2, 2, CV_8UC1);
    m.at<std::uint8_t>(0, 0) = 1; m.at<std::uint8_t>(0, 1) = 2;
    m.at<std::uint8_t>(1, 0) = 3; m.at<std::uint8_t>(1, 1) = 250;
    const auto p = (tmpdir() / "g.png").string();
    ASSERT_TRUE(cv::imwrite(p, m));
    const auto d = io::read_image(p);
    EXPECT_EQ(d.image.format(), PixelFormat::Gray8);
    EXPECT_EQ(d.image.row(1)[1], 250);
}

TEST(ImageIo, RejectsAlpha16BitAndCorrupt) {
    const auto d = tmpdir();
    cv::Mat rgba(2, 2, CV_8UC4, cv::Scalar(1, 2, 3, 4));
    ASSERT_TRUE(cv::imwrite((d / "a.png").string(), rgba));
    EXPECT_EQ(category_of([&] { io::read_image((d / "a.png").string()); }), ErrorCategory::UnsupportedFormat);
    cv::Mat g16(2, 2, CV_16UC1, cv::Scalar(1000));
    ASSERT_TRUE(cv::imwrite((d / "g16.png").string(), g16));
    try {
        io::read_image((d / "g16.png").string());
        FAIL();
    } catch (const Error& e) {
        EXPECT_EQ(e.category(), ErrorCategory::UnsupportedFormat);
        EXPECT_NE(std::string(e.what()).find("16-bit"), std::string::npos);
    }
    std::ofstream((d / "bad.png").string()) << "this is not an image";
    EXPECT_EQ(category_of([&] { io::read_image((d / "bad.png").string()); }), ErrorCategory::IoFailure);
    EXPECT_EQ(category_of([&] { io::read_image((d / "missing.png").string()); }), ErrorCategory::IoFailure);
    std::ofstream((d / "empty.png").string());
    EXPECT_EQ(category_of([&] { io::read_image((d / "empty.png").string()); }), ErrorCategory::IoFailure);
}

TEST(ImageIo, JpegDecodes) {
    cv::Mat m(8, 8, CV_8UC3, cv::Scalar(10, 20, 30));
    const auto p = (tmpdir() / "x.jpg").string();
    ASSERT_TRUE(cv::imwrite(p, m));
    const auto d = io::read_image(p);
    EXPECT_EQ(d.image.format(), PixelFormat::Rgb8);
    EXPECT_EQ(d.image.width(), 8);
}

TEST(ImageIo, ExifOrientationIgnored) {
    // Checked-in fixture: a 2x1 JPEG with EXIF Orientation=6 (rotate 90 CW).
    // Stored order must be preserved: width 2, height 1.
    const std::string p = std::string(FRAMEPIPE_FIXTURE_DIR) + "/exif_orientation6.jpg";
    if (!fs::exists(p)) GTEST_SKIP() << "fixture missing: " << p;
    const auto d = io::read_image(p);
    EXPECT_EQ(d.image.width(), 4);
    EXPECT_EQ(d.image.height(), 2);
}

TEST(ImageIo, MaxDimensionEnforcedAtDecode) {
    cv::Mat m(10, 70, CV_8UC1, cv::Scalar(0));
    const auto p = (tmpdir() / "wide.png").string();
    ASSERT_TRUE(cv::imwrite(p, m));
    EXPECT_EQ(category_of([&] { io::read_image(p, 64); }), ErrorCategory::InvalidImage);
}

TEST(ImageIo, QuantizationFormula) {
    EXPECT_EQ(io::quantize(0.0f), 0);
    EXPECT_EQ(io::quantize(1.0f), 255);
    EXPECT_EQ(io::quantize(-0.5f), 0);
    EXPECT_EQ(io::quantize(2.0f), 255);
    EXPECT_EQ(io::quantize(0.5f), 128);  // floor(127.5 + 0.5) = 128
    EXPECT_EQ(io::quantize(127.0f / 255.0f), 127);
    for (int v = 0; v < 256; ++v) EXPECT_EQ(io::quantize(static_cast<float>(v) / 255.0f), v);
    EXPECT_EQ(io::quantize(std::nanf("")), 0);
}

TEST(ImageIo, QuantizedPngRoundTrip) {
    auto f = make_float(5, 2, [](int x, int y) { return (x + 5 * y) / 9.0; }, 3);
    const auto q = io::quantize_image(f.cview());
    const auto p = (tmpdir() / "q.png").string();
    io::write_png_gray(p, q);
    const auto d = io::read_image(p);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 5; ++x) EXPECT_EQ(d.image.row(y)[x], io::quantize(f.at(x, y)));
}
