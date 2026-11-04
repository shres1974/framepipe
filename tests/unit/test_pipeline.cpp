#include <cstring>

#include "framepipe/errors.hpp"
#include "test_util.hpp"

using namespace fptest;

TEST(Pipeline, HandFixtureGray1x1) {
    // Gray8 1x1 = 128 stays 128/255 through resize and blur; edges are zero.
    HostU8Image in(1, 1, PixelFormat::Gray8);
    in.row(0)[0] = 128;
    for (auto [w, h] : {std::pair{1, 1}, {5, 3}, {16, 16}}) {
        for (OutputStage st : {OutputStage::Resize, OutputStage::Blur, OutputStage::Edges}) {
            Pipeline p(cfg(w, h, st));
            p.prepare(spec_of(in.view()));
            HostFloatImage out(w, h);
            p.process_into(in.view(), out.view());
            const float expect = st == OutputStage::Edges ? 0.0f : 128.0f / 255.0f;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) ASSERT_NEAR(out.at(x, y), expect, 1e-5) << to_string(st);
        }
    }
}

TEST(Pipeline, OutputStagePrefixMatchesCapture) {
    const auto in = random_u8(37, 29, PixelFormat::Rgb8, 42, 3);
    for (OutputStage st : {OutputStage::Resize, OutputStage::Blur, OutputStage::Edges}) {
        Pipeline p(cfg(23, 31, st));
        p.prepare(spec_of(in.view()));
        HostFloatImage out(23, 31, 2);
        p.process_into(in.view(), out.view());
        const auto cap = p.process_capture_stages(in.view());
        const Stage final_stage = st == OutputStage::Edges ? Stage::Edges
                                  : st == OutputStage::Blur ? Stage::BlurVertical
                                                            : Stage::Resized;
        ASSERT_NE(cap.get(final_stage), nullptr);
        expect_identical(*cap.get(final_stage), out, to_string(st));
        expect_padding_untouched(out);
        // Only the required prefix is computed.
        if (st == OutputStage::Resize) {
            EXPECT_EQ(cap.get(Stage::BlurHorizontal), nullptr);
        }
        if (st != OutputStage::Edges) {
            EXPECT_EQ(cap.get(Stage::SobelX), nullptr);
        }
    }
}

TEST(Pipeline, ExplicitCudaNeverFallsBack) {
    PipelineConfig c = cfg(8, 8);
    c.backend = Backend::Cuda;
    if (query_backend(Backend::Cuda).usable) GTEST_SKIP() << "a CUDA device is usable here";
    try {
        Pipeline p(c);
        FAIL() << "explicit cuda request must not construct a CPU pipeline";
    } catch (const Error& e) {
        EXPECT_EQ(e.category(), ErrorCategory::BackendUnavailable);
        EXPECT_EQ(exit_code_for(e.category()), 4);
    }
}

TEST(Pipeline, WrongDeviceIndexIsReported) {
    PipelineConfig c = cfg(8, 8);
    c.backend = Backend::Cuda;
    c.device_index = 99;
    EXPECT_THROW(Pipeline p(c), Error);
}

TEST(Pipeline, AutoSelectsAndReportsReason) {
    PipelineConfig c = cfg(8, 8);
    c.backend = Backend::Auto;
    Pipeline p(c);
    const bool cuda = query_backend(Backend::Cuda).usable;
    EXPECT_EQ(p.selected_backend(), cuda ? Backend::Cuda : Backend::CpuReference);
    EXPECT_NE(p.configuration().selection_reason.find("auto"), std::string::npos);
}

TEST(Pipeline, MustPrepareFirst) {
    Pipeline p(cfg(8, 8));
    const auto in = random_u8(8, 8, PixelFormat::Gray8, 1);
    HostFloatImage out(8, 8);
    EXPECT_THROW(p.process_into(in.view(), out.view()), Error);
}

TEST(Pipeline, RejectsWrongOutputSize) {
    Pipeline p(cfg(8, 8));
    const auto in = random_u8(8, 8, PixelFormat::Gray8, 1);
    p.prepare(spec_of(in.view()));
    HostFloatImage out(9, 8);
    EXPECT_THROW(p.process_into(in.view(), out.view()), Error);
}

TEST(Pipeline, RejectsShapeMismatchWithoutReprepare) {
    Pipeline p(cfg(8, 8));
    p.prepare({10, 10, PixelFormat::Gray8});
    const auto in = random_u8(10, 10, PixelFormat::Rgb8, 1);
    HostFloatImage out(8, 8);
    EXPECT_THROW(p.process_into(in.view(), out.view()), Error);
}

TEST(Pipeline, RejectsOverlappingBuffers) {
    Pipeline p(cfg(4, 4, OutputStage::Resize));
    p.prepare({4, 4, PixelFormat::Gray8});
    HostFloatImage buf(4, 8);
    FloatImageView out = buf.view();
    out.height = 4;
    out.extent_bytes = 64;
    InputImageView in{reinterpret_cast<const std::uint8_t*>(buf.raw()), 4, 4, 4, 16, PixelFormat::Gray8};
    try {
        p.process_into(in, out);
        FAIL();
    } catch (const Error& e) {
        EXPECT_NE(std::string(e.what()).find("overlap"), std::string::npos);
    }
}

TEST(Pipeline, PrepareRejectsOversizeInputPerConfig) {
    PipelineConfig c = cfg(8, 8);
    c.resources.max_dimension = 64;
    Pipeline p(c);
    EXPECT_THROW(p.prepare({65, 10, PixelFormat::Gray8}), Error);
    EXPECT_NO_THROW(p.prepare({64, 10, PixelFormat::Gray8}));
}

TEST(Pipeline, CroppedInputViewMatchesCopiedImage) {
    // A crop (non-contiguous sub-view) of a larger image must equal processing a packed copy.
    const auto big = random_u8(50, 40, PixelFormat::Rgb8, 77);
    const int cx = 7, cy = 5, cw = 21, ch = 17;
    InputImageView crop{big.row(cy) + cx * 3, cw, ch, big.stride_bytes(),
                        big.stride_bytes() * (ch - 1) + static_cast<std::size_t>(cw) * 3, PixelFormat::Rgb8};
    HostU8Image packed(cw, ch, PixelFormat::Rgb8);
    for (int y = 0; y < ch; ++y) std::memcpy(packed.row(y), big.row(cy + y) + cx * 3, static_cast<std::size_t>(cw) * 3);
    Pipeline p(cfg(13, 11));
    p.prepare({cw, ch, PixelFormat::Rgb8});
    HostFloatImage a(13, 11), b(13, 11);
    p.process_into(crop, a.view());
    p.process_into(packed.view(), b.view());
    expect_identical(a, b, "crop");
}
