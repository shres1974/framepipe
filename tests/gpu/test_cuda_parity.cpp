// CUDA parity, boundary, ownership, and error-path tests (spec 12, 13, 16).
// Every test SKIPS visibly (GTEST_SKIP with a reason) when no usable CUDA
// device exists. A skipped run is NOT evidence of GPU correctness.
//
// Intended for Compute Sanitizer as well:
//   compute-sanitizer --tool memcheck --error-exitcode 99 build/cuda/bin/framepipe_cuda_tests

#include <cuda_runtime.h>

#include "framepipe/errors.hpp"
#include "test_util.hpp"
#include "validation/compare.hpp"

using namespace fptest;

namespace {

bool cuda_ok(std::string* why) {
    const auto a = query_backend(Backend::Cuda);
    if (why) *why = a.reason;
    return a.usable;
}

#define REQUIRE_CUDA()                                               \
    do {                                                             \
        std::string why_;                                            \
        if (!cuda_ok(&why_)) GTEST_SKIP() << "no usable CUDA device: " << why_; \
    } while (0)

PipelineConfig cuda_cfg(int wo, int ho, OutputStage s = OutputStage::Edges, int r = 3, double sigma = 1.2) {
    auto c = cfg(wo, ho, s, r, sigma);
    c.backend = Backend::Cuda;
    return c;
}

void expect_stage_parity(const HostU8Image& in, const PipelineConfig& gc) {
    PipelineConfig rc = gc;
    rc.backend = Backend::CpuReference;
    Pipeline ref(rc), gpu(gc);
    ASSERT_EQ(gpu.selected_backend(), Backend::Cuda);
    ref.prepare(spec_of(in.view()));
    gpu.prepare(spec_of(in.view()));
    const auto rs = ref.process_capture_stages(in.view());
    const auto gs = gpu.process_capture_stages(in.view());
    for (Stage s : stages_for(gc.output_stage, true)) {
        ASSERT_NE(rs.get(s), nullptr) << to_string(s);
        ASSERT_NE(gs.get(s), nullptr) << to_string(s);
        const auto r = compare_float_images(rs.get(s)->cview(), gs.get(s)->cview(), tolerance_for(s), s == Stage::Edges);
        EXPECT_TRUE(r.passed) << to_string(s) << ": " << r.message << " max_err=" << r.max_abs_error << " at ("
                              << r.max_error_x << "," << r.max_error_y << ") ref=" << r.reference_at_max
                              << " gpu=" << r.candidate_at_max << " input " << in.width() << "x" << in.height();
    }
    // Normal (non-capture) path with a padded caller output.
    HostFloatImage a(gc.resize.width, gc.resize.height, 0), b(gc.resize.width, gc.resize.height, 5);
    ref.process_into(in.view(), a.view());
    gpu.process_into(in.view(), b.view());
    const Stage fs = gc.output_stage == OutputStage::Edges  ? Stage::Edges
                     : gc.output_stage == OutputStage::Blur ? Stage::BlurVertical
                                                            : Stage::Resized;
    const auto r = compare_float_images(a.cview(), b.cview(), tolerance_for(fs), fs == Stage::Edges);
    EXPECT_TRUE(r.passed) << "normal path: " << r.message;
    expect_padding_untouched(b);
}

}  // namespace

TEST(CudaParity, LaunchBoundaryAndSmallGeometry) {
    REQUIRE_CUDA();
    std::uint32_t seed = 100;
    for (auto [w, h] : {std::pair{1, 1}, {1, 7}, {7, 1}, {2, 2}, {3, 3}, {15, 17}, {16, 16}, {17, 19}, {31, 33},
                        {257, 129}}) {
        for (PixelFormat f : {PixelFormat::Gray8, PixelFormat::Rgb8}) {
            ++seed;
            const auto in = random_u8(w, h, f, seed, seed % 4);
            expect_stage_parity(in, cuda_cfg(w, h));                      // same size
            expect_stage_parity(in, cuda_cfg(w * 2 + 1, h + 3));          // upscale, non-integer
            expect_stage_parity(in, cuda_cfg(std::max(1, w / 3), std::max(1, h / 2)));  // downscale
        }
    }
}

TEST(CudaParity, ResizeRatiosAndOneByOne) {
    REQUIRE_CUDA();
    const auto in = random_u8(64, 48, PixelFormat::Rgb8, 7);
    for (auto [w, h] : {std::pair{64, 48}, {1, 1}, {128, 96}, {37, 23}, {1279, 7}}) {
        expect_stage_parity(in, cuda_cfg(w, h, OutputStage::Resize));
    }
}

TEST(CudaParity, GaussianRadiiAndSigmas) {
    REQUIRE_CUDA();
    const auto in = random_u8(97, 61, PixelFormat::Rgb8, 11, 3);
    for (int r : {0, 1, 3, 7, 15})
        for (double s : {0.1, 1.2, 100.0}) expect_stage_parity(in, cuda_cfg(53, 41, OutputStage::Edges, r, s));
}

TEST(CudaParity, IntensityExtremesAndStructuredPatterns) {
    REQUIRE_CUDA();
    for (int v : {0, 255, 128}) {
        HostU8Image in(33, 17, PixelFormat::Rgb8, 0, static_cast<std::uint8_t>(v));
        expect_stage_parity(in, cuda_cfg(19, 21));
    }
    HostU8Image chk(40, 40, PixelFormat::Gray8);
    for (int y = 0; y < 40; ++y)
        for (int x = 0; x < 40; ++x) chk.row(y)[x] = ((x / 3 + y / 3) % 2) ? 255 : 0;
    expect_stage_parity(chk, cuda_cfg(40, 40));
    HostU8Image ramp(64, 8, PixelFormat::Gray8);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 64; ++x) ramp.row(y)[x] = static_cast<std::uint8_t>(x * 4);
    expect_stage_parity(ramp, cuda_cfg(64, 8));
}

TEST(CudaParity, CroppedInputView) {
    REQUIRE_CUDA();
    const auto big = random_u8(80, 60, PixelFormat::Rgb8, 5);
    InputImageView crop{big.row(9) + 11 * 3, 37, 29, big.stride_bytes(), big.stride_bytes() * 28 + 37 * 3, PixelFormat::Rgb8};
    PipelineConfig gc = cuda_cfg(23, 19), rc = gc;
    rc.backend = Backend::CpuReference;
    Pipeline ref(rc), gpu(gc);
    ref.prepare({37, 29, PixelFormat::Rgb8});
    gpu.prepare({37, 29, PixelFormat::Rgb8});
    HostFloatImage a(23, 19), b(23, 19);
    ref.process_into(crop, a.view());
    gpu.process_into(crop, b.view());
    EXPECT_TRUE(compare_float_images(a.cview(), b.cview(), tolerance_for(Stage::Edges), true).passed);
}

TEST(CudaParity, PinnedAndPageableTransferPathsAgree) {
    REQUIRE_CUDA();
    const auto in = random_u8(129, 65, PixelFormat::Rgb8, 3, 7);
    PipelineConfig pinned = cuda_cfg(64, 32), pageable = pinned;
    pageable.resources.max_pinned_host_bytes = 0;  // forces pageable direct copies
    Pipeline a(pinned), b(pageable);
    a.prepare(spec_of(in.view()));
    b.prepare(spec_of(in.view()));
    EXPECT_EQ(a.workspace().transfer_mode, "pinned_staging");
    EXPECT_EQ(b.workspace().transfer_mode, "pageable_direct");
    HostFloatImage oa(64, 32), ob(64, 32, 3);
    a.process_into(in.view(), oa.view());
    b.process_into(in.view(), ob.view());
    expect_identical(oa, ob, "pinned vs pageable");
}

TEST(CudaParity, DeviceResidentPathMatches) {
    REQUIRE_CUDA();
    const auto in = random_u8(200, 100, PixelFormat::Rgb8, 21);
    Pipeline p(cuda_cfg(100, 50));
    p.prepare(spec_of(in.view()));
    ASSERT_TRUE(p.supports_device_pipeline_timing());
    HostFloatImage a(100, 50), b(100, 50);
    p.process_into(in.view(), a.view());
    p.device_resident_upload(in.view());
    const double ms = p.device_resident_run_timed_ms();
    EXPECT_GT(ms, 0.0);
    p.device_resident_download(b.view());
    expect_identical(a, b, "device-resident");
}

TEST(CudaLifecycle, NoDeviceAllocationInSteadyState) {
    REQUIRE_CUDA();
    const auto in = random_u8(320, 240, PixelFormat::Rgb8, 1);
    Pipeline p(cuda_cfg(160, 120));
    p.prepare(spec_of(in.view()));
    HostFloatImage out(160, 120);
    p.process_into(in.view(), out.view());
    std::size_t free0 = 0, total = 0;
    ASSERT_EQ(cudaMemGetInfo(&free0, &total), cudaSuccess);
    const auto ws0 = p.workspace();
    for (int i = 0; i < 300; ++i) p.process_into(in.view(), out.view());
    std::size_t free1 = 0;
    ASSERT_EQ(cudaMemGetInfo(&free1, &total), cudaSuccess);
    EXPECT_EQ(p.workspace().device_allocated_bytes, ws0.device_allocated_bytes);
    // Driver-level free memory is not an exact measure of app buffers; allow 2 MiB of noise.
    EXPECT_LE(free0 > free1 ? free0 - free1 : 0, std::size_t{2} << 20) << "device memory grew in steady state";
}

TEST(CudaLifecycle, ReprepareShapeChangesAndInstances) {
    REQUIRE_CUDA();
    Pipeline a(cuda_cfg(32, 32, OutputStage::Blur, 1, 0.5));
    Pipeline b(cuda_cfg(32, 32, OutputStage::Blur, 15, 20.0));
    for (auto [w, h] : {std::pair{64, 48}, {17, 19}, {64, 48}, {1, 1}}) {
        const auto in = random_u8(w, h, PixelFormat::Rgb8, static_cast<std::uint32_t>(w * h));
        a.prepare(spec_of(in.view()));
        b.prepare(spec_of(in.view()));
        HostFloatImage oa(32, 32), ob(32, 32), oa2(32, 32);
        a.process_into(in.view(), oa.view());
        b.process_into(in.view(), ob.view());
        a.process_into(in.view(), oa2.view());
        expect_identical(oa, oa2, "instance parameters must be independent");
    }
    std::size_t f0 = 0, t = 0, f1 = 0;
    cudaMemGetInfo(&f0, &t);
    for (int i = 0; i < 20; ++i) a.prepare({64 + i % 2, 48, PixelFormat::Rgb8});
    cudaMemGetInfo(&f1, &t);
    EXPECT_LE(f0 > f1 ? f0 - f1 : 0, std::size_t{4} << 20) << "re-prepare must replace, not accumulate";
}

TEST(CudaErrors, BudgetExceededIsAllocationFailure) {
    REQUIRE_CUDA();
    PipelineConfig c = cuda_cfg(1280, 720);
    c.resources.max_device_bytes = 1024;
    Pipeline p(c);
    try {
        p.prepare({1920, 1080, PixelFormat::Rgb8});
        FAIL() << "expected AllocationFailure";
    } catch (const Error& e) {
        EXPECT_EQ(e.category(), ErrorCategory::AllocationFailure);
    }
    EXPECT_FALSE(p.is_prepared());
    // Destruction after failure must be safe; a fresh prepare within budget works.
}

TEST(CudaErrors, WrongDeviceIndex) {
    REQUIRE_CUDA();
    int n = 0;
    cudaGetDeviceCount(&n);
    PipelineConfig c = cuda_cfg(8, 8);
    c.device_index = n + 3;
    try {
        Pipeline p(c);
        FAIL();
    } catch (const Error& e) {
        EXPECT_EQ(e.category(), ErrorCategory::BackendUnavailable);
    }
}

TEST(CudaErrors, ExplicitCudaUnavailableNeverFallsBack) {
    std::string why;
    if (cuda_ok(&why)) GTEST_SKIP() << "device usable; this checks the no-device path";
    EXPECT_THROW(Pipeline p(cuda_cfg(8, 8)), Error);
}
