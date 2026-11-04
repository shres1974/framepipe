// Lifecycle and resource-ownership tests (spec Section 13).

#include "framepipe/errors.hpp"
#include "test_util.hpp"

using namespace fptest;

TEST(Lifecycle, RepeatedPrepareAndShapeChanges) {
    const std::size_t before = tracked_host_live_bytes();
    {
        Pipeline p(cfg(32, 24));
        for (int i = 0; i < 5; ++i) {
            for (auto [w, h, f] : {std::tuple{64, 48, PixelFormat::Rgb8}, {17, 19, PixelFormat::Gray8},
                                   {1, 1, PixelFormat::Rgb8}}) {
                p.prepare({w, h, f});
                const auto in = random_u8(w, h, f, static_cast<std::uint32_t>(w * h + i));
                HostFloatImage out(32, 24);
                p.process_into(in.view(), out.view());
            }
        }
    }
    EXPECT_EQ(tracked_host_live_bytes(), before) << "pipeline leaked host workspace";
}

TEST(Lifecycle, SteadyStateRetainedAllocationIsStable) {
    Pipeline p(cfg(64, 48));
    const auto in = random_u8(128, 96, PixelFormat::Rgb8, 5);
    HostFloatImage out(64, 48);
    p.prepare(spec_of(in.view()));
    p.process_into(in.view(), out.view());
    const std::size_t bytes = tracked_host_live_bytes();
    const std::size_t count = tracked_host_allocation_count();
    for (int i = 0; i < 200; ++i) p.process_into(in.view(), out.view());
    EXPECT_EQ(tracked_host_live_bytes(), bytes);
    EXPECT_EQ(tracked_host_allocation_count(), count) << "allocations in steady-state frame loop";
}

TEST(Lifecycle, ReprepareSameShapeKeepsRetainedBytes) {
    Pipeline p(cfg(64, 48));
    p.prepare({100, 80, PixelFormat::Rgb8});
    const std::size_t a = tracked_host_live_bytes();
    p.prepare({100, 80, PixelFormat::Rgb8});
    EXPECT_EQ(tracked_host_live_bytes(), a) << "prepare grew a cache instead of replacing workspace";
}

TEST(Lifecycle, FailureHalfwayThroughAllocation) {
    const std::size_t before = tracked_host_live_bytes();
    {
        Pipeline p(cfg(32, 24));
        p.prepare({40, 30, PixelFormat::Gray8});
        ASSERT_TRUE(p.is_prepared());
        // Second allocation inside prepare fails (gray succeeds, A fails).
        framepipe::testing::fail_host_allocation_after(2);
        EXPECT_THROW(
            {
                try {
                    p.prepare({50, 30, PixelFormat::Gray8});
                } catch (const Error& e) {
                    EXPECT_EQ(e.category(), ErrorCategory::AllocationFailure);
                    EXPECT_EQ(e.context().operation, "prepare");
                    throw;
                }
            },
            Error);
        framepipe::testing::fail_host_allocation_after(0);
        EXPECT_FALSE(p.is_prepared()) << "failed prepare must leave the pipeline unprepared";
        const auto in = random_u8(50, 30, PixelFormat::Gray8, 1);
        HostFloatImage out(32, 24);
        EXPECT_THROW(p.process_into(in.view(), out.view()), Error);
        // Recovers with a fresh prepare.
        p.prepare({50, 30, PixelFormat::Gray8});
        p.process_into(in.view(), out.view());
        EXPECT_TRUE(std::isfinite(out.at(0, 0)));
    }
    EXPECT_EQ(tracked_host_live_bytes(), before);
}

TEST(Lifecycle, DestructionAfterFailedOperation) {
    const std::size_t before = tracked_host_live_bytes();
    {
        Pipeline p(cfg(16, 16));
        p.prepare({20, 20, PixelFormat::Rgb8});
        const auto wrong = random_u8(21, 20, PixelFormat::Rgb8, 3);
        HostFloatImage out(16, 16);
        EXPECT_THROW(p.process_into(wrong.view(), out.view()), Error);
    }
    EXPECT_EQ(tracked_host_live_bytes(), before);
}

TEST(Lifecycle, IndependentInstancesHaveIndependentParameters) {
    Pipeline a(cfg(32, 32, OutputStage::Blur, 1, 0.5));
    Pipeline b(cfg(32, 32, OutputStage::Blur, 15, 20.0));
    const auto in = random_u8(40, 40, PixelFormat::Gray8, 9);
    a.prepare(spec_of(in.view()));
    b.prepare(spec_of(in.view()));
    HostFloatImage oa(32, 32), ob(32, 32), oa2(32, 32);
    a.process_into(in.view(), oa.view());
    b.process_into(in.view(), ob.view());
    a.process_into(in.view(), oa2.view());
    expect_identical(oa, oa2, "instance a affected by instance b");
    bool differs = false;
    for (int y = 0; y < 32 && !differs; ++y)
        for (int x = 0; x < 32; ++x) differs = differs || oa.at(x, y) != ob.at(x, y);
    EXPECT_TRUE(differs);
}

TEST(Lifecycle, MovedFromPipelineIsSafe) {
    Pipeline a(cfg(8, 8));
    a.prepare({8, 8, PixelFormat::Gray8});
    Pipeline b(std::move(a));
    const auto in = random_u8(8, 8, PixelFormat::Gray8, 2);
    HostFloatImage out(8, 8);
    b.process_into(in.view(), out.view());
    EXPECT_TRUE(std::isfinite(out.at(3, 3)));
}

TEST(Lifecycle, HostBufferMoveOnlyNoDoubleFree) {
    const std::size_t before = tracked_host_live_bytes();
    {
        HostBuffer a(1000);
        HostBuffer b(std::move(a));
        EXPECT_EQ(a.size(), 0u);
        EXPECT_EQ(b.size(), 1000u);
        HostBuffer c;
        c = std::move(b);
        c = HostBuffer(10);
        EXPECT_EQ(tracked_host_live_bytes(), before + 10);
    }
    EXPECT_EQ(tracked_host_live_bytes(), before);
}
