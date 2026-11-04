// cpu-omp backend: row-parallel results must be bit-identical to cpu-ref
// because each output row is computed by exactly the same code.

#include "framepipe/errors.hpp"
#include "test_util.hpp"

using namespace fptest;

TEST(CpuParallel, MatchesReferenceExactly) {
    if (!query_backend(Backend::CpuParallel).usable) GTEST_SKIP() << "cpu-omp not built (OpenMP unavailable)";
    for (int threads : {1, 2, 3, 8}) {
        for (auto [wi, hi, wo, ho] : {std::tuple{1, 1, 1, 1}, {257, 129, 128, 64}, {31, 33, 64, 3}, {640, 480, 320, 240}}) {
            const auto in = random_u8(wi, hi, PixelFormat::Rgb8, static_cast<std::uint32_t>(threads * 1000 + wi), 5);
            PipelineConfig rc = cfg(wo, ho);
            PipelineConfig pc = rc;
            pc.backend = Backend::CpuParallel;
            pc.cpu.threads = threads;
            Pipeline ref(rc), par(pc);
            EXPECT_EQ(par.selected_backend(), Backend::CpuParallel);
            ref.prepare(spec_of(in.view()));
            par.prepare(spec_of(in.view()));
            HostFloatImage a(wo, ho), b(wo, ho);
            ref.process_into(in.view(), a.view());
            par.process_into(in.view(), b.view());
            expect_identical(a, b, "cpu-omp vs cpu-ref");
        }
    }
}

TEST(CpuParallel, UnavailableIsExplicit) {
    if (query_backend(Backend::CpuParallel).usable) GTEST_SKIP() << "cpu-omp is built";
    PipelineConfig c = cfg(8, 8);
    c.backend = Backend::CpuParallel;
    EXPECT_THROW(Pipeline p(c), Error);
}
