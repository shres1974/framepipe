// Benchmark harness integration: output files, schema basics, honesty rules.

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "benchmark/harness.hpp"
#include "framepipe/errors.hpp"
#include "test_util.hpp"

using namespace fptest;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {
fs::path setup() {
    const auto d = fs::temp_directory_path() / ("framepipe_bench_" + std::to_string(::getpid()) + "_" +
                                                ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::remove_all(d);
    fs::create_directories(d);
    cv::Mat a(48, 64, CV_8UC3, cv::Scalar(10, 100, 200));
    cv::Mat b(30, 40, CV_8UC1, cv::Scalar(77));
    for (int i = 0; i < 48; ++i) a.at<cv::Vec3b>(i, i % 64) = {255, 255, 255};
    cv::imwrite((d / "a.png").string(), a);
    cv::imwrite((d / "b.png").string(), b);
    std::ofstream(d / "m.jsonl") << R"({"frame_id":"a","path":"a.png"})" "\n" R"({"frame_id":"b","path":"b.png"})" "\n";
    return d;
}

std::vector<json> read_lines(const fs::path& p) {
    std::vector<json> v;
    std::ifstream in(p);
    std::string line;
    while (std::getline(in, line))
        if (!line.empty()) v.push_back(json::parse(line));
    return v;
}
}  // namespace

TEST(Harness, CpuRunProducesAllArtifacts) {
    const auto d = setup();
    bench::BenchmarkOptions o;
    o.manifest_path = (d / "m.jsonl").string();
    o.config_overrides_json = R"({"schema_version":1,"resize":{"width":32,"height":24}})";
    o.backends = {"cpu-ref", "cuda"};
    o.scopes = {"host_processing", "device_pipeline", "application", "startup", "batch"};
    o.warmup = 1;
    o.iterations = 3;
    o.repetitions = 2;
    o.output_dir = (d / "out").string();
    o.command_line = "test";
    const int rc = bench::run_benchmark(o);
    const bool cuda_usable = query_backend(Backend::Cuda).usable;
    if (!cuda_usable) {
        EXPECT_EQ(rc, 0) << "unavailable optional backend must not fail the CPU run";
    }
    for (const char* f : {"run.json", "environment.json", "resolved_config.json", "inputs.jsonl", "validation.json",
                          "samples.jsonl", "summary.json", "summary.csv", "analysis.md"}) {
        EXPECT_TRUE(fs::exists(d / "out" / f)) << f;
    }
    EXPECT_FALSE(fs::exists(d / "out" / "scratch"));
    const auto samples = read_lines(d / "out" / "samples.jsonl");
    // cpu-ref: host_processing 2 frames*3 iters + application 2*3 + startup 1 + batch 1 = 14 per repetition
    std::size_t cpu = 0, gpu = 0;
    for (const auto& s : samples) {
        if (s["backend"] == "cpu-ref") ++cpu;
        if (s["backend"] == "cuda") ++gpu;
        EXPECT_GT(s["duration_ms"].get<double>(), 0.0);
        if (s["backend"] != "cuda") {
            EXPECT_NE(s["metric"], "gpu_device_pipeline_ms") << "device_pipeline must not be synthesized for CPU";
        }
        if (s["backend"] == "cpu-ref" && s["scope"] == "host_processing") {
            EXPECT_EQ(s["metric"], "cpu_core_wall_ms");
        }
        for (const char* k : {"run_id", "backend", "implementation", "metric", "frame_id", "input_width", "output_width",
                              "repetition", "iteration", "duration_ms", "success", "math_config_hash"})
            EXPECT_TRUE(s.contains(k)) << k;
    }
    EXPECT_EQ(cpu, 2u * 14u);
    if (!cuda_usable) {
        EXPECT_EQ(gpu, 0u) << "no GPU samples may exist when CUDA is unavailable";
        const auto summary = json::parse(std::ifstream(d / "out" / "summary.json"));
        bool found = false;
        for (const auto& b : summary["backends"]) {
            if (b["backend"] == "cuda") {
                found = true;
                EXPECT_EQ(b["status"], "unavailable");
            }
        }
        EXPECT_TRUE(found);
        for (const auto& r : summary["ratios"]) EXPECT_TRUE(r["host_processing_speedup_vs_scalar_cpu_ref"].is_null());
        const auto val = json::parse(std::ifstream(d / "out" / "validation.json"));
        for (const auto& b : val["backends"])
            if (b["backend"] == "cuda") {
                EXPECT_EQ(b["status"], "not_executed");
            }
    }
    const auto run = json::parse(std::ifstream(d / "out" / "run.json"));
    EXPECT_EQ(run["backend_run_order"].size(), 2u);
    EXPECT_NE(run["backend_run_order"][0]["order"], run["backend_run_order"][1]["order"]) << "order must rotate";
}

TEST(Harness, RefusesToOverwrite) {
    const auto d = setup();
    fs::create_directories(d / "out");
    bench::BenchmarkOptions o;
    o.manifest_path = (d / "m.jsonl").string();
    o.backends = {"cpu-ref"};
    o.scopes = {"host_processing"};
    o.output_dir = (d / "out").string();
    EXPECT_THROW(bench::run_benchmark(o), Error);
}

TEST(Harness, OmpAndOpenCvBaselineValidatedAndLabeled) {
    const auto d = setup();
    bench::BenchmarkOptions o;
    o.manifest_path = (d / "m.jsonl").string();
    o.config_overrides_json = R"({"schema_version":1,"resize":{"width":32,"height":24}})";
    o.backends = {"cpu-ref", "cpu-omp", "opencv-cpu"};
    o.scopes = {"host_processing"};
    o.warmup = 0;
    o.iterations = 2;
    o.repetitions = 1;
    o.omp_threads = 2;
    o.output_dir = (d / "out").string();
    bench::run_benchmark(o);
    const auto val = json::parse(std::ifstream(d / "out" / "validation.json"));
    for (const auto& b : val["backends"]) {
        if (b["backend"] == "cpu-omp" && query_backend(Backend::CpuParallel).usable) {
            EXPECT_EQ(b["status"], "passed");
        }
        if (b["backend"] == "opencv-cpu") {
            EXPECT_TRUE(b.contains("matched_semantics"));
        }
    }
    const auto summary = json::parse(std::ifstream(d / "out" / "summary.json"));
    for (const auto& b : summary["backends"]) {
        if (b["backend"] == "opencv-cpu") {
            EXPECT_EQ(b["threads"], 1);
        }
        if (b["backend"] == "cpu-omp" && b["status"] == "ok") {
            EXPECT_EQ(b["threads"], 2);
        }
    }
}

TEST(Harness, BadArguments) {
    bench::BenchmarkOptions o;
    o.manifest_path = "x";
    o.output_dir = "y";
    o.backends = {"gpu"};
    o.scopes = {"host_processing"};
    EXPECT_THROW(bench::run_benchmark(o), Error);
    o.backends = {"cpu-ref"};
    o.scopes = {"gpu_time"};
    EXPECT_THROW(bench::run_benchmark(o), Error);
}
