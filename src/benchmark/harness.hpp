#pragma once
// Benchmark harness (spec Sections 17-19).
//
// Metric names (never an ambiguous "GPU time"):
//   cpu_core_wall_ms                CPU backends, scope host_processing
//   gpu_transfer_inclusive_wall_ms  CUDA, scope host_processing
//   gpu_device_pipeline_ms          CUDA, scope device_pipeline (GPU events)
//   application_wall_ms             scope application (read..encode+close)
//   startup_wall_ms                 scope startup (backend creation..prepared)
//   batch_wall_ms                   scope batch (whole manifest, one sample per repetition)

#include <string>
#include <vector>

namespace framepipe::bench {

struct BenchmarkOptions {
    std::string manifest_path;
    std::string config_path;          // may be empty: built-in defaults
    std::string config_overrides_json;  // CLI overrides applied after the file, may be empty
    std::vector<std::string> backends;  // cpu-ref, cpu-omp, cuda, opencv-cpu
    std::vector<std::string> scopes;    // host_processing, device_pipeline, application, startup, batch
    std::string workload = "repeat";    // "repeat": each frame measured repeatedly; "ring": cycle frames
    int warmup = 20;
    int iterations = 200;
    int repetitions = 5;
    std::string output_dir;
    bool overwrite = false;
    std::string command_line;
    int omp_threads = 0;     // threads for cpu-omp; 0 = config cpu.threads
    int opencv_threads = 1;  // OpenCV baseline thread policy, recorded separately
};

// Returns a process exit code. Writes the result directory.
int run_benchmark(const BenchmarkOptions& opt);

}  // namespace framepipe::bench
