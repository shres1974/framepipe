#pragma once
// Validated pipeline configuration (spec Section 8).

#include <cstdint>
#include <string>

namespace framepipe {

enum class Backend { Auto, CpuReference, CpuParallel, Cuda };
enum class OutputStage { Resize, Blur, Edges };
enum class CudaImplementation { Basic };

const char* to_string(Backend b) noexcept;          // "auto", "cpu-ref", "cpu-omp", "cuda"
const char* to_string(OutputStage s) noexcept;      // "resize", "blur", "edges"
const char* to_string(CudaImplementation i) noexcept;  // "basic"

// Parsers throw Error(InvalidConfiguration) listing accepted values.
Backend parse_backend(const std::string& s);
OutputStage parse_output_stage(const std::string& s);
CudaImplementation parse_cuda_implementation(const std::string& s);

struct ResizeConfig {
    int width = 1280;
    int height = 720;
    std::string mode = "bilinear_half_pixel";
    std::string aspect_policy = "stretch";
};

struct GaussianConfig {
    int radius = 3;
    double sigma = 1.2;
};

struct CpuConfig {
    int threads = 1;
};

struct CudaConfig {
    CudaImplementation implementation = CudaImplementation::Basic;
};

struct ResourceConfig {
    int max_dimension = 16384;
    std::uint64_t max_device_bytes = 1073741824ULL;
    std::uint64_t max_pinned_host_bytes = 268435456ULL;
    int max_in_flight_frames = 1;
};

struct PipelineConfig {
    int schema_version = 1;
    Backend backend = Backend::Auto;
    int device_index = 0;
    OutputStage output_stage = OutputStage::Edges;
    ResizeConfig resize;
    GaussianConfig gaussian;
    std::string border = "replicate";
    CpuConfig cpu;
    CudaConfig cuda;
    ResourceConfig resources;
};

// Supported parameter ranges.
constexpr int kMaxGaussianRadius = 15;
constexpr double kMinSigma = 0.1;
constexpr double kMaxSigma = 100.0;
constexpr int kMaxCpuThreads = 1024;
constexpr int kMaxInFlightFrames = 8;

// Throws Error(InvalidConfiguration) describing the first violation.
void validate_config(const PipelineConfig& c);

// Applies a JSON document on top of `base`. Keys present override `base`;
// unknown keys, wrong types, and unsupported enum values are errors. The
// result is validated before return.
PipelineConfig apply_config_json(const std::string& json_text, const PipelineConfig& base = PipelineConfig{});

// Fully resolved JSON (all fields), stable key order.
std::string config_to_json(const PipelineConfig& c, int indent = 2);

// SHA-256 over a canonical encoding of every parameter that affects intended
// mathematical output (stage, resize, gaussian, border). Backend and
// performance settings are excluded by design and recorded separately.
std::string math_config_hash(const PipelineConfig& c);

}  // namespace framepipe
