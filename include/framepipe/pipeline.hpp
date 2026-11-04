#pragma once
// Public synchronous processing API (spec Section 10). No CUDA types appear here.
//
// Ownership: the Pipeline owns intermediate workspace and immutable parameter
// tables. The caller owns input and output buffers. process_into is
// synchronous: on successful return the output is ready for host access and
// the input may be reused. A Pipeline instance is NOT thread-safe; use one
// instance per caller. Independent instances may run concurrently.

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "framepipe/config.hpp"
#include "framepipe/image.hpp"
#include "framepipe/image_view.hpp"

namespace framepipe {

enum class Stage { Gray, Resized, BlurHorizontal, BlurVertical, SobelX, SobelY, Edges };
constexpr std::size_t kStageCount = 7;
const char* to_string(Stage s) noexcept;

// Stages computed for an output-stage selection, in execution order.
std::vector<Stage> stages_for(OutputStage s, bool include_sobel_components);

// Host copies of intermediate stages from the validation path.
struct StageCapture {
    std::array<std::optional<HostFloatImage>, kStageCount> images;
    const HostFloatImage* get(Stage s) const noexcept;
    HostFloatImage& emplace(Stage s, int width, int height);
};

struct ResolvedConfiguration {
    PipelineConfig config;
    Backend selected_backend = Backend::CpuReference;
    std::string selection_reason;
    std::string implementation;  // e.g. "scalar", "openmp(threads=4)", "cuda-basic"
    std::string math_config_hash;
};

struct WorkspaceInfo {
    std::size_t host_logical_bytes = 0;
    std::size_t host_allocated_bytes = 0;
    std::size_t device_logical_bytes = 0;
    std::size_t device_allocated_bytes = 0;
    std::size_t pinned_host_bytes = 0;
    std::string transfer_mode;  // CUDA: "pinned_staging" or "pageable_direct"; CPU: ""
};

class BackendImpl;  // private

class Pipeline {
public:
    // Resolves the backend. An explicit Cuda request never falls back to CPU;
    // it throws Error(BackendUnavailable). Auto selects CUDA when usable.
    explicit Pipeline(PipelineConfig config);
    ~Pipeline();
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Allocates workspace and parameter tables for one input shape/format.
    // Calling again with a different shape replaces the workspace.
    void prepare(const InputSpec& input);

    // Runs the configured prefix of the pipeline. Output dimensions must equal
    // (resize.width, resize.height). Input and output must not overlap.
    void process_into(InputImageView input, FloatImageView output);

    // Validation path: computes every selected stage plus signed Sobel
    // components (for edges) and returns host copies. Not for benchmarks.
    StageCapture process_capture_stages(InputImageView input);

    Backend selected_backend() const noexcept;
    const ResolvedConfiguration& configuration() const noexcept;
    bool is_prepared() const noexcept;
    InputSpec prepared_input() const;
    int output_width() const noexcept;
    int output_height() const noexcept;
    WorkspaceInfo workspace() const;

    // Device-resident timing hooks (CUDA only). These exist so the benchmark
    // can measure gpu_device_pipeline_ms with input already on the device.
    bool supports_device_pipeline_timing() const noexcept;
    void device_resident_upload(InputImageView input);
    double device_resident_run_timed_ms();  // GPU-event interval, stop event synchronized
    void device_resident_download(FloatImageView output);

private:
    void require_prepared(const char* op) const;
    std::unique_ptr<BackendImpl> backend_;
    ResolvedConfiguration resolved_;
    std::optional<InputSpec> prepared_;
};

const char* backend_name(Backend b) noexcept;

struct BackendAvailability {
    bool built = false;
    bool usable = false;
    std::string reason;
};
BackendAvailability query_backend(Backend b, int device_index = 0);

struct CudaDeviceInfo {
    int index = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t total_memory_bytes = 0;
    int multiprocessors = 0;
};

struct CudaRuntimeReport {
    bool built = false;
    std::string toolkit_version;     // compile-time CUDA version, empty if not built
    std::string architectures;       // CMAKE_CUDA_ARCHITECTURES at build time
    std::optional<int> runtime_version;
    std::optional<int> driver_version;
    int device_count = 0;
    std::string error;               // empty on success
    std::vector<CudaDeviceInfo> devices;
};
CudaRuntimeReport query_cuda_runtime();

}  // namespace framepipe
