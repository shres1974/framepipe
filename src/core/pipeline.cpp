#include "framepipe/pipeline.hpp"

#include "core/backend.hpp"
#include "framepipe/build_info.hpp"
#include "framepipe/errors.hpp"

namespace framepipe {

const char* to_string(Stage s) noexcept {
    switch (s) {
        case Stage::Gray: return "gray";
        case Stage::Resized: return "resize";
        case Stage::BlurHorizontal: return "blur_horizontal";
        case Stage::BlurVertical: return "blur";
        case Stage::SobelX: return "sobel_x";
        case Stage::SobelY: return "sobel_y";
        case Stage::Edges: return "edges";
    }
    return "unknown";
}

std::vector<Stage> stages_for(OutputStage s, bool include_sobel_components) {
    std::vector<Stage> v{Stage::Gray, Stage::Resized};
    if (s == OutputStage::Resize) return v;
    v.push_back(Stage::BlurHorizontal);
    v.push_back(Stage::BlurVertical);
    if (s == OutputStage::Blur) return v;
    if (include_sobel_components) {
        v.push_back(Stage::SobelX);
        v.push_back(Stage::SobelY);
    }
    v.push_back(Stage::Edges);
    return v;
}

const HostFloatImage* StageCapture::get(Stage s) const noexcept {
    const auto& o = images[static_cast<std::size_t>(s)];
    return o ? &*o : nullptr;
}

HostFloatImage& StageCapture::emplace(Stage s, int width, int height) {
    auto& o = images[static_cast<std::size_t>(s)];
    o.emplace(width, height);
    return *o;
}

const char* backend_name(Backend b) noexcept { return to_string(b); }

#if !FRAMEPIPE_HAVE_CUDA
std::unique_ptr<BackendImpl> make_cuda_backend(const PipelineConfig&) {
    throw Error(ErrorCategory::BackendUnavailable,
                "the cuda backend was not built (configure with -DFRAMEPIPE_ENABLE_CUDA=ON)", {"select", "cuda", "", ""});
}
BackendAvailability cuda_availability(int) {
    return {false, false, "not built: configured with FRAMEPIPE_ENABLE_CUDA=OFF"};
}
CudaRuntimeReport cuda_runtime_report() {
    CudaRuntimeReport r;
    r.built = false;
    r.error = "CUDA backend not built";
    return r;
}
#endif

BackendAvailability query_backend(Backend b, int device_index) {
    switch (b) {
        case Backend::CpuReference: return {true, true, "always available"};
        case Backend::CpuParallel:
            if (FRAMEPIPE_HAVE_OPENMP) return {true, true, "OpenMP available"};
            return {false, false, "not built: OpenMP was not found or FRAMEPIPE_ENABLE_OPENMP=OFF"};
        case Backend::Cuda: return cuda_availability(device_index);
        case Backend::Auto: return {true, true, "selects cuda when usable, otherwise cpu-ref"};
    }
    return {false, false, "unknown backend"};
}

CudaRuntimeReport query_cuda_runtime() { return cuda_runtime_report(); }

Pipeline::Pipeline(PipelineConfig config) {
    validate_config(config);
    resolved_.config = config;
    resolved_.math_config_hash = math_config_hash(config);
    Backend chosen = config.backend;
    switch (config.backend) {
        case Backend::Auto: {
            const auto cuda = cuda_availability(config.device_index);
            if (cuda.usable) {
                chosen = Backend::Cuda;
                resolved_.selection_reason = "auto: CUDA device " + std::to_string(config.device_index) + " is usable";
            } else {
                chosen = Backend::CpuReference;
                resolved_.selection_reason = "auto: CUDA unavailable (" + cuda.reason + "); selected cpu-ref";
            }
            break;
        }
        case Backend::CpuReference:
            resolved_.selection_reason = "explicitly requested";
            break;
        case Backend::CpuParallel: {
            const auto a = query_backend(Backend::CpuParallel);
            if (!a.usable) throw Error(ErrorCategory::BackendUnavailable, "cpu-omp " + a.reason, {"select", "cpu-omp", "", ""});
            resolved_.selection_reason = "explicitly requested";
            break;
        }
        case Backend::Cuda: {
            // Explicit CUDA requests never fall back to the CPU.
            const auto a = cuda_availability(config.device_index);
            if (!a.usable) {
                throw Error(ErrorCategory::BackendUnavailable, "cuda backend requested but unusable: " + a.reason,
                            {"select", "cuda", "", ""});
            }
            resolved_.selection_reason = "explicitly requested";
            break;
        }
    }
    resolved_.selected_backend = chosen;
    if (chosen == Backend::Cuda) {
        backend_ = make_cuda_backend(config);
    } else {
        backend_ = make_cpu_backend(config, chosen == Backend::CpuParallel);
    }
    resolved_.implementation = backend_->implementation();
}

Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;

void Pipeline::prepare(const InputSpec& input) {
    const auto& c = resolved_.config;
    validate_dimensions(input.width, input.height, c.resources.max_dimension, "input image");
    if (input.format != PixelFormat::Gray8 && input.format != PixelFormat::Rgb8) {
        throw Error(ErrorCategory::UnsupportedFormat, "input pixel format must be gray8 or rgb8");
    }
    prepared_.reset();  // a failed prepare leaves the pipeline unprepared
    try {
        backend_->prepare(input);
    } catch (Error& e) {
        throw Error(e.category(), e.message(),
                    {"prepare", backend_name(resolved_.selected_backend), shape_string(input), e.context().detail});
    }
    prepared_ = input;
}

void Pipeline::require_prepared(const char* op) const {
    if (!prepared_) {
        throw Error(ErrorCategory::InvalidArgument, "pipeline must be prepared before processing",
                    {op, backend_name(resolved_.selected_backend), "", ""});
    }
}

namespace {
void check_input_matches(const InputImageView& in, const InputSpec& prepared, int max_dim, const char* op,
                         Backend b) {
    validate_view(in, max_dim);
    if (spec_of(in) != prepared) {
        throw Error(ErrorCategory::InvalidImage,
                    "input shape " + shape_string(spec_of(in)) + " does not match the prepared shape " +
                        shape_string(prepared) + "; call prepare() for the new shape",
                    {op, backend_name(b), shape_string(spec_of(in)), ""});
    }
}
}  // namespace

void Pipeline::process_into(InputImageView input, FloatImageView output) {
    require_prepared("process_into");
    const auto& c = resolved_.config;
    check_input_matches(input, *prepared_, c.resources.max_dimension, "process_into", resolved_.selected_backend);
    validate_view(output);
    if (output.width != c.resize.width || output.height != c.resize.height) {
        throw Error(ErrorCategory::InvalidImage,
                    "output view is " + shape_string(output.width, output.height) + " but the configuration requires " +
                        shape_string(c.resize.width, c.resize.height),
                    {"process_into", backend_name(resolved_.selected_backend), shape_string(*prepared_), ""});
    }
    if (byte_ranges_overlap(input.data, input.extent_bytes, output.data, output.extent_bytes)) {
        throw Error(ErrorCategory::InvalidImage, "input and output buffers overlap",
                    {"process_into", backend_name(resolved_.selected_backend), shape_string(*prepared_), ""});
    }
    backend_->process_into(input, output);
}

StageCapture Pipeline::process_capture_stages(InputImageView input) {
    require_prepared("process_capture_stages");
    check_input_matches(input, *prepared_, resolved_.config.resources.max_dimension, "process_capture_stages",
                        resolved_.selected_backend);
    return backend_->capture(input);
}

bool Pipeline::supports_pipelined_batch() const noexcept { return backend_->supports_pipelined_batch(); }

void Pipeline::process_batch(const BatchNextFn& next, const BatchEmitFn& emit) {
    require_prepared("process_batch");
    const auto& c = resolved_.config;
    const int max_dim = c.resources.max_dimension;
    const Backend b = resolved_.selected_backend;
    const InputSpec prepared = *prepared_;
    // Wrap the caller's pullers to enforce the same per-frame shape/view
    // validation the synchronous process_into path applies.
    BatchNext bnext = [&](BatchFrame& out) -> bool {
        InputImageView in;
        std::int64_t seq = 0;
        if (!next(in, seq)) return false;
        check_input_matches(in, prepared, max_dim, "process_batch", b);
        out.input = in;
        out.sequence = seq;
        return true;
    };
    BatchEmit bemit = [&](std::int64_t seq, ConstFloatImageView view) { emit(seq, view); };
    backend_->process_batch(bnext, bemit);
}

Backend Pipeline::selected_backend() const noexcept { return resolved_.selected_backend; }
const ResolvedConfiguration& Pipeline::configuration() const noexcept { return resolved_; }
bool Pipeline::is_prepared() const noexcept { return prepared_.has_value(); }
InputSpec Pipeline::prepared_input() const {
    require_prepared("prepared_input");
    return *prepared_;
}
int Pipeline::output_width() const noexcept { return resolved_.config.resize.width; }
int Pipeline::output_height() const noexcept { return resolved_.config.resize.height; }
WorkspaceInfo Pipeline::workspace() const { return backend_->workspace(); }

bool Pipeline::supports_device_pipeline_timing() const noexcept { return backend_->supports_device_pipeline_timing(); }
void Pipeline::device_resident_upload(InputImageView input) {
    require_prepared("device_resident_upload");
    check_input_matches(input, *prepared_, resolved_.config.resources.max_dimension, "device_resident_upload",
                        resolved_.selected_backend);
    backend_->device_resident_upload(input);
}
double Pipeline::device_resident_run_timed_ms() {
    require_prepared("device_resident_run_timed_ms");
    return backend_->device_resident_run_timed_ms();
}
void Pipeline::device_resident_download(FloatImageView output) {
    require_prepared("device_resident_download");
    validate_view(output);
    if (output.width != output_width() || output.height != output_height()) {
        throw Error(ErrorCategory::InvalidImage, "output view dimensions do not match configuration");
    }
    backend_->device_resident_download(output);
}

}  // namespace framepipe
