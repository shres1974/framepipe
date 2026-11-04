#pragma once
// Private backend interface. The caller selects a backend once; the pipeline
// prepares resources; repeated frames reuse them (spec Section 5).

#include <memory>
#include <string>

#include "framepipe/errors.hpp"
#include "framepipe/pipeline.hpp"

namespace framepipe {

class BackendImpl {
public:
    virtual ~BackendImpl() = default;
    virtual Backend kind() const noexcept = 0;
    virtual std::string implementation() const = 0;

    // Allocates workspace/tables for `input`. Must leave the backend in an
    // unprepared state if it throws (strong cleanup, no leaks).
    virtual void prepare(const InputSpec& input) = 0;
    virtual void process_into(const InputImageView& input, const FloatImageView& output) = 0;
    virtual StageCapture capture(const InputImageView& input) = 0;
    virtual WorkspaceInfo workspace() const = 0;

    virtual bool supports_device_pipeline_timing() const noexcept { return false; }
    virtual void device_resident_upload(const InputImageView&) {
        throw Error(ErrorCategory::BackendUnavailable, "device-resident timing is only available on GPU backends");
    }
    virtual double device_resident_run_timed_ms() {
        throw Error(ErrorCategory::BackendUnavailable, "device-resident timing is only available on GPU backends");
    }
    virtual void device_resident_download(const FloatImageView&) {
        throw Error(ErrorCategory::BackendUnavailable, "device-resident timing is only available on GPU backends");
    }
};

std::unique_ptr<BackendImpl> make_cpu_backend(const PipelineConfig& config, bool parallel);

// Provided by src/cuda when FRAMEPIPE_HAVE_CUDA, otherwise stubs in pipeline.cpp.
std::unique_ptr<BackendImpl> make_cuda_backend(const PipelineConfig& config);
BackendAvailability cuda_availability(int device_index);
CudaRuntimeReport cuda_runtime_report();

}  // namespace framepipe
