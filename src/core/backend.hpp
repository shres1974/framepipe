#pragma once
// Private backend interface. The caller selects a backend once; the pipeline
// prepares resources; repeated frames reuse them (spec Section 5).

#include <functional>
#include <memory>
#include <string>

#include "framepipe/errors.hpp"
#include "framepipe/pipeline.hpp"

namespace framepipe {

// Optional pipelined-batch interface (optimization: transfer/compute overlap).
//
// A backend that can keep several frames in flight (e.g. CUDA with multiple
// streams and per-slot buffers) overrides supports_pipelined_batch() to return
// true and implements process_batch(). The batch runner drives it: `next`
// yields the frames to process in submission order, and `emit` is called once
// per frame, also in submission order, with the processed output.
//
// Contracts that a pipelined implementation must honor so the runner keeps its
// existing semantics:
//   * emit() is called in the SAME order frames were produced by next().
//   * On any error, the pipeline drains all in-flight work before the
//     exception propagates; no output buffer is left half-written.
//   * The prepared shape applies to every frame; a shape change is handled by
//     the runner draining and re-preparing, exactly as in the synchronous path.
struct BatchFrame {
    InputImageView input;
    std::int64_t sequence = 0;  // opaque token echoed back to emit()
};
// Returns false when there are no more frames. On true, `out` is populated.
using BatchNext = std::function<bool(BatchFrame& out)>;
// Receives one completed frame. The FloatImageView is valid only for the
// duration of the call (it aliases backend-owned staging); copy out if needed.
using BatchEmit = std::function<void(std::int64_t sequence, ConstFloatImageView output)>;

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

    // Pipelined batch processing with transfer/compute overlap. Default: not
    // supported, and the runner falls back to repeated process_into() calls.
    // When supported, the runner calls process_batch() instead, which pulls
    // frames via `next` and reports results via `emit`, both in submission
    // order. The backend must be prepared() for the common shape first.
    virtual bool supports_pipelined_batch() const noexcept { return false; }
    virtual void process_batch(const BatchNext& /*next*/, const BatchEmit& /*emit*/) {
        throw Error(ErrorCategory::BackendUnavailable, "pipelined batch is not supported by this backend");
    }

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
