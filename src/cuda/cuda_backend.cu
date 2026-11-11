// Basic custom CUDA backend (spec Section 12). One explicit non-blocking
// stream; persistent pitched device workspace allocated in prepare(); a
// two-buffer output-resolution workspace; per-instance device parameter
// tables (no process-wide constant memory); synchronous facade that waits on
// the owned stream (never the whole device) before returning.

#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "core/backend.hpp"
#include "core/tables.hpp"
#include "cuda/cuda_raii.hpp"
#include "cuda/kernels.cuh"

namespace framepipe {

namespace {

using namespace framepipe::cuda;

constexpr std::size_t kPitchEstimate = 512;  // conservative estimate for preflight only

std::size_t round_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

class CudaBackend final : public BackendImpl {
public:
    explicit CudaBackend(const PipelineConfig& c) : config_(c) {
        check(cudaSetDevice(c.device_index), "select_device");
        stream_ = Stream("stream_create");
        start_ = Event("event_create");
        stop_ = Event("event_create");
    }

    ~CudaBackend() override {
        // Complete owned work before members are released (Stream::reset also syncs).
        if (stream_.get()) (void)cudaStreamSynchronize(stream_.get());
    }

    Backend kind() const noexcept override { return Backend::Cuda; }
    std::string implementation() const override { return "cuda-basic(block=16x16)"; }

    void prepare(const InputSpec& in) override {
        set_device();
        if (stream_.get()) check(cudaStreamSynchronize(stream_.get()), "prepare_drain", shape_);
        release_workspace();  // replace, not grow a cache of shapes
        shape_ = shape_string(in);
        const int wo = config_.resize.width, ho = config_.resize.height;
        const int ch = channel_count(in.format);
        const std::size_t in_row = static_cast<std::size_t>(in.width) * static_cast<std::size_t>(ch);

        // ---- preflight memory budget (estimate with conservative pitch)
        const std::size_t est_in = round_up(in_row, kPitchEstimate) * static_cast<std::size_t>(in.height);
        const std::size_t est_gray =
            round_up(static_cast<std::size_t>(in.width) * 4, kPitchEstimate) * static_cast<std::size_t>(in.height);
        const std::size_t est_out =
            round_up(static_cast<std::size_t>(wo) * 4, kPitchEstimate) * static_cast<std::size_t>(ho);
        const std::size_t table_bytes =
            static_cast<std::size_t>(wo + ho) * 12 + static_cast<std::size_t>(2 * config_.gaussian.radius + 1) * 4;
        const std::size_t est_total = est_in + est_gray + 2 * est_out + table_bytes;
        if (est_total > config_.resources.max_device_bytes) {
            throw Error(ErrorCategory::AllocationFailure,
                        "estimated device workspace " + std::to_string(est_total) +
                            " bytes exceeds resources.max_device_bytes " +
                            std::to_string(config_.resources.max_device_bytes),
                        {"prepare", "cuda", shape_, ""});
        }
        std::size_t free_b = 0, total_b = 0;
        check(cudaMemGetInfo(&free_b, &total_b), "mem_get_info", shape_);
        if (est_total + (64u << 20) > free_b) {  // leave headroom for runtime overhead
            throw Error(ErrorCategory::AllocationFailure,
                        "estimated device workspace " + std::to_string(est_total) +
                            " bytes does not fit in currently free device memory " + std::to_string(free_b),
                        {"prepare", "cuda", shape_, ""});
        }

        // ---- allocate into locals; commit only on full success
        DeviceImage d_in(in.width * ch, in.height, 1, "alloc_input");
        DeviceImage d_gray(in.width, in.height, sizeof(float), "alloc_gray");
        DeviceImage d_a(wo, ho, sizeof(float), "alloc_workspace_a");
        DeviceImage d_b(wo, ho, sizeof(float), "alloc_workspace_b");

        const ResizeTables t = build_resize_tables(in.width, in.height, wo, ho);
        const std::vector<float> w = build_gaussian_weights(config_.gaussian.radius, config_.gaussian.sigma);
        DeviceBuffer d_tables(table_bytes, "alloc_tables");
        auto* base = d_tables.as<unsigned char>();
        const std::size_t xw = static_cast<std::size_t>(wo) * 4, yh = static_cast<std::size_t>(ho) * 4;
        DevTables dt{};
        dt.x0 = reinterpret_cast<const std::int32_t*>(base);
        dt.x1 = reinterpret_cast<const std::int32_t*>(base + xw);
        dt.wx = reinterpret_cast<const float*>(base + 2 * xw);
        dt.y0 = reinterpret_cast<const std::int32_t*>(base + 3 * xw);
        dt.y1 = reinterpret_cast<const std::int32_t*>(base + 3 * xw + yh);
        dt.wy = reinterpret_cast<const float*>(base + 3 * xw + 2 * yh);
        const float* dw = reinterpret_cast<const float*>(base + 3 * xw + 3 * yh);
        // Tables are uploaded here, outside steady-state measurement. Copies
        // are enqueued on the owned (non-blocking) stream and then waited on:
        // a plain cudaMemcpy from pageable memory may return before its DMA
        // completes and is not ordered with a non-blocking stream.
        auto up = [&](const void* dst, const void* src, std::size_t n) {
            check(cudaMemcpyAsync(const_cast<void*>(dst), src, n, cudaMemcpyHostToDevice, s()), "upload_tables", shape_);
        };
        try {
            up(dt.x0, t.x.i0.data(), xw);
            up(dt.x1, t.x.i1.data(), xw);
            up(dt.wx, t.x.w.data(), xw);
            up(dt.y0, t.y.i0.data(), yh);
            up(dt.y1, t.y.i1.data(), yh);
            up(dt.wy, t.y.w.data(), yh);
            up(dw, w.data(), w.size() * 4);
            check(cudaStreamSynchronize(s()), "upload_tables_complete", shape_);
        } catch (...) {
            // Host source vectors and device locals are about to be destroyed:
            // drain any enqueued copy first (best effort, original error kept).
            (void)cudaStreamSynchronize(s());
            (void)cudaGetLastError();
            throw;
        }

        // Pinned staging, bounded by max_pinned_host_bytes; otherwise use
        // pageable memory directly (recorded as transfer_mode).
        const std::size_t pin_in = in_row * static_cast<std::size_t>(in.height);
        const std::size_t pin_out = static_cast<std::size_t>(wo) * 4 * static_cast<std::size_t>(ho);
        PinnedBuffer p_in, p_out;
        if (pin_in + pin_out <= config_.resources.max_pinned_host_bytes) {
            p_in = PinnedBuffer(pin_in, "alloc_pinned_input");
            p_out = PinnedBuffer(pin_out, "alloc_pinned_output");
        }

        d_in_ = std::move(d_in);
        d_gray_ = std::move(d_gray);
        d_a_ = std::move(d_a);
        d_b_ = std::move(d_b);
        d_tables_ = std::move(d_tables);
        tables_ = dt;
        d_weights_ = dw;
        pin_in_ = std::move(p_in);
        pin_out_ = std::move(p_out);
        input_ = in;
        prepared_ = true;
    }

    void process_into(const InputImageView& in, const FloatImageView& out) override {
        set_device();
        guard([&] {
            upload(in);
            const DeviceImage& result = enqueue_pipeline();
            download_and_wait(result, out);
        });
    }

    StageCapture capture(const InputImageView& in) override {
        set_device();
        StageCapture cap;
        // Temporaries live outside guard() so that, on failure, guard drains
        // the stream BEFORE any device buffer used by in-flight work is freed.
        std::deque<DeviceImage> tmp;
        auto alloc = [&](int w, int h) -> DeviceImage& { return tmp.emplace_back(w, h, 4, "alloc_capture"); };
        guard([&] {
            upload(in);
            const int wo = config_.resize.width, ho = config_.resize.height, r = config_.gaussian.radius;
            DeviceImage& gray = alloc(in.width, in.height);
            DeviceImage& rs = alloc(wo, ho);
            launch(launch_grayscale(dev_u8(d_in_), dev_f32(gray), s()), "grayscale");
            launch(launch_resize(as_const(dev_f32(gray)), dev_f32(rs), tables_, s()), "resize");
            fetch(cap, Stage::Gray, gray);
            fetch(cap, Stage::Resized, rs);
            if (config_.output_stage == OutputStage::Resize) return;
            DeviceImage& h = alloc(wo, ho);
            DeviceImage& v = alloc(wo, ho);
            launch(launch_gaussian_horizontal(as_const(dev_f32(rs)), dev_f32(h), d_weights_, r, s()), "gaussian_horizontal");
            launch(launch_gaussian_vertical(as_const(dev_f32(h)), dev_f32(v), d_weights_, r, s()), "gaussian_vertical");
            fetch(cap, Stage::BlurHorizontal, h);
            fetch(cap, Stage::BlurVertical, v);
            if (config_.output_stage == OutputStage::Blur) return;
            DeviceImage& gx = alloc(wo, ho);
            DeviceImage& gy = alloc(wo, ho);
            DeviceImage& e = alloc(wo, ho);
            launch(launch_sobel_components(as_const(dev_f32(v)), dev_f32(gx), dev_f32(gy), s()), "sobel_components");
            launch(launch_sobel_edges(as_const(dev_f32(v)), dev_f32(e), s()), "sobel_edges");
            fetch(cap, Stage::SobelX, gx);
            fetch(cap, Stage::SobelY, gy);
            fetch(cap, Stage::Edges, e);
        });
        return cap;
    }

    WorkspaceInfo workspace() const override {
        WorkspaceInfo w;
        const std::size_t tables = d_tables_.size();
        w.device_logical_bytes = d_in_.logical_bytes() + d_gray_.logical_bytes() + d_a_.logical_bytes() +
                                 d_b_.logical_bytes() + tables;
        w.device_allocated_bytes = d_in_.allocated_bytes() + d_gray_.allocated_bytes() + d_a_.allocated_bytes() +
                                   d_b_.allocated_bytes() + tables;
        w.pinned_host_bytes = pin_in_.size() + pin_out_.size();
        w.transfer_mode = pin_in_.size() ? "pinned_staging" : "pageable_direct";
        return w;
    }

    bool supports_device_pipeline_timing() const noexcept override { return true; }

    void device_resident_upload(const InputImageView& in) override {
        set_device();
        guard([&] {
            upload(in);
            check(cudaStreamSynchronize(s()), "upload_complete", shape_);
        });
    }

    double device_resident_run_timed_ms() override {
        set_device();
        float ms = 0.0f;
        guard([&] {
            check(cudaStreamSynchronize(s()), "pre_timing_drain", shape_);  // no outstanding measured work
            check(cudaEventRecord(start_.get(), s()), "event_record", shape_);
            last_result_ = &enqueue_pipeline();
            check(cudaEventRecord(stop_.get(), s()), "event_record", shape_);
            check(cudaEventSynchronize(stop_.get()), "pipeline_complete", shape_);
            check(cudaEventElapsedTime(&ms, start_.get(), stop_.get()), "event_elapsed", shape_);
        });
        return static_cast<double>(ms);
    }

    void device_resident_download(const FloatImageView& out) override {
        set_device();
        if (!last_result_) throw Error(ErrorCategory::InvalidArgument, "no device-resident result to download");
        guard([&] { download_and_wait(*last_result_, out); });
    }

private:
    cudaStream_t s() const noexcept { return stream_.get(); }
    void set_device() { check(cudaSetDevice(config_.device_index), "select_device", shape_); }

    void release_workspace() noexcept {
        prepared_ = false;
        last_result_ = nullptr;
        d_in_.reset();
        d_gray_.reset();
        d_a_.reset();
        d_b_.reset();
        d_tables_.reset();
        pin_in_.reset();
        pin_out_.reset();
    }

    // On any failure: stop dependent work by draining the owned stream
    // (best effort, preserving the original error) and never write output.
    template <class F>
    void guard(F&& f) {
        try {
            f();
        } catch (...) {
            (void)cudaStreamSynchronize(s());
            (void)cudaGetLastError();  // clear non-sticky error state
            last_result_ = nullptr;
            throw;
        }
    }

    void launch(cudaError_t launch_err, const char* stage) {
        check(launch_err, stage, shape_);  // launch-configuration errors, immediately
#if FRAMEPIPE_CUDA_DEBUG_SYNC
        check(cudaStreamSynchronize(s()), stage, shape_);  // diagnostics only
#endif
    }

    DevU8 dev_u8(const DeviceImage& d) const {
        return {static_cast<const std::uint8_t*>(d.ptr()), input_.width, input_.height, d.pitch(),
                channel_count(input_.format)};
    }
    static DevF32 dev_f32(const DeviceImage& d) {
        return {static_cast<float*>(d.ptr()), d.width(), d.height(), d.pitch()};
    }

    // Staging reuse invariant: every public entry point waits on the owned
    // stream before returning (download_and_wait / upload_complete / event
    // sync / guard on error), so pinned staging is never refilled while a
    // previous transfer from it is still in flight.
    void upload(const InputImageView& in) {
        const std::size_t row = static_cast<std::size_t>(in.width) * static_cast<std::size_t>(channel_count(in.format));
        if (pin_in_.size()) {
            // Honor caller stride while packing into pinned staging.
            for (int y = 0; y < in.height; ++y) {
                std::memcpy(pin_in_.data() + static_cast<std::size_t>(y) * row, row_ptr(in, y), row);
            }
            check(cudaMemcpy2DAsync(d_in_.ptr(), d_in_.pitch(), pin_in_.data(), row, row,
                                    static_cast<std::size_t>(in.height), cudaMemcpyHostToDevice, s()),
                  "upload_input", shape_);
        } else {
            check(cudaMemcpy2DAsync(d_in_.ptr(), d_in_.pitch(), in.data, in.stride_bytes, row,
                                    static_cast<std::size_t>(in.height), cudaMemcpyHostToDevice, s()),
                  "upload_input", shape_);
        }
    }

    // Two-buffer schedule. Returns the device image holding the final output.
    const DeviceImage& enqueue_pipeline() {
        const int r = config_.gaussian.radius;
        // Fused grayscale + resize: reads u8 input directly into d_a_, skipping
        // the intermediate d_gray_ write/read. Bit-identical to the separate
        // grayscale and resize kernels (see k_grayscale_resize).
        launch(launch_grayscale_resize(dev_u8(d_in_), dev_f32(d_a_), tables_, s()), "grayscale_resize");
        if (config_.output_stage == OutputStage::Resize) return d_a_;
        launch(launch_gaussian_horizontal(as_const(dev_f32(d_a_)), dev_f32(d_b_), d_weights_, r, s()),
               "gaussian_horizontal");
        launch(launch_gaussian_vertical(as_const(dev_f32(d_b_)), dev_f32(d_a_), d_weights_, r, s()),
               "gaussian_vertical");
        if (config_.output_stage == OutputStage::Blur) return d_a_;
        launch(launch_sobel_edges(as_const(dev_f32(d_a_)), dev_f32(d_b_), s()), "sobel_edges");
        return d_b_;
    }

    void download_and_wait(const DeviceImage& src, const FloatImageView& out) {
        const std::size_t row = static_cast<std::size_t>(out.width) * sizeof(float);
        if (pin_out_.size()) {
            check(cudaMemcpy2DAsync(pin_out_.data(), row, src.ptr(), src.pitch(), row,
                                    static_cast<std::size_t>(out.height), cudaMemcpyDeviceToHost, s()),
                  "download_output", shape_);
            check(cudaStreamSynchronize(s()), "pipeline_complete", shape_);
            for (int y = 0; y < out.height; ++y) {
                std::memcpy(row_ptr(out, y), pin_out_.data() + static_cast<std::size_t>(y) * row, row);
            }
        } else {
            check(cudaMemcpy2DAsync(out.data, out.stride_bytes, src.ptr(), src.pitch(), row,
                                    static_cast<std::size_t>(out.height), cudaMemcpyDeviceToHost, s()),
                  "download_output", shape_);
            check(cudaStreamSynchronize(s()), "pipeline_complete", shape_);
        }
    }

    void fetch(StageCapture& cap, Stage st, const DeviceImage& d) {
        auto& h = cap.emplace(st, d.width(), d.height());
        check(cudaMemcpy2DAsync(h.raw(), h.stride_bytes(), d.ptr(), d.pitch(), d.row_bytes(),
                                static_cast<std::size_t>(d.height()), cudaMemcpyDeviceToHost, s()),
              "capture_download", shape_);
        check(cudaStreamSynchronize(s()), "capture_complete", shape_);
    }

    PipelineConfig config_;
    Stream stream_;
    Event start_, stop_;
    InputSpec input_{};
    std::string shape_;
    bool prepared_ = false;
    DeviceImage d_in_, d_gray_, d_a_, d_b_;
    DeviceBuffer d_tables_;
    DevTables tables_{};
    const float* d_weights_ = nullptr;
    PinnedBuffer pin_in_, pin_out_;
    const DeviceImage* last_result_ = nullptr;
};

}  // namespace

std::unique_ptr<BackendImpl> make_cuda_backend(const PipelineConfig& config) {
    return std::make_unique<CudaBackend>(config);
}

}  // namespace framepipe
