// Basic custom CUDA backend (spec Section 12). The synchronous facade uses one
// explicit non-blocking stream and a single workspace slot; it waits on that
// stream (never the whole device) before returning. The pipelined batch path
// (process_batch) adds N independent slots, each with its own stream, device
// workspace, and pinned staging, and overlaps upload/compute/download across
// frames to hide the PCIe transfer that dominates per-frame GPU time.

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
constexpr int kDefaultPipelineDepth = 3;     // slots kept in flight by process_batch

std::size_t round_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

// One independent unit of work: its own stream, device workspace, parameter
// tables, pinned staging, and timing events. The synchronous facade uses a
// single Slot; the pipelined batch path uses several. Device parameter tables
// are per-slot so slots never share mutable device state.
struct Slot {
    Stream stream;
    Event start, stop;
    DeviceImage d_in, d_gray, d_a, d_b;
    DeviceBuffer d_tables;
    DevTables tables{};
    const float* d_weights = nullptr;
    PinnedBuffer pin_in, pin_out;

    void reset() noexcept {
        if (stream.get()) (void)cudaStreamSynchronize(stream.get());
        d_in.reset();
        d_gray.reset();
        d_a.reset();
        d_b.reset();
        d_tables.reset();
        pin_in.reset();
        pin_out.reset();
        d_weights = nullptr;
    }
};

class CudaBackend final : public BackendImpl {
public:
    explicit CudaBackend(const PipelineConfig& c) : config_(c) {
        check(cudaSetDevice(c.device_index), "select_device");
        // Slot 0 is the primary slot used by the synchronous facade.
        slots_.resize(1);
        slots_[0].stream = Stream("stream_create");
        slots_[0].start = Event("event_create");
        slots_[0].stop = Event("event_create");
    }

    ~CudaBackend() override {
        for (auto& sl : slots_)
            if (sl.stream.get()) (void)cudaStreamSynchronize(sl.stream.get());
    }

    Backend kind() const noexcept override { return Backend::Cuda; }
    std::string implementation() const override { return "cuda-basic(block=16x16)"; }

    void prepare(const InputSpec& in) override {
        set_device();
        drain_all();
        release_extra_slots();  // steady state keeps a single prepared slot
        release_slot(slots_[0]);
        shape_ = shape_string(in);
        input_ = in;
        allocate_slot(slots_[0]);
        prepared_ = true;
    }

    void process_into(const InputImageView& in, const FloatImageView& out) override {
        set_device();
        Slot& sl = slots_[0];
        guard(sl, [&] {
            upload(sl, in);
            const DeviceImage& result = enqueue_pipeline(sl);
            download_and_wait(sl, result, out);
        });
    }

    bool supports_pipelined_batch() const noexcept override { return true; }

    // Software pipeline over `depth_` slots. Keeps up to depth_ frames in
    // flight: while slot i computes, slot i-1 downloads and slot i+1 uploads.
    // emit() is called in submission order. On error, every in-flight slot is
    // drained before the exception propagates and no result is emitted for an
    // incomplete frame.
    void process_batch(const BatchNext& next, const BatchEmit& emit) override {
        if (!prepared_) throw Error(ErrorCategory::InvalidArgument, "process_batch requires a prepared pipeline");
        set_device();
        const int depth = ensure_slots();

        // A frame whose kernels have been enqueued on a slot and whose output
        // download is pending. We finish (wait + emit) in submission order.
        struct InFlight {
            int slot = -1;
            std::int64_t sequence = 0;
            const DeviceImage* result = nullptr;
        };
        std::deque<InFlight> inflight;

        auto finish_oldest = [&] {
            InFlight f = inflight.front();
            inflight.pop_front();
            Slot& sl = slots_[static_cast<std::size_t>(f.slot)];
            // Output staging for this slot; copy out within emit()'s lifetime.
            const std::size_t row = static_cast<std::size_t>(config_.resize.width) * sizeof(float);
            if (sl.pin_out.size()) {
                check(cudaMemcpy2DAsync(sl.pin_out.data(), row, f.result->ptr(), f.result->pitch(), row,
                                        static_cast<std::size_t>(config_.resize.height), cudaMemcpyDeviceToHost,
                                        sl.stream.get()),
                      "download_output", shape_);
            }
            check(cudaStreamSynchronize(sl.stream.get()), "pipeline_complete", shape_);
            ConstFloatImageView view;
            view.width = config_.resize.width;
            view.height = config_.resize.height;
            if (sl.pin_out.size()) {
                view.data = reinterpret_cast<const float*>(sl.pin_out.data());
                view.stride_bytes = row;
            } else {
                // No pinned staging: read directly from the device is not
                // allowed on the host, so fall back to a synchronous 2D copy
                // into a per-slot host buffer.
                host_fallback_.assign(static_cast<std::size_t>(config_.resize.height) * row / sizeof(float), 0.0f);
                check(cudaMemcpy2D(host_fallback_.data(), row, f.result->ptr(), f.result->pitch(), row,
                                   static_cast<std::size_t>(config_.resize.height), cudaMemcpyDeviceToHost),
                      "download_output", shape_);
                view.data = host_fallback_.data();
                view.stride_bytes = row;
            }
            view.extent_bytes = static_cast<std::size_t>(config_.resize.height) * row;
            emit(f.sequence, view);
        };

        try {
            BatchFrame frame;
            bool more = next(frame);
            while (more || !inflight.empty()) {
                // Launch new work while frames remain and a slot is free.
                while (more && static_cast<int>(inflight.size()) < depth) {
                    const int slot = next_slot_;
                    next_slot_ = (next_slot_ + 1) % depth;
                    Slot& sl = slots_[static_cast<std::size_t>(slot)];
                    // The slot must be idle before we refill its staging. It is
                    // idle because we never keep more than `depth` frames in
                    // flight and finish them in order, but synchronize to be safe
                    // when the ring wraps.
                    check(cudaStreamSynchronize(sl.stream.get()), "slot_reuse_drain", shape_);
                    upload(sl, frame.input);
                    const DeviceImage& result = enqueue_pipeline(sl);
                    inflight.push_back({slot, frame.sequence, &result});
                    more = next(frame);
                }
                if (!inflight.empty()) finish_oldest();
            }
        } catch (...) {
            drain_all();
            (void)cudaGetLastError();
            throw;
        }
    }

    StageCapture capture(const InputImageView& in) override {
        set_device();
        Slot& sl = slots_[0];
        StageCapture cap;
        std::deque<DeviceImage> tmp;
        auto alloc = [&](int w, int h) -> DeviceImage& { return tmp.emplace_back(w, h, 4, "alloc_capture"); };
        guard(sl, [&] {
            upload(sl, in);
            const int wo = config_.resize.width, ho = config_.resize.height, r = config_.gaussian.radius;
            DeviceImage& gray = alloc(in.width, in.height);
            DeviceImage& rs = alloc(wo, ho);
            launch(sl, launch_grayscale(dev_u8(sl.d_in), dev_f32(gray), sl.stream.get()), "grayscale");
            launch(sl, launch_resize(as_const(dev_f32(gray)), dev_f32(rs), sl.tables, sl.stream.get()), "resize");
            fetch(sl, cap, Stage::Gray, gray);
            fetch(sl, cap, Stage::Resized, rs);
            if (config_.output_stage == OutputStage::Resize) return;
            DeviceImage& h = alloc(wo, ho);
            DeviceImage& v = alloc(wo, ho);
            launch(sl, launch_gaussian_horizontal(as_const(dev_f32(rs)), dev_f32(h), sl.d_weights, r, sl.stream.get()),
                   "gaussian_horizontal");
            launch(sl, launch_gaussian_vertical(as_const(dev_f32(h)), dev_f32(v), sl.d_weights, r, sl.stream.get()),
                   "gaussian_vertical");
            fetch(sl, cap, Stage::BlurHorizontal, h);
            fetch(sl, cap, Stage::BlurVertical, v);
            if (config_.output_stage == OutputStage::Blur) return;
            DeviceImage& gx = alloc(wo, ho);
            DeviceImage& gy = alloc(wo, ho);
            DeviceImage& e = alloc(wo, ho);
            launch(sl, launch_sobel_components(as_const(dev_f32(v)), dev_f32(gx), dev_f32(gy), sl.stream.get()),
                   "sobel_components");
            launch(sl, launch_sobel_edges(as_const(dev_f32(v)), dev_f32(e), sl.stream.get()), "sobel_edges");
            fetch(sl, cap, Stage::SobelX, gx);
            fetch(sl, cap, Stage::SobelY, gy);
            fetch(sl, cap, Stage::Edges, e);
        });
        return cap;
    }

    WorkspaceInfo workspace() const override {
        WorkspaceInfo w;
        for (const auto& sl : slots_) {
            const std::size_t tables = sl.d_tables.size();
            w.device_logical_bytes += sl.d_in.logical_bytes() + sl.d_gray.logical_bytes() + sl.d_a.logical_bytes() +
                                      sl.d_b.logical_bytes() + tables;
            w.device_allocated_bytes += sl.d_in.allocated_bytes() + sl.d_gray.allocated_bytes() +
                                        sl.d_a.allocated_bytes() + sl.d_b.allocated_bytes() + tables;
            w.pinned_host_bytes += sl.pin_in.size() + sl.pin_out.size();
        }
        w.transfer_mode = slots_[0].pin_in.size() ? "pinned_staging" : "pageable_direct";
        return w;
    }

    bool supports_device_pipeline_timing() const noexcept override { return true; }

    void device_resident_upload(const InputImageView& in) override {
        set_device();
        Slot& sl = slots_[0];
        guard(sl, [&] {
            upload(sl, in);
            check(cudaStreamSynchronize(sl.stream.get()), "upload_complete", shape_);
        });
    }

    double device_resident_run_timed_ms() override {
        set_device();
        Slot& sl = slots_[0];
        float ms = 0.0f;
        guard(sl, [&] {
            check(cudaStreamSynchronize(sl.stream.get()), "pre_timing_drain", shape_);
            check(cudaEventRecord(sl.start.get(), sl.stream.get()), "event_record", shape_);
            last_result_ = &enqueue_pipeline(sl);
            check(cudaEventRecord(sl.stop.get(), sl.stream.get()), "event_record", shape_);
            check(cudaEventSynchronize(sl.stop.get()), "pipeline_complete", shape_);
            check(cudaEventElapsedTime(&ms, sl.start.get(), sl.stop.get()), "event_elapsed", shape_);
        });
        return static_cast<double>(ms);
    }

    void device_resident_download(const FloatImageView& out) override {
        set_device();
        if (!last_result_) throw Error(ErrorCategory::InvalidArgument, "no device-resident result to download");
        Slot& sl = slots_[0];
        guard(sl, [&] { download_and_wait(sl, *last_result_, out); });
    }

private:
    void set_device() { check(cudaSetDevice(config_.device_index), "select_device", shape_); }

    void drain_all() noexcept {
        for (auto& sl : slots_)
            if (sl.stream.get()) (void)cudaStreamSynchronize(sl.stream.get());
    }

    void release_slot(Slot& sl) noexcept {
        prepared_ = false;
        last_result_ = nullptr;
        sl.reset();
    }

    void release_extra_slots() noexcept {
        for (std::size_t i = 1; i < slots_.size(); ++i) slots_[i].reset();
        slots_.resize(1);
        next_slot_ = 0;
    }

    // Ensures `depth_` slots exist and are allocated for the current shape.
    // Returns the achievable depth (may be 1 if extra slots do not fit budget).
    int ensure_slots() {
        const int want = depth_ > 0 ? depth_ : kDefaultPipelineDepth;
        for (int i = 1; i < want; ++i) {
            slots_.emplace_back();
            Slot& sl = slots_.back();
            sl.stream = Stream("stream_create");
            sl.start = Event("event_create");
            sl.stop = Event("event_create");
            try {
                allocate_slot(sl);
            } catch (const Error&) {
                // Could not fit another slot: drop it and run at the depth we
                // managed to allocate (at least 1).
                sl.reset();
                slots_.pop_back();
                break;
            }
        }
        next_slot_ = 0;
        return static_cast<int>(slots_.size());
    }

    // Allocates the device workspace, uploads immutable tables, and (budget
    // permitting) pinned staging into `sl` for the current input_/config_.
    void allocate_slot(Slot& sl) {
        const int wo = config_.resize.width, ho = config_.resize.height;
        const int ch = channel_count(input_.format);
        const std::size_t in_row = static_cast<std::size_t>(input_.width) * static_cast<std::size_t>(ch);

        const std::size_t est_in = round_up(in_row, kPitchEstimate) * static_cast<std::size_t>(input_.height);
        const std::size_t est_gray =
            round_up(static_cast<std::size_t>(input_.width) * 4, kPitchEstimate) * static_cast<std::size_t>(input_.height);
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
        if (est_total + (64u << 20) > free_b) {
            throw Error(ErrorCategory::AllocationFailure,
                        "estimated device workspace " + std::to_string(est_total) +
                            " bytes does not fit in currently free device memory " + std::to_string(free_b),
                        {"prepare", "cuda", shape_, ""});
        }

        DeviceImage d_in(input_.width * ch, input_.height, 1, "alloc_input");
        DeviceImage d_gray(input_.width, input_.height, sizeof(float), "alloc_gray");
        DeviceImage d_a(wo, ho, sizeof(float), "alloc_workspace_a");
        DeviceImage d_b(wo, ho, sizeof(float), "alloc_workspace_b");

        const ResizeTables t = build_resize_tables(input_.width, input_.height, wo, ho);
        const std::vector<float> wts = build_gaussian_weights(config_.gaussian.radius, config_.gaussian.sigma);
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
        auto up = [&](const void* dst, const void* src, std::size_t n) {
            check(cudaMemcpyAsync(const_cast<void*>(dst), src, n, cudaMemcpyHostToDevice, sl.stream.get()),
                  "upload_tables", shape_);
        };
        try {
            up(dt.x0, t.x.i0.data(), xw);
            up(dt.x1, t.x.i1.data(), xw);
            up(dt.wx, t.x.w.data(), xw);
            up(dt.y0, t.y.i0.data(), yh);
            up(dt.y1, t.y.i1.data(), yh);
            up(dt.wy, t.y.w.data(), yh);
            up(dw, wts.data(), wts.size() * 4);
            check(cudaStreamSynchronize(sl.stream.get()), "upload_tables_complete", shape_);
        } catch (...) {
            (void)cudaStreamSynchronize(sl.stream.get());
            (void)cudaGetLastError();
            throw;
        }

        const std::size_t pin_in = in_row * static_cast<std::size_t>(input_.height);
        const std::size_t pin_out = static_cast<std::size_t>(wo) * 4 * static_cast<std::size_t>(ho);
        PinnedBuffer p_in, p_out;
        if (pin_in + pin_out <= config_.resources.max_pinned_host_bytes) {
            p_in = PinnedBuffer(pin_in, "alloc_pinned_input");
            p_out = PinnedBuffer(pin_out, "alloc_pinned_output");
        }

        sl.d_in = std::move(d_in);
        sl.d_gray = std::move(d_gray);
        sl.d_a = std::move(d_a);
        sl.d_b = std::move(d_b);
        sl.d_tables = std::move(d_tables);
        sl.tables = dt;
        sl.d_weights = dw;
        sl.pin_in = std::move(p_in);
        sl.pin_out = std::move(p_out);
    }

    template <class F>
    void guard(Slot& sl, F&& f) {
        try {
            f();
        } catch (...) {
            (void)cudaStreamSynchronize(sl.stream.get());
            (void)cudaGetLastError();
            last_result_ = nullptr;
            throw;
        }
    }

    void launch(Slot& sl, cudaError_t launch_err, const char* stage) {
        check(launch_err, stage, shape_);
#if FRAMEPIPE_CUDA_DEBUG_SYNC
        check(cudaStreamSynchronize(sl.stream.get()), stage, shape_);
#else
        (void)sl;
#endif
    }

    DevU8 dev_u8(const DeviceImage& d) const {
        return {static_cast<const std::uint8_t*>(d.ptr()), input_.width, input_.height, d.pitch(),
                channel_count(input_.format)};
    }
    static DevF32 dev_f32(const DeviceImage& d) {
        return {static_cast<float*>(d.ptr()), d.width(), d.height(), d.pitch()};
    }

    void upload(Slot& sl, const InputImageView& in) {
        const std::size_t row = static_cast<std::size_t>(in.width) * static_cast<std::size_t>(channel_count(in.format));
        if (sl.pin_in.size()) {
            for (int y = 0; y < in.height; ++y) {
                std::memcpy(sl.pin_in.data() + static_cast<std::size_t>(y) * row, row_ptr(in, y), row);
            }
            check(cudaMemcpy2DAsync(sl.d_in.ptr(), sl.d_in.pitch(), sl.pin_in.data(), row, row,
                                    static_cast<std::size_t>(in.height), cudaMemcpyHostToDevice, sl.stream.get()),
                  "upload_input", shape_);
        } else {
            check(cudaMemcpy2DAsync(sl.d_in.ptr(), sl.d_in.pitch(), in.data, in.stride_bytes, row,
                                    static_cast<std::size_t>(in.height), cudaMemcpyHostToDevice, sl.stream.get()),
                  "upload_input", shape_);
        }
    }

    const DeviceImage& enqueue_pipeline(Slot& sl) {
        const int r = config_.gaussian.radius;
        launch(sl, launch_grayscale(dev_u8(sl.d_in), dev_f32(sl.d_gray), sl.stream.get()), "grayscale");
        launch(sl, launch_resize(as_const(dev_f32(sl.d_gray)), dev_f32(sl.d_a), sl.tables, sl.stream.get()), "resize");
        if (config_.output_stage == OutputStage::Resize) return sl.d_a;
        launch(sl, launch_gaussian_horizontal(as_const(dev_f32(sl.d_a)), dev_f32(sl.d_b), sl.d_weights, r, sl.stream.get()),
               "gaussian_horizontal");
        launch(sl, launch_gaussian_vertical(as_const(dev_f32(sl.d_b)), dev_f32(sl.d_a), sl.d_weights, r, sl.stream.get()),
               "gaussian_vertical");
        if (config_.output_stage == OutputStage::Blur) return sl.d_a;
        launch(sl, launch_sobel_edges(as_const(dev_f32(sl.d_a)), dev_f32(sl.d_b), sl.stream.get()), "sobel_edges");
        return sl.d_b;
    }

    void download_and_wait(Slot& sl, const DeviceImage& src, const FloatImageView& out) {
        const std::size_t row = static_cast<std::size_t>(out.width) * sizeof(float);
        if (sl.pin_out.size()) {
            check(cudaMemcpy2DAsync(sl.pin_out.data(), row, src.ptr(), src.pitch(), row,
                                    static_cast<std::size_t>(out.height), cudaMemcpyDeviceToHost, sl.stream.get()),
                  "download_output", shape_);
            check(cudaStreamSynchronize(sl.stream.get()), "pipeline_complete", shape_);
            for (int y = 0; y < out.height; ++y) {
                std::memcpy(row_ptr(out, y), sl.pin_out.data() + static_cast<std::size_t>(y) * row, row);
            }
        } else {
            check(cudaMemcpy2DAsync(out.data, out.stride_bytes, src.ptr(), src.pitch(), row,
                                    static_cast<std::size_t>(out.height), cudaMemcpyDeviceToHost, sl.stream.get()),
                  "download_output", shape_);
            check(cudaStreamSynchronize(sl.stream.get()), "pipeline_complete", shape_);
        }
    }

    void fetch(Slot& sl, StageCapture& cap, Stage st, const DeviceImage& d) {
        auto& h = cap.emplace(st, d.width(), d.height());
        check(cudaMemcpy2DAsync(h.raw(), h.stride_bytes(), d.ptr(), d.pitch(), d.row_bytes(),
                                static_cast<std::size_t>(d.height()), cudaMemcpyDeviceToHost, sl.stream.get()),
              "capture_download", shape_);
        check(cudaStreamSynchronize(sl.stream.get()), "capture_complete", shape_);
    }

    PipelineConfig config_;
    InputSpec input_{};
    std::string shape_;
    bool prepared_ = false;
    int depth_ = kDefaultPipelineDepth;
    int next_slot_ = 0;
    std::vector<Slot> slots_;
    std::vector<float> host_fallback_;  // pageable download staging for process_batch
    const DeviceImage* last_result_ = nullptr;
};

}  // namespace

std::unique_ptr<BackendImpl> make_cuda_backend(const PipelineConfig& config) {
    return std::make_unique<CudaBackend>(config);
}

}  // namespace framepipe
