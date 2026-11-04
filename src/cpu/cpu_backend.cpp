// CPU backends: scalar reference ("cpu-ref") and OpenMP row-parallel
// ("cpu-omp"). Both run the same operator code from cpu_ops.cpp; cpu-omp only
// partitions output rows, with an implicit barrier between dependent stages.

#include <cstring>

#include "core/backend.hpp"
#include "core/tables.hpp"
#include "cpu/cpu_ops.hpp"
#if FRAMEPIPE_HAVE_OPENMP
#include "cpu/cpu_parallel.hpp"
#endif

namespace framepipe {

namespace {

class CpuBackend final : public BackendImpl {
public:
    CpuBackend(const PipelineConfig& c, bool parallel) : config_(c), parallel_(parallel) {}

    Backend kind() const noexcept override { return parallel_ ? Backend::CpuParallel : Backend::CpuReference; }

    std::string implementation() const override {
#if FRAMEPIPE_HAVE_OPENMP
        if (parallel_) {
            return std::string("openmp-rows(threads=") + std::to_string(config_.cpu.threads) +
                   ",schedule=" + cpu::schedule_name() + ")";
        }
#endif
        return "scalar";
    }

    void prepare(const InputSpec& in) override {
        // Build into locals; commit only on success so a failure halfway
        // through allocation leaves no partially prepared state.
        const int wo = config_.resize.width, ho = config_.resize.height;
        ResizeTables tables = build_resize_tables(in.width, in.height, wo, ho);
        std::vector<float> weights = build_gaussian_weights(config_.gaussian.radius, config_.gaussian.sigma);
        HostFloatImage gray(in.width, in.height);
        HostFloatImage a, b;
        if (config_.output_stage != OutputStage::Resize) {
            a = HostFloatImage(wo, ho);
            b = HostFloatImage(wo, ho);
        }
        tables_ = std::move(tables);
        weights_ = std::move(weights);
        gray_ = std::move(gray);
        a_ = std::move(a);
        b_ = std::move(b);
        input_ = in;
    }

    void process_into(const InputImageView& in, const FloatImageView& out) override {
        const int r = config_.gaussian.radius;
        rows(in.height, [&](int y0, int y1) { cpu::grayscale(in, gray_.view(), y0, y1); });
        switch (config_.output_stage) {
            case OutputStage::Resize:
                rows(out.height, [&](int y0, int y1) { cpu::resize_bilinear(gray_.cview(), out, tables_, y0, y1); });
                return;
            case OutputStage::Blur:
                rows(out.height, [&](int y0, int y1) { cpu::resize_bilinear(gray_.cview(), a_.view(), tables_, y0, y1); });
                rows(out.height, [&](int y0, int y1) { cpu::gaussian_horizontal(a_.cview(), b_.view(), weights_, r, y0, y1); });
                rows(out.height, [&](int y0, int y1) { cpu::gaussian_vertical(b_.cview(), out, weights_, r, y0, y1); });
                return;
            case OutputStage::Edges:
                // Two-buffer workspace: resize->A, hblur A->B, vblur B->A, sobel A->out.
                rows(out.height, [&](int y0, int y1) { cpu::resize_bilinear(gray_.cview(), a_.view(), tables_, y0, y1); });
                rows(out.height, [&](int y0, int y1) { cpu::gaussian_horizontal(a_.cview(), b_.view(), weights_, r, y0, y1); });
                rows(out.height, [&](int y0, int y1) { cpu::gaussian_vertical(b_.cview(), a_.view(), weights_, r, y0, y1); });
                rows(out.height, [&](int y0, int y1) { cpu::sobel_edges(a_.cview(), out, y0, y1); });
                return;
        }
    }

    StageCapture capture(const InputImageView& in) override {
        StageCapture cap;
        const int wo = config_.resize.width, ho = config_.resize.height, r = config_.gaussian.radius;
        auto& g = cap.emplace(Stage::Gray, in.width, in.height);
        cpu::grayscale(in, g.view(), 0, in.height);
        auto& rs = cap.emplace(Stage::Resized, wo, ho);
        cpu::resize_bilinear(g.cview(), rs.view(), tables_, 0, ho);
        if (config_.output_stage == OutputStage::Resize) return cap;
        auto& h = cap.emplace(Stage::BlurHorizontal, wo, ho);
        cpu::gaussian_horizontal(rs.cview(), h.view(), weights_, r, 0, ho);
        auto& v = cap.emplace(Stage::BlurVertical, wo, ho);
        cpu::gaussian_vertical(h.cview(), v.view(), weights_, r, 0, ho);
        if (config_.output_stage == OutputStage::Blur) return cap;
        auto& sx = cap.emplace(Stage::SobelX, wo, ho);
        auto& sy = cap.emplace(Stage::SobelY, wo, ho);
        cpu::sobel_components(v.cview(), sx.view(), sy.view(), 0, ho);
        auto& e = cap.emplace(Stage::Edges, wo, ho);
        cpu::sobel_edges(v.cview(), e.view(), 0, ho);
        return cap;
    }

    WorkspaceInfo workspace() const override {
        WorkspaceInfo w;
        auto logical = [](const HostFloatImage& im) {
            return im.empty() ? std::size_t{0}
                              : static_cast<std::size_t>(im.width()) * static_cast<std::size_t>(im.height()) * sizeof(float);
        };
        const std::size_t tables = (tables_.x.i0.size() + tables_.y.i0.size()) * 12 + weights_.size() * sizeof(float);
        w.host_logical_bytes = logical(gray_) + logical(a_) + logical(b_) + tables;
        w.host_allocated_bytes = gray_.allocated_bytes() + a_.allocated_bytes() + b_.allocated_bytes() + tables;
        return w;
    }

private:
    template <class F>
    void rows(int n, F&& f) {
#if FRAMEPIPE_HAVE_OPENMP
        if (parallel_) {
            cpu::parallel_rows(config_.cpu.threads, n, f);
            return;
        }
#endif
        f(0, n);
    }

    PipelineConfig config_;
    bool parallel_;
    InputSpec input_{};
    ResizeTables tables_;
    std::vector<float> weights_;
    HostFloatImage gray_, a_, b_;
};

}  // namespace

std::unique_ptr<BackendImpl> make_cpu_backend(const PipelineConfig& config, bool parallel) {
    return std::make_unique<CpuBackend>(config, parallel);
}

}  // namespace framepipe
