#include "core/tables.hpp"

#include <cmath>

#include "framepipe/config.hpp"
#include "framepipe/errors.hpp"

namespace framepipe {

AxisTable build_axis_table(int input_extent, int output_extent) {
    if (input_extent <= 0 || output_extent <= 0) {
        throw Error(ErrorCategory::Internal, "build_axis_table requires positive extents");
    }
    AxisTable t;
    const auto n = static_cast<std::size_t>(output_extent);
    t.i0.resize(n);
    t.i1.resize(n);
    t.w.resize(n);
    for (int d = 0; d < output_extent; ++d) {
        // u = (x + 0.5) * Wi / Wo - 0.5, evaluated left to right as specified.
        const double u = (static_cast<double>(d) + 0.5) * static_cast<double>(input_extent) /
                             static_cast<double>(output_extent) - 0.5;
        const double raw = std::floor(u);  // floor, not truncation
        const auto raw_i = static_cast<long long>(raw);
        t.w[static_cast<std::size_t>(d)] = static_cast<float>(u - raw);  // weight from the UNCLAMPED raw index
        t.i0[static_cast<std::size_t>(d)] = clamp_index(raw_i, input_extent);
        t.i1[static_cast<std::size_t>(d)] = clamp_index(raw_i + 1, input_extent);
    }
    return t;
}

ResizeTables build_resize_tables(int in_w, int in_h, int out_w, int out_h) {
    return {build_axis_table(in_w, out_w), build_axis_table(in_h, out_h)};
}

std::vector<float> build_gaussian_weights(int radius, double sigma) {
    if (radius < 0 || radius > kMaxGaussianRadius || !std::isfinite(sigma) || sigma < kMinSigma || sigma > kMaxSigma) {
        throw Error(ErrorCategory::InvalidConfiguration, "gaussian parameters out of range");
    }
    const auto n = static_cast<std::size_t>(2 * radius + 1);
    std::vector<double> un(n);
    double sum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        const double v = std::exp(-(static_cast<double>(k) * static_cast<double>(k)) / (2.0 * sigma * sigma));
        un[static_cast<std::size_t>(k + radius)] = v;
    }
    for (std::size_t i = 0; i < n; ++i) sum += un[i];  // sum over -r..r in increasing order
    std::vector<float> w(n);
    for (std::size_t i = 0; i < n; ++i) w[i] = static_cast<float>(un[i] / sum);
    return w;
}

}  // namespace framepipe
