#include "validation/compare.hpp"

#include <algorithm>
#include <cmath>

#include <nlohmann/json.hpp>

#include "framepipe/errors.hpp"

namespace framepipe {

Tolerance tolerance_for(Stage s) noexcept {
    switch (s) {
        case Stage::Gray: return {1e-7, 1e-6};
        case Stage::Resized: return {2e-6, 2e-6};
        case Stage::BlurHorizontal:
        case Stage::BlurVertical: return {1e-5, 1e-5};
        case Stage::SobelX:
        case Stage::SobelY: return {2e-5, 1e-5};
        case Stage::Edges: return {1e-5, 1e-5};
    }
    return {0, 0};
}

CompareResult compare_float_images(const ConstFloatImageView& ref, const ConstFloatImageView& cand, Tolerance tol,
                                   bool range01) {
    CompareResult r;
    r.tolerance = tol;
    r.dimensions_match = ref.width == cand.width && ref.height == cand.height;
    if (!r.dimensions_match) {
        r.message = "dimension mismatch: reference " + shape_string(ref.width, ref.height) + ", candidate " +
                    shape_string(cand.width, cand.height);
        return r;
    }
    double sum_abs = 0.0, sum_sq = 0.0;
    long long out_of_range = 0;
    long long finite_pairs = 0;
    for (int y = 0; y < ref.height; ++y) {
        const float* pr = row_ptr(ref, y);
        const float* pc = row_ptr(cand, y);
        for (int x = 0; x < ref.width; ++x) {
            const double rv = pr[x], cv = pc[x];
            ++r.pixel_count;
            const bool rf = std::isfinite(rv), cf = std::isfinite(cv);
            if (!rf) ++r.nonfinite_reference;
            if (!cf) ++r.nonfinite_candidate;
            if (!rf || !cf) {
                ++r.failing_pixels;
                continue;
            }
            ++finite_pairs;
            if (range01 && (cv < 0.0 || cv > 1.0)) ++out_of_range;
            const double e = std::fabs(cv - rv);
            sum_abs += e;
            sum_sq += e * e;
            if (e > tol.atol + tol.rtol * std::fabs(rv)) ++r.failing_pixels;
            if (e > r.max_abs_error || r.max_error_x < 0) {
                r.max_abs_error = e;
                r.max_error_x = x;
                r.max_error_y = y;
                r.reference_at_max = rv;
                r.candidate_at_max = cv;
            }
        }
    }
    const double n = static_cast<double>(finite_pairs > 0 ? finite_pairs : 1);
    r.mean_abs_error = sum_abs / n;
    const double mse = sum_sq / n;
    r.rmse = std::sqrt(mse);
    r.exact_match = (mse == 0.0) && r.nonfinite_candidate == 0 && r.nonfinite_reference == 0;
    if (mse > 0.0) r.psnr_db = 10.0 * std::log10(1.0 / mse);  // peak = 1.0 for normalized images
    r.failing_pixels += out_of_range;
    r.passed = r.failing_pixels == 0;
    if (r.nonfinite_candidate || r.nonfinite_reference) {
        r.message = "non-finite values present (reference=" + std::to_string(r.nonfinite_reference) +
                    ", candidate=" + std::to_string(r.nonfinite_candidate) + ")";
    } else if (out_of_range) {
        r.message = std::to_string(out_of_range) + " candidate values outside [0, 1]";
    } else if (!r.passed) {
        r.message = std::to_string(r.failing_pixels) + " pixels exceed tolerance";
    } else {
        r.message = "all pixels within tolerance";
    }
    return r;
}

std::string compare_result_json(const CompareResult& r, int indent) {
    nlohmann::ordered_json j;
    j["status"] = r.passed ? "passed" : "failed";
    j["dimensions_match"] = r.dimensions_match;
    j["tolerance"] = {{"atol", r.tolerance.atol}, {"rtol", r.tolerance.rtol}};
    j["pixel_count"] = r.pixel_count;
    j["failing_pixels"] = r.failing_pixels;
    j["nonfinite_reference"] = r.nonfinite_reference;
    j["nonfinite_candidate"] = r.nonfinite_candidate;
    j["max_abs_error"] = r.max_abs_error;
    j["mean_abs_error"] = r.mean_abs_error;
    j["rmse"] = r.rmse;
    if (r.max_error_x >= 0) {
        j["max_error_at"] = {{"x", r.max_error_x},
                             {"y", r.max_error_y},
                             {"reference", r.reference_at_max},
                             {"candidate", r.candidate_at_max}};
    } else {
        j["max_error_at"] = nullptr;
    }
    j["exact_match"] = r.exact_match;
    j["psnr_db"] = r.psnr_db ? nlohmann::ordered_json(*r.psnr_db) : nlohmann::ordered_json(nullptr);
    j["message"] = r.message;
    return j.dump(indent);
}

}  // namespace framepipe

namespace framepipe {

HostU8Image error_map(const ConstFloatImageView& ref, const ConstFloatImageView& cand, Tolerance tol) {
    if (ref.width != cand.width || ref.height != cand.height) {
        throw Error(ErrorCategory::InvalidArgument, "error_map: dimension mismatch " + shape_string(ref.width, ref.height) +
                                                         " vs " + shape_string(cand.width, cand.height));
    }
    HostU8Image out(ref.width, ref.height, PixelFormat::Gray8);
    for (int y = 0; y < ref.height; ++y) {
        const float* pr = row_ptr(ref, y);
        const float* pc = row_ptr(cand, y);
        std::uint8_t* dst = out.row(y);
        for (int x = 0; x < ref.width; ++x) {
            const double rv = pr[x], cv = pc[x];
            if (!std::isfinite(rv) || !std::isfinite(cv)) {
                dst[x] = 255;
                continue;
            }
            const double limit = tol.atol + tol.rtol * std::fabs(rv);
            const double e = std::fabs(cv - rv);
            const double frac = limit > 0.0 ? e / limit : (e > 0.0 ? 1.0 : 0.0);
            dst[x] = static_cast<std::uint8_t>(std::lround(std::min(frac, 1.0) * 255.0));
        }
    }
    return out;
}

}  // namespace framepipe
