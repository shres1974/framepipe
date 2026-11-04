#pragma once
// Float image comparison under the per-pixel rule
//   abs(c - r) <= atol + rtol * abs(r)
// (spec Section 16.2). Thresholds are fixed per stage and never relaxed
// automatically.

#include <optional>
#include <string>

#include "framepipe/image_view.hpp"
#include "framepipe/pipeline.hpp"

namespace framepipe {

struct Tolerance {
    double atol;
    double rtol;
};

// Normative tolerances for each stage.
Tolerance tolerance_for(Stage s) noexcept;

struct CompareResult {
    bool passed = false;
    bool dimensions_match = false;
    long long pixel_count = 0;
    long long failing_pixels = 0;
    long long nonfinite_reference = 0;
    long long nonfinite_candidate = 0;
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double rmse = 0.0;
    int max_error_x = -1;
    int max_error_y = -1;
    double reference_at_max = 0.0;
    double candidate_at_max = 0.0;
    std::optional<double> psnr_db;  // nullopt when MSE == 0 (exact match)
    bool exact_match = false;
    Tolerance tolerance{0, 0};
    std::string message;
};

// Requires `range01` to additionally check candidate values lie in [0, 1].
CompareResult compare_float_images(const ConstFloatImageView& reference, const ConstFloatImageView& candidate,
                                   Tolerance tol, bool range01 = false);

std::string compare_result_json(const CompareResult& r, int indent = -1);

}  // namespace framepipe
