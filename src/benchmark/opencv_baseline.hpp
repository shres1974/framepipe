#pragma once
// Separately labeled OpenCV imgproc CPU baseline (spec 18.4). NOT a FramePipe
// backend: it is never selectable for `process`, and its timings are only
// treated as a matched comparison when its output passes the FramePipe
// tolerance against cpu-ref for the same input (checked by the harness).
//
// Explicit settings: CV_32F throughout; grayscale uses the FramePipe integer
// convention (shared input adaptation, included in timing); cv::resize with
// INTER_LINEAR; cv::sepFilter2D with FramePipe's FP32 Gaussian weights and
// BORDER_REPLICATE; cv::Sobel(ksize=3, CV_32F, BORDER_REPLICATE); magnitude
// divided by sqrt(32) and clamped to [0, 1].

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "framepipe/config.hpp"
#include "framepipe/image_view.hpp"

namespace framepipe::bench {

class OpenCvBaseline {
public:
    explicit OpenCvBaseline(const PipelineConfig& config, int threads);
    void prepare(const InputSpec& spec);
    void process_into(const InputImageView& in, const FloatImageView& out);
    std::string implementation() const;
    int threads() const noexcept { return threads_; }

private:
    PipelineConfig config_;
    int threads_;
    cv::Mat gray_, resized_, blurred_, gx_, gy_, mag_;
    cv::Mat kernel_;
};

}  // namespace framepipe::bench
