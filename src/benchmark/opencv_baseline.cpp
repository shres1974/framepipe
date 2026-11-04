#include "benchmark/opencv_baseline.hpp"

#include <opencv2/imgproc.hpp>

#include "core/tables.hpp"
#include "cpu/cpu_ops.hpp"

namespace framepipe::bench {

OpenCvBaseline::OpenCvBaseline(const PipelineConfig& config, int threads) : config_(config), threads_(threads) {
    cv::setNumThreads(threads_);
    const auto w = build_gaussian_weights(config.gaussian.radius, config.gaussian.sigma);
    kernel_ = cv::Mat(1, static_cast<int>(w.size()), CV_32F);
    for (std::size_t i = 0; i < w.size(); ++i) kernel_.at<float>(0, static_cast<int>(i)) = w[i];
}

std::string OpenCvBaseline::implementation() const {
    return "opencv-imgproc " + std::string(CV_VERSION) + " (threads=" + std::to_string(threads_) + ")";
}

void OpenCvBaseline::prepare(const InputSpec& spec) {
    gray_.create(spec.height, spec.width, CV_32F);
    const int wo = config_.resize.width, ho = config_.resize.height;
    resized_.create(ho, wo, CV_32F);
    blurred_.create(ho, wo, CV_32F);
    gx_.create(ho, wo, CV_32F);
    gy_.create(ho, wo, CV_32F);
    mag_.create(ho, wo, CV_32F);
}

void OpenCvBaseline::process_into(const InputImageView& in, const FloatImageView& out) {
    cv::setNumThreads(threads_);
    FloatImageView gv{gray_.ptr<float>(0), gray_.cols, gray_.rows, gray_.step[0], gray_.step[0] * gray_.rows};
    cpu::grayscale(in, gv, 0, in.height);
    cv::Mat dst(out.height, out.width, CV_32F, out.data, out.stride_bytes);
    if (config_.output_stage == OutputStage::Resize) {
        cv::resize(gray_, dst, dst.size(), 0, 0, cv::INTER_LINEAR);
        return;
    }
    cv::resize(gray_, resized_, resized_.size(), 0, 0, cv::INTER_LINEAR);
    if (config_.output_stage == OutputStage::Blur) {
        cv::sepFilter2D(resized_, dst, CV_32F, kernel_, kernel_, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        return;
    }
    cv::sepFilter2D(resized_, blurred_, CV_32F, kernel_, kernel_, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::Sobel(blurred_, gx_, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(blurred_, gy_, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::magnitude(gx_, gy_, mag_);
    mag_.convertTo(dst, CV_32F, 1.0 / static_cast<double>(kEdgeNormalization));
    cv::min(dst, 1.0, dst);
    cv::max(dst, 0.0, dst);
}

}  // namespace framepipe::bench
