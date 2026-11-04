#include "benchmark/stats.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace framepipe::bench {

double median_of(std::vector<double> v) {
    if (v.empty()) throw std::invalid_argument("median of empty set");
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

double p95_nearest_rank(std::vector<double> v) {
    if (v.empty()) throw std::invalid_argument("p95 of empty set");
    std::sort(v.begin(), v.end());
    // ceil(0.95 * N) computed exactly in integers: ceil(95N / 100).
    const std::size_t n = v.size();
    const std::size_t rank = (95 * n + 99) / 100;
    return v[rank - 1];
}

Stats compute_stats(std::vector<double> samples) {
    if (samples.empty()) throw std::invalid_argument("no samples");
    Stats s;
    s.count = samples.size();
    s.min = *std::min_element(samples.begin(), samples.end());
    s.max = *std::max_element(samples.begin(), samples.end());
    double sum = 0;
    for (double x : samples) sum += x;
    s.mean = sum / static_cast<double>(s.count);
    if (s.count >= 2) {
        double ss = 0;
        for (double x : samples) ss += (x - s.mean) * (x - s.mean);
        s.stddev = std::sqrt(ss / static_cast<double>(s.count - 1));
    }
    s.median = median_of(samples);
    s.p95 = p95_nearest_rank(std::move(samples));
    return s;
}

nlohmann::ordered_json stats_json(const Stats& s) {
    nlohmann::ordered_json j;
    j["count"] = s.count;
    j["min_ms"] = s.min;
    j["max_ms"] = s.max;
    j["mean_ms"] = s.mean;
    j["stddev_ms"] = s.stddev ? nlohmann::ordered_json(*s.stddev) : nlohmann::ordered_json(nullptr);
    j["median_ms"] = s.median;
    j["p95_ms"] = s.p95;
    return j;
}

}  // namespace framepipe::bench
