#pragma once
// Descriptive statistics over raw samples (spec 18.3).
//   median: middle value, or mean of the two middle values
//   p95:    nearest rank, sorted[ceil(0.95 * N) - 1]
//   stddev: sample standard deviation (N-1), absent for N < 2

#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

namespace framepipe::bench {

struct Stats {
    std::size_t count = 0;
    double min = 0, max = 0, mean = 0, median = 0, p95 = 0;
    std::optional<double> stddev;
};

// Throws std::invalid_argument for an empty sample set.
Stats compute_stats(std::vector<double> samples);
double median_of(std::vector<double> v);
double p95_nearest_rank(std::vector<double> v);

nlohmann::ordered_json stats_json(const Stats& s);

}  // namespace framepipe::bench
