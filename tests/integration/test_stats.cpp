#include "benchmark/stats.hpp"

#include <gtest/gtest.h>

using namespace framepipe::bench;

TEST(Stats, MedianOddEven) {
    EXPECT_EQ(median_of({3, 1, 2}), 2);
    EXPECT_EQ(median_of({4, 1, 3, 2}), 2.5);
    EXPECT_EQ(median_of({7}), 7);
}

TEST(Stats, P95NearestRank) {
    std::vector<double> v;
    for (int i = 1; i <= 100; ++i) v.push_back(i);
    EXPECT_EQ(p95_nearest_rank(v), 95);  // ceil(95)-1 = index 94 -> 95
    v.push_back(101);                    // N=101: ceil(95.95)=96 -> index 95 -> 96
    EXPECT_EQ(p95_nearest_rank(v), 96);
    EXPECT_EQ(p95_nearest_rank({5}), 5);
    EXPECT_EQ(p95_nearest_rank({1, 2}), 2);  // ceil(1.9)=2 -> index 1
    std::vector<double> twenty;
    for (int i = 1; i <= 20; ++i) twenty.push_back(i);
    EXPECT_EQ(p95_nearest_rank(twenty), 19);  // ceil(19)=19 -> index 18 -> 19
}

TEST(Stats, FullSummary) {
    const auto s = compute_stats({2, 4, 4, 4, 5, 5, 7, 9});
    EXPECT_EQ(s.count, 8u);
    EXPECT_EQ(s.min, 2);
    EXPECT_EQ(s.max, 9);
    EXPECT_EQ(s.mean, 5);
    EXPECT_EQ(s.median, 4.5);
    ASSERT_TRUE(s.stddev.has_value());
    EXPECT_NEAR(*s.stddev, 2.13808993529939, 1e-12);  // sample stddev (N-1)
}

TEST(Stats, SingleSampleHasNoStddev) {
    const auto s = compute_stats({1.5});
    EXPECT_FALSE(s.stddev.has_value());
    EXPECT_TRUE(stats_json(s)["stddev_ms"].is_null());
}

TEST(Stats, EmptyThrows) { EXPECT_THROW(compute_stats({}), std::invalid_argument); }
