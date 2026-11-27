#include "01_welford.hpp"
#include "mini_test.hpp"
#include <random>
#include <vector>

TEST(stable_against_two_pass) {
    std::mt19937_64 rng(0); std::normal_distribution<double> g(0, 1);
    std::vector<double> xs; RunningStats s;
    for (int i = 0; i < 100000; ++i) { xs.push_back(1e9 + g(rng)); s.push(xs.back()); }
    double m = 0; for (double x : xs) m += x - 1e9; m = 1e9 + m / xs.size();
    double v = 0; for (double x : xs) v += (x - m) * (x - m); v /= (xs.size() - 1);
    CHECK(s.count() == xs.size());
    CHECK_NEAR(s.mean(), m, 1e-5);
    CHECK_NEAR(s.variance(), v, 1e-6 * v);
}
TEST(small_n) { RunningStats s; s.push(3.0); CHECK_NEAR(s.variance(), 0.0, 0); s.push(5.0); CHECK_NEAR(s.variance(), 2.0, 1e-12); }
