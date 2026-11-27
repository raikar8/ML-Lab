#include "07_overlap_add.hpp"
#include "mini_test.hpp"
#include <cmath>
#include <random>

TEST(equals_direct) {
    std::mt19937 rng(1); std::normal_distribution<float> g;
    for (std::size_t L : {1u, 5u, 70u}) {
        std::vector<float> h(L), x(64 * 12); for (auto& v : h) v = g(rng); for (auto& v : x) v = g(rng);
        OverlapAdd ola(h, 64); std::vector<float> y(x.size());
        for (std::size_t b = 0; b < x.size(); b += 64) ola.process(&x[b], &y[b]);
        double err = 0;
        for (std::size_t n = 0; n < x.size(); ++n) { double r = 0; for (std::size_t k = 0; k < L && k <= n; ++k) r += h[k] * x[n - k]; err = std::max(err, std::fabs(r - y[n])); }
        CHECK(err < 1e-3);
    }
}
