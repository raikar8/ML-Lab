#include "12_conv2d_im2col.hpp"
#include "mini_test.hpp"
#include <cmath>
#include <random>

TEST(matches_direct) {
    std::mt19937 rng(0); std::normal_distribution<float> g;
    const int C = 3, H = 9, W = 8, Co = 4, k = 3;
    std::vector<float> x(C * H * W), w(Co * C * k * k), b(Co);
    for (auto& v : x) v = g(rng); for (auto& v : w) v = g(rng); for (auto& v : b) v = g(rng);
    for (int stride : {1, 2}) for (int pad : {0, 1}) {
        int Ho = out_size(H, k, stride, pad), Wo = out_size(W, k, stride, pad);
        std::vector<float> y1(Co * Ho * Wo), y2(Co * Ho * Wo);
        conv2d_im2col(x.data(), C, H, W, w.data(), b.data(), Co, k, stride, pad, y1.data());
        conv2d_direct(x.data(), C, H, W, w.data(), b.data(), Co, k, stride, pad, y2.data());
        double e = 0; for (std::size_t i = 0; i < y1.size(); ++i) e = std::max(e, (double)std::fabs(y1[i] - y2[i]));
        CHECK(e < 1e-4);
    }
    CHECK(out_size(224, 7, 2, 3) == 112);
}
