#include "11_streaming_conv1d.hpp"
#include "mini_test.hpp"
#include <cmath>
#include <random>

TEST(stream_equals_offline) {
    const int ci = 3, co = 2, k = 3, d = 4, T = 60;
    std::mt19937 rng(0); std::normal_distribution<float> g;
    std::vector<float> w(co * ci * k), b(co), x(ci * T);
    for (auto& v : w) v = g(rng); for (auto& v : b) v = g(rng); for (auto& v : x) v = g(rng);
    std::vector<float> ref(co * T, 0);
    for (int o = 0; o < co; ++o) for (int n = 0; n < T; ++n) { float a = b[o];
        for (int i = 0; i < ci; ++i) for (int j = 0; j < k; ++j) { int idx = n - (k - 1 - j) * d; if (idx >= 0) a += w[(o * ci + i) * k + j] * x[i * T + idx]; }
        ref[o * T + n] = a; }
    StreamingConv1d c(ci, co, k, d, w, b);
    std::vector<float> y(co * T); int pos = 0;
    for (int t : {1, 9, 20, 5, 25}) {
        std::vector<float> chunk(ci * t), out(co * t);
        for (int i = 0; i < ci; ++i) for (int n = 0; n < t; ++n) chunk[i * t + n] = x[i * T + pos + n];
        c.process(chunk.data(), t, out.data());
        for (int o = 0; o < co; ++o) for (int n = 0; n < t; ++n) y[o * T + pos + n] = out[o * t + n];
        pos += t;
    }
    double err = 0; for (int i = 0; i < co * T; ++i) err = std::max(err, (double)std::fabs(y[i] - ref[i]));
    CHECK(pos == T); CHECK(err < 1e-4);
}
