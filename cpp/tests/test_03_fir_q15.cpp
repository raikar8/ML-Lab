#include <algorithm>
#include "03_fir_q15.hpp"
#include "mini_test.hpp"
#include <cmath>
#include <random>

TEST(matches_float_and_blocks) {
    const int N = 1000, L = 31;
    std::vector<double> hf(L); double s = 0;
    for (int k = 0; k < L; ++k) { hf[k] = 0.5 - 0.5 * std::cos(2 * M_PI * (k + 1) / (L + 1)); s += hf[k]; }
    std::vector<int16_t> h(L); for (int k = 0; k < L; ++k) { hf[k] /= s * 1.05; h[k] = (int16_t)std::lround(hf[k] * 32768); }
    std::mt19937 rng(0); std::uniform_real_distribution<double> u(-0.5, 0.5);
    std::vector<int16_t> x(N); std::vector<double> xf(N);
    for (int i = 0; i < N; ++i) { x[i] = (int16_t)std::lround(u(rng) * 32768); xf[i] = x[i] / 32768.0; }
    std::vector<int16_t> y1(N), y2(N);
    FirQ15 f(h); f.process(x.data(), y1.data(), N);
    FirQ15 g(h); int pos = 0; for (int b : {1, 7, 64, 3, 500, 425}) { g.process(x.data() + pos, y2.data() + pos, b); pos += b; }
    CHECK(pos == N); CHECK(y1 == y2);
    double maxerr = 0;
    for (int n = 0; n < N; ++n) { double r = 0; for (int k = 0; k < L && k <= n; ++k) r += hf[k] * xf[n - k]; maxerr = std::max(maxerr, std::fabs(r - y1[n] / 32768.0)); }
    CHECK(maxerr < 5e-4);
    f.reset(); std::vector<int16_t> y3(N); f.process(x.data(), y3.data(), N); CHECK(y3 == y1);
}
