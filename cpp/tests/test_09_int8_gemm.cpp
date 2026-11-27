#include "09_int8_gemm.hpp"
#include "mini_test.hpp"
#include <random>
#include <vector>

TEST(multiplier) {
    for (double M : {0.3, 0.000123, 0.9999}) { int32_t m0; int sh; quantize_multiplier(M, m0, sh);
        CHECK(m0 >= (1 << 30)); CHECK_NEAR(m0 / 2147483648.0 * std::ldexp(1.0, -sh), M, M * 1e-8); }
    int32_t m0; int sh; quantize_multiplier(0.25, m0, sh);
    CHECK(multiply_by_quantized_multiplier(1000, m0, sh) == 250);
    CHECK(multiply_by_quantized_multiplier(-1002, m0, sh) == -251 || multiply_by_quantized_multiplier(-1002, m0, sh) == -250);
}
TEST(linear_within_one_lsb) {
    const int N = 8, K = 64, C = 6; std::mt19937 rng(0); std::uniform_real_distribution<double> ux(-0.5, 2.0); std::normal_distribution<double> gw(0, 0.1);
    std::vector<double> xf(N * K), wf(C * K), bf(C);
    for (auto& v : xf) v = ux(rng); for (auto& v : wf) v = gw(rng);
    for (auto& v : bf) v = gw(rng);
    double sx = 2.5 / 255; int zpx = (int)std::lround(-128 + 0.5 / sx);
    std::vector<int8_t> xq(N * K), wq(C * K); std::vector<double> sw(C);
    for (int i = 0; i < N * K; ++i) xq[i] = (int8_t)std::clamp(std::lround(xf[i] / sx) + zpx, -128L, 127L);
    for (int c = 0; c < C; ++c) { double m = 0; for (int k = 0; k < K; ++k) m = std::max(m, std::fabs(wf[c * K + k])); sw[c] = m / 127;
        for (int k = 0; k < K; ++k) wq[c * K + k] = (int8_t)std::lround(wf[c * K + k] / sw[c]); }
    double sy = 4.0 / 255; int zpy = 0;
    std::vector<int32_t> b(C), M0(C); std::vector<int> sh(C);
    for (int c = 0; c < C; ++c) { b[c] = (int32_t)std::lround(bf[c] / (sx * sw[c])); quantize_multiplier(sx * sw[c] / sy, M0[c], sh[c]); }
    std::vector<int8_t> y(N * C);
    int8_linear(xq.data(), N, K, zpx, wq.data(), C, b.data(), M0.data(), sh.data(), zpy, y.data());
    int maxd = 0;
    for (int n = 0; n < N; ++n) for (int c = 0; c < C; ++c) {
        double acc = b[c] * sx * sw[c]; for (int k = 0; k < K; ++k) acc += sx * (xq[n * K + k] - zpx) * sw[c] * wq[c * K + k];
        long ref = std::clamp(std::lround(acc / sy) + zpy, -128L, 127L); maxd = std::max(maxd, (int)std::labs(ref - y[n * C + c])); }
    CHECK(maxd <= 1);
}
