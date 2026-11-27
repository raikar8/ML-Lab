#include "15_tiled_matmul.hpp"
#include "mini_test.hpp"
#include <chrono>
#include <cmath>
#include <random>

static double err(const std::vector<float>& a, const std::vector<float>& b) { double e = 0; for (std::size_t i = 0; i < a.size(); ++i) e = std::max(e, (double)std::fabs(a[i] - b[i])); return e; }
TEST(correct_odd_sizes) {
    std::mt19937 rng(0); std::normal_distribution<float> g; const int M = 37, K = 53, N = 29;
    std::vector<float> A(M * K), B(K * N), C0(M * N), C1(M * N), C2(M * N);
    for (auto& v : A) v = g(rng); for (auto& v : B) v = g(rng);
    matmul_naive(A.data(), B.data(), C0.data(), M, K, N); matmul_ikj(A.data(), B.data(), C1.data(), M, K, N);
    matmul_tiled(A.data(), B.data(), C2.data(), M, K, N, 16);
    CHECK(err(C0, C1) < 1e-3); CHECK(err(C0, C2) < 1e-3);
}
TEST(timing_report) {
    const int n = 512; std::vector<float> A(n * n, 1.f), B(n * n, 0.5f), C(n * n);
    auto t = [&](auto f) { auto s = std::chrono::steady_clock::now(); f(); return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count(); };
    double tn = t([&] { matmul_naive(A.data(), B.data(), C.data(), n, n, n); });
    double ti = t([&] { matmul_ikj(A.data(), B.data(), C.data(), n, n, n); });
    double tt = t([&] { matmul_tiled(A.data(), B.data(), C.data(), n, n, n, 64); });
    std::printf("  512^3: naive %.1f ms, ikj %.1f ms, tiled %.1f ms\n", tn, ti, tt);
    CHECK_NEAR(C[123], 256.0, 1e-3);
}
