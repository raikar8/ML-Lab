/*
title: int8 GEMM with int32 accumulation and fixed-point requantization (TFLite/CMSIS-NN style)
week: 22
level: hard
time: 45
tags: c++, quantization, embedded, gemm

* `void quantize_multiplier(double M, int32_t& M0, int& shift)` — for 0 < M < 1: `M = M0·2^-31·2^-shift`, `M0` in
  [2^30, 2^31) (use `std::frexp`).
* `int32_t multiply_by_quantized_multiplier(int32_t acc, int32_t M0, int shift)` — rounding high multiply
  `(int64(acc)·M0 + 2^30) >> 31`, then rounding arithmetic right shift by `shift` (round half away from zero is fine).
* `void int8_linear(const int8_t* x, int N, int K, int32_t zp_x, const int8_t* w, int C, const int32_t* bias,
  const int32_t* M0, const int* shift, int32_t zp_y, int8_t* y)` — `y[n][c] = clamp(zp_y + requant(Σ_k (x[n][k] − zp_x)·w[c][k]
  + bias[c]), −128, 127)` with symmetric per-channel weights (zp_w = 0). Weights are row-major `C × K`.
*/
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

inline void quantize_multiplier(double M, int32_t& M0, int& shift) {
    // BEGIN SOLUTION
    int e = 0;
    double m = std::frexp(M, &e);           // M = m * 2^e, m in [0.5, 1)
    int64_t q = std::llround(m * (1LL << 31));
    if (q == (1LL << 31)) { q /= 2; ++e; }
    M0 = static_cast<int32_t>(q);
    shift = -e;
    // END SOLUTION
}
inline int32_t multiply_by_quantized_multiplier(int32_t acc, int32_t M0, int shift) {
    // BEGIN SOLUTION
    int64_t hi = (static_cast<int64_t>(acc) * M0 + (1LL << 30)) >> 31;
    if (shift <= 0) return static_cast<int32_t>(hi);
    int64_t r = 1LL << (shift - 1);
    return static_cast<int32_t>(hi >= 0 ? (hi + r) >> shift : -((-hi + r) >> shift));
    // END SOLUTION
    // STUB: return 0;
}
inline void int8_linear(const int8_t* x, int N, int K, int32_t zp_x, const int8_t* w, int C, const int32_t* bias,
                        const int32_t* M0, const int* shift, int32_t zp_y, int8_t* y) {
    // BEGIN SOLUTION
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            int32_t acc = bias[c];
            for (int k = 0; k < K; ++k) acc += (static_cast<int32_t>(x[n * K + k]) - zp_x) * w[c * K + k];
            int32_t v = zp_y + multiply_by_quantized_multiplier(acc, M0[c], shift[c]);
            y[n * C + c] = static_cast<int8_t>(std::clamp(v, -128, 127));
        }
    // END SOLUTION
}
