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
    // TODO: implement
}
inline int32_t multiply_by_quantized_multiplier(int32_t acc, int32_t M0, int shift) {
    // TODO: implement
    return 0;
}
inline void int8_linear(const int8_t* x, int N, int K, int32_t zp_x, const int8_t* w, int C, const int32_t* bias,
                        const int32_t* M0, const int* shift, int32_t zp_y, int8_t* y) {
    // TODO: implement
}
