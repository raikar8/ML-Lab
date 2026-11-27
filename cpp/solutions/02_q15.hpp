/*
title: Q15 fixed-point arithmetic with rounding and saturation
week: 21
level: easy
time: 20
tags: c++, fixed-point, dsp, embedded

`int16_t` Q15 helpers: `to_q15(float)` (round, saturate), `from_q15(int16_t)`, `sat16(int32_t)`,
`q15_add` (saturating), `q15_mul` (32-bit product, +2^14 rounding, >>15, saturate — note −1 × −1 must give 32767),
and `q15_mac(int64_t acc, int16_t a, int16_t b)` returning `acc + a*b` (Q30 accumulator for dot products).
*/
#pragma once
#include <cmath>
#include <cstdint>

inline int16_t sat16(int32_t v) {
    // BEGIN SOLUTION
    return static_cast<int16_t>(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
    // END SOLUTION
    // STUB: return 0;
}
inline int16_t to_q15(float x) {
    // BEGIN SOLUTION
    long v = std::lround(static_cast<double>(x) * 32768.0);
    return sat16(static_cast<int32_t>(v > 40000 ? 40000 : (v < -40000 ? -40000 : v)));
    // END SOLUTION
    // STUB: return 0;
}
inline float from_q15(int16_t q) { return static_cast<float>(q) / 32768.0f; }
inline int16_t q15_add(int16_t a, int16_t b) {
    // BEGIN SOLUTION
    return sat16(static_cast<int32_t>(a) + b);
    // END SOLUTION
    // STUB: return 0;
}
inline int16_t q15_mul(int16_t a, int16_t b) {
    // BEGIN SOLUTION
    int32_t p = static_cast<int32_t>(a) * b;
    return sat16((p + (1 << 14)) >> 15);
    // END SOLUTION
    // STUB: return 0;
}
inline int64_t q15_mac(int64_t acc, int16_t a, int16_t b) {
    // BEGIN SOLUTION
    return acc + static_cast<int32_t>(a) * b;
    // END SOLUTION
    // STUB: return 0;
}
