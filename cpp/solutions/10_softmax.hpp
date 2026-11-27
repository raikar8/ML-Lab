/*
title: Numerically stable softmax, log-softmax and cross-entropy in C++
week: 10
level: easy
time: 20
tags: c++, numerics

For a row of `n` logits: `float logsumexp(const float* z, int n)` (subtract the max), `void softmax(const float* z, float*
out, int n)`, `float cross_entropy(const float* z, int n, int label)` = `logsumexp − z[label]`, and
`void softmax_rows(const float* z, float* out, int rows, int n)`. Must not overflow for logits like ±1000.
*/
#pragma once
#include <algorithm>
#include <cmath>

inline float logsumexp(const float* z, int n) {
    // BEGIN SOLUTION
    float m = *std::max_element(z, z + n);
    double s = 0;
    for (int i = 0; i < n; ++i) s += std::exp(static_cast<double>(z[i] - m));
    return m + static_cast<float>(std::log(s));
    // END SOLUTION
    // STUB: return 0.0f;
}
inline void softmax(const float* z, float* out, int n) {
    // BEGIN SOLUTION
    // subtract the max and normalise by the sum (more accurate than exp(z - lse) in float)
    float m = *std::max_element(z, z + n);
    double s = 0;
    for (int i = 0; i < n; ++i) { out[i] = std::exp(z[i] - m); s += out[i]; }
    for (int i = 0; i < n; ++i) out[i] = static_cast<float>(out[i] / s);
    // END SOLUTION
}
inline float cross_entropy(const float* z, int n, int label) {
    // BEGIN SOLUTION
    return logsumexp(z, n) - z[label];
    // END SOLUTION
    // STUB: return 0.0f;
}
inline void softmax_rows(const float* z, float* out, int rows, int n) {
    // BEGIN SOLUTION
    for (int r = 0; r < rows; ++r) softmax(z + r * n, out + r * n, n);
    // END SOLUTION
}
