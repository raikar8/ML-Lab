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
    // TODO: implement
    return 0.0f;
}
inline void softmax(const float* z, float* out, int n) {
    // TODO: implement
}
inline float cross_entropy(const float* z, int n, int label) {
    // TODO: implement
    return 0.0f;
}
inline void softmax_rows(const float* z, float* out, int rows, int n) {
    // TODO: implement
}
