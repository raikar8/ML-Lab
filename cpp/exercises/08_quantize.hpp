/*
title: Per-channel symmetric int8 weight quantization and per-tensor asymmetric activations
week: 22
level: easy
time: 25
tags: c++, quantization, embedded

Row-major weights `w` with `rows` output channels × `cols` inputs.
* `std::vector<float> per_channel_scales(const std::vector<float>& w, int rows, int cols)` — `max|row| / 127` (use 1.0 for
  an all-zero row).
* `std::vector<int8_t> quantize_weights(w, rows, cols, scales)` — `clamp(round(w/s), −127, 127)`.
* `struct AsymParams { float scale; int32_t zero_point; }` and `AsymParams asym_params(float lo, float hi)` for int8 range
  [−128, 127], widening [lo, hi] to include 0, `zero_point = round(−128 − lo/scale)` clamped to the range.
* `int8_t quantize_act(float x, AsymParams p)` and `float dequantize_act(int8_t q, AsymParams p)`.
*/
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct AsymParams { float scale; int32_t zero_point; };

inline std::vector<float> per_channel_scales(const std::vector<float>& w, int rows, int cols) {
    std::vector<float> s(rows, 1.0f);
    // TODO: implement
    return s;
}
inline std::vector<int8_t> quantize_weights(const std::vector<float>& w, int rows, int cols, const std::vector<float>& s) {
    std::vector<int8_t> q(w.size());
    // TODO: implement
    return q;
}
inline AsymParams asym_params(float lo, float hi) {
    // TODO: implement
    return {1.0f, 0};
}
inline int8_t quantize_act(float x, AsymParams p) {
    // TODO: implement
    return 0;
}
inline float dequantize_act(int8_t q, AsymParams p) { return p.scale * (static_cast<int32_t>(q) - p.zero_point); }
