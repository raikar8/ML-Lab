/*
title: Streaming causal dilated Conv1d with state buffers (multi-channel)
week: 7
level: medium
time: 35
tags: c++, streaming, audio, real-time

`StreamingConv1d(int c_in, int c_out, int k, int dilation, std::vector<float> w [c_out×c_in×k], std::vector<float> b)`.
Channel-major frames: an input chunk is `c_in × t` (row = channel). `void process(const float* in, int t, float* out)`
writes `c_out × t`: `y[o][n] = b[o] + Σ_i Σ_j w[o][i][j] · x[i][n − (k−1−j)·dilation]`, where past samples come from an
internal history of the last `(k−1)·dilation` frames per channel (zeros at start). Pre-allocate the history in the
constructor; `process` may use a member scratch buffer sized for the largest chunk seen (grow only if needed). `reset()`.
*/
#pragma once
#include <algorithm>
#include <vector>

class StreamingConv1d {
public:
    StreamingConv1d(int c_in, int c_out, int k, int dilation, std::vector<float> w, std::vector<float> b)
        : ci_(c_in), co_(c_out), k_(k), d_(dilation), ctx_((k - 1) * dilation), w_(std::move(w)), b_(std::move(b)),
          hist_(static_cast<std::size_t>(c_in) * ctx_, 0.f) {}
    void reset() { std::fill(hist_.begin(), hist_.end(), 0.f); }
    void process(const float* in, int t, float* out) {
        // BEGIN SOLUTION
        const int L = ctx_ + t;
        buf_.resize(static_cast<std::size_t>(ci_) * L);
        for (int i = 0; i < ci_; ++i) {
            std::copy(hist_.begin() + i * ctx_, hist_.begin() + (i + 1) * ctx_, buf_.begin() + i * L);
            std::copy(in + i * t, in + (i + 1) * t, buf_.begin() + i * L + ctx_);
        }
        for (int o = 0; o < co_; ++o)
            for (int n = 0; n < t; ++n) {
                float acc = b_[o];
                for (int i = 0; i < ci_; ++i)
                    for (int j = 0; j < k_; ++j) acc += w_[(o * ci_ + i) * k_ + j] * buf_[i * L + n + j * d_];
                out[o * t + n] = acc;
            }
        for (int i = 0; i < ci_; ++i)
            std::copy(buf_.begin() + i * L + t, buf_.begin() + (i + 1) * L, hist_.begin() + i * ctx_);
        // END SOLUTION
    }

private:
    int ci_, co_, k_, d_, ctx_;
    std::vector<float> w_, b_, hist_, buf_;
};
