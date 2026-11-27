/*
title: Block convolution with overlap-add (streaming long FIR / room impulse response)
week: 8
level: medium
time: 30
tags: c++, dsp, streaming, audio

`OverlapAdd(std::vector<float> h, std::size_t block)` — filters a stream in blocks of exactly `block` samples. For each
input block, compute its **full linear convolution** with `h` (length `block + L − 1`, direct O(block·L) is fine — the
point is the bookkeeping), add the saved tail from previous blocks to the first `L − 1` outputs, emit `block` samples,
and keep the remaining `L − 1` as the new tail. `void process(const float* in, float* out)` (one block). Output must
equal direct convolution of the whole signal (truncated to its length).
*/
#pragma once
#include <cstddef>
#include <algorithm>
#include <vector>

class OverlapAdd {
public:
    OverlapAdd(std::vector<float> h, std::size_t block) : h_(std::move(h)), B_(block), tail_(h_.size() - 1, 0.f), y_(B_ + h_.size() - 1) {}
    void process(const float* in, float* out) {
        // BEGIN SOLUTION
        const std::size_t L = h_.size();
        std::fill(y_.begin(), y_.end(), 0.f);
        for (std::size_t n = 0; n < B_; ++n)
            for (std::size_t k = 0; k < L; ++k) y_[n + k] += in[n] * h_[k];
        for (std::size_t i = 0; i + 1 < L; ++i) y_[i] += tail_[i];
        for (std::size_t n = 0; n < B_; ++n) out[n] = y_[n];
        for (std::size_t i = 0; i + 1 < L; ++i) tail_[i] = y_[B_ + i];
        // END SOLUTION
    }

private:
    std::vector<float> h_;
    std::size_t B_;
    std::vector<float> tail_, y_;
};
