/*
title: Streaming fixed-point FIR filter (Q15 coefficients, Q30 accumulator)
week: 21
level: medium
time: 30
tags: c++, fixed-point, dsp, streaming, real-time

`FirQ15(std::vector<int16_t> taps)` keeps a circular delay line so it can process audio **block by block** with no
allocation in `process`. `void process(const int16_t* in, int16_t* out, std::size_t n)`: for each sample, store it in the
delay line, accumulate `Σ h[k]·x[n−k]` in `int64_t`, round (+2^14), shift >>15, saturate. Processing a signal in blocks of
any size must equal processing it in one call. Also `void reset()`.
*/
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

class FirQ15 {
public:
    explicit FirQ15(std::vector<int16_t> taps) : h_(std::move(taps)), delay_(h_.size(), 0) {}
    void reset() {
        // BEGIN SOLUTION
        std::fill(delay_.begin(), delay_.end(), 0); pos_ = 0;
        // END SOLUTION
    }
    void process(const int16_t* in, int16_t* out, std::size_t n) {
        // BEGIN SOLUTION
        const std::size_t L = h_.size();
        for (std::size_t i = 0; i < n; ++i) {
            delay_[pos_] = in[i];
            int64_t acc = 0;
            std::size_t idx = pos_;
            for (std::size_t k = 0; k < L; ++k) {
                acc += static_cast<int32_t>(h_[k]) * delay_[idx];
                idx = idx == 0 ? L - 1 : idx - 1;
            }
            int64_t v = (acc + (1 << 14)) >> 15;
            out[i] = static_cast<int16_t>(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
            pos_ = pos_ + 1 == L ? 0 : pos_ + 1;
        }
        // END SOLUTION
    }

private:
    std::vector<int16_t> h_, delay_;
    std::size_t pos_ = 0;
};
