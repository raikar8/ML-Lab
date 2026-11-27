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
        // TODO: implement
    }
    void process(const int16_t* in, int16_t* out, std::size_t n) {
        // TODO: implement
    }

private:
    std::vector<int16_t> h_, delay_;
    std::size_t pos_ = 0;
};
