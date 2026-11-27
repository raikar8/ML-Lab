/*
title: Online mean/variance (Welford) — numerically stable streaming statistics
week: 1
level: easy
time: 15
tags: c++, numerics, streaming

Implement `RunningStats` with `void push(double x)`, `std::size_t count()`, `double mean()`, `double variance()`
(sample variance, n−1; 0 when n < 2). Must be accurate for data like 1e9 + small noise, where E[x²] − E[x]² fails.
*/
#pragma once
#include <cstddef>

class RunningStats {
public:
    void push(double x) {
        // BEGIN SOLUTION
        ++n_;
        double d = x - mean_;
        mean_ += d / static_cast<double>(n_);
        m2_ += d * (x - mean_);
        // END SOLUTION
    }
    std::size_t count() const { return n_; }
    double mean() const { return mean_; }
    double variance() const {
        // BEGIN SOLUTION
        return n_ > 1 ? m2_ / static_cast<double>(n_ - 1) : 0.0;
        // END SOLUTION
        // STUB: return 0.0;
    }

private:
    std::size_t n_ = 0;
    double mean_ = 0.0, m2_ = 0.0;
};
