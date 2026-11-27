/*
title: Biquad filter — RBJ low-pass design and Direct Form II Transposed
week: 21
level: medium
time: 30
tags: c++, dsp, filters

* `BiquadCoeffs lowpass(double fs, double f0, double Q)` — RBJ cookbook low-pass, normalised by a0:
  `w0 = 2π f0/fs, α = sin(w0)/(2Q)`, `b0 = (1−cos w0)/2, b1 = 1−cos w0, b2 = b0, a0 = 1+α, a1 = −2cos w0, a2 = 1−α`.
* `Biquad` (float state, **Direct Form II Transposed**): `float process(float x)` with
  `y = b0 x + z1; z1 = b1 x − a1 y + z2; z2 = b2 x − a2 y`. `void reset()`.
* `double magnitude_at(const BiquadCoeffs&, double fs, double f)` — |H(e^{jω})| evaluated analytically.
*/
#pragma once
#include <cmath>
#include <complex>

struct BiquadCoeffs { double b0, b1, b2, a1, a2; };

inline BiquadCoeffs lowpass(double fs, double f0, double Q) {
    // TODO: implement
    return {0, 0, 0, 0, 0};
}

inline double magnitude_at(const BiquadCoeffs& k, double fs, double f) {
    // TODO: implement
    return 0.0;
}

class Biquad {
public:
    explicit Biquad(BiquadCoeffs c) : c_(c) {}
    float process(float x) {
        // TODO: implement
        return 0.0f;
    }
    void reset() { z1_ = z2_ = 0.0; }

private:
    BiquadCoeffs c_;
    double z1_ = 0.0, z2_ = 0.0;
};
