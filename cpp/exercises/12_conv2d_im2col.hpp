/*
title: Conv2d via im2col + GEMM (NCHW, single image), and a direct reference
week: 7
level: medium
time: 40
tags: c++, cnn, gemm, embedded

* `int out_size(int n, int k, int stride, int pad)`.
* `void im2col(const float* x, int C, int H, int W, int k, int stride, int pad, float* cols)` — `cols` is
  `(C·k·k) × (Ho·Wo)` row-major; row index `(c·k + ki)·k + kj`, zero for padded positions.
* `void conv2d_im2col(const float* x, int C, int H, int W, const float* w [Co×C×k×k], const float* b, int Co, int k,
  int stride, int pad, float* y [Co×Ho×Wo])` — im2col then a plain triple-loop GEMM `y = W·cols + b`.
* `void conv2d_direct(...)` same signature, nested loops — used as the reference.
*/
#pragma once
#include <vector>

inline int out_size(int n, int k, int stride, int pad) { return (n + 2 * pad - k) / stride + 1; }

inline void im2col(const float* x, int C, int H, int W, int k, int stride, int pad, float* cols) {
    // TODO: implement
}

inline void conv2d_im2col(const float* x, int C, int H, int W, const float* w, const float* b, int Co, int k,
                          int stride, int pad, float* y) {
    // TODO: implement
}

inline void conv2d_direct(const float* x, int C, int H, int W, const float* w, const float* b, int Co, int k,
                          int stride, int pad, float* y) {
    const int Ho = out_size(H, k, stride, pad), Wo = out_size(W, k, stride, pad);
    for (int o = 0; o < Co; ++o)
        for (int oh = 0; oh < Ho; ++oh)
            for (int ow = 0; ow < Wo; ++ow) {
                float acc = b ? b[o] : 0.f;
                for (int c = 0; c < C; ++c)
                    for (int ki = 0; ki < k; ++ki)
                        for (int kj = 0; kj < k; ++kj) {
                            const int ih = oh * stride - pad + ki, iw = ow * stride - pad + kj;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) acc += w[((o * C + c) * k + ki) * k + kj] * x[(c * H + ih) * W + iw];
                        }
                y[(o * Ho + oh) * Wo + ow] = acc;
            }
}
