/*
title: Cache-blocked matrix multiply (and why it beats the naive loop order)
week: 15
level: medium
time: 35
tags: c++, performance, gemm, kernels

Row-major `A (M×K)`, `B (K×N)`, `C (M×N)`; all functions **overwrite** C.
* `matmul_naive` — i, j, k loop order (strided access to B: cache-unfriendly).
* `matmul_ikj` — i, k, j order (unit-stride inner loop over B and C rows, vectorisable).
* `matmul_tiled(..., int T)` — block over i, k, j in tiles of size T (edges handled), ikj order inside a tile, so tiles of
  A, B and C stay in cache and each loaded element is reused ~T times.
The test checks correctness and prints timings; on most machines tiled/ikj are several times faster than naive for 512².
*/
#pragma once
#include <algorithm>
#include <vector>

inline void matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0;
            for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

inline void matmul_ikj(const float* A, const float* B, float* C, int M, int K, int N) {
    // TODO: implement
}

inline void matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N, int T = 64) {
    // TODO: implement
}
