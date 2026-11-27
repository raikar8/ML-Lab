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
    // BEGIN SOLUTION
    std::fill(C, C + static_cast<std::size_t>(M) * N, 0.f);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            const float a = A[i * K + k];
            const float* b = B + static_cast<std::size_t>(k) * N;
            float* c = C + static_cast<std::size_t>(i) * N;
            for (int j = 0; j < N; ++j) c[j] += a * b[j];
        }
    // END SOLUTION
}

inline void matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N, int T = 64) {
    // BEGIN SOLUTION
    std::fill(C, C + static_cast<std::size_t>(M) * N, 0.f);
    for (int i0 = 0; i0 < M; i0 += T)
        for (int k0 = 0; k0 < K; k0 += T)
            for (int j0 = 0; j0 < N; j0 += T) {
                const int i1 = std::min(i0 + T, M), k1 = std::min(k0 + T, K), j1 = std::min(j0 + T, N);
                for (int i = i0; i < i1; ++i)
                    for (int k = k0; k < k1; ++k) {
                        const float a = A[i * K + k];
                        const float* b = B + static_cast<std::size_t>(k) * N;
                        float* c = C + static_cast<std::size_t>(i) * N;
                        for (int j = j0; j < j1; ++j) c[j] += a * b[j];
                    }
            }
    // END SOLUTION
}
