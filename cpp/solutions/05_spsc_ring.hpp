/*
title: Lock-free single-producer single-consumer ring buffer (audio thread ↔ inference thread)
week: 21
level: hard
time: 40
tags: c++, real-time, concurrency, audio

`template <typename T, std::size_t N> class SpscRing` (N a power of two, checked with `static_assert`):
* `bool push(const T&)` — producer only; false when full. `bool pop(T&)` — consumer only; false when empty.
* `std::size_t write(const T* src, std::size_t n)` / `read(T* dst, std::size_t n)` — bulk versions returning how many
  were transferred (they may transfer fewer than n). `std::size_t size() const`.
* Indices are `std::atomic<std::size_t>` that only increase; slot = index & (N−1). The producer writes data **then**
  publishes `head_` with `memory_order_release`; the consumer reads `head_` with `memory_order_acquire` (and vice versa for
  `tail_`). No locks, no allocation after construction. The test runs a real producer thread and consumer thread.
*/
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>

template <typename T, std::size_t N>
class SpscRing {
    static_assert(N >= 2 && (N & (N - 1)) == 0, "N must be a power of two");

public:
    bool push(const T& v) { return write(&v, 1) == 1; }
    bool pop(T& v) { return read(&v, 1) == 1; }

    std::size_t write(const T* src, std::size_t n) {
        // BEGIN SOLUTION
        std::size_t h = head_.load(std::memory_order_relaxed);
        std::size_t t = tail_.load(std::memory_order_acquire);
        std::size_t k = std::min(n, N - (h - t));
        for (std::size_t i = 0; i < k; ++i) buf_[(h + i) & (N - 1)] = src[i];
        head_.store(h + k, std::memory_order_release);
        return k;
        // END SOLUTION
        // STUB: return 0;
    }
    std::size_t read(T* dst, std::size_t n) {
        // BEGIN SOLUTION
        std::size_t t = tail_.load(std::memory_order_relaxed);
        std::size_t h = head_.load(std::memory_order_acquire);
        std::size_t k = std::min(n, h - t);
        for (std::size_t i = 0; i < k; ++i) dst[i] = buf_[(t + i) & (N - 1)];
        tail_.store(t + k, std::memory_order_release);
        return k;
        // END SOLUTION
        // STUB: return 0;
    }
    std::size_t size() const { return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire); }

private:
    std::array<T, N> buf_{};
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};
