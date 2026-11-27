/*
title: Top-k indices with a min-heap (O(n log k)) — top-k sampling, beam search, MoE routing
week: 1
level: easy
time: 20
tags: c++, algorithms, heap

`std::vector<int> topk_indices(const std::vector<float>& v, int k)` — indices of the k largest values, sorted by value
descending (ties: smaller index first). Use `std::priority_queue` (or `std::make_heap`) as a size-k **min**-heap; do not
sort the whole array. If k ≥ n return all indices sorted.
*/
#pragma once
#include <algorithm>
#include <queue>
#include <vector>

inline std::vector<int> topk_indices(const std::vector<float>& v, int k) {
    // TODO: implement
    return {};
}
