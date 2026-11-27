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
    // BEGIN SOLUTION
    // "worse" = smaller value, or equal value with larger index
    auto worse = [&](int a, int b) { return v[a] < v[b] || (v[a] == v[b] && a > b); };
    auto cmp = [&](int a, int b) { return worse(b, a); };  // top of heap = worst kept element
    std::priority_queue<int, std::vector<int>, decltype(cmp)> heap(cmp);
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        if (static_cast<int>(heap.size()) < k) heap.push(i);
        else if (k > 0 && worse(heap.top(), i)) { heap.pop(); heap.push(i); }
    }
    std::vector<int> out;
    while (!heap.empty()) { out.push_back(heap.top()); heap.pop(); }
    std::reverse(out.begin(), out.end());
    return out;
    // END SOLUTION
    // STUB: return {};
}
