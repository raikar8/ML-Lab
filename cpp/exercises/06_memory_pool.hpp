/*
title: Fixed-block memory pool with O(1) allocate/free (no malloc in the real-time path)
week: 21
level: medium
time: 30
tags: c++, embedded, memory, real-time

`FixedPool(std::size_t block_size, std::size_t n_blocks)` allocates one buffer at construction. Free blocks form an
intrusive singly linked **free list** stored inside the free blocks themselves (block size is rounded up to hold a pointer
and to `alignof(std::max_align_t)`). `void* allocate()` → nullptr when exhausted; `void deallocate(void*)`;
`std::size_t available()`; `bool owns(const void*)`. All operations O(1).
*/
#pragma once
#include <cstddef>
#include <algorithm>
#include <memory>
#include <vector>

class FixedPool {
public:
    FixedPool(std::size_t block_size, std::size_t n_blocks) {
        // TODO: implement
    }
    void* allocate() {
        // TODO: implement
        return nullptr;
    }
    void deallocate(void* p) {
        // TODO: implement
    }
    std::size_t available() const { return avail_; }
    bool owns(const void* p) const {
        // TODO: implement
        return false;
    }

private:
    std::unique_ptr<std::max_align_t[]> mem_;
    unsigned char* base_ = nullptr;
    void* free_ = nullptr;
    std::size_t bs_ = 0, n_ = 0, avail_ = 0;
};
