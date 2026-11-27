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
        // BEGIN SOLUTION
        const std::size_t a = alignof(std::max_align_t);
        bs_ = (std::max(block_size, sizeof(void*)) + a - 1) / a * a;
        n_ = n_blocks;
        mem_.reset(new std::max_align_t[(bs_ * n_ + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t)]);
        base_ = reinterpret_cast<unsigned char*>(mem_.get());
        for (std::size_t i = 0; i < n_; ++i) {
            void* blk = base_ + i * bs_;
            *static_cast<void**>(blk) = free_;
            free_ = blk;
        }
        avail_ = n_;
        // END SOLUTION
    }
    void* allocate() {
        // BEGIN SOLUTION
        if (!free_) return nullptr;
        void* b = free_;
        free_ = *static_cast<void**>(b);
        --avail_;
        return b;
        // END SOLUTION
        // STUB: return nullptr;
    }
    void deallocate(void* p) {
        // BEGIN SOLUTION
        if (!p) return;
        *static_cast<void**>(p) = free_;
        free_ = p;
        ++avail_;
        // END SOLUTION
    }
    std::size_t available() const { return avail_; }
    bool owns(const void* p) const {
        // BEGIN SOLUTION
        auto q = static_cast<const unsigned char*>(p);
        return q >= base_ && q < base_ + bs_ * n_ && (q - base_) % bs_ == 0;
        // END SOLUTION
        // STUB: return false;
    }

private:
    std::unique_ptr<std::max_align_t[]> mem_;
    unsigned char* base_ = nullptr;
    void* free_ = nullptr;
    std::size_t bs_ = 0, n_ = 0, avail_ = 0;
};
