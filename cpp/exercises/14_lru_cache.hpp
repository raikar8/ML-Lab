/*
title: LRU cache in C++ (unordered_map + std::list, O(1))
week: 1
level: medium
time: 25
tags: c++, data-structures, classic

`template <typename K, typename V> class LRUCache` with `explicit LRUCache(std::size_t capacity)`,
`std::optional<V> get(const K&)` (marks as most recent), `void put(const K&, const V&)` (insert or update; evict the least
recently used when over capacity; capacity 0 stores nothing), `std::size_t size()`, and `std::vector<K> keys()` from most to
least recent. Use `std::list<std::pair<K,V>>` and `std::unordered_map<K, list iterator>`; move nodes with `splice`.
*/
#pragma once
#include <cstddef>
#include <list>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename K, typename V>
class LRUCache {
public:
    explicit LRUCache(std::size_t capacity) : cap_(capacity) {}
    std::optional<V> get(const K& k) {
        // TODO: implement
        return std::nullopt;
    }
    void put(const K& k, const V& v) {
        // TODO: implement
    }
    std::size_t size() const { return items_.size(); }
    std::vector<K> keys() const {
        std::vector<K> out;
        for (auto& kv : items_) out.push_back(kv.first);
        return out;
    }

private:
    std::size_t cap_;
    std::list<std::pair<K, V>> items_;
    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> map_;
};
