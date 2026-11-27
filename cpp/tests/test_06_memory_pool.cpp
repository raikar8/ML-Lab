#include "06_memory_pool.hpp"
#include "mini_test.hpp"
#include <cstdint>
#include <cstring>
#include <set>

TEST(pool) {
    FixedPool p(24, 10);
    CHECK(p.available() == 10);
    std::set<void*> got;
    for (int i = 0; i < 10; ++i) { void* b = p.allocate(); CHECK(b != nullptr); CHECK(p.owns(b)); CHECK(reinterpret_cast<std::uintptr_t>(b) % alignof(std::max_align_t) == 0); std::memset(b, 0xAB, 24); got.insert(b); }
    CHECK(got.size() == 10); CHECK(p.allocate() == nullptr); CHECK(p.available() == 0);
    void* one = *got.begin(); p.deallocate(one); CHECK(p.available() == 1); CHECK(p.allocate() == one);
    int x; CHECK(!p.owns(&x));
    for (void* b : got) p.deallocate(b);
    CHECK(p.available() == 10);
}
