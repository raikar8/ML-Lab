#include "05_spsc_ring.hpp"
#include "mini_test.hpp"
#include <chrono>
#include <thread>
#include <vector>

TEST(single_thread) {
    SpscRing<int, 8> r; int v;
    CHECK(!r.pop(v));
    for (int i = 0; i < 8; ++i) CHECK(r.push(i));
    CHECK(!r.push(99)); CHECK(r.size() == 8);
    int out[5]; CHECK(r.read(out, 5) == 5); CHECK(out[0] == 0 && out[4] == 4);
    int in[10] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}; CHECK(r.write(in, 10) == 5);
    int all[8]; CHECK(r.read(all, 8) == 8); CHECK(all[0] == 5 && all[3] == 10 && all[7] == 14);
}
TEST(two_threads_in_order) {
    SpscRing<int, 64> r; const int M = 200000;
    std::vector<int> got; got.reserve(M);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    auto alive = [&] { return std::chrono::steady_clock::now() < deadline; };
    std::thread prod([&] { for (int i = 0; i < M && alive();) { int blk[7]; int k = 0; for (; k < 7 && i + k < M; ++k) blk[k] = i + k; i += (int)r.write(blk, k); } });
    std::thread cons([&] { int buf[5]; while ((int)got.size() < M && alive()) { std::size_t k = r.read(buf, 5); for (std::size_t j = 0; j < k; ++j) got.push_back(buf[j]); } });
    prod.join(); cons.join();
    bool ok = (int)got.size() == M; for (int i = 0; ok && i < M; ++i) ok &= got[i] == i;
    CHECK(ok);
}
