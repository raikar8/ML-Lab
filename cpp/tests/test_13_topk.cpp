#include "13_topk.hpp"
#include "mini_test.hpp"
#include <numeric>
#include <random>

TEST(topk) {
    std::vector<float> v = {0.1f, 0.9f, 0.5f, 0.9f, -1.f};
    CHECK((topk_indices(v, 3) == std::vector<int>{1, 3, 2}));
    CHECK((topk_indices(v, 10) == std::vector<int>{1, 3, 2, 0, 4}));
    CHECK(topk_indices(v, 0).empty());
    std::mt19937 rng(0); std::uniform_real_distribution<float> u; std::vector<float> big(10000); for (auto& x : big) x = u(rng);
    auto t = topk_indices(big, 50);
    std::vector<int> idx(big.size()); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return big[a] > big[b]; });
    CHECK((t == std::vector<int>(idx.begin(), idx.begin() + 50)));
}
