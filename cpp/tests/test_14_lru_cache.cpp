#include "14_lru_cache.hpp"
#include "mini_test.hpp"
#include <string>

TEST(lru) {
    LRUCache<int, std::string> c(2);
    c.put(1, "a"); c.put(2, "b");
    CHECK(c.get(1).value() == "a");
    c.put(3, "c");
    CHECK(!c.get(2).has_value()); CHECK(c.size() == 2);
    c.put(1, "z");
    CHECK((c.keys() == std::vector<int>{1, 3})); CHECK(c.get(1).value() == "z");
    LRUCache<int, int> z(0); z.put(1, 1); CHECK(!z.get(1));
    LRUCache<int, int> big(100); for (int i = 0; i < 1000; ++i) big.put(i, i);
    CHECK(big.size() == 100 && big.keys()[0] == 999 && !big.get(899) && big.get(900).value() == 900);
}
