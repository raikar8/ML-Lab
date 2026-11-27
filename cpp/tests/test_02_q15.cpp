#include "02_q15.hpp"
#include "mini_test.hpp"

TEST(conversions) { CHECK(to_q15(0.5f) == 16384); CHECK(to_q15(1.0f) == 32767); CHECK(to_q15(-1.0f) == -32768); CHECK(to_q15(5.f) == 32767); }
TEST(arith) {
    CHECK(q15_mul(to_q15(0.5f), to_q15(-0.5f)) == -8192);
    CHECK(q15_mul(-32768, -32768) == 32767);
    CHECK(q15_add(30000, 10000) == 32767); CHECK(q15_add(-30000, -10000) == -32768);
    CHECK_NEAR(from_q15(q15_mul(to_q15(0.3f), to_q15(0.7f))), 0.21, 1e-4);
    CHECK(q15_mac(10, 100, 200) == 20010);
}
