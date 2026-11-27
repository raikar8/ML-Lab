#include "10_softmax.hpp"
#include "mini_test.hpp"

TEST(stable) {
    float z[3] = {1000.f, 1000.f, -1000.f}, p[3];
    softmax(z, p, 3);
    CHECK_NEAR(p[0], 0.5, 1e-6); CHECK_NEAR(p[2], 0.0, 1e-12);
    CHECK(std::isfinite(logsumexp(z, 3)));
    float a[3] = {1.f, 2.f, 3.f};
    CHECK_NEAR(cross_entropy(a, 3, 2), std::log(std::exp(1.) + std::exp(2.) + std::exp(3.)) - 3, 1e-5);
    float zz[6] = {1, 2, 3, 0, 0, 0}, out[6]; softmax_rows(zz, out, 2, 3);
    CHECK_NEAR(out[3], 1.0 / 3, 1e-6); CHECK_NEAR(out[0] + out[1] + out[2], 1.0, 1e-6);
}
