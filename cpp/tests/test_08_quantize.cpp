#include "08_quantize.hpp"
#include "mini_test.hpp"

TEST(weights) {
    std::vector<float> w = {0.5f, -1.0f, 0.25f, 0.0f, 0.0f, 0.0f, 10.f, 5.f, -2.5f};
    auto s = per_channel_scales(w, 3, 3);
    CHECK_NEAR(s[0], 1.0 / 127, 1e-7); CHECK_NEAR(s[1], 1.0, 0); CHECK_NEAR(s[2], 10.0 / 127, 1e-6);
    auto q = quantize_weights(w, 3, 3, s);
    CHECK(q[1] == -127 && q[0] == 64 && q[6] == 127 && q[7] == 64 && q[3] == 0);
}
TEST(activations) {
    auto p = asym_params(0.5f, 6.0f);  // widened to [0, 6]
    CHECK(p.zero_point == -128); CHECK_NEAR(p.scale, 6.0 / 255, 1e-7);
    CHECK_NEAR(dequantize_act(quantize_act(0.0f, p), p), 0.0, 0);
    auto p2 = asym_params(-1.f, 3.f);
    for (float x : {-1.f, 0.f, 1.234f, 3.f}) CHECK_NEAR(dequantize_act(quantize_act(x, p2), p2), x, p2.scale / 2 + 1e-6);
    CHECK(quantize_act(100.f, p2) == 127);
}
