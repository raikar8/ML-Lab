#include <algorithm>
#include "04_biquad.hpp"
#include "mini_test.hpp"

static double tone_gain(Biquad& b, double fs, double f) {
    b.reset(); double peak = 0;
    for (int n = 0; n < 20000; ++n) { double y = b.process((float)std::sin(2 * M_PI * f * n / fs)); if (n > 10000) peak = std::max(peak, std::fabs(y)); }
    return peak;
}
TEST(lowpass_response) {
    double fs = 16000; auto k = lowpass(fs, 1000, 0.7071);
    CHECK_NEAR(magnitude_at(k, fs, 0), 1.0, 1e-9);
    CHECK_NEAR(magnitude_at(k, fs, 1000), 0.7071, 1e-3);
    CHECK(magnitude_at(k, fs, 4000) < 0.1);
    Biquad b(k);
    CHECK_NEAR(tone_gain(b, fs, 200), magnitude_at(k, fs, 200), 2e-3);
    CHECK_NEAR(tone_gain(b, fs, 3000), magnitude_at(k, fs, 3000), 2e-3);
}
