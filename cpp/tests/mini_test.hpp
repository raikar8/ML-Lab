// A tiny test harness (no dependencies): TEST(name) { CHECK(...); CHECK_NEAR(a, b, tol); }
#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace mini_test {
struct Case { const char* name; std::function<void()> fn; };
inline std::vector<Case>& registry() { static std::vector<Case> r; return r; }
inline int& failures() { static int f = 0; return f; }
struct Reg { Reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };
}  // namespace mini_test

#define MT_CAT2(a, b) a##b
#define MT_CAT(a, b) MT_CAT2(a, b)
#define TEST(name) static void name(); static mini_test::Reg MT_CAT(reg_, name)(#name, name); static void name()
#define CHECK(cond) do { if (!(cond)) { std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++mini_test::failures(); } } while (0)
#define CHECK_NEAR(a, b, tol) do { double _a = (a), _b = (b); if (!(std::fabs(_a - _b) <= (tol))) { \
    std::printf("  FAIL %s:%d: %s = %.9g, expected %.9g (tol %g)\n", __FILE__, __LINE__, #a, _a, _b, (double)(tol)); \
    ++mini_test::failures(); } } while (0)

int main() {
    int before = 0;
    for (auto& c : mini_test::registry()) {
        before = mini_test::failures();
        try { c.fn(); } catch (const std::exception& e) { std::printf("  FAIL exception: %s\n", e.what()); ++mini_test::failures(); }
        std::printf("%s %s\n", mini_test::failures() == before ? "[ OK ]" : "[FAIL]", c.name);
    }
    std::printf(mini_test::failures() ? "%d check(s) failed\n" : "all passed\n", mini_test::failures());
    return mini_test::failures() ? 1 : 0;
}
