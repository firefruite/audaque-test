#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace test_util {

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

inline void check(bool condition, const std::string& expr, const std::string& file, int line) {
    g_tests_run++;
    if (condition) {
        g_tests_passed++;
    } else {
        g_tests_failed++;
        std::cerr << "  FAIL: " << expr << " at " << file << ":" << line << std::endl;
    }
}

inline void check_near(float a, float b, float tol, const std::string& expr_a,
                       const std::string& expr_b, const std::string& file, int line) {
    g_tests_run++;
    if (std::abs(a - b) <= tol) {
        g_tests_passed++;
    } else {
        g_tests_failed++;
        std::cerr << "  FAIL: " << expr_a << " (" << a << ") != " << expr_b
                  << " (" << b << ") +/- " << tol
                  << " at " << file << ":" << line << std::endl;
    }
}

inline void check_true(bool cond, const std::string& msg) {
    check(cond, msg, __FILE__, __LINE__);
}

inline int summary() {
    std::cout << "\n[" << g_tests_passed << "/" << g_tests_run << " passed, "
              << g_tests_failed << " failed]" << std::endl;
    return g_tests_failed > 0 ? 1 : 0;
}

} // namespace test_util

#define EXPECT_TRUE(cond) ::test_util::check((cond), #cond, __FILE__, __LINE__)
#define EXPECT_FALSE(cond) ::test_util::check(!(cond), "!(" #cond ")", __FILE__, __LINE__)
#define EXPECT_EQ(a, b) ::test_util::check((a) == (b), #a " == " #b, __FILE__, __LINE__)
#define EXPECT_NE(a, b) ::test_util::check((a) != (b), #a " != " #b, __FILE__, __LINE__)
#define EXPECT_NEAR(a, b, tol) ::test_util::check_near((a), (b), (tol), #a, #b, __FILE__, __LINE__)
