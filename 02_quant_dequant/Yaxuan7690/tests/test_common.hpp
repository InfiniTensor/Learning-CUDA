// =============================================================================
//  test_common.hpp -- 极简测试框架 (不依赖任何第三方库)
//
//  用法:
//      testfw::begin_section("xxx");
//      CHECK(cond);
//      CHECK_MSG(cond, "detail");
//      testfw::end_section();
//      return testfw::summary("test_xxx");
// =============================================================================
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace testfw {

struct State {
    long long checks = 0;
    long long failures = 0;
    long long section_failures = 0;
    std::string section;
    std::vector<std::string> messages;
};

inline State& state() {
    static State s;
    return s;
}

inline void begin_section(const char* name) {
    State& s = state();
    s.section = name;
    s.section_failures = 0;
    std::printf("[ RUN      ] %s\n", name);
    std::fflush(stdout);
}

inline void end_section() {
    State& s = state();
    if (s.section_failures == 0) {
        std::printf("[       OK ] %s\n", s.section.c_str());
    } else {
        std::printf(
            "[  FAILED  ] %s  (%lld failed assertions)\n", s.section.c_str(), s.section_failures);
    }
    std::fflush(stdout);
}

inline bool check(bool ok, const char* expr, const char* file, int line, const std::string& extra) {
    State& s = state();
    ++s.checks;
    if (ok)
        return true;
    ++s.failures;
    ++s.section_failures;
    std::string m = std::string(file) + ":" + std::to_string(line) + "  " + expr;
    if (!extra.empty())
        m += "   [" + extra + "]";
    if (s.messages.size() < 30)
        s.messages.push_back(s.section + " -> " + m);
    std::printf("    FAIL %s\n", m.c_str());
    std::fflush(stdout);
    return false;
}

inline int summary(const char* name) {
    State& s = state();
    std::printf("\n==================== %s ====================\n", name);
    std::printf("total assertions: %lld, failures: %lld\n", s.checks, s.failures);
    if (s.failures != 0) {
        std::printf("failure summary:\n");
        for (const std::string& m : s.messages)
            std::printf("  * %s\n", m.c_str());
        std::printf("RESULT: FAILED\n");
        return 1;
    }
    std::printf("RESULT: PASSED\n");
    return 0;
}

// ---- 便于拼接失败信息的小工具 -------------------------------------------
template <class T>
inline std::string s_of(T v) {
    return std::to_string(v);
}

inline std::string hex32(uint32_t v) {
    char b[32];
    std::snprintf(b, sizeof(b), "0x%08X", v);
    return std::string(b);
}

inline std::string f_of(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.10g", v);
    return std::string(b);
}

inline std::string pair_of(long long i, const std::string& a, const std::string& b) {
    return "i=" + std::to_string(i) + " a=" + a + " b=" + b;
}

}  // namespace testfw

#define CHECK(cond) testfw::check((cond), #cond, __FILE__, __LINE__, std::string())
#define CHECK_MSG(cond, extra) testfw::check((cond), #cond, __FILE__, __LINE__, (extra))
