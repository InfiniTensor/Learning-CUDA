#include "params.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// 去掉首尾空白
std::string Trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool ParseInt(const std::string& s, int* out) {
    char* end = NULL;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') return false;
    *out = (int)v;
    return true;
}

bool ParseFloat(const std::string& s, float* out) {
    char* end = NULL;
    float v = std::strtof(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') return false;
    *out = v;
    return true;
}

} // namespace

bool ParseParams(const char* path, NlmParams* out, std::string* err) {
    if (!path || !out) {
        if (err) *err = "ParseParams: 空指针参数";
        return false;
    }
    FILE* fp = std::fopen(path, "r");
    if (!fp) {
        if (err) *err = std::string("无法打开参数文件: ") + path;
        return false;
    }

    NlmParams p;  // 默认值与任务文档示例一致
    char line[512];
    int lineno = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        ++lineno;
        std::string s(line);
        // 去掉注释
        size_t sharp = s.find('#');
        if (sharp != std::string::npos) s = s.substr(0, sharp);
        s = Trim(s);
        if (s.empty()) continue;

        size_t eq = s.find('=');
        if (eq == std::string::npos) {
            std::fclose(fp);
            if (err) *err = "参数文件第 " + std::to_string(lineno) + " 行缺少 '='";
            return false;
        }
        std::string key = Trim(s.substr(0, eq));
        std::string val = Trim(s.substr(eq + 1));

        if (key == "patch_radius") {
            if (!ParseInt(val, &p.patch_radius)) {
                std::fclose(fp);
                if (err) *err = "patch_radius 非法取值: " + val;
                return false;
            }
        } else if (key == "search_radius") {
            if (!ParseInt(val, &p.search_radius)) {
                std::fclose(fp);
                if (err) *err = "search_radius 非法取值: " + val;
                return false;
            }
        } else if (key == "h") {
            if (!ParseFloat(val, &p.h)) {
                std::fclose(fp);
                if (err) *err = "h 非法取值: " + val;
                return false;
            }
        } else if (key == "sigma") {
            if (!ParseFloat(val, &p.sigma)) {
                std::fclose(fp);
                if (err) *err = "sigma 非法取值: " + val;
                return false;
            }
        } else {
            std::fclose(fp);
            if (err) *err = "未知参数 key: " + key;
            return false;
        }
    }
    std::fclose(fp);

    // 取值范围校验（覆盖任务基线 pr=3 / sr=10，上限为防御性约束）
    if (p.patch_radius < 1 || p.patch_radius > 8) {
        if (err) *err = "patch_radius 超出允许范围 [1,8]";
        return false;
    }
    if (p.search_radius < 1 || p.search_radius > 32) {
        if (err) *err = "search_radius 超出允许范围 [1,32]";
        return false;
    }
    if (!(p.h > 0.0f) || p.h > 1000.0f) {
        if (err) *err = "h 超出允许范围 (0,1000]";
        return false;
    }
    if (p.sigma < 0.0f || p.sigma > 1000.0f) {
        if (err) *err = "sigma 超出允许范围 [0,1000]";
        return false;
    }

    *out = p;
    return true;
}
