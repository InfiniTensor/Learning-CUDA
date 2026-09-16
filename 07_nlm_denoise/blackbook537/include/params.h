#ifndef NLM_PARAMS_H
#define NLM_PARAMS_H

#include <string>

// ============================================================================
// NLM 滤波参数（对应任务文档参数文件定义）
//   patch_radius  : patch 半径，实际 patch 大小 (2*patch_radius+1)^2
//   search_radius : 搜索窗口半径，实际窗口 (2*search_radius+1)^2
//   h             : 滤波强度
//   sigma         : 噪声标准差估计（8 位图像域 0-255）
// 任务基线：patch_radius=3, search_radius=10, h=10.0, sigma=25.0
// ============================================================================
struct NlmParams {
    int   patch_radius;   // >= 1，任务要求至少支持到 3
    int   search_radius;  // >= 1，任务要求至少支持到 10
    float h;              // > 0
    float sigma;          // >= 0

    NlmParams() : patch_radius(3), search_radius(10), h(10.0f), sigma(25.0f) {}
};

// 解析文本参数文件（"key = value" 格式，'#' 之后为注释）。
// 缺失字段使用默认值；未知 key / 非法取值 / 越界取值均返回 false 并填写 err。
// 取值范围约束：pr ∈ [1,8]，sr ∈ [1,32]，h ∈ (0,1000]，sigma ∈ [0,1000]。
bool ParseParams(const char* path, NlmParams* out, std::string* err);

#endif // NLM_PARAMS_H
