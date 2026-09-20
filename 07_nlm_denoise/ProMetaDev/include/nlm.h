// nlm.h - CPU 参考 / OpenCV 参考 / GPU 实现接口
#pragma once
#include "types.h"
#include <vector>

namespace nlm {

// ======================= 语义定义（本项目的规范语义） =======================
// 记 p 为中心像素、q 为搜索窗口内的候选像素、P 为以 p 为中心的 patch：
//   D(p,q)  = Σ_{c} Σ_{k∈P} (I_c(clamp(p+k)) - I_c(clamp(q+k)))^2      （逐通道求和平差）
//   w(p,q)  = exp( -max( D(p,q)/|P| - 2σ², 0 ) / h² )
//   out(p)  = Σ_q w(p,q) I(q) / Σ_q w(p,q)
// 其中：
//   - 候选 q 的搜索范围裁剪到图像内（边界像素候选数自然减少）
//   - patch 读取在边界处按"复制边缘"（clamp）处理，保证 patch 始终完整
//   - D 除以 |P| = (2rp+1)^2 得到逐像素平均平方差，故 2σ² 是"纯噪声下 D/|P| 的期望"

// ======================= CPU 参考实现（精确 NLM，OpenMP 并行） =======================
Image cpu_nlm(const Image& in, const NLMParams& p, double* out_ms = nullptr);

// ======================= OpenCV 参考实现（外部对照） =======================
// 调用 cv::fastNlMeansDenoising / fastNlMeansDenoisingColored。
// OpenCV 内部用"快速"近似（patch 权重按阈值二值化 + 不同边界策略），
// 与上面精确语义不会逐像素一致，仅用于独立交叉验证与加速比基准。
Image opencv_nlm(const Image& in, const NLMParams& p, double* out_ms = nullptr);

// ======================= GPU 实现 =======================
class GpuNLM {
public:
    GpuNLM();
    ~GpuNLM();
    // 输入输出均为 8 位整数图像；内部转为 float 计算，结果四舍五入并截断回 0-255
    Image denoise(const Image& in, const NLMParams& p, PerfStats* stats = nullptr);
    void release();

private:
    struct Impl;
    Impl* impl_;
};

// 权重查找表：把 w = exp(-max(x - 2σ²/h², 0)) 按 **指数 x = d/h²** 分档预计算。
// 注意必须按指数分档而不是按距离 d 分档：h 较小时 d 的档宽会被 h² 放大成很大的指数跨度，
// 导致查表权重严重失真（实测按 d 分档时 MAE 可达 3.7，按指数分档降到 0.02 量级）。
// 覆盖范围 x ∈ [0, LUT_XMAX]，超出部分权重已小于 exp(-20) ≈ 2e-9，可忽略。
constexpr double LUT_XMAX = 20.0;
void build_weight_lut(const NLMParams& p, std::vector<float>& lut);

} // namespace nlm
