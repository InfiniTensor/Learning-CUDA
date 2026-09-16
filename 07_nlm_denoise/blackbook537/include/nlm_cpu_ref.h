#ifndef NLM_CPU_REF_H
#define NLM_CPU_REF_H

#include "image_io.h"
#include "params.h"

// ============================================================================
// CPU 参考实现（正确性 ground truth + 加速比基线）
// 与 GPU kernel 语义严格一致：
//   - planar 浮点计算，RGB 跨通道联合 patch 距离、共享权重
//   - 边界：每次最终访问坐标独立 clamp（等价 OpenCV BORDER_REPLICATE）
//   - 权重：w = exp(-max(dist - 2*sigma^2*N_patch, 0) / h^2)
// 单线程、无第三方依赖。
// ============================================================================
void NlmDenoiseCpuRef(const ImageU8& src, ImageU8* dst, const NlmParams& params);

#endif // NLM_CPU_REF_H
