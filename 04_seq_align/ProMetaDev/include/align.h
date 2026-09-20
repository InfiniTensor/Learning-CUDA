// align.h - CPU 参考实现（穷举）与 GPU 比对器接口
#pragma once
#include "types.h"
#include <string>
#include <vector>

namespace sa {

// ======================= 单条 read 的带状 DP（窗口内自由起始） =======================
// 把整条 read 比对到窗口 ref[q-B, q+L+B) 内的某一段，返回最优得分与起始坐标。
// 打分：匹配 match、错配 mismatch、空位 gap（线性）。q 不可行时返回 INT32_MIN。
int32_t window_best_score(const RefGenome& ref, const char* read, int L, int64_t q,
                          const Config& cfg, int64_t* out_start);

// ======================= CPU 穷举参考 =======================
// 对参考上每一个可能的锚点求窗口最优，取全局最优（同分取最左）。
// 计算量与参考长度成正比，仅用于小规模正确性验证。
Hit cpu_align_one(const RefGenome& ref, const std::string& read, const Config& cfg);

std::vector<Hit> cpu_align_all(const RefGenome& ref, const std::vector<Read>& reads,
                               const Config& cfg);

// ======================= CPU seed-and-extend（与 GPU 同算法的性能基准） =======================
// 在主机端构建同样的 k-mer 索引，并按同样的"种子→候选→带状 DP 验证→归约"流程比对，
// 用 OpenMP 在 read 维度并行。用于给出可比的 CPU/GPU 加速比。
std::vector<Hit> cpu_seeded_align_all(const RefGenome& ref, const std::vector<Read>& reads,
                                      const Config& cfg, double* out_index_ms = nullptr);

// ======================= GPU 比对器 =======================
struct AlignStats {
    double build_index_ms = 0.0; // 建 k-mer 索引
    double upload_ms = 0.0;      // H2D 上传
    double seed_ms = 0.0;        // 种子提取 + 候选枚举
    double verify_ms = 0.0;      // 带状 DP 验证
    double reduce_ms = 0.0;      // 每 read 归约
    double total_ms = 0.0;       // 端到端
    int64_t index_entries = 0;   // 索引条目数（有效 k-mer 位置数）
    int64_t num_candidates = 0;  // 验证过的 (read, 起始点) 对数
    int64_t num_seeds = 0;       // 提取的种子总数
};

class GpuAligner {
public:
    GpuAligner();
    ~GpuAligner();

    // 上传参考基因组并构建 k-mer 索引（图不变时可复用）
    void build_index(const RefGenome& ref, const Config& cfg);

    // 批量比对
    std::vector<Hit> align(const std::vector<Read>& reads, const Config& cfg);

    const AlignStats& stats() const;

    void release();

private:
    struct Impl;
    Impl* impl_;
};

// 把全局坐标映射回 (seq_id, 序列内偏移)；越界返回 seq_id = -1
void global_to_local(const RefGenome& ref, int64_t gpos, int32_t& seq_id, int64_t& off);

} // namespace sa
